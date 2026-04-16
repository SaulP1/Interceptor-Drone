'''
#!/usr/bin/env python3
"""
intercept_static.py — Autonomous visual interception of a stationary target.

Uses visual servoing: the drone steers based on what the camera sees,
not based on knowing the target's world position.

Inputs:
  /interceptor/detection    — target offset from camera center (from camera_viewer.py)
  /quadrotor/owl/depth      — depth image to measure distance to target

Outputs:
  /mavros/setpoint_velocity/cmd_vel_unstamped — velocity commands to fly the drone
  /mavros/setpoint_position/local             — position commands for takeoff

Flight logic (proportional control):
  1. Takeoff and hover
  2. Wait for target detection
  3. Steer left/right/up/down to center the target in the camera
  4. Fly forward — speed proportional to distance (fast when far, slow when close)
  5. Declare interception when depth < INTERCEPT_RADIUS

Referenced from: Seabird's sweep_and_detect.py (flight + detection subscriber pattern)
Referenced from: keyboard_fly.py (MAVROS arming, offboard mode)
Referenced from: pi_tracker.py (center-line tracking — minimize offset to center)

Run:
    source /opt/ros/humble/setup.bash
    python3 ~/interceptor/perception/intercept_static.py

Then in another terminal, set OFFBOARD and arm:
    ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'OFFBOARD'}"
    ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"

Requires: PX4 + MAVROS + Isaac Sim + camera_viewer.py all running.
"""

import sys
import os
import json
import time
import threading
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from mavros_msgs.msg import State


# ═══════════════════════════════════════════════════════════════
# CONTROL PARAMETERS — tune these to change flight behavior
# ═══════════════════════════════════════════════════════════════

# Takeoff
TAKEOFF_ALT =  0.5               # meters — hover height before intercept
TAKEOFF_SETTLE_TIME = 5.0 #5.0        # seconds to stabilize after takeoff

# Forward speed control (proportional to depth)
MAX_FORWARD_SPEED = 5.0#1.5          # m/s — max speed when far from target
MIN_FORWARD_SPEED = 0.2          # m/s — creep speed when close
DEPTH_FAR = 8.0                  # meters — at this distance, use max speed
DEPTH_CLOSE = 1.0                # meters — at this distance, use min speed

# Lateral steering (proportional to offset_norm)
# offset_norm ranges from -1 to 1
# Gain converts offset to velocity: vel = gain * offset_norm
LATERAL_GAIN = 1.0               # m/s per unit offset (left/right)
VERTICAL_GAIN = 0.8              # m/s per unit offset (up/down)

# Interception
INTERCEPT_DEPTH = 0.3#0.8            # meters — declare intercept when closer than this
INTERCEPT_CONFIRM_FRAMES = 10    # need this many consecutive close frames

# Search behavior (when target not visible)
HOVER_TIMEOUT = 10.0 #30.0             # seconds — reset timeout if no detection

# Safety
MAX_ALTITUDE = 20.0 #3.0               # meters — never fly higher than this
MIN_ALTITUDE = 0.3               # meters — never fly lower than this

# Depth sampling — sample a patch around detection center to get target depth
DEPTH_PATCH_RADIUS = 10          # pixels around detection center


class InterceptController(Node):
    """
    Visual servoing intercept controller.

    State machine:
      INIT → TAKEOFF → SEARCH → INTERCEPT → DONE

    INIT:      Pre-stream setpoints, wait for user to set OFFBOARD + arm
    TAKEOFF:   Climb to hover altitude, stabilize
    SEARCH:    Hover and wait for target detection
    INTERCEPT: Fly toward target using proportional control
    DONE:      Target intercepted, hover in place
    """

    def __init__(self):
        super().__init__("intercept_controller")

        # ── State ──
        self.state = "INIT"
        self.mavros_state = State()
        self.current_z = 0.0
        self.start_time = time.time()

        # ── Detection data (from camera_viewer.py) ──
        self.latest_detection = None
        self.detection_time = 0.0
        self.detection_lock = threading.Lock()

        # ── Depth data ──
        self.latest_depth = None
        self.depth_lock = threading.Lock()

        # ── Intercept tracking ──
        self.close_frame_count = 0
        self.intercept_declared = False

        # ── QoS profiles ──
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ──

        # MAVROS state (armed, mode)
        self.create_subscription(
            State, "/mavros/state", self._on_mavros_state, mavros_qos
        )

        # MAVROS pose (for current altitude)
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self._on_pose, mavros_qos
        )

        # Detection from camera_viewer.py
        self.create_subscription(
            String, "/interceptor/detection", self._on_detection, 10
        )

        # Depth image from Owl camera
        self.create_subscription(
            Image, "/quadrotor/owl/depth", self._on_depth, reliable_qos
        )

        # ── Publishers ──

        # Velocity commands — this is how we steer the drone
        # Twist message: linear.x=forward, linear.y=left, linear.z=up
        self.vel_pub = self.create_publisher(
            Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10
        )

        # Position setpoints for takeoff phase
        self.pos_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", mavros_qos
        )

        # ── Main control loop at 20Hz ──
        self.timer = self.create_timer(0.05, self._control_loop)
        self.loop_count = 0

        self.get_logger().info("=" * 55)
        self.get_logger().info("  INTERCEPTOR — Visual Servoing Controller")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  Intercept depth: {INTERCEPT_DEPTH}m")
        self.get_logger().info(f"  Max forward speed: {MAX_FORWARD_SPEED} m/s")
        self.get_logger().info(f"  Lateral gain: {LATERAL_GAIN}")
        self.get_logger().info(f"  Takeoff alt: {TAKEOFF_ALT}m")
        self.get_logger().info("  Waiting for MAVROS connection...")

    # ═══════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════

    def _on_mavros_state(self, msg):
        self.mavros_state = msg

    def _on_pose(self, msg):
        self.current_z = msg.pose.position.z

    def _on_detection(self, msg):
        """Receive detection from camera_viewer.py."""
        try:
            data = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = data
                self.detection_time = time.time()
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_depth(self, msg):
        """Receive depth image from Owl camera."""
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
            with self.depth_lock:
                self.latest_depth = depth.copy()
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    # ═══════════════════════════════════════════════════════
    # DEPTH SAMPLING
    # ═══════════════════════════════════════════════════════

    def _get_target_depth(self) -> float:
        """
        Sample depth at the detection center.

        Takes the median of a small patch around the detection center
        to be robust against noisy depth edges.

        Referenced from: Seabird's buoy_detector.py _sample_depth()
        and yolo_detector.py _back_project_bbox_center()

        Returns depth in meters, or -1.0 if no valid depth.
        """
        with self.detection_lock:
            det = self.latest_detection
        with self.depth_lock:
            depth = self.latest_depth

        if det is None or depth is None:
            return -1.0

        cx, cy = det["center_px"]
        h, w = depth.shape[:2]

        # Clamp to image bounds
        r = DEPTH_PATCH_RADIUS
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)

        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.1)]

        if len(valid) == 0:
            return -1.0

        return float(np.median(valid))

    # ═══════════════════════════════════════════════════════
    # COMMAND HELPERS
    # ═══════════════════════════════════════════════════════

    def _send_velocity(self, forward=0.0, left=0.0, up=0.0):
        """
        Send velocity command in drone body frame.

        MAVROS cmd_vel_unstamped uses ENU body frame:
          linear.x = forward (positive = forward)
          linear.y = left (positive = left)
          linear.z = up (positive = up)
        """
        msg = Twist()
        msg.linear.x = float(forward)
        msg.linear.y = float(left)
        msg.linear.z = float(up)
        self.vel_pub.publish(msg)

    def _send_position(self, x=0.0, y=0.0, z=0.0):
        """Send position setpoint (used for takeoff)."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.pos_pub.publish(msg)

    # ═══════════════════════════════════════════════════════
    # MAIN CONTROL LOOP (runs at 20Hz)
    # ═══════════════════════════════════════════════════════

    def _control_loop(self):
        self.loop_count += 1

        if self.state == "INIT":
            self._state_init()
        elif self.state == "TAKEOFF":
            self._state_takeoff()
        elif self.state == "SEARCH":
            self._state_search()
        elif self.state == "INTERCEPT":
            self._state_intercept()
        elif self.state == "DONE":
            self._state_done()

    # ── INIT: Pre-stream setpoints, wait for user to arm ──

    def _state_init(self):
        # Pre-stream position setpoints (PX4 requires this before OFFBOARD)
        self._send_position(0.0, 0.0, TAKEOFF_ALT)

        if self.mavros_state.mode == "":
            return  # not connected yet

        if self.loop_count == 1:
            self.get_logger().info(
                f"Connected. Mode: {self.mavros_state.mode}, "
                f"Armed: {self.mavros_state.armed}"
            )
            self.get_logger().info("Pre-streaming setpoints for 3 seconds...")

        # Pre-stream for 3 seconds before accepting commands
        if self.loop_count < 60:  # 60 * 0.05s = 3 seconds
            return

        # Prompt user to arm (only once)
        if self.loop_count == 60:
            self.get_logger().info(
                "\nReady. In another terminal, set OFFBOARD and arm:\n"
                "  source /opt/ros/humble/setup.bash\n"
                "  ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "
                "\"{base_mode: 0, custom_mode: 'OFFBOARD'}\"\n"
                "  ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "
                "\"{value: true}\"\n"
            )

        # Wait for armed + OFFBOARD (set by user in another terminal)
        if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
            self.get_logger().info("Armed and OFFBOARD — taking off")
            self.state = "TAKEOFF"
            self.start_time = time.time()

    # ── TAKEOFF: Climb to hover altitude ──

    def _state_takeoff(self):
        self._send_position(0.0, 0.0, TAKEOFF_ALT)

        elapsed = time.time() - self.start_time

        # Log progress
        if self.loop_count % 40 == 0:
            self.get_logger().info(
                f"[TAKEOFF] alt={self.current_z:.2f}m "
                f"target={TAKEOFF_ALT}m ({elapsed:.0f}s)"
            )

        # Wait for altitude + settle time
        if elapsed > TAKEOFF_SETTLE_TIME and self.current_z > TAKEOFF_ALT * 0.7:
            self.get_logger().info(
                f"Takeoff complete — alt={self.current_z:.2f}m. Searching..."
            )
            self.state = "SEARCH"
            self.start_time = time.time()

    # ── SEARCH: Hover and wait for detection ──

    def _state_search(self):
        # Hover in place
        self._send_velocity(0.0, 0.0, 0.0)

        # Check for detection
        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time

        det_age = time.time() - det_time if det is not None else 999.0

        if det is not None and det_age < 1.0:
            self.get_logger().info(
                "TARGET ACQUIRED — switching to INTERCEPT"
            )
            self.state = "INTERCEPT"
            self.close_frame_count = 0
            return

        # Log while searching
        elapsed = time.time() - self.start_time
        if self.loop_count % 60 == 0:
            self.get_logger().info(
                f"[SEARCH] Waiting for detection... ({elapsed:.0f}s)"
            )

        if elapsed > HOVER_TIMEOUT:
            self.get_logger().warn(
                f"No detection after {HOVER_TIMEOUT}s — still searching"
            )
            self.start_time = time.time()  # reset timeout

    # ── INTERCEPT: Fly toward target using visual servoing ──

    def _state_intercept(self):
        # Get latest detection
        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time

        det_age = time.time() - det_time if det is not None else 999.0

        # If we lost the target, hover and go back to search
        if det is None or det_age > 2.0:
            self._send_velocity(0.0, 0.0, 0.0)
            if self.loop_count % 40 == 0:
                self.get_logger().warn("[INTERCEPT] Lost target — hovering")
            if det_age > 5.0:
                self.get_logger().warn("[INTERCEPT] Target lost too long — back to SEARCH")
                self.state = "SEARCH"
                self.start_time = time.time()
            return

        # Get offset from camera center
        dx_norm, dy_norm = det["offset_norm"]

        # Get depth to target
        target_depth = self._get_target_depth()

        # ── Compute velocity commands ──

        # LATERAL (left/right): steer to center target horizontally
        # dx_norm > 0 means target is RIGHT of center
        # We need to move RIGHT, which in body frame is negative Y
        lateral_vel = -dx_norm * LATERAL_GAIN

        # VERTICAL (up/down): steer to center target vertically
        # dy_norm > 0 means target is BELOW center
        # We need to move DOWN, which is negative Z
        vertical_vel = -dy_norm * VERTICAL_GAIN

        # Clamp altitude for safety
        if self.current_z > MAX_ALTITUDE:
            vertical_vel = min(vertical_vel, 0.0)
        if self.current_z < MIN_ALTITUDE:
            vertical_vel = max(vertical_vel, 0.0)

        # FORWARD: speed proportional to depth
        if target_depth > 0:
            # Linear interpolation: far=max_speed, close=min_speed
            depth_fraction = (target_depth - DEPTH_CLOSE) / (DEPTH_FAR - DEPTH_CLOSE)
            depth_fraction = max(0.0, min(1.0, depth_fraction))
            forward_vel = MIN_FORWARD_SPEED + depth_fraction * (MAX_FORWARD_SPEED - MIN_FORWARD_SPEED)

            # Check for interception
            if target_depth < INTERCEPT_DEPTH:
                self.close_frame_count += 1
                if self.close_frame_count >= INTERCEPT_CONFIRM_FRAMES:
                    self.get_logger().info(
                        f"\n{'='*55}\n"
                        f"  TARGET INTERCEPTED at depth={target_depth:.2f}m\n"
                        f"{'='*55}"
                    )
                    self.state = "DONE"
                    self.intercept_declared = True
                    return
            else:
                self.close_frame_count = 0
        else:
            # No valid depth — use detection area as rough distance proxy
            # Larger area = closer, so go slower
            area = det.get("area_px", 0)
            if area > 10000:
                forward_vel = MIN_FORWARD_SPEED
            else:
                forward_vel = MAX_FORWARD_SPEED * 0.5

        # Send velocity command
        self._send_velocity(forward_vel, lateral_vel, vertical_vel)

        # Log periodically
        if self.loop_count % 20 == 0:
            depth_str = f"{target_depth:.2f}m" if target_depth > 0 else "N/A"
            self.get_logger().info(
                f"[INTERCEPT] depth={depth_str} "
                f"offset=({dx_norm:+.2f},{dy_norm:+.2f}) "
                f"vel=(fwd={forward_vel:.2f}, lat={lateral_vel:.2f}, vert={vertical_vel:.2f}) "
                f"close_frames={self.close_frame_count}"
            )

    # ── DONE: Hover in place ──

    def _state_done(self):
        self._send_velocity(0.0, 0.0, 0.0)

        if self.loop_count % 100 == 0:
            self.get_logger().info("[DONE] Intercepted — hovering in place")


def main():
    rclpy.init()
    node = InterceptController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nInterrupted — shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
'''




#!/usr/bin/env python3
"""
intercept_static.py — Autonomous visual interception using PID control.

Uses visual servoing with PID: the drone steers based on what the camera
sees, converting camera offsets to world-frame velocity commands.

IMPORTANT: MAVROS velocity commands are in LOCAL ENU frame (not body frame).
  linear.x = East velocity
  linear.y = North velocity  
  linear.z = Up velocity
We must rotate body-frame commands by the drone's yaw to get ENU commands.

Inputs:
  /interceptor/detection    — target offset from camera center (from camera_viewer.py)
  /quadrotor/owl/depth      — depth image to measure distance to target
  /mavros/local_position/pose — drone position and orientation (for yaw)

Outputs:
  /mavros/setpoint_velocity/cmd_vel_unstamped — velocity commands (ENU frame)
  /mavros/setpoint_position/local             — position commands for takeoff

Run:
    source /opt/ros/humble/setup.bash
    python3 ~/interceptor/perception/intercept_static.py

Then in another terminal, set OFFBOARD and arm:
    ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'OFFBOARD'}"
    ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"

Requires: PX4 + MAVROS + Isaac Sim + camera_viewer.py all running.
"""

import sys
import os
import json
import time
import threading
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from mavros_msgs.msg import State


# ═══════════════════════════════════════════════════════════════
# CONTROL PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Takeoff
TAKEOFF_ALT = 0.5
TAKEOFF_SETTLE_TIME = 2.0

# Forward speed (proportional to depth)
MAX_FORWARD_SPEED = 4.0
MIN_FORWARD_SPEED = 0.3
DEPTH_FAR = 8.0
DEPTH_CLOSE = 1.0

# PID Gains — Lateral (left/right steering)
LATERAL_KP = 2.0
LATERAL_KI = 0.1
LATERAL_KD = 0.5

# PID Gains — Vertical (up/down steering)
VERTICAL_KP = 1.5
VERTICAL_KI = 0.08
VERTICAL_KD = 0.4

# Interception
INTERCEPT_DEPTH = 0.3
INTERCEPT_CONFIRM_FRAMES = 10

# Search
HOVER_TIMEOUT = 30.0

# Safety
MAX_ALTITUDE = 20.0
MIN_ALTITUDE = 0.3
MAX_VELOCITY = 5.0

# Depth sampling
DEPTH_PATCH_RADIUS = 10

# Control loop
CONTROL_DT = 0.05  # 20Hz


# ═══════════════════════════════════════════════════════════════
# PID CONTROLLER
# ═══════════════════════════════════════════════════════════════

class PIDController:
    """
    PID controller for one axis.
    
    output = Kp * error + Ki * integral(error) + Kd * d(error)/dt
    """

    def __init__(self, kp, ki, kd, integral_max=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True

    def compute(self, error, dt):
        p_term = self.kp * error

        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral

        if self.first_call:
            derivative = 0.0
            self.first_call = False
        else:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        self.prev_error = error
        return p_term + i_term + d_term

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True


# ═══════════════════════════════════════════════════════════════
# QUATERNION TO YAW HELPER
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(x, y, z, w):
    """
    Extract yaw angle from quaternion.
    
    Returns yaw in radians. Yaw is the rotation around the Z (up) axis.
    0 = facing East (+X in ENU), pi/2 = facing North (+Y in ENU).
    
    This tells us which direction the drone's camera is pointing
    so we can convert body-frame commands to world-frame commands.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class InterceptController(Node):
    """
    Visual servoing intercept controller with PID.

    State machine:
      INIT → TAKEOFF → INTERCEPT → DONE
      (SEARCH state entered if target lost during INTERCEPT)
    """

    def __init__(self):
        super().__init__("intercept_controller")

        # ── State ──
        self.state = "INIT"
        self.mavros_state = State()
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0  # radians, ENU frame
        self.start_time = time.time()

        # ── Detection data ──
        self.latest_detection = None
        self.detection_time = 0.0
        self.detection_lock = threading.Lock()

        # ── Depth data ──
        self.latest_depth = None
        self.depth_lock = threading.Lock()

        # ── PID controllers ──
        self.pid_lateral = PIDController(LATERAL_KP, LATERAL_KI, LATERAL_KD)
        self.pid_vertical = PIDController(VERTICAL_KP, VERTICAL_KI, VERTICAL_KD)

        # ── Intercept tracking ──
        self.close_frame_count = 0
        self.intercept_declared = False
        self.last_control_time = time.time()

        # ── QoS ──
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ──
        self.create_subscription(State, "/mavros/state", self._on_mavros_state, mavros_qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self._on_pose, mavros_qos)
        self.create_subscription(String, "/interceptor/detection", self._on_detection, 10)
        self.create_subscription(Image, "/quadrotor/owl/depth", self._on_depth, reliable_qos)

        # ── Publishers ──
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)
        self.pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", mavros_qos)

        # ── Control loop 20Hz ──
        self.timer = self.create_timer(CONTROL_DT, self._control_loop)
        self.loop_count = 0

        self.get_logger().info("=" * 55)
        self.get_logger().info("  INTERCEPTOR — PID Visual Servoing Controller")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  Lateral  PID: P={LATERAL_KP} I={LATERAL_KI} D={LATERAL_KD}")
        self.get_logger().info(f"  Vertical PID: P={VERTICAL_KP} I={VERTICAL_KI} D={VERTICAL_KD}")
        self.get_logger().info(f"  Max fwd speed: {MAX_FORWARD_SPEED} m/s")
        self.get_logger().info(f"  Intercept depth: {INTERCEPT_DEPTH}m")
        self.get_logger().info("  Waiting for MAVROS connection...")

    # ═══════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════

    def _on_mavros_state(self, msg):
        self.mavros_state = msg

    def _on_pose(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
        q = msg.pose.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def _on_detection(self, msg):
        try:
            data = json.loads(msg.data)
            with self.detection_lock:
                self.latest_detection = data
                self.detection_time = time.time()
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_depth(self, msg):
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            with self.depth_lock:
                self.latest_depth = depth.copy()
        except Exception as e:
            self.get_logger().warn(f"Depth convert failed: {e}")

    # ═══════════════════════════════════════════════════════
    # DEPTH SAMPLING
    # ═══════════════════════════════════════════════════════

    def _get_target_depth(self):
        with self.detection_lock:
            det = self.latest_detection
        with self.depth_lock:
            depth = self.latest_depth

        if det is None or depth is None:
            return -1.0

        cx, cy = det["center_px"]
        h, w = depth.shape[:2]
        r = DEPTH_PATCH_RADIUS
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)

        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.1)]
        if len(valid) == 0:
            return -1.0
        return float(np.median(valid))

    # ═══════════════════════════════════════════════════════
    # COMMAND HELPERS
    # ═══════════════════════════════════════════════════════

    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def _send_body_velocity(self, forward, right, up):
        """
        Send velocity in BODY frame — automatically converts to ENU.

        Body frame (what the camera sees):
          forward = toward where camera points
          right   = to the right of camera
          up      = straight up

        ENU frame (what MAVROS expects):
          x = East
          y = North
          z = Up

        Conversion uses the drone's current yaw:
          enu_x = forward * cos(yaw) - right * sin(yaw)  ... wait, let me think...

        Actually:
          forward (body) = direction the drone faces
          If yaw=0, drone faces East (+X in ENU)
          If yaw=pi/2, drone faces North (+Y in ENU)

        So:
          vel_east  = forward * cos(yaw) + right * (-sin(yaw))
          vel_north = forward * sin(yaw) + right * cos(yaw)

        But "right" in body frame is perpendicular to forward:
          right_east  = -sin(yaw) ... no.

        Let me be precise:
          Body forward direction in ENU: (cos(yaw), sin(yaw))
          Body right direction in ENU:   (sin(yaw), -cos(yaw))
          
        Wait — ENU right of forward:
          If forward = (cos(yaw), sin(yaw)), 
          then right = rotate forward by -90° = (sin(yaw), -cos(yaw))

        So:
          vel_east  = forward * cos(yaw) + right * sin(yaw)
          vel_north = forward * sin(yaw) + right * (-cos(yaw))
          vel_up    = up
        """
        yaw = self.current_yaw

        vel_east  = forward * math.cos(yaw) + right * math.sin(yaw)
        vel_north = forward * math.sin(yaw) - right * math.cos(yaw)
        vel_up    = up

        msg = Twist()
        msg.linear.x = self._clamp(float(vel_east), MAX_VELOCITY)
        msg.linear.y = self._clamp(float(vel_north), MAX_VELOCITY)
        msg.linear.z = self._clamp(float(vel_up), MAX_VELOCITY)
        self.vel_pub.publish(msg)

    def _send_position(self, x=0.0, y=0.0, z=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.pos_pub.publish(msg)

    # ═══════════════════════════════════════════════════════
    # CONTROL LOOP
    # ═══════════════════════════════════════════════════════

    def _control_loop(self):
        self.loop_count += 1

        if self.state == "INIT":
            self._state_init()
        elif self.state == "TAKEOFF":
            self._state_takeoff()
        elif self.state == "SEARCH":
            self._state_search()
        elif self.state == "INTERCEPT":
            self._state_intercept()
        elif self.state == "DONE":
            self._state_done()

    # ── INIT ──

    def _state_init(self):
        self._send_position(0.0, 0.0, TAKEOFF_ALT)

        if self.mavros_state.mode == "":
            return

        if self.loop_count == 1:
            self.get_logger().info(
                f"Connected. Mode: {self.mavros_state.mode}, Armed: {self.mavros_state.armed}"
            )
            self.get_logger().info("Pre-streaming setpoints for 3 seconds...")

        if self.loop_count < 60:
            return

        if self.loop_count == 60:
            self.get_logger().info(
                "\nReady. In another terminal, set OFFBOARD and arm:\n"
                "  source /opt/ros/humble/setup.bash\n"
                "  ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "
                "\"{base_mode: 0, custom_mode: 'OFFBOARD'}\"\n"
                "  ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "
                "\"{value: true}\"\n"
            )

        if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
            self.get_logger().info("Armed and OFFBOARD — taking off")
            self.state = "TAKEOFF"
            self.start_time = time.time()

    # ── TAKEOFF ──

    def _state_takeoff(self):
        self._send_position(0.0, 0.0, TAKEOFF_ALT)

        elapsed = time.time() - self.start_time

        if self.loop_count % 40 == 0:
            self.get_logger().info(
                f"[TAKEOFF] alt={self.current_z:.2f}m target={TAKEOFF_ALT}m "
                f"yaw={math.degrees(self.current_yaw):.1f}° ({elapsed:.0f}s)"
            )

        if elapsed > TAKEOFF_SETTLE_TIME and self.current_z > TAKEOFF_ALT * 0.5:
            self.get_logger().info(
                f"Takeoff complete — alt={self.current_z:.2f}m, "
                f"yaw={math.degrees(self.current_yaw):.1f}°. Intercepting..."
            )
            self.pid_lateral.reset()
            self.pid_vertical.reset()
            self.last_control_time = time.time()
            self.state = "INTERCEPT"
            self.close_frame_count = 0
            self.start_time = time.time()

    # ── SEARCH ──

    def _state_search(self):
        self._send_body_velocity(0.0, 0.0, 0.0)

        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time

        det_age = time.time() - det_time if det is not None else 999.0

        if det is not None and det_age < 1.0:
            self.get_logger().info("TARGET ACQUIRED — switching to INTERCEPT")
            self.pid_lateral.reset()
            self.pid_vertical.reset()
            self.last_control_time = time.time()
            self.state = "INTERCEPT"
            self.close_frame_count = 0
            return

        elapsed = time.time() - self.start_time
        if self.loop_count % 60 == 0:
            self.get_logger().info(f"[SEARCH] Waiting for detection... ({elapsed:.0f}s)")

        if elapsed > HOVER_TIMEOUT:
            self.get_logger().warn(f"No detection after {HOVER_TIMEOUT}s — still searching")
            self.start_time = time.time()

    # ── INTERCEPT ──

    def _state_intercept(self):
        now = time.time()
        dt = max(0.01, min(0.1, now - self.last_control_time))
        self.last_control_time = now

        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time

        det_age = now - det_time if det is not None else 999.0

        # Lost target — hover
        if det is None or det_age > 2.0:
            self._send_body_velocity(0.0, 0.0, 0.0)
            if self.loop_count % 40 == 0:
                self.get_logger().warn("[INTERCEPT] Lost target — hovering")
            if det_age > 5.0:
                self.get_logger().warn("[INTERCEPT] Target lost — back to SEARCH")
                self.state = "SEARCH"
                self.start_time = time.time()
            return

        # Camera offset (error for PID)
        dx_norm, dy_norm = det["offset_norm"]

        # Depth to target
        target_depth = self._get_target_depth()

        # ── PID: Lateral (left/right) ──
        # dx_norm > 0 = target is RIGHT of center
        # PID output > 0 when target is right = we need to move right
        # In body frame: right is positive
        right_vel = self.pid_lateral.compute(dx_norm, dt)

        # ── PID: Vertical (up/down) ──
        # dy_norm > 0 = target is BELOW center
        # We need to move down = negative up velocity
        vertical_vel = -self.pid_vertical.compute(dy_norm, dt)

        # Clamp altitude
        if self.current_z > MAX_ALTITUDE:
            vertical_vel = min(vertical_vel, 0.0)
        if self.current_z < MIN_ALTITUDE:
            vertical_vel = max(vertical_vel, 0.0)

        # ── Forward speed (proportional to depth) ──
        if target_depth > 0:
            depth_fraction = (target_depth - DEPTH_CLOSE) / (DEPTH_FAR - DEPTH_CLOSE)
            depth_fraction = max(0.0, min(1.0, depth_fraction))
            forward_vel = MIN_FORWARD_SPEED + depth_fraction * (MAX_FORWARD_SPEED - MIN_FORWARD_SPEED)

            # Check interception
            if target_depth < INTERCEPT_DEPTH:
                self.close_frame_count += 1
                forward_vel = 0.0
                if self.close_frame_count >= INTERCEPT_CONFIRM_FRAMES:
                    self.get_logger().info(
                        f"\n{'='*55}\n"
                        f"  TARGET INTERCEPTED at depth={target_depth:.2f}m\n"
                        f"{'='*55}"
                    )
                    self.state = "DONE"
                    self.intercept_declared = True
                    return
            else:
                self.close_frame_count = 0
        else:
            area = det.get("area_px", 0)
            forward_vel = MIN_FORWARD_SPEED if area > 10000 else MAX_FORWARD_SPEED * 0.5

        # Send body-frame velocity (automatically converted to ENU)
        self._send_body_velocity(forward_vel, right_vel, vertical_vel)

        # Log
        if self.loop_count % 20 == 0:
            depth_str = f"{target_depth:.2f}m" if target_depth > 0 else "N/A"
            self.get_logger().info(
                f"[INTERCEPT] depth={depth_str} "
                f"offset=({dx_norm:+.2f},{dy_norm:+.2f}) "
                f"vel_body=(fwd={forward_vel:.2f}, right={right_vel:.2f}, up={vertical_vel:.2f}) "
                f"yaw={math.degrees(self.current_yaw):.1f}° "
                f"close={self.close_frame_count}"
            )

    # ── DONE ──

    def _state_done(self):
        self._send_body_velocity(0.0, 0.0, 0.0)
        if self.loop_count % 100 == 0:
            self.get_logger().info("[DONE] Intercepted — hovering in place")


def main():
    rclpy.init()
    node = InterceptController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nInterrupted — shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()