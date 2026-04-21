#!/usr/bin/env python3
"""
intercept_yaw.py — Autonomous visual interception with yaw tracking.

The drone TURNS to face the target instead of strafing sideways.
This is closer to how a real interceptor works:
  1. Rotate to point camera at target (yaw control)
  2. Adjust altitude to match target height (vertical control)
  3. Fly forward — speed based on depth (fast when far, slow when close)
  4. Declare interception when depth < threshold

Control breakdown:
  offset_norm dx → YAW rotation (turn to face target)
  offset_norm dy → VERTICAL velocity (match target height)
  depth          → FORWARD speed (proportional to distance)

Inputs:
  /interceptor/detection    — target offset from camera center (from camera_viewer.py)
  /quadrotor/owl/depth      — depth image for distance measurement
  /mavros/local_position/pose — drone position + orientation

Outputs:
  /mavros/setpoint_raw/attitude — attitude + thrust commands (for yaw)
  /mavros/setpoint_velocity/cmd_vel_unstamped — velocity commands (ENU)
  /interceptor/hit — signals world script on interception

Referenced from: intercept_static.py (state machine, depth sampling, detection subscriber)
Referenced from: pi_tracker.py (yaw toward target concept — minimize the offset line)

Run:
    source /opt/ros/humble/setup.bash
    python3 ~/interceptor/perception/intercept_yaw.py

Then in another terminal:
    ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'OFFBOARD'}"
    ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
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
from std_msgs.msg import String, Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


# ═══════════════════════════════════════════════════════════════
# CONTROL PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Takeoff
TAKEOFF_ALT = 0.5 
TAKEOFF_SETTLE_TIME = 2.0

# Forward speed
MAX_FORWARD_SPEED = 4.0          # m/s — hardware speed limit
MAX_DECEL = 3.0                  # m/s² — max braking deceleration (tune by testing)

# YAW PID — controls how aggressively the drone turns to face the target
# This is the key difference from intercept_static.py
# Higher P = snappier rotation toward target
# Higher D = dampens yaw oscillation (prevents spinning back and forth)
YAW_KP = 1.5          # rad/s per unit offset — how fast to turn
YAW_KI = 0.05         # fixes persistent yaw drift
YAW_KD = 0.9          # dampens yaw oscillation

# Camera horizontal FOV in radians (~90° from our lens settings)
# Used to convert pixel offset to angular offset
CAMERA_HFOV_RAD = math.radians(90.0)

# PID - correction on error, correct on past error, dampen based on error change rate
VERTICAL_KP = 8.0
VERTICAL_KI = 0.08
VERTICAL_KD = 0.4

# Small lateral PID for fine corrections (residual after yaw)
LATERAL_KP = 0.5
LATERAL_KI = 0.02
LATERAL_KD = 0.15

# Interception
INTERCEPT_DEPTH = 0.3
INTERCEPT_CONFIRM_FRAMES = 10



# Search
HOVER_TIMEOUT = 30.0

# Safety
MAX_ALTITUDE = 50.0
MIN_ALTITUDE = 0.3
MAX_VELOCITY = 3.0
MAX_YAW_RATE = 1.5    # rad/s — max yaw rotation speed

# Depth sampling
DEPTH_PATCH_RADIUS = 10

# Control loop
CONTROL_DT = 0.05  # 20Hz


# ═══════════════════════════════════════════════════════════════
# PID CONTROLLER
# ═══════════════════════════════════════════════════════════════

class PIDController:
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
# QUATERNION HELPERS
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(x, y, z, w):
    """Extract yaw (radians) from quaternion. 0=East, pi/2=North in ENU."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    """Convert yaw angle (radians) to quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class InterceptYawController(Node):
    """
    Visual servoing with yaw tracking.

    Instead of strafing to center the target, the drone ROTATES
    to face it and then flies forward. This keeps the target
    near camera center naturally.

    State machine: INIT → TAKEOFF → INTERCEPT → DONE
    """

    def __init__(self):
        super().__init__("intercept_yaw_controller")

        # ── State ──
        self.state = "INIT"
        self.mavros_state = State()
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0  # desired yaw angle
        self.start_time = time.time()

        # ── Detection ──
        self.latest_detection = None
        self.detection_time = 0.0
        self.detection_lock = threading.Lock()

        # ── Depth ──
        self.latest_depth = None
        self.depth_lock = threading.Lock()

        # ── PID controllers ──
        self.pid_yaw = PIDController(YAW_KP, YAW_KI, YAW_KD)
        self.pid_vertical = PIDController(VERTICAL_KP, VERTICAL_KI, VERTICAL_KD)
        self.pid_lateral = PIDController(LATERAL_KP, LATERAL_KI, LATERAL_KD)

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

        # Services
        self.arm_client = self.create_client (CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # Subscribers
        self.create_subscription(State, "/mavros/state", self._on_mavros_state, mavros_qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self._on_pose, mavros_qos)
        self.create_subscription(String, "/interceptor/detection", self._on_detection, 10)
        self.create_subscription(Image, "/quadrotor/owl/depth", self._on_depth, reliable_qos)

        # Publishers
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)
        self.pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", mavros_qos)
        self.hit_pub = self.create_publisher(Bool, "/interceptor/hit", 10)

        # Control loop
        self.timer = self.create_timer(CONTROL_DT, self._control_loop)
        self.loop_count = 0

        self.get_logger().info("=" * 55)
        self.get_logger().info("  INTERCEPTOR — Yaw Tracking Controller")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  Yaw PID: P={YAW_KP} I={YAW_KI} D={YAW_KD}")
        self.get_logger().info(f"  Vertical PID: P={VERTICAL_KP} I={VERTICAL_KI} D={VERTICAL_KD}")
        self.get_logger().info(f"  Max fwd speed: {MAX_FORWARD_SPEED} m/s")
        self.get_logger().info(f"  Max yaw rate: {MAX_YAW_RATE} rad/s")
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


    # DEPTH SAMPLING


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


    # COMMAND HELPERS


    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def _send_position_yaw(self, x, y, z, yaw):
        """Send position + yaw setpoint."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pos_pub.publish(msg)

    def _send_body_velocity_with_yaw(self, forward, right, up, yaw_rate):
        """
        Send velocity in body frame + yaw rate.

        Converts body forward/right to ENU using current yaw,
        then adds yaw_rate as angular.z for rotation.
        """
        yaw = self.current_yaw
        vel_east = forward * math.cos(yaw) + right * math.sin(yaw)
        vel_north = forward * math.sin(yaw) - right * math.cos(yaw)

        msg = Twist()
        msg.linear.x = self._clamp(float(vel_east), MAX_VELOCITY)
        msg.linear.y = self._clamp(float(vel_north), MAX_VELOCITY)
        msg.linear.z = self._clamp(float(up), MAX_VELOCITY)
        msg.angular.z = self._clamp(float(yaw_rate), MAX_YAW_RATE)
        self.vel_pub.publish(msg)


    # CONTROL LOOP

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

    # INIT

    def _state_init(self):
        # Pre-stream position setpoints (PX4 requires this before OFFBOARD)
        self._send_position_yaw(0.0, 0.0, TAKEOFF_ALT, self.current_yaw)

        if self.mavros_state.mode == "":
            return

        if self.loop_count == 1:
            self.get_logger().info(
                f"Connected. Mode: {self.mavros_state.mode}, Armed: {self.mavros_state.armed}"
            )
            self.get_logger().info("Pre-streaming setpoints for 3 seconds...")

        if self.loop_count < 60:
            return

        # Already armed and in OFFBOARD — proceed
        if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
            self.get_logger().info("Armed and OFFBOARD — taking off")
            self.target_yaw = self.current_yaw
            self.state = "TAKEOFF"
            self.start_time = time.time()
            return

        # Try to set OFFBOARD and arm (only once)
        if self.loop_count == 60:
            self.get_logger().info("Setting OFFBOARD and arming...")
            thread = threading.Thread(target=self._arm_in_background, daemon=True)
            thread.start()

    def _arm_in_background(self):
        """Run arming sequence in background thread to avoid blocking the control loop."""
        try:
            # Set OFFBOARD
            mode_req = SetMode.Request()
            mode_req.custom_mode = "OFFBOARD"
            future = self.mode_client.call_async(mode_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5)
            if future.result() is not None:
                self.get_logger().info(f"Set mode result: {future.result().mode_sent}")

            time.sleep(0.5)

            # Arm
            arm_req = CommandBool.Request()
            arm_req.value = True
            future = self.arm_client.call_async(arm_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5)
            if future.result() is not None:
                self.get_logger().info(f"Arm result: {future.result().success}")
        except Exception as e:
            self.get_logger().warn(f"Arming failed: {e}. Use Terminal 6 manually.")

    # TAKEOFF

    def _state_takeoff(self):
        self._send_position_yaw(0.0, 0.0, TAKEOFF_ALT, self.target_yaw)
        elapsed = time.time() - self.start_time

        if self.loop_count % 40 == 0:
            self.get_logger().info(
                f"[TAKEOFF] alt={self.current_z:.2f}m target={TAKEOFF_ALT}m "
                f"yaw={math.degrees(self.current_yaw):.1f}° ({elapsed:.0f}s)"
            )

        if elapsed > TAKEOFF_SETTLE_TIME and self.current_z > TAKEOFF_ALT * 0.5:
            self.get_logger().info(
                f"Takeoff complete — alt={self.current_z:.2f}m. Intercepting..."
            )
            self.pid_yaw.reset()
            self.pid_vertical.reset()
            self.pid_lateral.reset()
            self.last_control_time = time.time()
            self.state = "INTERCEPT"
            self.close_frame_count = 0
            self.start_time = time.time()

    # SEARCH

    def _state_search(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)

        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time
        det_age = time.time() - det_time if det is not None else 999.0

        if det is not None and det_age < 1.0:
            self.get_logger().info("TARGET ACQUIRED — switching to INTERCEPT")
            self.pid_yaw.reset()
            self.pid_vertical.reset()
            self.pid_lateral.reset()
            self.last_control_time = time.time()
            self.state = "INTERCEPT"
            self.close_frame_count = 0
            return

        elapsed = time.time() - self.start_time
        if self.loop_count % 60 == 0:
            self.get_logger().info(f"[SEARCH] Waiting for detection... ({elapsed:.0f}s)")
        if elapsed > HOVER_TIMEOUT:
            self.start_time = time.time()

    # INTERCEPT (yaw tracking)

    def _state_intercept(self):
        now = time.time()
        dt = max(0.01, min(0.1, now - self.last_control_time))
        self.last_control_time = now

        with self.detection_lock:
            det = self.latest_detection
            det_time = self.detection_time
        det_age = now - det_time if det is not None else 999.0

        # Lost target
        if det is None or det_age > 2.0:
            self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)
            if self.loop_count % 40 == 0:
                self.get_logger().warn("[INTERCEPT] Lost target — hovering")
            if det_age > 5.0:
                self.get_logger().warn("[INTERCEPT] Target lost — back to SEARCH")
                self.state = "SEARCH"
                self.start_time = time.time()
            return

        dx_norm, dy_norm = det["offset_norm"]
        target_depth = self._get_target_depth()

        # ── YAW: Turn to face the target ──
        # dx_norm is the horizontal offset in the image, range [-1, 1]
        # Convert to angular error: how many radians off-center is the target?
        # If camera HFOV is 90°, then dx_norm=1.0 means target is 45° to the right
        yaw_error = dx_norm * (CAMERA_HFOV_RAD / 2.0)

        # PID computes yaw rate (rad/s) to correct the error
        # Positive error (target right) → negative yaw rate (turn right in ENU)
        # In ENU: positive angular.z = counter-clockwise = turn left
        yaw_rate = -self.pid_yaw.compute(yaw_error, dt)
        yaw_rate = self._clamp(yaw_rate, MAX_YAW_RATE)

        # ── VERTICAL: Move up/down to match target height ──
        vertical_vel = -self.pid_vertical.compute(dy_norm, dt)
        if self.current_z > MAX_ALTITUDE:
            vertical_vel = min(vertical_vel, 0.0)
        if self.current_z < MIN_ALTITUDE:
            vertical_vel = max(vertical_vel, 0.0)

        # ── LATERAL: Small residual correction ──
        # After yaw brings the target roughly to center, this handles
        # the remaining pixel-level offset for precision
        right_vel = self.pid_lateral.compute(dx_norm, dt)

        # ── FORWARD: Speed based on depth ──
        if target_depth > 0:
            platform_limit = 12.0
            stopping_limit = math.sqrt(2 * MAX_DECEL * max(0.1, target_depth)) #ensures you can break to zero by the time you reach the target, with some margin for error
            forward_vel = min(platform_limit, stopping_limit, MAX_FORWARD_SPEED) 

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
                    hit_msg = Bool()
                    hit_msg.data = True
                    self.hit_pub.publish(hit_msg)
                    return
            else:
                self.close_frame_count = 0
        else:
            area = det.get("area_px", 0)
            forward_vel = 0.5 if area > 10000 else MAX_FORWARD_SPEED * 0.5

        # Send command: body velocity + yaw rate
        self._send_body_velocity_with_yaw(forward_vel, right_vel, vertical_vel, yaw_rate)

        # Log
        if self.loop_count % 20 == 0:
            depth_str = f"{target_depth:.2f}m" if target_depth > 0 else "N/A"
            self.get_logger().info(
                f"[INTERCEPT] depth={depth_str} "
                f"offset=({dx_norm:+.2f},{dy_norm:+.2f}) "
                f"yaw_err={math.degrees(yaw_error):+.1f}° "
                f"yaw_rate={yaw_rate:+.2f}rad/s "
                f"fwd={forward_vel:.2f} "
                f"yaw={math.degrees(self.current_yaw):.1f}° "
                f"close={self.close_frame_count}"
            )

    # END STATE UPON INTERECPTION

    def _state_done(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)
        if not hasattr(self, '_done_time'):
            self._done_time = time.time()
        if time.time() - self._done_time > 3.0:
            self.get_logger().info("[DONE] Shutting down.")
            raise SystemExit(0)
        if self.loop_count % 20 == 0:
            remaining = 3.0 - (time.time() - self._done_time)
            self.get_logger().info(f"[DONE] Intercepted — shutting down in {remaining:.1f}s")



def main():
    rclpy.init()
    node = InterceptYawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nInterrupted — shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()