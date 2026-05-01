#!/usr/bin/env python3
"""
intercept_yaw.py — Autonomous visual interception with yaw tracking.
                   Refactored for physical hardware (Starling 2 Max).

The drone TURNS to face the target instead of strafing sideways.
This is closer to how a real interceptor works:
  1. Rotate to point camera at target (yaw control)
  2. Adjust altitude to match target height (vertical control)
  3. Fly forward — speed based on depth (fast when far, slow when close)
  4. Declare interception when depth < threshold

Modifications for physical hardware:
  - Removed Isaac Sim depth dependency (/quadrotor/owl/depth).
  - Uses Geometric Pinhole Depth via IMX412 bounding box size.
  - Enforces PX4 MAX_VEL limits to prevent pitch-coupling fly-aways.
"""

import sys
import os
import json
import time
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "config"))
from interceptor_config import (
    # Topics
    DETECTION_TOPIC,
    MAVROS_SETPOINT_TOPIC,
    MAVROS_POS_TOPIC,
    MAVROS_STATE_TOPIC,
    # Camera & Target Math
    CAMERA_HFOV_RAD,
    CAMERA_FX,
    TARGET_SIZE_M,
    # Flight parameters
    TAKEOFF_ALT_M,
    FLIGHT_SPEED_MS,
    SETPOINT_RATE_HZ,
    # Safety Limits
    MAX_VEL_HORIZONTAL,
    MAX_VEL_UP,
    MAX_VEL_DOWN
)

# ═══════════════════════════════════════════════════════════════
# CONTROL PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Sourced from config:
TAKEOFF_ALT       = TAKEOFF_ALT_M           # m
MAX_FORWARD_SPEED = FLIGHT_SPEED_MS         # m/s — also acts as cruise cap
CONTROL_DT        = 1.0 / SETPOINT_RATE_HZ  # s
MAVROS_CONNECT_LOOPS = int(3.0 * SETPOINT_RATE_HZ)  # ~3 s of pre-stream

# Takeoff
TAKEOFF_SETTLE_TIME = 3.0      # s

# Forward speed
MAX_DECEL = 2.0                # m/s² — max braking deceleration

# YAW PID — controls how aggressively the drone turns to face the target
YAW_KP = 1.2                   # rad/s per unit offset
YAW_KI = 0.05                  # fixes persistent yaw drift
YAW_KD = 1.5                   # dampens yaw oscillation

# VERTICAL PID — up/down to match target height
# Softened significantly to prevent aggressive climbing when 
# the drone pitches forward (pitch-coupling).
VERTICAL_KP = 0.3
VERTICAL_KI = 0.0
VERTICAL_KD = 0.0

# Small lateral PID for fine corrections (residual after yaw)
LATERAL_KP = 1.0
LATERAL_KI = 0.02
LATERAL_KD = 0.15

# Interception
INTERCEPT_DEPTH = 0.3          # m
INTERCEPT_CONFIRM_FRAMES = 15

# Search
HOVER_TIMEOUT = 30.0           # s

# Safety
MAX_ALTITUDE = 50.0            # m
MIN_ALTITUDE = 0.3             # m
MAX_YAW_RATE = 1.5             # rad/s


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

    State machine: INIT → TAKEOFF → INTERCEPT → DONE
                                  ↘ SEARCH ↗
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
        self.target_yaw = 0.0
        
        # Takeoff holding variables
        self.takeoff_x = 0.0
        self.takeoff_y = 0.0
        
        self.start_time = time.time()

        # ── Detection ──
        self.latest_detection = None
        self.detection_time = 0.0

        # ── PID controllers ──
        self.pid_yaw = PIDController(YAW_KP, YAW_KI, YAW_KD)
        self.pid_vertical = PIDController(VERTICAL_KP, VERTICAL_KI, VERTICAL_KD)
        self.pid_lateral = PIDController(LATERAL_KP, LATERAL_KI, LATERAL_KD)

        # ── Intercept tracking ──
        self.close_frame_count = 0
        self.intercept_declared = False
        self.last_control_time = time.time()

        # ── Arm sequence state ──
        self._arm_seq_started = False

        # ── QoS ──
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Services
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # ── Subscribers ──
        self.create_subscription(State, MAVROS_STATE_TOPIC, self._on_mavros_state, mavros_qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self._on_pose, mavros_qos)
        self.create_subscription(String, DETECTION_TOPIC, self._on_detection, 10)

        # ── Publishers ──
        self.vel_pub = self.create_publisher(Twist, MAVROS_SETPOINT_TOPIC, 10)
        self.pos_pub = self.create_publisher(PoseStamped, MAVROS_POS_TOPIC, mavros_qos)
        self.hit_pub = self.create_publisher(Bool, "/interceptor/hit", 10)

        # ── Control loop ──
        self.timer = self.create_timer(CONTROL_DT, self._control_loop)
        self.loop_count = 0

        self.get_logger().info("=" * 55)
        self.get_logger().info("  INTERCEPTOR — Yaw Tracking Controller (Hardware Mode)")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  Pinhole Depth: fx={CAMERA_FX} tgt_size={TARGET_SIZE_M}m")
        self.get_logger().info(f"  Yaw PID: P={YAW_KP} I={YAW_KI} D={YAW_KD}")
        self.get_logger().info(f"  Vertical PID: P={VERTICAL_KP} I={VERTICAL_KI} D={VERTICAL_KD}")
        self.get_logger().info(f"  Max fwd speed: {MAX_FORWARD_SPEED} m/s")
        self.get_logger().info(f"  Velocity Caps: Up={MAX_VEL_UP} Down={MAX_VEL_DOWN} Horiz={MAX_VEL_HORIZONTAL}")
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
            self.latest_detection = data
            self.detection_time = time.time()
        except (json.JSONDecodeError, KeyError):
            pass

    # ═══════════════════════════════════════════════════════
    # DEPTH ESTIMATION
    # ═══════════════════════════════════════════════════════

    def _get_target_depth(self):
        """Estimate distance using Geometric Pinhole Method."""
        det = self.latest_detection
        if det is None:
            return -1.0

        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            return -1.0

        pixel_width = bbox[2] - bbox[0]
        pixel_height = bbox[3] - bbox[1]
        max_pixel_dim = max(pixel_width, pixel_height)

        if max_pixel_dim <= 0:
            return -1.0

        depth = (CAMERA_FX * TARGET_SIZE_M) / max_pixel_dim
        return float(depth)

    # ═══════════════════════════════════════════════════════
    # COMMAND HELPERS
    # ═══════════════════════════════════════════════════════

    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def _send_position_yaw(self, x, y, z, yaw):
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
        yaw = self.current_yaw
        vel_east = forward * math.cos(yaw) + right * math.sin(yaw)
        vel_north = forward * math.sin(yaw) - right * math.cos(yaw)

        msg = Twist()
        # Clamped safely to physical PX4 hardware limits
        msg.linear.x = self._clamp(float(vel_east), MAX_VEL_HORIZONTAL)
        msg.linear.y = self._clamp(float(vel_north), MAX_VEL_HORIZONTAL)
        msg.linear.z = max(-MAX_VEL_DOWN, min(MAX_VEL_UP, float(up)))
        msg.angular.z = self._clamp(float(yaw_rate), MAX_YAW_RATE)
        self.vel_pub.publish(msg)

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
        self._send_position_yaw(self.current_x, self.current_y, TAKEOFF_ALT, self.current_yaw)

        if self.mavros_state.mode == "":
            return

        if self.loop_count == 1:
            self.get_logger().info(f"Connected. Mode: {self.mavros_state.mode}")

        if self.loop_count < MAVROS_CONNECT_LOOPS:
            return

        if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
            self.get_logger().info("Armed and OFFBOARD — taking off")
            self.takeoff_x = self.current_x
            self.takeoff_y = self.current_y
            self.target_yaw = self.current_yaw
            self.state = "TAKEOFF"
            self.start_time = time.time()
            return

        if self.loop_count == MAVROS_CONNECT_LOOPS and not self._arm_seq_started:
            self.get_logger().info("Setting OFFBOARD and arming...")
            self._arm_seq_started = True
            self._begin_arm_sequence()

    def _begin_arm_sequence(self):
        if not self.mode_client.service_is_ready():
            self.get_logger().warn("ARM SEQ | /mavros/set_mode not ready — will retry")
            self._arm_seq_started = False  
            return
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.mode_client.call_async(req)
        future.add_done_callback(self._after_set_mode)

    def _after_set_mode(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"Set mode result: mode_sent={getattr(result, 'mode_sent', '?')}")
        except Exception as e:
            self.get_logger().warn(f"Set mode failed: {e}")
            return

        if not self.arm_client.service_is_ready():
            self.get_logger().warn("ARM SEQ | /mavros/cmd/arming not ready")
            return

        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        future.add_done_callback(self._after_arm)

    def _after_arm(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"Arm result: success={getattr(result, 'success', '?')}")
        except Exception as e:
            self.get_logger().warn(f"Arming failed: {e}")

    # ── TAKEOFF ──

    def _state_takeoff(self):
        self._send_position_yaw(self.takeoff_x, self.takeoff_y, TAKEOFF_ALT, self.target_yaw)
        elapsed = time.time() - self.start_time

        if self.loop_count % 40 == 0:
            self.get_logger().info(
                f"[TAKEOFF] alt={self.current_z:.2f}m target={TAKEOFF_ALT}m "
                f"yaw={math.degrees(self.current_yaw):.1f}° ({elapsed:.0f}s)"
            )

        if elapsed > TAKEOFF_SETTLE_TIME and self.current_z > TAKEOFF_ALT * 0.5:
            self.get_logger().info(f"Takeoff complete — alt={self.current_z:.2f}m. Searching...")
            self.pid_yaw.reset()
            self.pid_vertical.reset()
            self.pid_lateral.reset()
            self.last_control_time = time.time()
            self.state = "SEARCH"
            self.close_frame_count = 0
            self.start_time = time.time()

    # ── SEARCH ──

    def _state_search(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)

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

    # ── INTERCEPT (yaw tracking) ──

    def _state_intercept(self):
        now = time.time()
        dt = max(0.01, min(0.1, now - self.last_control_time))
        self.last_control_time = now

        det = self.latest_detection
        det_time = self.detection_time
        det_age = now - det_time if det is not None else 999.0

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

        # ── YAW ──
        yaw_error = dx_norm * (CAMERA_HFOV_RAD / 2.0)
        yaw_rate = -self.pid_yaw.compute(yaw_error, dt)
        yaw_rate = self._clamp(yaw_rate, MAX_YAW_RATE)

        # ── VERTICAL ──
        vertical_vel = -self.pid_vertical.compute(dy_norm, dt)
        if self.current_z > MAX_ALTITUDE:
            vertical_vel = min(vertical_vel, 0.0)
        if self.current_z < MIN_ALTITUDE:
            vertical_vel = max(vertical_vel, 0.0)

        # ── LATERAL ──
        right_vel = self.pid_lateral.compute(dx_norm, dt)

        # ── FORWARD ──
        if target_depth > 0:
            platform_limit = MAX_FORWARD_SPEED
            stopping_limit = math.sqrt(2 * MAX_DECEL * max(0.1, target_depth))
            forward_vel = min(platform_limit, stopping_limit)

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
            forward_vel = MAX_FORWARD_SPEED * 0.3 # Fallback speed

        self._send_body_velocity_with_yaw(forward_vel, right_vel, vertical_vel, yaw_rate)

        if self.loop_count % 20 == 0:
            depth_str = f"{target_depth:.2f}m" if target_depth > 0 else "N/A"
            self.get_logger().info(
                f"[INTERCEPT] depth={depth_str} "
                f"offset=({dx_norm:+.2f},{dy_norm:+.2f}) "
                f"yaw_rate={yaw_rate:+.2f} "
                f"vert_vel={vertical_vel:+.2f} "
                f"fwd={forward_vel:.2f}"
            )

    # ── DONE ──

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
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("\nInterrupted — shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()