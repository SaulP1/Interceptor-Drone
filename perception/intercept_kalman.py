#!/usr/bin/env python3
"""
intercept_kalman.py — Camera + Kalman filter interception of a moving target.
                      Refactored for ModalAI Starling 2 Max hardware.
                      Uses Monocular Geometric Pinhole Depth (no ToF).

CRITICAL CONFIG REMINDER:
Verify TARGET_SIZE_M in interceptor_config.py matches your physical balloon.
If the balloon is 0.5 m but config says 0.3 m, the drone will think the target
is closer than it actually is and brake too early.

Frame Architecture:
World frame is ENU. The MAVROS pose topic publishes body orientation as a
FLU→ENU quaternion (ROS convention; MAVROS converts PX4's NED+FRD at the
bridge). The extrinsics in interceptor_config.py come from
voxl-inspect-extrinsics, which expresses everything in the VOXL FRD body
frame. We therefore convert R_CAM_TO_BODY and T_CAM_WRT_BODY from FRD to
FLU once at module load with R_FRD_TO_FLU = diag(1, -1, -1). All in-script
math from that point on is FLU-body / ENU-world.

Camera frame: OpenCV convention (+X right, +Y down, +Z forward along optical
axis). Pinhole back-projection multiplies the unnormalized direction vector
(u, v, 1) by Z directly — Z from f·W/w_pixels is perpendicular depth, not
slant range, so no normalization.
"""

import sys
import os
import json
import time
import math
import logging

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
    DETECTION_TOPIC,
    MAVROS_SETPOINT_TOPIC,
    MAVROS_POS_TOPIC,
    MAVROS_STATE_TOPIC,
    # IMX412 intrinsics @ 1024x768
    CAMERA_FX,
    CAMERA_FY,
    CAMERA_CX,
    CAMERA_CY,
    CAMERA_HFOV_RAD,
    # Extrinsics: hires_front wrt body — VOXL FRD convention
    T_CAM_WRT_BODY,
    R_CAM_TO_BODY,
    # Pinhole Depth Parameters
    TARGET_SIZE_M,
    # Flight parameters (single source of truth)
    TAKEOFF_ALT_M,
    FLIGHT_SPEED_MS,
    SETPOINT_RATE_HZ,
)


# ═══════════════════════════════════════════════════════════════
# FRD → FLU CONVERSION FOR EXTRINSICS
# ═══════════════════════════════════════════════════════════════
# Config extrinsics are in VOXL FRD body frame (X=forward, Y=right, Z=down).
# MAVROS body quaternion is FLU (X=forward, Y=left,  Z=up).
# Conversion: flip Y and Z. R_FRD_TO_FLU = diag(1, -1, -1).
#
# Sanity check on the resulting R_CAM_TO_BODY_FLU:
#   cam +Z (optical axis) → FLU +X (forward)   — unchanged, both forward
#   cam +X (image right)  → FLU -Y (i.e. right) — flipped from FRD +Y
#   cam +Y (image down)   → FLU -Z (i.e. down)  — flipped from FRD +Z

R_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])
R_CAM_TO_BODY_FLU = R_FRD_TO_FLU @ R_CAM_TO_BODY
T_CAM_WRT_BODY_FLU = R_FRD_TO_FLU @ T_CAM_WRT_BODY


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

file_logger = logging.getLogger("intercept_kalman")
file_logger.setLevel(logging.DEBUG)
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.DEBUG)
_sh.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"))
file_logger.addHandler(_sh)
file_logger.propagate = False

def flog(msg, level=logging.INFO):
    file_logger.log(level, msg)

def flog_debug(msg):
    file_logger.debug(msg)


# ═══════════════════════════════════════════════════════════════
# CONTROL PARAMETERS
# ═══════════════════════════════════════════════════════════════
# Parameters that also exist in interceptor_config.py are sourced
# from there. Script-local constants remain only for things the
# config does not define (PID gains, velocity ceilings, miss-
# detection thresholds) and for INTERCEPT_DEPTH, which differs
# semantically from the config's INTERCEPT_RADIUS_M (Z along the
# optical axis vs. 3D Euclidean distance — not interchangeable).

# Sourced from config:
TAKEOFF_ALT       = TAKEOFF_ALT_M       # m
MAX_FORWARD_SPEED = FLIGHT_SPEED_MS     # m/s — also acts as cruise cap
CONTROL_DT        = 1.0 / SETPOINT_RATE_HZ
MAVROS_CONNECT_LOOPS = int(3.0 * SETPOINT_RATE_HZ)  # ~3 s before arm attempt

# Script-local (not duplicated in config):
TAKEOFF_SETTLE_TIME = 8.0    # s

MAX_DECEL = 3.0   # m/s²
# NOTE: with MAX_DECEL=3 and current MAX_FORWARD_SPEED, the soft brake's
# stopping_limit (sqrt(2·a·d)) only undercuts MAX_FORWARD_SPEED at very
# small depths. The brake doesn't engage in practice. Lower MAX_DECEL
# (~0.5) to make this a real soft brake.

# Yaw PID
YAW_KP = 0.4
YAW_KI = 0.0
YAW_KD = 0.0

# Vertical PID
VERTICAL_KP = 0.3
VERTICAL_KI = 0.0
VERTICAL_KD = 0.0

# Lateral PID
LATERAL_KP = 0.0
LATERAL_KI = 0.0
LATERAL_KD = 0.0

# Interception
INTERCEPT_DEPTH          = 0.5   # m — perpendicular Z, not radius
INTERCEPT_CONFIRM_FRAMES = 5

# Miss detection
MISS_LOST_TIMEOUT        = 8.0   # s — generous; consider 2-3 s
MISS_MAX_FLIGHT_TIME     = 90.0  # s
MISS_DISTANCE_INCREASING = 3.0   # s

# Safety
MAX_ALTITUDE = 50.0   # m
MIN_ALTITUDE = 0.5    # m

# PX4 velocity limits — script-local hard ceilings, independent of
# MAX_FORWARD_SPEED. The clamp in _send_body_velocity_with_yaw uses
# MAX_VEL_HORIZONTAL as the absolute cap on E/N components after the
# body→world rotation.
MAX_VEL_HORIZONTAL = 1.0    # m/s
MAX_VEL_UP         = 0.2    # m/s
MAX_VEL_DOWN       = 0.3    # m/s

MAX_YAW_RATE       = 0.785  # rad/s

# Kalman filter
KF_MIN_OBSERVATIONS  = 10
PREDICT_AHEAD_FACTOR = 1.2


# ═══════════════════════════════════════════════════════════════
# KALMAN FILTER
# ═══════════════════════════════════════════════════════════════

class KalmanFilter3D:
    def __init__(self, process_noise=0.5, measurement_noise=0.3):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 10.0
        self.Q_base = np.eye(6) * process_noise
        self.Q_base[3, 3] = process_noise * 2.0
        self.Q_base[4, 4] = process_noise * 2.0
        self.Q_base[5, 5] = process_noise * 2.0
        self.R = np.eye(3) * measurement_noise
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.initialized       = False
        self.observation_count = 0
        self.last_update_time  = None

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        Q = self.Q_base * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_obs):
        if not self.initialized:
            self.x[:3]             = z_obs
            self.x[3:]             = 0.0
            self.initialized       = True
            self.observation_count = 1
            self.last_update_time  = time.time()
            flog(f"KF INIT | first obs: ({z_obs[0]:.2f},{z_obs[1]:.2f},{z_obs[2]:.2f})")
            return

        now = time.time()
        if self.last_update_time is not None:
            dt = now - self.last_update_time
            if dt > 0.001:
                self.predict(dt)
        self.last_update_time = now

        y = z_obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.observation_count += 1

        flog_debug(
            f"KF UPDATE #{self.observation_count} | "
            f"obs=({z_obs[0]:.2f},{z_obs[1]:.2f},{z_obs[2]:.2f}) "
            f"pos=({self.x[0]:.2f},{self.x[1]:.2f},{self.x[2]:.2f}) "
            f"vel=({self.x[3]:.2f},{self.x[4]:.2f},{self.x[5]:.2f})")

    def get_position(self, dt_extrap=0.0):
        """Return position, optionally extrapolated forward by dt_extrap seconds.
        Pass the time elapsed since last_update_time when comparing against
        a real-time drone pose, otherwise the comparison is biased by however
        stale the KF is."""
        return self.x[:3].copy() + self.x[3:].copy() * dt_extrap

    def get_velocity(self):
        return self.x[3:].copy()

    def predict_position(self, dt_future):
        """Extrapolate dt_future seconds beyond the last update."""
        return self.x[:3].copy() + self.x[3:].copy() * dt_future

    def has_velocity_estimate(self):
        return self.observation_count >= KF_MIN_OBSERVATIONS

    def time_since_update(self, now):
        if self.last_update_time is None:
            return 0.0
        return now - self.last_update_time


# ═══════════════════════════════════════════════════════════════
# PID CONTROLLER
# ═══════════════════════════════════════════════════════════════

class PIDController:
    def __init__(self, kp, ki, kd, integral_max=2.0):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.integral_max = integral_max
        self.integral     = 0.0
        self.prev_error   = 0.0
        self.first_call   = True

    def compute(self, error, dt):
        p_term = self.kp * error
        self.integral = max(-self.integral_max,
                            min(self.integral_max, self.integral + error * dt))
        i_term = self.ki * self.integral
        if self.first_call:
            derivative      = 0.0
            self.first_call = False
        else:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term          = self.kd * derivative
        self.prev_error = error
        return p_term + i_term + d_term

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.first_call = True


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

def quat_to_rotation_matrix(qx, qy, qz, qw):
    """Quaternion → 3×3 rotation matrix (body FLU → world ENU when fed
    the MAVROS local_position/pose orientation)."""
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


# ═══════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════════

class InterceptKalmanController(Node):
    def __init__(self):
        super().__init__("intercept_kalman_controller")

        self.state      = "INIT"
        self.prev_state = None
        self.mavros_state = State()

        # ENU pose from MAVROS
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_z   = 0.0
        self.current_yaw = 0.0
        self._enu_quat   = (0.0, 0.0, 0.0, 1.0)

        # Takeoff holding position
        self.takeoff_x = 0.0
        self.takeoff_y = 0.0
        self.target_yaw = 0.0

        self.start_time        = time.time()
        self.flight_start_time = None

        self.latest_detection = None
        self.detection_time   = 0.0
        self.detection_count  = 0

        # KF and detections live on the spin thread; no lock needed.
        self.kf = KalmanFilter3D(process_noise=0.5, measurement_noise=0.3)

        self.pid_yaw      = PIDController(YAW_KP,      YAW_KI,      YAW_KD)
        self.pid_vertical = PIDController(VERTICAL_KP, VERTICAL_KI, VERTICAL_KD)
        self.pid_lateral  = PIDController(LATERAL_KP,  LATERAL_KI,  LATERAL_KD)

        self.close_frame_count     = 0
        self.intercept_declared    = False
        self.last_control_time     = time.time()

        self.last_detection_seen       = 0.0
        self.min_distance_seen         = float('inf')
        self.distance_increasing_since = None
        self.loop_count = 0

        # Arm sequence state — tracks async chained service calls.
        self._arm_seq_started = False

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        detection_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        self.arm_client  = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode,     "/mavros/set_mode")

        self.create_subscription(
            State,       MAVROS_STATE_TOPIC,           self._on_state,       mavros_qos)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._on_pose_mavros, mavros_qos)
        self.create_subscription(
            String,      DETECTION_TOPIC,              self._on_detection,   detection_qos)

        self.vel_pub = self.create_publisher(Twist,       MAVROS_SETPOINT_TOPIC, 10)
        self.pos_pub = self.create_publisher(PoseStamped, MAVROS_POS_TOPIC,      mavros_qos)
        self.hit_pub = self.create_publisher(Bool,        "/interceptor/hit",    10)

        self.timer = self.create_timer(CONTROL_DT, self._control_loop)

        flog("=" * 60)
        flog("INTERCEPTOR — Kalman + IMX412 (Pinhole Depth) | Starling 2 Max")
        flog("=" * 60)
        flog(f"MAVROS pose:  /mavros/local_position/pose  (ENU world / FLU body)")
        flog(f"Pinhole:      fx={CAMERA_FX:.0f} target_size={TARGET_SIZE_M}m")
        flog(f"Extrinsics (FRD→FLU converted):")
        flog(f"  T_FLU = ({T_CAM_WRT_BODY_FLU[0]:+.3f},"
             f" {T_CAM_WRT_BODY_FLU[1]:+.3f}, {T_CAM_WRT_BODY_FLU[2]:+.3f}) m")
        flog(f"  R_FLU =\n{R_CAM_TO_BODY_FLU}")
        flog(f"PID Yaw:      Kp={YAW_KP} Ki={YAW_KI} Kd={YAW_KD}")
        flog(f"PID Vert:     Kp={VERTICAL_KP} Ki={VERTICAL_KI} Kd={VERTICAL_KD}")
        flog(f"Vel limits:   horiz={MAX_VEL_HORIZONTAL} m/s  "
             f"up={MAX_VEL_UP} m/s  down={MAX_VEL_DOWN} m/s")
        flog(f"Yaw rate:     {MAX_YAW_RATE:.3f} rad/s")
        flog("Waiting for MAVROS...")

        self.get_logger().info("INTERCEPTOR — Kalman + IMX412 | Starling 2 Max")

    def _log_state(self, new_state, reason=""):
        if new_state != self.prev_state:
            msg = f"STATE | {self.prev_state or 'NONE'} -> {new_state}"
            if reason:
                msg += f" | {reason}"
            msg += f" | pos=({self.current_x:.2f},{self.current_y:.2f},{self.current_z:.2f})"
            flog(msg)
            self.get_logger().info(msg)
            self.prev_state = new_state

    # ══════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════

    def _on_state(self, msg):
        old_mode  = self.mavros_state.mode
        old_armed = self.mavros_state.armed
        self.mavros_state = msg
        if msg.mode != old_mode:
            flog(f"MAVROS | mode {old_mode} -> {msg.mode}")
        if msg.armed != old_armed:
            status = "ARMED" if msg.armed else "DISARMED"
            flog(f"MAVROS | {status}")

    def _on_pose_mavros(self, msg):
        """MAVROS local position pose. Position is ENU world; orientation is
        FLU body → ENU world quaternion."""
        self.current_x   = msg.pose.position.x
        self.current_y   = msg.pose.position.y
        self.current_z   = msg.pose.position.z
        q                = msg.pose.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self._enu_quat   = (q.x, q.y, q.z, q.w)

    def _on_detection(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_detection = data
            self.detection_time   = time.time()
            self.detection_count += 1
        except (json.JSONDecodeError, KeyError):
            pass

    # ══════════════════════════════════════════════════════════
    # PINHOLE DEPTH
    # ══════════════════════════════════════════════════════════

    def _get_target_depth(self):
        """Estimate perpendicular depth (Z along optical axis) from bbox.

        Z = fx * W_real / w_pixels. Uses max(width, height) on the assumption
        that the larger dimension is closest to transverse to the optical
        axis. For a roughly spherical balloon this is fine; for a tilted
        oblong target at the image edge the pinhole approximation frays."""
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

    # ══════════════════════════════════════════════════════════
    # WORLD POSITION ESTIMATE
    # ══════════════════════════════════════════════════════════

    def _estimate_target_world_pos(self, depth):
        """Back-project detection pixel to ENU world position.

        Pipeline:
          pixel (u, v)  → camera frame ray (OpenCV, +X right, +Y down, +Z fwd)
          × depth Z     → camera-frame point (no normalization — depth is Z
                          along optical axis, not slant range)
          × R_CAM_TO_BODY_FLU → body-frame FLU point
          + T_CAM_WRT_BODY_FLU (rotated to world)
          + body position (ENU)
          → ENU world point
        """
        det = self.latest_detection
        if det is None or depth <= 0:
            return None

        cx_px, cy_px = det["center_px"]

        # Camera-frame point. NO normalization: f·W/w returns perpendicular
        # depth Z, so multiplying the (u, v, 1) ray by Z directly yields the
        # correct 3D point. Normalizing first would treat depth as slant
        # range and shrink off-axis points by 1/|d|.
        p_cam = np.array([
            (cx_px - CAMERA_CX) / CAMERA_FX,
            (cy_px - CAMERA_CY) / CAMERA_FY,
            1.0,
        ]) * depth

        qx, qy, qz, qw = self._enu_quat
        R_body_to_world = quat_to_rotation_matrix(qx, qy, qz, qw)

        body_pos = np.array([self.current_x, self.current_y, self.current_z])

        p_world = (body_pos
                   + R_body_to_world @ T_CAM_WRT_BODY_FLU
                   + R_body_to_world @ (R_CAM_TO_BODY_FLU @ p_cam))
        return p_world

    # ══════════════════════════════════════════════════════════
    # COMMAND HELPERS
    # ══════════════════════════════════════════════════════════

    def _clamp(self, v, limit):
        return max(-limit, min(limit, v))

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
        """Convert body FRU command (forward, right, up) to ENU world velocity
        and publish on /mavros/setpoint_velocity/cmd_vel_unstamped."""
        yaw = self.current_yaw
        ve  = forward * math.cos(yaw) + right * math.sin(yaw)
        vn  = forward * math.sin(yaw) - right * math.cos(yaw)
        msg = Twist()
        msg.linear.x  = self._clamp(float(ve), MAX_VEL_HORIZONTAL)
        msg.linear.y  = self._clamp(float(vn), MAX_VEL_HORIZONTAL)
        msg.linear.z  = max(-MAX_VEL_DOWN, min(MAX_VEL_UP, float(up)))
        msg.angular.z = self._clamp(float(yaw_rate), MAX_YAW_RATE)
        self.vel_pub.publish(msg)

    # ══════════════════════════════════════════════════════════
    # CONTROL LOOP
    # ══════════════════════════════════════════════════════════

    def _control_loop(self):
        self.loop_count += 1
        dispatch = {
            "INIT":      self._state_init,
            "TAKEOFF":   self._state_takeoff,
            "ACQUIRE":   self._state_acquire,
            "INTERCEPT": self._state_intercept,
            "DONE":      self._state_done,
            "MISS":      self._state_miss,
        }
        dispatch.get(self.state, lambda: None)()

    # ── INIT ──

    def _transition_to_takeoff(self, reason):
        """Helper to lock in takeoff coordinates before transitioning."""
        self.takeoff_x = self.current_x
        self.takeoff_y = self.current_y
        self.target_yaw = self.current_yaw
        self.state = "TAKEOFF"
        self._log_state("TAKEOFF", reason)
        self.start_time = time.time()

    def _state_init(self):
        self._log_state("INIT", "startup")
        self._send_position_yaw(self.current_x, self.current_y, TAKEOFF_ALT, self.current_yaw)

        if self.mavros_state.mode == "":
            return

        if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
            self._transition_to_takeoff("armed and offboard")
            return

        if self.loop_count == MAVROS_CONNECT_LOOPS and not self._arm_seq_started:
            self._arm_seq_started = True
            self._begin_arm_sequence()

    # ── ARM SEQUENCE (async, no extra threads) ──
    #
    # Original code spawned a worker thread that called
    # rclpy.spin_until_future_complete from off-main-thread while
    # main was already spinning the same node — undefined behavior on
    # SingleThreadedExecutor. Replaced with add_done_callback chaining;
    # all callbacks fire on the main spin thread.

    def _begin_arm_sequence(self):
        if not self.mode_client.service_is_ready():
            flog("ARM SEQ | /mavros/set_mode not ready", logging.WARNING)
            self._arm_seq_started = False  # allow retry on next tick
            return
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.mode_client.call_async(req)
        future.add_done_callback(self._after_set_mode)

    def _after_set_mode(self, future):
        try:
            result = future.result()
            flog(f"ARM SEQ | OFFBOARD mode_sent={getattr(result, 'mode_sent', '?')}")
        except Exception as e:
            flog(f"ARM SEQ | set_mode failed: {e}", logging.WARNING)
            return

        if not self.arm_client.service_is_ready():
            flog("ARM SEQ | /mavros/cmd/arming not ready", logging.WARNING)
            return

        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        future.add_done_callback(self._after_arm)

    def _after_arm(self, future):
        try:
            result = future.result()
            flog(f"ARM SEQ | arming success={getattr(result, 'success', '?')}")
        except Exception as e:
            flog(f"ARM SEQ | arm failed: {e}", logging.WARNING)

    # ── TAKEOFF ──

    def _state_takeoff(self):
        self._send_position_yaw(self.takeoff_x, self.takeoff_y, TAKEOFF_ALT, self.target_yaw)
        elapsed = time.time() - self.start_time
        if elapsed > TAKEOFF_SETTLE_TIME and self.current_z > TAKEOFF_ALT * 0.5:
            self.state             = "ACQUIRE"
            self._log_state("ACQUIRE", f"takeoff done at {self.current_z:.2f}m")
            self.start_time        = time.time()
            self.flight_start_time = time.time()

    # ── ACQUIRE ──

    def _state_acquire(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)

        det      = self.latest_detection
        det_time = self.detection_time
        det_age  = time.time() - det_time if det else 999.0

        if det is not None and det_age < 1.0:
            depth = self._get_target_depth()
            if depth > 0:
                world_pos = self._estimate_target_world_pos(depth)
                if world_pos is not None:
                    self.kf.update(world_pos)

                    if self.kf.has_velocity_estimate():
                        self.pid_yaw.reset()
                        self.pid_vertical.reset()
                        self.pid_lateral.reset()
                        self.last_control_time   = time.time()
                        self.close_frame_count   = 0
                        self.last_detection_seen = time.time()
                        self.state = "INTERCEPT"
                        self._log_state("INTERCEPT", "Kalman velocity ready")
                        return

        elapsed = time.time() - self.start_time
        if elapsed > 30.0:
            self.state = "MISS"
            self._log_state("MISS", "acquire timeout")

    # ── INTERCEPT ──

    def _state_intercept(self):
        now = time.time()
        dt  = max(0.01, min(0.1, now - self.last_control_time))
        self.last_control_time = now
        flight_elapsed = now - self.flight_start_time if self.flight_start_time else 0.0

        if self.flight_start_time and flight_elapsed > MISS_MAX_FLIGHT_TIME:
            self.state = "MISS"
            self._log_state("MISS", "flight time limit")
            return

        det      = self.latest_detection
        det_time = self.detection_time
        det_age  = now - det_time if det else 999.0

        # Cache depth calculation
        target_depth = -1.0
        if det is not None and det_age < 1.0:
            self.last_detection_seen = now
            target_depth = self._get_target_depth()
            if target_depth > 0:
                world_pos = self._estimate_target_world_pos(target_depth)
                if world_pos is not None:
                    self.kf.update(world_pos)

        lost_duration = now - self.last_detection_seen
        if lost_duration > MISS_LOST_TIMEOUT:
            self.state = "MISS"
            self._log_state("MISS", f"lost {lost_duration:.1f}s")
            return

        # KF position extrapolated to "now" — without this, the distance and
        # overshoot logic compares real-time drone pose against a KF snapshot
        # timestamped at the last detection.
        kf_age  = self.kf.time_since_update(now)
        kf_pos  = self.kf.get_position(dt_extrap=kf_age)

        if det is not None and det_age < 2.0:
            # Active Visual Servoing Branch
            dx_norm, dy_norm = det["offset_norm"]

            yaw_error = dx_norm * (CAMERA_HFOV_RAD / 2.0)
            yaw_rate  = self._clamp(-self.pid_yaw.compute(yaw_error, dt), MAX_YAW_RATE)

            vertical_vel = -self.pid_vertical.compute(dy_norm, dt)
            if self.current_z > MAX_ALTITUDE: vertical_vel = min(vertical_vel, 0.0)
            if self.current_z < MIN_ALTITUDE: vertical_vel = max(vertical_vel, 0.0)

            right_vel = self.pid_lateral.compute(dx_norm, dt)

            dist = 0.0
            if target_depth > 0:
                drone_pos = np.array([self.current_x, self.current_y, self.current_z])
                dist      = np.linalg.norm(kf_pos - drone_pos)

                if dist < self.min_distance_seen:
                    self.min_distance_seen         = dist
                    self.distance_increasing_since = None
                elif self.distance_increasing_since is None:
                    self.distance_increasing_since = now
                elif now - self.distance_increasing_since > MISS_DISTANCE_INCREASING:
                    self.state = "MISS"
                    self._log_state("MISS", "overshoot")
                    return

                # Soft brake using stopping distance under MAX_DECEL.
                # NOTE: With current params this rarely undercuts MAX_FORWARD_SPEED
                # (see top of file). Lower MAX_DECEL to make it bite earlier.
                stopping_limit = math.sqrt(2 * MAX_DECEL * max(0.1, target_depth))
                forward_vel    = min(stopping_limit, MAX_FORWARD_SPEED)

                if target_depth < INTERCEPT_DEPTH:
                    self.close_frame_count += 1
                    forward_vel = 0.0
                    if self.close_frame_count >= INTERCEPT_CONFIRM_FRAMES:
                        self.state              = "DONE"
                        self._log_state("DONE", f"intercepted depth={target_depth:.2f}m")
                        self.intercept_declared = True
                        hit_msg      = Bool()
                        hit_msg.data = True
                        self.hit_pub.publish(hit_msg)
                        return
                else:
                    self.close_frame_count = 0
            else:
                forward_vel = MAX_FORWARD_SPEED * 0.3

            self._send_body_velocity_with_yaw(
                forward_vel, right_vel, vertical_vel, yaw_rate)

        else:
            # Blind Pursuit Branch
            if self.kf.has_velocity_estimate():
                drone_pos      = np.array([self.current_x, self.current_y, self.current_z])
                dist           = np.linalg.norm(kf_pos - drone_pos)
                time_to_target = dist / max(MAX_FORWARD_SPEED * 0.5, 0.1)
                # predict_position extrapolates from last update; add kf_age
                # so the prediction lands time_to_target seconds from now,
                # not from the last (potentially old) detection.
                predicted      = self.kf.predict_position(
                    kf_age + time_to_target * PREDICT_AHEAD_FACTOR)

                dx = predicted[0] - self.current_x
                dy = predicted[1] - self.current_y
                dz = predicted[2] - self.current_z

                target_yaw = math.atan2(dy, dx)
                yaw_error  = target_yaw - self.current_yaw
                while yaw_error >  math.pi: yaw_error -= 2 * math.pi
                while yaw_error < -math.pi: yaw_error += 2 * math.pi

                yaw_rate     = self._clamp(yaw_error * YAW_KP, MAX_YAW_RATE)
                vertical_vel = dz * 2.0
                forward_vel  = min(dist * 1.5, MAX_FORWARD_SPEED)

                self._send_body_velocity_with_yaw(
                    forward_vel, 0.0, vertical_vel, yaw_rate)
            else:
                self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)

    # ── DONE ──

    def _state_done(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, 0.0, 0.0)
        if not hasattr(self, '_done_time'):
            self._done_time = time.time()
        if time.time() - self._done_time > 3.0:
            raise SystemExit(0)

    # ── MISS ──

    def _state_miss(self):
        self._send_body_velocity_with_yaw(0.0, 0.0, -MAX_VEL_DOWN, 0.0)
        if not hasattr(self, '_miss_time'):
            self._miss_time = time.time()
        if self.current_z < 0.2 or (time.time() - self._miss_time > 15.0):
            raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = InterceptKalmanController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()