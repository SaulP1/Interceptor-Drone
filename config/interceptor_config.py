"""
intercepter_config.py — Single source of truth for the Intercepter project.

Every script imports from here. No magic numbers scattered across files.

Inspired by: Seabird's seabird_config.py (same idea — one config file
for the whole project). Adapted for the Intercepter's different hardware,
topics, and mission goals.

Location: ~/interceptor/config/interceptor_config.py
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

HOME = os.path.expanduser("~")
PROJECT_DIR    = f"{HOME}/interceptor"
ISAAC_SCRIPTS  = f"{PROJECT_DIR}/isaac_scripts"
PERCEPTION_DIR = f"{PROJECT_DIR}/perception"
CONFIG_DIR     = f"{PROJECT_DIR}/config"
LOGS_DIR     = "/app/logs"
DEBUG_FRAMES = "/app/logs/debug_frames"
MODELS_DIR     = f"{PROJECT_DIR}/models"


# ═══════════════════════════════════════════════════════════════
# ISAAC SIM PRIM PATHS
# ═══════════════════════════════════════════════════════════════

DRONE_PRIM_PATH  = "/World/quadrotor"
DRONE_BODY_PATH  = "/World/quadrotor/body"
CAMERA_PRIM_PATH = "/World/quadrotor/body/owl/camera"
TARGET_PRIM_PATH = "/World/target_block"


# ═══════════════════════════════════════════════════════════════
# ROS2 TOPICS
# ═══════════════════════════════════════════════════════════════

CAMERA_RGB_TOPIC   = "/hires_front_small_color"
CAMERA_DEPTH_TOPIC = "/tof_pc"
CAMERA_CONF_TOPIC  = "/tof_conf"
DRONE_POSE_TOPIC   = "/vvhub_body_wrt_local/pose"
DETECTION_TOPIC    = "/interceptor/detection"

MAVROS_SETPOINT_TOPIC = "/mavros/setpoint_velocity/cmd_vel_unstamped"
MAVROS_POS_TOPIC      = "/mavros/setpoint_position/local"
MAVROS_STATE_TOPIC    = "/mavros/state"

# ═══════════════════════════════════════════════════════════════
# CAMERA INTRINSICS — IMX412 @ 1024x768 (hires_front_small_color)
# Estimated from EXIF 35mm-equivalent focal length (17mm).
# fx/fy = (width/2) / tan(HFOV/2). Replace with calibrated values
# once cameracalibrator is run against a checkerboard.
# ═══════════════════════════════════════════════════════════════

CAMERA_IMG_W = 1024
CAMERA_IMG_H = 768
CAMERA_FX    = 483.0   # pixels
CAMERA_FY    = 483.0   # pixels
CAMERA_CX    = 512.0   # pixels
CAMERA_CY    = 384.0   # pixels

# Derived FOV — used as fallback if not computing from intrinsics
import math
CAMERA_HFOV_RAD = 2.0 * math.atan(CAMERA_IMG_W / (2.0 * CAMERA_FX))  # ~93 deg
CAMERA_VFOV_RAD = 2.0 * math.atan(CAMERA_IMG_H / (2.0 * CAMERA_FY))  # ~77 deg

# ═══════════════════════════════════════════════════════════════
# EXTRINSICS — hires_front (IMX412) wrt body frame
# Source: voxl-inspect-extrinsics, chaining body→imu_apps (#6)
# + imu_apps→hires_front (#3). imu_apps→body is identity rotation.
#
# T_cam_wrt_body: body→imu (0.030,-0.006,-0.012)
#               + imu→cam  (0.041, 0.006, 0.019)
#               = (0.071, 0.000, 0.007) meters
#
# R_cam_to_body: R_child_to_parent from entry #3
#   camera +Z (optical axis) → body +X (forward)
#   camera +X (image right)  → body +Y (right)
#   camera +Y (image down)   → body +Z (down)
# ═══════════════════════════════════════════════════════════════

import numpy as np

T_CAM_WRT_BODY = np.array([0.071, 0.000, 0.007])  # meters

R_CAM_TO_BODY = np.array([
    [0.0,  0.0,  1.0],
    [1.0,  0.0,  0.0],
    [0.0,  1.0,  0.0],
])

# ═══════════════════════════════════════════════════════════════
# TOF PARAMETERS
# ═══════════════════════════════════════════════════════════════

# /tof_pc grid dimensions (matches voxl-camera-server output)
TOF_WIDTH  = 180
TOF_HEIGHT = 240

# Half-angle of cone used to select ToF points along detection ray (degrees)
# Wider = more points averaged, less sensitive to pointing error.
# Narrower = more precise but needs good ray direction.
TOF_CONE_HALF_ANGLE_DEG = 5.0


# ═══════════════════════════════════════════════════════════════
# TARGET DETECTION — HSV Color Ranges
# ═══════════════════════════════════════════════════════════════
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
#
# Red wraps around the hue wheel, so we need two ranges.
# These are starting values — tune with camera_viewer.py's
# debug output if detection is unreliable.
#
# Referenced from: Seabird's buoy_detector.py COLOR_RANGES
# (same idea — HSV ranges for colored objects. Adapted for
# our target color and Isaac Sim's lighting.)

HSV_RED_RANGES = [
    (np.array([0,   160, 60]),  np.array([12,  255, 255])),  # lower red wrap
    (np.array([168, 160, 60]),  np.array([179, 255, 255])),  # upper red wrap
]

# Minimum blob area in pixels to count as a valid detection.
# Too small = noise. Too large = misses distant targets.
# Tune based on how far away the target is.
MIN_DETECTION_AREA_PX = 5000
MAX_DETECTION_AREA_PX = 500000


# ═══════════════════════════════════════════════════════════════
# TARGET PHYSICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════

TARGET_SIZE_M = 0.3            # side length of the cube (meters)
TARGET_COLOR_RGB = (1.0, 0.0, 0.0)  # red in USD (0-1 range)

# Static target position (matches cam_test1.py red block)
#TARGET_STATIC_POS = (2.0, 0.0, 0.5)
TARGET_STATIC_POS = (4.0, 2.0, 0.5)
# Moving target defaults (Phase 2)
TARGET_SPEED_MS      = 1.0     # meters per second
TARGET_X_RANGE       = (-3.0, 3.0)   # random start X range
TARGET_Y_MIN         = 3.0     # minimum distance in front of drone
TARGET_Y_MAX         = 8.0     # maximum distance in front of drone
TARGET_Z             = 0.5     # height above ground


# ═══════════════════════════════════════════════════════════════
# DRONE SPAWN
# ═══════════════════════════════════════════════════════════════

DRONE_SPAWN_POS = [0.0, 0.0, 0.07]   # matches cam_test1.py


# ═══════════════════════════════════════════════════════════════
# FLIGHT PARAMETERS
# ═══════════════════════════════════════════════════════════════

TAKEOFF_ALT_M     = 0.5     # default takeoff altitude
FLIGHT_SPEED_MS   = 6.0     # cruise speed for interception
SETPOINT_RATE_HZ  = 20      # must be > 2 Hz for PX4 OFFBOARD

# Position tolerance for "arrived at waypoint"
XY_TOLERANCE_M    = 0.3
Z_TOLERANCE_M     = 0.5

# Interception: how close to target counts as "intercepted"
INTERCEPT_RADIUS_M = 0.5


# ═══════════════════════════════════════════════════════════════
# DEBUG / LOGGING
# ═══════════════════════════════════════════════════════════════

SAVE_DEBUG_FRAMES    = True
DEBUG_FRAME_INTERVAL = 10    # save every Nth frame
MAX_DEBUG_FRAMES     = 500


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════

DETECTION_MODE = 'color'

YOLO_MODEL_PATH = "yolov8s.pt"
YOLO_CONFIDENCE = 0.6
YOLO_TARGET_CLASS = "car"
YOLO_INPUT_SIZE = 1280

def print_config():
    """Print config summary for verification."""
    print(f"[config] Project dir: {PROJECT_DIR}")
    print(f"[config] Camera topics: {CAMERA_RGB_TOPIC}, {CAMERA_DEPTH_TOPIC}")
    print(f"[config] Camera res: {CAMERA_IMG_W}x{CAMERA_IMG_H}")
    print(f"[config] Target color: RED")
    print(f"[config] Drone spawn: {DRONE_SPAWN_POS}")
    print(f"[config] Takeoff alt: {TAKEOFF_ALT_M}m")