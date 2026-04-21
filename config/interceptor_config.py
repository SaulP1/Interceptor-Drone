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
LOGS_DIR       = f"{PROJECT_DIR}/logs"
DEBUG_FRAMES   = f"{LOGS_DIR}/debug_frames"
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
# Camera topics — published by Pegasus ROS2CameraGraph inside Isaac
# These already exist when cam_test1.py runs.

CAMERA_RGB_TOPIC   = "/quadrotor/owl/rgb"
CAMERA_DEPTH_TOPIC = "/quadrotor/owl/depth"
CAMERA_INFO_TOPIC  = "/quadrotor/owl/camera_info"

# Drone state — published by Pegasus ROS2Backend
DRONE_POSE_TOPIC   = "/drone00/state/pose"

# Target state — published by world_moving_target.py (Phase 2)
TARGET_POSE_TOPIC     = "/interceptor/target_pose"
TARGET_VELOCITY_TOPIC = "/interceptor/target_velocity"

# Detection output — published by camera_viewer.py
DETECTION_TOPIC = "/interceptor/detection"

# MAVROS topics — used by flight scripts
MAVROS_SETPOINT_TOPIC = "/mavros/setpoint_position/local"
MAVROS_POSE_TOPIC     = "/mavros/local_position/pose"
MAVROS_STATE_TOPIC    = "/mavros/state"


# ═══════════════════════════════════════════════════════════════
# CAMERA PARAMETERS
# ═══════════════════════════════════════════════════════════════
# The Owl camera's actual intrinsics will come from /camera_info,
# but we define defaults here for reference and fallback.

CAMERA_IMG_W = 1280
CAMERA_IMG_H = 720


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
    (np.array([0,   200, 40]), np.array([10,  255, 255])),   # low-hue red
    (np.array([160, 200, 40]), np.array([179, 255, 255])),   # high-hue red
]

# Minimum blob area in pixels to count as a valid detection.
# Too small = noise. Too large = misses distant targets.
# Tune based on how far away the target is.
MIN_DETECTION_AREA_PX = 100
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
FLIGHT_SPEED_MS   = 2.0     # cruise speed for interception
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