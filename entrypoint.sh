#!/bin/bash
source /opt/ros/humble/setup.bash

case "$1" in
    camera_viewer)
        echo "[docker] Starting camera_viewer.py"
        exec python3 /app/perception/camera_viewer.py
        ;;
    intercept_static)
        echo "[docker] Starting intercept_static.py"
        exec python3 /app/perception/intercept_static.py
        ;;
    intercept_yaw)
        echo "[docker] Starting intercept_yaw.py"
        exec python3 /app/perception/intercept_yaw.py
        ;;
    keyboard_fly)
        echo "[docker] Starting keyboard_fly.py"
        exec python3 /app/perception/keyboard_fly.py
        ;;
    bash)
        exec /bin/bash
        ;;
    *)
        echo "Usage: docker run interceptor {camera_viewer|intercept_static|intercept_yaw|keyboard_fly|bash}"
        exit 1
        ;;
esac