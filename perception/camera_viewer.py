#!/usr/bin/env python3
"""
camera_viewer.py — ROS2 node that sees what the drone camera sees.

Subscribes to the drone's camera RGB topic, runs color detection,
draws the detection overlay, publishes detection data for the flight
script, and saves annotated frames to disk.

Referenced from: Seabird's sim_camera.py
  - Same ROS2 subscriber pattern for Isaac camera topics
  - Same frame conversion (ROS2 Image -> numpy)
  - Same QoS profile (BEST_EFFORT for Isaac sensor publishers)
Also referenced: pi_tracker.py (center-line visualization)

Run:
    source /opt/ros/humble/setup.bash
    python3 ~/interceptor/perception/camera_viewer.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "config"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "perception"))

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from interceptor_config import (
    CAMERA_RGB_TOPIC,
    DETECTION_TOPIC,
    DEBUG_FRAMES,
    SAVE_DEBUG_FRAMES,
    DEBUG_FRAME_INTERVAL,
    MAX_DEBUG_FRAMES,
    print_config,
)
from color_detector import ColorDetector


class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer")

        self.detector = ColorDetector()
        self.frame_count = 0
        self.detection_count = 0
        self.saved_count = 0
        self.last_log_time = time.time()

        os.makedirs(DEBUG_FRAMES, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image, CAMERA_RGB_TOPIC, self._on_image, sensor_qos,
        )

        self.detection_pub = self.create_publisher(
            String, DETECTION_TOPIC, 10,
        )

        self.get_logger().info("=" * 50)
        self.get_logger().info("Camera Viewer started")
        self.get_logger().info(f"  Subscribing to: {CAMERA_RGB_TOPIC}")
        self.get_logger().info(f"  Publishing to:  {DETECTION_TOPIC}")
        self.get_logger().info(f"  Debug frames:   {DEBUG_FRAMES}")
        self.get_logger().info("  Waiting for camera frames...")
        self.get_logger().info("=" * 50)

    def _on_image(self, msg: Image):
        """Called every time a new camera frame arrives from Isaac Sim."""
        self.frame_count += 1

        # Convert ROS2 Image -> OpenCV BGR array
        try:
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f"Frame conversion failed: {e}")
            return

        # Run detection
        detection = self.detector.detect(bgr)

        # Process detection
        if detection is not None:
            self.detection_count += 1

            # Draw overlay on frame
            self.detector.draw_detection(bgr, detection)

            # Publish detection as JSON for the flight script
            det_msg = String()
            det_msg.data = json.dumps({
                "label": detection.label,
                "confidence": detection.confidence,
                "center_px": list(detection.center_px),
                "bbox": list(detection.bbox),
                "area_px": detection.area_px,
                "offset_px": list(detection.offset_px),
                "offset_norm": list(detection.offset_norm),
                "frame_id": self.frame_count,
                "timestamp": time.time(),
                "img_wh": [msg.width, msg.height],
            })
            self.detection_pub.publish(det_msg)

        # Draw frame info
        status = "DETECTED" if detection else "SEARCHING"
        cv2.putText(
            bgr, f"Frame {self.frame_count} | {status}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            bgr, f"Detections: {self.detection_count}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Save debug frames to disk
        if (SAVE_DEBUG_FRAMES
                and self.frame_count % DEBUG_FRAME_INTERVAL == 0
                and self.saved_count < MAX_DEBUG_FRAMES):
            fname = f"frame_{self.frame_count:06d}.png"
            path = os.path.join(DEBUG_FRAMES, fname)
            cv2.imwrite(path, bgr)
            self.saved_count += 1

        # Periodic console logging (every 5 seconds)
        now = time.time()
        if now - self.last_log_time >= 5.0:
            if detection:
                dx_n, dy_n = detection.offset_norm
                self.get_logger().info(
                    f"[f{self.frame_count}] {status} | "
                    f"offset=({dx_n:+.2f},{dy_n:+.2f}) | "
                    f"area={detection.area_px:.0f}px | "
                    f"saved={self.saved_count}"
                )
            else:
                self.get_logger().info(
                    f"[f{self.frame_count}] {status} | "
                    f"saved={self.saved_count}"
                )
            self.last_log_time = now


def main():
    print_config()

    rclpy.init()
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"\nStopped — {node.frame_count} frames, "
            f"{node.detection_count} detections, "
            f"{node.saved_count} frames saved to {DEBUG_FRAMES}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()