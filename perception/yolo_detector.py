"""
yolo_detector.py — YOLOv8-based object detection for the Interceptor project.

Drop-in replacement for color_detector.py. Returns the same Detection
dataclass so camera_viewer.py and intercept_yaw.py need zero changes.

Uses ultralytics YOLOv8 for inference. Can use any pretrained or
custom-trained model.

Referenced from: Seabird's yolo_detector.py
  - Same concept: YOLO inference on camera frames
  - Same Detection output format
  - Adapted: single target class filtering, different confidence threshold,
    configurable target class from config file

Referenced from: color_detector.py
  - Same Detection dataclass
  - Same draw_detection() overlay with center-line
  - Same interface: detect(bgr_frame) → Detection or None

Run: imported by camera_viewer.py, not run directly.

Location: ~/interceptor/perception/yolo_detector.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "config"))
from interceptor_config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    YOLO_TARGET_CLASS,
    YOLO_INPUT_SIZE,
)

# Import Detection from color_detector so both detectors share the same dataclass
from color_detector import Detection

# Import ultralytics
from ultralytics import YOLO


class YoloDetector:
    """
    YOLOv8-based detector for the Interceptor project.

    Usage:
        detector = YoloDetector()
        detection = detector.detect(bgr_frame)
        if detection is not None:
            print(f"Target at {detection.center_px}, offset {detection.offset_norm}")
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = YOLO_CONFIDENCE,
        target_class: str = YOLO_TARGET_CLASS,
        input_size: int = YOLO_INPUT_SIZE,
    ):
        """
        Args:
            model_path:   path to YOLO weights (.pt file). If pretrained name
                          like "yolov8s.pt", ultralytics auto-downloads it.
            confidence:   minimum confidence threshold (0-1)
            target_class: COCO class name to track (e.g., "car", "person")
            input_size:   YOLO input resolution (larger = better small object detection)
        """
        print(f"[yolo] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_class = target_class
        self.input_size = input_size

        # Get the class index for our target class
        # model.names is a dict: {0: 'person', 1: 'bicycle', 2: 'car', ...}
        self.target_class_id = None
        for class_id, class_name in self.model.names.items():
            if class_name == self.target_class:
                self.target_class_id = class_id
                break

        if self.target_class_id is None:
            available = list(self.model.names.values())
            raise ValueError(
                f"[yolo] Class '{self.target_class}' not found in model. "
                f"Available classes: {available}"
            )

        print(f"[yolo] Model loaded. Tracking class: '{self.target_class}' (id={self.target_class_id})")
        print(f"[yolo] Confidence threshold: {self.confidence}")
        print(f"[yolo] Input size: {self.input_size}")

    def detect(self, bgr_frame: np.ndarray) -> Optional[Detection]:
        """
        Run YOLO detection on a single BGR frame.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR image (OpenCV format)

        Returns:
            Detection if target class is found above confidence, None otherwise.
            If multiple detections of the target class exist, returns the one
            with highest confidence.
        """
        h_img, w_img = bgr_frame.shape[:2]
        img_center_x = w_img // 2
        img_center_y = h_img // 2

        # Run YOLO inference
        # verbose=False suppresses per-frame logging
        results = self.model(
            bgr_frame,
            conf=self.confidence,
            imgsz=self.input_size,
            verbose=False,
        )

        # results is a list (one per image). We only pass one image.
        if not results or len(results) == 0:
            return None

        result = results[0]

        # Filter for our target class only
        best_detection = None
        best_conf = 0.0

        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id != self.target_class_id:
                continue

            if conf < self.confidence:
                continue

            if conf > best_conf:
                best_conf = conf
                # box.xyxy is [x1, y1, x2, y2] in pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                best_detection = (x1, y1, x2, y2, conf)

        if best_detection is None:
            return None

        x1, y1, x2, y2, conf = best_detection

        # Compute center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Compute area
        area = (x2 - x1) * (y2 - y1)

        # Compute offset from image center
        dx = cx - img_center_x
        dy = cy - img_center_y
        dx_norm = dx / (w_img / 2)
        dy_norm = dy / (h_img / 2)

        return Detection(
            label=self.target_class,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            center_px=(cx, cy),
            area_px=float(area),
            offset_px=(dx, dy),
            offset_norm=(round(dx_norm, 3), round(dy_norm, 3)),
        )

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding box + center-line on frame.
        Same visual output as color_detector.py's draw_detection().

        Referenced from: pi_tracker.py (center-line from screen center to target)
        Referenced from: color_detector.py (same overlay format)
        """
        h_img, w_img = frame.shape[:2]
        img_center = (w_img // 2, h_img // 2)

        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.center_px
        dx, dy = detection.offset_px
        dx_n, dy_n = detection.offset_norm

        # Crosshair at image center
        cv2.line(frame, (img_center[0] - 20, img_center[1]),
                 (img_center[0] + 20, img_center[1]), (100, 100, 100), 2)
        cv2.line(frame, (img_center[0], img_center[1] - 20),
                 (img_center[0], img_center[1] + 20), (100, 100, 100), 2)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Center-line: image center → target center
        cv2.arrowedLine(frame, img_center, (cx, cy), color, 2)

        # Target center dot
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Label with class name and confidence
        label = f"{detection.label} {detection.confidence:.0%} offset=({dx:+d},{dy:+d})px ({dx_n:+.2f},{dy_n:+.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Area info
        info = f"area={detection.area_px:.0f}px"
        cv2.putText(frame, info, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return frame