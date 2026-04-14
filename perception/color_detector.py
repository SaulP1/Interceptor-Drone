#!/usr/bin/env python3
"""
color_detector.py — HSV-based color detection for the Interceptor project.

This is the Phase 1-2 detection module. It does ONE thing well:
find a red object in a camera frame and return its pixel location.

Later, this gets swapped for yolo_detector.py without changing any
other code — the output format (Detection dataclass) is the same.

No ROS2, no MAVROS, no Isaac — pure image in, detection out.

Referenced from: Seabird's buoy_detector.py
  - Same HSV masking + morphology approach
  - Same contour-finding logic
  - Adapted: we detect one target (not multiple buoy colors),
    we compute the center-line vector for course correction,
    and we use different thresholds for our scene.

Location: ~/interceptor/perception/color_detector.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import os

# Add config to path
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "config"))
from interceptor_config import (
    HSV_RED_RANGES,
    MIN_DETECTION_AREA_PX,
    MAX_DETECTION_AREA_PX,
)


@dataclass
class Detection:
    """
    Single detected object — the universal output format.

    Both color_detector.py and (future) yolo_detector.py return this
    same dataclass. Flight scripts and the camera viewer don't care
    which detector produced it.

    Fields:
        label:       what was detected ("red_target")
        confidence:  0.0 - 1.0 (for HSV, based on blob area relative to max)
        bbox:        (x_min, y_min, x_max, y_max) in pixels
        center_px:   (cx, cy) center of bounding box in pixels
        area_px:     blob area in pixels
        offset_px:   (dx, dy) vector FROM image center TO detection center
                     This is the key value for course correction:
                     if dx > 0, target is right of center → steer right
                     if dy > 0, target is below center → steer down
        offset_norm: (dx, dy) normalized to [-1, 1] range
                     (-1,-1) = top-left corner, (1,1) = bottom-right
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_px: Tuple[int, int]
    area_px: float
    offset_px: Tuple[int, int]
    offset_norm: Tuple[float, float]


class ColorDetector:
    """
    HSV color-based detector for a red target.

    Usage:
        detector = ColorDetector()
        detection = detector.detect(bgr_frame)
        if detection is not None:
            print(f"Target at {detection.center_px}, offset {detection.offset_norm}")
    """

    def __init__(
        self,
        hsv_ranges: List[Tuple[np.ndarray, np.ndarray]] = None,
        min_area: int = MIN_DETECTION_AREA_PX,
        max_area: int = MAX_DETECTION_AREA_PX,
    ):
        """
        Args:
            hsv_ranges: list of (lower, upper) HSV bounds. Defaults to red.
            min_area:   minimum contour area in pixels to be a valid detection
            max_area:   maximum contour area in pixels
        """
        self.hsv_ranges = hsv_ranges or HSV_RED_RANGES
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, bgr_frame: np.ndarray) -> Optional[Detection]:
        """
        Run detection on a single BGR frame.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR image (OpenCV format)

        Returns:
            Detection if a red object is found, None otherwise.
            If multiple red blobs exist, returns the largest one.
        """
        h_img, w_img = bgr_frame.shape[:2]
        img_center_x = w_img // 2
        img_center_y = h_img // 2

        # ── Step 1: Convert to HSV ──
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # ── Step 2: Build color mask ──
        # Red spans two hue ranges (wraps around 0/180), so we OR them.
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # ── Step 3: Clean up mask with morphology ──
        # Open (remove noise) then Close (fill gaps)
        # Referenced from: Seabird buoy_detector.py uses same sequence
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)

        # ── Step 4: Find contours ──
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Step 5: Find largest valid contour ──
        best_contour = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if area > best_area:
                best_area = area
                best_contour = cnt

        if best_contour is None:
            return None

        # ── Step 6: Compute bounding box and center ──
        x, y, w, h = cv2.boundingRect(best_contour)
        cx = x + w // 2
        cy = y + h // 2

        # ── Step 7: Compute offset from image center ──
        # This is the core value for course correction.
        # Positive dx = target is RIGHT of center
        # Positive dy = target is BELOW center
        dx = cx - img_center_x
        dy = cy - img_center_y

        # Normalize to [-1, 1] range
        dx_norm = dx / (w_img / 2)
        dy_norm = dy / (h_img / 2)

        # Confidence: proportion of blob area to max expected area
        confidence = min(best_area / self.max_area, 1.0)

        return Detection(
            label="red_target",
            confidence=confidence,
            bbox=(x, y, x + w, y + h),
            center_px=(cx, cy),
            area_px=best_area,
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

        The center-line goes from the image center to the detection center.
        This is the visual representation of "how far off-center is the
        target" — the flight controller's job is to make this line shorter.

        Referenced from: pi_tracker.py's drawing logic
          - Same concept: line from screen center to target center
          - Same bounding box + label overlay
          - Adapted for our Detection dataclass format

        Args:
            frame:     BGR image to draw on (will be modified in place)
            detection: Detection object from detect()
            color:     BGR color for drawing

        Returns:
            The annotated frame (same reference, modified in place)
        """
        h_img, w_img = frame.shape[:2]
        img_center = (w_img // 2, h_img // 2)

        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.center_px
        dx, dy = detection.offset_px
        dx_n, dy_n = detection.offset_norm

        # Draw crosshair at image center
        cv2.line(frame, (img_center[0] - 20, img_center[1]),
                 (img_center[0] + 20, img_center[1]), (100, 100, 100), 2)
        cv2.line(frame, (img_center[0], img_center[1] - 20),
                 (img_center[0], img_center[1] + 20), (100, 100, 100), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw center-line: image center → target center
        # This is THE key visual — the flight script tries to shrink this line
        cv2.arrowedLine(frame, img_center, (cx, cy), color, 2)

        # Draw target center dot
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Label: offset in pixels and normalized
        label = f"RED offset=({dx:+d},{dy:+d})px ({dx_n:+.2f},{dy_n:+.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Confidence + area
        info = f"conf={detection.confidence:.2f} area={detection.area_px:.0f}px"
        cv2.putText(frame, info, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return frame