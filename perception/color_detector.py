#!/usr/bin/env python3
"""
color_detector.py — HSV-based color detection for the Interceptor project.

Measured balloon HSV (from test.py on saved debug frame):
    H ≈ 5.3,  S ≈ 178.7,  V ≈ 129.6
Thresholds set with ±15 H, ±60 S, ±60 V margin around measured values.

Location: ~/interceptor/perception/color_detector.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "interceptor", "config"))
from interceptor_config import MIN_DETECTION_AREA_PX, MAX_DETECTION_AREA_PX, HSV_RED_RANGES

# ── Measured balloon HSV: H≈5, S≈179, V≈130 ──────────────────────────────────
# Two ranges because red wraps around H=0/180.
# S_min=100: measured S=179, floor at 100 to reject washed-out surfaces.
# V range 60–220: measured V=130, excludes deep shadows and blown highlights.
# HSV_RED_RANGES_MEASURED = [
#     (np.array([0,   100, 60]), np.array([18,  255, 220])),   # H 0-18
#     (np.array([162, 100, 60]), np.array([179, 255, 220])),   # H 162-179 (wrap)
# ]
HSV_RED_RANGES_MEASURED = HSV_RED_RANGES

# Debug: save mask to /tmp for offline inspection
DEBUG_MASK = os.environ.get("INTERCEPTOR_DEBUG_MASK", "0") == "1"


@dataclass
class Detection:
    """
    Single detected object — universal output format.
    Identical to what yolo_detector.py returns so flight scripts are agnostic.

    offset_px:   (dx, dy) from image center to detection center.
                 dx>0 → target right of center; dy>0 → target below center.
    offset_norm: offset_px normalized to [-1, 1].
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
        if detection:
            print(detection.center_px, detection.offset_norm)
    """

    def __init__(
        self,
        hsv_ranges: List[Tuple[np.ndarray, np.ndarray]] = None,
        min_area: int = MIN_DETECTION_AREA_PX,
        max_area: int = MAX_DETECTION_AREA_PX,
    ):
        self.hsv_ranges = hsv_ranges if hsv_ranges is not None else HSV_RED_RANGES_MEASURED
        self.min_area = min_area
        self.max_area = max_area
        self._frame_count = 0

    def detect(self, bgr_frame: np.ndarray) -> Optional[Detection]:
        """
        Run detection on a single BGR frame.
        Returns the largest valid red blob, or None.

        bgr_frame must be a writable uint8 array (call .copy() after frombuffer).
        """
        self._frame_count += 1
        h_img, w_img = bgr_frame.shape[:2]

        # ── 1. BGR → HSV ──────────────────────────────────────────────────────
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # ── 2. Log center-pixel HSV every 30 frames for threshold calibration ─
        if self._frame_count % 30 == 1:
            cy_c, cx_c = h_img // 2, w_img // 2
            sample = hsv[
                max(0, cy_c - 20):cy_c + 20,
                max(0, cx_c - 20):cx_c + 20
            ].reshape(-1, 3).mean(axis=0)
            print(f"[color_detector] frame={self._frame_count} "
                  f"center HSV mean: H={sample[0]:.1f} S={sample[1]:.1f} V={sample[2]:.1f}")

        # ── 3. Build combined mask ─────────────────────────────────────────────
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        # ── 4. Log mask coverage and optionally save to /tmp ──────────────────
        mask_coverage = mask.sum() / (255 * h_img * w_img)
        if self._frame_count % 30 == 1:
            print(f"[color_detector] mask coverage: {mask_coverage:.4f} "
                  f"({int(mask_coverage * 100)}%)")

        if DEBUG_MASK and self._frame_count % 10 == 1:
            cv2.imwrite(f"/tmp/mask_{self._frame_count:06d}.png", mask)
            cv2.imwrite(f"/tmp/frame_{self._frame_count:06d}.png", bgr_frame)

        # ── 5. Morphology: open (kill noise) → close (fill gaps) ──────────────
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # ── 6. Find contours ───────────────────────────────────────────────────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── 7. Pick largest valid contour ──────────────────────────────────────
        best_contour = None
        best_area    = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area and area > best_area:
                best_area    = area
                best_contour = cnt

        if best_contour is None:
            if self._frame_count % 30 == 1:
                all_areas = [cv2.contourArea(c) for c in contours]
                print(f"[color_detector] No valid contour. "
                      f"Found {len(contours)} contours, areas: {sorted(all_areas, reverse=True)[:5]}")
            return None

        # ── 8. Bounding box + center ───────────────────────────────────────────
        x, y, w, h = cv2.boundingRect(best_contour)
        cx = x + w // 2
        cy = y + h // 2

        # ── 9. Offset from image center ────────────────────────────────────────
        img_cx = w_img // 2
        img_cy = h_img // 2
        dx = cx - img_cx
        dy = cy - img_cy
        dx_norm = dx / (w_img / 2.0)
        dy_norm = dy / (h_img / 2.0)

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
        Draw bounding box, center-line arrow, and labels on frame (in-place).
        frame must be writable (call .copy() after np.frombuffer).
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

        # Arrow: image center → target center
        cv2.arrowedLine(frame, img_center, (cx, cy), color, 2)

        # Target center dot
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Labels
        cv2.putText(frame,
                    f"RED offset=({dx:+d},{dy:+d})px ({dx_n:+.2f},{dy_n:+.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"conf={detection.confidence:.2f} area={detection.area_px:.0f}px",
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return frame