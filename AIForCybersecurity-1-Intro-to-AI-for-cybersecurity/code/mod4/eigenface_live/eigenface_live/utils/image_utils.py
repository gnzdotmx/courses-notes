"""Image conversion helpers for Qt and OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtGui import QImage


def bgr_frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage without copying when possible."""
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def crop_and_resize_face(
    gray_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    """Extract, square-pad, and resize a face region to fixed dimensions."""
    x, y, w, h = bbox
    height, width = gray_frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face bounding box.")

    face = gray_frame[y1:y2, x1:x2]
    side = max(face.shape[0], face.shape[1])
    pad_top = (side - face.shape[0]) // 2
    pad_left = (side - face.shape[1]) // 2
    square = cv2.copyMakeBorder(
        face,
        pad_top,
        side - face.shape[0] - pad_top,
        pad_left,
        side - face.shape[1] - pad_left,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return cv2.resize(square, size, interpolation=cv2.INTER_AREA)


def flatten_face(face: np.ndarray) -> np.ndarray:
    """Flatten a grayscale face image to a 1D feature vector."""
    return face.astype(np.float32).ravel() / 255.0
