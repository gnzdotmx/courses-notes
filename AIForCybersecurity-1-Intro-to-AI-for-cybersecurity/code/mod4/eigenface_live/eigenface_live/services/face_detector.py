"""Haar-cascade face detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from eigenface_live.config import FaceDetectionConfig
from eigenface_live.models.face_sample import FaceDetection


class FaceDetector:
    """Detect the largest frontal face in a BGR frame."""

    def __init__(self, config: FaceDetectionConfig) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")
        self._config = config

    def detect_largest(self, frame: np.ndarray) -> FaceDetection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._config.scale_factor,
            minNeighbors=self._config.min_neighbors,
            minSize=self._config.min_size,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center = (x + w / 2.0, y + h / 2.0)
        return FaceDetection(bbox=(int(x), int(y), int(w), int(h)), center=center)

    def draw_bbox(self, frame: np.ndarray, detection: FaceDetection, color: tuple[int, int, int]) -> np.ndarray:
        annotated = frame.copy()
        x, y, w, h = detection.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        return annotated
