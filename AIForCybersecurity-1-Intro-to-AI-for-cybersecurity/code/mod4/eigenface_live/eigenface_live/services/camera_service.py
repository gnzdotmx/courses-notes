"""macOS camera capture via AVFoundation."""

from __future__ import annotations

import platform
from typing import Protocol

import cv2
import numpy as np

from eigenface_live.config import CameraConfig


class CameraServiceError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class CameraServiceProtocol(Protocol):
    def open(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...


class MacCameraService:
    """OpenCV camera wrapper locked to macOS AVFoundation."""

    def __init__(self, config: CameraConfig) -> None:
        if platform.system() != "Darwin":
            raise CameraServiceError("This application supports macOS only.")
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        if self.is_open:
            return
        capture = cv2.VideoCapture(self._config.device_index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            raise CameraServiceError(
                "Could not open the camera. Grant camera access in "
                "System Settings → Privacy & Security → Camera."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = capture

    def read(self) -> np.ndarray:
        if not self.is_open or self._capture is None:
            raise CameraServiceError("Camera is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraServiceError("Failed to read a frame from the camera.")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> MacCameraService:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
