"""Live camera preview widget."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from eigenface_live.utils.image_utils import bgr_frame_to_qimage


class CameraWidget(QWidget):
    """Displays the latest camera frame with optional overlay text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("Camera preview")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(640, 480)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background-color: #111; color: #ccc; border: 1px solid #333;")

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        self.setLayout(layout)

    def show_frame(self, frame: np.ndarray) -> None:
        image = bgr_frame_to_qimage(frame)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def show_message(self, message: str) -> None:
        self._label.setPixmap(QPixmap())
        self._label.setText(message)
