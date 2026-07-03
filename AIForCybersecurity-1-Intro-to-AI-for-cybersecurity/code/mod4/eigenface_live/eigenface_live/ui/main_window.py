"""Main application window."""

from __future__ import annotations

import platform

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from eigenface_live.config import AppConfig
from eigenface_live.ui.controller import AppController
from eigenface_live.ui.train_tab import TrainTab
from eigenface_live.ui.verify_tab import VerifyTab
from eigenface_live.ui.widgets.camera_widget import CameraWidget


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        if platform.system() != "Darwin":
            raise RuntimeError("Eigenface Live supports macOS only.")

        self.setWindowTitle("Eigenface Live — Train & Verify")
        self.resize(1100, 760)

        self._controller = AppController(config)
        self._camera_widget = CameraWidget()
        self._overlay_label = QLabel("Camera idle")
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setStyleSheet("padding: 6px; background: #222; color: #eee;")

        tabs = QTabWidget()
        tabs.addTab(TrainTab(self._controller, self._camera_widget), "Train")
        tabs.addTab(VerifyTab(self._controller, self._camera_widget), "Verify")

        right = QVBoxLayout()
        right.addWidget(self._camera_widget, stretch=4)
        right.addWidget(self._overlay_label)
        right.addWidget(tabs, stretch=3)
        right_widget = QWidget()
        right_widget.setLayout(right)
        self.setCentralWidget(right_widget)

        self._controller.frame_ready.connect(self._on_frame)
        self._controller.start_camera()

    def _on_frame(self, frame: object, overlay: str) -> None:
        self._camera_widget.show_frame(frame)
        self._overlay_label.setText(overlay)

    def closeEvent(self, event) -> None:
        self._controller.stop_camera()
        super().closeEvent(event)
