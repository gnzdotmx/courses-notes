"""Enrollment tab."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eigenface_live.ui.controller import AppController
from eigenface_live.ui.widgets.camera_widget import CameraWidget
from eigenface_live.utils.paths import sanitize_identity


class TrainTab(QWidget):
    def __init__(self, controller: AppController, camera_widget: CameraWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._camera_widget = camera_widget

        self._identity_input = QLineEdit()
        self._identity_input.setPlaceholderText("e.g. alice")
        self._status_label = QLabel("Enter an identity and start enrollment.")
        self._progress = QProgressBar()
        self._progress.setRange(0, controller.config.enrollment.target_samples)

        self._start_button = QPushButton("Start enrollment")
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)

        form = QFormLayout()
        form.addRow("Identity", self._identity_input)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._cancel_button)

        instructions = QGroupBox("Instructions")
        instruction_layout = QVBoxLayout()
        instruction_layout.addWidget(
            QLabel(
                "1. Enter a short identity label.\n"
                "2. Click Start enrollment and grant camera access if prompted.\n"
                "3. Move your face slowly in oval shapes until capture completes.\n"
                "4. The PCA + MLP model is saved locally for verification."
            )
        )
        instructions.setLayout(instruction_layout)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(instructions)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self._start_button.clicked.connect(self._start_enrollment)
        self._cancel_button.clicked.connect(self._cancel_enrollment)
        controller.enrollment_progress.connect(self._on_progress)
        controller.enrollment_finished.connect(self._on_finished)
        controller.error_occurred.connect(self._on_error)

    def _start_enrollment(self) -> None:
        try:
            identity = sanitize_identity(
                self._identity_input.text(),
                self._controller.config.security,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid identity", str(exc))
            return

        self._progress.setValue(0)
        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._controller.begin_enrollment(identity)

    def _cancel_enrollment(self) -> None:
        self._controller.cancel_enrollment()
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._status_label.setText("Enrollment cancelled.")

    def _on_progress(self, progress: object) -> None:
        self._progress.setValue(progress.collected)
        self._status_label.setText(progress.message)

    def _on_finished(self, path: Path) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._status_label.setText(f"Model saved to {path.name}")
        QMessageBox.information(self, "Enrollment complete", f"Saved enrollment to:\n{path}")

    def _on_error(self, message: str) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        QMessageBox.critical(self, "Enrollment error", message)
