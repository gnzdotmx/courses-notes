"""Verification tab."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eigenface_live.models.face_sample import VerificationDecision
from eigenface_live.ui.controller import AppController
from eigenface_live.ui.widgets.camera_widget import CameraWidget


class VerifyTab(QWidget):
    def __init__(self, controller: AppController, camera_widget: CameraWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._camera_widget = camera_widget

        self._identity_combo = QComboBox()

        self._result_label = QLabel("Select an enrolled identity and start verification.")
        self._distance_label = QLabel("Distance: —")
        self._start_button = QPushButton("Start verification")
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._refresh_button = QPushButton("Refresh enrollments")

        form = QFormLayout()
        form.addRow("Enrolled identity", self._identity_combo)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._stop_button)
        buttons.addWidget(self._refresh_button)

        legend = QGroupBox("Decision legend")
        legend_layout = QVBoxLayout()
        legend_layout.addWidget(QLabel("MATCH: same person\nNO_MATCH: different person\nUNCERTAIN: re-position face and retry"))
        legend.setLayout(legend_layout)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(legend)
        layout.addWidget(self._result_label)
        layout.addWidget(self._distance_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self._start_button.clicked.connect(self._start_verification)
        self._stop_button.clicked.connect(self._stop_verification)
        self._refresh_button.clicked.connect(self._refresh_identities)
        controller.verification_result.connect(self._on_result)
        controller.error_occurred.connect(self._on_error)

        self._refresh_identities()

    def _refresh_identities(self) -> None:
        self._identity_combo.clear()
        identities = self._controller.repository.list_identities()
        self._identity_combo.addItems(identities)
        if not identities:
            self._result_label.setText("No enrollments found. Train a model first.")

    def _start_verification(self) -> None:
        identity = self._identity_combo.currentText().strip()
        if not identity:
            QMessageBox.warning(self, "No enrollment", "Train and save an identity first.")
            return
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._controller.begin_verification(identity)

    def _stop_verification(self) -> None:
        self._controller.stop_verification()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._result_label.setText("Verification stopped.")

    def _on_result(self, result: object) -> None:
        color = {
            VerificationDecision.MATCH: "#1f8f4d",
            VerificationDecision.NO_MATCH: "#b3261e",
            VerificationDecision.UNCERTAIN: "#9a6b00",
        }[result.decision]
        self._result_label.setText(
            f"{result.decision.value.upper()} — {result.identity} "
            f"(confidence {result.confidence:.0%})"
        )
        self._result_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._distance_label.setText(
            f"Distance {result.distance:.4f} / threshold {result.threshold:.4f}"
        )

    def _on_error(self, message: str) -> None:
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        QMessageBox.critical(self, "Verification error", message)
