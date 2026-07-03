"""Application controller coordinating camera, detection, and ML services."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from eigenface_live.config import AppConfig
from eigenface_live.models.enrollment import EnrollmentArtifact
from eigenface_live.models.face_sample import EnrollmentProgress, VerificationResult
from eigenface_live.services.camera_service import MacCameraService
from eigenface_live.services.face_detector import FaceDetector
from eigenface_live.services.face_recognition_service import FaceRecognitionService
from eigenface_live.services.frame_collector import OvalMotionCollector
from eigenface_live.services.model_persistence import ModelRepository
from eigenface_live.utils.image_utils import crop_and_resize_face, flatten_face


class AppController(QObject):
    """Facade between Qt UI and backend services."""

    frame_ready = pyqtSignal(np.ndarray, str)
    enrollment_progress = pyqtSignal(object)
    enrollment_finished = pyqtSignal(object)
    verification_result = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._camera = MacCameraService(config.camera)
        self._detector = FaceDetector(config.detection)
        self._collector = OvalMotionCollector(config.enrollment)
        self._recognition = FaceRecognitionService(config)
        self._repository = ModelRepository(config)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._mode = "idle"
        self._active_artifact: EnrollmentArtifact | None = None
        self._pending_identity = ""

    @property
    def repository(self) -> ModelRepository:
        return self._repository

    @property
    def config(self) -> AppConfig:
        return self._config

    def start_camera(self) -> None:
        try:
            self._camera.open()
            self._timer.start(int(1000 / self._config.camera.fps))
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def stop_camera(self) -> None:
        self._timer.stop()
        self._camera.close()
        self._mode = "idle"
        self._collector.stop()

    def begin_enrollment(self, identity: str) -> None:
        self._pending_identity = identity.strip()
        self._collector.start()
        self._mode = "enroll"

    def cancel_enrollment(self) -> None:
        self._collector.stop()
        self._mode = "idle"

    def begin_verification(self, identity: str) -> None:
        try:
            self._active_artifact = self._repository.load(identity)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self._mode = "idle"
            return
        self._mode = "verify"

    def stop_verification(self) -> None:
        self._active_artifact = None
        self._mode = "idle"

    def _on_tick(self) -> None:
        try:
            frame = self._camera.read()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.stop_camera()
            return

        overlay = "Live preview"
        detection = self._detector.detect_largest(frame)

        if detection is not None:
            frame = self._detector.draw_bbox(frame, detection, (80, 200, 120))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = crop_and_resize_face(gray, detection.bbox, self._config.enrollment.face_size)
            vector = flatten_face(face)
        else:
            vector = None

        if self._mode == "enroll":
            progress = self._collector.ingest(detection, vector)
            overlay = progress.message
            self.enrollment_progress.emit(progress)
            if self._collector.is_complete:
                try:
                    artifact = self._recognition.train(self._pending_identity, self._collector.samples)
                    path = self._repository.save(artifact)
                    self.enrollment_finished.emit(path)
                except Exception as exc:
                    self.error_occurred.emit(str(exc))
                finally:
                    self._collector.stop()
                    self._mode = "idle"

        elif self._mode == "verify" and self._active_artifact is not None and vector is not None:
            result = self._recognition.verify(self._active_artifact, vector)
            overlay = (
                f"{result.decision.value.upper()} — confidence {result.confidence:.0%} "
                f"(distance {result.distance:.3f} / {result.threshold:.3f})"
            )
            self.verification_result.emit(result)

        self.frame_ready.emit(frame, overlay)
