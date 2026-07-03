"""Collect training samples while the user traces oval face motion."""

from __future__ import annotations

from collections import deque

import numpy as np

from eigenface_live.config import EnrollmentConfig
from eigenface_live.models.face_sample import EnrollmentProgress, FaceDetection, FaceSample


class OvalMotionCollector:
    """
    Capture face frames only after the user moves their head through an oval-like path.

    Heuristic: track face-center positions; require sufficient span on both axes and
    periodic displacement before accepting a sample.
    """

    def __init__(self, config: EnrollmentConfig) -> None:
        self._config = config
        self._centers: deque[tuple[float, float]] = deque(maxlen=config.motion_window)
        self._samples: list[FaceSample] = []
        self._frames_since_capture = 0
        self._active = False

    @property
    def samples(self) -> list[FaceSample]:
        return list(self._samples)

    @property
    def is_complete(self) -> bool:
        return len(self._samples) >= self._config.target_samples

    def reset(self) -> None:
        self._centers.clear()
        self._samples.clear()
        self._frames_since_capture = 0
        self._active = False

    def start(self) -> None:
        self.reset()
        self._active = True

    def stop(self) -> None:
        self._active = False

    def ingest(
        self,
        detection: FaceDetection | None,
        face_vector: np.ndarray | None,
    ) -> EnrollmentProgress:
        if not self._active:
            return EnrollmentProgress(
                collected=len(self._samples),
                target=self._config.target_samples,
                motion_ready=False,
                message="Enrollment idle.",
            )

        if detection is None or face_vector is None:
            return EnrollmentProgress(
                collected=len(self._samples),
                target=self._config.target_samples,
                motion_ready=False,
                message="Center your face in the frame.",
            )

        self._centers.append(detection.center)
        self._frames_since_capture += 1
        motion_ready = self._has_oval_motion()

        if (
            motion_ready
            and self._frames_since_capture >= self._config.min_capture_interval_frames
            and not self.is_complete
        ):
            self._samples.append(FaceSample.from_vector(face_vector))
            self._frames_since_capture = 0

        if self.is_complete:
            self._active = False
            message = "Enrollment capture complete."
        elif not motion_ready:
            message = "Move your face slowly in oval shapes."
        else:
            message = "Good motion — keep tracing ovals."

        return EnrollmentProgress(
            collected=len(self._samples),
            target=self._config.target_samples,
            motion_ready=motion_ready,
            message=message,
        )

    def _has_oval_motion(self) -> bool:
        if len(self._centers) < self._config.motion_window // 2:
            return False
        xs = [c[0] for c in self._centers]
        ys = [c[1] for c in self._centers]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x < self._config.min_motion_span_x or span_y < self._config.min_motion_span_y:
            return False

        points = np.array(self._centers, dtype=np.float32)
        centered = points - points.mean(axis=0)
        if centered.shape[0] < 3:
            return False
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            return False
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[0] <= 1e-6:
            return False
        ratio = eigvals[-1] / eigvals[0]
        return 1.2 <= ratio <= 6.0
