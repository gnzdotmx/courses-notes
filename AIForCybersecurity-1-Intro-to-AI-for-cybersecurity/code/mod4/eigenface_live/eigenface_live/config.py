"""Application configuration (macOS-only deployment)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30


@dataclass(frozen=True, slots=True)
class FaceDetectionConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: tuple[int, int] = (80, 80)


@dataclass(frozen=True, slots=True)
class EnrollmentConfig:
    target_samples: int = 40
    min_motion_span_x: float = 40.0
    min_motion_span_y: float = 30.0
    motion_window: int = 45
    min_capture_interval_frames: int = 4
    face_size: tuple[int, int] = (92, 112)  # grayscale crop, LFW-like aspect


@dataclass(frozen=True, slots=True)
class ModelConfig:
    pca_components: int = 30
    mlp_hidden: tuple[int, ...] = (64,)
    mlp_max_iter: int = 400
    verification_threshold_percentile: float = 95.0
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    max_identity_length: int = 32
    identity_pattern: str = r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$"
    model_file_mode: int = 0o600
    model_dir_mode: int = 0o700


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    detection: FaceDetectionConfig = FaceDetectionConfig()
    enrollment: EnrollmentConfig = EnrollmentConfig()
    model: ModelConfig = ModelConfig()
    security: SecurityConfig = SecurityConfig()
    models_dir: Path = Path(__file__).resolve().parent.parent / "data" / "enrollments"


DEFAULT_CONFIG = AppConfig()
