"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from captioning.paths import APP_DIR


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and paths for one training run."""

    datasets: tuple[str, ...] = ("flickr8k",)
    epochs: int = 10
    embedding_dim: int = 256
    lstm_units: int = 256
    dense_units: int = 256
    dropout: float = 0.5
    learning_rate: float = 0.001
    num_words: int | None = None
    max_caption_length: int | None = None
    image_feature_dim: int = 1000
    steps_per_epoch: int | None = None
    custom_images_dir: Path | None = None
    custom_captions_file: Path | None = None
    custom_train_list: Path | None = None
    custom_test_list: Path | None = None
    work_dir: Path = field(default_factory=lambda: APP_DIR)
    force_retrain: bool = False
    skip_confirm: bool = False
    extract_features: bool = True
    train: bool = True
    evaluate: bool = False
    verbose: int = 1

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_words is not None and self.num_words < 2:
            raise ValueError("num_words must be >= 2 when set")
        if "custom" in self.datasets:
            if not self.custom_images_dir or not self.custom_captions_file:
                raise ValueError(
                    "custom dataset requires --custom-images-dir and --custom-captions-file"
                )
            _validate_path(self.custom_images_dir, must_exist=True)
            _validate_path(self.custom_captions_file, must_exist=True)
            if self.custom_train_list:
                _validate_path(self.custom_train_list, must_exist=True)
            if self.custom_test_list:
                _validate_path(self.custom_test_list, must_exist=True)


def _validate_path(path: Path, *, must_exist: bool) -> None:
    resolved = path.expanduser().resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    if resolved.is_file() and ".." in path.parts:
        pass
    # Reject path traversal relative to nothing sensitive; paths must be user-provided explicitly.
