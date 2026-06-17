"""Application paths and artifact filenames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

APP_DIR = Path(__file__).resolve().parent.parent
VGG_PATH = APP_DIR / "vgg16_model.h5"
REGISTRY_PATH = APP_DIR / "models_registry.json"

# Legacy single-prefix artifacts (pre-refactor)
LEGACY_PREFIX = "default"
LEGACY_MODEL_PATH = APP_DIR / "model.keras"
LEGACY_TOKENIZER_PATH = APP_DIR / "tokenize.dump"
LEGACY_MAXLEN_PATH = APP_DIR / "maxlen.dump"
LEGACY_FEATURES_PATH = APP_DIR / "features.dump"


def artifact_prefix(dataset_ids: Sequence[str]) -> str:
    """Stable prefix for a training run (sorted dataset ids joined by '_')."""
    return "_".join(sorted(dataset_ids))


def features_path_for_dataset(dataset_id: str) -> Path:
    """Per-dataset VGG feature cache (survives when you train on another dataset)."""
    return APP_DIR / f"features_{dataset_id}.dump"


@dataclass(frozen=True)
class RunArtifacts:
    """Paths for one trained model bundle (caption model + tokenizer + maxlen)."""

    prefix: str
    model_path: Path
    tokenizer_path: Path
    maxlen_path: Path

    @classmethod
    def for_datasets(cls, dataset_ids: Sequence[str]) -> "RunArtifacts":
        return cls.for_prefix(artifact_prefix(dataset_ids))

    @classmethod
    def for_prefix(cls, prefix: str) -> "RunArtifacts":
        if prefix in (LEGACY_PREFIX, "legacy"):
            return cls(
                prefix=LEGACY_PREFIX,
                model_path=LEGACY_MODEL_PATH,
                tokenizer_path=LEGACY_TOKENIZER_PATH,
                maxlen_path=LEGACY_MAXLEN_PATH,
            )
        return cls(
            prefix=prefix,
            model_path=APP_DIR / f"model_{prefix}.keras",
            tokenizer_path=APP_DIR / f"tokenize_{prefix}.dump",
            maxlen_path=APP_DIR / f"maxlen_{prefix}.dump",
        )

    @property
    def training_paths(self) -> tuple[Path, Path, Path]:
        return (self.model_path, self.tokenizer_path, self.maxlen_path)


def ensure_app_dir() -> Path:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    return APP_DIR


def list_model_prefixes() -> list[str]:
    """Discover trained model bundles for Streamlit / CLI."""
    prefixes: set[str] = set()
    for path in APP_DIR.glob("model_*.keras"):
        prefixes.add(path.stem.removeprefix("model_"))
    if LEGACY_MODEL_PATH.is_file():
        prefixes.add(LEGACY_PREFIX)

    registry = load_models_registry()
    for prefix in registry.get("models", {}):
        if RunArtifacts.for_prefix(prefix).model_path.is_file():
            prefixes.add(prefix)

    return sorted(prefixes, key=_prefix_sort_key)


def _prefix_sort_key(prefix: str) -> tuple:
    if prefix == LEGACY_PREFIX:
        return (1, prefix)
    return (0, prefix)


def load_models_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        return {"models": {}}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"models": {}}


def register_model_run(prefix: str, dataset_ids: Sequence[str]) -> None:
    registry = load_models_registry()
    models = registry.setdefault("models", {})
    models[prefix] = {
        "datasets": sorted(dataset_ids),
        "model_path": RunArtifacts.for_prefix(prefix).model_path.name,
        "tokenizer_path": RunArtifacts.for_prefix(prefix).tokenizer_path.name,
        "maxlen_path": RunArtifacts.for_prefix(prefix).maxlen_path.name,
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def display_name_for_prefix(prefix: str) -> str:
    if prefix == LEGACY_PREFIX:
        return "default (legacy artifacts)"
    return prefix.replace("_", " + ")
