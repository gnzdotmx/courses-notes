"""Dataset types and merge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a caption dataset."""

    id: str
    name: str
    downloadable: bool
    trainable: bool
    description: str
    when_to_use: str
    size_hint: str
    kaggle_handles: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class LoadedDataset:
    """Parsed captions and paths for one source."""

    spec_id: str
    images_dir: Path
    train_description: dict[str, list[str]]
    test_description: dict[str, list[str]]
    all_description: dict[str, list[str]] = field(default_factory=dict)


def prefix_descriptions(
    descriptions: dict[str, list[str]],
    prefix: str,
) -> dict[str, list[str]]:
    return {f"{prefix}:{key}": value for key, value in descriptions.items()}


def merge_descriptions(
    parts: list[dict[str, list[str]]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for part in parts:
        for key, captions in part.items():
            if key in merged:
                merged[key].extend(captions)
            else:
                merged[key] = list(captions)
    return merged
