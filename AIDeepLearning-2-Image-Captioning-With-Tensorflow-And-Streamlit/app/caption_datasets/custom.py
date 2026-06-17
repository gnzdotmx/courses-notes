"""Custom local dataset loader."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from captioning.config import TrainingConfig
from captioning.text_processing import wrap_caption
from caption_datasets.base import LoadedDataset
from utils.loaders import load_image_ids


def _parse_captions_file(path: Path) -> dict[str, list[str]]:
    description: dict[str, list[str]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() == ".csv":
        reader = csv.reader(text.splitlines())
        for row in reader:
            if len(row) < 2:
                continue
            image_id = row[0].split(".", 1)[0]
            caption = wrap_caption(row[1])
            description.setdefault(image_id, []).append(caption)
        return description

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line and " " not in line.split(",", 1)[0]:
            image_part, caption = line.split(",", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            image_part, caption = parts[0], " ".join(parts[1:])
        image_id = image_part.split(".", 1)[0]
        description.setdefault(image_id, []).append(wrap_caption(caption))
    return description


def _split_train_test(
    description: dict[str, list[str]],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ids = sorted(description.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_at = int(len(ids) * train_ratio)
    train_ids = set(ids[:split_at])
    train = {k: description[k] for k in ids if k in train_ids}
    test = {k: description[k] for k in ids if k not in train_ids}
    return train, test


def load(config: TrainingConfig) -> LoadedDataset:
    if not config.custom_images_dir or not config.custom_captions_file:
        raise ValueError("custom dataset requires --custom-images-dir and --custom-captions-file")

    images_dir = config.custom_images_dir.expanduser().resolve()
    captions_file = config.custom_captions_file.expanduser().resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not captions_file.is_file():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    description = _parse_captions_file(captions_file)

    if config.custom_train_list and config.custom_test_list:
        train_description = load_image_ids(description, str(config.custom_train_list))
        test_description = load_image_ids(description, str(config.custom_test_list))
    else:
        train_description, test_description = _split_train_test(description)

    return LoadedDataset(
        spec_id="custom",
        images_dir=images_dir,
        train_description=train_description,
        test_description=test_description,
        all_description=description,
    )
