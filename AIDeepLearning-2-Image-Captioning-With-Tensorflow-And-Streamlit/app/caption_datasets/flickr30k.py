"""Flickr30k dataset loader (Kaggle, with fallback handles)."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from captioning.config import TrainingConfig
from captioning.text_processing import wrap_caption
from caption_datasets.base import LoadedDataset
from caption_datasets.kaggle_download import (
    download_dataset_first_available,
    find_directory,
    find_file,
    kaggle_dataset_url,
)

# gpiosenka/flickr30k was removed from Kaggle (404). Try these in order.
KAGGLE_HANDLES: tuple[str, ...] = (
    "adityajn105/flickr30k",
    "eeshawn/flickr30k",
    "srinivasac/flickr30k-dataset",
    "hsankesara/flickr-image-dataset",
)

IMAGE_DIR_NAMES = (
    "flickr30k_images",
    "flickr30k-images",
    "Images",
    "images",
    "flickr30k",
    "Flickr30k",
)

CAPTION_FILE_NAMES = (
    "results_20130124.token",
    "flickr30k.token.txt",
    "Flickr30k.token.txt",
    "captions.txt",
    "results.txt",
    "captions.csv",
    "dataset_flickr30k.json",
)

SPLIT_TRAIN_NAMES = ("train_images.txt", "train.txt", "Flickr30k.trainImages.txt")
SPLIT_TEST_NAMES = ("test_images.txt", "test.txt", "Flickr30k.testImages.txt")


def _image_id_from_filename(filename: str) -> str:
    return filename.split(".", 1)[0]


def _parse_token_lines(lines: list[str]) -> dict[str, list[str]]:
    description: dict[str, list[str]] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line and "\t" in line.split("#", 1)[-1]:
            # Flickr30k official: image.jpg#0\tcaption text
            head, caption = line.split("\t", 1)
            filename = head.split("#", 1)[0]
        elif "\t" in line:
            filename, caption = line.split("\t", 1)
        elif "|" in line:
            filename, caption = line.split("|", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            filename, caption = parts[0], " ".join(parts[1:])
        image_id = _image_id_from_filename(filename.strip())
        description.setdefault(image_id, []).append(wrap_caption(caption.strip()))
    return description


def _parse_csv_captions(path: Path) -> dict[str, list[str]]:
    description: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        fields = {name.lower(): name for name in reader.fieldnames}
        image_key = next(
            (fields[k] for k in ("image", "image_name", "filename", "img", "file") if k in fields),
            reader.fieldnames[0],
        )
        caption_key = next(
            (
                fields[k]
                for k in ("caption", "comment", "sentence", "text", "description")
                if k in fields
            ),
            reader.fieldnames[-1],
        )
        for row in reader:
            filename = (row.get(image_key) or "").strip()
            caption = (row.get(caption_key) or "").strip()
            if not filename or not caption:
                continue
            image_id = _image_id_from_filename(filename)
            description.setdefault(image_id, []).append(wrap_caption(caption))
    return description


def _parse_json_captions(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    images = data.get("images", data)
    if not isinstance(images, list):
        raise ValueError(f"Unrecognized JSON layout: {path}")
    description: dict[str, list[str]] = {}
    for item in images:
        filename = item.get("filename") or item.get("file_name") or ""
        if not filename:
            continue
        image_id = _image_id_from_filename(filename)
        sentences = item.get("sentences") or item.get("captions") or []
        if isinstance(sentences, list):
            for sent in sentences:
                if isinstance(sent, dict):
                    text = sent.get("raw") or sent.get("caption") or ""
                else:
                    text = str(sent)
                if text.strip():
                    description.setdefault(image_id, []).append(wrap_caption(text.strip()))
        elif isinstance(sentences, str) and sentences.strip():
            description.setdefault(image_id, []).append(wrap_caption(sentences.strip()))
    return description


def _parse_caption_file(path: Path) -> dict[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _parse_csv_captions(path)
    if suffix == ".json":
        return _parse_json_captions(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_token_lines(lines)


def _find_images_dir(root: Path) -> Path | None:
    named = find_directory(root, IMAGE_DIR_NAMES)
    if named is not None:
        return named
    best_dir: Path | None = None
    best_count = 0
    for dirpath, _, files in os.walk(root):
        count = sum(1 for name in files if name.lower().endswith((".jpg", ".jpeg", ".png")))
        if count > best_count:
            best_count = count
            best_dir = Path(dirpath)
    return best_dir if best_count >= 100 else None


def _find_caption_file(root: Path) -> Path | None:
    found = find_file(root, CAPTION_FILE_NAMES)
    if found is not None:
        return found
    candidates: list[Path] = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if lower.endswith((".token", ".txt", ".csv", ".json")) and "caption" in lower:
                candidates.append(Path(dirpath) / name)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size)
    return find_file(root, ("captions.csv", "dataset_flickr30k.json"))


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
    del config
    root, handle = download_dataset_first_available(KAGGLE_HANDLES)
    print(f"Flickr30k source: {handle} ({kaggle_dataset_url(handle)})")

    images_dir = _find_images_dir(root)
    token_file = _find_caption_file(root)

    if not images_dir or not token_file:
        raise FileNotFoundError(
            "Flickr30k layout not recognized. Expected an images folder and a captions file "
            f"under {root} (downloaded from {handle})."
        )

    print(f"Flickr30k images: {images_dir}")
    print(f"Flickr30k captions: {token_file}")

    description = _parse_caption_file(token_file)
    if not description:
        raise ValueError(f"No captions parsed from {token_file}")

    train_list = find_file(root, SPLIT_TRAIN_NAMES)
    test_list = find_file(root, SPLIT_TEST_NAMES)

    if train_list and test_list:
        from utils.loaders import load_image_ids

        train_description = load_image_ids(description, str(train_list))
        test_description = load_image_ids(description, str(test_list))
    else:
        print("Flickr30k: no official split files; using 90/10 random split (seed=42).")
        train_description, test_description = _split_train_test(description)

    return LoadedDataset(
        spec_id="flickr30k",
        images_dir=images_dir,
        train_description=train_description,
        test_description=test_description,
        all_description=description,
    )
