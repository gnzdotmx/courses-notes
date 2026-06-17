"""Flickr8k dataset loader (Kaggle)."""

from __future__ import annotations

from pathlib import Path

from captioning.config import TrainingConfig
from captioning.text_processing import wrap_caption
from caption_datasets.base import LoadedDataset
from caption_datasets.kaggle_download import download_dataset, find_directory, find_file
from utils.loaders import load_image_ids

TEXT_HANDLE = "youssefaboelnasr/flickr8k-text"
IMAGE_HANDLE = "adityajn105/flickr8k"


def _parse_token_file(token_path: Path) -> dict[str, list[str]]:
    description: dict[str, list[str]] = {}
    text = token_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        token = line.split()
        if len(token) < 2:
            continue
        filename, caption_tokens = token[0], token[1:]
        image_id = filename.split(".", 1)[0]
        caption = wrap_caption(" ".join(caption_tokens))
        description.setdefault(image_id, []).append(caption)
    return description


def load(config: TrainingConfig) -> LoadedDataset:
    del config
    text_root = download_dataset(TEXT_HANDLE)
    image_root = download_dataset(IMAGE_HANDLE)

    token_file = find_file(
        text_root,
        ("Flickr8k.token.txt", "flickr8k.token.txt"),
    )
    train_list = find_file(
        text_root,
        ("Flickr_8k.trainImages.txt", "Flickr8k.trainImages.txt"),
    )
    test_list = find_file(
        text_root,
        ("Flickr_8k.testImages.txt", "Flickr8k.testImages.txt"),
    )
    images_dir = find_directory(image_root, ("Images", "images", "Flicker8k_Dataset"))

    if not token_file or not train_list or not test_list or not images_dir:
        raise FileNotFoundError(
            "Flickr8k layout not recognized under Kaggle download paths. "
            f"text_root={text_root}, image_root={image_root}"
        )

    description = _parse_token_file(token_file)
    train_description = load_image_ids(description, str(train_list))
    test_description = load_image_ids(description, str(test_list))

    return LoadedDataset(
        spec_id="flickr8k",
        images_dir=images_dir,
        train_description=train_description,
        test_description=test_description,
        all_description=description,
    )
