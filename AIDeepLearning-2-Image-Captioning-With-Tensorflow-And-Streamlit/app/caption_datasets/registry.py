"""Dataset catalog and orchestration."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from caption_datasets.base import DatasetSpec, LoadedDataset, merge_descriptions, prefix_descriptions

if TYPE_CHECKING:
    from captioning.config import TrainingConfig

LoaderFn = Callable[["TrainingConfig"], LoadedDataset]

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "flickr8k": DatasetSpec(
        id="flickr8k",
        name="Flickr8k",
        downloadable=True,
        trainable=True,
        description="8,000 images with five captions each; standard course baseline.",
        when_to_use="Learning, fast experiments, and reproducing the course pipeline.",
        size_hint="~1 GB (images + captions via Kaggle).",
        kaggle_handles=("youssefaboelnasr/flickr8k-text", "adityajn105/flickr8k"),
    ),
    "flickr30k": DatasetSpec(
        id="flickr30k",
        name="Flickr30k",
        downloadable=True,
        trainable=True,
        description="~31k images with five captions each; richer vocabulary than Flickr8k.",
        when_to_use="Better caption diversity while still feasible on a laptop.",
        size_hint="~4 GB via Kaggle (e.g. adityajn105/flickr30k).",
        kaggle_handles=(
            "adityajn105/flickr30k",
            "eeshawn/flickr30k",
            "srinivasac/flickr30k-dataset",
            "hsankesara/flickr-image-dataset",
        ),
        notes=(
            "On first download, open the Kaggle dataset page while logged in and accept "
            "the download rules. The loader tries several public mirrors if one fails."
        ),
    ),
    "coco": DatasetSpec(
        id="coco",
        name="MS COCO (Karpathy splits)",
        downloadable=False,
        trainable=False,
        description="Large-scale everyday images with multiple captions per image.",
        when_to_use="Production-quality baselines; needs substantial disk, RAM, and training time.",
        size_hint="Tens of GB for full images + annotations.",
        notes=(
            "Not enabled in this repo: MS COCO must be obtained from the official COCO "
            "website or a mirror you trust, then wired in via the custom dataset loader."
        ),
    ),
    "visual_genome": DatasetSpec(
        id="visual_genome",
        name="Visual Genome",
        downloadable=False,
        trainable=False,
        description="Dense scene graphs and region captions.",
        when_to_use="Research on relationships and dense descriptions; not a simple caption-only CSV.",
        size_hint="~20+ GB.",
        notes="Use --datasets custom with official Visual Genome exports.",
    ),
    "vizwiz": DatasetSpec(
        id="vizwiz",
        name="VizWiz",
        downloadable=False,
        trainable=False,
        description="Images taken by blind users with crowdsourced captions.",
        when_to_use="Accessibility-focused models; different caption style from Flickr/COCO.",
        size_hint="Several GB; requires registration on the VizWiz site.",
        notes="Use --datasets custom with VizWiz annotation files you download manually.",
    ),
    "nocaps": DatasetSpec(
        id="nocaps",
        name="NoCaps",
        downloadable=False,
        trainable=False,
        description="Evaluation benchmark for novel object captioning (COCO-like, held-out classes).",
        when_to_use="Testing generalization to new objects — not for training in this pipeline.",
        size_hint="Uses COCO validation images plus NoCaps annotations.",
        notes="Evaluation-only; train on COCO/Flickr then evaluate with a separate script.",
    ),
    "laion_400m": DatasetSpec(
        id="laion_400m",
        name="LAION-400M",
        downloadable=False,
        trainable=False,
        description="Web-scale image–text pairs (~400M).",
        when_to_use="Large multimodal pretraining in industrial clusters — not local laptop training.",
        size_hint="Multi-terabyte.",
        notes="Documented for reference only; not downloadable through this project.",
    ),
    "laion_5b": DatasetSpec(
        id="laion_5b",
        name="LAION-5B",
        downloadable=False,
        trainable=False,
        description="Web-scale image–text pairs (~5B).",
        when_to_use="Foundation-model scale training only.",
        size_hint="Petabyte-class aggregate size.",
        notes="Documented for reference only; not downloadable through this project.",
    ),
    "custom": DatasetSpec(
        id="custom",
        name="Custom (local paths)",
        downloadable=False,
        trainable=True,
        description="Your own images directory and caption file.",
        when_to_use="Private data, subsets of COCO/VizWiz/Visual Genome, or course extensions.",
        size_hint="User-defined.",
        notes="Provide --custom-images-dir and --custom-captions-file.",
    ),
}

_LOADER_MODULES: dict[str, str] = {
    "flickr8k": "caption_datasets.flickr8k",
    "flickr30k": "caption_datasets.flickr30k",
    "custom": "caption_datasets.custom",
}


def _get_loader(dataset_id: str) -> LoaderFn:
    module_name = _LOADER_MODULES.get(dataset_id)
    if not module_name:
        raise ValueError(f"No loader implemented for '{dataset_id}'.")
    module = importlib.import_module(module_name)
    return module.load


def list_datasets() -> None:
    print("Available datasets:\n")
    for spec in DATASET_REGISTRY.values():
        flags = []
        if spec.downloadable:
            flags.append("downloadable")
        if spec.trainable:
            flags.append("trainable")
        print(f"  {spec.id}")
        print(f"    Name: {spec.name} [{', '.join(flags) or 'reference only'}]")
        print(f"    {spec.description}")
        print(f"    When to use: {spec.when_to_use}")
        print(f"    Size: {spec.size_hint}")
        if spec.kaggle_handles:
            print(f"    Kaggle: {', '.join(spec.kaggle_handles)}")
        if spec.notes:
            print(f"    Note: {spec.notes}")
        print()


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_id}'. Choose from: {known}")
    return DATASET_REGISTRY[dataset_id]


def load_selected_datasets(config: "TrainingConfig") -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    list[tuple[str, Path]],
]:
    """
    Load and merge datasets.
    Returns train_description, test_description, image_roots for feature extraction.
    """
    train_parts: list[dict[str, list[str]]] = []
    test_parts: list[dict[str, list[str]]] = []
    image_roots: list[tuple[str, Path]] = []

    from captioning.text_processing import clean_descriptions

    for dataset_id in config.datasets:
        spec = get_dataset_spec(dataset_id)
        if not spec.trainable:
            raise ValueError(
                f"Dataset '{dataset_id}' is not trainable in this pipeline. {spec.notes}"
            )
        loader = _get_loader(dataset_id)

        print(f"\n=== Loading {spec.name} ===")
        loaded = loader(config)
        prefix = loaded.spec_id

        train = prefix_descriptions(
            clean_descriptions(loaded.train_description), prefix
        )
        test = prefix_descriptions(
            clean_descriptions(loaded.test_description), prefix
        )
        train_parts.append(train)
        test_parts.append(test)
        image_roots.append((prefix, loaded.images_dir.resolve()))

    train_description = merge_descriptions(train_parts)
    test_description = merge_descriptions(test_parts)
    print(
        f"\nCombined: {len(train_description)} train images, "
        f"{len(test_description)} test images"
    )
    return train_description, test_description, image_roots
