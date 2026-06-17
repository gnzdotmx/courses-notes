"""Interactive disk cleanup for training artifacts and dataset caches."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from captioning.paths import (
    APP_DIR,
    LEGACY_FEATURES_PATH,
    LEGACY_MAXLEN_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_TOKENIZER_PATH,
    REGISTRY_PATH,
    RunArtifacts,
    VGG_PATH,
    display_name_for_prefix,
    list_model_prefixes,
    load_models_registry,
)
from caption_datasets.registry import DATASET_REGISTRY


@dataclass(frozen=True)
class DeletableItem:
    """One file or directory the user may remove."""

    item_id: str
    path: Path
    category: str
    label: str
    note: str = ""

    @property
    def size_bytes(self) -> int:
        return _path_size(self.path)

    @property
    def size_human(self) -> str:
        return _human_size(self.size_bytes)

    @property
    def exists(self) -> bool:
        return self.path.is_file() or self.path.is_dir()


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def discover_deletable_items() -> list[DeletableItem]:
    """Scan app artifacts and known Kaggle caches."""
    items: list[DeletableItem] = []
    seen_paths: set[str] = set()

    def add(item: DeletableItem) -> None:
        key = str(item.path.resolve()) if item.path.exists() else str(item.path)
        if key in seen_paths:
            return
        if item.exists:
            seen_paths.add(key)
            items.append(item)

    for prefix in list_model_prefixes():
        arts = RunArtifacts.for_prefix(prefix)
        run_name = display_name_for_prefix(prefix)
        add(
            DeletableItem(
                item_id=f"model:{prefix}",
                path=arts.model_path,
                category="trained_model",
                label=f"Caption model — {run_name}",
                note=arts.model_path.name,
            )
        )
        if arts.tokenizer_path.is_file():
            add(
                DeletableItem(
                    item_id=f"tokenizer:{prefix}",
                    path=arts.tokenizer_path,
                    category="trained_model",
                    label=f"Tokenizer — {run_name}",
                    note=arts.tokenizer_path.name,
                )
            )
        if arts.maxlen_path.is_file():
            add(
                DeletableItem(
                    item_id=f"maxlen:{prefix}",
                    path=arts.maxlen_path,
                    category="trained_model",
                    label=f"Max caption length — {run_name}",
                    note=arts.maxlen_path.name,
                )
            )

    for path in sorted(APP_DIR.glob("features_*.dump")):
        dataset_id = path.stem.removeprefix("features_")
        spec = DATASET_REGISTRY.get(dataset_id)
        name = spec.name if spec else dataset_id
        add(
            DeletableItem(
                item_id=f"features:{dataset_id}",
                path=path,
                category="features",
                label=f"VGG16 feature cache — {name}",
                note="Re-extract with extract-features or train",
            )
        )

    if LEGACY_FEATURES_PATH.is_file():
        add(
            DeletableItem(
                item_id="features:legacy",
                path=LEGACY_FEATURES_PATH,
                category="features",
                label="VGG16 feature cache (legacy combined file)",
            )
        )

    if VGG_PATH.is_file():
        add(
            DeletableItem(
                item_id="shared:vgg16",
                path=VGG_PATH,
                category="shared",
                label="VGG16 feature extractor weights",
                note="Rebuilt automatically on next feature extraction",
            )
        )

    if REGISTRY_PATH.is_file():
        add(
            DeletableItem(
                item_id="meta:registry",
                path=REGISTRY_PATH,
                category="metadata",
                label="Model index (models_registry.json)",
                note="Safe to remove; recreated on next train",
            )
        )

    legacy_manifest = APP_DIR / "dataset_manifest.json"
    if legacy_manifest.is_file():
        add(
            DeletableItem(
                item_id="meta:dataset_manifest",
                path=legacy_manifest,
                category="metadata",
                label="Legacy dataset manifest",
            )
        )

    for spec in DATASET_REGISTRY.values():
        if not spec.downloadable or not spec.kaggle_handles:
            continue
        for handle in spec.kaggle_handles:
            try:
                from caption_datasets.kaggle_download import kaggle_cache_path

                cache = kaggle_cache_path(handle)
            except ImportError:
                continue
            if cache.is_dir():
                add(
                    DeletableItem(
                        item_id=f"kaggle:{spec.id}:{handle.replace('/', '_')}",
                        path=cache,
                        category="kaggle_dataset",
                        label=f"Kaggle download — {spec.name}",
                        note=f"{handle} → {cache}",
                    )
                )

    return items


_CATEGORY_ORDER = (
    "trained_model",
    "features",
    "shared",
    "metadata",
    "kaggle_dataset",
)
_CATEGORY_TITLES = {
    "trained_model": "Trained models (caption network + tokenizer + maxlen)",
    "features": "Extracted image features (VGG16 caches)",
    "shared": "Shared model files",
    "metadata": "Index / manifest files",
    "kaggle_dataset": "Downloaded datasets (Kaggle cache — images + captions on disk)",
}


def _print_item_menu(items: list[DeletableItem]) -> dict[int, DeletableItem]:
    print("\nThe following can be removed to free disk space.")
    print("You will be asked yes/no for each item.\n")

    by_category: dict[str, list[DeletableItem]] = {}
    for item in items:
        by_category.setdefault(item.category, []).append(item)

    index = 0
    index_map: dict[int, DeletableItem] = {}
    for category in _CATEGORY_ORDER:
        group = by_category.get(category)
        if not group:
            continue
        print(f"── {_CATEGORY_TITLES.get(category, category)} ──")
        for item in group:
            index += 1
            index_map[index] = item
            note = f" — {item.note}" if item.note else ""
            print(
                f"  [{index:2d}] {item.label}\n"
                f"       {item.path} ({item.size_human}){note}"
            )
        print()

    total = sum(item.size_bytes for item in items)
    print(f"Total if all selected: {_human_size(total)}\n")
    return index_map


def _prompt_selection(index_map: dict[int, DeletableItem], *, skip_confirm: bool) -> list[DeletableItem]:
    selected: list[DeletableItem] = []

    print("For each item, type y to delete, n to keep, or q to quit without deleting anything.\n")

    for num in sorted(index_map):
        item = index_map[num]
        while True:
            answer = input(f"  [{num:2d}] Delete? {item.label} [{item.size_human}] [y/N/q]: ").strip().lower()
            if answer in ("q", "quit"):
                print("Cleanup cancelled; nothing was deleted.")
                return []
            if answer in ("", "n", "no"):
                break
            if answer in ("y", "yes"):
                selected.append(item)
                break
            print("    Please enter y, n, or q.")

    if not selected:
        print("No items selected; nothing was deleted.")
        return []

    freed = sum(item.size_bytes for item in selected)
    print(f"\nSelected {len(selected)} item(s), about {_human_size(freed)} to free.")

    if not skip_confirm:
        answer = input("Proceed with deletion? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Deletion cancelled.")
            return []

    return selected


def _delete_item(item: DeletableItem) -> None:
    path = item.path
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    elif path.is_file():
        path.unlink()
        print(f"Removed file: {path}")
    else:
        print(f"Already gone: {path}")


def _update_registry_after_delete(deleted: list[DeletableItem]) -> None:
    if not REGISTRY_PATH.is_file():
        return
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    models: dict = registry.get("models", {})
    prefixes_to_remove: set[str] = set()
    for item in deleted:
        if item.item_id.startswith("model:"):
            prefixes_to_remove.add(item.item_id.split(":", 1)[1])

    for prefix in prefixes_to_remove:
        arts = RunArtifacts.for_prefix(prefix)
        if not arts.model_path.is_file():
            models.pop(prefix, None)

    if not models and REGISTRY_PATH.is_file():
        REGISTRY_PATH.unlink(missing_ok=True)
        print(f"Removed empty registry: {REGISTRY_PATH}")
    elif prefixes_to_remove:
        registry["models"] = models
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print("Updated models_registry.json")


def run_cleanup(*, skip_confirm: bool = False, dry_run: bool = False) -> None:
    """
    List deletable artifacts and ask the user to confirm each removal.
    Does not load TensorFlow.
    """
    items = discover_deletable_items()
    if not items:
        print("No cached artifacts or Kaggle datasets found to clean up.")
        return

    index_map = _print_item_menu(items)
    if dry_run:
        print("Dry run only; no files were deleted.")
        return

    selected = _prompt_selection(index_map, skip_confirm=skip_confirm)
    for item in selected:
        _delete_item(item)

    if selected:
        _update_registry_after_delete(selected)
        freed = sum(item.size_bytes for item in selected)
        print(f"\nCleanup complete. Freed about {_human_size(freed)}.")
    else:
        print("Cleanup finished with no deletions.")
