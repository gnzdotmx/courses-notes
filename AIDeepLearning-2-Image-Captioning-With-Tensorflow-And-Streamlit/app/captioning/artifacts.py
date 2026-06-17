"""Confirm and remove cached training artifacts."""

from __future__ import annotations

from captioning.paths import (
    RunArtifacts,
    features_path_for_dataset,
    load_models_registry,
)


def existing_artifacts(paths: tuple) -> list:
    from pathlib import Path

    return [p for p in paths if Path(p).is_file()]


def confirm_remove_artifacts(
    paths: list,
    *,
    reason: str,
    skip_confirm: bool = False,
) -> bool:
    """Ask user before deleting files. Returns True if removed or nothing to remove."""
    if not paths:
        return True

    print(reason)
    for path in paths:
        print(f"  - {path}")

    if not skip_confirm:
        answer = input("Delete these files and continue? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print(
                "Aborted. Existing artifacts were not modified. "
                "Train another prefix, or pass -y to confirm deletion."
            )
            return False

    for path in paths:
        path.unlink(missing_ok=True)
    print("Removed selected artifact(s).")
    return True


def prepare_training_artifacts(
    artifacts: RunArtifacts,
    dataset_ids: list[str],
    *,
    force_retrain: bool,
    skip_confirm: bool,
    force_feature_datasets: frozenset[str] | None,
    will_train: bool = True,
) -> bool:
    """
    Prepare artifacts for this run's prefix only.

    - Never deletes another prefix's model/tokenizer (e.g. flickr8k when training flickr30k).
    - Never deletes another dataset's features_*.dump (Kaggle downloads are untouched).
    - If the user declines deletion, aborts without removing anything.
    """
    to_remove: list = []

    if force_retrain and will_train:
        to_remove.extend(existing_artifacts(artifacts.training_paths))

    if force_feature_datasets:
        for dataset_id in force_feature_datasets:
            path = features_path_for_dataset(dataset_id)
            if path.is_file():
                to_remove.append(path)

    # Deduplicate
    seen: set = set()
    unique: list = []
    for path in to_remove:
        resolved = str(path.resolve()) if hasattr(path, "resolve") else str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if unique:
        datasets_note = ", ".join(sorted(dataset_ids))
        ok = confirm_remove_artifacts(
            unique,
            reason=(
                f"Rebuild artifacts for run '{artifacts.prefix}' ({datasets_note}). "
                "Other trained models and per-dataset feature files are left unchanged."
            ),
            skip_confirm=skip_confirm,
        )
        if not ok:
            return False

    prior_registry = load_models_registry().get("models", {})
    if artifacts.prefix in prior_registry and will_train and not force_retrain:
        if artifacts.model_path.is_file() and not skip_confirm:
            answer = input(
                f"{artifacts.model_path.name} already exists for this dataset selection. "
                "Skip training and use it? [Y/n]: "
            ).strip().lower()
            if answer in ("n", "no"):
                if not confirm_remove_artifacts(
                    list(artifacts.training_paths),
                    reason=f"Retrain run '{artifacts.prefix}' requires removing:",
                    skip_confirm=skip_confirm,
                ):
                    return False

    return True


def datasets_needing_feature_extract(
    dataset_ids: list[str],
    *,
    force: bool,
) -> set[str]:
    """Dataset ids that still need VGG feature extraction."""
    if force:
        return set(dataset_ids)
    missing = set()
    for dataset_id in dataset_ids:
        if not features_path_for_dataset(dataset_id).is_file():
            missing.add(dataset_id)
    return missing
