"""Kaggle dataset download with cache reuse."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kagglehub is required for dataset download. Install with: pip install kagglehub"
    ) from exc


def kaggle_dataset_url(handle: str) -> str:
    """Public Kaggle dataset page (open in browser to accept download rules)."""
    owner, slug = handle.split("/", 1)
    return f"https://www.kaggle.com/datasets/{owner}/{slug}"


def kaggle_cache_path(handle: str, version: str = "1") -> Path:
    owner, slug = handle.split("/", 1)
    return Path.home() / ".cache" / "kagglehub" / "datasets" / owner / slug / "versions" / version


def download_dataset(handle: str, *, force: bool = False) -> Path:
    """
    Download a Kaggle dataset if not cached.
    Requires Kaggle API credentials (kaggle.json) for first download.
    """
    cache = kaggle_cache_path(handle)
    if cache.is_dir() and not force:
        print(f"Using cached dataset: {handle} -> {cache}")
        return cache

    print(f"Downloading dataset: {handle} ...")
    print(f"  Dataset page: {kaggle_dataset_url(handle)}")
    try:
        path = kagglehub.dataset_download(handle)
    except KaggleApiHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        hint = (
            f"Open {kaggle_dataset_url(handle)} while logged into Kaggle, "
            "click Download / accept the dataset rules, then retry."
        )
        if status == 403:
            raise RuntimeError(
                f"Kaggle denied access to {handle} (HTTP 403). {hint}"
            ) from exc
        raise RuntimeError(f"Kaggle download failed for {handle}. {hint}") from exc
    resolved = Path(path).resolve()
    print(f"Dataset ready: {resolved}")
    return resolved


def download_dataset_first_available(
    handles: tuple[str, ...],
    *,
    force: bool = False,
) -> tuple[Path, str]:
    """Try each Kaggle handle until one downloads; aggregate errors if all fail."""
    errors: list[str] = []
    for handle in handles:
        try:
            return download_dataset(handle, force=force), handle
        except Exception as exc:
            errors.append(f"  • {handle}: {exc}")
    lines = "\n".join(errors)
    pages = "\n".join(f"  • {kaggle_dataset_url(h)}" for h in handles)
    raise RuntimeError(
        "Could not download dataset from any configured Kaggle source.\n"
        f"{lines}\n"
        "Open each page below (logged into Kaggle), accept dataset rules if prompted, "
        "then run training again:\n"
        f"{pages}"
    )


def find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for dirpath, _, files in os.walk(root):
        for name in names:
            if name in files:
                return Path(dirpath) / name
    return None


def find_directory(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    for dirpath, dirnames, _ in os.walk(root):
        for name in names:
            if name in dirnames:
                return Path(dirpath) / name
    return None
