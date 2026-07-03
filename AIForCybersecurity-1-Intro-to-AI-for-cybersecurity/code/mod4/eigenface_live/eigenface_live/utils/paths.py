"""Safe path helpers."""

from __future__ import annotations

import re
from pathlib import Path

from eigenface_live.config import SecurityConfig


def sanitize_identity(identity: str, security: SecurityConfig) -> str:
    """Validate and normalize an enrollment identity label."""
    name = identity.strip()
    if not name:
        raise ValueError("Identity name cannot be empty.")
    if len(name) > security.max_identity_length:
        raise ValueError(
            f"Identity name must be at most {security.max_identity_length} characters."
        )
    if not re.fullmatch(security.identity_pattern, name):
        raise ValueError(
            "Identity may contain letters, numbers, spaces, hyphens, and underscores only."
        )
    return name


def enrollment_model_path(models_dir: Path, identity: str, security: SecurityConfig) -> Path:
    """Resolve a safe on-disk path for a stored enrollment."""
    safe = sanitize_identity(identity, security)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", safe).strip("_").lower()
    if not slug:
        raise ValueError("Identity resolves to an empty filename.")
    base = models_dir.resolve()
    target = (base / f"{slug}.joblib").resolve()
    if base not in target.parents and target != base:
        raise ValueError("Invalid enrollment path.")
    return target
