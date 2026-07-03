"""Secure enrollment persistence."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

import joblib

from eigenface_live.config import AppConfig
from eigenface_live.models.enrollment import EnrollmentArtifact
from eigenface_live.utils.paths import enrollment_model_path, sanitize_identity


class ModelPersistenceError(RuntimeError):
    """Raised when enrollment artifacts cannot be stored or loaded safely."""


class ModelRepository:
    """Repository for signed enrollment artifacts stored on disk."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._models_dir = config.models_dir
        self._key_path = self._models_dir / ".signing_key"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._models_dir, self._config.security.model_dir_mode)
        if not self._key_path.exists():
            key = secrets.token_bytes(32)
            self._key_path.write_bytes(key)
            os.chmod(self._key_path, self._config.security.model_file_mode)

    def save(self, artifact: EnrollmentArtifact) -> Path:
        identity = sanitize_identity(artifact.identity, self._config.security)
        path = enrollment_model_path(self._models_dir, identity, self._config.security)
        metadata = self._metadata_from_artifact(artifact, identity)
        digest = self._sign(metadata)
        joblib.dump({"artifact": artifact, "metadata": metadata, "digest": digest}, path)
        os.chmod(path, self._config.security.model_file_mode)
        return path

    def load(self, identity: str) -> EnrollmentArtifact:
        path = enrollment_model_path(self._models_dir, identity, self._config.security)
        if not path.exists():
            raise ModelPersistenceError(f"No enrollment found for '{identity}'.")
        package = joblib.load(path)
        if not isinstance(package, dict):
            raise ModelPersistenceError("Enrollment file is corrupted.")
        metadata = package.get("metadata")
        digest = package.get("digest")
        artifact = package.get("artifact")
        if metadata is None or digest is None or artifact is None:
            raise ModelPersistenceError("Enrollment file is incomplete.")
        if not self._verify(metadata, digest):
            raise ModelPersistenceError("Enrollment integrity check failed.")
        if not isinstance(artifact, EnrollmentArtifact):
            raise ModelPersistenceError("Enrollment file has an unexpected format.")
        if not self._metadata_matches_artifact(metadata, artifact):
            raise ModelPersistenceError("Enrollment metadata does not match stored model.")
        return artifact

    def list_identities(self) -> list[str]:
        identities: list[str] = []
        for path in sorted(self._models_dir.glob("*.joblib")):
            try:
                package = joblib.load(path)
                metadata = package.get("metadata")
                if isinstance(metadata, dict) and "identity" in metadata:
                    identities.append(str(metadata["identity"]))
            except Exception:
                continue
        return identities

    def _metadata_from_artifact(self, artifact: EnrollmentArtifact, identity: str) -> dict[str, str | int | float]:
        return {
            "identity": identity,
            "version": artifact.app_version,
            "sample_count": artifact.sample_count,
            "threshold": artifact.threshold,
            "created_at": artifact.created_at.isoformat(),
        }

    def _metadata_matches_artifact(self, metadata: dict, artifact: EnrollmentArtifact) -> bool:
        return (
            metadata.get("identity") == artifact.identity
            and metadata.get("version") == artifact.app_version
            and metadata.get("sample_count") == artifact.sample_count
            and abs(float(metadata.get("threshold", -1)) - artifact.threshold) < 1e-9
            and metadata.get("created_at") == artifact.created_at.isoformat()
        )

    def _sign(self, metadata: dict) -> str:
        key = self._key_path.read_bytes()
        blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(key, blob, hashlib.sha256).hexdigest()

    def _verify(self, metadata: dict, digest: str) -> bool:
        expected = self._sign(metadata)
        return hmac.compare_digest(expected, digest)
