"""Eigenface training and verification service."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenface_live import __version__
from eigenface_live.config import AppConfig, ModelConfig
from eigenface_live.models.enrollment import EnrollmentArtifact
from eigenface_live.models.face_sample import (
    FaceSample,
    VerificationDecision,
    VerificationResult,
)


class FaceRecognitionService:
    """PCA + MLP enrollment with distance-based verification fallback."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def train(self, identity: str, samples: list[FaceSample]) -> EnrollmentArtifact:
        if len(samples) < 12:
            raise ValueError("Need at least 12 face samples to train.")

        matrix = np.vstack([sample.vector for sample in samples])
        labels = np.zeros(len(samples), dtype=np.int64)

        n_components = min(
            self._config.model.pca_components,
            matrix.shape[0] - 1,
            matrix.shape[1],
        )
        if n_components < 5:
            raise ValueError("Not enough variation across captured samples.")

        pca = PCA(n_components=n_components, whiten=True, random_state=self._config.model.random_state)
        projections = pca.fit_transform(matrix)
        prototype = projections.mean(axis=0)

        distances = pairwise_distances(projections, prototype.reshape(1, -1), metric="euclidean").ravel()
        threshold = float(np.percentile(distances, self._config.model.verification_threshold_percentile))

        negatives = self._synthesize_negatives(matrix, count=max(20, len(samples) // 2))
        train_x = np.vstack([matrix, negatives])
        train_y = np.concatenate([labels, np.ones(len(negatives), dtype=np.int64)])

        classifier = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=self._config.model.mlp_hidden,
                        max_iter=self._config.model.mlp_max_iter,
                        random_state=self._config.model.random_state,
                    ),
                ),
            ]
        )
        classifier.fit(train_x, train_y)

        return EnrollmentArtifact(
            identity=identity,
            created_at=datetime.now(timezone.utc),
            face_size=self._config.enrollment.face_size,
            pca=pca,
            classifier=classifier,
            prototype=prototype,
            threshold=threshold,
            sample_count=len(samples),
            app_version=__version__,
        )

    def verify(self, artifact: EnrollmentArtifact, face_vector: np.ndarray) -> VerificationResult:
        projection = artifact.pca.transform(face_vector.reshape(1, -1))
        distance = float(pairwise_distances(projection, artifact.prototype.reshape(1, -1)).ravel()[0])

        probability = float(artifact.classifier.predict_proba(face_vector.reshape(1, -1))[0][0])
        distance_match = distance <= artifact.threshold
        model_match = probability >= 0.5

        if distance_match and model_match:
            decision = VerificationDecision.MATCH
            confidence = min(1.0, (probability + (1.0 - distance / max(artifact.threshold, 1e-6))) / 2.0)
        elif not distance_match and not model_match:
            decision = VerificationDecision.NO_MATCH
            confidence = min(1.0, max(1.0 - probability, distance / max(artifact.threshold, 1e-6)))
        else:
            decision = VerificationDecision.UNCERTAIN
            confidence = 0.5

        return VerificationResult(
            decision=decision,
            confidence=confidence,
            distance=distance,
            threshold=artifact.threshold,
            identity=artifact.identity,
        )

    def _synthesize_negatives(self, positives: np.ndarray, count: int) -> np.ndarray:
        rng = np.random.default_rng(self._config.model.random_state)
        mean = positives.mean(axis=0)
        std = positives.std(axis=0) + 1e-3
        noise = rng.normal(0.0, 1.5, size=(count, positives.shape[1]))
        negatives = np.clip(mean + noise * std, 0.0, 1.0)
        return negatives.astype(np.float32)
