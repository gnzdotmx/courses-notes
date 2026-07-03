import unittest

import numpy as np

from eigenface_live.config import DEFAULT_CONFIG, SecurityConfig
from eigenface_live.models.face_sample import FaceDetection, FaceSample, VerificationDecision
from eigenface_live.services.face_recognition_service import FaceRecognitionService
from eigenface_live.services.frame_collector import OvalMotionCollector
from eigenface_live.utils.paths import sanitize_identity


class PathUtilsTests(unittest.TestCase):
    def test_sanitize_identity_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_identity("  ", SecurityConfig())

    def test_sanitize_identity_accepts_safe_names(self) -> None:
        self.assertEqual(sanitize_identity("Alice-1", SecurityConfig()), "Alice-1")


class OvalMotionCollectorTests(unittest.TestCase):
    def test_collects_samples_after_oval_motion(self) -> None:
        config = DEFAULT_CONFIG.enrollment
        collector = OvalMotionCollector(config)
        collector.start()

        vector = np.ones(config.face_size[0] * config.face_size[1], dtype=np.float32) * 0.5
        progress = None
        for angle in np.linspace(0, 2 * np.pi, 80):
            cx = 200 + 60 * np.cos(angle)
            cy = 180 + 40 * np.sin(angle)
            detection = FaceDetection(bbox=(int(cx - 40), int(cy - 50), 80, 100), center=(cx, cy))
            progress = collector.ingest(detection, vector)

        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertGreater(progress.collected, 0)


class FaceRecognitionServiceTests(unittest.TestCase):
    def test_train_and_verify_same_person(self) -> None:
        config = DEFAULT_CONFIG
        service = FaceRecognitionService(config)
        dim = config.enrollment.face_size[0] * config.enrollment.face_size[1]
        rng = np.random.default_rng(42)
        base = rng.random(dim, dtype=np.float32)
        samples = [
            FaceSample.from_vector(np.clip(base + rng.normal(0, 0.02, dim), 0, 1))
            for _ in range(20)
        ]
        artifact = service.train("tester", samples)
        probe = np.clip(base + rng.normal(0, 0.02, dim), 0, 1)
        result = service.verify(artifact, probe)
        self.assertIn(result.decision, {VerificationDecision.MATCH, VerificationDecision.UNCERTAIN})


if __name__ == "__main__":
    unittest.main()
