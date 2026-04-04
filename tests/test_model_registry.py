import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.hmm import HMMConfig, HMMStockPredictor
from model.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def _make_trained_predictor(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        predictor = HMMStockPredictor(HMMConfig(n_hidden_states=2, n_iter=10))
        summary = predictor.train(X)
        return predictor, summary

    def test_empty_registry_has_no_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            self.assertFalse(registry.has_any_version())
            self.assertEqual(registry.list_versions(), [])

    def test_save_version_creates_file_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            predictor, summary = self._make_trained_predictor()
            version = registry.save_version(predictor, "GOOGL", summary)
            self.assertEqual(version.version, 1)
            self.assertTrue(Path(tmp, "model_v1.pkl").exists())
            self.assertTrue(registry.has_any_version())

    def test_list_versions_grows_with_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            predictor, summary = self._make_trained_predictor()
            registry.save_version(predictor, "GOOGL", summary)
            registry.save_version(predictor, "GOOGL", summary)
            self.assertEqual(len(registry.list_versions()), 2)

    def test_load_latest_returns_most_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            predictor, summary = self._make_trained_predictor()
            registry.save_version(predictor, "GOOGL", summary)
            registry.save_version(predictor, "GOOGL", summary)
            loaded = registry.load_latest()
            self.assertIsNotNone(loaded)
            self.assertTrue(hasattr(loaded.model, "transmat_"))

    def test_load_version_loads_specific_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            predictor, summary = self._make_trained_predictor()
            v1 = registry.save_version(predictor, "GOOGL", summary)
            registry.save_version(predictor, "GOOGL", summary)
            loaded = registry.load_version(1)
            self.assertEqual(loaded.training_history[0].n_samples, v1.n_samples)

    def test_load_version_raises_for_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            with self.assertRaises(ValueError):
                registry.load_version(99)

    def test_load_latest_raises_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            with self.assertRaises(RuntimeError):
                registry.load_latest()


if __name__ == "__main__":
    unittest.main()
