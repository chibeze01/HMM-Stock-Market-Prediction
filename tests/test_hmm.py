import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.hmm import HMMConfig, HMMStockPredictor


def _synthetic_features(rows: int = 60) -> np.ndarray:
    x = np.linspace(0, 3 * np.pi, rows)
    return np.column_stack((np.sin(x), np.cos(x)))


class HMMStockPredictorTests(unittest.TestCase):
    def test_train_and_predict(self):
        config = HMMConfig(n_hidden_states=2, n_iter=50, random_state=7)
        model = HMMStockPredictor(config)
        features = _synthetic_features()

        summary = model.train(features)

        self.assertEqual(summary.n_samples, features.shape[0])
        self.assertIsInstance(model.predict_next_day_state(features), int)
        self.assertEqual(len(model.training_history), 1)

    def test_fine_tune_and_history_grows(self):
        model = HMMStockPredictor(HMMConfig(n_hidden_states=2, n_iter=50))
        base = _synthetic_features()
        model.train(base)

        extra = _synthetic_features(rows=30)
        model.fine_tune(extra)

        self.assertGreaterEqual(len(model.training_history), 2)

    def test_save_and_load_roundtrip(self):
        model = HMMStockPredictor(HMMConfig(n_hidden_states=2, n_iter=50))
        features = _synthetic_features()
        model.train(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)
            loaded = HMMStockPredictor.load(path)

        self.assertEqual(loaded.config.n_hidden_states, model.config.n_hidden_states)
        self.assertIsInstance(loaded.predict_next_day_state(features), int)


if __name__ == "__main__":
    unittest.main()
