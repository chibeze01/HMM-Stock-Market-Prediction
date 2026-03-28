import unittest

import numpy as np
import pandas as pd

from model.evaluation import (
    compute_regime_summary,
    rolling_directional_accuracy,
    rolling_log_likelihood,
    run_evaluation,
)
from model.hmm import HMMConfig, HMMStockPredictor


def _dummy_frame(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100, 110, rows) + np.random.normal(0, 0.5, rows)
    frame = pd.DataFrame({"Close": close}, index=dates)
    frame["Returns"] = frame["Close"].pct_change()
    frame = frame.dropna()
    return frame


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.frame = _dummy_frame()
        features = np.column_stack(
            (
                self.frame["Returns"].values.reshape(-1, 1),
                self.frame["Returns"].values.reshape(-1, 1),
            )
        )
        self.model = HMMStockPredictor(HMMConfig(n_hidden_states=2, n_iter=50))
        self.model.train(features)
        self.features = features

    def test_regime_summary_columns(self):
        hidden_states = self.model.model.predict(self.features)
        summary = compute_regime_summary(self.frame, hidden_states)
        self.assertIn("mean_return", summary.columns)
        self.assertEqual(len(summary), self.model.config.n_hidden_states)

    def test_directional_accuracy_series(self):
        hidden_states = self.model.model.predict(self.features)
        accuracy = rolling_directional_accuracy(self.frame, hidden_states, window=5)
        self.assertGreaterEqual(accuracy.shape[0], 1)

    def test_log_likelihood_series(self):
        series = rolling_log_likelihood(self.model, self.features, self.frame.index, window=5)
        self.assertGreaterEqual(series.shape[0], 1)

    def test_run_evaluation_bundle(self):
        bundle = run_evaluation(self.model, self.frame, self.features, window=5)
        self.assertFalse(bundle.regime_summary.empty)
        self.assertIsNotNone(bundle.rolling_accuracy.index)


if __name__ == "__main__":
    unittest.main()
