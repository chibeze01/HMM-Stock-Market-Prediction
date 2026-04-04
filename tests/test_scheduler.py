import tempfile
import unittest

import numpy as np
import pandas as pd

from model.drift_detector import DriftConfig
from model.hmm import HMMConfig
from model.retraining_pipeline import RetrainingConfig, RetrainingPipeline
from model.scheduler import RetrainingScheduler
from model.utils import PreprocessingConfig


def _synthetic_fetch(ticker, start, end, **kwargs):
    rng = np.random.default_rng(42)
    n = 300
    close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": 1_000_000,
        },
        index=dates,
    )


class TestRetrainingScheduler(unittest.TestCase):
    def _make_pipeline(self, tmp):
        config = RetrainingConfig(
            ticker="GOOGL",
            lookback_days=365,
            hmm_config=HMMConfig(n_hidden_states=2, n_iter=10),
            preprocess_config=PreprocessingConfig(),
            drift_config=DriftConfig(),
            registry_dir=tmp,
        )
        return RetrainingPipeline(config, _fetch_fn=_synthetic_fetch)

    def test_not_running_initially(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            scheduler = RetrainingScheduler(pipeline, interval_hours=24.0)
            self.assertFalse(scheduler.is_running)
            self.assertIsNone(scheduler.last_summary)

    def test_trigger_now_runs_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            scheduler = RetrainingScheduler(pipeline, interval_hours=24.0)
            summary = scheduler.trigger_now()
            self.assertTrue(summary.retrained)
            self.assertEqual(summary.reason, "initial_training")
            self.assertIsNotNone(scheduler.last_summary)

    def test_start_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            scheduler = RetrainingScheduler(pipeline, interval_hours=24.0)
            scheduler.start()
            self.assertTrue(scheduler.is_running)
            scheduler.stop()
            self.assertFalse(scheduler.is_running)


if __name__ == "__main__":
    unittest.main()
