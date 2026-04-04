import tempfile
import unittest

import numpy as np
import pandas as pd

from model.drift_detector import DriftConfig
from model.hmm import HMMConfig
from model.retraining_pipeline import RetrainingConfig, RetrainingPipeline
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


class TestRetrainingPipeline(unittest.TestCase):
    def _make_config(
        self,
        tmp,
        accuracy_floor=0.0,
        log_likelihood_floor=-999.0,
        consecutive_failures=3,
    ):
        return RetrainingConfig(
            ticker="GOOGL",
            lookback_days=365,
            hmm_config=HMMConfig(n_hidden_states=2, n_iter=10),
            preprocess_config=PreprocessingConfig(),
            drift_config=DriftConfig(
                accuracy_floor=accuracy_floor,
                log_likelihood_floor=log_likelihood_floor,
                consecutive_failures=consecutive_failures,
            ),
            registry_dir=tmp,
        )

    def test_first_cycle_trains_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = RetrainingPipeline(
                self._make_config(tmp), _fetch_fn=_synthetic_fetch
            )
            summary = pipeline.run_cycle()
            self.assertTrue(summary.retrained)
            self.assertEqual(summary.reason, "initial_training")
            self.assertIsNotNone(summary.model_version)

    def test_second_cycle_no_retrain_when_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = RetrainingPipeline(
                self._make_config(tmp), _fetch_fn=_synthetic_fetch
            )
            pipeline.run_cycle()
            summary = pipeline.run_cycle()
            self.assertFalse(summary.retrained)
            self.assertEqual(summary.reason, "no_drift")

    def test_cycle_retrains_when_drifting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(
                tmp,
                accuracy_floor=1.0,
                log_likelihood_floor=999.0,
                consecutive_failures=1,
            )
            pipeline = RetrainingPipeline(config, _fetch_fn=_synthetic_fetch)
            pipeline.run_cycle()  # initial training
            summary = pipeline.run_cycle()  # should detect drift and retrain
            self.assertTrue(summary.retrained)

    def test_cycle_summary_has_drift_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = RetrainingPipeline(
                self._make_config(tmp), _fetch_fn=_synthetic_fetch
            )
            summary = pipeline.run_cycle()
            self.assertIsNotNone(summary.drift_status)
            self.assertIsNotNone(summary.ran_at)

    def test_model_version_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config(
                tmp,
                accuracy_floor=1.0,
                log_likelihood_floor=999.0,
                consecutive_failures=1,
            )
            pipeline = RetrainingPipeline(config, _fetch_fn=_synthetic_fetch)
            s1 = pipeline.run_cycle()  # initial: v1
            s2 = pipeline.run_cycle()  # drift retrain: v2
            self.assertEqual(s1.model_version, 1)
            self.assertEqual(s2.model_version, 2)


if __name__ == "__main__":
    unittest.main()
