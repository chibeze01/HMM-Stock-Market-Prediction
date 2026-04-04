import math
import unittest

from model.drift_detector import DriftConfig, DriftDetector


class TestDriftDetector(unittest.TestCase):
    def _detector(self, **kwargs):
        return DriftDetector(
            DriftConfig(
                accuracy_floor=0.45,
                log_likelihood_floor=-5.0,
                consecutive_failures=3,
                **kwargs,
            )
        )

    def test_healthy_metrics_not_drifting(self):
        d = self._detector()
        status = d.check(accuracy=0.6, log_likelihood=-2.0)
        self.assertFalse(status.is_drifting)
        self.assertEqual(d.failure_streak, 0)

    def test_single_accuracy_failure_below_threshold(self):
        d = self._detector()
        status = d.check(accuracy=0.3, log_likelihood=-2.0)
        self.assertFalse(status.is_drifting)
        self.assertEqual(d.failure_streak, 1)

    def test_consecutive_failures_trigger_drift(self):
        d = self._detector()
        d.check(accuracy=0.3, log_likelihood=-2.0)
        d.check(accuracy=0.3, log_likelihood=-2.0)
        status = d.check(accuracy=0.3, log_likelihood=-2.0)
        self.assertTrue(status.is_drifting)

    def test_recovery_resets_streak(self):
        d = self._detector()
        d.check(accuracy=0.3, log_likelihood=-2.0)
        status = d.check(accuracy=0.6, log_likelihood=-2.0)
        self.assertFalse(status.is_drifting)
        self.assertEqual(d.failure_streak, 0)

    def test_reset_clears_streak(self):
        d = self._detector()
        d.check(accuracy=0.3, log_likelihood=-2.0)
        d.check(accuracy=0.3, log_likelihood=-2.0)
        d.reset()
        self.assertEqual(d.failure_streak, 0)

    def test_none_accuracy_uses_log_likelihood_only(self):
        d = self._detector()
        status = d.check(accuracy=None, log_likelihood=-8.0)
        self.assertFalse(status.is_drifting)
        self.assertEqual(d.failure_streak, 1)

    def test_log_likelihood_failure_with_consecutive_1(self):
        d = DriftDetector(
            DriftConfig(
                accuracy_floor=0.45,
                log_likelihood_floor=-5.0,
                consecutive_failures=1,
            )
        )
        status = d.check(accuracy=0.6, log_likelihood=-8.0)
        self.assertTrue(status.is_drifting)

    def test_nan_log_likelihood_counts_as_failure(self):
        d = self._detector()
        status = d.check(accuracy=0.6, log_likelihood=float("nan"))
        self.assertEqual(d.failure_streak, 1)
        self.assertIn("log_likelihood", status.reason)

    def test_both_none_is_healthy(self):
        d = self._detector()
        status = d.check(accuracy=None, log_likelihood=None)
        self.assertFalse(status.is_drifting)
        self.assertEqual(d.failure_streak, 0)
        self.assertEqual(status.reason, "metrics_healthy")


if __name__ == "__main__":
    unittest.main()
