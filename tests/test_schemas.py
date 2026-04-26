"""Unit tests for api.schemas Pydantic models."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainRequest(unittest.TestCase):
    def test_defaults(self):
        from api.schemas import TrainRequest

        req = TrainRequest(ticker="AAPL", start_date="2020-01-01", end_date="2023-12-31")
        self.assertEqual(req.ticker, "AAPL")
        self.assertEqual(req.n_hidden_states, 4)
        self.assertEqual(req.covariance_type, "diag")
        self.assertEqual(req.n_iter, 500)
        self.assertEqual(req.random_state, 42)
        self.assertIn("returns", req.features)
        self.assertIn("volatility", req.features)
        self.assertEqual(req.volatility_window, 5)
        self.assertEqual(req.momentum_window, 3)

    def test_custom(self):
        from api.schemas import TrainRequest

        req = TrainRequest(
            ticker="GOOGL",
            start_date="2021-06-01",
            end_date="2023-06-01",
            features=["returns", "momentum"],
            n_hidden_states=3,
            covariance_type="full",
            n_iter=200,
            random_state=7,
            volatility_window=10,
            momentum_window=5,
        )
        self.assertEqual(req.n_hidden_states, 3)
        self.assertEqual(req.features, ["returns", "momentum"])
        self.assertEqual(req.volatility_window, 10)

    def test_bad_hidden_states(self):
        from api.schemas import TrainRequest

        with self.assertRaises(Exception):
            TrainRequest(
                ticker="AAPL",
                start_date="2020-01-01",
                end_date="2023-12-31",
                n_hidden_states=1,
            )

    def test_empty_ticker_rejected(self):
        from api.schemas import TrainRequest

        with self.assertRaises(Exception):
            TrainRequest(ticker="", start_date="2020-01-01", end_date="2023-12-31")


class TestFineTuneRequest(unittest.TestCase):
    def test_fields(self):
        from api.schemas import FineTuneRequest

        req = FineTuneRequest(model_id="abc-123", new_end_date="2024-06-01")
        self.assertEqual(req.model_id, "abc-123")
        self.assertEqual(req.new_end_date, "2024-06-01")


class TestPredictRequest(unittest.TestCase):
    def test_fields(self):
        from api.schemas import PredictRequest

        req = PredictRequest(model_id="abc-123")
        self.assertEqual(req.model_id, "abc-123")


class TestTrainResponse(unittest.TestCase):
    def test_fields(self):
        from api.schemas import TrainResponse

        resp = TrainResponse(
            model_id="abc",
            training_summary={"log_likelihood": -42.0, "n_samples": 100},
            evaluation={"regime_summary": {}},
        )
        self.assertEqual(resp.model_id, "abc")
        self.assertIn("log_likelihood", resp.training_summary)


class TestPredictResponse(unittest.TestCase):
    def test_fields(self):
        from api.schemas import PredictResponse

        resp = PredictResponse(
            predicted_state=2,
            probabilities=[0.1, 0.3, 0.6],
            regime_label="State 2: bullish",
        )
        self.assertEqual(resp.predicted_state, 2)
        self.assertEqual(len(resp.probabilities), 3)


class TestHealthResponse(unittest.TestCase):
    def test_fields(self):
        from api.schemas import HealthResponse

        resp = HealthResponse(status="ok", version="0.1.0")
        self.assertEqual(resp.status, "ok")


if __name__ == "__main__":
    unittest.main()
