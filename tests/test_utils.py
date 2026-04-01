import unittest

import pandas as pd

from model.utils import (
    PreprocessedData,
    PreprocessingConfig,
    fetch_stock_data,
    preprocess_data,
)


def _build_dummy_frame():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    close = [100, 101, 103, 102, 101, 105, 104, 106]
    return pd.DataFrame({"Close": close}, index=dates)


class UtilsTestCase(unittest.TestCase):
    def test_fetch_stock_data_validates_inputs(self):
        with self.assertRaises(ValueError):
            fetch_stock_data("", "2024-01-01", "2024-01-31")
        with self.assertRaises(ValueError):
            fetch_stock_data("AAPL", "2024-01-10", "2024-01-01")

    def test_preprocess_data_does_not_mutate_source(self):
        source = _build_dummy_frame()
        result = preprocess_data(source)

        self.assertIsInstance(result, PreprocessedData)
        self.assertNotIn("Returns", source.columns)
        self.assertEqual(result.frame.shape[0], result.features.shape[0])
        self.assertEqual(result.frame.shape[0], result.states.shape[0])

    def test_preprocess_data_custom_features(self):
        source = _build_dummy_frame()
        config = PreprocessingConfig(features=("momentum",))
        result = preprocess_data(source, config)

        self.assertEqual(result.features.shape[1], 1)
        self.assertIn("Momentum", result.frame.columns)
        self.assertTrue(set(result.frame["State"].unique()).issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
