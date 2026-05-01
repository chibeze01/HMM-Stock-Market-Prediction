import unittest

import numpy as np
import pandas as pd

from model.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    WalkForwardConfig,
    walk_forward_backtest,
)


class TestBacktestConfig(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = BacktestConfig()
        self.assertEqual(cfg.initial_capital, 10_000.0)
        self.assertEqual(cfg.position_size, 1.0)
        self.assertEqual(cfg.commission, 0.001)
        self.assertEqual(cfg.slippage, 0.0005)
        self.assertIsNone(cfg.long_states)

    def test_rejects_negative_initial_capital(self):
        with self.assertRaises(ValueError):
            BacktestConfig(initial_capital=-1.0)

    def test_rejects_zero_initial_capital(self):
        with self.assertRaises(ValueError):
            BacktestConfig(initial_capital=0.0)

    def test_rejects_position_size_out_of_range(self):
        with self.assertRaises(ValueError):
            BacktestConfig(position_size=0.0)
        with self.assertRaises(ValueError):
            BacktestConfig(position_size=1.5)

    def test_rejects_negative_commission(self):
        with self.assertRaises(ValueError):
            BacktestConfig(commission=-0.01)

    def test_rejects_negative_slippage(self):
        with self.assertRaises(ValueError):
            BacktestConfig(slippage=-0.001)

    def test_accepts_custom_long_states(self):
        cfg = BacktestConfig(long_states=frozenset({0, 2}))
        self.assertEqual(cfg.long_states, frozenset({0, 2}))


class TestTradeAndResult(unittest.TestCase):
    def test_trade_creation(self):
        trade = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=100.0,
            exit_price=105.0,
            direction="long",
            pnl=5.0,
            state=0,
        )
        self.assertEqual(trade.pnl, 5.0)
        self.assertEqual(trade.direction, "long")

    def test_backtest_result_creation(self):
        equity = pd.Series([10000.0, 10050.0, 10020.0])
        result = BacktestResult(
            trades=[],
            equity_curve=equity,
            total_return=0.002,
            annualized_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=-0.003,
            win_rate=0.0,
            n_trades=0,
            benchmark_return=0.01,
        )
        self.assertEqual(result.n_trades, 0)
        self.assertAlmostEqual(result.sharpe_ratio, 1.2)

    def test_backtest_result_with_trades(self):
        trade = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=100.0,
            exit_price=105.0,
            direction="long",
            pnl=50.0,
            state=0,
        )
        result = BacktestResult(
            trades=[trade],
            equity_curve=pd.Series([10000.0, 10050.0]),
            total_return=0.005,
            annualized_return=0.10,
            sharpe_ratio=1.5,
            max_drawdown=-0.001,
            win_rate=1.0,
            n_trades=1,
            benchmark_return=0.01,
        )
        self.assertEqual(result.n_trades, 1)
        self.assertEqual(result.trades[0].pnl, 50.0)


def _make_regime_summary():
    """Regime summary with states 0 (bullish) and 1 (bearish)."""
    return pd.DataFrame(
        {
            "mean_return": [0.002, -0.001],
            "volatility": [0.01, 0.02],
            "avg_close": [150.0, 148.0],
            "sample_count": [100, 80],
        },
        index=pd.Index([0, 1], name="HiddenState"),
    )


class TestDetectLongStates(unittest.TestCase):
    def test_detects_positive_mean_return_states(self):
        engine = BacktestEngine()
        summary = _make_regime_summary()
        long_states = engine._detect_long_states(summary)
        self.assertEqual(long_states, frozenset({0}))

    def test_all_bearish_returns_empty(self):
        engine = BacktestEngine()
        summary = pd.DataFrame(
            {"mean_return": [-0.001, -0.002]},
            index=pd.Index([0, 1], name="HiddenState"),
        )
        long_states = engine._detect_long_states(summary)
        self.assertEqual(long_states, frozenset())

    def test_multiple_bullish_states(self):
        engine = BacktestEngine()
        summary = pd.DataFrame(
            {"mean_return": [0.003, -0.001, 0.001]},
            index=pd.Index([0, 1, 2], name="HiddenState"),
        )
        long_states = engine._detect_long_states(summary)
        self.assertEqual(long_states, frozenset({0, 2}))


class TestGenerateSignals(unittest.TestCase):
    def test_signals_match_long_states(self):
        engine = BacktestEngine()
        hidden_states = np.array([0, 1, 0, 1, 0])
        long_states = frozenset({0})
        signals = engine._generate_signals(hidden_states, long_states)
        np.testing.assert_array_equal(signals, np.array([1, 0, 1, 0, 1]))

    def test_no_long_states_all_zero(self):
        engine = BacktestEngine()
        hidden_states = np.array([0, 1, 2])
        signals = engine._generate_signals(hidden_states, frozenset())
        np.testing.assert_array_equal(signals, np.array([0, 0, 0]))


def _make_frame_and_signals(n=20):
    """Build a simple frame with Close/Returns and binary signals."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    frame = pd.DataFrame({"Close": close}, index=dates)
    frame["Returns"] = frame["Close"].pct_change().fillna(0)
    # Alternate: long on even days, flat on odd
    signals = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    return frame, signals


class TestSimulateTrades(unittest.TestCase):
    def test_produces_trades_and_equity_curve(self):
        engine = BacktestEngine()
        frame, signals = _make_frame_and_signals()
        trades, equity = engine._simulate_trades(frame, signals)
        self.assertIsInstance(trades, list)
        self.assertGreater(len(trades), 0)
        self.assertIsInstance(equity, pd.Series)
        self.assertEqual(len(equity), len(frame))
        # equity starts at initial capital
        self.assertAlmostEqual(equity.iloc[0], engine.config.initial_capital)

    def test_no_signals_no_trades(self):
        engine = BacktestEngine()
        frame, _ = _make_frame_and_signals()
        signals = np.zeros(len(frame), dtype=int)
        trades, equity = engine._simulate_trades(frame, signals)
        self.assertEqual(len(trades), 0)
        # equity stays flat
        self.assertTrue((equity == engine.config.initial_capital).all())

    def test_commission_reduces_pnl(self):
        cfg_no_cost = BacktestConfig(commission=0.0, slippage=0.0)
        cfg_with_cost = BacktestConfig(commission=0.01, slippage=0.0)
        frame, signals = _make_frame_and_signals()
        trades_no, eq_no = BacktestEngine(cfg_no_cost)._simulate_trades(frame, signals)
        trades_yes, eq_yes = BacktestEngine(cfg_with_cost)._simulate_trades(frame, signals)
        total_pnl_no = sum(t.pnl for t in trades_no)
        total_pnl_yes = sum(t.pnl for t in trades_yes)
        self.assertGreater(total_pnl_no, total_pnl_yes)

    def test_trade_fields_populated(self):
        engine = BacktestEngine()
        frame, signals = _make_frame_and_signals()
        trades, _ = engine._simulate_trades(frame, signals)
        t = trades[0]
        self.assertIsInstance(t.entry_date, pd.Timestamp)
        self.assertIsInstance(t.exit_date, pd.Timestamp)
        self.assertGreater(t.entry_price, 0)
        self.assertGreater(t.exit_price, 0)
        self.assertEqual(t.direction, "long")


class TestFinancialMetrics(unittest.TestCase):
    def test_sharpe_ratio_positive_returns(self):
        engine = BacktestEngine()
        daily_returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.005])
        sharpe = engine._sharpe_ratio(daily_returns)
        self.assertGreater(sharpe, 0)

    def test_sharpe_ratio_zero_std(self):
        engine = BacktestEngine()
        daily_returns = pd.Series([0.0, 0.0, 0.0])
        sharpe = engine._sharpe_ratio(daily_returns)
        self.assertEqual(sharpe, 0.0)

    def test_sharpe_ratio_empty(self):
        engine = BacktestEngine()
        sharpe = engine._sharpe_ratio(pd.Series([], dtype=float))
        self.assertEqual(sharpe, 0.0)

    def test_max_drawdown_known_curve(self):
        engine = BacktestEngine()
        equity = pd.Series([100.0, 110.0, 90.0, 95.0])
        dd = engine._max_drawdown(equity)
        self.assertAlmostEqual(dd, -20.0 / 110.0, places=4)

    def test_max_drawdown_monotonic_increasing(self):
        engine = BacktestEngine()
        equity = pd.Series([100.0, 105.0, 110.0, 115.0])
        dd = engine._max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)

    def test_max_drawdown_empty(self):
        engine = BacktestEngine()
        dd = engine._max_drawdown(pd.Series([], dtype=float))
        self.assertEqual(dd, 0.0)


def _make_backtest_inputs(n=100):
    """Build frame, hidden_states, regime_summary for a full run() test."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, n))
    frame = pd.DataFrame({"Close": close}, index=dates)
    frame["Returns"] = frame["Close"].pct_change().fillna(0)
    # 2 states: 0=bullish, 1=bearish alternating in blocks
    hidden_states = np.array([0 if i < n // 2 else 1 for i in range(n)])
    regime_summary = pd.DataFrame(
        {
            "mean_return": [0.001, -0.001],
            "volatility": [0.01, 0.015],
            "avg_close": [105.0, 100.0],
            "sample_count": [n // 2, n // 2],
        },
        index=pd.Index([0, 1], name="HiddenState"),
    )
    return frame, hidden_states, regime_summary


class TestBacktestEngineRun(unittest.TestCase):
    def test_run_returns_backtest_result(self):
        engine = BacktestEngine()
        frame, hidden_states, regime_summary = _make_backtest_inputs()
        result = engine.run(frame, hidden_states, regime_summary)
        self.assertIsInstance(result, BacktestResult)
        self.assertGreaterEqual(result.n_trades, 0)
        self.assertEqual(len(result.equity_curve), len(frame))

    def test_run_with_explicit_long_states(self):
        cfg = BacktestConfig(long_states=frozenset({0}))
        engine = BacktestEngine(cfg)
        frame, hidden_states, regime_summary = _make_backtest_inputs()
        result = engine.run(frame, hidden_states, regime_summary)
        self.assertIsInstance(result, BacktestResult)

    def test_run_all_bearish_no_trades(self):
        engine = BacktestEngine()
        frame, hidden_states, _ = _make_backtest_inputs()
        # All states have negative returns
        regime_summary = pd.DataFrame(
            {"mean_return": [-0.001, -0.002]},
            index=pd.Index([0, 1], name="HiddenState"),
        )
        result = engine.run(frame, hidden_states, regime_summary)
        self.assertEqual(result.n_trades, 0)

    def test_run_computes_benchmark_return(self):
        engine = BacktestEngine()
        frame, hidden_states, regime_summary = _make_backtest_inputs()
        result = engine.run(frame, hidden_states, regime_summary)
        expected_benchmark = (frame["Close"].iloc[-1] - frame["Close"].iloc[0]) / frame[
            "Close"
        ].iloc[0]
        self.assertAlmostEqual(result.benchmark_return, expected_benchmark, places=4)


class TestWalkForwardConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = WalkForwardConfig()
        self.assertEqual(cfg.n_splits, 5)
        self.assertEqual(cfg.train_size, 252)
        self.assertEqual(cfg.test_size, 63)

    def test_rejects_invalid_splits(self):
        with self.assertRaises(ValueError):
            WalkForwardConfig(n_splits=0)

    def test_rejects_invalid_train_size(self):
        with self.assertRaises(ValueError):
            WalkForwardConfig(train_size=0)

    def test_rejects_invalid_test_size(self):
        with self.assertRaises(ValueError):
            WalkForwardConfig(test_size=0)


class TestWalkForwardBacktest(unittest.TestCase):
    def test_walk_forward_returns_results_list(self):
        # Build enough data for at least 2 splits: 2*(50+25)=150 rows
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, n))
        raw_data = pd.DataFrame({"Close": close}, index=dates)

        wf_cfg = WalkForwardConfig(n_splits=2, train_size=50, test_size=25)
        results = walk_forward_backtest(
            raw_data=raw_data,
            wf_config=wf_cfg,
            backtest_config=BacktestConfig(),
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, BacktestResult)

    def test_walk_forward_insufficient_data(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        raw_data = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)
        wf_cfg = WalkForwardConfig(n_splits=2, train_size=50, test_size=25)
        with self.assertRaises(ValueError):
            walk_forward_backtest(
                raw_data=raw_data,
                wf_config=wf_cfg,
                backtest_config=BacktestConfig(),
            )


if __name__ == "__main__":
    unittest.main()
