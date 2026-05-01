from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import compute_regime_summary
from .hmm import HMMConfig, HMMStockPredictor
from .logging_utils import get_logger
from .utils import PreprocessingConfig, preprocess_data

logger = get_logger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size: float = 1.0
    commission: float = 0.001
    slippage: float = 0.0005
    long_states: frozenset[int] | None = None

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")
        if not (0 < self.position_size <= 1.0):
            raise ValueError("position_size must be in (0, 1].")
        if self.commission < 0:
            raise ValueError("commission must be >= 0.")
        if self.slippage < 0:
            raise ValueError("slippage must be >= 0.")


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str
    pnl: float
    state: int


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    benchmark_return: float


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def _detect_long_states(self, regime_summary: pd.DataFrame) -> frozenset[int]:
        bullish = regime_summary.index[regime_summary["mean_return"] > 0]
        return frozenset(int(s) for s in bullish)

    def _generate_signals(
        self, hidden_states: np.ndarray, long_states: frozenset[int]
    ) -> np.ndarray:
        return np.isin(hidden_states, list(long_states)).astype(int)

    # ⚡ Bolt: Replaced row-by-row iteration in Python with vectorized array operations.
    # Uses `np.diff` to identify entry/exit indices and evaluates returns using
    # fast array calculations, yielding significant speedups for long backtests.
    def _simulate_trades(
        self, frame: pd.DataFrame, signals: np.ndarray
    ) -> tuple[list[Trade], pd.Series]:
        cfg = self.config
        capital = float(cfg.initial_capital)
        equity = np.full(len(frame), capital)
        dates = frame.index
        close = frame["Close"].to_numpy()
        n_days = len(frame)

        if n_days <= 1:
            return [], pd.Series(equity, index=dates)

        # Ensure mask is strictly 1 or 0 and we don't enter on day 0
        mask = (signals == 1).astype(int)
        mask[0] = 0

        diff = np.diff(mask, prepend=0)
        entries = np.where(diff == 1)[0]
        exits = np.where(diff == -1)[0]

        if len(entries) == 0:
            return [], pd.Series(equity, index=dates)

        if len(entries) > len(exits):
            exits = np.append(exits, n_days - 1)

        n_trades = len(entries)

        entry_prices = close[entries]
        exit_prices = close[exits]

        cost = cfg.commission + cfg.slippage
        gross_returns = (exit_prices - entry_prices) / entry_prices
        net_returns = gross_returns - 2 * cost

        multipliers = 1 + cfg.position_size * net_returns
        capitals_after = capital * np.cumprod(multipliers)

        capitals_before = np.empty(n_trades, dtype=float)
        capitals_before[0] = capital
        if n_trades > 1:
            capitals_before[1:] = capitals_after[:-1]

        pnls = capitals_before * cfg.position_size * net_returns

        trades = [None] * n_trades
        for i in range(n_trades):
            trades[i] = Trade(
                entry_date=dates[entries[i]],
                exit_date=dates[exits[i]],
                entry_price=entry_prices[i],
                exit_price=exit_prices[i],
                direction="long",
                pnl=pnls[i],
                state=0,
            )

            start_idx = exits[i]
            end_idx = exits[i + 1] if i + 1 < n_trades else n_days
            equity[start_idx:end_idx] = capitals_after[i]

        return trades, pd.Series(equity, index=dates)

    def _sharpe_ratio(self, daily_returns: pd.Series) -> float:
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) == 0:
            return 0.0
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return float(drawdown.min())

    def run(
        self,
        frame: pd.DataFrame,
        hidden_states: np.ndarray,
        regime_summary: pd.DataFrame,
    ) -> BacktestResult:
        long_states = self.config.long_states
        if long_states is None:
            long_states = self._detect_long_states(regime_summary)

        signals = self._generate_signals(hidden_states, long_states)
        trades, equity_curve = self._simulate_trades(frame, signals)

        n_trades = len(trades)
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        n_days = len(frame)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        daily_returns = equity_curve.pct_change().dropna()
        sharpe = self._sharpe_ratio(daily_returns)
        max_dd = self._max_drawdown(equity_curve)
        win_rate = sum(1 for t in trades if t.pnl > 0) / n_trades if n_trades else 0.0
        benchmark_return = (frame["Close"].iloc[-1] - frame["Close"].iloc[0]) / frame["Close"].iloc[
            0
        ]

        logger.info(
            "Backtest complete: trades=%d sharpe=%.2f max_dd=%.2f%% return=%.2f%%",
            n_trades,
            sharpe,
            max_dd * 100,
            total_return * 100,
        )
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            benchmark_return=benchmark_return,
        )


@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int = 5
    train_size: int = 252
    test_size: int = 63

    def __post_init__(self):
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        if self.train_size < 1:
            raise ValueError("train_size must be >= 1.")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1.")


def walk_forward_backtest(
    raw_data: pd.DataFrame,
    wf_config: WalkForwardConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    preprocess_config: PreprocessingConfig | None = None,
    hmm_config: HMMConfig | None = None,
) -> list[BacktestResult]:
    wf_cfg = wf_config or WalkForwardConfig()
    bt_cfg = backtest_config or BacktestConfig()
    pp_cfg = preprocess_config or PreprocessingConfig()
    hmm_cfg = hmm_config or HMMConfig()

    total_needed = wf_cfg.n_splits * wf_cfg.test_size + wf_cfg.train_size
    if len(raw_data) < total_needed:
        raise ValueError(f"Insufficient data: need {total_needed} rows, have {len(raw_data)}.")

    results: list[BacktestResult] = []
    engine = BacktestEngine(bt_cfg)

    for split in range(wf_cfg.n_splits):
        train_start = split * wf_cfg.test_size
        train_end = train_start + wf_cfg.train_size
        test_end = train_end + wf_cfg.test_size

        train_raw = raw_data.iloc[train_start:train_end]
        test_raw = raw_data.iloc[train_end:test_end]

        train_data = preprocess_data(train_raw, pp_cfg)
        test_data = preprocess_data(test_raw, pp_cfg)

        predictor = HMMStockPredictor(hmm_cfg)
        predictor.train(train_data.features)

        hidden_states = predictor.model.predict(test_data.features)
        regime_summary = compute_regime_summary(
            train_data.frame, predictor.model.predict(train_data.features)
        )

        result = engine.run(test_data.frame, hidden_states, regime_summary)
        results.append(result)
        logger.info(
            "Walk-forward split %d/%d: trades=%d sharpe=%.2f",
            split + 1,
            wf_cfg.n_splits,
            result.n_trades,
            result.sharpe_ratio,
        )

    return results
