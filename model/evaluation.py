from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import HMMStockPredictor
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationBundle:
    regime_summary: pd.DataFrame
    rolling_accuracy: pd.Series
    rolling_log_likelihood: pd.Series


def compute_regime_summary(frame: pd.DataFrame, hidden_states: np.ndarray) -> pd.DataFrame:
    """
    Aggregates realized metrics for each inferred regime.
    """
    if len(frame) != len(hidden_states):
        raise ValueError("frame and hidden_states must have the same length.")
    summary = (
        frame.assign(HiddenState=hidden_states.flatten())
        .groupby("HiddenState")
        .agg(
            mean_return=("Returns", "mean"),
            volatility=("Returns", "std"),
            avg_close=("Close", "mean"),
            sample_count=("Close", "count"),
        )
        .sort_index()
    )
    logger.debug("Computed regime summary for %s states", summary.shape[0])
    return summary


def _infer_direction_map(frame: pd.DataFrame, hidden_states: np.ndarray) -> dict[int, int]:
    grouped = (
        frame.assign(HiddenState=hidden_states.flatten()).groupby("HiddenState")["Returns"].mean()
    )
    return dict(zip(grouped.index, np.where(grouped >= 0, 1, -1), strict=False))


def rolling_directional_accuracy(
    frame: pd.DataFrame,
    hidden_states: np.ndarray,
    window: int = 20,
    direction_map: dict[int, int] | None = None,
) -> pd.Series:
    """
    Computes a rolling proportion of days where the inferred regime direction
    matches the realized sign of returns.
    """
    if window < 2:
        raise ValueError("window must be >= 2.")
    if len(frame) != len(hidden_states):
        raise ValueError("frame and hidden_states must have the same length.")

    direction_map = direction_map or _infer_direction_map(frame, hidden_states)

    # ⚡ Bolt: Replace np.vectorize with direct array indexing for contiguous integer lookup.
    # np.vectorize iterates in Python space which is an anti-pattern. Array indexing
    # maps integer state values natively in C, offering significant performance gains.
    max_state = max(direction_map.keys()) if direction_map else 0
    if hidden_states.size > 0:
        max_state = max(max_state, int(np.max(hidden_states)))
    mapping_arr = np.array([direction_map.get(i, 0) for i in range(max_state + 1)])
    predicted_direction = mapping_arr[hidden_states.flatten()]

    realized_direction = np.sign(frame["Returns"])
    accuracy = (predicted_direction == np.sign(realized_direction)).astype(int)
    result = pd.Series(accuracy, index=frame.index).rolling(window=window).mean().dropna()
    logger.debug("Calculated rolling accuracy window=%s points=%s", window, result.shape[0])
    return result


def rolling_log_likelihood(
    model: HMMStockPredictor, X: np.ndarray, index: pd.Index, window: int = 20
) -> pd.Series:
    """
    Smooths per-observation log likelihood into a rolling diagnostic series.
    """
    if window < 2:
        raise ValueError("window must be >= 2.")
    if not hasattr(model.model, "_compute_log_likelihood"):
        raise AttributeError("Model must provide _compute_log_likelihood.")
    log_likelihood = model.model._compute_log_likelihood(X)
    per_sample = np.logaddexp.reduce(log_likelihood, axis=1)
    series = pd.Series(per_sample, index=index).rolling(window=window).mean().dropna()
    logger.debug("Computed rolling log-likelihood window=%s points=%s", window, series.shape[0])
    return series


def run_evaluation(
    model: HMMStockPredictor,
    frame: pd.DataFrame,
    features: np.ndarray,
    window: int = 20,
) -> EvaluationBundle:
    """
    Convenience helper that generates common evaluation artifacts
    used by the UI.
    """
    hidden_states = model.model.predict(features)
    regime_df = compute_regime_summary(frame, hidden_states)
    accuracy = rolling_directional_accuracy(frame, hidden_states, window=window)
    log_like = rolling_log_likelihood(model, features, frame.index, window=window)
    logger.info(
        "Evaluation completed: regimes=%s accuracy_points=%s loglike_points=%s",
        regime_df.shape[0],
        accuracy.shape[0],
        log_like.shape[0],
    )
    return EvaluationBundle(regime_df, accuracy, log_like)
