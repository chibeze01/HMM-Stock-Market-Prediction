from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .hmm import HMMStockPredictor


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
    return summary


def _infer_direction_map(frame: pd.DataFrame, hidden_states: np.ndarray) -> Dict[int, int]:
    grouped = (
        frame.assign(HiddenState=hidden_states.flatten())
        .groupby("HiddenState")["Returns"]
        .mean()
    )
    return grouped.apply(lambda v: 1 if v >= 0 else -1).to_dict()


def rolling_directional_accuracy(
    frame: pd.DataFrame,
    hidden_states: np.ndarray,
    window: int = 20,
    direction_map: Optional[Dict[int, int]] = None,
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
    predicted_direction = np.vectorize(direction_map.get)(hidden_states.flatten())
    realized_direction = np.sign(frame["Returns"])
    accuracy = (predicted_direction == np.sign(realized_direction)).astype(int)
    return (
        pd.Series(accuracy, index=frame.index)
        .rolling(window=window)
        .mean()
        .dropna()
    )


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
    return (
        pd.Series(per_sample, index=index)
        .rolling(window=window)
        .mean()
        .dropna()
    )


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
    return EvaluationBundle(regime_df, accuracy, log_like)
