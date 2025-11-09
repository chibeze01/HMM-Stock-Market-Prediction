import datetime as dt
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:  # pragma: no cover - prefer hmmlearn when available
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - fallback for offline environments
    from .simple_hmm import GaussianHMM


@dataclass(frozen=True)
class HMMConfig:
    n_hidden_states: int = 4
    n_iter: int = 500
    covariance_type: str = "diag"
    random_state: Optional[int] = 42

    def __post_init__(self):
        if self.n_hidden_states < 2:
            raise ValueError("n_hidden_states must be >= 2.")
        if self.n_iter < 10:
            raise ValueError("n_iter must be >= 10.")
        if self.covariance_type not in {"full", "diag", "tied", "spherical"}:
            raise ValueError("Invalid covariance_type value.")


@dataclass
class TrainingSummary:
    timestamp: dt.datetime
    log_likelihood: float
    n_samples: int


class HMMStockPredictor:
    """
    A class for a Hidden Markov Model (HMM) based stock market predictor.
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        """
        Initializes the HMMStockPredictor.
        """
        self.config = config or HMMConfig()
        self.model = self._build_model()
        self.training_history: List[TrainingSummary] = []
        self._X_history: Optional[np.ndarray] = None

    def _build_model(self) -> GaussianHMM:
        return GaussianHMM(
            n_components=self.config.n_hidden_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )

    def _ensure_fitted(self) -> None:
        if not hasattr(self.model, "transmat_"):
            raise RuntimeError("Model has not been trained.")

    def train(self, X_train: np.ndarray) -> TrainingSummary:
        """
        Trains the HMM model on the provided training data.
        """
        self.model = self._build_model()
        self.model.fit(X_train)
        self._X_history = np.array(X_train, copy=True)
        summary = TrainingSummary(
            timestamp=dt.datetime.utcnow(),
            log_likelihood=self.model.score(X_train),
            n_samples=X_train.shape[0],
        )
        self.training_history.append(summary)
        return summary

    def fine_tune(self, X_new: np.ndarray, retain_history: bool = True) -> TrainingSummary:
        """
        Fine-tunes the HMM model on new data. Optionally retains earlier observations.
        """
        if retain_history and self._X_history is not None:
            data = np.vstack([self._X_history, X_new])
        else:
            data = X_new
        return self.train(data)

    def predict_next_day_state(self, X: np.ndarray) -> int:
        """
        Predicts the most likely hidden state for the next day.
        """
        self._ensure_fitted()
        hidden_states = self.model.predict(X)
        last_hidden_state = hidden_states[-1]
        most_likely_next_state = np.argmax(self.model.transmat_[last_hidden_state])
        return int(most_likely_next_state)

    def regime_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Returns posterior probabilities for each state over the provided timeline.
        """
        self._ensure_fitted()
        return self.model.predict_proba(X)

    def save(self, path: Union[Path, str]) -> Path:
        """
        Persists the trained model and metadata to disk.
        """
        self._ensure_fitted()
        target = Path(path)
        payload = {
            "config": asdict(self.config),
            "model": self.model,
            "training_history": self.training_history,
        }
        with open(target, "wb") as fh:
            pickle.dump(payload, fh)
        return target

    @classmethod
    def load(cls, path: Path) -> "HMMStockPredictor":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        predictor = cls(HMMConfig(**payload["config"]))
        predictor.model = payload["model"]
        predictor.training_history = payload.get("training_history", [])
        return predictor
