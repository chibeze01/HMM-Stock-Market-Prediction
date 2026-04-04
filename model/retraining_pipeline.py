from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .drift_detector import DriftConfig, DriftDetector, DriftStatus
from .evaluation import run_evaluation
from .hmm import HMMConfig, HMMStockPredictor
from .logging_utils import get_logger
from .model_registry import ModelRegistry, ModelVersion
from .utils import PreprocessingConfig, fetch_stock_data, preprocess_data

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrainingConfig:
    ticker: str = "GOOGL"
    lookback_days: int = 365
    hmm_config: HMMConfig = field(default_factory=HMMConfig)
    preprocess_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    registry_dir: str = "models"


@dataclass
class CycleSummary:
    ran_at: dt.datetime
    retrained: bool
    reason: str
    drift_status: DriftStatus
    model_version: int | None


class RetrainingPipeline:
    def __init__(
        self,
        config: RetrainingConfig | None = None,
        _fetch_fn=None,
    ):
        self.config = config or RetrainingConfig()
        self.registry = ModelRegistry(self.config.registry_dir)
        self.drift_detector = DriftDetector(self.config.drift_config)
        self._fetch_fn = _fetch_fn or fetch_stock_data
        self.logger = get_logger(self.__class__.__name__)

    def run_cycle(self) -> CycleSummary:
        end = dt.date.today()
        start = end - dt.timedelta(days=self.config.lookback_days)
        self.logger.info(
            "Running retraining cycle for %s (%s → %s)",
            self.config.ticker,
            start,
            end,
        )

        data = self._fetch_fn(
            self.config.ticker, start, end, force_refresh=True
        )
        preprocessed = preprocess_data(data, self.config.preprocess_config)

        if not self.registry.has_any_version():
            self.logger.info("No existing model — performing initial training")
            predictor = HMMStockPredictor(self.config.hmm_config)
            summary = predictor.train(preprocessed.features)
            version = self.registry.save_version(
                predictor, self.config.ticker, summary
            )
            self.drift_detector.reset()
            return CycleSummary(
                ran_at=dt.datetime.utcnow(),
                retrained=True,
                reason="initial_training",
                drift_status=DriftStatus(False, "initial_training", None, None),
                model_version=version.version,
            )

        predictor = self.registry.load_latest()
        bundle = run_evaluation(
            predictor,
            preprocessed.frame,
            preprocessed.features,
            window=self.config.drift_config.eval_window,
        )
        last_accuracy = (
            bundle.rolling_accuracy.iloc[-1]
            if not bundle.rolling_accuracy.empty
            else None
        )
        last_log_lik = (
            bundle.rolling_log_likelihood.iloc[-1]
            if not bundle.rolling_log_likelihood.empty
            else None
        )
        drift_status = self.drift_detector.check(last_accuracy, last_log_lik)
        self.logger.info(
            "Drift check: is_drifting=%s streak=%s reason=%s",
            drift_status.is_drifting,
            self.drift_detector.failure_streak,
            drift_status.reason,
        )

        if drift_status.is_drifting:
            self.logger.info("Drift detected — retraining model")
            predictor = HMMStockPredictor(self.config.hmm_config)
            summary = predictor.train(preprocessed.features)
            version = self.registry.save_version(
                predictor, self.config.ticker, summary
            )
            self.drift_detector.reset()
            return CycleSummary(
                ran_at=dt.datetime.utcnow(),
                retrained=True,
                reason=drift_status.reason,
                drift_status=drift_status,
                model_version=version.version,
            )

        return CycleSummary(
            ran_at=dt.datetime.utcnow(),
            retrained=False,
            reason="no_drift",
            drift_status=drift_status,
            model_version=None,
        )
