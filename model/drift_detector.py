from __future__ import annotations

import math
from dataclasses import dataclass

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DriftConfig:
    accuracy_floor: float = 0.45
    log_likelihood_floor: float = -5.0
    consecutive_failures: int = 3
    eval_window: int = 20


@dataclass(frozen=True)
class DriftStatus:
    is_drifting: bool
    reason: str
    accuracy: float | None
    log_likelihood: float | None


class DriftDetector:
    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()
        self._failure_streak: int = 0

    def check(self, accuracy: float | None, log_likelihood: float | None) -> DriftStatus:
        failed = False
        reasons: list[str] = []
        if accuracy is not None and accuracy < self.config.accuracy_floor:
            failed = True
            reasons.append(
                f"accuracy={accuracy:.3f} < floor={self.config.accuracy_floor}"
            )
        if log_likelihood is not None and (
            math.isnan(log_likelihood)
            or log_likelihood < self.config.log_likelihood_floor
        ):
            failed = True
            reasons.append(
                f"log_likelihood={log_likelihood:.3f} < floor={self.config.log_likelihood_floor}"
            )
        if failed:
            self._failure_streak += 1
        else:
            self._failure_streak = 0
        is_drifting = self._failure_streak >= self.config.consecutive_failures
        reason = "; ".join(reasons) if reasons else "metrics_healthy"
        logger.debug(
            "DriftDetector check: streak=%s is_drifting=%s reason=%s",
            self._failure_streak,
            is_drifting,
            reason,
        )
        return DriftStatus(
            is_drifting=is_drifting,
            reason=reason,
            accuracy=accuracy,
            log_likelihood=log_likelihood,
        )

    def reset(self) -> None:
        self._failure_streak = 0

    @property
    def failure_streak(self) -> int:
        return self._failure_streak
