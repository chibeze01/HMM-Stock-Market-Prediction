from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from .logging_utils import get_logger
from .retraining_pipeline import CycleSummary, RetrainingPipeline

logger = get_logger(__name__)


class RetrainingScheduler:
    def __init__(
        self,
        pipeline: RetrainingPipeline,
        interval_hours: float = 24.0,
    ):
        self._pipeline = pipeline
        self._interval_hours = interval_hours
        self._scheduler = BackgroundScheduler(daemon=True)
        self._last_summary: CycleSummary | None = None

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.add_job(
                self._run,
                "interval",
                hours=self._interval_hours,
                id="retrain",
            )
            self._scheduler.start()
            logger.info(
                "RetrainingScheduler started with interval=%.1fh",
                self._interval_hours,
            )

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("RetrainingScheduler stopped")

    def trigger_now(self) -> CycleSummary:
        summary = self._pipeline.run_cycle()
        self._last_summary = summary
        return summary

    def _run(self) -> None:
        try:
            self._last_summary = self._pipeline.run_cycle()
        except Exception:
            logger.exception("Scheduled retraining cycle failed")

    @property
    def is_running(self) -> bool:
        return self._scheduler.running

    @property
    def last_summary(self) -> CycleSummary | None:
        return self._last_summary
