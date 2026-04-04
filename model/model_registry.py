from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

from .hmm import HMMConfig, HMMStockPredictor, TrainingSummary
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    version: int
    ticker: str
    trained_at: str  # ISO datetime string
    log_likelihood: float
    n_samples: int
    model_path: str  # filename relative to registry_dir


class ModelRegistry:
    def __init__(self, registry_dir: Path | str = "models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.registry_dir / "registry.json"

    def _load_metadata(self) -> list[dict]:
        if not self._metadata_path.exists():
            return []
        with open(self._metadata_path) as fh:
            return json.load(fh)

    def _save_metadata(self, entries: list[dict]) -> None:
        with open(self._metadata_path, "w") as fh:
            json.dump(entries, fh, indent=2)

    def save_version(
        self,
        predictor: HMMStockPredictor,
        ticker: str,
        summary: TrainingSummary,
    ) -> ModelVersion:
        entries = self._load_metadata()
        next_version = len(entries) + 1
        filename = f"model_v{next_version}.pkl"
        model_path = self.registry_dir / filename
        predictor.save(model_path)
        version = ModelVersion(
            version=next_version,
            ticker=ticker,
            trained_at=summary.timestamp.isoformat(),
            log_likelihood=summary.log_likelihood,
            n_samples=summary.n_samples,
            model_path=filename,
        )
        entries.append(asdict(version))
        self._save_metadata(entries)
        logger.info("Saved model version %s to %s", next_version, model_path)
        return version

    def list_versions(self) -> list[ModelVersion]:
        return [ModelVersion(**e) for e in self._load_metadata()]

    def has_any_version(self) -> bool:
        return len(self._load_metadata()) > 0

    def load_version(self, version: int) -> HMMStockPredictor:
        entries = self._load_metadata()
        matched = [e for e in entries if e["version"] == version]
        if not matched:
            raise ValueError(f"Model version {version} not found in registry.")
        path = self.registry_dir / matched[0]["model_path"]
        logger.info("Loading model version %s from %s", version, path)
        return HMMStockPredictor.load(path)

    def load_latest(self) -> HMMStockPredictor:
        entries = self._load_metadata()
        if not entries:
            raise RuntimeError("No model versions in registry.")
        latest = max(entries, key=lambda e: e["version"])
        path = self.registry_dir / latest["model_path"]
        logger.info(
            "Loading latest model version %s from %s", latest["version"], path
        )
        return HMMStockPredictor.load(path)
