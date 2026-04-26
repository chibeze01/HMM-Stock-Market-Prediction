"""Thread-safe in-memory model store with bounded capacity."""

from __future__ import annotations

import datetime as dt
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_MODELS = 50


@dataclass
class ModelEntry:
    model: Any
    evaluation: Any
    ticker: str
    dataset: Any = None
    preprocess_config: Any = None
    created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)


class ModelRegistry:
    """Stores trained models keyed by UUID, evicting oldest when full."""

    def __init__(self, max_models: int = MAX_MODELS):
        self._store: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()
        self._max_models = max_models

    def add(self, entry: ModelEntry) -> str:
        model_id = str(uuid.uuid4())
        with self._lock:
            if len(self._store) >= self._max_models:
                oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
                del self._store[oldest_key]
            self._store[model_id] = entry
        return model_id

    def get(self, model_id: str) -> ModelEntry | None:
        with self._lock:
            return self._store.get(model_id)

    def update(self, model_id: str, **kwargs: Any) -> bool:
        with self._lock:
            entry = self._store.get(model_id)
            if entry is None:
                return False
            for key, value in kwargs.items():
                setattr(entry, key, value)
            return True

    def delete(self, model_id: str) -> bool:
        with self._lock:
            if model_id in self._store:
                del self._store[model_id]
                return True
            return False

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())
