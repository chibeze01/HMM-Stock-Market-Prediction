"""Integration tests for the FastAPI endpoints using Starlette TestClient."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.testclient import TestClient


def _make_mock_stock_data(rows: int = 100) -> pd.DataFrame:
    """Create minimal stock DataFrame that preprocess_data expects."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    return pd.DataFrame({"Close": close}, index=dates)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the model registry between tests to avoid state leaks."""
    from api.main import registry

    registry._store.clear()
    yield


@pytest.fixture
def mock_fetch():
    """Patch fetch_stock_data to return synthetic data (no network)."""
    mock_data = _make_mock_stock_data(200)
    with patch("api.main.fetch_stock_data", return_value=mock_data) as m:
        yield m


@pytest.fixture
def client(mock_fetch):
    """Sync test client wired to the FastAPI ASGI app."""
    from api.main import app

    return TestClient(app)


def _train_model(client) -> str:
    """Helper: train a model and return its model_id."""
    resp = client.post(
        "/train",
        json={
            "ticker": "TEST",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "features": ["returns", "volatility"],
            "n_hidden_states": 3,
            "n_iter": 50,
            "random_state": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["model_id"]


# ── Health ─────────────────────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ── Train → Predict roundtrip ─────────────────────────────────────────────────


def test_train_returns_model_id(client):
    model_id = _train_model(client)
    assert isinstance(model_id, str)
    assert len(model_id) > 0


def test_train_response_has_evaluation(client):
    resp = client.post(
        "/train",
        json={
            "ticker": "TEST",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "features": ["returns", "volatility"],
            "n_hidden_states": 3,
            "n_iter": 50,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "training_summary" in body
    assert "evaluation" in body


def test_predict_after_train(client):
    model_id = _train_model(client)
    resp = client.post("/predict", json={"model_id": model_id})
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_state" in body
    assert "probabilities" in body
    assert isinstance(body["probabilities"], list)
    assert "regime_label" in body


# ── Fine-tune ─────────────────────────────────────────────────────────────────


def test_fine_tune_after_train(client):
    model_id = _train_model(client)
    resp = client.post(
        "/fine-tune",
        json={
            "model_id": model_id,
            "new_end_date": "2024-06-01",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == model_id
    assert "training_summary" in body


# ── Error cases ───────────────────────────────────────────────────────────────


def test_predict_unknown_model_returns_404(client):
    resp = client.post("/predict", json={"model_id": "nonexistent-id"})
    assert resp.status_code == 404


def test_fine_tune_unknown_model_returns_404(client):
    resp = client.post(
        "/fine-tune",
        json={
            "model_id": "nonexistent-id",
            "new_end_date": "2024-06-01",
        },
    )
    assert resp.status_code == 404


def test_train_validation_error(client):
    resp = client.post(
        "/train",
        json={
            "ticker": "",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        },
    )
    assert resp.status_code == 422


# ── Models listing & deletion ─────────────────────────────────────────────────


def test_list_models(client):
    model_id = _train_model(client)
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert model_id in [m["model_id"] for m in body]


def test_delete_model(client):
    model_id = _train_model(client)
    resp = client.delete(f"/models/{model_id}")
    assert resp.status_code == 200
    # Predict should now 404
    resp = client.post("/predict", json={"model_id": model_id})
    assert resp.status_code == 404


def test_delete_nonexistent_model_returns_404(client):
    resp = client.delete("/models/nonexistent-id")
    assert resp.status_code == 404


# ── Evaluation endpoint ──────────────────────────────────────────────────────


def test_evaluation_endpoint(client):
    model_id = _train_model(client)
    resp = client.get(f"/evaluation/{model_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "regime_summary" in body
    assert "rolling_accuracy" in body
    assert "rolling_log_likelihood" in body


def test_evaluation_unknown_model_returns_404(client):
    resp = client.get("/evaluation/nonexistent-id")
    assert resp.status_code == 404
