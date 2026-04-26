"""Pydantic v2 request/response models for the HMM prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    start_date: str
    end_date: str
    features: list[str] = ["returns", "volatility"]
    n_hidden_states: int = Field(default=4, ge=2)
    covariance_type: str = "diag"
    n_iter: int = Field(default=500, ge=10)
    random_state: int | None = 42
    volatility_window: int = Field(default=5, ge=2)
    momentum_window: int = Field(default=3, ge=2)


class FineTuneRequest(BaseModel):
    model_id: str
    new_end_date: str
    extend_start_date: str | None = None


class PredictRequest(BaseModel):
    model_id: str


class TrainResponse(BaseModel):
    model_id: str
    training_summary: dict
    evaluation: dict


class PredictResponse(BaseModel):
    predicted_state: int
    probabilities: list[float]
    regime_label: str


class HealthResponse(BaseModel):
    status: str
    version: str
