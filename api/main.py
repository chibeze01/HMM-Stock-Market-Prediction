"""FastAPI application for HMM stock market prediction."""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import FastAPI, HTTPException

from model.evaluation import run_evaluation
from model.hmm import HMMConfig, HMMStockPredictor
from model.utils import (
    PreprocessingConfig,
    fetch_stock_data,
    merge_preprocessed,
    preprocess_data,
)

from .model_registry import ModelEntry, ModelRegistry
from .schemas import (
    FineTuneRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

app = FastAPI(title="HMM Stock Predictor API", version="0.1.0")
registry = ModelRegistry()

# Per-model locks to serialize concurrent train/fine-tune on the same model_id
_model_locks: dict[str, asyncio.Lock] = {}


def _get_model_lock(model_id: str) -> asyncio.Lock:
    if model_id not in _model_locks:
        _model_locks[model_id] = asyncio.Lock()
    return _model_locks[model_id]


def _eval_to_dict(evaluation) -> dict:
    """Convert EvaluationBundle to JSON-serializable dict."""
    regime = evaluation.regime_summary
    return {
        "regime_summary": regime.reset_index().to_dict(orient="records"),
        "rolling_accuracy": evaluation.rolling_accuracy.tolist(),
        "rolling_log_likelihood": evaluation.rolling_log_likelihood.tolist(),
    }


def _training_summary_to_dict(summary) -> dict:
    return {
        "timestamp": summary.timestamp.isoformat(),
        "log_likelihood": float(summary.log_likelihood),
        "n_samples": int(summary.n_samples),
    }


def _describe_state(regime_summary, state_id: int) -> str:
    if regime_summary is None or state_id not in regime_summary.index:
        return f"State {state_id}"
    row = regime_summary.loc[state_id]
    mean_ret = row["mean_return"]
    vol = row["volatility"]
    direction = "bullish" if mean_ret > 0 else "bearish"
    return f"State {state_id}: {direction} (avg return {mean_ret:.2%}, volatility {vol:.2%})"


def _do_train(req: TrainRequest):
    """CPU-bound training work — runs in thread pool."""
    preprocess_cfg = PreprocessingConfig(
        features=tuple(req.features),
        volatility_window=req.volatility_window,
        momentum_window=req.momentum_window,
    )
    model_cfg = HMMConfig(
        n_hidden_states=req.n_hidden_states,
        covariance_type=req.covariance_type,
        n_iter=req.n_iter,
        random_state=req.random_state,
    )
    raw = fetch_stock_data(req.ticker, req.start_date, req.end_date)
    dataset = preprocess_data(raw, preprocess_cfg)

    if dataset.features.shape[0] < model_cfg.n_hidden_states * 3:
        raise ValueError(
            f"Need at least {model_cfg.n_hidden_states * 3} observations. "
            f"Got {dataset.features.shape[0]}."
        )

    predictor = HMMStockPredictor(model_cfg)
    summary = predictor.train(dataset.features)
    evaluation = run_evaluation(predictor, dataset.frame, dataset.features)

    eval_dict = _eval_to_dict(evaluation)
    summary_dict = _training_summary_to_dict(summary)

    entry = ModelEntry(
        model=predictor,
        ticker=req.ticker,
        evaluation=eval_dict,
        dataset=dataset,
        preprocess_config=preprocess_cfg,
    )
    model_id = registry.add(entry)

    return model_id, summary_dict, eval_dict


def _do_fine_tune(model_id: str, req: FineTuneRequest):
    """CPU-bound fine-tune work — runs in thread pool."""
    entry = registry.get(model_id)
    if entry is None:
        return None

    preprocess_cfg = entry.preprocess_config
    dataset = entry.dataset

    new_raw = fetch_stock_data(
        entry.ticker,
        req.new_end_date if req.extend_start_date is None else req.extend_start_date,
        req.new_end_date,
    )
    new_dataset = preprocess_data(new_raw, preprocess_cfg)
    if new_dataset.features.size == 0:
        raise ValueError("No usable new data for fine-tune window.")

    combined = merge_preprocessed(dataset, new_dataset)
    predictor = entry.model
    summary = predictor.fine_tune(combined.features)
    evaluation = run_evaluation(predictor, combined.frame, combined.features)

    eval_dict = _eval_to_dict(evaluation)
    summary_dict = _training_summary_to_dict(summary)

    entry.evaluation = eval_dict
    entry.dataset = combined

    return summary_dict, eval_dict


def _do_predict(model_id: str):
    """CPU-bound prediction — runs in thread pool."""
    entry = registry.get(model_id)
    if entry is None:
        return None

    predictor = entry.model
    dataset = entry.dataset
    predicted_state = predictor.predict_next_day_state(dataset.features)
    proba = predictor.regime_probabilities(dataset.features)[-1]

    # Build regime label from evaluation summary
    eval_data = entry.evaluation
    regime_records = eval_data.get("regime_summary", [])
    regime_label = f"State {predicted_state}"
    for rec in regime_records:
        idx = rec.get("HiddenState", rec.get("index", None))
        if idx == predicted_state:
            mr = rec.get("mean_return", 0)
            vol = rec.get("volatility", 0)
            direction = "bullish" if mr > 0 else "bearish"
            regime_label = (
                f"State {predicted_state}: {direction} (avg return {mr:.2%}, volatility {vol:.2%})"
            )
            break

    return {
        "predicted_state": int(predicted_state),
        "probabilities": [float(p) for p in proba],
        "regime_label": regime_label,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=app.version)


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    loop = asyncio.get_event_loop()
    try:
        model_id, summary_dict, eval_dict = await loop.run_in_executor(
            None, partial(_do_train, req)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrainResponse(
        model_id=model_id,
        training_summary=summary_dict,
        evaluation=eval_dict,
    )


@app.post("/fine-tune", response_model=TrainResponse)
async def fine_tune(req: FineTuneRequest):
    entry = registry.get(req.model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found.")

    lock = _get_model_lock(req.model_id)
    async with lock:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, partial(_do_fine_tune, req.model_id, req))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found.")

    summary_dict, eval_dict = result
    return TrainResponse(
        model_id=req.model_id,
        training_summary=summary_dict,
        evaluation=eval_dict,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(_do_predict, req.model_id))
    if result is None:
        raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found.")
    return PredictResponse(**result)


@app.get("/evaluation/{model_id}")
async def get_evaluation(model_id: str):
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return entry.evaluation


@app.get("/models")
async def list_models():
    result = []
    for mid in registry.list_ids():
        entry = registry.get(mid)
        if entry:
            result.append(
                {
                    "model_id": mid,
                    "ticker": entry.ticker,
                    "created_at": entry.created_at.isoformat(),
                }
            )
    return result


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if not registry.delete(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    _model_locks.pop(model_id, None)
    return {"status": "deleted", "model_id": model_id}
