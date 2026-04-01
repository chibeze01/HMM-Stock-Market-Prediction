import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.evaluation import EvaluationBundle, run_evaluation
from model.hmm import HMMConfig, HMMStockPredictor, TrainingSummary
from model.logging_utils import LOG_FILE, configure_logging
from model.utils import (
    PreprocessedData,
    PreprocessingConfig,
    fetch_stock_data,
    merge_preprocessed,
    preprocess_data,
)

st.set_page_config(page_title="HMM Stock Predictor", layout="wide")
configure_logging()
logger = logging.getLogger(__name__)


class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        logs = st.session_state.get("log_messages", [])
        logs.append(msg)
        st.session_state["log_messages"] = logs[-500:]


def initialize_session_state() -> None:
    defaults = {
        "model": None,
        "preprocessed": None,
        "evaluation": None,
        "training_summary": None,
        "training_window": None,
        "preprocess_config": None,
        "model_config": None,
        "ticker": None,
        "run_history": [],
        "last_prediction": None,
        "log_messages": [],
        "_streamlit_log_attached": False,
        "success_message": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_app() -> None:
    st.session_state.clear()
    logger.info("Session reset triggered by user.")
    st.experimental_rerun()


def describe_state(summary: pd.DataFrame, state_id: int) -> str:
    if summary is None or state_id not in summary.index:
        return f"State {state_id}"
    mean_return = summary.loc[state_id, "mean_return"]
    volatility = summary.loc[state_id, "volatility"]
    direction = "bullish" if mean_return > 0 else "bearish"
    return (
        f"State {state_id}: {direction} (avg return {mean_return:.2%}, volatility {volatility:.2%})"
    )


def ensure_enough_observations(observations: np.ndarray, config: HMMConfig) -> None:
    if observations.shape[0] < config.n_hidden_states * 3:
        raise ValueError(
            f"Need at least {config.n_hidden_states * 3} observations "
            "before training. Try expanding the date range."
        )


initialize_session_state()

if not st.session_state["_streamlit_log_attached"]:
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    st.session_state["_streamlit_log_attached"] = True

st.title("HMM Stock Market Predictor")
st.caption("End-to-end workflow for training, fine-tuning, and interpreting a regime model.")

# Sidebar controls ------------------------------------------------------------------
st.sidebar.header("Data Configuration")
ticker = st.sidebar.text_input("Stock Ticker", st.session_state["ticker"] or "GOOGL").upper()
start_date = st.sidebar.date_input(
    "Training Start Date",
    (
        st.session_state["training_window"][0]
        if st.session_state["training_window"]
        else dt.date(2020, 1, 1)
    ),
)
end_date = st.sidebar.date_input(
    "Training End Date",
    (
        st.session_state["training_window"][1]
        if st.session_state["training_window"]
        else dt.date(2023, 12, 31)
    ),
)

feature_labels = {
    "Daily Returns": "returns",
    "Rolling Volatility": "volatility",
    "Momentum": "momentum",
}
selected_feature_labels = st.sidebar.multiselect(
    "Features for Observations",
    options=list(feature_labels.keys()),
    default=["Daily Returns", "Rolling Volatility"],
)
selected_features = tuple(feature_labels[label] for label in selected_feature_labels)

vol_window = st.sidebar.slider("Volatility Window", min_value=3, max_value=30, value=5)
mom_window = st.sidebar.slider("Momentum Window", min_value=3, max_value=30, value=3)

st.sidebar.header("Model Parameters")
hidden_states = st.sidebar.slider(
    "Hidden States",
    min_value=2,
    max_value=8,
    value=4,
    help=(
        "Choose how many latent regimes the HMM should learn "
        "(higher values capture more nuanced behaviors but need more data)."
    ),
)
covariance_type = st.sidebar.selectbox(
    "Covariance Type", options=["diag", "full", "spherical", "tied"], index=0
)
n_iter = st.sidebar.slider("Training Iterations", min_value=100, max_value=2000, value=500, step=50)
random_state = st.sidebar.number_input("Random Seed", value=42)

st.sidebar.divider()
reset_clicked = st.sidebar.button("Reset Session", use_container_width=True, on_click=reset_app)

train_can_click = len(selected_features) > 0
train_clicked = st.sidebar.button(
    "Train / Re-train Model",
    use_container_width=True,
    type="primary",
    disabled=not train_can_click,
    help="Select at least one feature to train the model." if not train_can_click else None,
)

st.sidebar.header("Fine-Tune")
fine_tune_end_date = st.sidebar.date_input("Extend data up to", dt.date.today())
can_fine_tune = (
    st.session_state["model"] is not None and st.session_state["preprocessed"] is not None
)
fine_tune_clicked = st.sidebar.button(
    "Fine-Tune with Recent Data",
    use_container_width=True,
    disabled=not can_fine_tune,
    help="Train a model first before fine-tuning." if not can_fine_tune else None,
)

if st.session_state.get("success_message"):
    st.sidebar.success(st.session_state["success_message"])
    st.session_state["success_message"] = None

def train_pipeline(
    ticker_symbol: str,
    start: dt.date,
    end: dt.date,
    preprocess_cfg: PreprocessingConfig,
    model_cfg: HMMConfig,
) -> tuple[HMMStockPredictor, PreprocessedData, EvaluationBundle, TrainingSummary]:
    raw = fetch_stock_data(ticker_symbol, start, end)
    dataset = preprocess_data(raw, preprocess_cfg)
    ensure_enough_observations(dataset.features, model_cfg)
    predictor = HMMStockPredictor(model_cfg)
    summary = predictor.train(dataset.features)
    evaluation = run_evaluation(predictor, dataset.frame, dataset.features)
    return predictor, dataset, evaluation, summary


if train_clicked:
    try:
        preprocess_cfg = PreprocessingConfig(
            features=selected_features,
            volatility_window=vol_window,
            momentum_window=mom_window,
        )
        model_cfg = HMMConfig(
            n_hidden_states=hidden_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=int(random_state),
        )
        with st.spinner("Training model..."):
            model, dataset, evaluation, summary = train_pipeline(
                ticker, start_date, end_date, preprocess_cfg, model_cfg
            )
        logger.info(
            "Training succeeded for %s (%s → %s) states=%s features=%s",
            ticker,
            start_date,
            end_date,
            hidden_states,
            selected_features,
        )
        st.session_state.update(
            {
                "model": model,
                "preprocessed": dataset,
                "evaluation": evaluation,
                "training_summary": summary,
                "training_window": (start_date, end_date),
                "preprocess_config": preprocess_cfg,
                "model_config": model_cfg,
                "ticker": ticker,
                "run_history": [
                    {
                        "type": "train",
                        "summary": summary,
                        "window": (start_date, end_date),
                    }
                ],
                "last_prediction": None,
            }
        )
        st.sidebar.success("Model trained successfully.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        st.sidebar.error(f"Training failed: {exc}")


if fine_tune_clicked:
    current_end = st.session_state["training_window"][1]
    fine_tune_start = current_end + dt.timedelta(days=1)
    if fine_tune_end_date <= fine_tune_start:
        st.sidebar.warning("Choose an end date after the latest trained date.")
    else:
        try:
            preprocess_cfg = st.session_state["preprocess_config"]
            with st.spinner("Fine-tuning with latest data..."):
                new_raw = fetch_stock_data(ticker, fine_tune_start, fine_tune_end_date)
                new_dataset = preprocess_data(new_raw, preprocess_cfg)
                if new_dataset.features.size == 0:
                    raise ValueError("No usable new data was returned for this window.")
                combined = merge_preprocessed(st.session_state["preprocessed"], new_dataset)
                summary = st.session_state["model"].fine_tune(combined.features)
                evaluation = run_evaluation(
                    st.session_state["model"], combined.frame, combined.features
                )
            logger.info(
                "Fine-tuned model for %s adding window %s → %s",
                ticker,
                fine_tune_start,
                fine_tune_end_date,
            )
            st.session_state.update(
                {
                    "preprocessed": combined,
                    "evaluation": evaluation,
                    "training_summary": summary,
                    "training_window": (start_date, end_date),
                    "preprocess_config": preprocess_cfg,
                    "model_config": model_cfg,
                    "ticker": ticker,
                    "run_history": [
                        {
                            "type": "train",
                            "summary": summary,
                            "window": (start_date, end_date),
                        }
                    ],
                    "last_prediction": None,
                    "success_message": "Model trained successfully.",
                }
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed: %s", exc)
            st.sidebar.error(f"Training failed: {exc}")


if fine_tune_clicked:
    current_end = st.session_state["training_window"][1]
    fine_tune_start = current_end + dt.timedelta(days=1)
    if fine_tune_end_date <= fine_tune_start:
        st.sidebar.warning("Choose an end date after the latest trained date.")
    else:
        current_end = st.session_state["training_window"][1]
        fine_tune_start = current_end + dt.timedelta(days=1)
        if fine_tune_end_date <= fine_tune_start:
            st.sidebar.warning("Choose an end date after the latest trained date.")
        else:
            try:
                preprocess_cfg = st.session_state["preprocess_config"]
                with st.spinner("Fine-tuning with latest data..."):
                    new_raw = fetch_stock_data(ticker, fine_tune_start, fine_tune_end_date)
                    new_dataset = preprocess_data(new_raw, preprocess_cfg)
                    if new_dataset.features.size == 0:
                        raise ValueError("No usable new data was returned for this window.")
                    combined = merge_preprocessed(st.session_state["preprocessed"], new_dataset)
                    summary = st.session_state["model"].fine_tune(combined.features)
                    evaluation = run_evaluation(
                        st.session_state["model"], combined.frame, combined.features
                    )
                logger.info(
                    "Fine-tuned model for %s adding window %s → %s",
                    ticker,
                    fine_tune_start,
                    fine_tune_end_date,
                )
                st.session_state.update(
                    {
                        "preprocessed": combined,
                        "evaluation": evaluation,
                        "training_summary": summary,
                        "training_window": (
                            st.session_state["training_window"][0],
                            fine_tune_end_date,
                        ),
                    }
                )
                history = st.session_state["run_history"]
                history.append(
                    {
                        "type": "fine-tune",
                        "summary": summary,
                        "window": (fine_tune_start, fine_tune_end_date),
                    }
                )
            logger.info(
                "Fine-tuned model for %s adding window %s → %s",
                ticker,
                fine_tune_start,
                fine_tune_end_date,
            )
            st.session_state.update(
                {
                    "preprocessed": combined,
                    "evaluation": evaluation,
                    "training_summary": summary,
                    "training_window": (
                        st.session_state["training_window"][0],
                        fine_tune_end_date,
                    ),
                        "success_message": "Fine-tuning complete.",
                }
            )
            history = st.session_state["run_history"]
            history.append(
                {
                    "type": "fine-tune",
                    "summary": summary,
                    "window": (fine_tune_start, fine_tune_end_date),
                }
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fine-tuning failed: %s", exc)
            st.sidebar.error(f"Fine-tuning failed: {exc}")


# Main layout -----------------------------------------------------------------------
if st.session_state["model"] is None:
    st.info("Train the model using the controls on the left to unlock evaluation and predictions.")
else:
    summary = st.session_state["training_summary"]
    dataset = st.session_state["preprocessed"]
    evaluation = st.session_state["evaluation"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", st.session_state["ticker"])
    col2.metric("Observations", dataset.features.shape[0])
    col3.metric("Hidden States", st.session_state["model_config"].n_hidden_states)
    col4.metric("Log Likelihood", f"{summary.log_likelihood:.2f}")

    st.subheader("Training & Fine-Tune History")
    history_rows = []
    for record in st.session_state["run_history"]:
        row = {
            "Type": record["type"],
            "Window": f"{record['window'][0]} → {record['window'][1]}",
            "Timestamp (UTC)": record["summary"].timestamp.strftime("%Y-%m-%d %H:%M"),
            "Log Likelihood": round(record["summary"].log_likelihood, 2),
            "Samples": record["summary"].n_samples,
        }
        history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, height=180)

    st.subheader("Evaluation")
    eval_tabs = st.tabs(["Regime Summary", "Rolling Accuracy", "Log Likelihood"])
    with eval_tabs[0]:
        st.dataframe(
            evaluation.regime_summary.style.format(
                {"mean_return": "{:.2%}", "volatility": "{:.2%}", "avg_close": "{:.2f}"}
            ),
            use_container_width=True,
        )
    with eval_tabs[1]:
        if evaluation.rolling_accuracy.empty:
            st.info("Rolling accuracy series will appear once enough data is available.")
        else:
            st.line_chart(evaluation.rolling_accuracy)
    with eval_tabs[2]:
        if evaluation.rolling_log_likelihood.empty:
            st.info("Rolling log-likelihood requires additional observations.")
        else:
            st.line_chart(evaluation.rolling_log_likelihood)

    st.subheader("Prediction")
    if st.button("Predict Next Regime", type="primary"):
        try:
            predicted_state = st.session_state["model"].predict_next_day_state(dataset.features)
            proba = st.session_state["model"].regime_probabilities(dataset.features)[-1]
            message = describe_state(evaluation.regime_summary, predicted_state)
            st.session_state["last_prediction"] = {
                "state": predicted_state,
                "probabilities": proba,
                "message": message,
            }
            logger.info(
                "Generated prediction state=%s probability=%.2f", predicted_state, proba.max()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed: %s", exc)
            st.error(f"Prediction failed: {exc}")

    if st.session_state["last_prediction"]:
        pred = st.session_state["last_prediction"]
        st.success(f"Predicted next regime: {pred['message']}")
        prob_df = pd.DataFrame(
            {"State": list(range(len(pred["probabilities"]))), "Probability": pred["probabilities"]}
        ).set_index("State")
        st.bar_chart(prob_df)

    st.subheader("Recent Data")
    st.dataframe(dataset.frame.tail(10), use_container_width=True)
    st.line_chart(dataset.frame["Close"])

st.subheader("Debug Logs")
logs_display = st.session_state.get("log_messages", [])
if logs_display:
    st.code("\n".join(logs_display[-200:]), language="text")
else:
    st.caption("Logs will appear here after actions are taken.")
st.caption(f"Full log file written to: {LOG_FILE}")
