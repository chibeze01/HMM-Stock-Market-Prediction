import datetime as dt
import logging
import os
import sys

import httpx
import pandas as pd
import streamlit as st

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backtesting import (
    BacktestConfig,
    BacktestEngine,
)
from model.evaluation import EvaluationBundle, run_evaluation
from model.hmm import HMMConfig, HMMStockPredictor, TrainingSummary
from model.logging_utils import LOG_FILE, configure_logging

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

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
        "model_id": None,
        "evaluation": None,
        "training_summary": None,
        "training_window": None,
        "ticker": None,
        "run_history": [],
        "last_prediction": None,
        "log_messages": [],
        "_streamlit_log_attached": False,
        "success_message": None,
        "backtest_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_app() -> None:
    st.session_state.clear()
    logger.info("Session reset triggered by user.")
    st.experimental_rerun()


def _api_post(path: str, payload: dict) -> httpx.Response:
    return httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=120.0)


def _api_get(path: str) -> httpx.Response:
    return httpx.get(f"{API_BASE_URL}{path}", timeout=30.0)


def _api_delete(path: str) -> httpx.Response:
    return httpx.delete(f"{API_BASE_URL}{path}", timeout=30.0)


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
    help=(
        "Select the data inputs (features) that the Hidden Markov Model "
        "will use to learn market regimes."
    ),
)
selected_features = [feature_labels[label] for label in selected_feature_labels]

vol_window = st.sidebar.slider("Volatility Window", min_value=3, max_value=30, value=5)
mom_window = st.sidebar.slider("Momentum Window", min_value=3, max_value=30, value=3)

st.sidebar.header("Model Parameters")
hidden_states = st.sidebar.slider(
    "Hidden States",
    min_value=2,
    max_value=8,
    value=4,
    help=(
        "Choose how many latent regimes the HMM should learn (higher values "
        "capture more nuanced behaviors but need more data)."
    ),
)
covariance_type = st.sidebar.selectbox(
    "Covariance Type", options=["diag", "full", "spherical", "tied"], index=0
)
n_iter = st.sidebar.slider(
    "Training Iterations", min_value=100, max_value=2000, value=500, step=50
)
random_state = st.sidebar.number_input("Random Seed", value=42)

st.sidebar.divider()
reset_clicked = st.sidebar.button(  # noqa: F841
    "Reset Session", use_container_width=True, on_click=reset_app
)

train_can_click = len(selected_features) > 0
train_clicked = st.sidebar.button(
    "Train / Re-train Model",
    use_container_width=True,
    type="primary",
    disabled=not train_can_click,
    help=(
        "Select at least one feature to train the model."
        if not train_can_click
        else "Train the model with the selected configuration."
    ),
)

st.sidebar.header("Fine-Tune")
fine_tune_end_date = st.sidebar.date_input("Extend data up to", dt.date.today())
can_fine_tune = st.session_state["model_id"] is not None
fine_tune_clicked = st.sidebar.button(
    "Fine-Tune with Recent Data",
    use_container_width=True,
    disabled=not can_fine_tune,
    help=(
        "Train a model first before fine-tuning."
        if not can_fine_tune
        else "Fine-tune the model with recent data."
    ),
)

if st.session_state.get("success_message"):
    st.sidebar.success(st.session_state["success_message"])
    st.session_state["success_message"] = None

# ── Train ────────────────────────────────────────────────────────────────────────

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
        with st.spinner("Training model via API..."):
            resp = _api_post(
                "/train",
                {
                    "ticker": ticker,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "features": selected_features,
                    "n_hidden_states": hidden_states,
                    "covariance_type": covariance_type,
                    "n_iter": n_iter,
                    "random_state": int(random_state),
                    "volatility_window": vol_window,
                    "momentum_window": mom_window,
                },
            )
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"API error ({resp.status_code}): {detail}")

        body = resp.json()
        logger.info(
            "Training succeeded for %s (%s -> %s) states=%s features=%s",
            ticker,
            start_date,
            end_date,
            hidden_states,
            selected_features,
        )
        st.session_state.update(
            {
                "model_id": body["model_id"],
                "evaluation": body["evaluation"],
                "training_summary": body["training_summary"],
                "training_window": (start_date, end_date),
                "ticker": ticker,
                "run_history": [
                    {
                        "type": "train",
                        "summary": body["training_summary"],
                        "window": (str(start_date), str(end_date)),
                    }
                ],
                "last_prediction": None,
                "success_message": "Model trained successfully.",
            }
        )
        st.rerun()
    except httpx.ConnectError:
        st.sidebar.error(
            f"Cannot connect to API at {API_BASE_URL}. Is the server running?"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        st.sidebar.error(f"Training failed: {exc}")

# ── Fine-Tune ────────────────────────────────────────────────────────────────────

if fine_tune_clicked:
    try:
        with st.spinner("Fine-tuning model via API..."):
            resp = _api_post(
                "/fine-tune",
                {
                    "model_id": st.session_state["model_id"],
                    "new_end_date": str(fine_tune_end_date),
                },
            )
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"API error ({resp.status_code}): {detail}")

        body = resp.json()
        logger.info("Fine-tuned model for %s up to %s", ticker, fine_tune_end_date)
        st.session_state.update(
            {
                "evaluation": body["evaluation"],
                "training_summary": body["training_summary"],
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
                "summary": body["training_summary"],
                "window": (
                    str(st.session_state["training_window"][0]),
                    str(fine_tune_end_date),
                ),
            }
        )
        st.session_state["success_message"] = "Fine-tuning complete."
        st.rerun()
    except httpx.ConnectError:
        st.sidebar.error(
            f"Cannot connect to API at {API_BASE_URL}. Is the server running?"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fine-tuning failed: %s", exc)
        st.sidebar.error(f"Fine-tuning failed: {exc}")

# ── Main layout ──────────────────────────────────────────────────────────────────

if st.session_state["model_id"] is None:
    st.info(
        "**Getting Started**\n"
        "1. **Select a Ticker:** Enter a stock ticker in the sidebar.\n"
        "2. **Choose Features:** Select the data inputs (e.g., Returns, Volatility).\n"
        "3. **Train Model:** Click 'Train / Re-train Model' to learn market regimes.",
        icon="👈",
    )
else:
    summary = st.session_state["training_summary"]
    evaluation = st.session_state["evaluation"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", st.session_state["ticker"])
    col2.metric("Log Likelihood", f"{summary['log_likelihood']:.2f}")
    col3.metric("Samples", summary["n_samples"])

    st.subheader("Training & Fine-Tune History")
    history_rows = []
    for record in st.session_state["run_history"]:
        s = record["summary"]
        row = {
            "Type": record["type"],
            "Window": f"{record['window'][0]} -> {record['window'][1]}",
            "Timestamp (UTC)": s.get("timestamp", ""),
            "Log Likelihood": round(s.get("log_likelihood", 0), 2),
            "Samples": s.get("n_samples", 0),
        }
        history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, height=180)

    st.subheader("Evaluation")
    eval_tabs = st.tabs(["Regime Summary", "Rolling Accuracy", "Log Likelihood"])
    with eval_tabs[0]:
        regime_data = evaluation.get("regime_summary", [])
        if regime_data:
            regime_df = pd.DataFrame(regime_data)
            st.dataframe(regime_df, use_container_width=True)
        else:
            st.info(
                "**No Regime Summary Available**\n\n"
                "The model hasn't generated a regime summary yet. Ensure the model has "
                "been trained successfully with sufficient data.",
                icon="📊",
            )
    with eval_tabs[1]:
        if evaluation.rolling_accuracy.empty:
            st.info(
                "**Not Enough Data for Rolling Accuracy**\n\n"
                "The rolling accuracy series will appear once enough data is available.\n"
                "Try expanding your training date range to generate this metric.",
                icon="📈",
            )
        else:
            st.info("Rolling accuracy series will appear once enough data is available.")
    with eval_tabs[2]:
        if evaluation.rolling_log_likelihood.empty:
            st.info(
                "**Not Enough Data for Log Likelihood**\n\n"
                "Rolling log-likelihood requires additional observations to compute.\n"
                "Try expanding your training date range to generate this metric.",
                icon="📉",
            )
        else:
            st.info("Rolling log-likelihood requires additional observations.")

    st.subheader("Prediction")
    if st.button("Predict Next Regime", type="primary"):
        try:
            resp = _api_post("/predict", {"model_id": st.session_state["model_id"]})
            if resp.status_code != 200:
                detail = resp.json().get("detail", resp.text)
                raise RuntimeError(f"API error ({resp.status_code}): {detail}")
            pred = resp.json()
            st.session_state["last_prediction"] = pred
            logger.info("Generated prediction state=%s", pred["predicted_state"])
        except httpx.ConnectError:
            st.error(
                f"Cannot connect to API at {API_BASE_URL}. Is the server running?"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed: %s", exc)
            st.error(f"Prediction failed: {exc}")

    if st.session_state["last_prediction"]:
        pred = st.session_state["last_prediction"]
        st.success(f"Predicted next regime: {pred['regime_label']}")
        prob_df = pd.DataFrame(
            {
                "State": list(range(len(pred["probabilities"]))),
                "Probability": pred["probabilities"],
            }
        ).set_index("State")
        st.bar_chart(prob_df)
    else:
        st.info(
            "**No Prediction Available**\n\n"
            "Click **Predict Next Regime** above to forecast the current market state "
            "based on the latest data.",
            icon="🔮",
        )

    st.subheader("Backtesting")
    with st.expander("Backtest Settings", expanded=False):
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            bt_commission = st.number_input(
                "Commission (per side)", value=0.001, min_value=0.0,
                max_value=0.1, step=0.0005, format="%.4f",
            )
            bt_slippage = st.number_input(
                "Slippage (per side)", value=0.0005, min_value=0.0,
                max_value=0.1, step=0.0005, format="%.4f",
            )
        with bt_col2:
            bt_position_size = st.slider(
                "Position Size", min_value=0.1, max_value=1.0, value=1.0, step=0.1,
            )
            bt_initial_capital = st.number_input(
                "Initial Capital ($)", value=10_000.0, min_value=100.0, step=1000.0,
            )
    if st.button("Run Backtest", type="primary"):
        try:
            bt_cfg = BacktestConfig(
                initial_capital=bt_initial_capital,
                position_size=bt_position_size,
                commission=bt_commission,
                slippage=bt_slippage,
            )
            engine = BacktestEngine(bt_cfg)
            hidden_states = st.session_state["model"].model.predict(dataset.features)
            with st.spinner("Running backtest..."):
                result = engine.run(dataset.frame, hidden_states, evaluation.regime_summary)
            st.session_state["backtest_result"] = result
            logger.info(
                "Backtest complete: trades=%d sharpe=%.2f return=%.2f%%",
                result.n_trades, result.sharpe_ratio, result.total_return * 100,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backtest failed: %s", exc)
            st.error(f"Backtest failed: {exc}")

    if st.session_state["backtest_result"] is not None:
        result = st.session_state["backtest_result"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{result.total_return:.2%}")
        m2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        m3.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
        m4.metric("Win Rate", f"{result.win_rate:.1%}" if result.n_trades else "N/A")
        m5.metric("Trades", result.n_trades)

        r1, r2 = st.columns(2)
        r1.metric("Annualized Return", f"{result.annualized_return:.2%}")
        r2.metric("Benchmark (Buy & Hold)", f"{result.benchmark_return:.2%}")

        st.line_chart(result.equity_curve, use_container_width=True)

        if result.trades:
            trade_rows = [
                {
                    "Entry": t.entry_date.strftime("%Y-%m-%d"),
                    "Exit": t.exit_date.strftime("%Y-%m-%d"),
                    "Entry $": round(t.entry_price, 2),
                    "Exit $": round(t.exit_price, 2),
                    "PnL": round(t.pnl, 2),
                }
                for t in result.trades
            ]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)
    else:
        st.info(
            "**Backtest Not Run**\n\n"
            "Configure your settings above and click **Run Backtest** to simulate "
            "trading performance using the trained model.",
            icon="📊",
        )

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
