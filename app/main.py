import streamlit as st
import datetime
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import fetch_stock_data, preprocess_data
from model.hmm import HMMStockPredictor

# --- App Configuration ---
st.set_page_config(page_title="HMM Stock Predictor", layout="wide")

# --- App State Initialization ---
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'states' not in st.session_state:
    st.session_state['states'] = None

# --- UI Components ---
st.title("HMM Stock Market Predictor")

st.sidebar.header("Model Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "GOOGL")
start_date = st.sidebar.date_input("Training Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("Training End Date", datetime.date(2023, 12, 31))

if st.sidebar.button("Train Initial Model"):
    with st.spinner("Fetching data and training model..."):
        # Fetch and process data
        data = fetch_stock_data(ticker, start_date, end_date)
        processed_data, states = preprocess_data(data)

        st.session_state['data'] = processed_data
        st.session_state['states'] = states

        # Train HMM model
        model = HMMStockPredictor()
        model.train(states)
        st.session_state['model'] = model

        st.sidebar.success("Model trained successfully!")

st.header("Model Status")
if st.session_state['model'] is None:
    st.info("The model has not been trained yet. Please train the model using the sidebar.")
else:
    st.success("Model is trained and ready.")

    # --- Fine-Tuning Section ---
    st.sidebar.header("Fine-Tune Model")
    fine_tune_end_date = st.sidebar.date_input("Fine-Tune with Data Up To", datetime.date.today())

    if st.sidebar.button("Fine-Tune on Recent Data"):
        if st.session_state['model'] is None:
            st.sidebar.error("Train an initial model first.")
        else:
            with st.spinner("Fetching recent data and fine-tuning model..."):
                # Fetch recent data
                recent_data = fetch_stock_data(ticker, end_date, fine_tune_end_date)

                if not recent_data.empty:
                    # Preprocess recent data
                    processed_recent_data, new_states = preprocess_data(recent_data)

                    # Fine-tune the model
                    st.session_state['model'].fine_tune(new_states)

                    # Update data in session state
                    st.session_state['data'] = pd.concat([st.session_state['data'], processed_recent_data])
                    st.session_state['states'] = pd.concat([pd.DataFrame(st.session_state['states']), pd.DataFrame(new_states)]).values

                    st.sidebar.success("Model fine-tuned successfully!")
                else:
                    st.sidebar.warning("No new data available for fine-tuning.")

    # --- Prediction Section ---
    st.header("Prediction")
    if st.button("Predict Next Day's Market Movement"):
        if st.session_state['model'] is None:
            st.error("Please train a model before making predictions.")
        else:
            with st.spinner("Making prediction..."):
                # Use all available states for prediction
                all_states = st.session_state['states']

                # Predict the next state
                predicted_state = st.session_state['model'].predict_next_day_state(all_states)

                # Interpret the state
                state_interpretation = {
                    0: "Large Drop",
                    1: "Small Drop",
                    2: "Small Rise",
                    3: "Large Rise"
                }

                st.write(f"### Predicted Market Movement: **{state_interpretation.get(predicted_state, 'Unknown')}**")

# --- Data Display ---
st.header("Stock Data")
if st.session_state['data'] is not None:
    st.dataframe(st.session_state['data'].tail(10))
    st.line_chart(st.session_state['data']['Close'])
else:
    st.info("No data loaded yet.")