# AI Coding Agent Instructions for HMM Stock Market Prediction

## Architecture Overview

This is a Streamlit-based Hidden Markov Model (HMM) stock market predictor with a clear separation of concerns:

- **`app/main.py`**: Streamlit frontend with session state management for model persistence and interactive training/prediction workflows
- **`model/hmm.py`**: Core HMM implementation using `hmmlearn.GaussianHMM` with 4-state discretization (large drop, small drop, small rise, large rise)
- **`model/utils.py`**: Data fetching via `yfinance` and preprocessing with fixed return thresholds (-1%, 0%, +1%)

## Key Patterns & Conventions

### State Management Pattern

The app uses Streamlit session state to persist trained models across user interactions:

```python
if 'model' not in st.session_state:
    st.session_state['model'] = None
```

Always check for `None` before using session state objects and provide clear error messages when models aren't trained.

### Data Flow Architecture

1. **Fetch**: `fetch_stock_data()` → raw OHLC data via yfinance
2. **Preprocess**: `preprocess_data()` → discretized states using fixed bins `[-inf, -0.01, 0, 0.01, inf]`
3. **Train**: `HMMStockPredictor.train()` → fits GaussianHMM with 4 states
4. **Predict**: `predict_next_day_state()` → uses transition matrix to predict most likely next state

### State Interpretation Convention

States are consistently mapped across the codebase:

- `0`: Large Drop (< -1%)
- `1`: Small Drop (-1% to 0%)
- `2`: Small Rise (0% to 1%)
- `3`: Large Rise (> 1%)

## Critical Implementation Details

### Path Resolution Pattern

All model imports require adding the project root to Python path:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This is necessary because `app/main.py` needs to import from the `model/` directory.

### Fine-Tuning Approach

Fine-tuning is implemented as complete model retraining on new data, not incremental learning. The `fine_tune()` method calls `model.fit(X_new)` directly.

### Data Concatenation Strategy

When fine-tuning, both processed data and states are concatenated:

```python
st.session_state['data'] = pd.concat([st.session_state['data'], processed_recent_data])
st.session_state['states'] = pd.concat([pd.DataFrame(st.session_state['states']), pd.DataFrame(new_states)]).values
```

## Development Workflow

### Running the Application

```bash
streamlit run app/main.py
```

### Dependencies

All dependencies are in `requirements.txt`: `streamlit`, `yfinance`, `numpy`, `pandas`, `hmmlearn`

### Testing Strategy

The app relies on interactive testing through the Streamlit interface. Test with different tickers (e.g., "GOOGL", "AAPL") and date ranges to verify model training and prediction workflows.

## Common Modification Patterns

### Adding New Features

- **New states**: Modify bins in `preprocess_data()` and update state interpretation dictionaries
- **Additional indicators**: Add calculations in `preprocess_data()` and update HMM input features
- **UI components**: Follow the sidebar configuration → main area display pattern established in `main.py`

### Model Enhancements

- **Different HMM configurations**: Modify `n_hidden_states` and `covariance_type` in `HMMStockPredictor.__init__()`
- **Alternative algorithms**: Replace `GaussianHMM` while maintaining the same interface (`train()`, `predict_next_day_state()`)
