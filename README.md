# HMM Stock Market Predictor

Interactive Streamlit application that trains a Hidden Markov Model (HMM) on historical
stock prices, fine-tunes it with new data, and surfaces regime diagnostics to explain
predicted market states.

## Feature Highlights
- Deterministic data pipeline with caching, retries, and configurable feature engineering
  (returns, rolling volatility, momentum, custom bins).
- Configurable Gaussian HMM (state count, covariance type, training iterations, random seed)
  with fine-tuning, evaluation metrics, and model persistence.
- Streamlit workflow covering training, fine-tuning, evaluation dashboards, predictions,
  and history tracking with guardrails for invalid inputs.
- Automated unit tests plus GitHub Actions CI, code formatting (Black), linting (Ruff),
  and optional pre-commit hooks.
- Lightweight fallback implementation of `GaussianHMM` when `hmmlearn` is unavailable,
  so local development works even without external wheels.

## Project Structure
```
app/                # Streamlit UI
model/              # Modeling, preprocessing, evaluation, fallback HMM
tests/              # Unit & smoke tests
.github/workflows/  # CI pipeline
```

## Getting Started
1. **Prerequisites**
   - Python 3.10+
   - pip
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # optional tooling
   pip install -r requirements-dev.txt
   ```
   If `hmmlearn` cannot be installed (e.g., no network), the app automatically falls back
   to the bundled simplified implementation.
3. **Run tests**
   ```bash
   python -m unittest discover -s tests -v
   ```
4. **Launch Streamlit**
   ```bash
   streamlit run app/main.py
   ```
   Use the sidebar to configure tickers, date ranges, features, and model hyperparameters.

## Development Workflow
- **Formatting / Linting**
  ```bash
  black .
  ruff check .
  ```
- **Pre-commit hooks**
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```
- **Continuous Integration**
  Pushing to `main`/`master` or opening a PR runs linting/tests via GitHub Actions
  (`.github/workflows/ci.yml`).

## Deployment
- **Streamlit entry point**: `streamlit run app/main.py`
- **Container build**:
  ```bash
  docker build -t hmm-stock .
  docker run -p 8501:8501 hmm-stock
  ```
  (See `Dockerfile` for details; container exposes Streamlit on port 8501.)

## Troubleshooting
- **No data returned**: Ensure the ticker/date range is valid and markets were open.
- **Insufficient observations**: Expand the training window or reduce the number of hidden states.
- **Dependency issues**: Install from `requirements-dev.txt` and consider using the fallback HMM.
