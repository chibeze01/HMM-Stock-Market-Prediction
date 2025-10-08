import yfinance as yf
import numpy as np
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """
    Preprocesses the stock data for the HMM model.
    """
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()

    # Drop missing values
    data.dropna(inplace=True)

    # Discretize the returns into a number of states
    # For this example, we'll use 4 states:
    # 0: large drop
    # 1: small drop
    # 2: small rise
    # 3: large rise
    bins = [-np.inf, -0.01, 0, 0.01, np.inf]
    data['State'] = pd.cut(data['Returns'], bins=bins, labels=False)

    return data, data['State'].values.reshape(-1, 1)