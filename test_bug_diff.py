import numpy as np
import pandas as pd
from model.backtesting import BacktestEngine, BacktestConfig

frame = pd.DataFrame({
    "Close": np.random.rand(1_000) * 100 + 50,
}, index=pd.date_range("2000-01-01", periods=1_000))

# _generate_signals actually already returns an array of 0s and 1s!
# np.isin(hidden_states, list(long_states)).astype(int)
# Let's verify this.
engine = BacktestEngine()
hidden_states = np.array([0, 1, 2, 3, 1, 0, 4, 1])
long_states = frozenset([1, 4])
signals = engine._generate_signals(hidden_states, long_states)
print(signals)
