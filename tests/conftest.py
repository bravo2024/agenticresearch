"""Shared fixtures. Tests run offline: price history and the LLM are replaced
with synthetic data and scripted replies, so results are exact and repeatable."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402  (Streamlit calls at import time are no-ops outside `streamlit run`)


def make_df(close, start="2020-01-01") -> pd.DataFrame:
    """OHLCV frame around a close series, on business days."""
    close = pd.Series(np.asarray(close, dtype=float))
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({
        "Open": close.values,
        "High": close.values * 1.01,
        "Low": close.values * 0.99,
        "Close": close.values,
        "Volume": np.full(len(close), 1_000_000.0),
    }, index=idx)


@pytest.fixture
def fake_history(monkeypatch):
    """Route get_history to a dict of synthetic frames: {ticker: DataFrame}."""
    frames = {}

    def _get_history(ticker, interval="1d", period="2y"):
        return frames.get(ticker, pd.DataFrame()).copy()

    monkeypatch.setattr(app, "get_history", _get_history)
    app.pair_correlation.clear()
    return frames
