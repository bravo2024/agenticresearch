"""The research chart is built from a price frame. No network, no Streamlit render."""
import numpy as np
import pandas as pd

import app
from conftest import make_df


def _names(fig):
    return [t.name for t in fig.data]


def test_figure_has_candles_sma_bb_and_panels():
    df = make_df(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 80)))
    fig = app.build_research_figure(
        df, overlays=("sma20", "bb"), panels=("volume", "rsi", "macd"),
    )
    names = _names(fig)
    assert "Price" in names
    assert "SMA20" in names
    assert "BB upper" in names and "BB lower" in names
    assert "Volume" in names
    assert "RSI" in names
    assert "MACD" in names and "MACD signal" in names


def test_comparison_is_rebased_on_a_second_axis():
    df = make_df(np.linspace(100, 130, 40))
    other = make_df(np.linspace(50, 80, 40))
    fig = app.build_research_figure(df, compare=other, compare_name="BTC-USD", panels=())
    trace = next(t for t in fig.data if t.name and "rebased" in t.name)
    assert trace.name.startswith("BTC-USD")
    assert float(trace.y[0]) == 100
    assert abs(float(trace.y[-1]) - 160) < 1e-6


def test_rule_marks_and_split_land_on_the_figure():
    close = np.concatenate([
        np.linspace(100, 80, 40),
        np.linspace(80, 140, 80),
    ])
    df = make_df(close)
    marks = app.rule_trade_marks(df, "rsi_oversold")
    assert marks
    assert {m["side"] for m in marks} <= {"buy", "sell"}
    fig = app.build_research_figure(
        df, marks=marks, split_at=df.index[len(df) // 2], panels=("rsi",),
    )
    names = _names(fig)
    assert "buy" in names or "sell" in names
    assert fig.layout.shapes


def test_empty_frame_draws_nothing():
    assert app.build_research_figure(pd.DataFrame()) is None
