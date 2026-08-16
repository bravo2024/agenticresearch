"""The numbers the agent quotes come from these functions, so they are tested
on series where the right answer is known in advance."""
import numpy as np
import pandas as pd
import pytest

import app
from conftest import make_df


# ── RSI ──────────────────────────────────────────────────────
def test_rsi_bounds_and_warmup():
    close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300)))
    rsi = app.rsi_series(close)
    assert rsi.iloc[:14].isna().all()            # no reading before 14 bars
    assert rsi.dropna().between(0, 100).all()


def test_rsi_extremes():
    assert app.rsi_series(pd.Series(np.arange(1, 61, dtype=float))).iloc[-1] == 100
    assert app.rsi_series(pd.Series(np.arange(60, 0, -1, dtype=float))).iloc[-1] == 0


# ── Market / cost routing ───────────────────────────────────
@pytest.mark.parametrize("ticker,market,bench", [
    ("RELIANCE.NS", "india", "^NSEI"),
    ("SBIN.BO", "india", "^NSEI"),
    ("AAPL", "us", "^GSPC"),
    ("ETH-USD", "crypto", "BTC-USD"),
    ("BTC-USD", "crypto", None),
    ("^NSEI", "india", None),
    ("GC=F", "other", None),
    ("INR=X", "other", None),
])
def test_market_and_benchmark(ticker, market, bench):
    assert app.market_of(ticker) == market
    assert app.benchmark_for(ticker) == bench


def test_indian_costs_are_higher_than_us():
    # STT on both sides makes Indian delivery trades several times dearer.
    assert app.TRADE_COSTS["india"] > 2 * app.TRADE_COSTS["us"]


# ── Ticker normalisation (offline paths only) ───────────────
@pytest.mark.parametrize("raw,expected", [
    ("RELIANCE.NS", "RELIANCE.NS"),
    ("^nsei", "^NSEI"),
    ("gc=f", "GC=F"),
    ("btc-usd", "BTC-USD"),
    ("Infosys", "INFY.NS"),          # alias table, no network
    ("", ""),
])
def test_normalize_ticker_offline(raw, expected):
    assert app.normalize_ticker(raw) == expected


# ── Backtest accounting ─────────────────────────────────────
def test_always_long_equals_buy_and_hold_minus_entry_cost():
    close = pd.Series(np.linspace(100, 150, 200), index=pd.bdate_range("2020-01-01", periods=200))
    held = pd.Series(1, index=close.index)
    cost = 0.002
    s = app._segment_stats(close, held, cost, None)
    expected = ((1 - cost) * (150 / 100) - 1) * 100
    assert s["strategy_return_pct"] == pytest.approx(expected, abs=0.01)
    assert s["buy_and_hold_pct"] == pytest.approx(50.0, abs=0.01)
    assert s["time_in_market_pct"] == 100.0


def test_flat_strategy_makes_nothing():
    close = pd.Series(np.linspace(100, 150, 200), index=pd.bdate_range("2020-01-01", periods=200))
    s = app._segment_stats(close, pd.Series(0, index=close.index), 0.002, None)
    assert s["strategy_return_pct"] == 0
    assert s["trades"] == 0 and s["win_rate_pct"] is None


def test_round_trip_trade_pnl_includes_both_sides():
    close = pd.Series([100, 100, 110, 110, 110], index=pd.bdate_range("2020-01-01", periods=5),
                      dtype=float)
    held = pd.Series([0, 0, 1, 0, 0], index=close.index)     # bought at close of day 1, sold day 2
    s = app._segment_stats(close, held, 0.001, None)
    assert s["trades"] == 1
    assert s["avg_win_pct"] == pytest.approx((110 / 100 - 1 - 0.002) * 100, abs=0.01)


def test_backtest_has_no_lookahead(fake_history, monkeypatch):
    """A signal seen at day k's close must not earn day k's own move."""
    n = 400
    close = np.full(n, 100.0)
    k = 350
    close[k:] = 150.0                                   # +50% jump on day k
    fake_history["TEST.NS"] = make_df(close)

    def signal_on_jump_day(df, rule, p):
        pos = pd.Series(0, index=df.index)
        pos.iloc[k:] = 1                                # rule "sees" the jump at k's close
        return pos

    monkeypatch.setattr(app, "_rule_positions", signal_on_jump_day)
    out = app.backtest_rule("TEST.NS", "rsi_oversold", cost_per_side=0.0)
    assert out["out_of_sample"]["strategy_return_pct"] == pytest.approx(0.0, abs=1e-9)
    assert out["out_of_sample"]["buy_and_hold_pct"] == pytest.approx(50.0, abs=0.01)


def test_backtest_split_is_chronological_and_disjoint(fake_history):
    rng = np.random.default_rng(1)
    fake_history["TEST.NS"] = make_df(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1000))))
    fake_history["^NSEI"] = make_df(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1000))))
    out = app.backtest_rule("TEST.NS", "macd_cross")
    ins, oos = out["in_sample"], out["out_of_sample"]
    assert ins["to"] < oos["from"]
    assert out["benchmark"] == "^NSEI" and "index_return_pct" in oos
    assert out["cost_per_side_pct"] == pytest.approx(app.TRADE_COSTS["india"] * 100)


def test_backtest_rejects_unknown_rule_and_short_history(fake_history):
    fake_history["TEST.NS"] = make_df(np.linspace(100, 120, 1000))
    assert "error" in app.backtest_rule("TEST.NS", "astrology")
    fake_history["SHORT.NS"] = make_df(np.linspace(100, 120, 100))
    assert "error" in app.backtest_rule("SHORT.NS", "macd_cross")


@pytest.mark.parametrize("rule", list(app.BACKTEST_RULES))
def test_every_rule_produces_binary_positions(rule):
    df = make_df(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.02, 500))))
    pos = app._rule_positions(df, rule, {})
    assert set(pos.unique()) <= {0, 1}


# ── Indicators ──────────────────────────────────────────────
def test_indicators_on_steady_uptrend(fake_history):
    fake_history["UP.NS"] = make_df(np.linspace(100, 200, 400))
    fake_history["^NSEI"] = make_df(np.linspace(100, 200, 400))
    ind = app.indicators("UP.NS")
    assert ind["lagging"]["sma50_vs_sma200"].startswith("golden cross")
    assert ind["leading"]["rsi_zone"] == "overbought (>70)"
    # Stock and index move identically, so there is no excess return.
    assert ind["relative_strength"]["excess_return_3m_pct"] == pytest.approx(0, abs=0.01)


def test_indicators_kind_filter(fake_history):
    fake_history["UP.NS"] = make_df(np.linspace(100, 200, 400))
    ind = app.indicators("UP.NS", kind="volatility")
    assert "volatility" in ind and "leading" not in ind and "lagging" not in ind


# ── Lead-lag ────────────────────────────────────────────────
def test_lag_detects_a_true_lead(fake_history):
    rng = np.random.default_rng(3)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    b = np.roll(a, 1)                         # b repeats a's move one day later
    b[0] = a[0]
    fake_history["A.NS"] = make_df(a)
    fake_history["B.NS"] = make_df(b)
    out = app.pair_correlation("A.NS", "B.NS", lag=2)
    ll = out["lead_lag_correlations"]
    assert ll["a_today_vs_b_in_1d"] > 0.95
    assert abs(ll["b_today_vs_a_in_1d"]) < 0.2
    assert "timing_caveat" not in out          # same exchange, same clock


def test_lag_warns_across_trading_sessions(fake_history):
    rng = np.random.default_rng(4)
    fake_history["CL=F"] = make_df(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    fake_history["INR=X"] = make_df(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    assert "timing_caveat" in app.pair_correlation("CL=F", "INR=X", lag=1)


# ── LLM reply parsing ───────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ('{"done": true}', {"done": True}),
    ('```json\n{"done": true}\n```', {"done": True}),
    ('Sure! {"tools": [{"name": "x", "args": {"a": 1}}]} hope that helps',
     {"tools": [{"name": "x", "args": {"a": 1}}]}),
    ('{"thought": "cut off mid', None),
    ("no json here", None),
    ("[1, 2]", None),
])
def test_parse_json_obj(raw, expected):
    assert app._parse_json_obj(raw) == expected
