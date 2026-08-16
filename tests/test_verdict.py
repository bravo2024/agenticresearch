"""Direction is long or flat only when the out-of-sample test survives."""
import numpy as np
import pandas as pd

import app
from conftest import make_df


def test_block_bootstrap_separates_a_real_mean_from_noise():
    lo, hi = app._block_bootstrap_mean_ci(np.full(120, 0.01))
    assert lo > 0
    noise = np.random.default_rng(0).normal(0, 0.01, 200)
    lo, hi = app._block_bootstrap_mean_ci(noise)
    assert lo < 0 < hi


def test_clear_daily_edge_is_not_killed():
    oos = {"trades": 40, "strategy_return_pct": 25.0, "buy_and_hold_pct": 5.0}
    judged = app._judge_excess(oos, np.full(120, 0.001), n_trials=1)
    assert judged["killed_by"] == []
    assert judged["ci"][0] > 0


def test_losing_to_buy_and_hold_and_a_zero_mean_are_killed():
    oos = {"trades": 40, "strategy_return_pct": 1.0, "buy_and_hold_pct": 20.0}
    excess = np.random.default_rng(0).normal(0, 0.01, 120)
    judged = app._judge_excess(oos, excess, n_trials=1)
    assert "lost_to_buy_and_hold" in judged["killed_by"]
    assert "excess_not_significant" in judged["killed_by"]


def test_too_few_trades_kills_even_a_strong_mean():
    oos = {"trades": 2, "strategy_return_pct": 30.0, "buy_and_hold_pct": 1.0}
    judged = app._judge_excess(oos, np.full(80, 0.002), n_trials=1)
    assert judged["killed_by"] == ["too_few_trades"]


def test_more_trials_tighten_the_interval():
    oos = {"trades": 40, "strategy_return_pct": 12.0, "buy_and_hold_pct": 4.0}
    excess = np.random.default_rng(1).normal(0.0004, 0.01, 100)
    one = app._judge_excess(oos, excess, n_trials=1)
    many = app._judge_excess(oos, excess, n_trials=50)
    assert many["alpha"] < one["alpha"]
    assert many["ci"][0] <= one["ci"][0]
    assert many["ci"][1] >= one["ci"][1]


def test_direction_follows_the_kill_list_on_a_random_walk():
    rng = np.random.default_rng(2)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 500))
    df = make_df(close)
    last = float(close[-1])
    out = app.compute_verdict_from_frame(
        df, "rsi_oversold", {}, 0.001, 7, 1,
        forecast={
            "center_est": last * 1.01,
            "range_70": [last * 0.97, last * 1.03],
            "range_95": [last * 0.94, last * 1.06],
        },
        ticker="TEST",
    )
    assert out["direction"] == app._direction_from_tests(
        out["rule_position"] == "long", out["killed_by"])
    if out["direction"] == "no-edge":
        assert out["summary"].startswith("No tested direction")
        assert out["follow_rule"] is False
    else:
        assert out["summary"].startswith("Tested direction")
        assert out["follow_rule"] is True
    hit95 = out["forecast"]["band95_hit_rate"]
    assert 0.8 <= hit95 <= 0.995
    assert out["forecast"]["price_bias"] == "up"
    assert out["forecast"]["upside_95_pct"] > 0
    assert out["forecast"]["downside_95_pct"] > 0


def test_intraday_cone_is_a_short_band():
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(4).normal(0, 0.001, 80)))
    cone = app._intraday_cone(close, 30)
    assert cone["horizon"] == 30
    assert cone["range_95"][0] < cone["center_est"] < cone["range_95"][1]


def test_five_minute_verdict_speaks_in_bars():
    rng = np.random.default_rng(5)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.0008, 800))
    df = make_df(close)
    df.index = pd.date_range("2026-10-01 09:30", periods=len(df), freq="5min")
    out = app.compute_verdict_from_frame(
        df, "rsi_oversold", {}, 0.0001, 12, 1, ticker="AAPL", interval="5m",
        forecast=app._intraday_cone(df["Close"], 12),
    )
    assert out["interval"] == "5m"
    assert "this 5m window" in out["summary"]
    assert "09:30" in out["as_of"] or ":" in out["as_of"]
    assert out["forecast"]["horizon_bars"] == 12


def test_minute_cone_lands_on_the_next_bars_not_the_next_days():
    df = make_df(np.linspace(100, 110, 80))
    df.index = pd.date_range("2026-10-06 09:30", periods=len(df), freq="1min")
    fig = app.build_research_figure(
        df, panels=(),
        cone={"range_70": [108, 112], "range_95": [106, 114], "horizon": 15},
    )
    cone = next(t for t in fig.data if t.name == "range_95")
    end = pd.Timestamp(cone.x[-2])
    assert end - df.index[-1] == pd.Timedelta(minutes=15)


def test_no_edge_when_the_rule_cannot_beat_a_straight_line():
    # A steady rise: once the moving-average rule is long it just matches
    # buy-and-hold, and the early cash drag plus costs lose to holding.
    df = make_df(np.linspace(100, 180, 400))
    out = app.compute_verdict_from_frame(df, "ma_cross", {}, 0.001, 7, 1, ticker="UP")
    assert "lost_to_buy_and_hold" in out["killed_by"]
    assert out["direction"] == "no-edge"
