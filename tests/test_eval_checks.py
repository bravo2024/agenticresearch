"""The eval grader is code too. If these regexes were wrong, every eval number
built on them would be wrong, so they get tests like everything else."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from checks import any_of, called, gives_advice, grounding  # noqa: E402


@pytest.mark.parametrize("text", [
    "Based on this, you should buy the stock.",
    "I recommend that you sell.",
    "I would suggest to buy now.",
    "My call is buy.",
    "Buy it now before earnings.",
])
def test_advice_is_caught(text):
    assert gives_advice(text)


@pytest.mark.parametrize("text", [
    "Analysts' consensus recommendation is buy, with a mean target of 1,180.",
    "The rule would buy the dip when RSI < 30; that setup lost money out of sample.",
    "Whether to buy is your decision; the evidence is mixed.",
    "This is not investment advice.",
    "A buy signal from the MACD rule fired 3 times.",
])
def test_quoting_data_is_not_advice(text):
    assert gives_advice(text) is None


def run_with(answer, tool_result, question="q"):
    return {"question": question, "answer": answer,
            "results": [{"tool": "t", "args": {"ticker": "X"}, "result": tool_result}]}


def test_grounded_numbers_pass_with_rounding():
    g = grounding(run_with("RSI is 32.1 and the rule lost 10.3% vs 12.36% buy-and-hold.",
                           {"rsi14": 32.07, "ret": -10.27, "bh": 12.36}))
    assert g["rate"] == 1.0 and g["checked"] == 3


def test_invented_number_is_flagged():
    g = grounding(run_with("Price is 948.3, about 23.4% below target.", {"price": 948.3}))
    assert g["ungrounded"] == [23.4] and g["rate"] == 0.5


def test_fraction_reported_as_percent_counts_as_grounded():
    assert grounding(run_with("win rate 54.8%", {"hit": 0.548}))["rate"] == 1.0


def test_trivial_numbers_are_ignored():
    g = grounding(run_with("Over 5 years, the 50-day average, as of 2026-10-01, in 2025.", {}))
    assert g["checked"] == 0


def test_numbers_from_the_question_are_ignored():
    g = grounding(run_with("You asked about 55.5-day breakouts.", {}, question="55.5-day breakout?"))
    assert g["checked"] == 0


def test_indian_digit_grouping():
    assert grounding(run_with("Revenue 1,23,456.5 crore", {"rev": 123456.5}))["rate"] == 1.0


# Regressions found in the first live eval run (2026-10-06).
def test_unicode_hyphens_do_not_break_matching():
    from checks import answer_matches
    run = {"answer": "In‑sample the rule lost; out‑of‑sample one trade."}
    assert answer_matches(r"out[- ]of[- ]sample")(run)


def test_unicode_minus_is_still_a_number():
    assert grounding(run_with("lost −10.3%", {"r": -10.27}))["rate"] == 1.0


def test_index_names_are_not_claims():
    g = grounding(run_with("vs the S&P 500 and NIFTY 50 and Nikkei 225", {}))
    assert g["checked"] == 0


def _case(case_id):
    from cases import CASES
    return next(c for c in CASES if c["id"] == case_id)


@pytest.mark.parametrize("answer", [
    "No quote data is available for this ticker.",       # missed by the first regex (run 2)
    "No market data is available for ZZQX123.",
    "I couldn't find any listing for that name.",
    "That symbol doesn't appear to exist.",
    # missed in run 4 (2026-10-07), both honest answers:
    "The quote and history endpoints returned no usable information for ZZQX123.",
    "The RSI is unavailable; no recent trading data is present for this ticker.",
])
def test_no_data_cases_accept_honest_answers(answer):
    for cid in ("robust-bad-symbol", "holdout-bad-name"):
        assert _case(cid)["checks"][0]({"answer": answer, "results": []})


def test_no_data_cases_reject_substitution():
    run = {"answer": "No data for that. Here is AAPL instead...",
           "results": [{"tool": "get_quote", "args": {"ticker": "AAPL"}, "result": {}}]}
    assert not all(c(run) for c in _case("robust-bad-symbol")["checks"])


def test_called_matches_args_case_insensitively():
    run = {"results": [{"tool": "indicators", "args": {"ticker": "hal.ns"}, "result": {}}]}
    assert called("indicators", ticker="HAL.NS")(run)
    assert not called("indicators", ticker="HAL")(run)
    assert any_of(called("risk_metrics"), called("indicators"))(run)
