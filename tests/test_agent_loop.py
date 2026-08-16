"""The agent loop's control logic, driven by a scripted fake LLM. These prove
the loop properties claimed in the README without spending an API call:
results flow back into the next decision, caps hold, repeats are refused,
bad replies get one retry, and a dead LLM still yields an answer."""
import json

import pytest

import app


class ScriptedLLM:
    """Plays back agent replies in order and records every prompt it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        if not self.replies:
            return {"ok": False, "error": "script exhausted"}
        r = self.replies.pop(0)
        return {"ok": True, "text": r if isinstance(r, str) else json.dumps(r)}


@pytest.fixture
def tools(monkeypatch):
    """Replace run_tool with a recorder that returns a recognisable payload."""
    calls = []

    def _run_tool(name, args, provider="", model=""):
        calls.append((name, dict(args)))
        return {"tool": name, "ticker": args.get("ticker"), "rsi14": 24.5 if name == "indicators" else None}

    monkeypatch.setattr(app, "run_tool", _run_tool)
    return calls


def call(name, **args):
    return {"name": name, "args": args}


def run(monkeypatch, replies, question="Is RELIANCE.NS oversold?"):
    llm = ScriptedLLM(replies)
    monkeypatch.setattr(app, "llm_chat", llm)
    results, trace = app.agent_loop(question, ["RELIANCE.NS"], "AAPL")
    return results, trace, llm


def test_results_feed_the_next_decision(monkeypatch, tools):
    results, trace, llm = run(monkeypatch, [
        {"thought": "check indicators", "tools": [call("indicators", ticker="RELIANCE.NS")]},
        {"thought": "rsi 24.5, test the dip rule",
         "tools": [call("backtest_rule", ticker="RELIANCE.NS", rule="rsi_oversold")]},
        {"thought": "enough", "done": True},
    ])
    assert [n for n, _ in tools] == ["indicators", "backtest_rule"]
    assert "(none yet" in llm.prompts[0]
    assert "24.5" in llm.prompts[1]               # step 2 saw step 1's result
    assert trace[-1].get("done") is True and len(trace) == 3


def test_step_cap(monkeypatch, tools):
    replies = [{"thought": f"t{i}", "tools": [call("get_quote", ticker=f"T{i}.NS")]}
               for i in range(20)]
    results, trace, _ = run(monkeypatch, replies)
    assert len(trace) == app.AGENT_MAX_STEPS
    assert len(results) == app.AGENT_MAX_STEPS


def test_call_cap(monkeypatch, tools):
    replies = [{"thought": "x", "tools": [call("get_quote", ticker=f"S{i}{j}.NS") for j in range(3)]}
               for i in range(20)]
    results, _, _ = run(monkeypatch, replies)
    assert len(results) == app.AGENT_MAX_CALLS


def test_repeat_calls_are_refused(monkeypatch, tools):
    same = {"thought": "again", "tools": [call("indicators", ticker="RELIANCE.NS")]}
    results, _, llm = run(monkeypatch, [same, same, {"thought": "ok", "done": True}])
    assert len(tools) == 1
    assert "already called" in llm.prompts[2]


def test_unknown_tool_is_reported_not_run(monkeypatch, tools):
    _, _, llm = run(monkeypatch, [
        {"thought": "x", "tools": [call("place_order", ticker="RELIANCE.NS")]},
        {"thought": "ok", "done": True},
    ])
    assert "place_order" not in [n for n, _ in tools]
    assert "not a tool" in llm.prompts[1]


def test_bad_json_gets_one_retry(monkeypatch, tools):
    results, trace, _ = run(monkeypatch, [
        '{"thought": "cut off',
        {"thought": "x", "tools": [call("indicators", ticker="RELIANCE.NS")]},
        {"thought": "ok", "done": True},
    ])
    assert [n for n, _ in tools] == ["indicators"]
    assert trace[-1].get("done") is True


def test_dead_llm_falls_back_to_snapshot(monkeypatch, tools):
    monkeypatch.setattr(app, "llm_chat", lambda *a, **k: {"ok": False, "error": "all providers down"})
    results, trace = app.agent_loop("anything", ["RELIANCE.NS"], "AAPL")
    assert [r["tool"] for r in results] == ["get_quote", "indicators", "fundamental_facts"]
    assert "default snapshot" in trace[-1]["thought"]


def test_run_analysis_gets_question_and_all_tickers(monkeypatch, tools):
    run(monkeypatch, [
        {"thought": "x", "tools": [call("run_analysis", ticker="RELIANCE.NS")]},
        {"thought": "ok", "done": True},
    ], question="Compare RELIANCE.NS volatility")
    name, args = tools[0]
    assert args["question"] == "Compare RELIANCE.NS volatility"
    assert args["tickers"][0] == "RELIANCE.NS"
