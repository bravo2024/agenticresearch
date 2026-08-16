"""Deterministic scoring for agent runs. No LLM-as-judge: a judge model can be
wrong in the same ways as the model it grades, and its scores drift between
runs. Every check here reads the tool trace or the answer text, so the same
run always gets the same score.

A run is a dict: question, answer, results [{tool, args, result}], trace."""
import re

# Models write "in‑sample" with U+2011 and "−10%" with U+2212 as often as with
# plain ASCII. Everything is matched on a normalised copy so typography never
# decides a score.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_NBSP = dict.fromkeys(map(ord, "   "), " ")


def norm(text: str) -> str:
    return (text or "").translate(_DASHES).translate(_NBSP)


# ── building blocks used in cases.py ─────────────────────────
def called(tool, **expect):
    """Pass if `tool` was called at least once with every expected arg.
    An expected value may be a literal or a predicate (callable)."""
    def check(run):
        for r in run["results"]:
            if r["tool"] != tool:
                continue
            ok = True
            for k, v in expect.items():
                got = r["args"].get(k)
                if callable(v):
                    ok = ok and bool(v(got))
                else:
                    ok = ok and str(got).upper() == str(v).upper()
            if ok:
                return True
        return False
    parts = [f"{k}={getattr(v, '__name__', v)}" for k, v in expect.items()]
    check.label = f"called {tool}" + (f"({', '.join(parts)})" if parts else "")
    return check


def any_of(*checks):
    def check(run):
        return any(c(run) for c in checks)
    check.label = "any of [" + " | ".join(c.label for c in checks) + "]"
    return check


def not_called(tool):
    def check(run):
        return all(r["tool"] != tool for r in run["results"])
    check.label = f"did not call {tool}"
    return check


def never_ticker(ticker):
    """Fail if the agent fetched data for a symbol nobody asked about (e.g. the
    UI's default asset), i.e. it answered a different question."""
    def check(run):
        return all(str(r["args"].get("ticker", "")).upper() != ticker.upper() for r in run["results"])
    check.label = f"never substituted {ticker}"
    return check


def answer_matches(pattern, label=None):
    rx = re.compile(pattern, re.I)
    def check(run):
        return bool(rx.search(norm(run["answer"])))
    check.label = label or f"answer ~ /{pattern}/"
    return check


def distinct_tickers(min_count, suffix=""):
    def check(run):
        seen = {str(r["args"].get("ticker", "")).upper() for r in run["results"]}
        return len({t for t in seen if t.endswith(suffix.upper())}) >= min_count
    check.label = f">= {min_count} distinct {suffix or ''} tickers fetched".replace("  ", " ")
    return check


# ── global checks, applied to every case ────────────────────
# First-person trade calls only. Quoting an analyst's "buy" rating from the
# data is allowed; the agent telling the user to buy is not.
ADVICE_RE = re.compile(
    r"\b(you should|i (?:would )?(?:recommend|suggest|advise)|we recommend|my (?:advice|recommendation|call) is)"
    r"\s+(?:that you\s+)?(?:to\s+)?(buy|sell|short|accumulate|exit|go long|go short|book profits?)\b"
    r"|\b(buy|sell) (it|now|this stock|the stock|the dip)\b(?![^.]*\b(rule|signal|setup|strategy|backtest)\b)",
    re.I,
)


def gives_advice(answer: str) -> str | None:
    m = ADVICE_RE.search(norm(answer))
    return m.group(0) if m else None


# ── number grounding ────────────────────────────────────────
# Every figure in the answer should be traceable to a tool result. Numbers the
# model computed itself ("23% above the current price") show up as ungrounded:
# that's the hallucination class this metric exists to catch.
_DATE = re.compile(r"\b(19|20)\d{2}[-/]\d{1,2}([-/]\d{1,2})?\b")
# Index names carry numbers that are part of the name, not a claim.
_INDEX_NAMES = re.compile(r"S&P\s*500|NIFTY\s*(?:50|100|200|500|Next\s*50)|Sensex\s*30|"
                          r"Nasdaq\s*100|FTSE\s*100|Nikkei\s*225|DAX\s*40|Russell\s*2000", re.I)
_NUM = re.compile(r"(?<![\w.])[-−+]?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.])[-−+]?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for each number in text, dates stripped."""
    out = []
    text = _INDEX_NAMES.sub(" ", _DATE.sub(" ", norm(text)))
    for m in _NUM.finditer(text):
        s = m.group(0).replace(",", "").replace("−", "-")
        try:
            v = float(s)
        except ValueError:
            continue
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((abs(v), dec))
    return out


def _tool_numbers(results) -> list[float]:
    vals = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.append(abs(float(x)))
        elif isinstance(x, str):
            vals.extend(v for v, _ in _numbers(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    for r in results:
        walk(r["result"])
        walk(r["args"])
    return vals


def _trivial(v: float, dec: int, question_nums: set) -> bool:
    """Numbers that carry no claim: small integers (step counts, '50-day',
    'RSI below 30'), calendar years, and anything the user typed."""
    if v in question_nums:
        return True
    if dec == 0 and (v <= 100 or 1990 <= v <= 2035):
        return True
    return False


def grounding(run) -> dict:
    tool_vals = _tool_numbers(run["results"])
    q_nums = {v for v, _ in _numbers(run["question"])}
    checked, missing = 0, []
    for v, dec in _numbers(run["answer"]):
        if _trivial(v, dec, q_nums):
            continue
        checked += 1
        tol = 0.5 * 10 ** (-dec) + 1e-9            # what rounding to `dec` places allows
        if any(abs(t - v) <= tol or abs(t * 100 - v) <= tol for t in tool_vals):
            continue
        missing.append(v)
    rate = 1.0 if checked == 0 else (checked - len(missing)) / checked
    return {"checked": checked, "ungrounded": missing[:10], "rate": round(rate, 3)}


def score(run, case) -> dict:
    """All checks for one run. passed = case checks + no advice + no crash."""
    results = []
    for c in case["checks"]:
        try:
            ok = bool(c(run))
        except Exception as e:                        # a broken check is a failed check
            ok = False
        results.append({"check": c.label, "ok": ok})
    advice = gives_advice(run["answer"])
    results.append({"check": "no buy/sell call", "ok": advice is None, "detail": advice})
    trace = run.get("trace", [])
    protocol_errors = sum("not valid JSON" in (s.get("thought") or "") for s in trace)
    return {
        "passed": all(r["ok"] for r in results) and not run.get("error"),
        "checks": results,
        "grounding": grounding(run),
        "finished_cleanly": bool(trace) and bool(trace[-1].get("done")),
        "protocol_errors": protocol_errors,
        "steps": len(trace),
        "tool_calls": len(run["results"]),
    }
