"""Run the eval set against the live agent and score it.

    python evals/run_evals.py                      # all cases, once
    python evals/run_evals.py --repeats 3          # LLM output varies; see how much
    python evals/run_evals.py --only entry-oversold-bajaj --verbose
    python evals/run_evals.py --category entry

Writes evals/results/<timestamp>.json (every run, full trace) and .md (summary).
Uses live market data and the free LLM providers, so a single run is a
sample, not a constant: report pass rates over repeats when it matters.
"""
import argparse
import json
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import app  # noqa: E402
from cases import CASES  # noqa: E402
from checks import score  # noqa: E402


def run_case(case, provider, model):
    t0 = time.time()
    try:
        r = app.answer_question(case["question"], provider=provider, model=model)
        run = {"question": case["question"], "answer": r["answer"], "results": r["results"],
               "trace": r["trace"], "provider": r.get("provider"), "model": r.get("model")}
    except Exception as e:
        run = {"question": case["question"], "answer": "", "results": [], "trace": [],
               "error": f"{type(e).__name__}: {e}"[:300]}
    run["latency_s"] = round(time.time() - t0, 1)
    run["score"] = score(run, case)
    return run


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def summarise(runs):
    n = len(runs)
    s = [r["score"] for r in runs]
    by_cat = defaultdict(list)
    for r in runs:
        by_cat[r["category"]].append(r["score"]["passed"])
    case_checks = [c for x in s for c in x["checks"] if c["check"] != "no buy/sell call"]
    g_checked = sum(x["grounding"]["checked"] for x in s)
    g_missing = sum(len(x["grounding"]["ungrounded"]) for x in s)
    held = [r["score"]["passed"] for r in runs if r.get("holdout")]
    tuned = [r["score"]["passed"] for r in runs if not r.get("holdout")]
    return {
        "runs": n,
        "pass_rate": pct(sum(x["passed"] for x in s), n),
        "pass_rate_dev_cases": pct(sum(tuned), len(tuned)),
        "pass_rate_holdout": pct(sum(held), len(held)),
        "check_pass_rate": pct(sum(c["ok"] for c in case_checks), len(case_checks)),
        "advice_violations": sum(not c["ok"] for x in s for c in x["checks"] if c["check"] == "no buy/sell call"),
        "numbers_grounded": pct(g_checked - g_missing, g_checked),
        "finished_cleanly": pct(sum(x["finished_cleanly"] for x in s), n),
        "protocol_errors": sum(x["protocol_errors"] for x in s),
        "crashes": sum(bool(r.get("error")) for r in runs),
        "mean_steps": round(statistics.mean(x["steps"] for x in s), 2),
        "mean_tool_calls": round(statistics.mean(x["tool_calls"] for x in s), 2),
        "median_latency_s": statistics.median(r["latency_s"] for r in runs),
        "by_category": {k: pct(sum(v), len(v)) for k, v in sorted(by_cat.items())},
        "models": dict(Counter(f"{r.get('provider')}/{r.get('model')}" for r in runs)),
    }


def markdown(summary, runs, args):
    lines = [f"# Eval run {datetime.now():%Y-%m-%d %H:%M}",
             f"provider `{args.provider}` · repeats {args.repeats} · {summary['runs']} runs", "",
             "| metric | value |", "|---|---|"]
    for k, v in summary.items():
        if k not in ("by_category", "models"):
            lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines += ["", "| category | pass rate |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["by_category"].items()]
    lines += ["", "## Failures", ""]
    fails = [r for r in runs if not r["score"]["passed"]]
    if not fails:
        lines.append("none")
    for r in fails:
        bad = [c["check"] + (f" ({c['detail']})" if c.get("detail") else "")
               for c in r["score"]["checks"] if not c["ok"]]
        called = ", ".join(f"{x['tool']}({x['args'].get('ticker')})" for x in r["results"])
        lines.append(f"- **{r['id']}**: failed {'; '.join(bad) or r.get('error')}  \n  called: {called or 'nothing'}")
    lines += ["", "## Ungrounded numbers", ""]
    for r in runs:
        g = r["score"]["grounding"]
        if g["ungrounded"]:
            lines.append(f"- {r['id']}: {g['ungrounded']} ({g['rate']:.0%} of {g['checked']} grounded)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="case ids")
    ap.add_argument("--category")
    ap.add_argument("--provider", default="", help="pin a provider; default is the auto failover order")
    ap.add_argument("--model", default="")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between runs (free-tier rate limits)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = [c for c in CASES
             if (not args.only or c["id"] in args.only)
             and (not args.category or c["category"] == args.category)]
    if not cases:
        sys.exit("no cases matched")

    runs = []
    total = len(cases) * args.repeats
    for rep in range(args.repeats):
        for case in cases:
            run = run_case(case, args.provider, args.model)
            run.update(id=case["id"], category=case["category"], repeat=rep,
                       holdout=bool(case.get("holdout")))
            runs.append(run)
            sc = run["score"]
            mark = "PASS" if sc["passed"] else "FAIL"
            print(f"[{len(runs):>3}/{total}] {mark}  {case['id']:<26} steps={sc['steps']} "
                  f"calls={sc['tool_calls']} grounded={sc['grounding']['rate']:.0%} {run['latency_s']}s",
                  flush=True)
            if args.verbose or not sc["passed"]:
                for c in sc["checks"]:
                    if args.verbose or not c["ok"]:
                        print(f"        {'ok ' if c['ok'] else 'XX '} {c['check']}"
                              + (f"  -> {c['detail']}" if c.get("detail") else ""))
                if run.get("error"):
                    print(f"        error: {run['error']}")
                print("        called:", ", ".join(f"{x['tool']}({x['args'].get('ticker')})" for x in run["results"]))
            time.sleep(args.sleep)

    summary = summarise(runs)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "args": vars(args), "runs": runs}, default=str, indent=1),
        encoding="utf-8")
    md = markdown(summary, runs, args)
    (out_dir / f"{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"saved evals/results/{stamp}.json and .md")


if __name__ == "__main__":
    main()
