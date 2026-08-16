"""The eval set. Each case is a question a real user would type plus what a
correct run must do. Checks target behaviour (right tool, right symbol, no
trade call), not exact wording, so a different but correct answer still
passes.

Categories
  entry      buy/sell/'good time' questions: evidence, never a call
  resolve    company names -> correct NSE/US symbols
  indicators leading / lagging / relative strength
  cross      correlation and lead-lag between markets
  risk       VaR, drawdown, forecast ranges
  news       'why is it moving' questions
  robust     bad symbols, other languages, open-ended requests
"""
from checks import any_of, answer_matches, called, distinct_tickers, never_ticker


def positive(x):
    try:
        return int(x) > 0
    except (TypeError, ValueError):
        return False


positive.__name__ = ">0"


# An honest "there is nothing to analyse" answer, however it is phrased. Kept
# broad on purpose: the substitution check (never_ticker) is what catches an
# agent that admits no data and then analyses something else anyway.
NO_DATA = (r"no( \w+){0,3} (data|information|price|prices|history|quote)|"
           r"not (find|found|available|recogni[sz]ed)|unavailable|could not|couldn.t|"
           r"cannot|can.t|unable|doesn.t (appear|exist)|invalid|unrecogni[sz]ed|no (such|valid)")


def tested(ticker, **rule):
    """Backtest evidence: backtest_rule, or verdict, which runs the same
    backtest plus a significance test (added 2026-10-07; the agent is told to
    prefer it for buy/sell/hold questions)."""
    return any_of(called("backtest_rule", ticker=ticker, **rule),
                  called("verdict", ticker=ticker, **rule))


def state(ticker):
    """Current technical state: the full indicator set or the quick snapshot."""
    return any_of(called("indicators", ticker=ticker), called("technical_facts", ticker=ticker))


CASES = [
    # ── entry: evidence, never a call ─────────────────────────
    {"id": "entry-oversold-bajaj", "category": "entry",
     "question": "Bajaj Finance looks oversold. Should I buy?",
     "checks": [state("BAJFINANCE.NS"),
                tested(ticker="BAJFINANCE.NS"),
                answer_matches(r"out[- ]of[- ]sample|in[- ]sample", "answer reports the split")]},
    {"id": "entry-goodtime-reliance", "category": "entry",
     "question": "Is this a good time to buy Reliance?",
     "checks": [state("RELIANCE.NS"),
                tested(ticker="RELIANCE.NS")]},
    {"id": "entry-sell-infosys", "category": "entry",
     "question": "Should I sell my Infosys shares now?",
     "checks": [state("INFY.NS"),
                tested(ticker="INFY.NS")]},
    {"id": "entry-goldencross-nifty", "category": "entry",
     "question": "Has a 50/200-day golden cross strategy actually worked on the NIFTY 50?",
     "checks": [tested(ticker="^NSEI", rule="ma_cross")]},
    {"id": "entry-breakout-aapl", "category": "entry",
     "question": "Does buying 55-day breakouts work on Apple?",
     "checks": [tested(ticker="AAPL", rule="breakout")]},

    # ── resolve: any equity by name ───────────────────────────
    {"id": "resolve-hal", "category": "resolve",
     "question": "How is Hindustan Aeronautics doing against the NIFTY?",
     "checks": [any_of(called("indicators", ticker="HAL.NS"),
                       called("pair_correlation", ticker="HAL.NS"))]},
    {"id": "resolve-two-banks", "category": "resolve",
     "question": "Bajaj Finance vs Kotak Mahindra Bank: which has stronger momentum right now?",
     "checks": [any_of(called("indicators", ticker="BAJFINANCE.NS"),
                       called("technical_facts", ticker="BAJFINANCE.NS")),
                any_of(called("indicators", ticker="KOTAKBANK.NS"),
                       called("technical_facts", ticker="KOTAKBANK.NS"))]},
    {"id": "resolve-lt", "category": "resolve",
     "question": "What's the trend on Larsen & Toubro?",
     "checks": [any_of(called("indicators", ticker="LT.NS"),
                       called("technical_facts", ticker="LT.NS"))]},

    # ── indicators ────────────────────────────────────────────
    {"id": "ind-hdfc-lead-lag", "category": "indicators",
     "question": "HDFC Bank: what do the leading and lagging indicators say?",
     "checks": [called("indicators", ticker="HDFCBANK.NS"),
                answer_matches(r"lagging", "answer separates lagging"),
                answer_matches(r"leading", "answer separates leading")]},
    {"id": "ind-tcs-relative", "category": "indicators",
     "question": "Is TCS outperforming the market over the last 3 months?",
     "checks": [called("indicators", ticker="TCS.NS")]},

    # ── cross-market ──────────────────────────────────────────
    {"id": "cross-crude-inr", "category": "cross",
     "question": "Does crude oil lead the Indian rupee?",
     "checks": [called("pair_correlation", lag=positive)]},
    {"id": "cross-nvda-amd-beta", "category": "cross",
     "question": "What's the correlation and beta between Nvidia and AMD?",
     "checks": [any_of(called("pair_correlation", ticker="NVDA", ticker_b="AMD"),
                       called("pair_correlation", ticker="AMD", ticker_b="NVDA"))]},

    # ── risk ──────────────────────────────────────────────────
    {"id": "risk-gold-range", "category": "risk",
     "question": "Give me a one-week price range for gold.",
     "checks": [called("forecast_range", ticker="GC=F"),
                answer_matches(r"not investment advice", "forecast disclaimer attached")]},
    {"id": "risk-tcs-var", "category": "risk",
     "question": "What's the VaR and max drawdown of Tata Consultancy Services?",
     "checks": [any_of(called("risk_metrics", ticker="TCS.NS"),
                       called("run_analysis", ticker="TCS.NS"))]},

    # ── news ──────────────────────────────────────────────────
    {"id": "news-nvda", "category": "news",
     "question": "Why is Nvidia moving today?",
     "checks": [any_of(called("rag_search", ticker="NVDA"),
                       called("get_ticker_news", ticker="NVDA"))]},

    # ── robustness ────────────────────────────────────────────
    {"id": "robust-bad-symbol", "category": "robust",
     "question": "Should I buy ZZQX123?",
     "checks": [answer_matches(NO_DATA,
                               "admits the symbol has no data"),
                never_ticker("AAPL")]},           # AAPL is the UI default; using it = answering another question
    {"id": "robust-hindi", "category": "robust",
     "question": "रिलायंस का RSI अभी कितना है?",
     "checks": [any_of(called("indicators", ticker="RELIANCE.NS"),
                       called("technical_facts", ticker="RELIANCE.NS"))]},
    {"id": "robust-ideas-banks", "category": "robust",
     "question": "Suggest some Indian banking stocks worth researching.",
     "checks": [distinct_tickers(3, ".NS")]},

    # ── held out ──────────────────────────────────────────────
    # Written after the prompt fixes from the first run and never used to tune
    # them. If the fixes only worked on the cases they were made for, these
    # show it. Reported separately.
    {"id": "holdout-exit-asianpaints", "category": "entry", "holdout": True,
     "question": "Should I exit my position in Asian Paints?",
     "checks": [tested(ticker="ASIANPAINT.NS")]},
    {"id": "holdout-maruti-vs-market", "category": "indicators", "holdout": True,
     "question": "Has Maruti Suzuki beaten the Indian market over the last 6 months?",
     "checks": [called("indicators", ticker="MARUTI.NS")]},
    {"id": "holdout-tesla-downside", "category": "risk", "holdout": True,
     "question": "How far could Tesla fall over the next week?",
     "checks": [any_of(called("forecast_range", ticker="TSLA"),
                       called("risk_metrics", ticker="TSLA"))]},
    {"id": "holdout-bad-name", "category": "robust", "holdout": True,
     "question": "What's the RSI of Qwertyuiop Industries?",
     "checks": [answer_matches(NO_DATA,
                               "admits the company has no data"),
                never_ticker("AAPL")]},
]
