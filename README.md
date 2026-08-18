# Real-Time Market Research Assistant

A tool-using LLM agent for market questions. Ask in plain language ("what is driving NVDA today?", "BTC-USD vs gold volatility", "has Maruti Suzuki beaten the Indian market?") and the agent picks tools step by step, pulls live data from Yahoo Finance, runs the numbers in Python and answers only from what the tools returned.

Live app: https://agenticresearch.streamlit.app

## How it works

- **Any Yahoo Finance symbol.** Tickers, company names and common names ("crude", "gold", "S&P 500") resolve to symbols, through a built-in alias list first and Yahoo search second. Indian names prefer the NSE listing. One question can cover several assets across the US, India, crypto, commodities and indices.
- **Agent loop.** Each step the LLM reads every result so far and picks the next one to three tool calls, or stops. A finding can trigger the next check: RSI under 30 leads to a backtest of the RSI-dip rule. Limit of 6 steps and 14 calls; repeated calls are refused. Every step is shown in the chat.
- **Numbers come from code.** The LLM decides what to analyse; Python computes it. Free-form analysis runs as model-written pandas code in a subprocess, with up to two repair attempts if it fails.
- **News retrieval.** Headlines are embedded with `all-MiniLM-L6-v2` and retrieved by cosine similarity, across peer tickers for sector questions. Each story carries provider, date and URL.
- **Provider failover.** Groq, Gemini (free tier), NVIDIA, Kilo and opencode, in that order. Providers without a key are skipped; Kilo and opencode work without one.

## Agent tools

| Tool | Returns |
|---|---|
| `get_quote` | Live price and day change |
| `get_history` | OHLCV for a chosen interval |
| `fundamental_facts` | Valuation, margins, growth, beta, analyst targets, dividend yield |
| `technical_facts` | RSI(14), MACD and signal, Bollinger band |
| `indicators` | Leading (RSI, stochastic, rate of change, OBV divergence), lagging (50/200-day MA, MACD), volatility (ATR, Bollinger squeeze), relative strength vs NIFTY 50 or S&P 500 |
| `risk_metrics` | Volatility, VaR 95%, max drawdown, Sharpe |
| `forecast_range` | GARCH(1,1) 70% and 95% price ranges |
| `pair_correlation` | Correlation, beta and covariance between two assets; with `lag`, which one moves first |
| `backtest_rule` | RSI dip, MA cross, breakout or MACD rule after costs, in-sample vs out-of-sample |
| `verdict` | Tested direction for one rule (long, flat or no edge) with volatility ranges |
| `rag_search` | Most relevant headlines for the question |
| `get_ticker_news` | Recent headlines for the ticker |
| `run_analysis` | Model-written Python run on the full data |
| `chart_spec` | Chart settings for the price tab |

## Evidence, not calls

Asked "should I buy Bajaj Finance?", the agent does not say buy or sell. It reports what the indicators read today and tests whether that setup has made money on that stock:

- Long-only daily rules with costs on every entry and exit: STT on both sides, stamp duty on the buy and slippage for NSE/BSE (about 0.3% round trip); spread and slippage for US stocks.
- History is split 70/30 by date and both parts are compared with buy-and-hold and the index over the same dates.
- The result is one of: held up, failed out of sample, no edge after costs, or too few trades.

Limits: fills at the signal day's close, only currently listed stocks, approximate costs. Renamed or demerged companies need their current name.

## Evaluation

`evals/` holds 22 questions (18 development, 4 held out) with checks on which tools the agent must call, whether every number in the answer appears in tool output, and whether it avoids buy/sell advice. Each run uses live market data and live LLMs, so results vary. The figures below are 3 repeats of every case (66 runs) with automatic failover across Groq, Gemini and NVIDIA; most answers came from Nemotron-3-Super-120B.

| Metric | Result |
|---|---|
| Tasks passing all 3 runs (pass^3) | 86% (19/22) |
| Task success per run | 94% (62/66) |
| Held-out task success per run | 100% (12/12) |
| Required tools called | 97% (67/69 checks) |
| Numbers grounded in tool output | 97% |
| Buy/sell advice given | 0 |
| Mean agent steps / tool calls | 3.0 / 2.5 |

```bash
python evals/run_evals.py --repeats 3
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model (about 90 MB). Provider keys are optional and are read from Streamlit secrets, environment variables or a `.env` file next to `app.py`:

```
GROQ_API_KEY=...
NVIDIA_API_KEY=...
GEMINI_API_KEY=...
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run offline with synthetic prices and scripted LLM replies.

Not investment advice. Ranges are estimates from historical volatility and prices can move outside them.
