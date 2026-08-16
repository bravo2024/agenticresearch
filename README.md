# Real-Time Market Intelligence Assistant

**Powered by :::** yfinance · pandas · NumPy · sentence-transformers (RAG) · arch (GARCH) · frontier LLMs (opencode-free · NVIDIA Build · Groq · kilo-free)

An agentic market analyst. Give it a question in plain English — "what's driving NVDA today?", "BTC-USD vs gold volatility?", "crude & Reliance correlation?" — and it takes it from there: resolves every symbol named, pulls live quotes, history, news, and fundamentals from yfinance, runs the numbers, and answers with sources.

## What it does well

- **Any asset, by any name.** The agent resolves what you actually said — tickers, company names, colloquial handles ("crude", "gold", "S&P 500") — and routes to the right symbol automatically. One question can span multiple assets at once, cross-currency and cross-market (US, India, crypto, commodities, indices).
- **Works in steps, like an analyst.** Each turn the model reads every result so far and picks the next tool, or stops. RSI under 30 leads it to backtest the RSI-dip rule; a backtest with too few trades leads it to retry on a longer window. Up to 6 steps, repeated calls refused, and every step is shown in the chat so you can see why it did what it did.
- **Answers from data, not memory.** The model decides the analysis; Python does the arithmetic. Analysis code is executed in a code-interpreter subprocess with the app's live data injected, and a repair loop self-corrects failing scripts. Every figure returned is a real, computed number.
- **Live news + true RAG.** Headlines are embedded with `sentence-transformers` and retrieved by cosine similarity across the news corpus and peer tickers — every story carries provider, date, and URL, so the agent's claims come with receipts.
- **Multilingual.** Ask in Hindi, English, or any language — the analyst answers in the same language, figures and all.
- **Resilient by design.** Four providers — opencode-free, NVIDIA Build, Groq, kilo-free — with automatic cross-provider fallback and per-model retries. Two run keyless; set one optional key and the answer always gets through.

## Tools the agent can call

| Tool | What it returns |
|---|---|
| `get_quote` | Live price and day change, computed from previous close |
| `get_history` | OHLCV over a chosen interval |
| `rag_search` | Semantically retrieved headlines with source and date |
| `get_ticker_news` | Recent headlines relevant to the ticker |
| `technical_facts` | RSI, MACD, Bollinger, SMA/EMA crossovers, momentum flags |
| `fundamental_facts` | Valuation, margins, growth, beta, analyst targets, dividend yield |
| `pair_correlation` | Aligned correlation, beta, and covariance between any two assets; with `lag`, which one moves first |
| `indicators` | Leading (RSI, stochastic, rate of change, OBV divergence), lagging (50/200-day MAs, MACD), volatility (ATR, Bollinger squeeze) and relative strength vs NIFTY 50 / S&P 500 |
| `backtest_rule` | Has this setup actually paid on this stock? RSI dip, MA cross, breakout or MACD rule, after costs, in-sample vs out-of-sample |
| `run_analysis` | Model-written Python, executed in a subprocess with full data |

## Methods, briefly

- **Risk & technical.** Volatility, VaR95, max drawdown, Sharpe, RSI, MACD, Bollinger bands — all computed from live history.
- **Forecasts.** `forecast_range()` fits a GARCH(1,1) variance model with `arch` and returns 70%/95% volatility-based price bands — an honest range, not a coin-flip direction call.
- **Sentiment.** Per-headline LLM scoring with a lexicon fallback, aggregated and plotted.
- **News retrieval.** Cosine top-k over sentence embeddings — sector questions extend the corpus across peer tickers.
- **Correlations.** Deterministic pandas alignment of any two assets, with betas computed in both directions.

## Evidence, not calls

Ask "should I buy Bajaj Finance?" and the analyst won't say buy or sell. It
shows what the leading and lagging indicators read today, then tests whether
that setup has made money on that stock before:

- Long-only daily rules, costs charged on every entry and exit. For NSE/BSE
  that means STT on both sides, stamp duty on the buy and a little slippage
  (about 0.3% round trip); US stocks get spread and slippage only.
- History is split 70/30 by date. The first part is where you'd have found
  the idea, the last part is what it never saw. Both are compared with
  buy-and-hold and with the index over the same dates.
- The verdict is blunt: held up, didn't survive out of sample, no edge after
  costs, or too few trades to say.

Known limits: fills are assumed at the signal day's close, only stocks still
listed can be tested, and the cost table is an approximation (STT changes
with the budget). Renamed or demerged companies need their new name.

Any equity works by name. "Hindustan Aeronautics" resolves to `HAL.NS`
through Yahoo's search, with the NSE listing preferred over BSE.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) for news retrieval.

Set one or more provider keys as environment variables (or cloud secrets) and the chain uses them automatically:

```
OPENCODE_ZEN_API_KEY = "..."
NVIDIA_API_KEY       = "..."
GROQ_API_KEY         = "..."
```

No keys are required — opencode-free and kilo-free run keyless.

---

**Disclaimer.** This is an educational research tool, not investment advice. Forecasts are probabilistic estimates based on historical volatility; markets can move beyond any range. Do your own research before any financial decision.
