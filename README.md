# Real-Time Market Intelligence Assistant

**Powered by :::** yfinance · pandas · NumPy · sentence-transformers (RAG) · arch (GARCH) · frontier LLMs (opencode-free · NVIDIA Build · Groq · kilo-free)

An agentic market analyst. Give it a question in plain English — "what's driving NVDA today?", "BTC-USD vs gold volatility?", "crude & Reliance correlation?" — and it takes it from there: resolves every symbol named, pulls live quotes, history, news, and fundamentals from yfinance, runs the numbers, and answers with sources.

## What it does well

- **Any asset, by any name.** The agent resolves what you actually said — tickers, company names, colloquial handles ("crude", "gold", "S&P 500") — and routes to the right symbol automatically. One question can span multiple assets at once, cross-currency and cross-market (US, India, crypto, commodities, indices).
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
| `pair_correlation` | Aligned correlation, beta, and covariance between any two assets |
| `run_analysis` | Model-written Python, executed in a subprocess with full data |

## Methods, briefly

- **Risk & technical.** Volatility, VaR95, max drawdown, Sharpe, RSI, MACD, Bollinger bands — all computed from live history.
- **Forecasts.** `forecast_range()` fits a GARCH(1,1) variance model with `arch` and returns 70%/95% volatility-based price bands — an honest range, not a coin-flip direction call.
- **Sentiment.** Per-headline LLM scoring with a lexicon fallback, aggregated and plotted.
- **News retrieval.** Cosine top-k over sentence embeddings — sector questions extend the corpus across peer tickers.
- **Correlations.** Deterministic pandas alignment of any two assets, with betas computed in both directions.

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
