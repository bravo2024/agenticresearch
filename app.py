"""
Real-Time Market Research Assistant — Agentic Edition
======================================================
Vivek's original style: wide layout, light theme, sidebar-driven tabs,
price chart up front. A complete real-time market research tool: live
market data, RAG retrieval over news, real computed analytics, and an
LLM agent (opencode-free / nvidia-build / groq) that answers with
evidence.

Tabs (all driven by the LEFT SIDEBAR — no per-tab inputs):
  1. 📈 Price Chart     — line or candlestick, MA20/50, volume, auto-refresh
  2. 💬 Ask the Analyst — LLM agent with follow-up memory + suggested questions
  3. 🧭 Sentiment       — every headline scored, daily series, summaries
  4. 🛡 Risk & Technical— vol/VaR/drawdown/Sharpe + RSI/MACD/Bollinger
  5. 📊 Raw Data + News — history table, CSV download, expandable headlines
  6. 📄 Project Overview— architecture, stack, methodology, coverage notes

Portfolio Edition · CPU-only · Streamlit Cloud ready
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — wide layout, light theme (original style)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Market Research Assistant by Vivek Bose",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

VIOLET = "#7B2D8E"
GOLD = "#C9A84C"

st.markdown(f"""
<style>
    .hero {{ text-align: center; padding: 0.8rem 0 0.2rem 0; }}
    .hero h1 {{ color: {VIOLET}; font-size: 2.1rem; margin: 0; }}
    .hero p {{ color: #555; font-size: 1rem; margin: 0.2rem 0 0 0; }}
    .card {{ background: #f8f6fb; border: 1px solid #e4ddf0; border-radius: 8px;
            padding: 0.8rem 1rem; margin: 0.3rem 0; }}
    .metric-big {{ font-size: 1.6rem; font-weight: 700; color: {VIOLET}; }}
    .metric-label {{ color: #777; font-size: 0.78rem; text-transform: uppercase;
                    letter-spacing: 0.05em; }}
    .tag {{ display: inline-block; background: {VIOLET}; color: white; border-radius: 4px;
           padding: 0.12rem 0.5rem; font-size: 0.75rem; font-weight: 600; }}
    .tag-gold {{ background: {GOLD}; color: #222; }}
    .chat-user {{ background: {VIOLET}; color: white; border-radius: 10px;
                 padding: 0.55rem 0.9rem; margin: 0.3rem 0; max-width: 85%; }}
    .chat-agent {{ background: #f8f6fb; border: 1px solid #e4ddf0; color: #222;
                  border-radius: 10px; padding: 0.55rem 0.9rem; margin: 0.3rem 0; }}
    a.src-link {{ color: {VIOLET}; text-decoration: none; font-weight: 600; }}
    a.src-link:hover {{ text-decoration: underline; }}
    [data-testid="stAppViewContainer"] .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
    [data-testid="stSidebar"] button {{ font-size: 0.82rem; padding: 2px 6px; }}
    [data-testid="stSidebar"] button p {{ font-size: 0.82rem; line-height: 1.2rem; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LLM PROVIDERS (multi-provider inference with fallback)
# ─────────────────────────────────────────────────────────────
ZEN_BASE = "https://opencode.ai/zen/v1"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"

LLM_PROVIDERS = {
    "opencode-free": {
        "base": ZEN_BASE,
        "key_env": "OPENCODE_ZEN_API_KEY",
        "max_tokens": 4096,
        "models": [
            "deepseek-v4-flash-free",
            "mimo-v2.5-free",
            "ling-3.0-flash-free",
            "nemotron-3-ultra-free",
        ],
        "tag": "free",
        "anonymous": True,  # blank key works — free proxy
    },
    "nvidia-build": {
        "base": NVIDIA_BASE,
        "key_env": "NVIDIA_API_KEY",
        "max_tokens": 8192,
        "models": [
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-v4-pro",
            "z-ai/glm-5.2",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "minimaxai/minimax-m3",
        ],
        "tag": "nvidia",
    },
    "groq": {
        "base": GROQ_BASE,
        "key_env": "GROQ_API_KEY",
        "max_tokens": 32768,
        "models": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3.6-27b",
        ],
        "tag": "groq",
    },
    "kilo-free": {
        "base": "https://api.kilo.ai/api/openrouter",
        "key_env": "KILO_API_KEY",
        "max_tokens": 8192,
        "models": [
            "kilo-auto/free",
            "kilo-auto/efficient",
            "kilo-auto/balanced",
            "kilo-auto/frontier",
            "stepfun/step-3.7-flash:free",
            "inclusionai/ling-3.0-flash:free",
            "poolside/laguna-s-2.1:free",
            "cohere/north-mini-code:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ],
        "tag": "kilo-free",
        "anonymous": True,  # blank key works — public gateway
        "kilo_headers": True,  # needs the Kilo Code extension referer headers
    },
}

TICKERS = [
    "AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "NFLX", "NVDA", "PYPL", "INTC",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS",
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "SOL-USD",
    "^NSEI", "^NSEBANK", "^BSESN", "^DJI", "^GSPC", "^IXIC", "^FTSE", "^N225",
    "GC=F", "CL=F", "SI=F", "NG=F", "HG=F",
]

WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "BTC-USD", "RELIANCE.NS", "^NSEI", "GC=F"]

# Company names for news relevance matching (ticker -> aliases)
COMPANY_NAMES = {
    "AAPL": ["apple", "tim cook", "iphone", "macbook"],
    "MSFT": ["microsoft", "windows", "azure", "satya"],
    "GOOG": ["alphabet", "google", "chrome", "pixel", "deepmind"],
    "TSLA": ["tesla", "elon", "cybertruck", "model 3", "model y"],
    "AMZN": ["amazon", "aws", "bezos", "prime video"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "zuckerberg"],
    "NFLX": ["netflix", "streaming"],
    "NVDA": ["nvidia", "jensen", "cuda", "hopper", "blackwell", "rtx"],
    "PYPL": ["paypal", "venmo"],
    "INTC": ["intel"],
    "RELIANCE.NS": ["reliance", "mukesh ambani", "jio"],
    "TCS.NS": ["tata consultancy", "tcs "],
    "INFY.NS": ["infosys"],
    "HDFCBANK.NS": ["hdfc bank", "hdfcbank"],
    "ICICIBANK.NS": ["icici"],
    "LT.NS": ["larsen", "l&t "],
    "SBIN.NS": ["state bank of india", "sbi"],
    "BTC-USD": ["bitcoin", "btc", "crypto"],
    "ETH-USD": ["ethereum", "eth"],
    "BNB-USD": ["bnb", "binance"],
    "ADA-USD": ["cardano", "ada"],
    "XRP-USD": ["xrp", "ripple"],
    "SOL-USD": ["solana", "sol"],
    "^NSEI": ["nifty"],
    "^NSEBANK": ["bank nifty", "nifty bank"],
    "^BSESN": ["sensex", "bse"],
    "^DJI": ["dow jones", "dow "],
    "^GSPC": ["s&p 500", "sp 500", "s&p", "spx", "spx500", "sp500", "s&p500"],
    "^IXIC": ["nasdaq composite", "nasdaq"],
    "^FTSE": ["ftse"],
    "^N225": ["nikkei"],
    "GC=F": ["gold", "bullion"],
    "CL=F": ["crude", "oil", "wti", "brent"],
    "SI=F": ["silver"],
    "NG=F": ["natural gas"],
    "HG=F": ["copper"],
}


def resolve_tickers(text: str) -> list[str]:
    """Return EVERY ticker mentioned in the text. Known symbols match as an
    exact substring (longest first); company/asset aliases from COMPANY_NAMES
    match as whole words (so "oil" never matches "boil"). Unknown symbols
    (e.g. AMD) are left unresolved here — the router may still use them."""
    upper = text.upper()
    found = []
    for t in sorted(TICKERS, key=len, reverse=True):
        if t in upper and t not in found:
            found.append(t)
    for t, aliases in COMPANY_NAMES.items():
        if t in found:
            continue
        for a in aliases:
            pat = re.compile(r"\b" + re.escape(a.strip().upper()) + r"\b")
            if pat.search(upper):
                found.append(t)
                break
    return found


def news_relevant(ticker: str, headline: str, summary: str = "") -> bool:
    """True if a headline plausibly concerns the ticker (symbol or company names)."""
    text = (headline + " " + (summary or "")).lower()
    ticker_base = ticker.split(".")[0].split("=")[0].replace("^", "").lower()
    if ticker_base and ticker_base in text:
        return True
    aliases = COMPANY_NAMES.get(ticker, [])
    return any(a in text for a in aliases)

PERIOD_FOR_INTERVAL = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "90m": "60d",
    "60m": "730d", "1h": "730d",
    "1d": "2y", "5d": "2y", "1wk": "5y", "1mo": "10y", "3mo": "10y",
}

REPORT_DIR = Path(__file__).parent / "saved_reports"


def get_key(env_name: str) -> str:
    try:
        key = st.secrets.get(env_name, "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get(env_name, "")
    if not key:
        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{env_name}="):
                    key = line.split("=", 1)[1].strip()
    return key


def _provider_headers(cfg: dict, key: str) -> dict:
    """Request headers for a provider; kilo-free (public gateway) needs the
    referer headers the Kilo Code extension sends."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if cfg.get("kilo_headers"):
        headers.update({"HTTP-Referer": "https://kilo.ai/", "X-Title": "Kilo Code"})
    return headers


def llm_chat(messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 0, max_retries: int = 3,
             provider: str = "opencode-free", model: str = "") -> dict:
    """Call the selected provider; on failure, fall back across providers."""
    order = []
    prov_cfg = LLM_PROVIDERS.get(provider)
    if prov_cfg:
        models = [model] if model and model in prov_cfg["models"] else prov_cfg["models"]
        order.append((provider, prov_cfg, models))
    seen = {provider}
    for pname in ["opencode-free", "nvidia-build", "groq", "kilo-free"]:
        if pname in seen or pname not in LLM_PROVIDERS:
            continue
        seen.add(pname)
        order.append((pname, LLM_PROVIDERS[pname], LLM_PROVIDERS[pname]["models"]))

    errors = []
    for pname, cfg, models in order:
        key = "" if cfg.get("anonymous") else get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            errors.append(f"{pname}: no key")
            continue
        mt = max_tokens if max_tokens > 0 else cfg.get("max_tokens", 4096)
        for m in models:
            for _ in range(max_retries):
                try:
                    r = requests.post(
                        f"{cfg['base']}/chat/completions",
                        headers=_provider_headers(cfg, key),
                        json={
                            "model": m,
                            "messages": messages,
                            "max_tokens": mt,
                            "temperature": temperature,
                        },
                        timeout=60,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        content = data["choices"][0]["message"].get("content") or ""
                        if not content.strip():
                            errors.append(f"{pname}/{m}: empty content")
                            continue
                        return {
                            "ok": True,
                            "provider": pname,
                            "model": m,
                            "text": content,
                            "tokens": data.get("usage", {}).get("total_tokens", 0),
                        }
                    errors.append(f"{pname}/{m}: HTTP {r.status_code}")
                except Exception as e:
                    errors.append(f"{pname}/{m}: {str(e)[:60]}")
            time.sleep(1)

    return {"ok": False, "error": "; ".join(errors[-8:])}


def llm_chat_stream(messages: list[dict], temperature: float = 0.2,
                    max_tokens: int = 0,
                    provider: str = "opencode-free", model: str = ""):
    """Generator: stream a completion from the provider fallback chain.

    Yields text chunks as they arrive (OpenAI-compatible SSE). If the chosen
    provider/model fails or returns empty, silently advances to the next in
    the chain. Records the winning provider/model in session_state so the
    caller can display it.
    """
    order = []
    prov_cfg = LLM_PROVIDERS.get(provider)
    if prov_cfg:
        models = [model] if model and model in prov_cfg["models"] else prov_cfg["models"]
        order.append((provider, prov_cfg, models))
    seen = {provider}
    for pname in ["opencode-free", "nvidia-build", "groq", "kilo-free"]:
        if pname in seen or pname not in LLM_PROVIDERS:
            continue
        seen.add(pname)
        order.append((pname, LLM_PROVIDERS[pname], LLM_PROVIDERS[pname]["models"]))

    errors = []
    for pname, cfg, models in order:
        key = "" if cfg.get("anonymous") else get_key(cfg["key_env"])
        if not cfg.get("anonymous") and not key:
            errors.append(f"{pname}: no key")
            continue
        mt = max_tokens if max_tokens > 0 else cfg.get("max_tokens", 4096)
        for m in models:
            try:
                r = requests.post(
                    f"{cfg['base']}/chat/completions",
                    headers=_provider_headers(cfg, key),
                    json={
                        "model": m,
                        "messages": messages,
                        "max_tokens": mt,
                        "temperature": temperature,
                        "stream": True,
                    },
                    timeout=120,
                    stream=True,
                )
                if r.status_code != 200:
                    errors.append(f"{pname}/{m}: HTTP {r.status_code}")
                    continue
                chunks = []
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0].get("delta", {}) or {}
                        piece = delta.get("content")
                        if piece:
                            chunks.append(piece)
                            yield piece
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                text = "".join(chunks)
                if text.strip():
                    st.session_state["_llm_provider"] = pname
                    st.session_state["_llm_model"] = m
                    return
                errors.append(f"{pname}/{m}: empty stream")
            except Exception as e:
                errors.append(f"{pname}/{m}: {str(e)[:60]}")
            time.sleep(1)

    yield f"\n\n_LLM unavailable: {'; '.join(errors[-4:])}_"


# ─────────────────────────────────────────────────────────────
# DATA LAYER (yfinance, cached)
# ─────────────────────────────────────────────────────────────
def interval_period(interval: str) -> str:
    return PERIOD_FOR_INTERVAL.get(interval, "2y")


@st.cache_data(ttl=300, show_spinner=False)
def get_history(ticker: str, interval: str = "1d", period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(ticker: str) -> dict:
    """Live quote with correct fast_info keys; day change computed from prev close."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = info.get("lastPrice") or info.get("last_price")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        day_change_pct = None
        if price and prev_close:
            day_change_pct = (price - prev_close) / prev_close * 100
        if price is None:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return {
            "ticker": ticker,
            "price": price,
            "day_change_pct": round(day_change_pct, 2) if day_change_pct is not None else None,
            "volume": info.get("lastVolume") or info.get("last_volume"),
            "market_cap": info.get("marketCap") or info.get("market_cap"),
            "prev_close": prev_close,
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
        }
    except Exception as e:
        return {"ticker": ticker, "price": None, "error": str(e)[:120]}


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_facts(ticker: str) -> dict:
    """Curated fundamentals from yfinance `.info`: valuation, margins, leverage,
    dividend, beta and analyst targets. Cached 1h."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"ticker": ticker, "error": str(e)[:120]}

    def num(k):
        v = info.get(k)
        return float(v) if isinstance(v, (int, float)) else v

    raw = {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency"),
        "pe_trailing": num("trailingPE") or num("forwardPE"),
        "pe_forward": num("forwardPE"),
        "eps": num("trailingEps") or num("forwardEps"),
        "price_to_book": num("priceToBook"),
        "price_to_sales": num("priceToSalesTrailing12Months"),
        "revenue": num("totalRevenue"),
        "revenue_growth_pct": num("revenueGrowth") * 100
            if isinstance(info.get("revenueGrowth"), (int, float)) else None,
        "gross_margin_pct": num("grossMargins") * 100
            if isinstance(info.get("grossMargins"), (int, float)) else None,
        "net_margin_pct": num("profitMargins") * 100
            if isinstance(info.get("profitMargins"), (int, float)) else None,
        "beta": num("beta"),
        "debt_to_equity": num("debtToEquity"),
        "current_ratio": num("currentRatio"),
        "dividend_yield_pct": num("dividendRate") / num("regularMarketPrice") * 100
            if isinstance(info.get("dividendRate"), (int, float))
            and isinstance(info.get("regularMarketPrice"), (int, float))
            else (num("dividendYield")
                  if isinstance(info.get("dividendYield"), (int, float)) else None),
        "payout_ratio": num("payoutRatio"),
        "fifty_two_week_high": num("fiftyTwoWeekHigh"),
        "fifty_two_week_low": num("fiftyTwoWeekLow"),
        "shares_outstanding": num("sharesOutstanding") or num("impliedSharesOutstanding"),
        "analyst_recommendation": info.get("recommendationKey"),
        "analyst_target_mean": num("targetMeanPrice"),
        "analyst_target_high": num("targetHighPrice"),
        "analyst_target_low": num("targetLowPrice"),
        "earnings_quarterly_growth_pct": num("earningsQuarterlyGrowth") * 100
            if isinstance(info.get("earningsQuarterlyGrowth"), (int, float)) else None,
    }
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in raw.items() if v is not None}


@st.cache_data(ttl=3600, show_spinner=False)
def get_calendar(ticker: str) -> dict:
    cal = yf.Ticker(ticker).calendar
    if not cal:
        return {}
    out = {}
    for k in ("Earnings Date", "Dividend Date", "Ex-Dividend Date"):
        v = cal.get(k)
        if isinstance(v, (list, tuple)):
            v = str(v[0]) if v else None
        out[k] = str(v) if v else None
    return out


@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker: str, limit: int = 12) -> list[dict]:
    news = yf.Ticker(ticker).news or []
    items = []
    for n in news[:limit]:
        c = n.get("content", {})
        items.append({
            "title": c.get("title", ""),
            "summary": c.get("summary", "") or c.get("description", ""),
            "date": c.get("pubDate", ""),
            "provider": (c.get("provider") or {}).get("displayName", "")
                       if isinstance(c.get("provider"), dict) else c.get("provider", ""),
            "url": (c.get("canonicalUrl") or {}).get("url", "")
                   if isinstance(c.get("canonicalUrl"), dict) else "",
        })
    return items


@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_news(ticker: str, limit: int = 15) -> list[dict]:
    """Ticker-relevant news only — yfinance's feed is market-wide, so we
    filter by symbol / company-name aliases. Returns the filtered set when
    at least one headline matches; falls back to top stories otherwise
    (flagged so the UI/agent can be honest about coverage)."""
    all_news = get_news(ticker, limit=limit)
    relevant = [n for n in all_news if news_relevant(ticker, n["title"], n.get("summary", ""))]
    if relevant:
        for n in relevant:
            n["_filtered"] = True
        return relevant
    # Fallback: market-wide top stories, clearly labelled
    for n in all_news[:5]:
        n["_filtered"] = False
    return all_news[:5]


def risk_metrics(ticker: str) -> dict:
    df = get_history(ticker, "1d", "1y")
    if df.empty or len(df) < 30:
        return {"error": "not enough data"}
    ret = df["Close"].pct_change().dropna()
    daily_vol = ret.std()
    annual_vol = daily_vol * np.sqrt(252)
    var95 = -np.percentile(ret, 5)
    cum = (1 + ret).cumprod()
    dd = (cum / cum.cummax()) - 1
    max_dd = dd.min()
    sharpe = (ret.mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0
    return {
        "daily_vol_pct": round(daily_vol * 100, 3),
        "annual_vol_pct": round(annual_vol * 100, 1),
        "var95_1d_pct": round(var95 * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe_annual": round(sharpe, 2),
        "days": int(len(ret)),
    }


def technical_facts(ticker: str) -> dict:
    df = get_history(ticker, "1d", "6mo")
    if df.empty or len(df) < 30:
        return {"error": "not enough data"}
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 0))
    rsi = rsi.where(loss > 0, 100.0).where(gain > 0, 0.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    last = close.iloc[-1]
    return {
        "price": round(float(last), 2),
        "rsi14": round(float(rsi.iloc[-1]), 1),
        "macd": round(float(macd.iloc[-1]), 3),
        "macd_signal": round(float(signal.iloc[-1]), 3),
        "bollinger_upper": round(float(upper.iloc[-1]), 2),
        "bollinger_lower": round(float(lower.iloc[-1]), 2),
        "in_bollinger_band": bool(lower.iloc[-1] <= last <= upper.iloc[-1]),
    }


def forecast_range(ticker: str, horizon_days: int = 7) -> dict:
    """Volatility-based probabilistic price range (GARCH + drift), NOT a
    directional prediction. Uses the arch package (CPU, ~1s)."""
    df = get_history(ticker, "1d", "1y")
    if df.empty or len(df) < 60:
        return {"error": "not enough history for a volatility estimate"}
    close = df["Close"].dropna()
    ret = close.pct_change().dropna() * 100
    last = float(close.iloc[-1])

    try:
        from arch import arch_model
        m = arch_model(ret, vol="GARCH", p=1, q=1, mean="AR", lags=1, rescale=True)
        fit = m.fit(disp="off")
        fc = fit.forecast(horizon=horizon_days, reindex=False)
        # GARCH variance mean-reverts, so it is forecast one step at a time
        # (h.1 ... h.N). The horizon variance is the sum of that path; taking
        # the first step and scaling by sqrt(h) assumes flat vol and throws
        # away the term structure the model was fitted to produce.
        # rescale=True may have fitted on a scaled series, so undo that.
        step_var = fc.variance.iloc[0].to_numpy() / (fit.scale ** 2)
        step_mean = fc.mean.iloc[0].to_numpy() / fit.scale
        daily_vol_pct = float(np.sqrt(step_var.mean()))
        horizon_vol = float(np.sqrt(step_var.sum()))
        # Drift is read off the forecast mean path. Looking it up by the name
        # "AR[1]" never worked: arch names the lag after the series (Close[1]).
        drift = float(step_mean.sum())
    except Exception:
        # Fallback: historical vol, zero drift. A random walk is the honest
        # default when the GARCH fit fails.
        daily_vol_pct = float(ret.std())
        horizon_vol = daily_vol_pct * np.sqrt(horizon_days)
        drift = 0.0

    drift_pct = drift / horizon_days if horizon_days else 0.0
    center = last * (1 + drift / 100)
    r70 = last * (1 + (drift - 1.04 * horizon_vol) / 100), last * (1 + (drift + 1.04 * horizon_vol) / 100)
    r95 = last * (1 + (drift - 1.96 * horizon_vol) / 100), last * (1 + (drift + 1.96 * horizon_vol) / 100)
    return {
        "ticker": ticker,
        "last_price": round(last, 2),
        "horizon_days": horizon_days,
        "daily_vol_pct": round(daily_vol_pct, 2),
        "drift_pct": round(drift_pct, 3),
        "center_est": round(center, 2),
        "range_70": [round(r70[0], 2), round(r70[1], 2)],
        "range_95": [round(r95[0], 2), round(r95[1], 2)],
        "method": "GARCH(1,1)-AR(1) volatility projection over the full forecast variance path — quantifies uncertainty around the current price, it is not a directional call",
    }


# Code-execution subprocess: the LLM writes the analysis, Python runs the real math.
CODE_INTERPRETER_HEADER = textwrap.dedent("""
    import warnings
    warnings.filterwarnings("ignore")
    import json
    import yfinance as yf
    import pandas as pd
    import numpy as np

    def load(ticker, period="1y", interval="1d"):
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df

    def load_many(tickers, period="1y", interval="1d"):
        closes = {}
        for t in tickers:
            df = load(t, period, interval)
            if not df.empty:
                closes[t] = df["Close"].rename(t)
        out = pd.concat(closes, axis=1).dropna()
        return out
    """)

CODE_INTERPRETER_PROMPT = (
    "You are a quantitative analyst. Write a short Python script that answers the "
    "user's question about the given ticker. Rules:\n"
    "1. Use pandas, numpy, yfinance (already imported: yf, pd, np, and a "
    "`load(ticker, period, interval)` helper).\n"
    "2. A variable BUNDLE (dict) is pre-loaded with the app's FULL computed data: "
    "quote, complete OHLCV history (as dataframe + list), ticker-filtered news, "
    "calendar, risk metrics, technical facts, and fundamentals. Prefer BUNDLE data "
    "when it answers the question; call load() only for periods/intervals BUNDLE "
    "does not cover (e.g. intraday).\n"
    "2b. You may query yfinance directly for anything not in BUNDLE: "
    "yf.Ticker(ticker).info (P/E, EPS, revenue, margins...), .get_income_stmt(), "
    ".get_balance_sheet(), .get_cashflow(), .get_analyst_price_targets(), "
    ".get_recommendations(), .get_holders(), .option_chain().\n"
    "2c. For multi-ticker comparisons use the provided `load_many(tickers)` "
    "helper — it returns a wide Close DataFrame (one column per ticker, aligned "
    "by date). NEVER call yf.download([...]) with multiple tickers: it returns "
    "MultiIndex columns and breaks indexing.\n"
    "2d. NEVER test a DataFrame or Series for truthiness (e.g. `if df:`, "
    "`if corr:`, `if not df[col]:`). That raises 'The truth value of a DataFrame "
    "is ambiguous'. Check emptiness with `df.empty`, `df[col].isna().any()`, "
    "`.any()` or `.all()` instead.\n"
    "2e. BUNDLE['history'] is a LIST OF DICTS (not a DataFrame). Build your frame "
    "with `pd.DataFrame(BUNDLE['history'])` and parse the date column with "
    "pd.to_datetime(...).\n"
    "3. Compute real numbers and print them with print().\n"
    "4. Keep it under 45 lines. Return the script in FULL — never truncate it "
    "mid-way. Output ONLY the Python code — no markdown fences, no explanation.\n"
    "5. Never print the whole dataframe; print metrics, not data dumps.\n"
    "6. If the computation needs a package that may not exist, use numpy/pandas "
    "only (they are guaranteed available).\n"
)


@st.cache_data(ttl=1800, show_spinner=False)
def pair_correlation(ticker_a: str, ticker_b: str, period: str = "1y") -> dict:
    """Deterministic two-ticker link metrics computed in-app with pandas (no
    LLM-written code): aligned-close correlation, both beta directions, and
    current prices. Never fails the way LLM code can."""
    try:
        a = get_history(ticker_a, "1d", period)
        b = get_history(ticker_b, "1d", period)
    except Exception as e:
        return {"ticker_a": ticker_a, "ticker_b": ticker_b, "error": str(e)[:150]}
    if a.empty or b.empty:
        return {"ticker_a": ticker_a, "ticker_b": ticker_b,
                "error": "not enough history for one of the tickers"}
    ra = a["Close"].pct_change().dropna()
    rb = b["Close"].pct_change().dropna()
    df = pd.concat({"a": ra, "b": rb}, axis=1).dropna()
    if len(df) < 30:
        return {"ticker_a": ticker_a, "ticker_b": ticker_b,
                "error": f"only {len(df)} overlapping trading days: not enough to correlate"}
    corr = df["a"].corr(df["b"])
    var_b = df["b"].var()
    var_a = df["a"].var()
    beta_a_on_b = (df["a"].cov(df["b"]) / var_b) if var_b else None
    beta_b_on_a = (df["b"].cov(df["a"]) / var_a) if var_a else None
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "correlation": round(float(corr), 3),
        "covariance": round(float(df["a"].cov(df["b"])), 6),
        "beta_of_a_wrt_b": round(float(beta_a_on_b), 3) if beta_a_on_b else None,
        "beta_of_b_wrt_a": round(float(beta_b_on_a), 3) if beta_b_on_a else None,
        "n_overlap_days": int(len(df)),
        "period": period,
        "a": {
            "last_close": round(float(a["Close"].iloc[-1]), 2),
            "last_day_change_pct": round(float(df["a"].iloc[-1] * 100), 3),
        },
        "b": {
            "last_close": round(float(b["Close"].iloc[-1]), 2),
            "last_day_change_pct": round(float(df["b"].iloc[-1] * 100), 3),
        },
        "method": "deterministic pandas computation on aligned daily closes (no code interpreter)",
    }


def data_bundle(ticker: str) -> dict:
    """Everything the app computed for this ticker, as one JSON-serializable dict.
    This is the agent's FULL data access — quote, history, news, calendar, risk,
    technicals, fundamentals — not a summary."""
    bundle = {
        "ticker": ticker,
        "quote": get_quote(ticker),
        "calendar": get_calendar(ticker),
    }
    try:
        df = get_history(ticker, "1d", "1y")
        if not df.empty:
            bundle["history"] = df.reset_index().to_dict("records")
            bundle["history_summary"] = {
                "rows": int(len(df)),
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
                "columns": list(df.columns),
                "last_close": float(df["Close"].iloc[-1]),
                "last_volume": float(df["Volume"].iloc[-1]) if "Volume" in df else None,
            }
    except Exception as e:
        bundle["history_error"] = str(e)[:200]
    try:
        bundle["news"] = get_ticker_news(ticker, limit=15)
    except Exception as e:
        bundle["news_error"] = str(e)[:200]
    try:
        rm = risk_metrics(ticker)
        if "error" not in rm:
            bundle["risk_metrics"] = rm
    except Exception:
        pass
    try:
        tf = technical_facts(ticker)
        if "error" not in tf:
            bundle["technical_facts"] = tf
    except Exception:
        pass
    try:
        ff = fundamental_facts(ticker)
        if "error" not in ff:
            bundle["fundamental_facts"] = ff
    except Exception:
        pass
    return bundle


def run_analysis_code(code: str, bundle: dict | None = None, timeout: int = 120) -> dict:
    """Execute LLM-written analysis code in a code-interpreter subprocess.

    If a data bundle is provided, it is injected as BUNDLE so the script has
    the app's full computed data without re-fetching.
    Note: executed locally as the app user with normal filesystem/network access — do not deploy unauthenticated."""
    parts = [CODE_INTERPRETER_HEADER]
    if bundle:
        # Embed the bundle as a JSON string literal and json.loads it at runtime.
        # No text substitution, so "true"/"false"/"null" inside strings are never touched.
        parts.append(f"BUNDLE = json.loads({json.dumps(json.dumps(bundle, default=str))})")
    parts.append(code)
    full = "\n".join(parts)
    try:
        compile(full, "<generated>", "exec")
    except SyntaxError as e:
        return {"ok": False, "stdout": "", "stderr": f"SyntaxError: {e}", "exit_code": 1}
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        script_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=tempfile.gettempdir(), env=env,
        )
        out = (proc.stdout or "")[-6000:]
        err = (proc.stderr or "")[-2000:]
        return {
            "ok": proc.returncode == 0,
            "stdout": out.strip(),
            "stderr": err.strip(),
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)[:500], "exit_code": -2}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def agent_analysis(question: str, ticker: str,
                   provider: str = "opencode-free", model: str = "",
                   tickers: list[str] | None = None) -> dict:
    """Tool: LLM writes analysis code → subprocess runs it with the FULL data
    bundle injected → real output returned.

    This is the 'Code Interpreter' pattern: the model decides the analysis,
    Python does the arithmetic, nothing is hallucinated. If the first script
    fails, the error is fed back to the LLM for one repair attempt.
    tickers: full list of symbols the question asks about (primary first);
    the script can use load_many(tickers) to compare them (never yf.download).
    Note: executed locally as the app user with normal filesystem/network access — do not deploy unauthenticated."""
    if not ticker:
        return {"error": "no ticker detected in the question"}
    bundle = data_bundle(ticker)
    ctx_tickers = tickers or [ticker]
    prompt = (
        f"Ticker: {ticker}\n"
        f"Question: {question}\n\n"
        f"Relevant ticker(s) for this question (primary first): {', '.join(ctx_tickers)}\n"
        f"Use load_many({ctx_tickers}) to pull aligned Close prices for ALL of them "
        f"when the question compares or correlates. Do not use yf.download([...]).\n\n"
        f"BUNDLE keys available: {list(bundle.keys())}\n"
        f"History rows: {bundle.get('history_summary', {}).get('rows', 0)} "
        f"({bundle.get('history_summary', {}).get('start')} → {bundle.get('history_summary', {}).get('end')})\n\n"
        "Write the Python script now."
    )
    resp = llm_chat([
        {"role": "system", "content": CODE_INTERPRETER_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=0.0, max_tokens=1600, provider=provider, model=model)
    if not resp.get("ok") or not resp.get("text"):
        return {"error": resp.get("error", "code generation failed")}

    code = resp["text"].strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)

    result = run_analysis_code(code, bundle=bundle)

    # Repair loop: if execution failed, send the error back once and re-run.
    repairs = 0
    while not result.get("ok") and result.get("exit_code") not in (-1, -2) and repairs < 2:
        repairs += 1
        fix_hint = ""
        err = (result.get("stderr") or "").lower()
        if "ambiguous" in err or "truth value" in err or "dataframe" in err:
            fix_hint = ("HINT: you used a DataFrame/Series as a boolean. Never write "
                        "`if df:`/`if corr:`/`if not df[col]:` — use `df.empty`, "
                        "`.isna().any()`, `.any()`/`.all()`. Multi-ticker data must "
                        "go through load_many(tickers), never yf.download([...]).\n")
        if "keyerror" in err or "level" in err or "reindex" in err:
            fix_hint = ("HINT: column/index mismatch. Prefer load_many(tickers) for "
                        "aligned data, build frames from BUNDLE['history'] with "
                        "pd.DataFrame(...) + pd.to_datetime, and align with "
                        "`.reindex()`/`.dropna()` on the date index.\n")
        if "syntaxerror" in err or "unterminated" in err or "invalid" in err or "unexpected" in err:
            fix_hint = ("HINT: your script has a Python syntax error (unterminated "
                        "string, unbalanced quotes/parentheses, stray backslash). "
                        "Return the COMPLETE corrected script — every string "
                        "closed, all quotes balanced, nothing truncated.\n")
        fix_prompt = (
            f"Your previous script failed with this error:\n"
            f"--- stderr ---\n{(result.get('stderr') or '')[:1000]}\n---\n\n"
            f"{fix_hint}"
            f"Fix the bug and return the COMPLETE corrected script only "
            f"(no markdown fences, no explanation). Original task:\n"
            f"Ticker: {ticker}\nQuestion: {question}\n"
            f"Relevant ticker(s): {', '.join(ctx_tickers)}"
        )
        fix = llm_chat([
            {"role": "system", "content": CODE_INTERPRETER_PROMPT},
            {"role": "user", "content": fix_prompt},
        ], temperature=0.0, max_tokens=1600, provider=provider, model=model)
        if not fix.get("ok") or not fix.get("text"):
            break
        code = fix["text"].strip()
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        result = run_analysis_code(code, bundle=bundle)

    return {
        "question": question,
        "ticker": ticker,
        "code": code[:1500],
        "execution": result,
        "repairs": repairs,
        "bundle_keys": list(bundle.keys()),
        "note": "Numbers above were computed by Python execution, not by the LLM.",
    }


# ─────────────────────────────────────────────────────────────
# RAG LAYER — semantic retrieval over the yfinance news corpus
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(ttl=1800, show_spinner=False)
def rag_corpus(ticker: str, extra_tickers: tuple = ()) -> list[dict]:
    """Build a retrievable news corpus: the ticker's filtered headlines +
    summaries + market context, optionally extended with peer tickers
    (yfinance caps ~10 news items per symbol, so multi-ticker corpora give
    semantic retrieval real material). Each doc carries its date, provider
    and URL so the LLM can cite it."""
    docs = []

    def add_docs(symbol: str):
        news = get_ticker_news(symbol, limit=15)
        for n in news:
            text = f"{n['title']}. {n.get('summary', '')}".strip()
            docs.append({
                "text": f"[{symbol}] {text}"[:800],
                "title": n["title"],
                "summary": n.get("summary", "")[:400],
                "date": n["date"][:10],
                "provider": n["provider"],
                "url": n["url"],
                "ticker": symbol,
            })
        df = get_history(symbol, "1d", "6mo")
        if not df.empty:
            docs.append({
                "text": (
                    f"[{symbol}] price context: last close {df['Close'].iloc[-1]:.2f}, "
                    f"6-month high {df['High'].max():.2f}, low {df['Low'].min():.2f}, "
                    f"avg volume {df['Volume'].mean():,.0f}."
                ),
                "title": f"{symbol} price context (computed)",
                "summary": "", "date": str(df.index[-1].date()),
                "provider": "yfinance history", "url": "", "ticker": symbol,
            })

    add_docs(ticker)
    for extra in extra_tickers:
        if extra and extra != ticker:
            try:
                add_docs(extra)
            except Exception:
                continue
    return docs


def embed_texts(texts: list[str]):
    return get_embedder().encode(texts, normalize_embeddings=True)


def rag_retrieve(question: str, ticker: str, top_k: int = 5,
                 extra_tickers: tuple = ()) -> list[dict]:
    """Semantic retrieval: embed the question, cosine-match against the
    news corpus (optionally extended with peers), return top-k stories."""
    docs = rag_corpus(ticker, tuple(extra_tickers or ()))
    if not docs:
        return []
    q_vec = embed_texts([question])[0]
    doc_vecs = embed_texts([d["text"] for d in docs])
    scores = doc_vecs @ q_vec  # normalized embeddings → cosine sim
    order = np.argsort(scores)[::-1][:top_k]
    hits = []
    for i in order:
        hits.append({**docs[i], "score": round(float(scores[i]), 4)})
    return hits


def rag_search(question: str, ticker: str, top_k: int = 5,
               extra_tickers: tuple = ()) -> dict:
    """Tool: semantically retrieve the most relevant stories for a question.
    This is true RAG — embeddings + cosine retrieval over the news corpus,
    not just the latest headlines. Pass extra_tickers to widen the corpus
    (e.g. peers for comparison questions)."""
    if not ticker:
        return {"error": "no ticker"}
    hits = rag_retrieve(question, ticker, top_k, tuple(extra_tickers or ()))
    if not hits:
        return {"error": "no corpus for ticker"}
    return {
        "query": question,
        "ticker": ticker,
        "n_hits": len(hits),
        "hits": [
            {"title": h["title"], "summary": h["summary"], "date": h["date"],
             "provider": h["provider"], "url": h["url"], "score": h["score"],
             "ticker": h.get("ticker", ticker)}
            for h in hits
        ],
    }


# ─────────────────────────────────────────────────────────────
# SENTIMENT (LLM-scored, lexicon fallback, full story list)
# ─────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains", "record",
    "upgrade", "upgraded", "buy", "bullish", "strong", "growth", "grow", "raised",
    "raise", "profit", "profits", "soar", "soars", "jump", "jumps", "win", "wins",
    "positive", "outperform", "expands", "expansion", "optimistic", "momentum",
}
NEGATIVE_WORDS = {
    "drop", "drops", "fall", "falls", "fell", "plunge", "plunges", "downgrade",
    "downgraded", "sell", "bearish", "weak", "weakness", "decline", "declines",
    "loss", "losses", "cut", "cuts", "warning", "warns", "warned", "lawsuit",
    "investigation", "probe", "miss", "misses", "missed", "fraud", "crisis",
    "negative", "underperform", "layoff", "layoffs", "restructuring", "slump",
}


def lexicon_score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", t))
    neg = sum(1 for w in NEGATIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", t))
    if pos > neg:
        return 1.0
    if neg > pos:
        return -1.0
    return 0.0


def sentiment_score(ticker: str, lookback_days: int = 7,
                    provider: str = "opencode-free", model: str = "") -> dict:
    news = get_ticker_news(ticker, limit=15)
    if not news:
        return {"error": "no recent news"}

    any_filtered = any(n.get("_filtered") for n in news)
    news_source = "ticker-filtered" if any_filtered else "market-wide-fallback"

    windowed = []
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    for n in news:
        try:
            d = datetime.fromisoformat(n["date"].replace("Z", "+00:00")).replace(tzinfo=None)
            n["_dt"] = d
            if d >= cutoff:
                windowed.append(n)
        except (ValueError, TypeError):
            windowed.append(n)
    if not windowed:
        windowed = news[:8]

    lines = "\n".join(
        f"{i+1}. [{n['date'][:10]}] ({n['provider']}) {n['title']}" for i, n in enumerate(windowed)
    )
    prompt = (
        "You are a financial sentiment analyst. Score each headline below as "
        "bullish (+1), neutral (0), or bearish (-1) for the listed company, based "
        "only on the text. Return STRICT JSON: {\"items\": [{\"idx\": 1, \"score\": 1, "
        "\"ticker\": \"\"}], \"overall\": {\"mean\": 0.0, \"verdict\": \"bullish|neutral|bearish\"}}.\n\n"
        f"HEADLINES:\n{lines}"
    )
    resp = llm_chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=700,
                    provider=provider, model=model)
    if not resp.get("ok") or not resp.get("text"):
        return {"error": resp.get("error", "LLM returned empty response")}
    resp_model = resp.get("model", "?")

    parsed, scores, by_day = None, [], {}
    try:
        raw = resp["text"].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            parsed = json.loads(raw[start:i + 1])
                            break
            if parsed is None:
                raise
        for item in parsed.get("items", []):
            idx = int(item.get("idx", 0)) - 1
            if 0 <= idx < len(windowed):
                windowed[idx]["_score"] = max(-1.0, min(1.0, float(item.get("score", 0))))
                scores.append(windowed[idx]["_score"])
                day = windowed[idx].get("_dt", datetime(1970, 1, 1)).date().isoformat()
                by_day.setdefault(day, []).append(windowed[idx]["_score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    if not scores:
        scores, by_day = [], {}
        for n in windowed:
            n["_score"] = lexicon_score(n["title"] + " " + n.get("summary", ""))
            scores.append(n["_score"])
            day = n.get("_dt", datetime(1970, 1, 1)).date().isoformat()
            by_day.setdefault(day, []).append(n["_score"])
        fallback = True
    else:
        fallback = False

    daily = {d: round(float(np.mean(v)), 3) for d, v in sorted(by_day.items())}
    mean = float(np.mean(scores))
    verdict = "bullish" if mean > 0.15 else ("bearish" if mean < -0.15 else "neutral")

    # Full story list, sorted by |score| — the agent's complete read
    stories = sorted(windowed, key=lambda n: abs(n.get("_score", 0)), reverse=True)
    return {
        "ticker": ticker,
        "n_headlines": len(windowed),
        "mean_score": round(mean, 3),
        "verdict": verdict,
        "daily": daily,
        "scoring": "lexicon-fallback" if fallback else "llm",
        "news_source": news_source,
        "model": resp_model,
        "stories": [
            {"title": n["title"], "summary": n.get("summary", ""), "score": n.get("_score", 0),
             "provider": n["provider"], "url": n["url"], "date": n["date"][:10]}
            for n in stories
        ],
    }


# ─────────────────────────────────────────────────────────────
# AGENT CORE
# ─────────────────────────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "Why is NVDA up today?",
    "AAPL risk check: volatility, VaR, 1-week range",
    "Market mood on TSLA?",
    "MSFT technicals: RSI, MACD, Bollinger",
    "BTC-USD: earnings + risk snapshot",
    "NVDA vs AMD: volatility face-off",
]


def run_tool(name: str, args: dict, provider: str = "opencode-free", model: str = "") -> dict:
    ticker = args.get("ticker", "").upper().strip()
    if not ticker:
        return {"error": "no ticker"}
    if name == "get_quote":
        return get_quote(ticker)
    if name == "get_history":
        return get_history(ticker, args.get("interval", "1d"), args.get("period", "1y")).tail(50).to_dict("records")
    if name == "get_calendar":
        return get_calendar(ticker)
    if name == "rag_search":
        extra = tuple(args.get("extra_tickers") or [])
        return rag_search(args.get("question", ""), ticker, int(args.get("top_k", 5)), extra)
    if name in ("get_news", "get_ticker_news"):
        return get_ticker_news(ticker, int(args.get("limit", 10)))
    if name == "sentiment":
        return sentiment_score(ticker, int(args.get("lookback_days", 7)), provider, model)
    if name == "risk_metrics":
        return risk_metrics(ticker)
    if name == "technical_facts":
        return technical_facts(ticker)
    if name == "fundamental_facts":
        return fundamental_facts(ticker)
    if name == "forecast_range":
        return forecast_range(ticker, int(args.get("horizon_days", 7)))
    if name == "run_analysis":
        return agent_analysis(args.get("question", ""), ticker, provider, model,
                              tickers=args.get("tickers") or None)
    if name == "pair_correlation":
        tb = str(args.get("ticker_b", "")).strip().upper()
        if not tb:
            return {"error": "pair_correlation needs both ticker (a) and ticker_b (b)"}
        return pair_correlation(ticker, tb)
    return {"error": f"unknown tool {name}"}


# Tool manifest — what the agent can call. The LLM decides, we execute.
AGENT_TOOL_MANIFEST = [
    {
        "name": "get_quote",
        "description": "Live quote: price, day change %, volume, market cap, prev close, day high/low.",
        "args": {"ticker": "string symbol e.g. AAPL"},
    },
    {
        "name": "get_history",
        "description": "OHLCV price history. interval: 1d/1wk/1mo; period: 1mo/6mo/1y/2y/5y/10y.",
        "args": {"ticker": "string", "interval": "1d", "period": "1y"},
    },
    {
        "name": "rag_search",
        "description": "SEMANTIC RETRIEVAL (embedding-based RAG) over the news corpus: ask a question, get the top-k most relevant dated stories with provider, URL and relevance score. Pass extra_tickers (list, e.g. peers) to widen the corpus for comparison/sector questions. Use this for news/sentiment/why/compare questions.",
        "args": {"ticker": "string", "question": "the user's full question", "top_k": 5, "extra_tickers": ["MSFT", "GOOG"]},
    },
    {
        "name": "get_ticker_news",
        "description": "Recent headlines relevant to this ticker (filtered from the market feed).",
        "args": {"ticker": "string", "limit": 10},
    },
{
        "name": "technical_facts",
        "description": "Computed technicals from price history: RSI(14), MACD + signal, Bollinger band.",
        "args": {"ticker": "string"},
    },
    {
        "name": "fundamental_facts",
        "description": "Fundamentals from yfinance .info: valuation (P/E, EPS, P/B, P/S), revenue + margins, beta, debt/equity, dividend yield, 52-week range, analyst recommendation and price targets.",
        "args": {"ticker": "string"},
    },
    {
        "name": "pair_correlation",
        "description": "Deterministic correlation + beta between two tickers (computed in-app with pandas, cannot fail). Use for correlation/beta/co-movement/link questions between two assets instead of run_analysis. Args: ticker = first symbol, ticker_b = second symbol.",
        "args": {"ticker": "e.g. AAPL", "ticker_b": "e.g. MSFT"},
    },
    {
        "name": "run_analysis",
        "description": "Write and execute a Python script (pandas/numpy/yfinance) to compute ANY quantitative answer: volatility, VaR, Sharpe, RSI, MACD, correlation, regression, backtest, GARCH volatility projection, price ranges. Returns real computed numbers via Python execution.",
        "args": {"ticker": "string", "question": "the user's full question to compute"},
    },
]

SUGGESTION_BASKET = ["MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

SUGGESTION_RE = re.compile(
    r"\b(suggest|suggestion|recommend|recommendation|best stock|top stock|"
    r"other stock|other shares|what else|watchlist|pick|idea)\b"
    r"|\u0938\u0941\u091d\u093e|\u0938\u093f\u092b\u093e\u0930\u093f\u0936|"
    r"\u0914\u0930 \u0936\u0947\u092f\u0930|\u0930\u093f\u0915\u092e\u0947\u0902\u0921",
    re.I,
)


def agent_select_tools(question: str, tickers: list[str], selected: str,
                       provider: str = "opencode-free", model: str = "") -> list[tuple]:
    """The LLM chooses which tools to call (function-calling style).

    tickers: every symbol resolved from the question (may be empty).
    selected: the asset currently selected in the sidebar UI.

    Returns a list of (tool_name, args) tuples. Falls back to a sensible
    default pair if the LLM's JSON cannot be parsed.
    """
    manifest_text = "\n".join(
        f"- {t['name']}({json.dumps(t['args'])}) — {t['description']}" for t in AGENT_TOOL_MANIFEST
    )
    ticker_ctx = ", ".join(tickers) if tickers else "none detected"
    prompt = (
        f"User question: {question}\n"
        f"Ticker(s) detected in question: {ticker_ctx}\n"
        f"Selected asset in the UI: {selected}\n\n"
        f"Available tools:\n{manifest_text}\n\n"
        "Return STRICT JSON listing the tools to call, in the order you want their "
        "results: {\"tools\": [{\"name\": \"get_quote\", \"args\": {\"ticker\": \"AAPL\"}}]}. "
        "The \"ticker\" argument accepts ANY valid yfinance symbol the user mentions "
        "(AAPL, AMD, BTC-USD, GBPUSD=X, ^SPX...), even if it is not in the list above. "
        "For multi-ticker questions, call get_quote for EACH ticker asked about. "
        "Include get_history when the question needs price trends or computations, "
        "run_analysis for anything requiring calculation (volatility, VaR, "
        "Sharpe, RSI, forecasts, correlations), and fundamental_facts for valuation "
        "or earnings questions. Use at most 4 tools (up to 10 for suggestion "
        "questions below). If the user asks for stock suggestions or "
        "recommendations of OTHER stocks (e.g. 'suggest other stocks', "
        "'कुछ और शेयर सुझाओ', 'recommendations'), call get_quote AND "
        "fundamental_facts for several well-known candidates (e.g. MSFT, GOOGL, "
        "AMZN, NVDA, TSLA) so the suggestions come from real data. "
        "If no tool helps, return "
        "{\"tools\": []}."
    )
    resp = llm_chat([
        {"role": "system", "content": (
            "You are a tool-selection router. Output ONLY the JSON object, "
            "no markdown, no commentary.")},
        {"role": "user", "content": prompt},
    ], temperature=0.0, max_tokens=500, provider=provider, model=model)

    primary = tickers[0] if tickers else selected
    selected_calls = []
    parsed = None
    if resp.get("ok") and resp.get("text"):
        raw = resp["text"].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            parsed = json.loads(raw[start:i + 1])
                            break
        if isinstance(parsed, dict):
            valid = {t["name"] for t in AGENT_TOOL_MANIFEST}
            for item in parsed.get("tools", [])[:12]:
                name = item.get("name", "")
                if name in valid:
                    args = item.get("args") or {}
                    if name == "run_analysis":
                        args["question"] = question
                    if "ticker" not in args:
                        args["ticker"] = primary
                    selected_calls.append((name, args))

    # Deterministic fallback if selection failed
    if not selected_calls:
        selected_calls = [
            ("get_quote", {"ticker": primary}),
            ("run_analysis", {"ticker": primary, "question": question}),
            ("fundamental_facts", {"ticker": primary}),
        ]

    # Suggestion/recommendation requests: the question names no other ticker,
    # so fetch a candidate basket to make suggestions come from real data.
    # Deterministic — never depends on the router LLM deciding to do it.
    if SUGGESTION_RE.search(question):
        asked = {t.upper() for t in tickers}
        for s in SUGGESTION_BASKET:
            if s in asked or any(str(a.get("ticker", "")).upper() == s for _, a in selected_calls):
                continue
            selected_calls.append(("get_quote", {"ticker": s}))
            selected_calls.append(("fundamental_facts", {"ticker": s}))
    return selected_calls


def prepare_answer(question: str, provider: str = "opencode-free", model: str = "",
                   history: list[dict] | None = None) -> dict:
    """Agentic flow: resolve every ticker asked about → LLM picks tools →
    we execute → build synthesis context. Returns the messages to stream
    plus metadata for the UI.
    """
    tickers = resolve_tickers(question)
    selected = st.session_state.get("_sidebar_ticker", "AAPL") or "AAPL"
    if not tickers:
        tickers = [selected]

    tool_calls = agent_select_tools(question, tickers, selected, provider, model)

    # Data guarantee: every ticker the user asked about gets at least one call.
    used = {str(a.get("ticker", "")).upper() for _, a in tool_calls}
    for t in tickers:
        if t.upper() not in used:
            tool_calls.append(("get_quote", {"ticker": t}))

    # Correlation/link questions get the deterministic pair tool so they can
    # never collapse on buggy LLM-written code.
    if (len(tickers) >= 2
            and re.search(r"\b(correl|beta|co.mov|covary|cointegrat|diversif|link)\b",
                          question, re.I)
            and not any(n == "pair_correlation" for n, _ in tool_calls)):
        tool_calls.append(("pair_correlation",
                           {"ticker": tickers[0], "ticker_b": tickers[1]}))

    # run_analysis always learns every asked ticker (primary first).
    for name, args in tool_calls:
        if name == "run_analysis":
            args["tickers"] = tickers

    results = []
    for name, args in tool_calls:
        try:
            results.append({"tool": name, "args": args,
                            "result": run_tool(name, args, provider, model)})
        except Exception as e:
            results.append({"tool": name, "args": args,
                            "result": {"error": str(e)[:150]}})

    ctx_lines = []
    for r in results:
        payload = r["result"]
        tk = (r.get("args") or {}).get("ticker", "?")
        if r["tool"] in ("run_analysis", "rag_search"):
            ctx_lines.append(f"## tool: {r['tool']} · ticker: {tk}\n{json.dumps(payload, default=str)[:6000]}")
        else:
            ctx_lines.append(f"## tool: {r['tool']} · ticker: {tk}\n{json.dumps(payload, default=str)}")
    context = "\n\n".join(ctx_lines)

    system = (
        "You are a market-intelligence analyst. You answer from data, with "
        "evidence, always in the context of the overall market (indices, sector, "
        "macro) AND specifically for the ticker(s) the user asked about or "
        "selected. Guidelines:\n"
        "1. Base your answer strictly on the tool data provided.\n"
        "2. Cite headline sources with their provider and date when you mention news.\n"
        "3. Present facts (vol, VaR, RSI, earnings date, valuation) and clearly label any interpretation.\n"
        "4. ALWAYS answer the user's question directly from the data. Never refuse, "
        "never say 'I cannot determine', never return an empty reply. When the exact "
        "metric is missing, give the closest available figure and name the gap.\n"
        "5. When multiple tickers were asked about, answer for each one as a clear "
        "per-ticker breakdown, and compare them when the question asks.\n"
        "6. Provide information and analysis; avoid personalized trade advice or "
        "price targets.\n"
        "7. FORECASTS: when asked for an outlook/forecast/prediction, give a "
        "probabilistic range based on computed volatility (from run_analysis output), "
        "framed with confidence, and ALWAYS append the standard warning below verbatim.\n"
        "8. Report numbers exactly as produced by tool output or run_analysis "
        "execution results.\n"
        "9. End with a one-line 'Bottom line' summary.\n"
        "10. If the user asked for suggestions or recommendations of OTHER stocks, "
        "recommend ONLY from the candidate data present in TOOL DATA — never "
        "recommend a ticker whose data is absent, and cite each candidate's "
        "price and valuation from the tool output.\n\n"
        "STANDARD WARNING (append verbatim to any forecast):\n"
        "\"⚠️ This is not investment advice. Forecasts are probabilistic "
        "estimates based on historical volatility; markets can move beyond any "
        "range. Do your own research before making any financial decision.\""
    )
    user_msg = (
        f"QUESTION: {question}\n"
        f"SELECTED ASSET (sidebar): {selected}\n"
        f"ASKED TICKER(S): {', '.join(tickers)}\n\n"
        f"TOOL DATA:\n{context}\n\n"
        "Answer in 2-4 short paragraphs per relevant ticker, framed in market "
        "context, with the Bottom line."
    )

    messages = [{"role": "system", "content": system}]
    if history:
        for turn in history[-100:]:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({"role": "user", "content": user_msg})

    return {
        "ticker": selected,
        "tickers": tickers,
        "tools_used": [t for t, _ in tool_calls],
        "messages": messages,
    }


def answer_question(question: str, provider: str = "opencode-free", model: str = "",
                    history: list[dict] | None = None) -> dict:
    """Non-streaming path: run tools then get the full answer."""
    prep = prepare_answer(question, provider, model, history)
    resp = llm_chat(prep["messages"], provider=provider, model=model)
    return {
        "answer": resp.get("text", f"LLM error: {resp.get('error', '?')}"),
        "ticker": prep["ticker"],
        "tools_used": prep["tools_used"],
        "model": resp.get("model", "?"),
        "provider": resp.get("provider", provider),
    }


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def fresh_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def link_html(url: str, text: str | None = None) -> str:
    """Clickable source link as real HTML anchor (opens in new tab)."""
    if not url:
        return ""
    label = text or "Read full story ↗"
    return f'<a class="src-link" href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'


def render_story_card(s: dict, show_summary: bool = False):
    icon = "🟢" if s["score"] > 0.3 else ("🔴" if s["score"] < -0.3 else "⚪")
    meta = f'{s["provider"]} · {s["date"]}'
    link = link_html(s.get("url", ""), "Read full story ↗")
    html = (
        f'<div class="card"><small>{icon} <b>{s["title"]}</b> '
        f'<span class="tag tag-gold">{s["score"]:+.1f}</span></small><br>'
        f'<small style="color:#777;">{meta} · {link}</small>'
    )
    if show_summary and s.get("summary"):
        html += f'<br><small style="color:#444;">{s["summary"][:220]}</small>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def hero():
    st.markdown(
        """
        <div class="hero">
            <h1>📡 Real-Time Market Research Assistant</h1>
            <p>Live quotes · headline sentiment · risk analytics · technical indicators · RAG retrieval · LLM analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar():
    with st.sidebar:
        st.markdown("### 📈 Asset Symbol Options")

        ticker_input = st.text_input("Type your ticker (overrides dropdown if filled):", "AAPL").upper()
        ticker_options = [ticker_input] + [t for t in TICKERS if t != ticker_input]
        ticker_symbol = st.selectbox("Or select any ticker from list:", ticker_options, index=0)
        st.session_state["_sidebar_ticker"] = ticker_symbol

        st.markdown('<small style="color:#555; font-weight:600;">Quick picks</small>', unsafe_allow_html=True)
        wcols = st.columns(4)
        for i, w in enumerate(WATCHLIST):
            with wcols[i % 4]:
                if st.button(w, key=f"w_{w}", use_container_width=True):
                    ticker_symbol = w
                    st.session_state["_sidebar_ticker"] = w

        interval_options = ['1d', '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '5d', '1wk', '1mo', '3mo']
        selected_interval = st.selectbox("Select Time Interval:", interval_options, index=0)

        lookback_days = st.slider("Sentiment Lookback (days)", 1, 30, 7)

        st.markdown("---")
        st.markdown("### 🤖 LLM Provider")
        provider = st.selectbox("Provider:", list(LLM_PROVIDERS.keys()), index=0)
        model_choice = st.selectbox(
            "Model:", LLM_PROVIDERS[provider]["models"], index=0,
            help=f"{LLM_PROVIDERS[provider]['tag']} · auto-fallback across providers on failure",
        )

        auto_refresh = st.checkbox("🔄 Auto-refresh price chart (60s)", value=False)

        st.markdown("---")
        run = st.button("▶️ Run / Refresh All Tabs", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            '<small style="color:#888;">Facts, sources, and quantified uncertainty — '
            'every figure computed from live data.<br>Portfolio Edition · opencode-free + NVIDIA + Groq</small>',
            unsafe_allow_html=True,
        )
        return {
            "ticker": ticker_symbol,
            "interval": selected_interval,
            "lookback": lookback_days,
            "provider": provider,
            "model": model_choice,
            "run": run,
            "auto_refresh": auto_refresh,
        }


# ── Tab 1: Price Chart ──
def draw_candles(ax, df, window: int = 120):
    """Hand-rolled candlesticks (no mplfinance dependency)."""
    d = df.tail(window)
    up = d["Close"] >= d["Open"]
    for i in range(len(d)):
        o, c, h, l = d["Open"].iloc[i], d["Close"].iloc[i], d["High"].iloc[i], d["Low"].iloc[i]
        color = "#1f8a4c" if up.iloc[i] else "#c62828"
        ax.plot([i, i], [l, h], color=color, lw=0.7)
        body_bottom = min(o, c)
        body_top = max(o, c)
        ax.add_patch(Rectangle((i - 0.35, body_bottom), 0.7, max(body_top - body_bottom, 1e-9),
                               facecolor=color, edgecolor=color))
    ax.set_xlim(-0.6, len(d) - 0.4)
    # X ticks: dates every ~nth bar
    step = max(1, len(d) // 6)
    ticks = list(range(0, len(d), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.index[i].strftime("%m-%d") for i in ticks], rotation=30, ha="right")
    return d


def price_chart_tab(ticker: str, interval: str, auto_refresh: bool):
    st.markdown(f"### 📈 {ticker} — Price Chart")
    st.caption(
        f"Interval: {interval} → history window: {interval_period(interval)} · "
        f"data as of {fresh_stamp()} · historical data with moving averages and volume"
    )

    chart_type = st.radio("Chart type:", ["Line + MA", "Candlestick"], horizontal=True)

    # Auto-refresh: when enabled, Streamlit re-runs this fragment every 60s.
    # yfinance calls are still throttled by the 5-minute data cache.
    frag = st.fragment(run_every=60 if auto_refresh else None)

    @frag
    def _chart_fragment(ticker: str, interval: str, chart_type: str):
        df = get_history(ticker, interval, interval_period(interval))
        if df.empty:
            st.warning(f"No data found for {ticker} at {interval}.")
            return

        q = get_quote(ticker)
        price = q.get("price")
        chg = q.get("day_change_pct")
        if price:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Last Price", f"${price:,.2f}",
                         delta=f"{chg:.2f}%" if chg is not None else None)
            col_b.metric("Bars", f"{len(df):,}")
            col_c.metric("History Window", interval_period(interval))

        if chart_type == "Candlestick":
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(11, 6.5), sharex=False,
                gridspec_kw={"height_ratios": [3, 1]}, dpi=110,
            )
            d = draw_candles(ax1, df)
            ax1.set_ylabel("Price")
            ax1.set_title(f"{ticker} — {interval} candlesticks (last {len(d)})")
            ax1.grid(True, alpha=0.25)
            ax2.bar(range(len(d)), d["Volume"], color="#9ca3af", alpha=0.6)
            ax2.set_ylabel("Volume")
            ax2.grid(True, alpha=0.25)
        else:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(11, 6.5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}, dpi=110,
            )
            ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.6, label="Close")
            if len(df) >= 20:
                ax1.plot(df.index, df["Close"].rolling(20).mean(), color="#f59e0b",
                         lw=1.2, alpha=0.85, label="MA20")
            if len(df) >= 50:
                ax1.plot(df.index, df["Close"].rolling(50).mean(), color="#7B2D8E",
                         lw=1.2, alpha=0.75, label="MA50")
            ax1.set_ylabel("Price")
            ax1.set_title(f"{ticker} — {interval} bars")
            ax1.legend(fontsize=9, loc="upper left")
            ax1.grid(True, alpha=0.25)
            ax2.bar(df.index, df["Volume"], color="#9ca3af", alpha=0.6, width=0.8)
            ax2.set_ylabel("Volume")
            ax2.grid(True, alpha=0.25)
            span_days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
            if span_days < 3:
                fmt = "%H:%M"
            elif span_days < 400:
                fmt = "%Y-%m-%d"
            else:
                fmt = "%Y-%m"
            for ax in (ax1, ax2):
                ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
                ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        st.pyplot(fig)

    _chart_fragment(ticker, interval, chart_type)


# ── Tab 2: Ask the Analyst (streaming, follow-up memory) ──
def chat_tab(ticker: str, provider: str, model: str):
    st.markdown("### 💬 Ask the Analyst")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    # Render existing history (Streamlit-native chat bubbles)
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["text"])
            if msg["role"] == "agent":
                tools = ", ".join(msg.get("tools", []))
                st.markdown(
                    f'<small style="color:#7B2D8E;">🤖 <b>{msg.get("model","?")}</b> · '
                    f'{msg.get("provider","?")} · tools: {tools}</small>',
                    unsafe_allow_html=True,
                )

    # Suggested questions (set a pending question)
    st.markdown(
        '<span class="metric-label">Suggested questions</span>',
        unsafe_allow_html=True,
    )
    sug_cols = st.columns(3)
    pending = None
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        with sug_cols[i % 3]:
            if st.button(f"💡 {sq}", key=f"sug_{i}", use_container_width=True):
                pending = sq

    # st.chat_input: submit-once semantics — no reload loop, clears after send
    question = st.chat_input(f"e.g. Why is {ticker} up today?")
    if question is None:
        question = pending

    if question and question.strip():
        q_text = question.strip()
        with st.chat_message("user"):
            st.markdown(q_text)

        with st.chat_message("assistant"):
            with st.spinner("Running tools..."):
                prep = prepare_answer(q_text, provider, model,
                                      history=st.session_state.chat_turns)
            streamed = st.write_stream(
                llm_chat_stream(prep["messages"], provider=provider, model=model)
            )
            prov = st.session_state.get("_llm_provider", provider)
            mdl = st.session_state.get("_llm_model", model)
            tools = ", ".join(prep["tools_used"])
            tickers = ", ".join(prep.get("tickers", [])) or prep["ticker"]
            st.markdown(
                f'<small style="color:#7B2D8E;">🤖 <b>{mdl}</b> · {prov} · '
                f'tickers: {tickers} · tools: {tools}</small>',
                unsafe_allow_html=True,
            )

        st.session_state.chat_history.append({"role": "user", "text": q_text})
        st.session_state.chat_history.append({
            "role": "agent", "text": streamed, "tools": prep["tools_used"],
            "model": mdl, "provider": prov,
        })
        st.session_state.chat_turns.append({"q": q_text, "a": streamed})

    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.chat_turns = []
            st.rerun()

    st.markdown("---")
    st.caption("Computed from live data · Informational, not investment advice.")


# ── Tab 3: Sentiment (full story list) ──
def sentiment_tab(ticker: str, lookback: int, provider: str, model: str):
    st.markdown(f"### 🧭 Sentiment — {ticker}")
    st.caption(f"lookback: {lookback} days (sidebar slider) · scored by {provider}/{model}")

    with st.spinner(f"Scoring headlines with the LLM ({provider})..."):
        res = sentiment_score(ticker, int(lookback), provider, model)
    if "error" in res:
        st.warning(f"Sentiment unavailable: {res['error']}")
        return

    verdict = res["verdict"]
    vcolor = "#15803d" if verdict == "bullish" else ("#b91c1c" if verdict == "bearish" else "#b45309")
    st.markdown(
        f'<div class="card"><span class="metric-label">Tone (last {res["n_headlines"]} headlines · '
        f'{res.get("scoring","llm")} · {res.get("model","")})</span>'
        f'<div class="metric-big" style="color:{vcolor};">{res["mean_score"]:.2f} · {verdict.upper()}</div></div>',
        unsafe_allow_html=True,
    )

    if res.get("daily"):
        daily_df = pd.DataFrame({"date": list(res["daily"]), "score": list(res["daily"].values())})
        st.bar_chart(daily_df.set_index("date"))

    st.markdown("---")
    st.markdown("**All scored headlines (ranked by score magnitude)**")
    for s in res["stories"]:
        render_story_card(s, show_summary=True)


# ── Tab 4: Risk & Technical ──
def risk_tech_tab(ticker: str):
    st.markdown(f"### 🛡 Risk & 📐 Technical — {ticker}")
    st.caption("Descriptive analytics — historical figures, volatility-based ranges, and indicator facts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk Snapshot (1y daily)**")
        m = risk_metrics(ticker)
        if "error" in m:
            st.warning(m["error"])
        else:
            col_a, col_b = st.columns(2)
            col_a.metric("Daily Vol", f"{m['daily_vol_pct']}%")
            col_a.metric("Max Drawdown", f"{m['max_drawdown_pct']}%")
            col_b.metric("Annual Vol", f"{m['annual_vol_pct']}%")
            col_b.metric("Sharpe (ann.)", f"{m['sharpe_annual']}")
            st.caption(f"95% VaR (1d): {m['var95_1d_pct']}% · {m['days']} trading days · historical method")

    with c2:
        st.markdown("**Technical Facts (6mo daily)**")
        t = technical_facts(ticker)
        if "error" in t:
            st.warning(t["error"])
        else:
            col_c, col_d = st.columns(2)
            col_c.metric("RSI(14)", t["rsi14"])
            col_c.metric("MACD", t["macd"], delta=round(t["macd"] - t["macd_signal"], 3))
            col_d.metric("Bollinger Upper", t["bollinger_upper"])
            col_d.metric("Bollinger Lower", t["bollinger_lower"])
            band = "inside band" if t["in_bollinger_band"] else "OUTSIDE band"
            st.caption(f"Price {t['price']} · {band} · indicators shown as facts")

    cal = get_calendar(ticker)
    if cal:
        st.markdown("**📅 Upcoming events**")
        for k, v in cal.items():
            if v:
                st.markdown(f"- {k}: **{v}**")


# ── Tab 5: Raw Data + News ──
def data_tab(ticker: str, interval: str):
    st.markdown(f"### 📊 Raw Data — {ticker} ({interval})")
    st.caption(f"data as of {fresh_stamp()}")
    df = get_history(ticker, interval, interval_period(interval))
    if df.empty:
        st.warning(f"No data found for {ticker} at {interval}.")
        return

    st.dataframe(df.tail(100), width="stretch")
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(),
        file_name=f"{ticker}_{interval}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### 📰 Latest Headlines")
    news = get_news(ticker, limit=10)
    if not news:
        st.info("No recent news.")
    for n in news:
        link = link_html(n.get("url", ""), "Read full story ↗")
        summary_html = f'<br><small style="color:#444;">{n["summary"][:200]}</small>' if n.get("summary") else ""
        st.markdown(
            f'<div class="card"><small><b>{n["title"]}</b></small><br>'
            f'<small style="color:#777;">{n["provider"]} · {n["date"][:10]} · {link}</small>'
            f'{summary_html}</div>',
            unsafe_allow_html=True,
        )


# ── Tab 6: Project Overview (technical documentation) ──
def overview_tab():
    st.markdown("### 📄 Project Overview & Technical Documentation")

    st.markdown(
        '<div class="card">'
        '<span class="metric-label">Project Overview</span><br>'
        '<span style="color:#555;">A single-file market research workstation that combines '
        'live quotes, OHLCV charting, headline sentiment, risk analytics, technical '
        'indicators, semantic news retrieval (RAG), and a tool-using LLM agent. Every '
        'figure is computed from live data; every answer carries sources and quantified '
        'uncertainty.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    stat_a, stat_b, stat_c, stat_d = st.columns(4)
    stat_a.metric("Tickers tracked", len(TICKERS))
    stat_b.metric("LLM providers", len(LLM_PROVIDERS))
    stat_c.metric("Agent tools", len(AGENT_TOOL_MANIFEST))
    stat_d.metric("Tabs", 6)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏗 Architecture")
        st.markdown(
            """
            | Layer | Component |
            |---|---|
            | **Data** | `yfinance` — OHLCV history, live quotes, news feed, earnings calendar |
            | **Retrieval** | sentence-transformers embeddings + cosine top-k over the news corpus (RAG) |
            | **Compute** | code-interpreter subprocess — the LLM writes pandas/numpy code, Python executes it |
            | **LLM** | opencode-free · NVIDIA Build · Groq · kilo-free (OpenAI-compatible endpoints, fallback chain) |
            | **Agent** | LLM-driven tool selection → execution → synthesis (citations instructed) |
            | **Sentiment** | LLM JSON scoring with deterministic lexicon fallback |
            | **Risk** | historical vol, VaR, max drawdown, Sharpe (descriptive) |
            | **Technical** | RSI(14), MACD(12,26,9), Bollinger(20,2) — facts only |
            | **UI** | Streamlit · wide layout · sidebar is the single source of truth |
            """
        )
        st.markdown("#### 🧠 Agent Loop")
        st.markdown(
            """
            1. **Tool selection** — the LLM reads a tool manifest and chooses
               what to call (quote, history, RAG, calendar, code execution).
            2. **Execution** — tools run against live data; `run_analysis`
               writes Python, the subprocess computes real numbers.
            3. **Retrieval** — `rag_search` embeds the question and returns the
               most relevant dated stories with provider, URL and score.
            4. **Synthesis** — the LLM answers using only tool output, is
               instructed to cite the retrieved sources (attached with
               provider, date and URL), and attaches the standard warning to
               any forecast.
            5. **Repair loop** — if generated code fails, the error is returned
               to the LLM for up to two correction attempts.

            *Note: generated code executes locally as the app user with
            normal filesystem/network access — do not deploy unauthenticated.*
            """
        )
    with c2:
        st.markdown("#### 🤖 LLM Providers")
        st.markdown(
            """
            | Provider | Endpoint | Models |
            |---|---|---|
            | **opencode-free** | `opencode.ai/zen/v1` | deepseek-v4-flash-free, mimo-v2.5-free, ling-3.0-flash-free, nemotron-3-ultra-free |
            | **nvidia-build** | `integrate.api.nvidia.com/v1` | deepseek-v4-flash/pro, **glm-5.2**, nemotron-ultra-253b, minimax-m3 |
            | **groq** | `api.groq.com/openai/v1` | **gpt-oss-120b**, llama-3.3-70b, llama-3.1-8b, qwen3.6-27b |

            **Fallback chain:** selected provider → its remaining models →
            opencode-free → nvidia-build → groq → kilo-free. Empty or
            rate-limited responses automatically continue down the chain,
            keeping the app available across vendors.
            """
        )

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🔍 Retrieval (RAG)")
        st.markdown(
            """
            - **Corpus** — per-ticker filtered headlines + summaries + computed
              price context; extendable with peer tickers for sector questions.
            - **Embeddings** — `all-MiniLM-L6-v2` (384-d, normalized), cached.
            - **Retrieval** — cosine similarity between the question embedding
              and every corpus document; top-k returned with relevance score.
            - **Grounding** — every retrieved story carries provider, date and
              URL, so any claim the model cites can be traced to its source.
            - **Scale** — yfinance provides ~10 stories per symbol; the corpus
              extends across peer tickers for sector-wide queries.

            **Sentiment scoring:** one LLM call scores each headline as
            +1/0/-1 in strict JSON; unparseable responses fall back to a
            deterministic financial lexicon (+30/−30 terms) for uninterrupted
            availability.
            """
        )
    with c4:
        st.markdown("#### 📐 Risk & Technical Definitions")
        st.markdown(
            """
            - **Daily vol** — std of daily returns; **annualized** × √252.
            - **VaR 95% (1d)** — 5th percentile of the daily-return distribution
              (historical method).
            - **Max drawdown** — worst peak-to-trough decline of cumulative returns.
            - **Sharpe (annual)** — mean daily return ÷ daily vol × √252, 0% risk-free.
            - **RSI(14)** — momentum oscillator; >70 overbought, <30 oversold.
            - **MACD(12,26,9)** — EMA difference vs its 9-period signal line.
            - **Bollinger(20,2)** — 20-day mean ± 2σ band.

            *All indicators are computed from live data and shown as facts.*
            """
        )

    st.markdown("---")
    st.markdown("#### 💻 Run & Deploy")
    st.markdown(
        """
        ```bash
        pip install -r requirements.txt   # streamlit, yfinance, pandas, numpy, matplotlib, requests, sentence-transformers
        streamlit run app.py
        ```
        **Streamlit Cloud:** push to GitHub → connect → add provider keys as
        secrets (`OPENCODE_ZEN_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`) or
        leave them in `~/.hermes/.env`. CPU-only (~350 MB RAM), free tier ready.

        #### 🔒 Data & Coverage Notes
        - News coverage spans the latest ~10–15 headlines per ticker, plus
          multi-ticker retrieval for sector-wide questions.
        - Sentiment is computed from headline tone; retrieved stories are
          attached with provider, date and URL for traceability.
        - Risk metrics use historical data with clearly documented methods.
        - Free LLM tiers are rate-limited; the fallback chain keeps answers
          flowing across opencode-free, NVIDIA Build, and Groq.
        - Uncertainty is quantified as volatility-based ranges, with a standard
          risk disclaimer attached to every forecast-style answer.
        """
    )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#888; font-size:0.8rem;">'
        '<b>Real-Time Market Research Assistant</b> · Portfolio Edition · '
        'yfinance · sentence-transformers · opencode-free + NVIDIA Build + Groq · all figures computed from live data'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN — sidebar drives everything
# ─────────────────────────────────────────────────────────────
def main():
    hero()
    ui = sidebar()

    if ui["run"]:
        st.cache_data.clear()
        st.rerun()

    ticker = ui["ticker"]
    interval = ui["interval"]
    provider = ui["provider"]
    model = ui["model"]

    tabs = st.tabs([
        "📈 Price Chart",
        "💬 Ask the Analyst",
        "🧭 Sentiment",
        "🛡 Risk & Technical",
        "📊 Raw Data + News",
        "📄 Project Overview",
    ])

    with tabs[0]:
        price_chart_tab(ticker, interval, ui["auto_refresh"])
    with tabs[1]:
        chat_tab(ticker, provider, model)
    with tabs[2]:
        sentiment_tab(ticker, ui["lookback"], provider, model)
    with tabs[3]:
        risk_tech_tab(ticker)
    with tabs[4]:
        data_tab(ticker, interval)
    with tabs[5]:
        overview_tab()


if __name__ == "__main__":
    main()
