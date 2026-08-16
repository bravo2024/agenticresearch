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

try:
    # Use the OS certificate store, so HTTPS still verifies on machines whose
    # antivirus re-signs TLS traffic (Kaspersky, ESET...). Verification stays on.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

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
        # Fallback only; the live /models list replaces it once fetched.
        "models": [
            "deepseek-v4-flash-free",
            "nemotron-3-ultra-free",
            "space-bunny-free",
        ],
        "tag": "free",
        "anonymous": True,  # blank key works — free proxy
    },
    "nvidia-build": {
        "base": NVIDIA_BASE,
        "key_env": "NVIDIA_API_KEY",
        "max_tokens": 8192,
        "models": [
            "nvidia/nemotron-3-ultra-550b-a55b",
            "nvidia/nemotron-3-super-120b-a12b",
            "deepseek-ai/deepseek-v4.1-flash",
            "moonshotai/kimi-k3",
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
    # Gemini API, free tier only. Free vs paid is decided by the Google Cloud
    # project behind the key: a key from an AI Studio project with no billing
    # account linked can never be charged (over quota it returns 429, which
    # fails over to the next provider). The app also only ever picks Flash /
    # Flash-Lite text models, which the free tier covers; Pro, image, audio
    # and live models are never selected. A Google AI Pro subscription does
    # not apply to the API either way.
    "gemini-free": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "max_tokens": 8192,
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        "tag": "gemini",
    },
    "kilo-free": {
        "base": "https://api.kilo.ai/api/openrouter",
        "key_env": "KILO_API_KEY",
        "max_tokens": 8192,
        # Free tier needs no sign-in. kilo-auto/efficient|balanced|frontier
        # need a Kilo account (401 PAID_MODEL_AUTH_REQUIRED without one).
        "models": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "stepfun/step-3.7-flash:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "inclusionai/ling-3.0-flash-sante:free",
            "poolside/laguna-s-2.1:free",
            "kilo-auto/free",
            "cohere/north-mini-code:free",
        ],
        "tag": "kilo-free",
        "anonymous": True,  # blank key works — public gateway
        "app_attribution": True,  # OpenRouter-style HTTP-Referer / X-Title for this app
    },
}
# Auto mode walks this order; a pinned provider is tried first, then the rest.
# Keyed providers lead (fast, strong, and skipped instantly without a key),
# then Kilo's keyless free tier, then opencode's, which answered 1 of 14 free
# models on 2026-10-07. Curated `models` are the fallback when /models cannot
# be reached, and the names shown first when they are still live.
LLM_PROVIDER_ORDER = ["groq", "gemini-free", "nvidia-build", "kilo-free", "opencode-free"]
_MODELS_CACHE: dict[str, tuple[float, list[str]]] = {}
_MODELS_TTL = 600

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
    (e.g. AMD) are left unresolved here — the agent may still use them."""
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


@st.cache_data(ttl=86400, show_spinner=False)
def normalize_ticker(raw: str) -> str:
    """Turn whatever the agent wrote ("Tata Motors", "TATAMOTORS", "AMD") into
    a yfinance symbol. The hardcoded list only knows a handful of Indian names,
    so anything else goes through Yahoo's search. A bare Indian symbol has no
    US listing, so the search prefers the NSE line (.NS), then BSE (.BO)."""
    t = (raw or "").strip()
    if not t:
        return ""
    up = t.upper()
    # Already yfinance-shaped: RELIANCE.NS, ^NSEI, GC=F, BTC-USD, INR=X
    if up in TICKERS or re.search(r"[.=^-]", up):
        return up
    hits = resolve_tickers(t)
    if hits:
        return hits[0]
    try:
        quotes = yf.Search(t, max_results=8).quotes or []
    except Exception:
        return up
    syms = [q.get("symbol", "") for q in quotes
            if q.get("quoteType") in ("EQUITY", "ETF", "INDEX") and q.get("symbol")]
    if not syms:
        return up
    if up in syms:
        return up
    for suffix in (".NS", ".BO"):
        for s in syms:
            if s.endswith(suffix):
                return s
    return syms[0]


def market_of(ticker: str) -> str:
    """Which cost table and benchmark apply to this symbol."""
    t = ticker.upper()
    if t.endswith((".NS", ".BO")) or t in ("^NSEI", "^NSEBANK", "^BSESN"):
        return "india"
    if t.endswith("-USD"):
        return "crypto"
    if t.endswith(("=F", "=X")):
        return "other"
    return "us"


def benchmark_for(ticker: str) -> str | None:
    """The index a stock is judged against. None for indices themselves and
    for futures/FX, where 'beat the market' has no obvious meaning."""
    t = ticker.upper()
    if t.startswith("^") or t == "BTC-USD":
        return None
    return {"india": "^NSEI", "us": "^GSPC", "crypto": "BTC-USD"}.get(market_of(t))


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


def _key_for(cfg: dict) -> str:
    """The provider's key if one is configured. Keyless providers still use a
    key when present (official, higher limits) and go anonymous otherwise."""
    return get_key(cfg["key_env"]) or ""


APP_REFERER = "https://github.com/bravo2024/agenticresearch"
APP_TITLE = "Agentic Market Research Assistant"


def _provider_headers(cfg: dict, key: str) -> dict:
    """Request headers for a provider. OpenRouter-style gateways (Kilo) get
    the standard attribution headers naming this app."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if cfg.get("app_attribution"):
        headers.update({"HTTP-Referer": APP_REFERER, "X-Title": APP_TITLE})
    return headers


def discover_models(provider: str, timeout: int = 10) -> list[str]:
    """Model ids from a provider's OpenAI-compatible /models endpoint.

    Cached for 10 minutes. Returns [] when the provider is unknown, the
    endpoint is down, or a keyed provider has no key. A miss is not cached,
    so the next sidebar load tries again.
    """
    cfg = LLM_PROVIDERS.get(provider)
    if not cfg:
        return []
    now = time.time()
    hit = _MODELS_CACHE.get(provider)
    if hit and now - hit[0] < _MODELS_TTL:
        return hit[1]
    key = _key_for(cfg)
    if not cfg.get("anonymous") and not key:
        return []
    ids: list[str] = []
    try:
        r = requests.get(
            f"{cfg['base'].rstrip('/')}/models",
            headers=_provider_headers(cfg, key),
            timeout=timeout,
        )
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        # Gemini's OpenAI-compatible list says "models/gemini-2.5-flash";
        # chat/completions wants the bare id.
        ids = [i[len("models/"):] if i.startswith("models/") else i for i in ids]
    except Exception:
        ids = []
    if ids:
        _MODELS_CACHE[provider] = (now, ids)
    return ids


# Providers advertise far more than chat models, and the strongest ones are
# not listed first. Discovered ids are ranked by these patterns (earlier =
# tried first); anything matching MODEL_SKIP is not a chat model at all.
MODEL_RANK = {
    "opencode-free": [r"deepseek", r"kimi", r"glm", r"nemotron", r"qwen", r"minimax", r"free"],
    "groq": [r"gpt-oss-120b", r"kimi-k", r"llama-3\.3-70b", r"qwen", r"gpt-oss-20b", r"llama-3\.1-8b"],
    # Free-tier text models only: Flash first, then Flash-Lite. No Pro, and
    # nothing marked preview/exp, image, audio, live or TTS.
    "gemini-free": [r"^gemini-\d+(\.\d+)?-flash$", r"^gemini-\d+(\.\d+)?-flash-lite$"],
    # Nemotron-3 answers in ~1s on the free endpoints; DeepSeek / Kimi are
    # strong but often queue past 60s there (2026-10-07), so they come after.
    "nvidia-build": [r"nemotron-3-ultra", r"nemotron-3-super", r"deepseek-v4", r"kimi-k3",
                     r"qwen3", r"llama-3\.3-70b"],
    "kilo-free": [r"nemotron-3-super", r"step-3", r"nemotron-3-ultra", r"ling-3", r"laguna-s",
                  r"dots-3", r"kilo-auto/free", r"north", r"openrouter/free"],
}
MODEL_SKIP = re.compile(
    r"embed|guard|safety|safeguard|retriever|reward|vlm|vision|omni|ocr|whisper|tts|clip|"
    r"parse|orpheus|allam|audio|image|coder-6|codellama|llama2|lfm-2|nano", re.I)
MODELS_PER_PROVIDER = 5          # failover tries at most this many per provider

# (provider, model) -> epoch seconds until it may be tried again.
# (provider, "*") blocks the whole provider (bad key).
_MODEL_HOLD: dict[tuple[str, str], float] = {}
_PERMANENT_HTTP = {400, 401, 402, 403, 404, 410, 422}
_HOLD_PERMANENT_S = 1800
_HOLD_TRANSIENT_S = 60
_HOLD_SLOW_S = 600               # a model that timed out is queueing; give it 10 minutes
TIMEOUT = "timeout"


def _hold(pname: str, model: str, status) -> None:
    """Remember a failure so the next call skips this model for a while."""
    cfg = LLM_PROVIDERS.get(pname, {})
    if status in _PERMANENT_HTTP:
        hold = _HOLD_PERMANENT_S
    elif status == TIMEOUT:
        hold = _HOLD_SLOW_S
    else:
        hold = _HOLD_TRANSIENT_S
    _MODEL_HOLD[(pname, model)] = time.time() + hold
    if status == 401 and not cfg.get("anonymous"):
        _MODEL_HOLD[(pname, "*")] = time.time() + _HOLD_PERMANENT_S


# Providers where only MODEL_RANK matches may ever be called, even if pinned
# from the sidebar: this is what keeps Gemini on free-tier models.
FREE_TIER_LOCKED = {"gemini-free"}


def _allowed(pname: str, model: str) -> bool:
    if pname not in FREE_TIER_LOCKED:
        return True
    return any(re.search(p, model, re.I) for p in MODEL_RANK.get(pname, []))


def _usable(pname: str, model: str) -> bool:
    now = time.time()
    return _MODEL_HOLD.get((pname, "*"), 0) < now and _MODEL_HOLD.get((pname, model), 0) < now


def _rank_models(pname: str, ids: list[str]) -> list[str]:
    cfg = LLM_PROVIDERS.get(pname, {})
    ids = [i for i in ids if not MODEL_SKIP.search(i)]
    if cfg.get("anonymous"):
        ids = [i for i in ids if "free" in i.lower()]
    ranked, seen = [], set()
    for pat in MODEL_RANK.get(pname, []):
        rx = re.compile(pat, re.I)
        for i in ids:
            if i not in seen and rx.search(i):
                ranked.append(i)
                seen.add(i)
    return ranked


def available_models(provider: str, free_only: bool = True) -> list[str]:
    """Live models for the picker: curated ids that are still advertised,
    then the rest of the live list ranked strongest-first. Falls back to the
    curated list when discovery fails."""
    cfg = LLM_PROVIDERS.get(provider, {})
    static = list(cfg.get("models", []))
    discovered = discover_models(provider)
    if not discovered:
        return static
    curated = [m for m in static if m in discovered and _allowed(provider, m)]
    ranked = _rank_models(provider, discovered)
    if free_only:
        ranked = [m for m in ranked if "free" in m.lower()] or ranked
    # Only ranked models are offered: providers advertise dozens of ids that
    # 404 or are not chat models (NVIDIA lists 80, a handful answer).
    rest = [m for m in ranked if m not in curated]
    return (curated + rest) or static


def provider_status() -> list[dict]:
    """One row per provider for the sidebar: key state and live model count."""
    rows = []
    for pname in LLM_PROVIDER_ORDER:
        cfg = LLM_PROVIDERS[pname]
        if cfg.get("anonymous"):
            key_state = "keyless"
        else:
            key_state = "key set" if get_key(cfg["key_env"]) else f"no key ({cfg['key_env']})"
        if key_state.startswith("no key") or not _usable(pname, "*"):
            live = []
        else:
            live = [m for m in available_models(pname) if _usable(pname, m)]
        rows.append({"provider": pname, "key": key_state, "live_models": len(live),
                     "first": live[0] if live else None})
    return rows


def _models_for(pname: str) -> list[str]:
    """Models the failover walks for one provider, without a network call:
    the ranked live list when discovery is cached, else the curated list."""
    cfg = LLM_PROVIDERS[pname]
    hit = _MODELS_CACHE.get(pname)
    if hit and time.time() - hit[0] < _MODELS_TTL:
        live = set(hit[1])
        models = [m for m in cfg["models"] if m in live]
        models += [m for m in _rank_models(pname, hit[1]) if m not in models]
    else:
        models = list(cfg["models"])
    return models


def _call_order(provider: str, model: str) -> list[tuple]:
    """Providers to try, pinned one first. A discovered model id is tried
    first on its provider even when it is not in the static list. Models
    recently seen failing are skipped, and each provider contributes at most
    MODELS_PER_PROVIDER, so one dead endpoint cannot stall an answer."""
    order = list(LLM_PROVIDER_ORDER)
    if provider in order:
        order.remove(provider)
        order.insert(0, provider)
    out = []
    for pname in order:
        cfg = LLM_PROVIDERS[pname]
        models = [m for m in _models_for(pname) if _usable(pname, m) and _allowed(pname, m)]
        if pname == provider and model and _allowed(pname, model):
            if model in models:
                models.remove(model)
            models.insert(0, model)
        out.append((pname, cfg, models[:MODELS_PER_PROVIDER]))
    return out


def warm_model_registry() -> None:
    """Fetch every provider's /models in parallel so the first answer already
    walks live, ranked models instead of the built-in fallback lists."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(LLM_PROVIDER_ORDER)) as ex:
        list(ex.map(discover_models, LLM_PROVIDER_ORDER))


def _post_chat(cfg: dict, key: str, payload: dict, stream: bool = False, timeout=(10, 45)):
    return requests.post(
        f"{cfg['base']}/chat/completions",
        headers=_provider_headers(cfg, key),
        json=payload, timeout=timeout, stream=stream,
    )


def llm_chat(messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 0, max_retries: int = 2,
             provider: str = "", model: str = "") -> dict:
    """Call the selected provider; on failure, fall back across providers.

    Permanent errors (bad model id, no access, gone) move straight to the next
    model and park this one for 30 minutes. Rate limits and server errors get
    one more try, then a 60-second hold."""
    errors = []
    for pname, cfg, models in _call_order(provider, model):
        key = _key_for(cfg)
        if not cfg.get("anonymous") and not key:
            errors.append(f"{pname}: no key")
            continue
        mt = max_tokens if max_tokens > 0 else cfg.get("max_tokens", 4096)
        payload = {"messages": messages, "max_tokens": mt, "temperature": temperature}
        for m in models:
            if not _usable(pname, m):
                continue
            for attempt in range(max_retries):
                status = None
                try:
                    r = _post_chat(cfg, key, {**payload, "model": m})
                    status = r.status_code
                    if status == 200:
                        data = r.json()
                        choices = data.get("choices") or []
                        content = (choices[0].get("message") or {}).get("content") if choices else ""
                        if content and content.strip():
                            return {
                                "ok": True,
                                "provider": pname,
                                "model": m,
                                "text": content,
                                "tokens": (data.get("usage") or {}).get("total_tokens", 0),
                            }
                        errors.append(f"{pname}/{m}: empty reply")
                        _hold(pname, m, None)
                        break
                    errors.append(f"{pname}/{m}: HTTP {status}")
                except requests.Timeout:
                    status = TIMEOUT
                    errors.append(f"{pname}/{m}: timed out")
                except Exception as e:
                    errors.append(f"{pname}/{m}: {str(e)[:60]}")
                if status in _PERMANENT_HTTP or status == TIMEOUT or attempt == max_retries - 1:
                    _hold(pname, m, status)
                    break
                time.sleep(1.5)
            if not _usable(pname, "*"):
                break                      # bad key: skip the rest of this provider

    return {"ok": False, "error": "; ".join(errors[-8:])}


def llm_chat_stream(messages: list[dict], temperature: float = 0.2,
                    max_tokens: int = 0,
                    provider: str = "", model: str = ""):
    """Generator: stream a completion from the provider fallback chain.

    Yields text chunks as they arrive (OpenAI-compatible SSE). If the chosen
    provider/model fails or returns empty, silently advances to the next in
    the chain, with the same failure memory as llm_chat. Records the winning
    provider/model in session_state so the caller can display it.
    """
    errors = []
    for pname, cfg, models in _call_order(provider, model):
        key = _key_for(cfg)
        if not cfg.get("anonymous") and not key:
            errors.append(f"{pname}: no key")
            continue
        mt = max_tokens if max_tokens > 0 else cfg.get("max_tokens", 4096)
        for m in models:
            if not _usable(pname, m):
                continue
            try:
                r = _post_chat(cfg, key, {
                    "model": m, "messages": messages, "max_tokens": mt,
                    "temperature": temperature, "stream": True,
                }, stream=True, timeout=120)
                if r.status_code != 200:
                    errors.append(f"{pname}/{m}: HTTP {r.status_code}")
                    _hold(pname, m, r.status_code)
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
                if "".join(chunks).strip():
                    st.session_state["_llm_provider"] = pname
                    st.session_state["_llm_model"] = m
                    return
                errors.append(f"{pname}/{m}: empty stream")
                _hold(pname, m, None)
            except requests.Timeout:
                errors.append(f"{pname}/{m}: timed out")
                _hold(pname, m, TIMEOUT)
            except Exception as e:
                errors.append(f"{pname}/{m}: {str(e)[:60]}")
                _hold(pname, m, None)

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
    # NSE sometimes ships today's row with no close yet; one NaN at the end
    # turns every "last value" downstream into NaN.
    if "Close" in df:
        df = df.dropna(subset=["Close"])
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


def rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. Shared by technical_facts, indicators and backtest_rule so
    all three report the same number."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 0))
    rsi = rsi.where(loss > 0, 100.0).where(gain > 0, 0.0)
    rsi.iloc[:n] = np.nan                  # not enough bars yet; don't let day 1 read as "0"
    return rsi


def technical_facts(ticker: str) -> dict:
    df = get_history(ticker, "1d", "6mo")
    if df.empty or len(df) < 30:
        return {"error": "not enough data"}
    close = df["Close"]
    rsi = rsi_series(close)
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


def indicators(ticker: str, kind: str = "all") -> dict:
    """Leading, lagging and volatility readings in one place, plus relative
    strength against the market's index.

    Lagging indicators confirm a trend after it has started (moving averages,
    MACD). Leading ones are momentum oscillators that can turn before price
    does (RSI, stochastic, rate of change, OBV) — and they give more false
    signals for exactly that reason. Neither says anything about whether a
    signal has paid in the past; that is backtest_rule's job."""
    df = get_history(ticker, "1d", "2y")
    if df.empty or len(df) < 60:
        return {"ticker": ticker, "error": "not enough daily history"}
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df else None
    last = float(close.iloc[-1])
    out = {"ticker": ticker, "price": round(last, 2), "as_of": str(df.index[-1].date())}
    kind = (kind or "all").lower()

    if kind in ("all", "lagging"):
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        lag = {
            "sma50": round(float(sma50.iloc[-1]), 2),
            "price_vs_sma50_pct": round((last / sma50.iloc[-1] - 1) * 100, 2),
            "macd": round(float(macd.iloc[-1]), 3),
            "macd_signal": round(float(signal.iloc[-1]), 3),
            "macd_state": "above signal" if macd.iloc[-1] > signal.iloc[-1] else "below signal",
        }
        if sma200.notna().iloc[-1]:
            above = sma50 > sma200
            flips = above.ne(above.shift()) & sma200.notna()
            lag["sma200"] = round(float(sma200.iloc[-1]), 2)
            lag["price_vs_sma200_pct"] = round((last / sma200.iloc[-1] - 1) * 100, 2)
            lag["sma50_vs_sma200"] = "golden cross (50 above 200)" if above.iloc[-1] else "death cross (50 below 200)"
            if flips.any():
                lag["days_since_last_cross"] = int((df.index[-1] - flips[flips].index[-1]).days)
        out["lagging"] = lag

    if kind in ("all", "leading"):
        rsi = rsi_series(close)
        lo14, hi14 = low.rolling(14).min(), high.rolling(14).max()
        stoch_k = 100 * (close - lo14) / (hi14 - lo14)
        r = float(rsi.iloc[-1])
        lead = {
            "rsi14": round(r, 1),
            "rsi_zone": "oversold (<30)" if r < 30 else "overbought (>70)" if r > 70 else "neutral",
            "stochastic_k14": round(float(stoch_k.iloc[-1]), 1),
            "stochastic_d3": round(float(stoch_k.rolling(3).mean().iloc[-1]), 1),
            "roc_20d_pct": round(float(last / close.iloc[-21] - 1) * 100, 2),
        }
        if volume is not None and volume.iloc[-60:].sum() > 0:
            # OBV divergence: volume flow and price disagreeing over the last
            # 20 sessions is the classic early warning that a move is thin.
            obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
            obv_up = obv.iloc[-1] > obv.iloc[-21]
            px_up = close.iloc[-1] > close.iloc[-21]
            lead["obv_20d"] = "rising" if obv_up else "falling"
            lead["obv_price_divergence"] = (
                "none" if obv_up == px_up
                else "bearish (price up, volume flow down)" if px_up
                else "bullish (price down, volume flow up)")
        out["leading"] = lead

    if kind in ("all", "volatility"):
        prev = close.shift()
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        mid, sd = close.rolling(20).mean(), close.rolling(20).std()
        bbw = (4 * sd / mid * 100).dropna()
        out["volatility"] = {
            "atr14": round(float(atr.iloc[-1]), 2),
            "atr_pct_of_price": round(float(atr.iloc[-1] / last * 100), 2),
            "bollinger_width_pct": round(float(bbw.iloc[-1]), 2),
            # Where today's band width sits in the last year: a low percentile
            # is a squeeze, which tends to come before a bigger move.
            "bollinger_width_percentile_1y": round(float((bbw.iloc[-252:] <= bbw.iloc[-1]).mean() * 100), 0),
            "realised_vol_20d_annual_pct": round(float(close.pct_change().iloc[-20:].std() * np.sqrt(252) * 100), 1),
        }

    bench = benchmark_for(ticker)
    if bench and kind in ("all", "relative"):
        b = get_history(bench, "1d", "2y")
        if not b.empty:
            both = pd.concat({"s": close, "b": b["Close"]}, axis=1).dropna()
            rs = {"benchmark": bench}
            for label, n in (("1m", 21), ("3m", 63), ("6m", 126)):
                if len(both) > n:
                    s_ret = both["s"].iloc[-1] / both["s"].iloc[-n - 1] - 1
                    b_ret = both["b"].iloc[-1] / both["b"].iloc[-n - 1] - 1
                    rs[f"excess_return_{label}_pct"] = round((s_ret - b_ret) * 100, 2)
            out["relative_strength"] = rs
    return out


# Cost per side as a fraction of trade value, slippage included. India
# delivery: STT 0.1% on buy and on sell, stamp duty 0.015% on the buy,
# exchange + SEBI + GST a few thousandths of a percent, plus ~0.03% slippage
# (zero-brokerage broker assumed). The budget changes STT now and then, so
# these are defaults, not a quote.
TRADE_COSTS = {"india": 0.0015, "us": 0.0005, "crypto": 0.0010, "other": 0.0005}
# 1m is about 7 days on Yahoo, 5m and 15m about 60. The horizon is the cone
# ahead: 30 minutes, one hour, two hours. The cost is a spread per side.
# Delivery STT on every one-minute flip is not what an intraday bar pays.
INTRADAY_INTERVALS = ("1m", "5m", "15m")
INTRADAY_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d"}
INTRADAY_HORIZON_BARS = {"1m": 30, "5m": 12, "15m": 8}
INTRADAY_COSTS = {"india": 0.0002, "us": 0.0001, "crypto": 0.0005, "other": 0.0002}

BACKTEST_RULES = {
    "rsi_oversold": "buy when RSI(14) < entry (30), sell when RSI > exit (50)",
    "ma_cross": "hold while SMA(fast=50) is above SMA(slow=200)",
    "breakout": "buy on a close above the prior entry-day high (55), sell on a close below the prior exit-day low (20)",
    "macd_cross": "hold while MACD is above its signal line",
}


def _rule_positions(df: pd.DataFrame, rule: str, p: dict) -> pd.Series:
    """1 on days the rule wants to be long at the close, else 0."""
    close = df["Close"]
    if rule == "rsi_oversold":
        rsi = rsi_series(close)
        entry, exit_ = float(p.get("entry", 30)), float(p.get("exit", 50))
        pos, held = [], 0
        for v in rsi.to_numpy():
            if not held and v < entry:
                held = 1
            elif held and v > exit_:
                held = 0
            pos.append(held)
        return pd.Series(pos, index=close.index)
    if rule == "ma_cross":
        fast = close.rolling(int(p.get("fast", 50))).mean()
        slow = close.rolling(int(p.get("slow", 200))).mean()
        return (fast > slow).astype(int)
    if rule == "breakout":
        hi = df["High"].rolling(int(p.get("entry", 55))).max().shift()
        lo = df["Low"].rolling(int(p.get("exit", 20))).min().shift()
        pos, held = [], 0
        for c, h, l in zip(close.to_numpy(), hi.to_numpy(), lo.to_numpy()):
            if not held and c > h:
                held = 1
            elif held and c < l:
                held = 0
            pos.append(held)
        return pd.Series(pos, index=close.index)
    if rule == "macd_cross":
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        return (macd > macd.ewm(span=9, adjust=False).mean()).astype(int)
    raise ValueError(f"unknown rule {rule}; choose from {list(BACKTEST_RULES)}")


def _segment_stats(close: pd.Series, held: pd.Series, cost: float,
                   bench: pd.Series | None) -> dict:
    """Trade list and summary for one slice of history."""
    ret = close.pct_change().fillna(0)
    sides = held.diff().abs().fillna(held.iloc[0])
    equity = (1 + held * ret - sides * cost).cumprod()
    trades = []
    block = (held != held.shift()).cumsum()
    for _, days in held[held == 1].groupby(block[held == 1]):
        i0 = close.index.get_loc(days.index[0])
        entry = close.iloc[max(i0 - 1, 0)]
        exit_ = close.loc[days.index[-1]]
        trades.append(float(exit_ / entry - 1 - 2 * cost))
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    stats = {
        "from": str(close.index[0].date()),
        "to": str(close.index[-1].date()),
        "strategy_return_pct": round((equity.iloc[-1] - 1) * 100, 2),
        "buy_and_hold_pct": round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win_pct": round(np.mean(wins) * 100, 2) if wins else None,
        "avg_loss_pct": round(np.mean(losses) * 100, 2) if losses else None,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if wins and losses and sum(losses) < 0 else None,
        "max_drawdown_pct": round(float((equity / equity.cummax() - 1).min() * 100), 2),
        "time_in_market_pct": round(float(held.mean() * 100), 1),
    }
    if bench is not None:
        b = bench.reindex(close.index).dropna()
        if len(b) > 1:
            stats["index_return_pct"] = round((b.iloc[-1] / b.iloc[0] - 1) * 100, 2)
    return stats


def backtest_rule(ticker: str, rule: str = "rsi_oversold", params: dict | None = None,
                  years: int = 5, cost_per_side: float | None = None) -> dict:
    """Has this setup actually paid on this stock? Long-only, daily bars,
    with costs, judged separately on the first 70% of history (where you'd
    have found the idea) and the last 30% (which the rule never saw).

    The signal is read at a day's close and the trade is assumed filled at
    that same close — optimistic by a few ticks, which the slippage in the
    cost table is meant to cover."""
    params = params or {}
    df = get_history(ticker, "1d", f"{int(years)}y")
    if df.empty or len(df) < 300:
        return {"ticker": ticker, "error": "need at least ~300 daily bars to split in/out of sample"}
    try:
        pos = _rule_positions(df, rule, params)
    except ValueError as e:
        return {"ticker": ticker, "error": str(e)}
    held = pos.shift(1).fillna(0)          # today's P&L comes from yesterday's decision
    market = market_of(ticker)
    cost = TRADE_COSTS[market] if cost_per_side is None else float(cost_per_side)
    bench_sym = benchmark_for(ticker)
    bench = None
    if bench_sym:
        b = get_history(bench_sym, "1d", f"{int(years)}y")
        bench = None if b.empty else b["Close"]

    close = df["Close"]
    cut = int(len(close) * 0.7)
    ins = _segment_stats(close.iloc[:cut], held.iloc[:cut], cost, bench)
    oos = _segment_stats(close.iloc[cut:], held.iloc[cut:], cost, bench)

    if (oos["trades"] or 0) < 5:
        verdict = "too few trades in the out-of-sample period to judge"
    elif oos["strategy_return_pct"] > oos["buy_and_hold_pct"] and (oos["profit_factor"] or 0) > 1:
        verdict = "held up out of sample: beat buy-and-hold after costs"
    elif ins["strategy_return_pct"] > ins["buy_and_hold_pct"]:
        verdict = "worked in sample but did not survive out of sample"
    else:
        verdict = "no edge after costs: buy-and-hold did better in both periods"

    return {
        "ticker": ticker,
        "rule": rule,
        "rule_description": BACKTEST_RULES.get(rule),
        "params": params,
        "market": market,
        "cost_per_side_pct": round(cost * 100, 3),
        "benchmark": bench_sym,
        "in_sample": ins,
        "out_of_sample": oos,
        "verdict": verdict,
        "method": "long-only daily backtest, costs and slippage per side, 70/30 chronological split; past results, not a recommendation",
    }


def _block_bootstrap_mean_ci(values, n_boot: int = 800, alpha: float = 0.05,
                             seed: int = 0) -> tuple[float, float] | None:
    """95% (or tighter) interval for the mean, resampling in blocks so a
    streak of up days counts as one draw. Returns None when the sample is
    too short to mean anything."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 30:
        return None
    block = max(5, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)) % n
    means = x[idx].reshape(n_boot, -1)[:, :n].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _judge_excess(oos: dict, excess: np.ndarray, n_trials: int) -> dict:
    """Kill a rule that loses to buy-and-hold, has too few trades, or whose
    daily edge still includes zero after a block bootstrap. Extra rules tried
    in the same session tighten the interval (Bonferroni)."""
    n_trials = max(int(n_trials), 1)
    alpha = 0.05 / n_trials
    killed = []
    # Five closes the same gate as the backtest text. A 50/200 cross often
    # trades only a handful of times in the last 30% of five years; twenty
    # would reject every trend rule before the return test gets a say.
    if (oos.get("trades") or 0) < 5:
        killed.append("too_few_trades")
    if oos.get("strategy_return_pct", 0) <= oos.get("buy_and_hold_pct", 0):
        killed.append("lost_to_buy_and_hold")
    ci = _block_bootstrap_mean_ci(excess, alpha=alpha)
    if ci is None:
        killed.append("not_enough_oos_days")
    elif ci[0] <= 0 <= ci[1]:
        killed.append("excess_not_significant")
    mean = float(np.nanmean(excess)) if len(excess) else 0.0
    std = float(np.nanstd(excess, ddof=1)) if len(excess) > 2 else 0.0
    t_stat = mean / (std / np.sqrt(len(excess))) if std > 0 and len(excess) > 2 else 0.0
    return {
        "killed_by": killed,
        "alpha": alpha,
        "ci": ci,
        "mean": mean,
        "t_stat": t_stat,
        "n": int(len(excess)),
    }


def _range_calibration(close: pd.Series, horizon: int = 7, lookback: int = 63,
                       eval_days: int = 252) -> dict:
    """How often a band built only from past data contained the later close,
    and how often the sign of a clear trailing drift matched the later move.

    Rolling mean and volatility, not a fresh GARCH fit each day: one fit per
    day would take minutes, and this scores the same 1.04 / 1.96 band shape
    the live cone uses. A day counts as a drift call only when the lookback
    mean is at least one standard error from zero.
    """
    px = close.dropna().astype(float)
    ret = px.pct_change()
    horizon = max(int(horizon), 1)
    rows = []
    for i in range(lookback, len(px) - horizon):
        window = ret.iloc[i - lookback + 1:i + 1].dropna()
        if len(window) < lookback // 2:
            continue
        mu, sig = float(window.mean()), float(window.std(ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            continue
        t_stat = mu / (sig / np.sqrt(len(window)))
        last, fut = float(px.iloc[i]), float(px.iloc[i + horizon])
        drift, vol = mu * horizon, sig * np.sqrt(horizon)
        lo70, hi70 = last * (1 + drift - 1.04 * vol), last * (1 + drift + 1.04 * vol)
        lo95, hi95 = last * (1 + drift - 1.96 * vol), last * (1 + drift + 1.96 * vol)
        pred = 1 if t_stat > 1 else (-1 if t_stat < -1 else 0)
        rows.append((lo70 <= fut <= hi70, lo95 <= fut <= hi95, pred, fut > last))
    if len(rows) > eval_days:
        rows = rows[-eval_days:]
    if len(rows) < 40:
        return {"error": "not enough history to score the bands", "n": len(rows)}
    hit70, hit95, pred, up = (np.array([r[i] for r in rows]) for i in range(4))
    called = pred != 0
    n_called = int(called.sum())
    correct = np.zeros(len(rows), dtype=bool)
    correct[called] = ((pred[called] == 1) & up[called]) | ((pred[called] == -1) & ~up[called])
    hit = float(correct[called].mean()) if n_called else None
    base = float(up.mean())
    usable = False
    if n_called >= 30 and hit is not None:
        score = np.where(called, correct.astype(float), np.nan)
        ci = _block_bootstrap_mean_ci(score[called], alpha=0.05, seed=1)
        usable = bool(ci and ci[0] > base and hit > base)
    return {
        "n": int(len(rows)),
        "horizon_days": horizon,
        "band70_hit_rate": round(float(hit70.mean()), 3),
        "band95_hit_rate": round(float(hit95.mean()), 3),
        "drift_calls": n_called,
        "drift_sign_hit_rate": round(hit, 3) if hit is not None else None,
        "up_rate_base": round(base, 3),
        "drift_usable": usable,
        "method": "rolling 63-day mean and volatility, last year scored, horizon as given",
    }


def _direction_from_tests(position_long: bool, killed_by: list) -> str:
    if killed_by:
        return "no-edge"
    return "long" if position_long else "flat"


def _intraday_cone(close: pd.Series, horizon_bars: int) -> dict:
    """A short forward band from the last 63 bars. No daily GARCH fit."""
    px = close.dropna().astype(float)
    horizon_bars = max(int(horizon_bars), 1)
    window = px.pct_change().dropna().iloc[-63:]
    if len(window) < 20:
        return {"error": "not enough bars for an intraday band"}
    mu, sig = float(window.mean()), float(window.std(ddof=1))
    if not np.isfinite(sig) or sig <= 0:
        return {"error": "intraday volatility is zero"}
    last = float(px.iloc[-1])
    drift, vol = mu * horizon_bars, sig * np.sqrt(horizon_bars)
    return {
        "center_est": last * (1 + drift),
        "range_70": [last * (1 + drift - 1.04 * vol), last * (1 + drift + 1.04 * vol)],
        "range_95": [last * (1 + drift - 1.96 * vol), last * (1 + drift + 1.96 * vol)],
        "horizon": horizon_bars,
        "horizon_bars": horizon_bars,
        "method": "rolling 63-bar mean and volatility",
    }


def compute_verdict_from_frame(df: pd.DataFrame, rule: str, params: dict | None,
                               cost: float, horizon_days: int, n_trials: int,
                               benchmark: pd.Series | None = None,
                               forecast: dict | None = None,
                               ticker: str = "", interval: str = "1d") -> dict:
    """Direction from a tested rule, plus whether the forward band has been honest.

    direction is long or flat only when the out-of-sample edge survives costs,
    a minimum trade count, and a block-bootstrap interval. Otherwise it is
    no-edge, even if the rule wants to be long today.
    """
    params = params or {}
    if rule not in BACKTEST_RULES:
        return {"ticker": ticker, "error": f"unknown rule {rule}; choose from {list(BACKTEST_RULES)}"}
    if df is None or df.empty or len(df) < 300 or "Close" not in df:
        return {"ticker": ticker, "interval": interval,
                "error": f"need at least 300 {interval} bars"}
    try:
        pos = _rule_positions(df, rule, params)
    except ValueError as e:
        return {"ticker": ticker, "error": str(e)}
    held = pos.shift(1).fillna(0)
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0)
    sides = held.diff().abs().fillna(held.iloc[0])
    strat = held * ret - sides * cost
    excess = (strat - ret).to_numpy()
    cut = int(len(close) * 0.7)
    oos_stats = _segment_stats(close.iloc[cut:], held.iloc[cut:], cost, benchmark)
    ins_stats = _segment_stats(close.iloc[:cut], held.iloc[:cut], cost, benchmark)
    judged = _judge_excess(oos_stats, excess[cut:], n_trials)
    position_long = bool(pos.iloc[-1] == 1)
    direction = _direction_from_tests(position_long, judged["killed_by"])

    oos_ret = ret.iloc[cut:]
    oos_held = held.iloc[cut:]
    long_days = oos_ret[oos_held == 1]
    flat_days = oos_ret[oos_held == 0]
    base_up = float((oos_ret > 0).mean()) if len(oos_ret) else 0.0
    up_when_long = float((long_days > 0).mean()) if len(long_days) else None
    lift = (up_when_long - base_up) if up_when_long is not None else None

    cal = _range_calibration(close, horizon_days)
    fc = forecast or {}
    last = float(close.iloc[-1])
    center = fc.get("center_est")
    band95 = fc.get("range_95") or [None, None]
    upside = downside = None
    bias = None
    if center is not None and band95[0] is not None:
        upside = (band95[1] - last) / last * 100
        downside = (last - band95[0]) / last * 100
        if center > last * 1.001:
            bias = "up"
        elif center < last * 0.999:
            bias = "down"
        else:
            bias = "flat"

    ci = judged["ci"]
    excess_bps = judged["mean"] * 1e4
    intraday = interval in INTRADAY_INTERVALS
    summary = _verdict_summary(
        direction, rule, position_long, judged["killed_by"], excess_bps, lift,
        cal, upside, downside, bias, n_trials,
        unit="bar" if intraday else "day",
        window_phrase=f"this {interval} window" if intraday else "the last year",
    )
    as_of = close.index[-1]
    as_of_s = as_of.strftime("%Y-%m-%d %H:%M") if intraday else str(as_of.date())
    return {
        "ticker": ticker,
        "interval": interval,
        "rule": rule,
        "rule_description": BACKTEST_RULES[rule],
        "as_of": as_of_s,
        "direction": direction,
        "rule_position": "long" if position_long else "flat",
        "follow_rule": direction != "no-edge",
        "killed_by": judged["killed_by"],
        "trials": int(max(n_trials, 1)),
        "ci_level": round(1 - judged["alpha"], 4),
        "oos": {
            "strategy_return_pct": oos_stats["strategy_return_pct"],
            "buy_and_hold_pct": oos_stats["buy_and_hold_pct"],
            "trades": oos_stats["trades"],
            "excess_mean_bps": round(excess_bps, 2),
            "t_stat": round(judged["t_stat"], 2),
            "bootstrap_ci_bps": [round(ci[0] * 1e4, 2), round(ci[1] * 1e4, 2)] if ci else None,
            "days": judged["n"],
        },
        "in_sample_return_pct": ins_stats["strategy_return_pct"],
        "direction_stats": {
            "oos_days_long": int(len(long_days)),
            "mean_return_when_long_bps": round(float(long_days.mean()) * 1e4, 2) if len(long_days) else None,
            "mean_return_when_flat_bps": round(float(flat_days.mean()) * 1e4, 2) if len(flat_days) else None,
            "up_rate_when_long": round(up_when_long, 3) if up_when_long is not None else None,
            "up_rate_base": round(base_up, 3),
            "lift": round(lift, 3) if lift is not None else None,
        },
        "forecast": {
            "horizon_days": int(horizon_days),
            "horizon_bars": int((fc or {}).get("horizon_bars") or (horizon_days if intraday else 0)) or None,
            "center_est": center,
            "range_70": fc.get("range_70"),
            "range_95": fc.get("range_95"),
            "price_bias": bias,
            "upside_95_pct": round(upside, 2) if upside is not None else None,
            "downside_95_pct": round(downside, 2) if downside is not None else None,
            **{k: cal.get(k) for k in (
                "band70_hit_rate", "band95_hit_rate", "drift_sign_hit_rate",
                "drift_calls", "drift_usable", "up_rate_base", "method", "error", "n",
            )},
        },
        "summary": summary,
        "method": (
            f"{interval} bars, spread {cost * 100:.3f}% per side, out-of-sample excess, "
            "block-bootstrap interval"
            if intraday else
            "daily bars, delivery costs, out-of-sample excess, block-bootstrap interval; "
            "live cone is GARCH(1,1)-AR(1); band hit rate is a rolling-vol walk-forward"
        ),
    }


def _verdict_summary(direction, rule, position_long, killed_by, excess_bps, lift,
                     cal, upside, downside, bias, n_trials,
                     unit: str = "day", window_phrase: str = "the last year") -> str:
    position = "long" if position_long else "flat"
    lift_pp = (lift or 0) * 100
    band70 = cal.get("band70_hit_rate")
    if direction == "no-edge":
        why = ", ".join(killed_by) or "no measured edge"
        text = (f"No tested direction on {rule}: killed by {why}. "
                f"The rule is {position} at the last close, and that position is not backed by the out-of-sample test.")
    else:
        side = "long" if direction == "long" else "flat, stay out"
        text = (f"Tested direction: {side}. The {rule} rule is {position} at the last close. "
                f"Out-of-sample excess over buy-and-hold is {excess_bps:.1f} bps per {unit}. "
                f"Days in the trade beat the base rate of up days by {lift_pp:.1f} percentage points.")
    if upside is not None and downside is not None:
        text += (f" The 95% band leaves {upside:.1f}% of room above the last price and {downside:.1f}% below"
                 f" (center bias {bias}).")
    if band70 is not None:
        text += f" A 70% band built from past data contained the later close on {band70:.0%} of {window_phrase}."
    if cal.get("drift_usable"):
        text += " The trailing drift's direction has also held up on that window."
    elif cal.get("drift_sign_hit_rate") is not None:
        text += " The trailing drift's direction has not been reliable, so the stance comes from the rule test."
    if n_trials > 1:
        text += f" Judged against {n_trials} rules tried in this session."
    return text


def _session_trials(ticker: str, rule: str, interval: str = "1d") -> int:
    """How many distinct ticker/rule/interval triples verdict has scored.
    Repeating the same triple does not tighten the bar again."""
    try:
        seen = st.session_state.setdefault("_verdict_rules", [])
        token = f"{ticker}|{rule}|{interval}"
        if token not in seen:
            seen.append(token)
        return len(seen)
    except Exception:
        return 1


@st.cache_data(ttl=3600, show_spinner=False)
def _verdict_cached(ticker: str, rule: str, params_json: str, years: int,
                    horizon_days: int, n_trials: int, interval: str = "1d") -> dict:
    params = json.loads(params_json) if params_json else {}
    intraday = interval in INTRADAY_INTERVALS
    if intraday:
        df = get_history(ticker, interval, INTRADAY_PERIOD[interval])
        forecast = _intraday_cone(df["Close"], horizon_days) if not df.empty and "Close" in df else {}
        if forecast.get("error"):
            forecast = {}
        bench = None
        cost = INTRADAY_COSTS[market_of(ticker)]
    else:
        df = get_history(ticker, "1d", f"{int(years)}y")
        bench_sym = benchmark_for(ticker)
        bench = None
        if bench_sym:
            b = get_history(bench_sym, "1d", f"{int(years)}y")
            bench = None if b.empty else b["Close"]
        forecast = forecast_range(ticker, horizon_days)
        if forecast.get("error"):
            forecast = {}
        cost = TRADE_COSTS[market_of(ticker)]
    return compute_verdict_from_frame(
        df, rule, params, cost, horizon_days, n_trials,
        bench, forecast, ticker, interval,
    )


def verdict(ticker: str, rule: str = "rsi_oversold", params: dict | None = None,
            years: int = 5, horizon_days: int = 7, n_trials: int | None = None,
            interval: str = "1d") -> dict:
    """Tested direction for one rule on one symbol. Python computes it.

    interval 1d uses five years and a GARCH cone. 1m, 5m, and 15m use the
    bars Yahoo actually keeps and a rolling-volatility cone a short way ahead.
    """
    if interval not in ("1d",) + INTRADAY_INTERVALS:
        interval = "1d"
    if interval in INTRADAY_HORIZON_BARS:
        horizon_days = INTRADAY_HORIZON_BARS[interval]
    else:
        horizon_days = min(max(int(horizon_days), 1), 20)
    trials = _session_trials(ticker, rule, interval) if n_trials is None else max(int(n_trials), 1)
    try:
        return _verdict_cached(ticker, rule, json.dumps(params or {}, sort_keys=True),
                               int(years), horizon_days, trials, interval)
    except Exception as e:
        return {"ticker": ticker, "rule": rule, "interval": interval, "error": str(e)[:200]}


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
def pair_correlation(ticker_a: str, ticker_b: str, period: str = "1y", lag: int = 0) -> dict:
    """Deterministic two-ticker link metrics computed in-app with pandas (no
    LLM-written code): aligned-close correlation, both beta directions, and
    current prices. Never fails the way LLM code can.

    lag > 0 adds shifted correlations: does today's move in A line up with
    B's move 1..lag days later (and the other way round)? That's a hint about
    who moves first, not proof — a real signal still has to pass backtest_rule."""
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
    lead_lag = {}
    for k in range(1, min(int(lag), 10) + 1):
        lead_lag[f"a_today_vs_b_in_{k}d"] = round(float(df["a"].corr(df["b"].shift(-k))), 3)
        lead_lag[f"b_today_vs_a_in_{k}d"] = round(float(df["b"].corr(df["a"].shift(-k))), 3)
    out = {
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
    if lead_lag:
        out["lead_lag_correlations"] = lead_lag
        ma, mb = market_of(ticker_a), market_of(ticker_b)
        if ma != mb or {ma, mb} & {"other", "crypto"}:   # futures/FX/crypto: near-24h sessions
            # NSE shuts at 15:30 IST, US futures and FX trade on after that, so
            # "the same date" is a different moment for each. A 1-day lag
            # correlation across such pairs is often just that clock gap.
            out["timing_caveat"] = ("these markets close at different times; a 1-day lag "
                                    "correlation can come from the closing-time gap rather "
                                    "than one market leading the other — treat it as a "
                                    "hypothesis to test, not a signal")
    return out


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
        ind = indicators(ticker)
        if "error" not in ind:
            bundle["indicators"] = ind
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
                   provider: str = "", model: str = "",
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
                    provider: str = "", model: str = "") -> dict:
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
    "Bajaj Finance looks oversold. Has buying RSI dips actually worked on it?",
    "HDFC Bank: leading vs lagging indicators, and how is it doing against NIFTY?",
    "Does crude oil lead the Indian rupee? Check a 5-day lag",
]


def run_tool(name: str, args: dict, provider: str = "", model: str = "") -> dict:
    ticker = normalize_ticker(str(args.get("ticker", "")))
    if not ticker:
        return {"error": "no ticker"}
    if name == "indicators":
        return indicators(ticker, args.get("kind", "all"))
    if name == "backtest_rule":
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        return backtest_rule(ticker, args.get("rule", "rsi_oversold"), params,
                             int(args.get("years", 5)))
    if name == "verdict":
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        raw_trials = args.get("n_trials")
        return verdict(
            ticker, args.get("rule") or "rsi_oversold", params,
            int(args.get("years") or 5), int(args.get("horizon_days") or 7),
            None if raw_trials in (None, "") else int(raw_trials),
            str(args.get("interval") or "1d"),
        )
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
        tb = normalize_ticker(str(args.get("ticker_b", "")))
        if not tb:
            return {"error": "pair_correlation needs both ticker (a) and ticker_b (b)"}
        return pair_correlation(ticker, tb, str(args.get("period", "1y")),
                                lag=int(args.get("lag", 0) or 0))
    if name == "chart_spec":
        spec_args = dict(args)
        spec_args["ticker"] = ticker
        return set_chart_spec(spec_args)
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
        "description": "Deterministic correlation + beta between two tickers (computed in-app with pandas, cannot fail). Use for correlation/beta/co-movement/link questions between two assets instead of run_analysis. Args: ticker = first symbol, ticker_b = second symbol. Set lag (1-10) for 'does A lead B' / 'who moves first' questions: adds correlations of today's move in one with the other's move 1..lag days later.",
        "args": {"ticker": "e.g. AAPL", "ticker_b": "e.g. MSFT", "lag": 0, "period": "1y (or 2y/5y)"},
    },
    {
        "name": "risk_metrics",
        "description": "Deterministic 1-year risk snapshot: daily and annual volatility, 1-day historical VaR 95%, max drawdown, Sharpe. Use for risk / VaR / drawdown / volatility questions instead of run_analysis.",
        "args": {"ticker": "string"},
    },
    {
        "name": "forecast_range",
        "description": "Volatility-based 70% and 95% price ranges over the next N trading days (GARCH(1,1) on daily returns). A range, never a direction call. Use for outlook / forecast / 'where could it be in a week' questions.",
        "args": {"ticker": "string", "horizon_days": 5},
    },
    {
        "name": "indicators",
        "description": "Leading indicators (RSI, stochastic, 20-day rate of change, OBV divergence), lagging indicators (50/200-day moving averages, golden/death cross, MACD), volatility (ATR, Bollinger width and squeeze percentile, realised vol) and relative strength vs the market index (NIFTY 50 for Indian stocks, S&P 500 for US). kind: all/leading/lagging/volatility/relative.",
        "args": {"ticker": "e.g. TATAMOTORS.NS", "kind": "all"},
    },
    {
        "name": "backtest_rule",
        "description": "Test whether a trading setup has actually paid on this ticker: long-only daily backtest with realistic costs (Indian STT/stamp duty for NSE/BSE), split into in-sample (first 70%) and out-of-sample (last 30%), compared with buy-and-hold and the index. Use for 'is this a good entry', 'should I buy', 'does RSI/MA/breakout work on X' questions. rule: rsi_oversold (params entry, exit), ma_cross (fast, slow), breakout (entry, exit lookback days), macd_cross.",
        "args": {"ticker": "string", "rule": "rsi_oversold", "params": {"entry": 30, "exit": 50}, "years": 5},
    },
    {
        "name": "verdict",
        "description": "Tested direction for one rule: long, flat, or no-edge. interval is 1d (five years, GARCH cone), or 1m, 5m, 15m (the bars Yahoo keeps, a short rolling-vol cone). Call this for buy, sell, hold, outlook, or 'which way' questions, including intraday. Quote direction and summary. Do not invent a different direction. rule: rsi_oversold, ma_cross, breakout, macd_cross.",
        "args": {"ticker": "string", "rule": "rsi_oversold", "interval": "1d", "years": 5, "horizon_days": 7},
    },
    {
        "name": "chart_spec",
        "description": "Tell the price chart which symbol to draw and which layers to show. Python draws the candles. Use when the user asks to chart, plot, or compare two symbols. range is 1M, 3M, 6M, 1Y, or 5Y. ticker_b is an optional second Yahoo symbol drawn rebased to 100. overlays: sma20, sma50, sma200, bb. panels: volume, rsi, macd. marks=backtest plus a rule (rsi_oversold, ma_cross, breakout, macd_cross) draws that rule's trades and the 70/30 split.",
        "args": {"ticker": "string", "range": "1Y", "ticker_b": "", "overlays": ["sma50", "sma200", "bb"], "panels": ["volume", "rsi", "macd"], "marks": "none", "rule": ""},
    },
    {
        "name": "run_analysis",
        "description": "Write and execute a Python script (pandas/numpy/yfinance) to compute ANY quantitative answer: volatility, VaR, Sharpe, RSI, MACD, correlation, regression, backtest, GARCH volatility projection, price ranges. Returns real computed numbers via Python execution.",
        "args": {"ticker": "string", "question": "the user's full question to compute"},
    },
]

def _parse_json_obj(raw: str) -> dict | None:
    """First JSON object in an LLM reply, tolerating code fences and chatter."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


AGENT_MAX_STEPS = 6     # LLM turns before we stop and answer with what we have
AGENT_MAX_CALLS = 14    # tool calls across all turns

AGENT_LOOP_PROMPT = (
    "You are a market research agent that works in steps. Each turn you see the "
    "user's question and every tool result so far, then decide the next move. "
    "Reply with ONLY one JSON object, either\n"
    '{"thought": "<under 40 words: what the last results showed and what you need next>", '
    '"tools": [{"name": "...", "args": {...}}]}   (1-3 calls, run this turn)\n'
    "or\n"
    '{"thought": "<why the evidence now answers the question>", "done": true}\n\n'
    "How to work:\n"
    "- Look, then decide. Start with what tells you what you are dealing with, "
    "then follow up on what the results actually show. If indicators show RSI "
    "under 30, backtest rsi_oversold; if a trend is in place, backtest ma_cross or "
    "breakout; if a backtest has too few trades, retry with years=10 or another "
    "rule; if a symbol returns an error, retry with the company name.\n"
    "- Never repeat a call with the same arguments.\n"
    "- Prefer the deterministic tools (indicators, risk_metrics, forecast_range, "
    "backtest_rule, verdict, pair_correlation) over get_history + run_analysis whenever they "
    "cover the question: they cannot miscompute. 'Outperforming the market / vs "
    "NIFTY / vs the S&P' is indicators (relative_strength).\n"
    "- Stay on the symbols the user asked about. If one has no data after one "
    "retry, stop and report that; never switch to the UI-selected asset or any "
    "other ticker in its place.\n"
    "- ticker takes any yfinance symbol (AAPL, BTC-USD, GC=F, ^NSEI, INR=X). Indian "
    "stocks use .NS (HAL.NS). If unsure, put the company name and it is looked up.\n"
    "- Buy, sell, hold, exit, outlook, or 'which way' questions: call verdict on the "
    "matching rule (oversold -> rsi_oversold, trend -> ma_cross, breakout -> breakout, "
    "MACD -> macd_cross). verdict.direction is the answer: long, flat, or no-edge. "
    "If the user says 1 minute, 5 minute, or 15 minute, set interval to 1m, 5m, or 15m. "
    "Otherwise leave interval at 1d. Also call chart_spec with marks=backtest and that same rule. "
    "Do not pick a direction yourself, and do not name a position size.\n"
    "- Correlation, beta, 'does A lead B': pair_correlation (set lag for lead questions), "
    "and chart_spec with ticker_b so both series are on the chart.\n"
    "- A request to chart or plot a symbol: chart_spec for that symbol. Python draws it.\n"
    "- Requests for other stock ideas: get_quote + fundamental_facts on several real "
    "candidates first, so any idea comes from data.\n"
    "- run_analysis is for computations no other tool covers; give it a precise question.\n"
    "- Say done as soon as the evidence answers the question. The written answer "
    "comes in a later step, not here."
)


def agent_loop(question: str, tickers: list[str], selected: str,
               provider: str = "", model: str = "",
               on_step=None) -> tuple[list[dict], list[dict]]:
    """The agent's control loop. Every turn the LLM re-reads the question and
    all observations so far, then picks the next tool calls or stops. Python
    only executes and records: it never decides what gets called.

    Returns (results, trace). results feed the final answer; trace is the
    step-by-step reasoning shown in the UI. on_step(step) fires after each turn."""
    manifest_text = "\n".join(
        f"- {t['name']}({json.dumps(t['args'])}) — {t['description']}" for t in AGENT_TOOL_MANIFEST
    )
    ticker_ctx = ", ".join(tickers) if tickers else "none detected"
    valid = {t["name"] for t in AGENT_TOOL_MANIFEST}
    primary = tickers[0] if tickers else selected
    results, trace, observations, seen = [], [], [], set()

    for step in range(1, AGENT_MAX_STEPS + 1):
        user = (
            f"User question: {question}\n"
            f"Ticker(s) detected in question: {ticker_ctx}\n"
            f"Selected asset in the UI: {selected}\n\n"
            f"Tools:\n{manifest_text}\n\n"
            "Observations so far:\n"
            + ("\n\n".join(observations) or "(none yet: this is your first step)")
            + f"\n\nStep {step} of {AGENT_MAX_STEPS}. Your JSON:"
        )
        msgs = [{"role": "system", "content": AGENT_LOOP_PROMPT},
                {"role": "user", "content": user}]
        # Reasoning models on the free tiers spend part of max_tokens thinking,
        # so the budget is generous and a cut-off reply gets one retry.
        resp = llm_chat(msgs, temperature=0.0, max_tokens=1500, provider=provider, model=model)
        decision = _parse_json_obj(resp.get("text", "")) if resp.get("ok") else None
        if not decision and resp.get("ok"):
            msgs += [{"role": "assistant", "content": resp.get("text", "")},
                     {"role": "user", "content": "That reply was cut off or not valid JSON. "
                      "Send the JSON object again, thought under 25 words."}]
            resp = llm_chat(msgs, temperature=0.0, max_tokens=1500, provider=provider, model=model)
            decision = _parse_json_obj(resp.get("text", "")) if resp.get("ok") else None
        if not decision:
            trace.append({"step": step, "thought": "agent reply was not valid JSON; stopping here", "calls": []})
            break

        thought = str(decision.get("thought", ""))[:300]
        calls = decision.get("tools") or []
        if decision.get("done") or not calls:
            trace.append({"step": step, "thought": thought, "calls": [], "done": True})
            if on_step:
                on_step(trace[-1])
            break

        ran = []
        for item in calls[:3]:
            if len(results) >= AGENT_MAX_CALLS:
                break
            name = item.get("name", "") if isinstance(item, dict) else ""
            if name not in valid:
                observations.append(f"[step {step}] '{name}' is not a tool")
                continue
            args = dict(item.get("args") or {})
            args["ticker"] = normalize_ticker(str(args.get("ticker", ""))) or primary
            if args.get("ticker_b"):
                args["ticker_b"] = normalize_ticker(str(args["ticker_b"]))
            if name == "run_analysis":
                args["question"] = args.get("question") or question
                args["tickers"] = list(dict.fromkeys([args["ticker"], *tickers]))
            key = name + json.dumps(args, sort_keys=True, default=str)
            shown = {k: v for k, v in args.items() if k != "tickers"}
            if key in seen:
                observations.append(f"[step {step}] {name}{json.dumps(shown, default=str)}: already called, see above")
                continue
            seen.add(key)
            try:
                result = run_tool(name, args, provider, model)
            except Exception as e:
                result = {"error": str(e)[:150]}
            results.append({"tool": name, "args": args, "result": result})
            observations.append(f"[step {step}] {name}{json.dumps(shown, default=str)} ->\n"
                                f"{json.dumps(result, default=str)[:2500]}")
            label = args["ticker"] + (f", {args['ticker_b']}" if args.get("ticker_b") else "")
            if args.get("rule"):
                label += f", {args['rule']}"
            ran.append(f"{name}({label})")

        trace.append({"step": step, "thought": thought, "calls": ran})
        if on_step:
            on_step(trace[-1])
        if len(results) >= AGENT_MAX_CALLS:
            break

    # The LLM providers can all be down or rate-limited at once; answer from a
    # basic snapshot rather than nothing.
    if not results:
        for name in ("get_quote", "indicators", "fundamental_facts"):
            args = {"ticker": primary}
            try:
                result = run_tool(name, args, provider, model)
            except Exception as e:
                result = {"error": str(e)[:150]}
            results.append({"tool": name, "args": args, "result": result})
        trace.append({"step": len(trace) + 1,
                      "thought": "no usable plan from the agent; fetched a default snapshot",
                      "calls": [f"{r['tool']}({primary})" for r in results]})
        if on_step:
            on_step(trace[-1])
    return results, trace


def prepare_answer(question: str, provider: str = "", model: str = "",
                   history: list[dict] | None = None, on_step=None) -> dict:
    """Agentic flow: resolve tickers named in the question → the agent loop
    gathers evidence step by step → build the synthesis context. Returns the
    messages to stream plus the trace for the UI.
    """
    tickers = resolve_tickers(question)
    selected = st.session_state.get("_sidebar_ticker", "AAPL") or "AAPL"

    results, trace = agent_loop(question, tickers, selected, provider, model, on_step)
    if not tickers:
        tickers = list(dict.fromkeys(r["args"]["ticker"] for r in results)) or [selected]

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
        "metric is missing, give the closest available figure and name the gap. "
        "If an asked symbol has no data at all, say so plainly and do not "
        "substitute analysis of a different asset.\n"
        "5. When multiple tickers were asked about, answer for each one as a clear "
        "per-ticker breakdown, and compare them when the question asks.\n"
        "6. Provide information and analysis. Do not invent a position size or an order.\n"
        "7. When verdict is in TOOL DATA, that payload's direction is the answer: "
        "long, flat, or no-edge. Lead with it and quote summary, the 70% and 95% "
        "ranges, and the band hit rates. Do not replace the direction with your own view. "
        "ALWAYS append the standard warning below verbatim.\n"
        "8. Report numbers exactly as produced by tool output or run_analysis "
        "execution results. Every figure you state must be in TOOL DATA: no "
        "support/resistance levels, index levels, volatility of assets that were "
        "not fetched, or earnings dates from memory. Market context without data "
        "behind it is fine as words, never as numbers. Simple arithmetic on tool "
        "figures (a % gap between two of them) is allowed.\n"
        "9. End with a one-line 'Bottom line' summary.\n"
        "10. If the user asked for suggestions or recommendations of OTHER stocks, "
        "recommend ONLY from the candidate data present in TOOL DATA — never "
        "recommend a ticker whose data is absent, and cite each candidate's "
        "price and valuation from the tool output.\n"
        "11. If the user asked which way a symbol is leaning and verdict is missing, "
        "say the tested direction was not computed. Do not fill the gap from memory.\n\n"
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
        "tools_used": [r["tool"] for r in results],
        "results": results,
        "trace": trace,
        "messages": messages,
    }


def answer_question(question: str, provider: str = "", model: str = "",
                    history: list[dict] | None = None) -> dict:
    """Non-streaming path: run tools then get the full answer."""
    prep = prepare_answer(question, provider, model, history)
    resp = llm_chat(prep["messages"], provider=provider, model=model)
    return {
        "answer": resp.get("text", f"LLM error: {resp.get('error', '?')}"),
        "ticker": prep["ticker"],
        "tools_used": prep["tools_used"],
        "results": prep["results"],
        "trace": prep["trace"],
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
        provider_choice = st.selectbox(
            "Provider:",
            ["Auto (failover)"] + LLM_PROVIDER_ORDER,
            help="Auto walks Groq, Gemini (free tier), NVIDIA, Kilo, then opencode, "
                 "skipping providers without a key and models that recently failed. "
                 "A pinned provider is tried first. Models load from each provider's /models endpoint.",
        )
        with st.expander("Provider status", expanded=False):
            for row in provider_status():
                live = f"{row['live_models']} live" if row["live_models"] else "unavailable"
                first = f" · first: `{row['first']}`" if row["first"] else ""
                st.markdown(f"**{row['provider']}** · {row['key']} · {live}{first}")
            st.caption("Keys are read from Streamlit secrets, environment variables, "
                       "or ~/.hermes/.env. Groq and NVIDIA keys are free to create.")
        if provider_choice == "Auto (failover)":
            provider, model_choice = "", ""
            st.caption("Models are chosen per call from the live failover chain.")
        else:
            provider = provider_choice
            model_opts = available_models(provider) or LLM_PROVIDERS[provider]["models"]
            model_choice = st.selectbox(
                "Model:", model_opts,
                help="Live list from /models, curated free models first. "
                     "Falls back to the built-in list if the endpoint is unreachable.",
            )

        auto_refresh = st.checkbox("🔄 Auto-refresh price chart (60s)", value=False)

        st.markdown("---")
        run = st.button("▶️ Run / Refresh All Tabs", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            '<small style="color:#888;">Facts, sources, and quantified uncertainty — '
            'every figure computed from live data.<br>Groq · Gemini · NVIDIA · Kilo · opencode, models loaded live</small>',
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
# Bars kept for each range button. Daily charts request enough history for the
# button; intraday charts use the sidebar interval's own window.
CHART_RANGES = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "5Y": 1260}
TAPE = ["NVDA", "BTC-USD", "RELIANCE.NS", "GC=F", "INR=X", "^NSEI"]
CHART_OVERLAYS = ("sma20", "sma50", "sma200", "bb")
CHART_PANELS = ("volume", "rsi", "macd")


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def chart_history(ticker: str, interval: str, range_key: str, need_long: bool) -> pd.DataFrame:
    """Bars for the chart. Daily + rule marks load five years so the 70/30
    split is the same window the backtest uses. Intraday keeps the interval window."""
    if interval != "1d":
        return get_history(ticker, interval, interval_period(interval))
    if need_long or range_key == "5Y":
        period = "5y"
    elif range_key in ("1Y", "6M"):
        period = "2y"
    else:
        period = "1y"
    df = get_history(ticker, "1d", period)
    if need_long:
        return df
    return df.tail(CHART_RANGES.get(range_key, 252))


def rule_trade_marks(df: pd.DataFrame, rule: str, params: dict | None = None) -> list[dict]:
    """Buy and sell dates using the same next-bar fill as the backtest."""
    if df is None or df.empty or rule not in BACKTEST_RULES:
        return []
    try:
        pos = _rule_positions(df, rule, params or {})
    except ValueError:
        return []
    held = pos.shift(1).fillna(0)
    flips = held.diff()
    flips.iloc[0] = held.iloc[0]
    marks = []
    for ts, v in flips.items():
        side = "buy" if v == 1 else "sell" if v == -1 else ""
        if side:
            marks.append({"date": ts, "side": side, "price": float(df.loc[ts, "Close"])})
    return marks


def _marks_on_view(marks: list[dict], view: pd.DataFrame) -> list[dict]:
    if view.empty or not marks:
        return []
    start, end = view.index[0], view.index[-1]
    kept = []
    for m in marks:
        ts = pd.Timestamp(m["date"]).tz_localize(None) if pd.Timestamp(m["date"]).tzinfo else pd.Timestamp(m["date"])
        if start <= ts <= end:
            kept.append({**m, "date": ts})
    return kept


def build_research_figure(df: pd.DataFrame, *, compare: pd.DataFrame | None = None,
                          compare_name: str = "",
                          overlays: tuple | list = CHART_OVERLAYS,
                          panels: tuple | list = CHART_PANELS,
                          marks: list[dict] | None = None,
                          split_at=None, cone: dict | None = None,
                          title: str = "") -> go.Figure | None:
    """Candles plus the panels and overlays the chart tab (or chart_spec) asked for.

    A comparison series is rebased to 100 on a second axis so a coin and a
    stock can share the time axis. Python draws every mark. The model only
    chooses which of these layers to turn on.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    overlays = tuple(overlays or ())
    panels = tuple(p for p in (panels or ()) if p in CHART_PANELS)
    if panels:
        rest = 0.42 / len(panels)
        heights = [0.58] + [rest] * len(panels)
    else:
        heights = [1]
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}] for _ in panels]
    fig = make_subplots(
        rows=1 + len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=heights, specs=specs,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#1f8a4c", decreasing_line_color="#c62828",
        increasing_fillcolor="#1f8a4c", decreasing_fillcolor="#c62828",
    ), row=1, col=1, secondary_y=False)

    close = df["Close"]
    sma_colors = {"sma20": "#f59e0b", "sma50": "#7B2D8E", "sma200": "#1565c0"}
    for n, key in ((20, "sma20"), (50, "sma50"), (200, "sma200")):
        if key in overlays and len(df) >= n:
            fig.add_trace(go.Scatter(
                x=df.index, y=close.rolling(n).mean(), name=key.upper(),
                line=dict(color=sma_colors[key], width=1.2),
            ), row=1, col=1, secondary_y=False)
    if "bb" in overlays and len(df) >= 20:
        mid, sd = close.rolling(20).mean(), close.rolling(20).std()
        fig.add_trace(go.Scatter(
            x=df.index, y=mid + 2 * sd, name="BB upper",
            line=dict(color="#90a4ae", width=1, dash="dot"),
        ), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df.index, y=mid - 2 * sd, name="BB lower",
            line=dict(color="#90a4ae", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(144,164,174,0.12)",
        ), row=1, col=1, secondary_y=False)
    if compare is not None and not compare.empty and "Close" in compare.columns:
        both = pd.concat({"a": close, "b": compare["Close"]}, axis=1).dropna()
        if len(both) > 2:
            fig.add_trace(go.Scatter(
                x=both.index, y=both["b"] / both["b"].iloc[0] * 100,
                name=f"{compare_name or 'compare'} rebased",
                line=dict(color="#C9A84C", width=1.4),
            ), row=1, col=1, secondary_y=True)

    marker_style = {
        "buy": ("triangle-up", "#1f8a4c"),
        "sell": ("triangle-down", "#c62828"),
        "news": ("circle", "#546e7a"),
    }
    for side, (symbol, color) in marker_style.items():
        pts = [m for m in (marks or []) if m.get("side") == side]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(m["date"]) for m in pts],
            y=[m.get("price") for m in pts],
            mode="markers", name=side,
            text=[m.get("title", side) for m in pts],
            marker=dict(symbol=symbol, size=9, color=color),
        ), row=1, col=1, secondary_y=False)

    if split_at is not None and len(df):
        ts = pd.Timestamp(split_at)
        if df.index[0] <= ts <= df.index[-1]:
            fig.add_vline(x=ts, line_dash="dash", line_color="#7B2D8E", row=1, col=1)

    if cone and cone.get("range_95") and len(df):
        last = df.index[-1]
        horizon = max(int(cone.get("horizon") or cone.get("horizon_bars") or 7), 1)
        step = df.index.to_series().diff().median()
        if pd.notna(step) and step < pd.Timedelta(days=1):
            future = pd.date_range(last, periods=horizon + 1, freq=step)[1:]
        else:
            future = pd.bdate_range(last, periods=horizon + 1)[1:]
        px = float(close.iloc[-1])
        if len(future):
            end = future[-1]
            fills = {"range_95": "rgba(123,45,142,0.10)", "range_70": "rgba(201,168,76,0.20)"}
            for key in ("range_95", "range_70"):
                band = cone.get(key)
                if not band or len(band) != 2:
                    continue
                fig.add_trace(go.Scatter(
                    x=[last, end, end, last],
                    y=[px, band[0], band[1], px],
                    fill="toself", name=key, mode="lines",
                    line=dict(width=0), fillcolor=fills[key],
                ), row=1, col=1, secondary_y=False)

    up = close >= df["Open"]
    row = 2
    if "volume" in panels and "Volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=["#1f8a4c" if flag else "#c62828" for flag in up],
        ), row=row, col=1)
        row += 1
    if "rsi" in panels and len(df) > 15:
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi_series(close), name="RSI",
            line=dict(color="#7B2D8E", width=1.2),
        ), row=row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#c62828", row=row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#1f8a4c", row=row, col=1)
        row += 1
    if "macd" in panels and len(df) > 26:
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        fig.add_trace(go.Bar(
            x=df.index, y=hist, name="MACD hist",
            marker_color=["#1f8a4c" if v >= 0 else "#c62828" for v in hist.fillna(0)],
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd, name="MACD", line=dict(color="#1565c0", width=1.1),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=signal, name="MACD signal", line=dict(color="#f59e0b", width=1.1),
        ), row=row, col=1)

    fig.update_layout(
        title=title,
        height=300 + 150 * max(len(panels), 0),
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    if compare is not None and not getattr(compare, "empty", True):
        fig.update_yaxes(title_text="Rebased 100", row=1, col=1, secondary_y=True)
    return fig


def set_chart_spec(args: dict) -> dict:
    """Record which ticker and layers the chart should show. Drawing happens
    in the price tab, from this record, not inside the model."""
    rng = str(args.get("range") or "1Y")
    if rng not in CHART_RANGES:
        rng = "1Y"
    overlays = [o for o in _as_str_list(args.get("overlays")) if o in CHART_OVERLAYS]
    panels = [p for p in _as_str_list(args.get("panels")) if p in CHART_PANELS]
    rule = str(args.get("rule") or "")
    if rule not in BACKTEST_RULES:
        rule = ""
    marks = "backtest" if str(args.get("marks") or "") == "backtest" and rule else "none"
    compare = normalize_ticker(str(args.get("ticker_b") or args.get("compare") or ""))
    spec = {
        "ticker": normalize_ticker(str(args.get("ticker") or "")),
        "range": rng,
        "compare": compare,
        "overlays": overlays or list(CHART_OVERLAYS),
        "panels": panels or list(CHART_PANELS),
        "marks": marks,
        "rule": rule,
    }
    try:
        st.session_state["chart_spec"] = spec
        if spec["ticker"]:
            st.session_state["chart_focus"] = spec["ticker"]
    except Exception:
        pass
    return spec


def _apply_chart_spec(spec: dict) -> None:
    """Push a new agent spec into the chart widgets once. Later clicks stay."""
    sig = json.dumps(spec, sort_keys=True, default=str)
    if not spec or st.session_state.get("_chart_spec_sig") == sig:
        return
    st.session_state["_chart_spec_sig"] = sig
    st.session_state["chart_range"] = spec.get("range") or "1Y"
    st.session_state["chart_compare"] = spec.get("compare") or ""
    ov = set(spec.get("overlays") or CHART_OVERLAYS)
    pn = set(spec.get("panels") or CHART_PANELS)
    st.session_state["ov_sma"] = bool(ov & {"sma20", "sma50", "sma200"})
    st.session_state["ov_bb"] = "bb" in ov
    st.session_state["ov_rsi"] = "rsi" in pn
    st.session_state["ov_macd"] = "macd" in pn
    st.session_state["ov_volume"] = "volume" in pn
    st.session_state["ov_marks"] = spec.get("marks") == "backtest"
    if spec.get("rule") in BACKTEST_RULES:
        st.session_state["ov_rule"] = spec["rule"]
    if spec.get("ticker"):
        st.session_state["chart_focus"] = spec["ticker"]


def price_chart_tab(ticker: str, interval: str, auto_refresh: bool):
    spec = st.session_state.get("chart_spec") or {}
    _apply_chart_spec(spec)
    focus = st.session_state.get("chart_focus") or ticker

    st.markdown(f"### 📈 {focus}")
    bench = benchmark_for(focus)
    st.caption(
        f"{market_of(focus)} · benchmark {bench or 'none'} · interval {interval} · "
        f"as of {fresh_stamp()} · any Yahoo Finance symbol"
    )

    tape_cols = st.columns(len(TAPE))
    for i, sym in enumerate(TAPE):
        q = get_quote(sym)
        price = q.get("price")
        chg = q.get("day_change_pct")
        tape_cols[i].metric(
            sym,
            f"{price:,.2f}" if price else "—",
            delta=f"{chg:.2f}%" if chg is not None else None,
        )
        if tape_cols[i].button(sym, key=f"tape_{sym}", use_container_width=True):
            st.session_state["chart_focus"] = sym
            st.rerun()

    if focus != ticker and st.button(f"Use sidebar ticker {ticker}", key="clear_focus"):
        st.session_state.pop("chart_focus", None)
        st.rerun()

    defaults = {
        "chart_range": "1Y",
        "chart_compare": "",
        "ov_sma": True,
        "ov_bb": True,
        "ov_rsi": True,
        "ov_macd": True,
        "ov_volume": True,
        "ov_marks": False,
        "ov_rule": "rsi_oversold",
        "ov_direction": True,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    range_key = st.radio("Range", list(CHART_RANGES), horizontal=True, key="chart_range")
    controls = st.columns([2, 1, 1, 1, 1, 1, 1])
    compare_sym = controls[0].text_input("Compare (any Yahoo symbol)", key="chart_compare")
    show_bench = controls[1].checkbox("Benchmark", value=True)
    show_sma = controls[2].checkbox("SMA", key="ov_sma")
    show_bb = controls[3].checkbox("Bollinger", key="ov_bb")
    show_rsi = controls[4].checkbox("RSI", key="ov_rsi")
    show_macd = controls[5].checkbox("MACD", key="ov_macd")
    show_volume = controls[6].checkbox("Volume", key="ov_volume")
    extra = st.columns([1, 1.6, 1, 1, 1])
    show_direction = extra[0].checkbox("Direction", key="ov_direction")
    rule = extra[1].selectbox("Rule", list(BACKTEST_RULES), key="ov_rule")
    show_marks = extra[2].checkbox("Rule marks", key="ov_marks")
    show_cone = extra[3].checkbox("GARCH cone", value=False)
    show_news = extra[4].checkbox("News dates", value=False)

    frag = st.fragment(run_every=60 if auto_refresh else None)

    @frag
    def _chart_fragment(focus: str, interval: str, range_key: str, compare_sym: str,
                        show_bench: bool, show_sma: bool, show_bb: bool, show_rsi: bool,
                        show_macd: bool, show_volume: bool, show_marks: bool, rule: str,
                        show_cone: bool, show_news: bool, show_direction: bool):
        need_long = bool(show_marks and interval == "1d")
        full = chart_history(focus, interval, range_key, need_long)
        if full.empty:
            st.warning(f"No data found for {focus} at {interval}.")
            return
        view = full.tail(CHART_RANGES.get(range_key, len(full))) if need_long else full
        q = get_quote(focus)
        price, chg = q.get("price"), q.get("day_change_pct")
        head = st.columns(3)
        if price:
            head[0].metric("Last", f"{price:,.4f}" if price < 5 else f"{price:,.2f}",
                           delta=f"{chg:.2f}%" if chg is not None else None)
        head[1].metric("Bars", f"{len(view):,}")
        head[2].metric("Class", market_of(focus))

        overlays = []
        if show_sma:
            overlays += ["sma20", "sma50", "sma200"]
        if show_bb:
            overlays.append("bb")
        panels = []
        if show_volume:
            panels.append("volume")
        if show_rsi:
            panels.append("rsi")
        if show_macd:
            panels.append("macd")

        marks, split_at = [], None
        short = interval in INTRADAY_INTERVALS
        if rule and show_marks and (need_long or short):
            marks = _marks_on_view(rule_trade_marks(full, rule), view)
            if len(full) > 10:
                split_at = full.index[int(len(full) * 0.7)]
        elif show_marks:
            st.caption("Rule marks run on 1d, 1m, 5m, and 15m.")

        other = (compare_sym or "").strip()
        if not other and show_bench:
            other = benchmark_for(focus) or ""
        compare_df = None
        if other and normalize_ticker(other) != focus:
            other = normalize_ticker(other)
            compare_df = chart_history(other, interval, range_key, False)

        card = None
        if show_direction and rule and (interval == "1d" or short):
            card = verdict(focus, rule, years=5, horizon_days=7, interval=interval)
        elif show_direction:
            st.caption("Direction runs on 1d, 1m, 5m, and 15m.")

        cone = None
        if (interval == "1d" or short) and (show_cone or card):
            band = (card or {}).get("forecast") if card and not card.get("error") else None
            if not band or not band.get("range_95"):
                if short:
                    band = _intraday_cone(full["Close"], INTRADAY_HORIZON_BARS[interval])
                elif show_cone:
                    band = forecast_range(focus)
                else:
                    band = None
            if band and band.get("range_95"):
                cone = band
        elif show_cone:
            st.caption("The volatility cone runs on 1d, 1m, 5m, and 15m.")

        if show_news:
            for story in get_ticker_news(focus, limit=8):
                raw = (story.get("date") or "")[:10]
                if not raw:
                    continue
                day = pd.Timestamp(raw)
                hits = view.index[view.index.normalize() == day]
                if len(hits):
                    ts = hits[-1]
                    marks.append({
                        "date": ts, "side": "news",
                        "price": float(view.loc[ts, "Close"]),
                        "title": story.get("title") or "news",
                    })

        fig = build_research_figure(
            view, compare=compare_df, compare_name=other,
            overlays=overlays, panels=panels, marks=marks,
            split_at=split_at, cone=cone, title=focus,
        )
        if fig is None:
            st.warning(f"No chart for {focus}.")
            return
        st.plotly_chart(fig, use_container_width=True)
        if card and card.get("error"):
            st.caption(card["error"])
        elif card:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Direction", card["direction"])
            c2.metric("Rule now", card["rule_position"])
            unit = "bar" if card.get("interval") in INTRADAY_INTERVALS else "day"
            c3.metric("OOS excess", f"{card['oos']['excess_mean_bps']:.1f} bps/{unit}")
            hit = (card.get("forecast") or {}).get("band70_hit_rate")
            c4.metric("70% band hit", f"{hit:.0%}" if hit is not None else "—")
            st.caption(card["summary"])
        bits = [f"range {range_key}"]
        if other:
            bits.append(f"compared with {other}, rebased to 100")
        if split_at is not None:
            bits.append(f"dashed line is the 70/30 split on {str(pd.Timestamp(split_at).date())}")
        if cone and interval in INTRADAY_INTERVALS:
            bits.append(f"shaded cone is the next {INTRADAY_HORIZON_BARS[interval]} {interval} bars")
        elif cone:
            bits.append("shaded cone is the live GARCH 70% and 95% band")
        st.caption(" · ".join(bits))

    _chart_fragment(
        focus, interval, range_key, compare_sym, show_bench, show_sma, show_bb,
        show_rsi, show_macd, show_volume, show_marks, rule, show_cone, show_news,
        show_direction,
    )


def trace_line(s: dict) -> str:
    """One agent step as a markdown line: what it concluded, what it called."""
    action = ", ".join(f"`{c}`" for c in s.get("calls", [])) or "done, writing the answer"
    return f"**Step {s['step']}** · {s.get('thought') or '…'}  \n→ {action}"


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
            if msg.get("trace"):
                with st.expander(f"Agent: {len(msg['trace'])} steps", expanded=False):
                    for s in msg["trace"]:
                        st.markdown(trace_line(s))
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
            with st.status("Agent working...", expanded=True) as status:
                prep = prepare_answer(q_text, provider, model,
                                      history=st.session_state.chat_turns,
                                      on_step=lambda s: st.markdown(trace_line(s)))
                status.update(
                    label=f"Agent: {len(prep['trace'])} steps · {len(prep['tools_used'])} tool calls",
                    state="complete", expanded=False)
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
            "trace": prep["trace"], "model": mdl, "provider": prov,
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
            | **LLM** | opencode-free · Groq · NVIDIA Build · kilo-free. Models load from each provider's `/models` endpoint; Auto fails over in that order |
            | **Agent** | observe → decide → act loop, up to 6 steps, then synthesis (citations instructed) |
            | **Sentiment** | LLM JSON scoring with deterministic lexicon fallback |
            | **Risk** | historical vol, VaR, max drawdown, Sharpe (descriptive) |
            | **Technical** | RSI(14), MACD(12,26,9), Bollinger(20,2) — facts only |
            | **UI** | Streamlit · wide layout · sidebar is the single source of truth |
            """
        )
        st.markdown("#### 🧠 Agent Loop")
        st.markdown(
            """
            1. **Decide** — the LLM reads the tool manifest plus every result
               so far and picks the next 1–3 calls, or says it has enough.
            2. **Act** — tools run against live data; `run_analysis` writes
               Python, the subprocess computes real numbers. Results go back
               to step 1, so a finding (RSI under 30) can trigger the next
               check (backtest the RSI-dip rule). Capped at 6 steps / 14
               calls; repeated calls are refused. Python never picks a tool.
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
            | **opencode-free** | `opencode.ai/zen/v1` | live `/models`, curated free ids first |
            | **groq** | `api.groq.com/openai/v1` | live `/models` when `GROQ_API_KEY` is set |
            | **nvidia-build** | `integrate.api.nvidia.com/v1` | live `/models` when `NVIDIA_API_KEY` is set |
            | **kilo-free** | `api.kilo.ai/api/openrouter` | live `/models`, Kilo referer headers |

            **Failover:** Auto, or the pinned provider first, then opencode-free →
            groq → nvidia-build → kilo-free. A model chosen from the live list
            is tried even when it is not in the built-in fallback list. The
            picker caches `/models` for 10 minutes and uses the built-in list
            when the endpoint is down.
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
            - **Leading vs lagging** — leading (RSI, stochastic, rate of change,
              OBV divergence) can turn before price; lagging (50/200-day MA,
              MACD) confirms a trend once it is under way.
            - **Relative strength** — stock return minus NIFTY 50 (India) or
              S&P 500 (US) over 1/3/6 months.
            - **Backtest** — long-only daily rule, costs per side (Indian STT +
              stamp duty for NSE/BSE), judged on the first 70% of history and
              again on the last 30% it never saw, against buy-and-hold.

            *All indicators are computed from live data and shown as facts.
            Buy/sell questions get evidence, never a call.*
            """
        )

    st.markdown("---")
    st.markdown("#### 💻 Run & Deploy")
    st.markdown(
        """
        ```bash
        pip install -r requirements.txt   # streamlit, yfinance, pandas, numpy, plotly, requests, sentence-transformers
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
          flowing across opencode-free, Groq, NVIDIA Build, and Kilo.
        - Uncertainty is quantified as volatility-based ranges, with a standard
          risk disclaimer attached to every forecast-style answer.
        - Backtests are optimistic in known ways: fills at the signal day's
          close, only stocks still listed can be tested (survivorship), costs are
          approximate. A rule that fails here would fail live; one that passes
          has only cleared the first bar.
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
    warm_model_registry()          # cached 10 min; parallel /models fetch
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
