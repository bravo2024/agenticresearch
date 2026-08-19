"""News falls back to Yahoo search and Google News when Yahoo's per-ticker
feed is empty. Every network call is stubbed, so these run offline.

Dates are generated relative to *now* because the pipeline enforces a
`max_age_days` floor — fixed calendar dates would rot the suite within a week.
"""
import time
from email.utils import formatdate

import pytest

import app


def rfc2822(days_ago: float) -> str:
    return formatdate(time.time() - days_ago * 86400, usegmt=True)


def rss(items: list[tuple[str, str, str, float]]) -> bytes:
    body = "".join(
        f"<item><title>{t} - {s}</title><source>{s}</source>"
        f"<link>{l}</link><pubDate>{rfc2822(a)}</pubDate></item>"
        for t, s, l, a in items
    )
    return f"<rss><channel>{body}</channel></rss>".encode()


class FakeTicker:
    def __init__(self, symbol):
        self.news = []


class FakeSearch:
    """Yahoo search: nothing for NSE symbols, news by company name."""
    by_query = {
        "TCS.NS": [],
        "Tata Consultancy Services": [
            {"title": "TCS wins a large deal", "publisher": "Mint", "link": "u1",
             "providerPublishTime": time.time() - 3600, "relatedTickers": ["TCS.NS"]},
            {"title": "Sensex closes flat", "publisher": "PTI", "link": "u2",
             "providerPublishTime": time.time() - 7200, "relatedTickers": []},
        ],
    }

    def __init__(self, query, max_results=8, news_count=8, **_):
        self.quotes = [{"symbol": "TCS.NS", "quoteType": "EQUITY",
                        "longname": "Tata Consultancy Services Limited"}]
        self.news = self.by_query.get(query, [])

    @classmethod
    def with_news(cls, mapping):
        class S(cls):
            by_query = mapping
        return S


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture
def offline(monkeypatch):
    calls = []

    def fake_get(url, params=None, **_):
        calls.append(params["q"])
        return FakeResponse(rss([
            ("TCS shares rise after results", "Reuters", "g1", 1),
            ("TCS wins a large deal", "Mint", "g2", 2),
        ]))

    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(app.yf, "Search", FakeSearch)
    monkeypatch.setattr(app.requests, "get", fake_get)
    app._news.clear()
    app._asset_name.clear()
    return calls


def test_asset_name_drops_legal_suffix(offline):
    assert app.asset_name("TCS.NS") == ("Tata Consultancy Services", "EQUITY")


def test_empty_feed_falls_back_to_search_and_google(offline):
    news = app.get_ticker_news("TCS.NS")
    titles = [n["title"] for n in news]
    # "TCS wins a large deal" exists in both feeds; the directly fetchable
    # Yahoo URL wins the dedupe, and it is also the fresher of the two.
    assert titles == ["TCS wins a large deal", "TCS shares rise after results"]
    assert all(n["_filtered"] for n in news)
    assert news[0]["url"] == "u1"
    assert offline == ["Tata Consultancy Services stock when:7d"]


def test_sentiment_no_longer_reports_missing_news(offline, monkeypatch):
    monkeypatch.setattr(app, "llm_chat", lambda *a, **k: {"ok": False, "error": "offline"})
    assert app.sentiment_score("TCS.NS") != {"error": "no recent news"}


def test_stale_google_items_are_dropped(monkeypatch):
    def fake_get(url, params=None, **_):
        return FakeResponse(rss([
            ("Fresh Nifty story", "Mint", "g1", 1),
            ("Ancient Nifty story", "Mint", "g2", 60),
        ]))

    monkeypatch.setattr(app.requests, "get", fake_get)
    got = app._google_news("Nifty 50", india=True, limit=15)
    assert [n["title"] for n in got] == ["Fresh Nifty story"]


def test_stale_search_items_are_dropped(monkeypatch):
    now = time.time()
    search_cls = FakeSearch.with_news({
        "ZZZ": [
            {"title": "ZZZ does a thing", "publisher": "Mint", "link": "u1",
             "providerPublishTime": now - 3600, "relatedTickers": ["ZZZ"]},
            {"title": "ZZZ did a thing long ago", "publisher": "Mint", "link": "u2",
             "providerPublishTime": now - 60 * 86400, "relatedTickers": ["ZZZ"]},
        ],
    })
    monkeypatch.setattr(app.yf, "Search", search_cls)
    got = app._search_news("ZZZ", "ZZZ corp", 15)
    assert [n["title"] for n in got] == ["ZZZ does a thing"]


@pytest.mark.parametrize("title", [
    "Reliance Industries Limited (RELIANCE.NS) stock price, news, quote and history",
    "Infosys Limited (INFY) Stock Price, News, Quote & History",
    "HDFC Bank Share Price - Live NSE: HDFCBANK Stock Price & Chart",
    "Reliance Industries Share Price Prediction for Tomorrow, 13 Oct",
    "INFY Share Price Today - Infosys on NSE/BSE",
    "Current price of Bitcoin for October 9, 2026",
])
def test_junk_title_blocks_quote_pages_and_prediction_spam(title):
    assert app.junk_title(title)


@pytest.mark.parametrize("title", [
    "RIL target price ahead of Jio IPO: 47% upside - Why Religare recommends",
    "TCS share price rebounds 14% from September low",
    "HDFC Bank shares surge 1.60% as Q2 results weekend nears",
    "Stock market today: S&P 500, Nasdaq slip as chip sell-off continues",
    "Nvidia stock sinks amid concerns over AI financing",
])
def test_junk_title_keeps_real_headlines(title):
    assert not app.junk_title(title)


def test_relevance_filter_runs_before_truncation(monkeypatch):
    """20 newer irrelevant items must not crowd out 2 older relevant ones."""
    now = time.time()
    irrelevant = [
        {"title": f"Sector noise story {i}", "publisher": "Wire", "link": f"w{i}",
         "providerPublishTime": now - 60 * i, "relatedTickers": ["TCS.NS"]}
        for i in range(20)
    ]
    relevant = [
        {"title": "TCS wins a large deal", "publisher": "Mint", "link": "u1",
         "providerPublishTime": now - 20 * 3600, "relatedTickers": ["TCS.NS"]},
        {"title": "TCS buyback announced", "publisher": "Mint", "link": "u2",
         "providerPublishTime": now - 21 * 3600, "relatedTickers": ["TCS.NS"]},
    ]
    search_cls = FakeSearch.with_news({"TCS.NS": [], "Tata Consultancy Services": irrelevant + relevant})
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(app.yf, "Search", search_cls)
    monkeypatch.setattr(app.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no rss")))
    app._news.clear()
    app._asset_name.clear()
    got = app.get_ticker_news("TCS.NS", limit=15)
    assert [n["title"] for n in got] == ["TCS wins a large deal", "TCS buyback announced"]


def test_dedupe_prefers_direct_url(monkeypatch):
    def fake_get(url, params=None, **_):
        return FakeResponse(rss([("TCS wins a large deal", "Mint",
                                  "https://news.google.com/rss/articles/CBMiX", 1)]))

    search_cls = FakeSearch.with_news({
        "TCS.NS": [],
        "Tata Consultancy Services": [
            {"title": "TCS wins a large deal", "publisher": "Mint",
             "link": "https://finance.yahoo.com/markets/article/tcs-deal.html",
             "providerPublishTime": time.time() - 7200, "relatedTickers": ["TCS.NS"]},
        ],
    })
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(app.yf, "Search", search_cls)
    monkeypatch.setattr(app.requests, "get", fake_get)
    app._news.clear()
    app._asset_name.clear()
    got = app.get_ticker_news("TCS.NS")
    assert len(got) == 1
    assert got[0]["url"] == "https://finance.yahoo.com/markets/article/tcs-deal.html"


def test_article_text_rejects_google_redirect_without_network():
    art = app.article_text("https://news.google.com/rss/articles/CBMiX")
    assert art.get("error") and not art.get("text")


def test_no_market_fallback_for_unknown_tickers(monkeypatch):
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(app.yf, "Search", FakeSearch)
    monkeypatch.setattr(app, "asset_name", lambda t: ("", ""))
    monkeypatch.setattr(app.requests, "get", lambda *a, **k: FakeResponse(b"<rss><channel></channel></rss>"))
    app._news.clear()
    assert app.get_ticker_news("ZZZZ.NS") == []
    assert app.sentiment_score("ZZZZ.NS") == {"error": "no recent news"}
