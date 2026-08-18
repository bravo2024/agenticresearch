"""Company names in a question resolve through Yahoo search, and model
answers are cleaned before they are shown. Search is stubbed, so these run
offline."""
import pytest

import app


@pytest.fixture
def fake_search(monkeypatch):
    listings = {
        "maruti suzuki": "MARUTI.NS",
        "hindustan aeronautics": "HAL.NS",
        "wipro": "WIPRO.NS",
    }
    monkeypatch.setattr(app, "_search_name", lambda name: listings.get(name.lower(), ""))


def test_unlisted_alias_resolves_by_search(fake_search):
    q = "Has Maruti Suzuki beaten the Indian market over the last 6 months?"
    assert app.resolve_names(q) == ["MARUTI.NS"]


def test_question_words_are_not_names(fake_search):
    assert app.resolve_names("Why is it up today?") == []


def test_unknown_company_resolves_to_nothing(fake_search):
    assert app.resolve_names("What's the RSI of Qwertyuiop Industries?") == []


def test_multiword_name_beats_single_words(fake_search):
    assert app.resolve_names("Is Hindustan Aeronautics overvalued?") == ["HAL.NS"]


def test_clean_answer_strips_citation_markers():
    raw = "Direction **no‑edge**【verdict†L1-L4】. Price — flat."
    assert app.clean_answer(raw) == "Direction **no-edge**. Price, flat."


def test_clean_stream_holds_back_open_marker():
    chunks = ["RSI is 28 【tech", "nical†L2】 so", " oversold."]
    assert "".join(app.clean_stream(iter(chunks))) == "RSI is 28 so oversold."
