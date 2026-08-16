"""Live model discovery, ranking, failure memory and the failover order.
No network: /models and chat calls are faked."""
import time

import pytest

import app


@pytest.fixture(autouse=True)
def clean_registry():
    app._MODELS_CACHE.clear()
    app._MODEL_HOLD.clear()
    yield
    app._MODELS_CACHE.clear()
    app._MODEL_HOLD.clear()


def test_call_order_keyed_first_and_pins_discovered_model():
    order = app._call_order("", "")
    assert [name for name, _, _ in order] == [
        "groq", "gemini-free", "nvidia-build", "kilo-free", "opencode-free"]

    pinned = app._call_order("kilo-free", "some-new-model-free")
    assert pinned[0][0] == "kilo-free"
    assert pinned[0][2][0] == "some-new-model-free"
    assert [name for name, _, _ in pinned][1:] == [
        "groq", "gemini-free", "nvidia-build", "opencode-free"]


def test_each_provider_contributes_at_most_the_cap():
    for _, _, models in app._call_order("", ""):
        assert len(models) <= app.MODELS_PER_PROVIDER


def test_discover_models_caches_strips_prefix_and_skips_keyless(monkeypatch):
    calls = {"n": 0}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "models/gemini-2.5-flash"}, {"id": "brand-new-free"}, {"id": ""}]}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return Resp()

    monkeypatch.setattr(app.requests, "get", fake_get)
    first = app.discover_models("opencode-free")
    second = app.discover_models("opencode-free")
    assert calls["n"] == 1 and first == second
    assert "gemini-2.5-flash" in first                     # "models/" prefix removed

    monkeypatch.setattr(app, "get_key", lambda _env: "")
    assert app.discover_models("groq") == []
    assert calls["n"] == 1                                 # no key, no request


def test_ranking_puts_strong_models_first_and_drops_non_chat():
    ids = ["meta/llama2-70b", "nvidia/nv-embedqa-1b", "moonshotai/kimi-k3",
           "nvidia/nemotron-3-ultra-550b-a55b", "deepseek-ai/deepseek-v4.1-flash",
           "nvidia/llama-3.1-nemoguard-8b-content-safety"]
    ranked = app._rank_models("nvidia-build", ids)
    # Fast Nemotron-3 first; DeepSeek / Kimi queue past 60s on the free tier.
    assert ranked[:3] == ["nvidia/nemotron-3-ultra-550b-a55b", "deepseek-ai/deepseek-v4.1-flash",
                          "moonshotai/kimi-k3"]
    assert not any("embed" in m or "guard" in m or "llama2" in m for m in ranked)


def test_keyless_providers_only_get_free_models():
    ranked = app._rank_models("kilo-free", ["z-ai/glm-5.3", "stepfun/step-3.7-flash:free",
                                            "kilo-auto/free", "deepseek/deepseek-v4-pro"])
    assert ranked == ["stepfun/step-3.7-flash:free", "kilo-auto/free"]


# ── Gemini stays on the free tier ───────────────────────────
@pytest.mark.parametrize("model,ok", [
    ("gemini-2.5-flash", True),
    ("gemini-2.5-flash-lite", True),
    ("gemini-3-flash", True),
    ("gemini-2.5-pro", False),
    ("gemini-2.5-flash-preview-09-2025", False),
    ("gemini-2.5-flash-image", False),
    ("gemini-2.5-flash-native-audio", False),
    ("gemini-live-2.5-flash", False),
])
def test_gemini_free_tier_allowlist(model, ok):
    assert app._allowed("gemini-free", model) is ok


def test_pinned_gemini_pro_is_refused():
    order = dict((n, m) for n, _, m in app._call_order("gemini-free", "gemini-2.5-pro"))
    assert "gemini-2.5-pro" not in order["gemini-free"]
    assert order["gemini-free"]                          # still has free models to try


def test_gemini_picker_never_lists_paid_models(monkeypatch):
    monkeypatch.setattr(app, "discover_models", lambda _p: [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash-image"])
    assert app.available_models("gemini-free") == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert app.available_models("gemini-free", free_only=False) == [
        "gemini-2.5-flash", "gemini-2.5-flash-lite"]


# ── failure memory ──────────────────────────────────────────
class FakeResp:
    def __init__(self, status, content="OK"):
        self.status_code = status
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": {}}


def test_permanent_failure_is_parked_and_skipped_next_time(monkeypatch):
    seen = []

    def fake_post(cfg, key, payload, stream=False, timeout=60):
        seen.append(payload["model"])
        return FakeResp(410 if payload["model"] == "dead-free" else 200)

    monkeypatch.setattr(app, "_post_chat", fake_post)
    monkeypatch.setattr(app, "LLM_PROVIDER_ORDER", ["kilo-free"])
    monkeypatch.setitem(app.LLM_PROVIDERS["kilo-free"], "models", ["dead-free", "live-free"])
    monkeypatch.setattr(app.time, "sleep", lambda s: None)

    r1 = app.llm_chat([{"role": "user", "content": "hi"}])
    assert r1["ok"] and r1["model"] == "live-free"
    assert seen == ["dead-free", "live-free"]                 # 410: no retry on the dead one

    seen.clear()
    r2 = app.llm_chat([{"role": "user", "content": "hi"}])
    assert seen == ["live-free"]                              # parked model skipped
    assert app._MODEL_HOLD[("kilo-free", "dead-free")] > time.time() + 1000


def test_rate_limit_gets_one_retry_then_short_hold(monkeypatch):
    seen = []

    def fake_post(cfg, key, payload, stream=False, timeout=60):
        seen.append(payload["model"])
        return FakeResp(429 if payload["model"] == "busy-free" else 200)

    monkeypatch.setattr(app, "_post_chat", fake_post)
    monkeypatch.setattr(app, "LLM_PROVIDER_ORDER", ["kilo-free"])
    monkeypatch.setitem(app.LLM_PROVIDERS["kilo-free"], "models", ["busy-free", "live-free"])
    monkeypatch.setattr(app.time, "sleep", lambda s: None)

    app.llm_chat([{"role": "user", "content": "hi"}])
    assert seen == ["busy-free", "busy-free", "live-free"]
    hold = app._MODEL_HOLD[("kilo-free", "busy-free")] - time.time()
    assert 0 < hold <= app._HOLD_TRANSIENT_S


def test_timeout_parks_model_for_ten_minutes_without_retry(monkeypatch):
    seen = []

    def fake_post(cfg, key, payload, stream=False, timeout=60):
        seen.append(payload["model"])
        if payload["model"] == "slow-free":
            raise app.requests.Timeout("read timed out")
        return FakeResp(200)

    monkeypatch.setattr(app, "_post_chat", fake_post)
    monkeypatch.setattr(app, "LLM_PROVIDER_ORDER", ["kilo-free"])
    monkeypatch.setitem(app.LLM_PROVIDERS["kilo-free"], "models", ["slow-free", "live-free"])
    monkeypatch.setattr(app.time, "sleep", lambda s: None)

    assert app.llm_chat([{"role": "user", "content": "hi"}])["model"] == "live-free"
    assert seen == ["slow-free", "live-free"]                  # no second wait on the slow one
    hold = app._MODEL_HOLD[("kilo-free", "slow-free")] - time.time()
    assert app._HOLD_TRANSIENT_S < hold <= app._HOLD_SLOW_S


def test_bad_key_blocks_whole_provider(monkeypatch):
    seen = []

    def fake_post(cfg, key, payload, stream=False, timeout=60):
        seen.append(payload["model"])
        return FakeResp(401)

    monkeypatch.setattr(app, "_post_chat", fake_post)
    monkeypatch.setattr(app, "LLM_PROVIDER_ORDER", ["groq"])
    monkeypatch.setattr(app, "get_key", lambda _env: "bad-key")
    monkeypatch.setattr(app.time, "sleep", lambda s: None)

    r = app.llm_chat([{"role": "user", "content": "hi"}])
    assert not r["ok"]
    assert len(seen) == 1                                     # stopped after the first 401
    assert not app._usable("groq", "anything")


def test_chart_spec_keeps_known_layers_only():
    spec = app.set_chart_spec({
        "ticker": "btc-usd",
        "range": "nope",
        "ticker_b": "eth-usd",
        "overlays": ["sma50", "not-a-layer"],
        "panels": ["rsi", "volume"],
        "marks": "backtest",
        "rule": "ma_cross",
    })
    assert spec["ticker"] == "BTC-USD"
    assert spec["compare"] == "ETH-USD"
    assert spec["range"] == "1Y"
    assert spec["overlays"] == ["sma50"]
    assert spec["panels"] == ["rsi", "volume"]
    assert spec["marks"] == "backtest"
    assert spec["rule"] == "ma_cross"
