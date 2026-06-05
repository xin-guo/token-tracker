import http.client
import json
from datetime import UTC, datetime
from urllib.error import URLError

import pytest

from src.adapters.types import UsageEntry
from src.analyzer import cost


def make_entry(**kw):
    defaults = dict(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        session_id="s1",
        message_id="m1",
        request_id="r1",
        model="claude-opus-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        cost_usd=None,
        project="proj",
        agent_id="claude",
    )
    defaults.update(kw)
    return UsageEntry(**defaults)


@pytest.fixture
def fixed_pricing(monkeypatch):
    """Inject deterministic pricing so tests never hit the network or cache file."""
    pricing = {
        "claude-opus-4-6": {
            "input_cost_per_token": 15e-6,
            "output_cost_per_token": 75e-6,
            "cache_creation_input_token_cost": 18.75e-6,
            "cache_read_input_token_cost": 1.5e-6,
        },
    }
    monkeypatch.setattr(cost, "_pricing", pricing)
    return pricing


def test_explicit_cost_is_passed_through(fixed_pricing):
    # When the entry already carries a cost, pricing must be ignored entirely.
    entry = make_entry(cost_usd=1.23, input_tokens=999_999)
    assert cost.calculate_cost(entry) == 1.23


def test_cost_computed_from_pricing(fixed_pricing):
    entry = make_entry(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    # 15 + 75 + 18.75 + 1.5
    assert cost.calculate_cost(entry) == pytest.approx(110.25)


def test_unknown_model_returns_zero(fixed_pricing):
    entry = make_entry(model="totally-unknown-xyz", input_tokens=1_000_000)
    assert cost.calculate_cost(entry) == 0.0


def test_model_resolved_by_substring(fixed_pricing):
    # A dated model id should resolve to its base pricing key via prefix match.
    entry = make_entry(model="claude-opus-4-6-20260101", input_tokens=1_000_000)
    assert cost.calculate_cost(entry) == pytest.approx(15.0)


@pytest.fixture
def openai_pricing(monkeypatch):
    pricing = {
        "gpt-5": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
        "gpt-5-mini": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 2e-6},
        "gpt-5-codex": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
    }
    monkeypatch.setattr(cost, "_pricing", pricing)
    return pricing


def test_gpt5_exact_match_does_not_leak_to_mini(openai_pricing):
    # Regression: old `in` substring matching could resolve "gpt-5" against
    # "gpt-5-mini" first and silently use the cheaper mini pricing.
    entry = make_entry(model="gpt-5", input_tokens=1_000_000)
    assert cost.calculate_cost(entry) == pytest.approx(1.25)


def test_dated_variant_resolves_to_longest_base(openai_pricing):
    # A dated codex id should resolve to gpt-5-codex (length 11), not gpt-5 (length 5).
    entry = make_entry(model="gpt-5-codex-2025-12-01", input_tokens=1_000_000)
    assert cost.calculate_cost(entry) == pytest.approx(1.25)


def test_base_name_resolves_to_dated_pricing(openai_pricing, monkeypatch):
    # Reverse-direction fallback: pricing only has dated keys, model is the base name.
    pricing = {
        "gpt-5-2025-08-07": {"input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6},
        "gpt-5-mini-2025-08-07": {"input_cost_per_token": 0.25e-6, "output_cost_per_token": 2e-6},
    }
    monkeypatch.setattr(cost, "_pricing", pricing)
    entry = make_entry(model="gpt-5", input_tokens=1_000_000)
    # Must pick gpt-5-2025-08-07 (shorter), not gpt-5-mini-2025-08-07.
    assert cost.calculate_cost(entry) == pytest.approx(1.25)


def test_fallback_pricing_includes_openai_models():
    pricing = cost._fallback_pricing()
    for k in ("gpt-5", "gpt-5-codex", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "codex-mini-latest"):
        assert k in pricing, f"fallback pricing missing {k}"
        assert pricing[k].get("input_cost_per_token", 0) > 0


def test_fresh_cache_is_used_without_fetching(tmp_path, monkeypatch):
    cache = tmp_path / "pricing_cache.json"
    cache.write_text('{"gpt-5": {"input_cost_per_token": 1e-6}}', encoding="utf-8")
    monkeypatch.setattr(cost, "CACHE_PATH", cache)
    monkeypatch.setattr(cost, "_cache_stale", lambda: False)
    # 新鲜缓存命中时绝不能联网
    monkeypatch.setattr(cost, "_fetch_and_cache", lambda: (_ for _ in ()).throw(AssertionError("不应联网")))
    assert cost._load_pricing() == {"gpt-5": {"input_cost_per_token": 1e-6}}


def test_stale_cache_kept_when_fetch_fails(tmp_path, monkeypatch):
    # 关键：缓存过期但联网失败时，应保留旧缓存而非掉到内置兜底
    cache = tmp_path / "pricing_cache.json"
    cache.write_text('{"gpt-5": {"input_cost_per_token": 9e-6}}', encoding="utf-8")
    monkeypatch.setattr(cost, "CACHE_PATH", cache)
    monkeypatch.setattr(cost, "_cache_stale", lambda: True)

    def boom():
        raise URLError("offline")

    monkeypatch.setattr(cost, "_fetch_and_cache", boom)
    assert cost._load_pricing() == {"gpt-5": {"input_cost_per_token": 9e-6}}


def test_builtin_fallback_only_when_no_cache_and_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cost, "CACHE_PATH", tmp_path / "missing.json")

    def boom():
        raise TimeoutError("offline")

    monkeypatch.setattr(cost, "_fetch_and_cache", boom)
    result = cost._load_pricing()
    assert "gpt-5" in result and "claude-opus-4-7" in result  # 命中内置兜底表


@pytest.mark.parametrize("exc", [
    http.client.IncompleteRead(b""),          # 截断的 HTTP/1.1 响应
    UnicodeDecodeError("utf-8", b"", 0, 1, "bad"),  # resp.read().decode() 失败（ValueError 子类）
    json.JSONDecodeError("bad", "", 0),        # 损坏的 JSON
])
def test_stale_cache_survives_middownload_errors(tmp_path, monkeypatch, exc):
    # 回归：抓取链中途抛 HTTPException/decode/JSON 错误时，必须用旧缓存兜底而非崩溃
    cache = tmp_path / "pricing_cache.json"
    cache.write_text('{"gpt-5": {"input_cost_per_token": 7e-6}}', encoding="utf-8")
    monkeypatch.setattr(cost, "CACHE_PATH", cache)
    monkeypatch.setattr(cost, "_cache_stale", lambda: True)

    def boom():
        raise exc

    monkeypatch.setattr(cost, "_fetch_and_cache", boom)
    assert cost._load_pricing() == {"gpt-5": {"input_cost_per_token": 7e-6}}
