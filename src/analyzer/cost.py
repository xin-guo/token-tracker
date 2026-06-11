import http.client
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

from ..adapters.types import UsageEntry

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
# 缓存放用户可写目录：包根目录在 site-packages 下是只读的，写失败会导致每次都联网
CACHE_DIR = Path(os.path.expanduser("~/.cache/token-tracker"))
CACHE_PATH = CACHE_DIR / "pricing_cache.json"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 定价表 7 天过期，过期后尝试刷新（失败仍用旧缓存兜底）
_INSECURE_TLS_ENV = "TT_PRICING_INSECURE_TLS"

_pricing: dict | None = None
_warned_insecure = False

# 未知的新 Claude 模型按系列退回最新已知价（这些 key 由 _fallback_pricing 保证存在）
_FAMILY_FALLBACK = (
    ("claude-opus", "claude-opus-4-8"),
    ("claude-sonnet", "claude-sonnet-4-6"),
    ("claude-haiku", "claude-haiku-4-5-20251001"),
    ("claude-fable", "claude-fable-5"),
)

# 解析不到定价的模型只提示一次，避免聚合时每条 entry 刷屏
_warned_unknown_models: set[str] = set()


def get_pricing() -> dict:
    global _pricing
    if _pricing is not None:
        return _pricing
    # 以 litellm 数据为准，_fallback_pricing 作为已知价底座补 litellm 尚未收录的新模型（如最新 Opus）
    _pricing = {**_fallback_pricing(), **_load_pricing()}
    return _pricing


def calculate_cost(entry: UsageEntry) -> float:
    if entry.cost_usd is not None:
        return entry.cost_usd

    pricing = get_pricing()
    model_key = _resolve_model_key(entry.model, pricing)
    if model_key is None:
        _warn_unknown_model_once(entry.model)
        return 0.0

    info = pricing[model_key]
    input_cost = info.get("input_cost_per_token", 0)
    output_cost = info.get("output_cost_per_token", 0)
    cache_creation_cost = info.get("cache_creation_input_token_cost", input_cost * 1.25)
    cache_read_cost = info.get("cache_read_input_token_cost", input_cost * 0.1)

    return (
        entry.input_tokens * input_cost
        + entry.output_tokens * output_cost
        + entry.cache_creation_tokens * cache_creation_cost
        + entry.cache_read_tokens * cache_read_cost
    )


def _resolve_model_key(model: str, pricing: dict) -> str | None:
    if not model:
        return None
    if model in pricing:
        return model

    ml = model.lower()
    # model 以 key 为前缀：处理 dated/variant 后缀（gpt-5-codex-2025-12-01 → gpt-5-codex）
    # 取最长匹配，避免 gpt-5 误吞 gpt-5-codex-* 这种更具体的 key
    prefix_keys = [k for k in pricing if ml.startswith(k.lower())]
    if prefix_keys:
        return max(prefix_keys, key=len)
    # 反向兜底：key 以 model + "-" 开头（gpt-5 命中 gpt-5-2025-08-07）
    # 加 "-" 锚点避免 gpt-5 撞上 gpt-5-mini
    suffix_keys = [k for k in pricing if k.lower().startswith(ml + "-")]
    if suffix_keys:
        return min(suffix_keys, key=len)

    # 同系列兜底：未知的新 Claude 模型（litellm 收录滞后）退回同系列最新已知价，避免成本静默归零
    for prefix, fallback_key in _FAMILY_FALLBACK:
        if ml.startswith(prefix) and fallback_key in pricing:
            return fallback_key
    return None


def _warn_unknown_model_once(model: str) -> None:
    # 全新系列（非 opus/sonnet/haiku/fable）连系列兜底都接不住，成本会按 $0 计；显形以免静默少算
    if model and model not in _warned_unknown_models:
        _warned_unknown_models.add(model)
        print(
            f"token-tracker: 未知模型 {model!r} 缺少定价，本次成本按 $0 计；"
            "litellm 收录后自动恢复，或在 cost.py 的 _fallback_pricing 补内置价",
            file=sys.stderr,
        )


def _load_pricing() -> dict:
    cached = _read_cache()
    if cached is not None and not _cache_stale():
        return cached

    # 缓存缺失或过期 → 尝试联网刷新；失败时优先用旧缓存（哪怕过期），最后才用内置兜底。
    # 异常集要覆盖整条抓取链：URLError/TimeoutError/ssl.SSLError/OSError（网络与 socket），
    # http.client.HTTPException（IncompleteRead 等截断响应），ValueError（JSON 解析 + decode 失败）。
    # 仍不用裸 except，避免吞掉 AttributeError/KeyError 这类真 bug。
    try:
        return _fetch_and_cache()
    except (URLError, TimeoutError, ssl.SSLError, OSError, http.client.HTTPException, ValueError):
        if cached is not None:
            return cached
        return _fallback_pricing()


def _read_cache() -> dict | None:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_stale() -> bool:
    try:
        age = time.time() - CACHE_PATH.stat().st_mtime
    except OSError:
        return True
    return age > CACHE_TTL_SECONDS


def _fetch_and_cache() -> dict:
    try:
        data = _fetch(verify=True)
    except ssl.SSLCertVerificationError:
        # 默认不静默降级 TLS：仅当用户显式 TT_PRICING_INSECURE_TLS=1 时才放行（抓的是公开定价表）
        if os.environ.get(_INSECURE_TLS_ENV) != "1":
            raise
        _warn_insecure_once()
        data = _fetch(verify=False)

    _write_cache(data)
    return data


def _fetch(verify: bool) -> dict:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(LITELLM_URL, headers={"User-Agent": "token-tracker/0.1"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        return json.loads(resp.read().decode())


def _write_cache(data: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


def _warn_insecure_once() -> None:
    global _warned_insecure
    if not _warned_insecure:
        print(
            f"token-tracker: 已按 {_INSECURE_TLS_ENV}=1 关闭 TLS 证书校验（仅用于抓取公开定价表）",
            file=sys.stderr,
        )
        _warned_insecure = True


# Anthropic 官方价（claude.com/pricing 2026-06 核对）。opus 4.5/4.6/4.7/4.8 同价。
_OPUS_PRICING = {
    "input_cost_per_token": 5e-6,
    "output_cost_per_token": 25e-6,
    "cache_creation_input_token_cost": 6.25e-6,
    "cache_read_input_token_cost": 0.5e-6,
}

# Fable 5 / Mythos 5 同价，是 Opus 档的 2 倍（$10/$50 每百万 token）
_FABLE_PRICING = {
    "input_cost_per_token": 10e-6,
    "output_cost_per_token": 50e-6,
    "cache_creation_input_token_cost": 12.5e-6,
    "cache_read_input_token_cost": 1.0e-6,
}


def _fallback_pricing() -> dict:
    return {
        "claude-fable-5": _FABLE_PRICING,
        "claude-opus-4-8": _OPUS_PRICING,
        "claude-opus-4-7": _OPUS_PRICING,
        "claude-opus-4-6": _OPUS_PRICING,
        "claude-opus-4-5": _OPUS_PRICING,
        "claude-sonnet-4-6": {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_creation_input_token_cost": 3.75e-6,
            "cache_read_input_token_cost": 0.3e-6,
        },
        "claude-haiku-4-5-20251001": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 5e-6,
            "cache_creation_input_token_cost": 1.25e-6,
            "cache_read_input_token_cost": 0.1e-6,
        },
        "gpt-5": {
            "input_cost_per_token": 1.25e-6,
            "output_cost_per_token": 10e-6,
            "cache_read_input_token_cost": 0.125e-6,
        },
        "gpt-5-codex": {
            "input_cost_per_token": 1.25e-6,
            "output_cost_per_token": 10e-6,
            "cache_read_input_token_cost": 0.125e-6,
        },
        "gpt-5-mini": {
            "input_cost_per_token": 0.25e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 0.025e-6,
        },
        "gpt-5-nano": {
            "input_cost_per_token": 0.05e-6,
            "output_cost_per_token": 0.4e-6,
            "cache_read_input_token_cost": 0.005e-6,
        },
        "gpt-5-pro": {
            "input_cost_per_token": 15e-6,
            "output_cost_per_token": 120e-6,
        },
        "codex-mini-latest": {
            "input_cost_per_token": 1.5e-6,
            "output_cost_per_token": 6e-6,
            "cache_read_input_token_cost": 0.375e-6,
        },
    }
