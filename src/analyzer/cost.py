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


def get_pricing() -> dict:
    global _pricing
    if _pricing is not None:
        return _pricing
    _pricing = _load_pricing()
    return _pricing


def calculate_cost(entry: UsageEntry) -> float:
    if entry.cost_usd is not None:
        return entry.cost_usd

    pricing = get_pricing()
    model_key = _resolve_model_key(entry.model, pricing)
    if model_key is None:
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
    return None


def _load_pricing() -> dict:
    cached = _read_cache()
    if cached is not None and not _cache_stale():
        return cached

    # 缓存缺失或过期 → 尝试联网刷新；失败时优先用旧缓存（哪怕过期），最后才用内置兜底
    try:
        return _fetch_and_cache()
    except (URLError, TimeoutError, ssl.SSLError, OSError, json.JSONDecodeError):
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


def _fallback_pricing() -> dict:
    return {
        "claude-opus-4-6": {
            "input_cost_per_token": 15e-6,
            "output_cost_per_token": 75e-6,
            "cache_creation_input_token_cost": 18.75e-6,
            "cache_read_input_token_cost": 1.5e-6,
        },
        "claude-opus-4-7": {
            "input_cost_per_token": 15e-6,
            "output_cost_per_token": 75e-6,
            "cache_creation_input_token_cost": 18.75e-6,
            "cache_read_input_token_cost": 1.5e-6,
        },
        "claude-sonnet-4-6": {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_creation_input_token_cost": 3.75e-6,
            "cache_read_input_token_cost": 0.3e-6,
        },
        "claude-haiku-4-5-20251001": {
            "input_cost_per_token": 0.8e-6,
            "output_cost_per_token": 4e-6,
            "cache_creation_input_token_cost": 1e-6,
            "cache_read_input_token_cost": 0.08e-6,
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
