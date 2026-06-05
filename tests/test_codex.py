import json
from pathlib import Path

from src.adapters import codex


def _write_session(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "session.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return p


def _token_count_event(rate_limits: dict, context_window: int | None = 258400) -> dict:
    info = {"total_token_usage": {"input_tokens": 1, "output_tokens": 1}}
    if context_window is not None:
        info["model_context_window"] = context_window
    return {
        "timestamp": "2026-06-04T20:00:00.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": info,
            "rate_limits": rate_limits,
        },
    }


def test_free_plan_7d_bucket_routed_correctly(tmp_path):
    # Free plan: primary is the 7-day window (10080 min), secondary is null.
    # Old code put primary into the 5h slot, leaving 7d empty.
    rl = {
        "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
        "plan_type": "free",
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.seven_day_pct == 42.0
    assert result.five_hour_pct is None
    assert result.plan_type == "free"
    assert result.context_window == 258400


def test_paid_plan_both_buckets_routed(tmp_path):
    rl = {
        "primary": {"used_percent": 12.0, "window_minutes": 300, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 60.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "plan_type": "pro",
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.five_hour_pct == 12.0
    assert result.seven_day_pct == 60.0
    assert result.plan_type == "pro"


def test_swapped_window_order_still_routed_by_window_minutes(tmp_path):
    # Defensive: if OpenAI ever swaps primary/secondary order, bucket assignment
    # must follow window_minutes, not positional convention.
    rl = {
        "primary": {"used_percent": 55.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 8.0, "window_minutes": 300, "resets_at": 9_999_999_999},
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.five_hour_pct == 8.0
    assert result.seven_day_pct == 55.0


def test_expired_reset_zeros_out_pct(tmp_path):
    rl = {
        "primary": {"used_percent": 99.0, "window_minutes": 10080, "resets_at": 1},
        "secondary": None,
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.seven_day_pct == 0.0
