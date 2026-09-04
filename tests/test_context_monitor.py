"""
Regression tests for the runtime context-window measurement + guard.

Covers:
  - context token accounting (input / output / total / remaining)
  - reported vs estimated token counts (never conflated)
  - context limit guard (refuses prompts that exceed the window)
  - output reservation (budget is reserved before the request)
  - priority-based context trimming (lowest priority dropped first,
    current user request / task state never blindly truncated)
  - context-status diagnostic payload
"""

from __future__ import annotations

import pytest

from ai.context_monitor import (
    ContextMonitor,
    ContextPriority,
    ContextUsage,
)


def _monitor(limit: int = 1000, reserve: int = 100) -> ContextMonitor:
    m = ContextMonitor()
    m.configure("test-model", context_limit=limit, default_output_reserve=reserve)
    return m


# ═══════════════════════════════════════════════════════════════
# Token accounting
# ═══════════════════════════════════════════════════════════════

def test_token_accounting_reported():
    m = _monitor(limit=10000)
    usage = m.record_request(
        model="qwen2.5:1.5b",
        prompt="some prompt",
        max_output=200,
        provider_usage={"prompt_eval_count": 18234, "eval_count": 921},
        response_text="ignored when reported",
    )
    assert usage.estimated is False
    assert usage.input_tokens == 18234
    assert usage.output_tokens == 921
    assert usage.total_tokens == 18234 + 921
    assert usage.max_output_requested == 200
    # remaining is clamped at 0 when usage exceeds the limit
    assert usage.remaining == max(0, 10000 - usage.total_tokens)


def test_token_accounting_estimated():
    m = _monitor(limit=100000)
    usage = m.record_request(
        model="qwen2.5:1.5b",
        prompt="hello world this is a prompt",
        response_text="a short answer",
    )
    assert usage.estimated is True
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.remaining == 100000 - usage.total_tokens


def test_reported_vs_estimated_never_conflated():
    m = _monitor(limit=50000)
    reported = m.record_request(
        provider_usage={"prompt_eval_count": 10, "eval_count": 5})
    estimated = m.record_request(prompt="estimate me", response_text="ok")
    assert reported.estimated is False
    assert estimated.estimated is True


def test_estimate_tokens_positive_and_monotonic():
    m = _monitor()
    small = m.estimate_tokens("hi")
    large = m.estimate_tokens("word " * 500)
    assert small >= 1
    assert large > small


# ═══════════════════════════════════════════════════════════════
# Context limit guard + output reservation
# ═══════════════════════════════════════════════════════════════

def test_guard_allows_small_prompt():
    m = _monitor(limit=10000, reserve=100)
    ok, reason, tokens = m.pre_request_check("a short prompt")
    assert ok is True
    assert reason == ""
    assert tokens >= 1


def test_guard_refuses_oversized_prompt():
    m = _monitor(limit=100, reserve=10)
    huge = "word " * 1000
    ok, reason, tokens = m.pre_request_check(huge)
    assert ok is False
    assert "exceeds" in reason
    assert tokens > 0


def test_guard_reserves_output_budget():
    # With a tiny limit, a large output reserve must leave no room.
    m = _monitor(limit=100, reserve=100)
    ok, reason, _ = m.pre_request_check("anything at all")
    assert ok is False
    assert "reserve" in reason


def test_guard_output_reservation_shrinks_budget():
    m = _monitor(limit=1000, reserve=10)
    prompt = "word " * 200  # ~200-260 tokens estimate
    ok_small_reserve, _, _ = m.pre_request_check(prompt, reserve_output=10)
    ok_big_reserve, _, _ = m.pre_request_check(prompt, reserve_output=900)
    # A bigger output reserve leaves less room for the prompt.
    assert ok_small_reserve is True
    assert ok_big_reserve is False


# ═══════════════════════════════════════════════════════════════
# Priority-based context trimming
# ═══════════════════════════════════════════════════════════════

def test_trim_drops_lowest_priority_first():
    m = _monitor(limit=200, reserve=0)
    items = [
        (ContextPriority.OLDER_CONTEXT, "old " * 100),          # low, big
        (ContextPriority.RELEVANT_HISTORY, "history " * 20),
        (ContextPriority.USER_REQUEST, "play believer on youtube"),
    ]
    kept = m.trim_to_fit(items, reserve_output=0)
    # The current user request always survives.
    assert "play believer on youtube" in kept
    # The big low-priority block is dropped first.
    assert ("old " * 100) not in kept


def test_trim_keeps_task_state_and_user_request():
    m = _monitor(limit=120, reserve=0)
    user_req = "the current user request"
    task_state = "the current task state"
    items = [
        (ContextPriority.OLDER_CONTEXT, "noise " * 200),
        (ContextPriority.LOCAL_KNOWLEDGE, "knowledge " * 50),
        (ContextPriority.TASK_STATE, task_state),
        (ContextPriority.USER_REQUEST, user_req),
    ]
    kept = m.trim_to_fit(items, reserve_output=0)
    assert user_req in kept
    assert task_state in kept


def test_trim_returns_priority_order():
    m = _monitor(limit=10000, reserve=0)
    items = [
        (ContextPriority.RELEVANT_HISTORY, "history"),
        (ContextPriority.USER_REQUEST, "request"),
        (ContextPriority.LIVE_STATE, "live"),
    ]
    kept = m.trim_to_fit(items, reserve_output=0)
    # Highest priority first.
    assert kept[0] == "request"


# ═══════════════════════════════════════════════════════════════
# Diagnostics / logging
# ═══════════════════════════════════════════════════════════════

def test_context_status_payload():
    m = _monitor(limit=4096, reserve=128)
    m.record_request(prompt="hello", response_text="hi")
    status = m.context_status()
    assert status["model"] == "test-model"
    assert status["context_limit"] == 4096
    assert status["default_output_reserve"] == 128
    assert status["requests_recorded"] == 1
    assert status["last"] is not None
    assert "input_tokens" in status["last"]
    assert "estimated" in status["last"]


def test_compact_log_has_no_prompt_content():
    usage = ContextUsage(
        model="qwen2.5:1.5b", context_limit=131072,
        input_tokens=18234, output_tokens=921, total_tokens=19155,
        remaining=111917, max_output_requested=200, estimated=False,
    )
    line = usage.compact_log()
    assert line.startswith("CONTEXT model=qwen2.5:1.5b")
    assert "input=18234" in line
    assert "output=921" in line
    assert "total=19155" in line
    assert "limit=131072" in line
    assert "remaining=111917" in line
    assert "reported" in line
    # The metric line must not leak any prompt/response text fields.
    assert "prompt" not in line.lower().replace("prompt_eval", "")


def test_cli_context_status_command(capsys):
    from knowledge.cli import main
    rc = main(["context-status"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "context_limit" in captured.out