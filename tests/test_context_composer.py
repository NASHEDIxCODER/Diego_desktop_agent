"""
Regression tests for agent.context_composer (Phase 21A — Task 18 asked
for the context-composer test suite; none existed).

Covers:
  - empty memory → empty context block
  - memories are ranked and injected into a compact block
  - hard token budget is respected (never balloons)
  - similar memories are compressed (deduped)
"""

from __future__ import annotations

import time

from agent.context_composer import ContextComposer


def _fake_embedding_fn(dim: int = 8):
    def fn(text: str):
        # Tiny deterministic bag-of-words embedding.
        vec = [0.0] * dim
        for i, ch in enumerate(text.lower()):
            vec[i % dim] += (ord(ch) % 7) / 7.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
    return fn


from types import SimpleNamespace


def _store_with(monkeypatch, specs):
    """Wire a fake learning engine into the composer's collection.
    `specs` are SimpleNamespace records shaped like the real engine's
    entries (category/key/value/confidence/frequency/timestamp)."""

    class FakeProfile:
        _facts = [s for s in specs if s.kind == "fact"]

    class FakePrefs:
        _preferences = [s for s in specs if s.kind == "pref"]

    class FakeHabits:
        _habits = [s for s in specs if s.kind == "habit"]

    class FakeSkills:
        _entries = [s for s in specs if s.kind == "skill"]

        @staticmethod
        def success_rate(action):
            return 0.8

    class FakeEngine:
        profile = FakeProfile()
        preferences = FakePrefs()
        habits = FakeHabits()
        skills = FakeSkills()

    import sys
    fake_mod = type(sys)("learning.learning_engine")
    fake_mod.learning_engine = FakeEngine()
    monkeypatch.setitem(sys.modules, "learning.learning_engine", fake_mod)

    import agent.context_composer as cc
    return cc


def _fact(key, value, confidence=0.9, timestamp=None):
    return SimpleNamespace(
        kind="fact", category="app", key=key, value=value,
        confidence=confidence, frequency=1.0,
        timestamp=timestamp if timestamp is not None else time.time())


def test_empty_memories_empty_context():
    composer = ContextComposer()
    composer.set_embedding_fn(_fake_embedding_fn())
    assert composer.compose("anything at all", max_tokens=200) == ""


def test_relevant_memory_is_injected(monkeypatch):
    cc = _store_with(monkeypatch, [_fact("browser", "prefers firefox")])
    composer = cc.ContextComposer()
    composer.set_embedding_fn(_fake_embedding_fn())
    block = composer.compose("which browser do I prefer", max_tokens=200)
    assert "firefox" in block


def test_token_budget_respected(monkeypatch):
    items = [_fact(f"{i}", "memory value " * 40 + str(i),
                   confidence=0.5 + i * 0.04) for i in range(10)]
    cc = _store_with(monkeypatch, items)
    composer = cc.ContextComposer()
    composer.set_embedding_fn(_fake_embedding_fn())
    block = composer.compose("memory", max_tokens=150)
    # ~4 chars/token heuristic ceiling + formatting slack.
    assert len(block) <= 150 * 4 + 200


def test_similar_memories_compressed(monkeypatch):
    # Compression kicks in when one source type has MORE than 5 items:
    # top-3 stay individual, the rest merge into one summary line.
    items = [_fact(str(i), f"repeated memory {i}") for i in range(8)]
    cc = _store_with(monkeypatch, items)
    composer = cc.ContextComposer()
    composer.set_embedding_fn(_fake_embedding_fn())
    block = composer.compose("repeated memory", max_tokens=600)
    lines = [l for l in block.splitlines()
             if l.strip().startswith("-")]
    # 3 individual entries + 1 merged summary line (values preserved
    # verbatim inside the summary); the block header is not an entry.
    assert len(lines) == 4
    assert block.count("repeated memory") == 8
    assert "; " in lines[-1]  # the merged summary line
