"""
Regression tests for the runtime audit fixes (B1, B2, B3, B4).

Covers:
  B1 — personality.task_confirmation() must never produce malformed
       fragments (". Easy.", "— all set.", "Alright,") when detail is
       empty; normal responses preserved when detail exists.
  B2 — ActionDispatcher failure strings must not short-circuit the
       fallback path; fallback success must be returned.
  B3 — MusicAgent must report honest results based on playerctl exit
       status (no-player → failure string; running player → success);
       Brain._verify must treat known failure strings as verification
       failure.
  B4 — audit harness dispatches through Brain._dispatch_and_verify
       (production path with retries), not raw dispatcher + verify.

Run with:
    python -m pytest tests/test_audit_fixes.py -v
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stubs)

from agent.brain import AgentBrain
from agent.personality import DiegoPersonality


# ═══════════════════════════════════════════════════════════════
# B1 — personality.task_confirmation()
# ═══════════════════════════════════════════════════════════════

class TestB1TaskConfirmation:
    """Empty detail must never yield malformed fragments."""

    def setup_method(self):
        self.p = DiegoPersonality()

    def test_empty_detail_never_fragment(self):
        """Exhaustively render every confirmation template with empty
        detail — no fragment like '. Easy.' may ever be produced."""
        # Drain the standalone pool exhaustively (recent-list resets
        # when exhausted, so 2x pool size guarantees full coverage).
        for _ in range(len(self.p._standalone_confirmations) * 2):
            r = self.p.task_confirmation("")
            assert r == r.strip(), f"leading/trailing whitespace: {r!r}"
            assert not r.startswith((".", ",", "—", "–", "-", ";", ":")), \
                f"leading punctuation fragment: {r!r}"
            assert not r.endswith(","), f"dangling comma: {r!r}"
            assert r.endswith((".", "!", "?")), f"no terminal punctuation: {r!r}"
            assert len(r) > 2, f"empty/too-short response: {r!r}"

    def test_no_detail_template_fragments(self):
        """The three observed malformed fragments must be impossible."""
        for _ in range(200):
            r = self.p.task_confirmation("")
            assert r != ". Easy.", "reproduced '. Easy.'"
            assert r != "— all set.", "reproduced '— all set.'"
            assert r != "Alright,", "reproduced 'Alright,'"

    def test_detail_preserved(self):
        """With detail present, every response is a clean sentence.

        Templates WITH a {detail} slot must include the detail;
        standalone templates ("It's done.") are also valid responses.
        """
        detail_templates = [t for t in self.p._task_confirmations
                            if "{detail}" in t]
        assert detail_templates, "detail templates must exist"
        for _ in range(200):
            r = self.p.task_confirmation("firefox is open")
            assert "{detail}" not in r, f"unrendered placeholder: {r!r}"
            assert r == r.strip(), f"whitespace: {r!r}"
            assert not r.startswith((".", ",", "—")), \
                f"fragment with detail: {r!r}"
            assert r.endswith((".", "!", "?")), f"not a sentence: {r!r}"

    def test_detail_template_renders_detail(self):
        """A {detail} template must include the detail when rendered."""
        detail_templates = [t for t in self.p._task_confirmations
                            if "{detail}" in t]
        for _ in range(200):
            r = self.p.task_confirmation("firefox is open")
            # If a detail template was chosen, the detail must appear.
            if any(t.format(detail="firefox is open") == r
                   for t in detail_templates):
                assert "firefox is open" in r

    def test_detail_with_leading_punct_template(self):
        """'{detail}. Easy.' with detail must render cleanly."""
        r = self.p._clean_fragment("{detail}. Easy.".format(detail="done"))
        assert r == "done. Easy."

    def test_clean_fragment_sanitizer(self):
        cf = DiegoPersonality._clean_fragment
        assert cf(". Easy.") == "Easy."
        assert cf("— all set.") == "all set."
        assert cf("Alright,") == "Alright."
        assert cf("  Done.  ") == "Done."
        assert cf("") == ""

    def test_no_unrendered_placeholder(self):
        """No rendered confirmation may contain '{detail}'."""
        for _ in range(200):
            assert "{detail}" not in self.p.task_confirmation("")
            assert "{detail}" not in self.p.task_confirmation("x")


# ═══════════════════════════════════════════════════════════════
# B2 — ActionDispatcher failure contract
# ═══════════════════════════════════════════════════════════════

class TestB2DispatcherFallback:
    """Failure strings must not short-circuit the fallback path."""

    def setup_method(self):
        from agent.action_dispatcher import ActionDispatcher
        self.d = ActionDispatcher()

    def test_failure_result_detection(self):
        assert self.d._is_failure_result("Couldn't find gnome-terminal")
        assert self.d._is_failure_result("Volume control unavailable")
        assert self.d._is_failure_result("Couldn't open https://x")
        assert not self.d._is_failure_result("Opened firefox")
        assert not self.d._is_failure_result("Opened https://youtube.com")
        assert not self.d._is_failure_result("")
        assert not self.d._is_failure_result(None)

    def test_fallback_runs_after_primary_failure(self):
        """Primary returns a failure string → fallback must be attempted
        and its success returned instead of the failure string."""
        async def scenario():
            with patch.object(self.d, "_execute_sync",
                              return_value="Couldn't find gnome-terminal"), \
                 patch.object(self.d, "_execute_fallback",
                              return_value="Opened terminal via gtk-launch"):
                result = await self.d.execute(
                    {"action": "desktop_open", "params": {"app": "terminal"}})
            return result

        result = asyncio.run(scenario())
        assert result == "Opened terminal via gtk-launch", \
            f"fallback result not used: {result!r}"

    def test_primary_failure_returned_when_fallback_fails(self):
        """Both primary and fallback fail → original failure string is
        preserved (so Brain verification sees the failure)."""
        async def scenario():
            with patch.object(self.d, "_execute_sync",
                              return_value="Couldn't find gnome-terminal"), \
                 patch.object(self.d, "_execute_fallback", return_value=None):
                return await self.d.execute(
                    {"action": "desktop_open", "params": {"app": "terminal"}})

        result = asyncio.run(scenario())
        assert result == "Couldn't find gnome-terminal"

    def test_success_short_circuits_fallback(self):
        """Primary success must NOT trigger the fallback."""
        async def scenario():
            with patch.object(self.d, "_execute_sync",
                              return_value="Opened firefox"), \
                 patch.object(self.d, "_execute_fallback") as fb:
                result = await self.d.execute(
                    {"action": "desktop_open", "params": {"app": "firefox"}})
            return result, fb

        result, fb = asyncio.run(scenario())
        assert result == "Opened firefox"
        fb.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# B3 — MusicAgent honest results + Brain verification
# ═══════════════════════════════════════════════════════════════

class TestB3MusicAgent:
    """playerctl exit status must be authoritative."""

    def setup_method(self):
        from services.music_agent import MusicAgent
        self.agent = MusicAgent()

    def test_pause_no_player(self):
        with patch("services.music_agent.shutil.which", return_value=None):
            result = asyncio.run(self.agent.pause())
        assert "Couldn't" in result, f"false success: {result!r}"

    def test_resume_no_player(self):
        with patch("services.music_agent.shutil.which", return_value=None):
            result = asyncio.run(self.agent.resume())
        assert "Nothing to resume" in result, f"false success: {result!r}"

    def test_next_no_player(self):
        with patch("services.music_agent.shutil.which", return_value=None):
            result = asyncio.run(self.agent.next())
        assert "Couldn't" in result, f"false success: {result!r}"

    def test_previous_no_player(self):
        with patch("services.music_agent.shutil.which", return_value=None):
            result = asyncio.run(self.agent.previous())
        assert "Couldn't" in result, f"false success: {result!r}"

    def test_stop_no_player(self):
        with patch("services.music_agent.shutil.which", return_value=None):
            result = asyncio.run(self.agent.stop())
        assert "Nothing to stop" in result, f"false success: {result!r}"

    def test_pause_playerctl_fails(self):
        """playerctl exists but exits non-zero (no MPRIS player)."""
        fake = type("R", (), {"returncode": 1})()
        with patch("services.music_agent.shutil.which", return_value="/usr/bin/playerctl"), \
             patch("services.music_agent.subprocess.run", return_value=fake):
            result = asyncio.run(self.agent.pause())
        assert "Couldn't" in result, f"false success: {result!r}"

    def test_pause_playerctl_succeeds(self):
        """Running player: playerctl exits 0 → success reported."""
        fake = type("R", (), {"returncode": 0})()
        with patch("services.music_agent.shutil.which", return_value="/usr/bin/playerctl"), \
             patch("services.music_agent.subprocess.run", return_value=fake):
            result = asyncio.run(self.agent.pause())
        assert result == "Music paused.", f"honest success lost: {result!r}"

    def test_resume_playerctl_succeeds(self):
        fake = type("R", (), {"returncode": 0})()
        with patch("services.music_agent.shutil.which", return_value="/usr/bin/playerctl"), \
             patch("services.music_agent.subprocess.run", return_value=fake):
            result = asyncio.run(self.agent.resume())
        assert result == "Resumed.", f"honest success lost: {result!r}"

    def test_playerctl_exception_is_failure(self):
        with patch("services.music_agent.shutil.which", return_value="/usr/bin/playerctl"), \
             patch("services.music_agent.subprocess.run",
                   side_effect=OSError("boom")):
            assert self.agent._playerctl("pause") is False


class TestB3BrainVerification:
    """Brain._verify must treat media failure strings as failures."""

    def setup_method(self):
        from agent.brain import AgentBrain
        self.brain = AgentBrain()

    def test_failure_markers(self):
        from agent.brain import AgentBrain as AB
        f = AB._is_dispatch_failure
        assert f("Nothing to resume.")
        assert f("Nothing to stop.")
        assert f("Couldn't pause — no player responded.")
        assert f("Skipping tracks isn't available right now.")
        assert f("MPV is not installed — can't play local files.")
        assert f("No local music found matching 'x'.")
        assert not f("Music paused.")
        assert not f("Resumed.")
        assert not f("Playing lofi hip hop coding.")
        assert not f("")

    def test_trust_dispatch_result(self):
        from agent.brain import AgentBrain as AB
        t = AB._trust_dispatch_result
        assert t("Music paused.") is True
        assert t("Resumed.") is True
        assert t("Nothing to resume.") is False
        assert t("Couldn't pause — no player responded.") is False
        assert t(None) is False

    def test_verify_music_resume_no_player(self):
        """music_resume with 'Nothing to resume.' must FAIL verification."""
        ok = asyncio.run(self.brain._verify(
            "music_resume", {}, "Nothing to resume."))
        assert ok is False, "false success: no-player resume verified"

    def test_verify_music_pause_no_player(self):
        ok = asyncio.run(self.brain._verify(
            "music_pause", {}, "Couldn't pause — no player responded."))
        assert ok is False

    def test_verify_music_pause_running_player(self):
        """Running player: honest success string must verify."""
        ok = asyncio.run(self.brain._verify(
            "music_pause", {}, "Music paused."))
        assert ok is True

    def test_verify_play_media_success(self):
        ok = asyncio.run(self.brain._verify(
            "play_media", {"query": "lo-fi"}, "Playing lofi hip hop coding."))
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# B4 — audit harness uses the production dispatch path
# ═══════════════════════════════════════════════════════════════

class TestB4AuditHarness:
    """The audit harness must dispatch via Brain._dispatch_and_verify."""

    def test_harness_uses_production_path(self):
        import inspect
        from debug import audit_execution_pipeline as audit

        src = inspect.getsource(audit.run_single_command)
        assert "_dispatch_and_verify" in src, \
            "harness must call Brain._dispatch_and_verify (production path)"
        # The old shortcut (raw dispatcher + separate verify) must be gone
        assert "brain._dispatcher.execute(action)" not in src, \
            "harness must not bypass retries via raw dispatcher.execute()"
        assert "brain._verify(" not in src, \
            "harness must not call _verify separately (retries bypassed)"

    def test_production_path_includes_retry(self):
        """_dispatch_and_verify must contain the retry loop."""
        import inspect
        from agent.brain import AgentBrain
        src = inspect.getsource(AgentBrain._dispatch_and_verify)
        assert "MAX_ACTION_RETRIES" in src
        assert "_adjust_params_for_retry" in src