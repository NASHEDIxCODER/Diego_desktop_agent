"""
ActionVerifier — Post-action UI change verification with auto-retry.

After every desktop action (click, type, navigate, etc.), verifies
whether the action actually changed the UI as expected:

  1. Capture screen AFTER the action
  2. Compare against the previous frame (from ScreenMemory)
  3. Check: Did anything change?
     - Frame hash different?
     - Window title changed?
     - OCR text changed?
     - UI element count changed?
     - Expected element appeared/disappeared?
  4. If NOT: auto-retry with fallback candidates
  5. If STILL no change: report failure with explanation

Retry strategies:
  - Click: try alternative UI elements with similar labels
  - Type: verify text appeared in target field
  - Navigate: verify URL changed
  - Open app: verify new window appeared
  - Scroll: verify content offset changed

Logging: [VERIFY]

Usage:
    from vision.action_verifier import action_verifier

    result = await action_verifier.verify_action("click", {"label": "Run"},
                                                  expected_outcome="Button should depress")
    if not result.success:
        # Auto-retry or report failure
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

class VerificationStatus(str, Enum):
    """Outcome of action verification."""
    VERIFIED = "verified"              # Action had expected effect
    NO_CHANGE = "no_change"            # UI didn't change at all
    PARTIAL_CHANGE = "partial_change"  # Something changed, but not expected
    UNEXPECTED = "unexpected"          # Something unexpected happened
    ERROR = "error"                    # Verification itself failed
    TIMEOUT = "timeout"                # UI didn't stabilize in time


@dataclass
class VerificationResult:
    """Result of action verification."""
    success: bool = False
    status: VerificationStatus = VerificationStatus.NO_CHANGE
    explanation: str = ""
    pre_action_hash: str = ""
    post_action_hash: str = ""
    elapsed_ms: float = 0.0

    # What changed
    frame_changed: bool = False
    window_changed: bool = False
    dialog_appeared: bool = False
    dialog_disappeared: bool = False
    text_changed: bool = False
    text_diff: str = ""                # human-readable text diff snippet
    element_count_delta: int = 0       # change in UI element count
    expected_element_found: bool = False

    # Auto-retry
    retry_count: int = 0
    retry_success: bool = False
    retry_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "explanation": self.explanation,
            "frame_changed": self.frame_changed,
            "window_changed": self.window_changed,
            "dialog_appeared": self.dialog_appeared,
            "dialog_disappeared": self.dialog_disappeared,
            "text_changed": self.text_changed,
            "text_diff": self.text_diff[:200],
            "element_count_delta": self.element_count_delta,
            "retry_count": self.retry_count,
            "retry_success": self.retry_success,
            "elapsed_ms": self.elapsed_ms,
        }


# ═══════════════════════════════════════════════════════════════
# ActionVerifier
# ═══════════════════════════════════════════════════════════════

class ActionVerifier:
    """
    Post-action verification with auto-retry.

    Captures the screen after every action, compares it to the
    previous state, and determines whether the action had the
    expected effect. If not, tries alternative candidates.
    """

    # How long to wait for UI to stabilize after an action
    UI_STABILIZE_MS: float = 300.0    # 300ms wait before checking
    MAX_RETRY_COUNT: int = 2          # Max auto-retries before giving up
    RETRY_DELAY_MS: float = 500.0     # Wait between retry attempts

    def __init__(self):
        self._pre_action_snapshot: Optional[Any] = None  # ScreenSnapshot
        self._post_action_snapshot: Optional[Any] = None
        self._retry_count: int = 0
        self._verification_count: int = 0
        self._failure_count: int = 0

    # ── Pre-action capture ───────────────────────────────

    def capture_pre_action(self) -> None:
        """
        Capture the screen state BEFORE an action is performed.

        Must be called BEFORE executing the action.
        """
        try:
            from vision.screen_memory import screen_memory
            self._pre_action_snapshot = screen_memory.latest()
            if self._pre_action_snapshot:
                logger.debug("[VERIFY] Pre-action capture: frame #%d hash=%s window='%s'",
                             self._pre_action_snapshot.frame_id,
                             self._pre_action_snapshot.frame_hash[:8],
                             self._pre_action_snapshot.window_title[:40])
            else:
                logger.debug("[VERIFY] Pre-action capture: NO previous snapshot available")
        except Exception as e:
            logger.warning("[VERIFY] Pre-action capture failed: %s", e)
            self._pre_action_snapshot = None

    # ── Verification ─────────────────────────────────────

    async def verify_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        expected_outcome: str = "",
        expected_element: Optional[str] = None,
        retry_candidates: Optional[List[Dict[str, Any]]] = None,
        retry_executor: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> VerificationResult:
        """
        Verify that an action had the expected effect.

        Workflow:
          1. Wait for UI to stabilize (UI_STABILIZE_MS)
          2. Trigger a fresh vision analysis
          3. Compare pre/post action screen states
          4. If unchanged, auto-retry with candidates
          5. Return verification result

        Args:
            action_type: "click", "type", "navigate", "scroll", "open_app"
            action_params: Parameters of the action
            expected_outcome: Human description of what should happen
            expected_element: Label of element that should appear/disappear
            retry_candidates: Alternative parameters to try on failure
            retry_executor: Async callable(param_dict) -> bool for retry

        Returns:
            VerificationResult with success/failure details.
        """
        t0 = time.perf_counter_ns()
        self._verification_count += 1
        result = VerificationResult()

        # ── Wait for UI to stabilize ───────────────────
        await asyncio.sleep(self.UI_STABILIZE_MS / 1000.0)

        # ── Trigger fresh vision analysis ──────────────
        try:
            from services.vision_service import vision_service
            ctx = await vision_service.analyze(force=True)
            if ctx.error:
                result.status = VerificationStatus.ERROR
                result.explanation = f"Vision analysis failed: {ctx.error}"
                result.elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
                logger.warning("[VERIFY] ERROR: %s", ctx.error)
                return result
        except Exception as e:
            result.status = VerificationStatus.ERROR
            result.explanation = f"Vision service error: {e}"
            result.elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
            logger.warning("[VERIFY] ERROR: vision_service raised %s", e)
            return result

        # ── Get post-action snapshot ───────────────────
        try:
            from vision.screen_memory import screen_memory
            self._post_action_snapshot = screen_memory.latest()
        except Exception as e:
            logger.debug("[VERIFY] Post-action snapshot error: %s", e)
            self._post_action_snapshot = None

        # ── Compare pre vs post ────────────────────────
        if self._pre_action_snapshot and self._post_action_snapshot:
            # Frame hash comparison
            pre_hash = self._pre_action_snapshot.frame_hash
            post_hash = self._post_action_snapshot.frame_hash
            result.pre_action_hash = pre_hash
            result.post_action_hash = post_hash

            if pre_hash != post_hash:
                result.frame_changed = True
                logger.info("[VERIFY] Frame changed: hash %s → %s", pre_hash[:8], post_hash[:8])
            else:
                logger.info("[VERIFY] Frame UNCHANGED: hash %s (pre == post)", pre_hash[:8])

            # Window title comparison
            pre_window = self._pre_action_snapshot.window_title
            post_window = self._post_action_snapshot.window_title
            if pre_window != post_window:
                result.window_changed = True
                logger.info("[VERIFY] Window changed: '%s' → '%s'", pre_window[:40], post_window[:40])

            # OCR text comparison
            pre_text = self._pre_action_snapshot.ocr_text
            post_text = self._post_action_snapshot.ocr_text
            if pre_text != post_text:
                result.text_changed = True
                result.text_diff = self._compute_text_diff(pre_text, post_text)
                logger.info("[VERIFY] Text changed: %d → %d chars", len(pre_text), len(post_text))

            # UI element count comparison
            pre_lines = len(self._pre_action_snapshot.ui_tree_text.split("\n")) if self._pre_action_snapshot.ui_tree_text else 0
            post_lines = len(self._post_action_snapshot.ui_tree_text.split("\n")) if self._post_action_snapshot.ui_tree_text else 0
            result.element_count_delta = post_lines - pre_lines
            if result.element_count_delta != 0:
                logger.info("[VERIFY] UI elements: %d → %d (delta=%+d)", pre_lines, post_lines, result.element_count_delta)

            # Check for dialog appearance
            pre_has_dialog = "dialog" in self._pre_action_snapshot.ui_tree_text.lower() if self._pre_action_snapshot.ui_tree_text else False
            post_has_dialog = "dialog" in self._post_action_snapshot.ui_tree_text.lower() if self._post_action_snapshot.ui_tree_text else False
            if not pre_has_dialog and post_has_dialog:
                result.dialog_appeared = True
                logger.info("[VERIFY] Dialog appeared")
            elif pre_has_dialog and not post_has_dialog:
                result.dialog_disappeared = True
                logger.info("[VERIFY] Dialog disappeared")

            # Expected element check
            if expected_element:
                pre_text_lower = pre_text.lower()
                post_text_lower = post_text.lower()
                expected_lower = expected_element.lower()
                was_present_before = expected_lower in pre_text_lower
                is_present_after = expected_lower in post_text_lower
                if not was_present_before and is_present_after:
                    result.expected_element_found = True
                    logger.info("[VERIFY] Expected element '%s' appeared", expected_element)
                elif was_present_before and not is_present_after:
                    # Element disappearing might also be the expected outcome
                    result.expected_element_found = True
                    logger.info("[VERIFY] Expected element '%s' disappeared (as expected)", expected_element)

        # ── Determine overall success ──────────────────
        if self._determine_success(result, action_type):
            result.success = True
            result.status = VerificationStatus.VERIFIED
            result.explanation = self._build_explanation(result, action_type, expected_outcome)
            result.elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
            logger.info("[VERIFY] VERIFIED: %s — %s (%.1fms)", action_type, result.explanation, result.elapsed_ms)
            return result

        # ── Partial change? ────────────────────────────
        if result.frame_changed or result.text_changed or result.window_changed:
            result.status = VerificationStatus.PARTIAL_CHANGE
            result.explanation = "UI changed but not as expected"
            result.elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
            logger.info("[VERIFY] PARTIAL_CHANGE: %s changed but not verified", action_type)
            result.success = True  # Something happened, count as partial success
            return result

        # ── NO CHANGE: Auto-retry ──────────────────────
        if retry_candidates and retry_executor and self._retry_count < self.MAX_RETRY_COUNT:
            result = await self._auto_retry(
                result, action_type, retry_candidates, retry_executor,
                expected_outcome, expected_element,
            )

        if not result.success:
            result.status = VerificationStatus.NO_CHANGE
            result.explanation = f"UI did not change after {action_type}. " + (
                "Retry also failed." if result.retry_count > 0 else "No retry candidates available.")
            self._failure_count += 1
            logger.warning("[VERIFY] NO_CHANGE: %s had no effect (retries=%d)", action_type, result.retry_count)

        result.elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return result

    def _determine_success(self, result: VerificationResult, action_type: str) -> bool:
        """Determine if the action was successful based on what changed."""
        if result.expected_element_found:
            return True

        if action_type == "click":
            # Click should cause SOME change (frame, window, dialog, or text)
            return result.frame_changed or result.window_changed or result.dialog_appeared or result.dialog_disappeared or result.text_changed

        if action_type == "type":
            # Typing should change text
            return result.text_changed

        if action_type == "navigate":
            # Navigation should change the frame
            return result.frame_changed or result.window_changed

        if action_type == "scroll":
            # Scrolling should change the frame
            return result.frame_changed

        if action_type == "open_app":
            # Opening an app should change the window
            return result.window_changed or result.frame_changed

        if action_type == "key_press":
            # Key presses may or may not change the frame
            return result.frame_changed or result.text_changed

        # Default: any change counts
        return result.frame_changed or result.text_changed or result.window_changed

    async def _auto_retry(
        self,
        result: VerificationResult,
        action_type: str,
        candidates: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Any],
        expected_outcome: str,
        expected_element: Optional[str],
    ) -> VerificationResult:
        """Auto-retry with alternative candidates."""
        for i, candidate in enumerate(candidates):
            self._retry_count += 1
            result.retry_count = self._retry_count
            result.retry_action = str(candidate.get("label", candidate.get("params", candidate)))[:60]

            logger.info("[VERIFY] Auto-retry #%d/%d: %s",
                        self._retry_count, self.MAX_RETRY_COUNT, result.retry_action)

            # Wait before retry
            await asyncio.sleep(self.RETRY_DELAY_MS / 1000.0)

            # Capture pre-retry state
            self.capture_pre_action()

            # Execute retry
            try:
                await executor(candidate)
            except Exception as e:
                logger.warning("[VERIFY] Retry execution failed: %s", e)
                continue

            # Verify retry
            retry_result = await self.verify_action(
                action_type, candidate,
                expected_outcome=expected_outcome,
                expected_element=expected_element,
                retry_candidates=None,  # No recursive retry
            )

            if retry_result.success:
                result.success = True
                result.retry_success = True
                result.status = VerificationStatus.VERIFIED
                result.explanation = f"Retry #{self._retry_count} succeeded: {retry_result.explanation}"
                result.frame_changed = retry_result.frame_changed
                result.text_changed = retry_result.text_changed
                logger.info("[VERIFY] RETRY SUCCESS: %s worked on attempt #%d",
                            result.retry_action, self._retry_count)
                return result

            logger.info("[VERIFY] Retry #%d also failed: %s",
                        self._retry_count, retry_result.explanation)

        result.retry_success = False
        logger.warning("[VERIFY] All %d retries failed", self._retry_count)
        return result

    def _build_explanation(
        self, result: VerificationResult, action_type: str, expected: str
    ) -> str:
        """Build a human-readable explanation of what happened."""
        parts: List[str] = []

        if expected:
            parts.append(expected)

        if result.frame_changed:
            parts.append("screen content updated")
        if result.window_changed:
            parts.append("active window changed")
        if result.dialog_appeared:
            parts.append("dialog appeared")
        if result.dialog_disappeared:
            parts.append("dialog closed")
        if result.text_changed:
            parts.append("visible text updated")
        if result.element_count_delta != 0:
            parts.append(f"UI elements {'increased' if result.element_count_delta > 0 else 'decreased'} by {abs(result.element_count_delta)}")

        if result.retry_success:
            parts.append(f"(after {result.retry_count} retry attempts)")

        return "; ".join(parts) if parts else f"No visible change after {action_type}"

    @staticmethod
    def _compute_text_diff(pre_text: str, post_text: str) -> str:
        """Compute a simple human-readable text diff snippet."""
        if not pre_text and not post_text:
            return ""
        if not pre_text:
            return f"Added: '{post_text[:100]}'"
        if not post_text:
            return f"Removed: '{pre_text[:100]}'"

        # Find common prefix and suffix
        min_len = min(len(pre_text), len(post_text))
        i = 0
        while i < min_len and pre_text[i] == post_text[i]:
            i += 1
        prefix = pre_text[:i]

        j_pre = len(pre_text) - 1
        j_post = len(post_text) - 1
        while j_pre > i and j_post > i and pre_text[j_pre] == post_text[j_post]:
            j_pre -= 1
            j_post -= 1

        removed = pre_text[i:j_pre + 1] if j_pre >= i else ""
        added = post_text[i:j_post + 1] if j_post >= i else ""

        parts = []
        if removed:
            parts.append(f"-'{removed[:50]}'")
        if added:
            parts.append(f"+'{added[:50]}'")
        return " ".join(parts)

    # ── Convenience verify functions ────────────────────

    async def verify_click(self, label: str, expected: str = "",
                            retry_click: Optional[Callable[[str], Any]] = None) -> VerificationResult:
        """Verify a click action."""
        return await self.verify_action(
            "click", {"label": label},
            expected_outcome=expected or f"Clicked '{label}'",
            expected_element=label,
            retry_candidates=[{"label": label}] if retry_click else None,
            retry_executor=lambda p: retry_click(p["label"]) if retry_click else None,
        )

    async def verify_type(self, text: str, expected: str = "") -> VerificationResult:
        """Verify a type action."""
        return await self.verify_action(
            "type", {"text": text},
            expected_outcome=expected or f"Typed '{text}'",
        )

    async def verify_navigate(self, url: str, expected: str = "") -> VerificationResult:
        """Verify a navigation action."""
        return await self.verify_action(
            "navigate", {"url": url},
            expected_outcome=expected or f"Navigated to {url}",
        )

    # ── Diagnostics ──────────────────────────────────────

    def reset_retry_count(self) -> None:
        self._retry_count = 0

    def report(self) -> Dict[str, Any]:
        """Return diagnostic summary."""
        return {
            "verification_count": self._verification_count,
            "failure_count": self._failure_count,
            "failure_rate": f"{self._failure_count / max(self._verification_count, 1):.1%}",
            "retry_count": self._retry_count,
            "has_pre_snapshot": self._pre_action_snapshot is not None,
        }


# Global singleton
action_verifier = ActionVerifier()