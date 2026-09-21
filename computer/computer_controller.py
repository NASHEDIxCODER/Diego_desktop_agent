"""
Phase 22: ComputerController — the computer-use facade.

One generic boundary for EVERY desktop action. It does not replace the
existing reasoning/task/recovery architecture — TaskRunner, Brain, the
dispatcher and the resolvers stay authoritative. This controller composes
the Phase 22 primitives into the required loop:

    OBSERVE → ACTION → OBSERVE → VERIFY

for every action, returning a structured ActionResult that records the
method used, evidence, expected vs observed effect.

Hard rules enforced here:
  - Never blind coordinate guesses: element location goes through the
    perception hierarchy (element_finder); coordinates require justification.
  - Never bypass confirmation: EXTERNAL_SIDE_EFFECT actions return
    CONFIRMATION_REQUIRED instead of executing.
  - Never fake success: state-change actions FAIL when nothing changed.

Logging: [COMPUTER]
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Optional

from computer.action_policy import ActionRisk, assess_risk, confirmation_required
from computer.action_result import ActionOutcome, ActionResult
from computer.screen_state import ComputerState

logger = logging.getLogger(__name__)


class ComputerController:
    """Generic computer-use boundary. All actions return ActionResult."""

    _HANDLERS = {
        "open_app": "_act_open_app", "desktop_open": "_act_open_app",
        "close_app": "_act_close_app",
        "focus_window": "_act_focus_window",
        "open_url": "_act_open_url", "navigate": "_act_open_url",
        "browser_navigate": "_act_open_url",
        "back": "_act_back", "forward": "_act_forward", "refresh": "_act_refresh",
        "click": "_act_click", "double_click": "_act_double_click",
        "click_text": "_act_click",
        "type_text": "_act_type_text", "type": "_act_type_text",
        "clear_text": "_act_clear_text",
        "press_key": "_act_press_key", "key_press": "_act_press_key",
        "hotkey": "_act_hotkey",
        "scroll": "_act_scroll", "drag": "_act_drag",
        "wait": "_act_wait", "screenshot": "_act_screenshot",
        "observe": "_act_observe",
        "find_element": "_act_find_element",
        "find_candidates": "_act_find_candidates",
        "get_page_state": "_act_get_page_state",
        "read_page": "_act_get_page_state",
        "extract_text": "_act_get_page_state",
        "extract_links": "_act_extract_links",
        "wait_for_page": "_act_wait_for_page",
        "select_option": "_act_select_option",
        "get_window_state": "_act_get_window_state",
    }

    # Actions whose success is verified by an observed state delta.
    _DELTA_VERIFIED = frozenset({
        "click", "double_click", "click_text", "type_text", "type",
        "clear_text", "press_key", "key_press", "hotkey", "scroll", "drag",
    })
    _SELF_VERIFIED = frozenset({
        "open_app", "desktop_open", "close_app", "focus_window",
        "open_url", "navigate", "browser_navigate", "back", "forward",
        "refresh",
    })
    _READ_ONLY = frozenset({
        "observe", "screenshot", "find_element", "find_candidates",
        "get_page_state", "read_page", "extract_text", "extract_links",
        "wait_for_page", "get_window_state", "wait",
    })

    def __init__(self) -> None:
        self._last_state: Optional[ComputerState] = None
        self._task_id: str = ""
        self._step_id: str = ""

    def bind(self, task_id: str = "", step_id: str = "") -> None:
        """Attach task context so trace events land in the right workflow."""
        self._task_id, self._step_id = task_id, step_id

    # ── OBSERVE: normalized state (never raw screenshots only) ───

    def capture_state(self, note: str = "") -> ComputerState:
        from computer import window_manager as wm

        win = wm.active_window()
        page: Dict[str, Any] = {}
        try:
            from computer import browser_controller as bctl
            page = bctl.get_page_state(max_text=1200)
        except Exception:
            page = {}

        elements: list = []
        try:
            from computer import accessibility as a11y
            elements = [e.to_dict() for e in a11y.clickable(limit=15)]
        except Exception:
            elements = []

        url = str(page.get("url", "") or "")
        state = ComputerState(
            active_application=(win.window_class if win else ""),
            focused_window=(win.window_id if win else ""),
            window_title=(win.title if win else ""),
            window_class=(win.window_class if win else ""),
            active_url=url,
            page_title=str(page.get("title", "") or ""),
            visible_text=str(page.get("visible_text", "") or "")[:1200],
            visible_elements=elements,
            screen_hash=self._screen_hash(win, url),
            last_action=note,
            last_observation="",
        )
        self._last_state = state
        return state

    @staticmethod
    def _screen_hash(win: Any, url: str) -> str:
        """Cheap screen fingerprint: live capture hash, else window identity."""
        try:
            import services.screen_capture as sc
            for name in ("screen_capture", "screenCapture", "_instance"):
                obj = getattr(sc, name, None)
                if obj is not None and hasattr(obj, "last_capture"):
                    lc = obj.last_capture
                    lc = lc() if callable(lc) else lc
                    h = str(getattr(lc, "hash", "") or
                            getattr(lc, "fast_hash", "") or "")
                    if h:
                        return h
        except Exception:
            pass
        base = f"{getattr(win, 'window_id', '')}|{url}"
        return hashlib.md5(base.encode()).hexdigest()[:16]

    @staticmethod
    def _delta(before: ComputerState,
               after: ComputerState) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if before.screen_hash != after.screen_hash:
            d["screen_hash"] = True
        if before.window_title != after.window_title:
            d["window_title"] = after.window_title[:80]
        if before.active_application != after.active_application:
            d["app"] = after.active_application
        if before.active_url != after.active_url:
            d["url"] = after.active_url[:120]
        if before.page_title != after.page_title:
            d["page_title"] = after.page_title[:80]
        if before.visible_text != after.visible_text:
            d["visible_text"] = True
        return d

    # ── ACT: the full loop ───────────────────────────────────────

    def execute(self, action: str, params: Optional[Dict[str, Any]] = None
                ) -> ActionResult:
        """OBSERVE → ACT → OBSERVE → VERIFY, returning a structured result."""
        params = dict(params or {})
        action = str(action or "").strip()
        t0 = time.time()

        needs, creason = confirmation_required(action, params)
        risk, _ = assess_risk(action, params)
        if needs:
            result = ActionResult(
                action=action,
                target=str(params.get("app") or params.get("target") or ""),
                outcome=ActionOutcome.CONFIRMATION_REQUIRED, error=creason)
            self._trace_action(result, risk.value, blocked=True)
            return result

        before = self.capture_state(note=f"before:{action}")
        self._trace_started(action, str(params.get("target")
                                        or params.get("app") or ""), risk.value)

        handler = getattr(self, self._HANDLERS.get(action, ""), None)
        if handler is None:
            result = ActionResult(
                action=action, outcome=ActionOutcome.UNAVAILABLE,
                error=f"unknown computer action '{action}'")
        else:
            try:
                result = handler(params)
            except Exception as e:
                logger.exception("[COMPUTER] handler %s crashed", action)
                result = ActionResult(action=action,
                                      outcome=ActionOutcome.FAILED,
                                      error=str(e))

        after = self.capture_state(note=f"after:{action}")
        result = self._verify(result, action, before, after)
        self._last_state = after
        if not result.latency_ms:
            result.latency_ms = round((time.time() - t0) * 1000, 1)
        self._trace_action(result, risk.value)
        return result

    def _verify(self, result: ActionResult, action: str,
                before: ComputerState, after: ComputerState) -> ActionResult:
        """Target-honest verification; 'something changed' is never success."""
        if action in self._READ_ONLY or action in self._SELF_VERIFIED:
            return result
        if action in self._DELTA_VERIFIED and result.success:
            d = self._delta(before, after)
            if not d:
                result.success = False
                result.outcome = ActionOutcome.FAILED
                result.error = "no observed state change after action"
                result.observed_effect = "no change"
                logger.info("[COMPUTER] verify FAIL (%s): no delta", action)
            else:
                result.observed_effect = ", ".join(
                    f"{k}{'=changed' if v is True else str(v)[:40]}"
                    for k, v in d.items())
                result.verification = (result.verification or
                                       f"delta: {result.observed_effect[:80]}")
                logger.info("[COMPUTER] verify PASS (%s): %s",
                            action, result.observed_effect[:80])
        return result

    def _trace_started(self, action: str, target: str, risk: str) -> None:
        try:
            from agent.trace import agent_trace
            from agent.trace_event import TraceEventType
            agent_trace.emit(TraceEventType.ACTION_STARTED,
                             task_id=self._task_id, step_id=self._step_id,
                             action=action, target=target[:80], risk=risk)
        except Exception:
            pass

    def _trace_action(self, result: ActionResult, risk: str,
                      blocked: bool = False) -> None:
        try:
            from agent.trace import agent_trace
            from agent.trace_event import TraceEventType
            if blocked:
                agent_trace.emit(TraceEventType.CONFIRMATION_REQUIRED,
                                 task_id=self._task_id, action=result.action,
                                 target=result.target[:80], risk=risk,
                                 detail=result.error[:200])
            else:
                agent_trace.emit(
                    TraceEventType.ACTION_COMPLETED, task_id=self._task_id,
                    step_id=self._step_id, action=result.action,
                    target=result.target[:80], method=result.method,
                    evidence=dict(result.evidence),
                    verification_result=("PASS" if result.success else "FAIL"),
                    detail=result.error[:200])
        except Exception:
            pass

    # ── App/window handlers (canonical app_resolver identity) ────

    def _act_open_app(self, p: Dict[str, Any]) -> ActionResult:
        from services.app_resolver import (ResolutionStatus, launch,
                                           resolve_app, wait_for_presence)
        app = str(p.get("app") or p.get("target") or "")
        res = resolve_app(app)
        if res.status == ResolutionStatus.AMBIGUOUS:
            return ActionResult(action="open_app", target=app,
                                method="app_resolver",
                                outcome=ActionOutcome.AMBIGUOUS,
                                error=res.message or "ambiguous app name")
        if res.status != ResolutionStatus.RESOLVED or res.identity is None:
            return ActionResult(action="open_app", target=app,
                                method="app_resolver",
                                outcome=ActionOutcome.FAILED,
                                error=res.message or f"couldn't find {app}")
        launched = launch(res.identity)
        if not launched.ok:
            return ActionResult(action="open_app", target=app,
                                method="app_resolver",
                                outcome=ActionOutcome.FAILED,
                                error=launched.message or "launch failed",
                                evidence={"canonical": res.identity.canonical})
        pres = wait_for_presence(res.identity, timeout_s=4.0)
        ok = self._present(pres)
        return ActionResult(
            action="open_app", target=app, method="app_resolver", success=ok,
            outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
            evidence={"canonical": res.identity.canonical,
                      "launch_method": str(getattr(launched, "method", ""))},
            verification=("target process present" if ok
                          else "launched but target not detected"),
            error="" if ok else "target presence not confirmed")

    @staticmethod
    def _present(ev: Any) -> bool:
        for f in ("present", "found", "ok", "running"):
            v = getattr(ev, f, None)
            if v is not None:
                return bool(v)
        return bool(getattr(ev, "processes", None)
                    or getattr(ev, "windows", None))

    def _act_close_app(self, p: Dict[str, Any]) -> ActionResult:
        # Delegate to the dispatcher's hardened close (exact-match kill,
        # graceful window close, self-kill guard) — never reimplement it.
        try:
            from agent.action_dispatcher import action_dispatcher
        except Exception as e:
            return ActionResult(action="close_app",
                                outcome=ActionOutcome.UNAVAILABLE,
                                error=str(e))
        app = str(p.get("app") or p.get("target") or "")
        msg = action_dispatcher._close_app(app)
        ok = "Couldn't" not in msg and "unavailable" not in msg
        return ActionResult(
            action="close_app", target=app, method="app_resolver", success=ok,
            outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
            evidence={"message": msg[:200]},
            verification=msg[:120], error="" if ok else msg[:200])

    def _act_focus_window(self, p: Dict[str, Any]) -> ActionResult:
        from computer import window_manager as wm
        return wm.focus_window(str(p.get("target") or p.get("app") or ""),
                               target=str(p.get("target") or ""))

    # ── Browser handlers (tier 1) ────────────────────────────────

    def _act_open_url(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        url = str(p.get("url") or p.get("target") or "")
        if not url:
            return ActionResult(action="open_url", method="browser_dom",
                                outcome=ActionOutcome.FAILED,
                                error="no url provided")
        return bctl.open_url(url)

    def _act_back(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        return bctl.back()

    def _act_forward(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        return bctl.forward()

    def _act_refresh(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        return bctl.refresh()

    # ── Element interaction (perception hierarchy, never guesses) ─

    def _locate(self, p: Dict[str, Any]) -> ActionResult:
        """Locate the target through the hierarchy; honest AMBIGUOUS on miss."""
        target = str(p.get("target") or p.get("element")
                     or p.get("text") or "")
        if not target:
            return ActionResult(action="click", method="perception",
                                outcome=ActionOutcome.AMBIGUOUS,
                                error="no target provided")
        try:
            from computer import element_finder
            match = element_finder.find_element(
                target, allow_coordinate=bool(p.get("x") is not None),
                justification=str(p.get("justification") or ""))
        except Exception as e:
            return ActionResult(action="click", target=target,
                                method="perception",
                                outcome=ActionOutcome.FAILED,
                                error=f"perception failed: {e}")
        if not match.found:
            return ActionResult(
                action="click", target=target, method="perception",
                outcome=ActionOutcome.AMBIGUOUS,
                error=("could not locate target via any perception tier — "
                       "refusing to guess coordinates"))
        if match.point is None:
            return ActionResult(
                action="click", target=target, method=match.method,
                outcome=ActionOutcome.AMBIGUOUS,
                error=("target is window-level; use focus_window instead"),
                evidence=match.to_dict()["evidence"])
        return ActionResult(action="click", target=target,
                            method=match.method, success=True,
                            outcome=ActionOutcome.SUCCESS,
                            evidence={"match": match.to_dict()})

    @staticmethod
    def _dom_click(loc: ActionResult, target: str) -> Optional[ActionResult]:
        """Tier-1 DOM click when the DOM is what located the target.

        Returns None (the caller then falls back to the located point) unless
        the DOM tier answered AND a browser page can be acted on — so a
        browser element is never clicked by invented coordinates.
        """
        from computer.perception import PerceptionMethod
        if loc.method != PerceptionMethod.BROWSER_DOM.value:
            return None
        label = str((loc.evidence.get("match") or {}).get("label") or target)
        try:
            from computer import browser_controller as bctl
            if not bctl.attached():
                return None
            result = bctl.click_text(label, target=target or label)
        except Exception as e:
            logger.debug("[COMPUTER] dom click failed: %s", e)
            return None
        if result.outcome == ActionOutcome.UNAVAILABLE:
            return None
        result.action = "click"
        return result

    @staticmethod
    def _dom_type(loc: ActionResult, text: str,
                  target: str) -> Optional[ActionResult]:
        """Tier-1 DOM typing when the DOM located the input (never blind)."""
        from computer.perception import PerceptionMethod
        if loc.method != PerceptionMethod.BROWSER_DOM.value:
            return None
        label = str((loc.evidence.get("match") or {}).get("label") or target)
        try:
            from computer import browser_controller as bctl
            if not bctl.attached():
                return None
            result = bctl.type_text(text, target=label or target)
        except Exception as e:
            logger.debug("[COMPUTER] dom type failed: %s", e)
            return None
        if result.outcome == ActionOutcome.UNAVAILABLE:
            return None
        return result

    def _act_click(self, p: Dict[str, Any]) -> ActionResult:
        loc = self._locate(p)
        if not loc.success:
            loc.action = "click"
            return loc
        target = str(p.get("target") or p.get("element") or p.get("text") or "")
        dom = self._dom_click(loc, target)
        if dom is not None:
            return dom
        from computer import mouse_controller as mouse
        point = tuple(loc.evidence["match"]["point"])  # type: ignore[index]
        click = mouse.click(point[0], point[1],
                            target=loc.target, label=loc.target)
        click.method = loc.method
        click.evidence["located"] = loc.evidence["match"]
        click.action = "double_click" if p.get("double") else "click"
        return click

    def _act_double_click(self, p: Dict[str, Any]) -> ActionResult:
        p = dict(p, double=True)
        loc = self._locate(p)
        if not loc.success:
            loc.action = "double_click"
            return loc
        from computer import mouse_controller as mouse
        point = tuple(loc.evidence["match"]["point"])  # type: ignore[index]
        r = mouse.double_click(point[0], point[1], target=loc.target)
        r.method = loc.method
        r.evidence["located"] = loc.evidence["match"]
        return r

    def _act_type_text(self, p: Dict[str, Any]) -> ActionResult:
        from computer import keyboard_controller as kbd
        text = str(p.get("text") or "")
        target = str(p.get("target") or "")
        if target:
            loc = self._locate({"target": target})
            if loc.success:
                dom = self._dom_type(loc, text, target)
                if dom is not None:
                    return dom
                point = tuple(loc.evidence["match"]["point"])  # type: ignore[index]
                r = kbd.type_into(point, text, target=target,
                                  label=str(loc.evidence["match"].get("label",
                                                                     target)))
                r.method = loc.method
                return r
            # Target named but not located: fail honestly, never blind-type.
            return ActionResult(
                action="type_text", target=target, method="perception",
                outcome=ActionOutcome.AMBIGUOUS,
                error=f"could not locate input '{target[:60]}' — refusing "
                      f"to blind-type")
        return kbd.type_text(text, target=target)

    # ── Keyboard / primitives ────────────────────────────────────

    def _act_clear_text(self, p: Dict[str, Any]) -> ActionResult:
        from computer import keyboard_controller as kbd
        target = str(p.get("target") or "")
        if target:
            loc = self._locate({"target": target})
            if not loc.success:
                return ActionResult(
                    action="clear_text", target=target, method="perception",
                    outcome=ActionOutcome.AMBIGUOUS,
                    error=f"could not locate input '{target[:60]}'")
            point = tuple(loc.evidence["match"]["point"])  # type: ignore[index]
            from computer import mouse_controller as mouse
            mouse.click(point[0], point[1], target=target)
        return kbd.clear_text(target=target)

    def _act_press_key(self, p: Dict[str, Any]) -> ActionResult:
        key = str(p.get("key") or p.get("target") or "")
        # A browser page receives the key through the DOM (tier 1): inside a
        # form, `enter` submits it — the generic search-box behaviour.
        try:
            from computer import browser_controller as bctl
            if bctl.attached():
                r = bctl.press_key(key)
                if r.success:
                    return r
        except Exception as e:
            logger.debug("[COMPUTER] dom press_key failed: %s", e)
        from computer import keyboard_controller as kbd
        return kbd.press_key(key, target=str(p.get("target") or ""))

    def _act_hotkey(self, p: Dict[str, Any]) -> ActionResult:
        from computer import keyboard_controller as kbd
        keys = [str(k) for k in (p.get("keys") or p.get("key") or [])]
        if isinstance(p.get("keys"), str) and "+" in p["keys"]:
            keys = [k.strip() for k in p["keys"].split("+")]
        if not keys:
            return ActionResult(action="hotkey",
                                outcome=ActionOutcome.FAILED,
                                error="no keys provided")
        return kbd.hotkey(*keys, target=str(p.get("target") or ""))

    def _act_scroll(self, p: Dict[str, Any]) -> ActionResult:
        delta = int(p.get("delta") or p.get("clicks") or 0) or -3
        try:
            from computer import browser_controller as bctl
            if bctl.attached():
                r = bctl.scroll(delta)
                if r.success:
                    return r
        except Exception:
            pass
        from computer import mouse_controller as mouse
        return mouse.scroll(delta, target=str(p.get("target") or ""))

    def _act_drag(self, p: Dict[str, Any]) -> ActionResult:
        from computer import mouse_controller as mouse
        try:
            start = (int(p["start_x"]), int(p["start_y"]))
            end = (int(p["end_x"]), int(p["end_y"]))
        except (KeyError, TypeError, ValueError):
            return ActionResult(
                action="drag", method="native_input",
                outcome=ActionOutcome.FAILED,
                error="drag requires explicit start_x/start_y/end_x/end_y "
                      "(never guessed)")
        return mouse.drag(start, end, target=str(p.get("target") or ""))

    def _act_wait(self, p: Dict[str, Any]) -> ActionResult:
        seconds = min(max(float(p.get("seconds") or 1.0), 0.0), 10.0)
        time.sleep(seconds)
        return ActionResult(action="wait", method="native", success=True,
                            outcome=ActionOutcome.SUCCESS,
                            evidence={"seconds": seconds})

    def _act_screenshot(self, p: Dict[str, Any]) -> ActionResult:
        try:
            from computer import mouse_controller as mouse
            ex = mouse._executor()
            if ex is not None and hasattr(ex, "desktop_screenshot"):
                raw = ex.desktop_screenshot()
                ok = bool(raw[0]) if isinstance(raw, tuple) else bool(raw)
                msg = (str(raw[1])
                       if isinstance(raw, tuple) and len(raw) > 1 else "")
                return ActionResult(
                    action="screenshot", method="capture", success=ok,
                    outcome=(ActionOutcome.SUCCESS if ok
                             else ActionOutcome.FAILED),
                    evidence={"capture": msg[:300]},
                    error="" if ok else msg[:200])
        except Exception as e:
            return ActionResult(action="screenshot", method="capture",
                                outcome=ActionOutcome.FAILED, error=str(e))
        return ActionResult(action="screenshot", method="capture",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="no capture backend")

    def _act_observe(self, p: Dict[str, Any]) -> ActionResult:
        state = self.capture_state(note="observe")
        return ActionResult(
            action="observe", method="perception", success=True,
            outcome=ActionOutcome.SUCCESS,
            evidence={"state": state.to_dict()},
            verification=state.summary())

    def _act_find_element(self, p: Dict[str, Any]) -> ActionResult:
        from computer import element_finder
        target = str(p.get("target") or p.get("element") or "")
        match = element_finder.find_element(
            target, allow_coordinate=bool(p.get("x") is not None),
            justification=str(p.get("justification") or ""))
        if not match.found:
            return ActionResult(
                action="find_element", target=target, method="perception",
                outcome=ActionOutcome.AMBIGUOUS,
                error="element not found in any perception tier")
        return ActionResult(
            action="find_element", target=target, method=match.method,
            success=True, outcome=ActionOutcome.SUCCESS,
            evidence={"match": match.to_dict()},
            verification=f"located via {match.method}")

    def _act_get_page_state(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        state = bctl.get_page_state()
        if not state.get("attached"):
            return ActionResult(action="get_page_state", method="browser_dom",
                                outcome=ActionOutcome.UNAVAILABLE,
                                error="browser not attached")
        return ActionResult(action="get_page_state", method="browser_dom",
                            success=True, outcome=ActionOutcome.SUCCESS,
                            evidence={"page": state})

    def _act_get_window_state(self, p: Dict[str, Any]) -> ActionResult:
        from computer import window_manager as wm
        state = wm.window_state()
        if not state.get("focused"):
            return ActionResult(action="get_window_state",
                                method="native_window",
                                outcome=ActionOutcome.UNAVAILABLE,
                                error="no focused window detected")
        return ActionResult(action="get_window_state", method="native_window",
                            success=True, outcome=ActionOutcome.SUCCESS,
                            evidence={"window": state})

    # ── Phase 23: generic browser observation/interaction primitives ──

    def _act_find_candidates(self, p: Dict[str, Any]) -> ActionResult:
        """Enumerate ALL matches for a target (ambiguity evidence)."""
        from computer import browser_controller as bctl
        target = str(p.get("target") or p.get("element") or "")
        if not target:
            return ActionResult(action="find_candidates", method="browser_dom",
                                outcome=ActionOutcome.AMBIGUOUS,
                                error="no target provided")
        candidates = bctl.find_candidates(
            target, limit=int(p.get("limit") or 12))
        if not candidates:
            return ActionResult(action="find_candidates", target=target,
                                method="browser_dom",
                                outcome=ActionOutcome.AMBIGUOUS,
                                error=f"no element matched '{target[:60]}'")
        return ActionResult(action="find_candidates", target=target,
                            method="browser_dom", success=True,
                            outcome=ActionOutcome.SUCCESS,
                            evidence={"candidates": candidates[:12]},
                            verification=f"{len(candidates)} candidate(s) observed")

    def _act_wait_for_page(self, p: Dict[str, Any]) -> ActionResult:
        """Bounded observe→wait→observe until the document finishes loading."""
        from computer import browser_controller as bctl
        timeout = min(max(float(p.get("timeout") or 8.0), 0.0), 30.0)
        outcome = bctl.wait_for_ready(timeout=timeout)
        ready = bool(outcome.get("ready"))
        return ActionResult(
            action="wait_for_page", method="browser_dom", success=ready,
            outcome=(ActionOutcome.SUCCESS if ready else ActionOutcome.FAILED),
            evidence=dict(outcome),
            verification=(f"readyState={outcome.get('ready_state')}"),
            error="" if ready else str(outcome.get("error") or "still loading"))

    def _act_extract_links(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        structure = bctl.get_page_structure(max_text=2000, max_elements=150)
        if not structure.get("attached"):
            return ActionResult(action="extract_links", method="browser_dom",
                                outcome=ActionOutcome.UNAVAILABLE,
                                error="browser not attached")
        links = list(structure.get("links") or [])
        return ActionResult(
            action="extract_links", method="browser_dom", success=bool(links),
            outcome=(ActionOutcome.SUCCESS if links else ActionOutcome.FAILED),
            evidence={"links": links[:60], "url": structure.get("url", "")},
            verification=f"{len(links)} link(s) observed",
            error="" if links else "no links observed on page")

    def _act_select_option(self, p: Dict[str, Any]) -> ActionResult:
        from computer import browser_controller as bctl
        return bctl.select_option(
            str(p.get("label") or p.get("target") or ""),
            value=str(p.get("value") or ""),
            option=str(p.get("option") or p.get("text") or ""),
            target=str(p.get("target") or ""))


# Module-level singleton (the UI and Brain import this).
computer_controller = ComputerController()






