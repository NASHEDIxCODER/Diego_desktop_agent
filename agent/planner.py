"""
AgentPlanner — Plans and executes multi-step desktop tasks.

Uses an LLM to decompose user requests into step-by-step plans.
Executes each step via AgentExecutor and tracks progress via AgentMemory.

Architecture:
  User request → LLM plan → Step-by-step execution → Result

Each step is one action (click, type, navigate, etc.).
Failed steps are retried with visual search recovery.
"""

import json
import logging
import shutil
import time
from typing import Optional, Dict, Any, List

from agent.memory import agent_memory
from agent.executor import agent_executor

logger = logging.getLogger(__name__)

# System prompt for the planner LLM
PLANNER_SYSTEM_PROMPT = """You are Leo, a desktop AI agent that controls the user's computer.

Given a user request and the current context, create a step-by-step plan.

Available actions:
- browser_navigate(url) — Navigate to a URL
- browser_click(selector) — Click a CSS selector
- browser_click_text(text) — Click visible text
- browser_type(text, selector?) — Type text, optionally into a selector
- browser_screenshot() — Take a screenshot
- browser_get_url() — Get current URL
- browser_get_text(selector) — Get element text
- browser_list_tabs() — List all tabs
- browser_new_tab(url) — Open new tab
- browser_switch_tab(index) — Switch to tab
- browser_close_tab() — Close current tab
- mouse_move(x, y) — Move mouse
- mouse_click(x?, y?) — Click
- mouse_double_click(x?, y?) — Double click
- keyboard_type(text) — Type text
- keyboard_press(key) — Press a key
- keyboard_hotkey(*keys) — Press key combination
- desktop_open(app) — Open application
- desktop_screenshot() — Take desktop screenshot
- clipboard_copy() — Copy selection
- clipboard_paste(text) — Paste text

Output ONLY a JSON array of steps. Each step has:
{"action": "action_name", "params": {"key": "value"}, "description": "what this does"}

Example:
[
  {"action": "browser_navigate", "params": {"url": "https://linkedin.com"}, "description": "Open LinkedIn"},
  {"action": "browser_click_text", "params": {"text": "Messaging"}, "description": "Open messages"},
  {"action": "browser_type", "params": {"text": "Hello"}, "description": "Type message"}
]
"""


class AgentPlanner:
    """
    Plans and executes multi-step desktop tasks.

    Uses an LLM to decompose requests into steps, then executes
    each step via AgentExecutor with error recovery.
    """

    def __init__(self):
        self._llm_client = None
        self._initialized = False
        self._experience_enabled = False

    def initialize(self) -> bool:
        """Initialize the planner."""
        logger.info("Initializing agent planner...")
        try:
            from ai.llm_client import llm_client as _lc
            self._llm_client = _lc
        except Exception as e:
            logger.warning("LLM client not available: %s", e)
            self._llm_client = None

        # Wire experience DB for self-improving planning
        try:
            from learning.experience_db import experience_db
            self._experience_db = experience_db
            self._experience_enabled = True
            logger.info("[Planner] Experience DB wired — self-improving enabled")
        except Exception as e:
            logger.debug("[Planner] Experience DB unavailable: %s", e)
            self._experience_db = None
            self._experience_enabled = False

        self._initialized = True
        logger.info("Agent planner initialized")
        return True

    def process_request(self, request: str) -> str:
        """
        Legacy method — kept for backward compatibility.
        Generates a plan and returns a status message.

        The Brain is the sole orchestrator. Execution flows through
        Brain → Dispatcher → Verifier → Learning. This method only
        generates the plan; it NEVER executes steps directly.

        Args:
            request: User's request text

        Returns:
            Status message about the plan.
        """
        logger.info("Agent planning request: %s", request)

        plan = self._generate_plan_sync(request)
        if not plan:
            logger.warning("Could not generate plan for: %s", request[:60])
            return "I'm sorry, I couldn't figure out how to do that. Could you be more specific?"

        logger.info("Plan generated with %d steps", len(plan))
        return f"I've planned {len(plan)} steps to accomplish this."

    def generate_plan_only(self, request: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generate a step-by-step plan WITHOUT executing it.

        This is the ONLY planning API used by the Brain. The Brain
        dispatches each step through the ActionDispatcher.

        Args:
            request: User's request text

        Returns:
            List of action step dicts, or None if no plan could be made.
        """
        logger.info("Agent generating plan: %s", request[:80])

        plan = self._generate_plan_sync(request)
        if plan:
            logger.info("Plan generated with %d steps", len(plan))
        return plan

    def _generate_plan_sync(self, request: str) -> Optional[List[Dict[str, Any]]]:
        """
        Synchronous plan generation.

        The LLM client's chat() is async. We run it in a fresh event
        loop via asyncio.run() so this method can be called from a
        thread (run_in_executor) without deadlocking.
        """
        import asyncio
        coro = None
        try:
            coro = self._generate_plan(request)
            return asyncio.run(coro)
        except RuntimeError:
            # Already inside an event loop — close the un-awaited coroutine
            # to suppress "coroutine was never awaited" warnings.
            if coro is not None:
                coro.close()
            logger.warning("[Planner] Cannot run async LLM inside running loop — using fallback")
            return self._fallback_plan(request)
        except Exception as e:
            if coro is not None:
                coro.close()
            logger.warning("[Planner] Async plan generation failed: %s", e)
            return self._fallback_plan(request)

    async def _generate_plan(self, request: str) -> Optional[List[Dict[str, Any]]]:
        """Generate a step-by-step plan using the LLM + experience DB."""
        # ── Check experience DB for previously successful plans ──
        experience_ctx = ""
        if self._experience_enabled and self._experience_db:
            try:
                # Query for best approach
                best = self._experience_db.best_approach(request, top_n=2)
                if best and best[0].get("success"):
                    exp = best[0]
                    experience_ctx = "\nPrevious successful approach:\n"
                    experience_ctx += f"  Steps: {' → '.join(exp.get('plan_steps', [])[:5])}\n"
                    experience_ctx += f"  Latency: {exp.get('latency_ms', 0):.0f}ms\n"
                    experience_ctx += f"  Result: {exp.get('result', '')}\n"

                # Actions to avoid
                avoid = self._experience_db.avoid_actions(request)
                if avoid:
                    experience_ctx += f"\nDo NOT use these actions (they failed before): {', '.join(avoid)}\n"
            except Exception as e:
                logger.debug("[Planner] Experience query failed: %s", e)

        if not self._llm_client:
            return self._fallback_plan(request)

        try:
            # Build context from memory
            context = f"Current URL: {agent_memory.browser_url or 'unknown'}\n"
            context += f"Tabs: {len(agent_memory.browser_tabs)} open\n"
            context += f"Last action: {agent_memory.last_action or 'none'}\n"
            context += experience_ctx

            prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser request: {request}\n\nPlan:"

            # CRITICAL FIX: chat() is async — MUST await it.
            # Previously this returned a coroutine object (always truthy),
            # which then failed in _parse_plan() with AttributeError.
            response = await self._llm_client.chat(prompt)
            if response:
                # Parse JSON from response
                return self._parse_plan(response)
        except Exception as e:
            logger.warning("Plan generation failed: %s", e)

        return self._fallback_plan(request)

    def _parse_plan(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse a JSON plan from LLM response text."""
        # Try to find JSON array in the response
        try:
            # Find first [ and last ]
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                json_str = text[start:end + 1]
                plan = json.loads(json_str)
                if isinstance(plan, list) and len(plan) > 0:
                    return plan
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Plan parsing failed: %s", e)
        return None

    def _fallback_plan(self, request: str) -> Optional[List[Dict[str, Any]]]:
        """Generate a simple fallback plan for common requests."""
        request_lower = request.lower()

        # Open website
        if any(phrase in request_lower for phrase in ["open ", "go to ", "navigate to "]):
            for word in request_lower.split():
                if word.startswith("http"):
                    return [{"action": "browser_navigate", "params": {"url": word}, "description": f"Navigate to {word}"}]
                if "." in word and len(word) > 3:
                    url = f"https://{word}"
                    return [{"action": "browser_navigate", "params": {"url": url}, "description": f"Navigate to {url}"}]

            # Try to find a website name
            for site in ["google", "youtube", "gmail", "linkedin", "github", "reddit"]:
                if site in request_lower:
                    return [{"action": "browser_navigate", "params": {"url": f"https://{site}.com"}, "description": f"Open {site}.com"}]

        # Search
        if any(phrase in request_lower for phrase in ["search for ", "search ", "find "]):
            query = request_lower.replace("search for ", "").replace("search ", "").replace("find ", "")
            return [
                {"action": "browser_navigate", "params": {"url": "https://google.com"}, "description": "Open Google"},
                {"action": "browser_type", "params": {"text": query}, "description": f"Type '{query}'"},
                {"action": "keyboard_press", "params": {"key": "enter"}, "description": "Press Enter"},
            ]

        # Screenshot
        if any(phrase in request_lower for phrase in ["screenshot", "take a picture", "what do you see"]):
            return [
                {"action": "desktop_screenshot", "params": {}, "description": "Take screenshot"},
                {"action": "browser_screenshot", "params": {}, "description": "Take browser screenshot"},
            ]

        # YouTube
        if any(word in request_lower for word in ["youtube", "play ", "video"]):
            return [{"action": "browser_navigate", "params": {"url": "https://youtube.com"}, "description": "Open YouTube"}]

        return None

    def _execute_action(self, action: str, params: Dict[str, Any]) -> tuple:
        """Execute a single action. Returns (success, message)."""
        executor = agent_executor

        # Browser actions
        if action == "browser_navigate":
            return executor.browser_navigate(params.get("url", ""))
        elif action == "browser_click":
            return executor.browser_click(params.get("selector", ""))
        elif action == "browser_click_text":
            return executor.browser_click_text(params.get("text", ""))
        elif action == "browser_type":
            return executor.browser_type(params.get("text", ""), params.get("selector"))
        elif action == "browser_screenshot":
            return executor.browser_screenshot()
        elif action == "browser_get_url":
            return executor.browser_get_url()
        elif action == "browser_get_text":
            return executor.browser_get_text(params.get("selector", ""))
        elif action == "browser_list_tabs":
            return executor.browser_list_tabs()
        elif action == "browser_new_tab":
            return executor.browser_new_tab(params.get("url", "about:blank"))
        elif action == "browser_close_tab":
            return executor.browser_close_tab()
        elif action == "browser_switch_tab":
            return executor.browser_switch_tab(params.get("index", 0))

        # Mouse actions
        elif action == "mouse_move":
            return executor.mouse_move(params.get("x", 0), params.get("y", 0))
        elif action == "mouse_click":
            return executor.mouse_click(params.get("x"), params.get("y"))
        elif action == "mouse_double_click":
            return executor.mouse_double_click(params.get("x"), params.get("y"))
        elif action == "scroll":
            return executor.scroll(params.get("clicks", 0), params.get("x"), params.get("y"))

        # Keyboard actions
        elif action == "keyboard_type":
            return executor.keyboard_type(params.get("text", ""))
        elif action == "keyboard_press":
            return executor.keyboard_press(params.get("key", ""))
        elif action == "keyboard_hotkey":
            return executor.keyboard_hotkey(*params.get("keys", []))

        # Clipboard
        elif action == "clipboard_copy":
            return executor.clipboard_copy()
        elif action == "clipboard_paste":
            return executor.clipboard_paste(params.get("text", ""))

        # Desktop
        elif action == "desktop_open":
            return executor.desktop_open(params.get("app", ""))
        elif action == "desktop_screenshot":
            return executor.desktop_screenshot()
        elif action == "desktop_get_position":
            return executor.desktop_get_position()

        return False, f"Unknown action: {action}"

    def _try_recovery(self, action: str, params: Dict[str, Any], error: str) -> Optional[str]:
        """Try to recover from a failed action using experience + heuristics."""
        logger.info("Attempting recovery for %s...", action)

        # ── Check experience DB for recovery strategies ──
        if self._experience_enabled and self._experience_db:
            try:
                approaches = self._experience_db.best_approach(
                    f"recover {action}", top_n=1)
                if approaches and approaches[0].get("recovery_action"):
                    recovery = approaches[0]["recovery_action"]
                    logger.info("[Planner] Experience-based recovery: %s", recovery)
                    # Try the recovery action
                    # Parse recovery as "action_name:param" format
                    if ":" in recovery:
                        rec_action, rec_param = recovery.split(":", 1)
                        rec_params = {"app": rec_param} if "desktop_open" in rec_action else {}
                        success, msg = self._execute_action(rec_action, rec_params)
                        if success:
                            return f"Recovered via {recovery}: {msg}"
            except Exception as e:
                logger.debug("[Planner] Recovery query failed: %s", e)

        # ── Heuristic recovery ──────────────────────────

        # Firefox failed → try Chrome
        if action == "desktop_open" and params.get("app", "").lower() in ("firefox", "firefox-esr", "firefox-bin"):
            alternatives = ["google-chrome", "chromium", "brave", "chromium-browser"]
            for alt in alternatives:
                if shutil.which(alt):
                    logger.info("[Planner] Firefox failed → trying %s", alt)
                    success, msg = self._execute_action("desktop_open", {"app": alt})
                    if success:
                        return f"Firefox wasn't available, used {alt} instead: {msg}"

        # Chrome failed → try Firefox
        if action == "desktop_open" and params.get("app", "").lower() in ("google-chrome", "chromium", "chrome"):
            if shutil.which("firefox"):
                logger.info("[Planner] Chrome failed → trying Firefox")
                success, msg = self._execute_action("desktop_open", {"app": "firefox"})
                if success:
                    return f"Chrome wasn't available, used Firefox instead: {msg}"

        # Browser navigate failed → try fallback
        if action == "browser_navigate":
            url = params.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                # Try adding https://
                retry_url = "https://" + url
                logger.info("[Planner] Retrying navigate with https: %s", retry_url)
                success, msg = self._execute_action("browser_navigate", {"url": retry_url})
                if success:
                    return msg

        # For browser click failures, try taking a screenshot first
        if "browser_click" in action and "timeout" in error.lower():
            try:
                from agent.browser import browser_controller
                path = browser_controller.screenshot()
                if path:
                    return f"Element not found. Screenshot saved to {path}. Try a different selector."
            except Exception:
                pass

        # For navigation failures, wait and retry
        if action == "browser_navigate":
            time.sleep(2)
            success, msg = self._execute_action(action, params)
            if success:
                return msg

        # For type failures, try clicking the field first
        if action == "browser_type" and params.get("selector"):
            click_success, _ = self._execute_action("browser_click", {"selector": params["selector"]})
            if click_success:
                success, msg = self._execute_action(action, params)
                if success:
                    return msg

        return None

    @property
    def is_available(self) -> bool:
        return self._initialized

    def close(self) -> None:
        self._initialized = False
        logger.info("Agent planner shut down")


# Global singleton
agent_planner = AgentPlanner()