"""
BrowserController — drives the USER'S EXISTING Chrome session.

Attaches over Chrome DevTools Protocol (CDP) via Playwright's
connect_over_cdp to the Chrome the user is ALREADY running, with its
existing profile, cookies, logged-in accounts, tabs and session state:

    existing Chrome session
        ↓ attach / reuse the current browser (never a second instance)
    existing profile + cookies + login state
        ↓
    BrowserGoalEngine

The session strategy lives in agent.chrome_session (CurrentChromeSession):
it dynamically discovers the running Chrome process, its REAL user-data dir
and profile (never hardcoded "Default"), validates the profile against the
running process, and attaches to the DevTools endpoint — supporting Chrome
144+ existing-session auto-connect (approval mode) where available.

There is NO fallback to a fresh/temporary profile for production tasks. If
the running Chrome cannot be attached, initialize() fails honestly with
BROWSER_SESSION_UNAVAILABLE plus the exact reason and what to enable.
The user's profile is never copied, written to, or modified.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from agent.chrome_session import (
    UNAVAILABLE as _BROWSER_SESSION_UNAVAILABLE,
    chrome_session,
)

logger = logging.getLogger(__name__)


@dataclass
class BrowserTab:
    """Represents a browser tab."""
    id: str
    url: str
    title: str
    active: bool = False


class BrowserController:
    """
    Controls the user's existing persistent Chrome browser.

    Strategy (enforced, never silently deviated from):
      1. Discover the CURRENT Chrome process, its real user-data dir and
         profile (dynamic discovery, never a hardcoded "Default").
      2. Attach to its DevTools/remote-debugging endpoint — including the
         Chrome 144+ auto-connect (approval-mode) endpoint.
      3. When NO Chrome is running at all, launch the user's REAL Chrome on
         their REAL profile with a debugging port (logins preserved).

    Never: a fresh/anonymous profile, a profile copy, or a second browser
    instance while the user's Chrome is already open.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._attached = False
        self._session_info = None
        self._discovered_info = None
        self._unavailable_reason = ""

    def initialize(self) -> bool:
        """
        Attach to the user's CURRENT Chrome session (see agent.chrome_session).

        On failure this fails HONESTLY: returns False and stores
        `unavailable_reason` starting with BROWSER_SESSION_UNAVAILABLE plus
        the exact reason and what needs to be enabled. No fresh profile is
        ever created as a fallback.

        Returns True if the existing session is attached.
        """
        logger.info("Initializing browser controller (existing session)...")
        try:
            info, pw, browser, context, page = chrome_session().attach()
        except Exception as e:
            # ChromeSessionUnavailable already carries reason + remediation;
            # anything else must still fail honestly instead of falling back.
            reason = getattr(e, "reason", "")
            remediation = getattr(e, "remediation", "")
            # The discovered session evidence survives the failure: the trace
            # still reports WHICH Chrome was found and why it is unusable.
            self._session_info = getattr(e, "info", None)
            self._discovered_info = getattr(e, "info", None)
            self._unavailable_reason = (
                f"{_BROWSER_SESSION_UNAVAILABLE}: "
                f"{reason or f'Chrome attach failed: {e}'}"
                + (f" Remediation: {remediation}" if remediation else ""))
            self._attached = False
            self._page = None
            # Keep the DISCOVERED identity (process/data dir/profile) so the
            # honest failure still records WHICH Chrome could not be attached.
            self._remember_discovery()
            logger.error("%s", self._unavailable_reason)
            return False

        self._playwright = pw
        self._browser = browser
        self._context = context
        self._page = page
        self._session_info = info
        self._attached = True
        self._unavailable_reason = ""
        logger.info("Attached to the existing Chrome session: %s",
                    info.evidence())
        return True

    # ── Session evidence (AgentTrace contract) ──────────

    def _remember_discovery(self) -> None:
        """Best-effort snapshot of the DISCOVERED Chrome (never raises).

        Used on the honest failure path so the trace still records which
        browser process/data dir/profile could not be attached.
        """
        try:
            from agent.chrome_session import discover_chrome_session
            self._discovered_info = discover_chrome_session()
        except Exception as e:      # discovery must never mask the failure
            logger.debug("session discovery for evidence failed: %s", e)

    @property
    def unavailable_reason(self) -> str:
        """BROWSER_SESSION_UNAVAILABLE + exact reason + remediation."""
        return self._unavailable_reason

    def session_evidence(self) -> dict:
        """Explicit browser-session evidence for the AgentTrace."""
        evidence = {
            "browser_process": "unknown",
            "user_data_dir": "unknown",
            "profile_directory": "unknown",
            "connection_method": "none",
            "authenticated_state": "unknown",
            "active_tab": "",
            "current_url": "",
        }
        info = self._session_info or self._discovered_info
        if info is not None:
            evidence.update(info.evidence())
        if self._page is not None:
            try:
                evidence["active_tab"] = str(self._page.title() or "")
            except Exception:
                pass
            try:
                evidence["current_url"] = str(self._page.url or "")
            except Exception:
                pass
        return evidence

    def unavailable_evidence(self) -> dict:
        """Session evidence for the honest unavailability path."""
        evidence = self.session_evidence()
        evidence["connection_method"] = "none"
        evidence["unavailable"] = (self._unavailable_reason
                                   or f"{_BROWSER_SESSION_UNAVAILABLE}: "
                                      f"not initialized")
        return evidence


    # ── Navigation ──────────────────────────────────────

    def navigate(self, url: str) -> bool:
        """Navigate to a URL. Returns True on success."""
        if not self._ensure_page():
            return False
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            agent_memory.set_browser_url(self._page.url)
            logger.info("Navigated to: %s", url)
            return True
        except Exception as e:
            logger.warning("Navigation failed: %s", e)
            return False

    def get_current_url(self) -> Optional[str]:
        """Get the current page URL."""
        if not self._page:
            return None
        try:
            return self._page.url
        except Exception:
            return None

    def get_page_title(self) -> Optional[str]:
        """Get the current page title."""
        if not self._page:
            return None
        try:
            return self._page.title()
        except Exception:
            return None

    # ── Tab Management ──────────────────────────────────

    def list_tabs(self) -> List[BrowserTab]:
        """List all open tabs."""
        tabs = []
        if not self._browser:
            return tabs

        try:
            pages = self._browser.pages if hasattr(self._browser, 'pages') else []
            for i, page in enumerate(pages):
                try:
                    tabs.append(BrowserTab(
                        id=str(i),
                        url=page.url,
                        title=page.title(),
                        active=page == self._page,
                    ))
                except Exception:
                    tabs.append(BrowserTab(id=str(i), url="about:blank", title="(unknown)"))
        except Exception as e:
            logger.warning("Failed to list tabs: %s", e)

        return tabs

    def switch_tab(self, index: int) -> bool:
        """Switch to a specific tab by index."""
        if not self._browser:
            return False
        try:
            pages = self._browser.pages if hasattr(self._browser, 'pages') else []
            if 0 <= index < len(pages):
                self._page = pages[index]
                self._page.bring_to_front()
                agent_memory.set_active_tab(index)
                logger.info("Switched to tab %d: %s", index, self._page.url)
                return True
        except Exception as e:
            logger.warning("Tab switch failed: %s", e)
        return False

    def new_tab(self, url: str = "about:blank") -> bool:
        """Open a new tab."""
        if not self._context:
            return False
        try:
            self._page = self._context.new_page()
            if url != "about:blank":
                self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            logger.info("New tab opened: %s", url)
            return True
        except Exception as e:
            logger.warning("New tab failed: %s", e)
            return False

    def close_tab(self) -> bool:
        """Close the current tab."""
        if not self._page:
            return False
        try:
            self._page.close()
            # Switch to another tab
            tabs = self.list_tabs()
            if tabs:
                self._page = self._browser.pages[0]
            else:
                self._page = None
            return True
        except Exception as e:
            logger.warning("Close tab failed: %s", e)
            return False

    # ── Page Interaction ────────────────────────────────

    def click(self, selector: str) -> bool:
        """Click an element by CSS selector."""
        if not self._ensure_page():
            return False
        try:
            self._page.click(selector, timeout=5000)
            return True
        except Exception as e:
            logger.warning("Click failed: %s", e)
            return False

    def click_text(self, text: str) -> bool:
        """Click an element containing specific text."""
        if not self._ensure_page():
            return False
        try:
            self._page.click(f"text={text}", timeout=5000)
            return True
        except Exception as e:
            logger.warning("Click text failed: %s", e)
            return False

    def type_text(self, text: str, selector: Optional[str] = None) -> bool:
        """Type text into an input field."""
        if not self._ensure_page():
            return False
        try:
            if selector:
                self._page.fill(selector, text, timeout=5000)
            else:
                self._page.keyboard.type(text)
            return True
        except Exception as e:
            logger.warning("Type text failed: %s", e)
            return False

    def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        if not self._ensure_page():
            return False
        try:
            self._page.keyboard.press(key)
            return True
        except Exception as e:
            logger.warning("Key press failed: %s", e)
            return False

    def get_text(self, selector: str) -> Optional[str]:
        """Get text content of an element."""
        if not self._ensure_page():
            return None
        try:
            return self._page.text_content(selector, timeout=5000)
        except Exception:
            return None

    def get_html(self, selector: str = "body") -> Optional[str]:
        """Get inner HTML of an element."""
        if not self._ensure_page():
            return None
        try:
            return self._page.inner_html(selector, timeout=5000)
        except Exception:
            return None

    def wait_for_selector(self, selector: str, timeout: float = 10.0) -> bool:
        """Wait for an element to appear."""
        if not self._ensure_page():
            return False
        try:
            self._page.wait_for_selector(selector, timeout=timeout * 1000)
            return True
        except Exception:
            return False

    def wait_for_load(self) -> bool:
        """Wait for page to load."""
        if not self._ensure_page():
            return False
        try:
            self._page.wait_for_load_state("networkidle", timeout=15000)
            return True
        except Exception:
            return False

    def screenshot(self, path: Optional[str] = None) -> Optional[str]:
        """Take a screenshot. Returns path to screenshot file."""
        if not self._ensure_page():
            return None
        if path is None:
            path = f"/tmp/Diego_screenshot_{int(time.time())}.png"
        try:
            self._page.screenshot(path=path, full_page=False)
            agent_memory.set_last_screenshot(path)
            return path
        except Exception as e:
            logger.warning("Screenshot failed: %s", e)
            return None

    def evaluate(self, script: str) -> Any:
        """Execute JavaScript in the page context."""
        if not self._ensure_page():
            return None
        try:
            return self._page.evaluate(script)
        except Exception as e:
            logger.warning("JS evaluate failed: %s", e)
            return None

    # ── State ───────────────────────────────────────────

    def _ensure_page(self) -> bool:
        """Ensure we have an active page ON the attached session.

        The page always comes from the ATTACHED context — the user's own
        Chrome. `new_page()` therefore opens a new TAB in that same Chrome
        window; it never creates a browser/context. An unattached controller
        creates nothing at all: returning False is what makes
        BROWSER_SESSION_UNAVAILABLE honest instead of a silent clean browser.
        """
        if self._page:
            return True
        if not self._attached or self._context is None:
            return False
        try:
            # A new TAB inside the user's existing Chrome — never a new
            # browser, never a fresh/temporary profile.
            self._page = self._context.new_page()
            return True
        except Exception:
            pass
        return False

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def is_available(self) -> bool:
        return self._page is not None

    def close(self) -> None:
        """Release browser resources."""
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._attached = False
        logger.info("Browser controller shut down")


# Global singleton
browser_controller = BrowserController()

# Import here to avoid circular imports
from agent.memory import agent_memory