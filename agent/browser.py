"""
BrowserController — Controls an existing persistent Chrome profile.

Uses Chrome DevTools Protocol (CDP) via Playwright's connect_over_cdp
to attach to a running Chrome instance with your existing profile.

This means:
- Existing tabs, cookies, sessions, bookmarks are preserved
- No login required
- No temporary profile
- Full browser automation capability

If attach mode fails, falls back to a dedicated persistent profile.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default Chrome remote debugging port
DEFAULT_CDP_PORT = 9222

# Path for persistent Diego browser profile
DIEGO_PROFILE_DIR = Path.home() / ".Diego" / "browser_profile"


@dataclass
class BrowserTab:
    """Represents a browser tab."""
    id: str
    url: str
    title: str
    active: bool = False


class BrowserController:
    """
    Controls an existing persistent Chrome browser.

    Strategy:
    1. Try to attach to running Chrome via CDP (chrome://inspect)
    2. If that fails, launch Chrome with a persistent Diego profile
    3. Never create temporary/anonymous profiles

    All operations are thread-safe and reuse the same connection.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._attached = False
        self._cdp_port = DEFAULT_CDP_PORT
        self._warning_shown = False

    def initialize(self) -> bool:
        """
        Initialize browser connection.

        Strategy:
        1. Try CDP attach to existing Chrome session (preserves all sessions/cookies)
        2. If CDP fails (Chrome 136+ security), launch with persistent Diego profile
        3. Never create temporary/anonymous profiles

        Chrome 136+ Note:
        Newer Chrome versions restrict CDP connections to the default profile
        for security reasons. When attach fails, we automatically fall back to
        a dedicated Diego persistent profile at ~/.Diego/browser_profile.
        This preserves sessions within Diego's profile but won't have the user's
        existing logged-in sessions from their default Chrome profile.

        Returns True if browser is available.
        """
        logger.info("Initializing browser controller...")

        # Step 1: Try CDP attach to already-running Chrome
        if self._try_cdp_attach():
            self._attached = True
            logger.info("Browser attached via CDP (existing Chrome session)")
            return True

        # Step 2: Try launching Chrome with CDP enabled
        launched = self._ensure_chrome_running()
        if launched and self._try_cdp_attach():
            self._attached = True
            logger.info("Browser attached via CDP (launched Chrome)")
            return True

        # Step 2b: A launched Chrome can take longer than the readiness probe
        # to expose CDP, so keep polling BEFORE attempting a persistent
        # launch — that profile is locked by the Chrome we just started and
        # a persistent launch would only fail on the profile lock.
        if self._cdp_port_open() and self._attach_with_retry():
            self._attached = True
            logger.info("Browser attached via CDP (retried while Chrome "
                        "started)")
            return True

        # Step 3: Fallback to persistent Diego profile
        # This is the recommended path for Chrome 136+ which blocks
        # attaching to the default profile via CDP
        logger.info(
            "CDP attach failed (Chrome 136+ may block default profile access). "
            "Falling back to persistent Diego profile at %s",
            DIEGO_PROFILE_DIR,
        )
        if self._try_persistent_launch():
            logger.info("Browser launched with persistent Diego profile")
            return True

        logger.warning("No browser available — agent mode limited to desktop "
                       "actions")
        return False

    def _attach_with_retry(self, timeout: float = 20.0,
                           poll_s: float = 1.0) -> bool:
        """Retry the CDP attach until the deadline (bounded, no busy loop)."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._try_cdp_attach():
                return True
            if not self._cdp_port_open():
                return False          # nothing to attach to any more
            time.sleep(poll_s)
        return False

    def _find_chrome_binary(self) -> Optional[str]:
        """Find the Chrome/Chromium binary on the system."""
        import shutil
        # Common Chrome binary names
        chrome_names = [
            "google-chrome",
            "google-chrome-stable",
            "chromium-browser",
            "chromium",
            "chrome",
        ]
        for name in chrome_names:
            path = shutil.which(name)
            if path:
                return path
        return None

    def _ensure_chrome_running(self) -> bool:
        """
        Ensure Chrome is running with remote debugging enabled.
        
        Checks if Chrome is already running on the CDP port.
        If not, launches Chrome with --remote-debugging-port.
        """
        import subprocess

        # A CDP endpoint that already answers the version probe is THE thing we
        # need: reuse it (a mere open TCP port is not proof of a CDP browser).
        if self._cdp_ready():
            logger.debug("Reusing the live CDP endpoint on port %d",
                         self._cdp_port)
            return True

        # Find Chrome binary
        chrome_path = self._find_chrome_binary()
        if not chrome_path:
            logger.warning("Chrome binary not found")
            return False

        # Chrome 136+ REFUSES --remote-debugging-port on the default profile
        # dir, so the CDP instance must run on the dedicated persistent Diego
        # profile (the same one the persistent-launch fallback uses). It stays
        # persistent, so a one-time login there is reused afterwards.
        user_data_dir = str(DIEGO_PROFILE_DIR)
        try:
            os.makedirs(user_data_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Cannot create CDP profile dir %s: %s",
                           user_data_dir, e)
            return False

        args = [
            chrome_path,
            f"--remote-debugging-port={self._cdp_port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            # Headless host: still expose CDP, just without a visible window.
            args.append("--headless=new")

        # Launch Chrome with remote debugging
        try:
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Detach from Diego's process group so the browser outlives
                # the agent process that started it.
                start_new_session=True,
            )
            logger.info("Launched Chrome with CDP on port %d (profile: %s)",
                       self._cdp_port, user_data_dir)
            # Wait for the CDP endpoint to actually answer (not a blind sleep).
            return self._wait_for_cdp_port()
        except Exception as e:
            logger.warning("Failed to launch Chrome: %s", e)
            return False

    def _cdp_port_open(self) -> bool:
        """True when something accepts TCP connections on the CDP port."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(1.0)
            return sock.connect_ex(("127.0.0.1", self._cdp_port)) == 0
        except Exception:
            return False
        finally:
            sock.close()

    def _cdp_ready(self) -> bool:
        """True when the CDP endpoint answers the version probe."""
        from urllib.request import urlopen
        try:
            with urlopen(f"http://127.0.0.1:{self._cdp_port}/json/version",
                         timeout=3) as resp:
                return getattr(resp, "status", 200) == 200
        except Exception:
            return False

    def _wait_for_cdp_port(self, timeout: float = 20.0) -> bool:
        """Poll until Chrome's CDP endpoint is genuinely serving."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._cdp_ready():
                return True
            time.sleep(0.5)
        logger.warning("CDP endpoint on port %d did not become ready in %.0fs",
                       self._cdp_port, timeout)
        return False

    def _try_cdp_attach(self) -> bool:
        """Try to attach to a running Chrome instance via CDP."""
        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            # Chrome binds its CDP endpoint on IPv4 loopback; "localhost" can
            # resolve to ::1 first and yield ECONNREFUSED.
            cdp_url = f"http://127.0.0.1:{self._cdp_port}"

            # Try to connect
            self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
            # `Browser` exposes contexts, not pages: reuse the first context
            # (and its first tab) when present, otherwise create them.
            self._context = (self._browser.contexts[0]
                             if self._browser.contexts
                             else self._browser.new_context())
            self._page = (self._context.pages[0]
                          if self._context.pages
                          else self._context.new_page())

            if self._page:
                logger.info("CDP attach successful: %s", self._page.url)
                return True

            return False

        except Exception as e:
            logger.debug("CDP attach failed: %s", e)
            return False

    def _try_persistent_launch(self) -> bool:
        """Launch Chrome with a persistent Diego profile."""
        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()

            # Ensure profile directory exists
            DIEGO_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

            # Launch with persistent context
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(DIEGO_PROFILE_DIR),
                headless=False,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._browser = self._context  # Persistent context IS the browser

            logger.info("Persistent browser launched: %s", DIEGO_PROFILE_DIR)
            return True

        except Exception as e:
            logger.debug("Persistent launch failed: %s", e)
            return False

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
        """Ensure we have an active page."""
        if self._page:
            return True
        if self._context:
            try:
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