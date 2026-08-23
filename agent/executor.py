"""
AgentExecutor — Executes actions on the desktop and browser.

Actions:
- Mouse: move, click, double-click, drag, scroll
- Keyboard: type, press key, hotkey
- Clipboard: read, write
- Browser: navigate, click, type, screenshot, evaluate
- Desktop: open application, focus window, screenshot
"""

import logging
import os
import shutil
import subprocess
import time
from typing import Optional, Dict, Any, List

from agent.memory import agent_memory
from agent.browser import browser_controller

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes actions on the desktop and browser.

    All actions return (success: bool, message: str).
    Failures are logged and returned to the planner for recovery.
    """

    def __init__(self):
        self._pyautogui = None
        self._pyperclip = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize desktop automation libraries."""
        logger.info("Initializing agent executor...")
        try:
            import pyautogui as _pag
            self._pyautogui = _pag
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE = 0.1  # Small delay between actions
        except ImportError:
            logger.warning("pyautogui not available — desktop automation limited")
            self._pyautogui = None

        try:
            import pyperclip as _pc
            self._pyperclip = _pc
        except ImportError:
            logger.debug("pyperclip not available — clipboard access limited")
            self._pyperclip = None

        self._initialized = True
        logger.info("Agent executor initialized")
        return True

    # ════════════════════════════════════════════════════
    # MOUSE ACTIONS
    # ════════════════════════════════════════════════════

    def mouse_move(self, x: int, y: int) -> tuple:
        """Move mouse to absolute coordinates."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            self._pyautogui.moveTo(x, y, duration=0.2)
            agent_memory.set_last_action(f"mouse_move({x}, {y})")
            return True, f"Mouse moved to ({x}, {y})"
        except Exception as e:
            return False, f"Mouse move failed: {e}"

    def mouse_click(self, x: Optional[int] = None, y: Optional[int] = None) -> tuple:
        """Click at current position or specified coordinates."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            if x is not None and y is not None:
                self._pyautogui.click(x, y)
            else:
                self._pyautogui.click()
            agent_memory.set_last_action(f"mouse_click({x}, {y})")
            return True, "Clicked"
        except Exception as e:
            return False, f"Click failed: {e}"

    def mouse_double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> tuple:
        """Double-click at current position or specified coordinates."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            if x is not None and y is not None:
                self._pyautogui.doubleClick(x, y)
            else:
                self._pyautogui.doubleClick()
            agent_memory.set_last_action(f"double_click({x}, {y})")
            return True, "Double clicked"
        except Exception as e:
            return False, f"Double click failed: {e}"

    def mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> tuple:
        """Drag from one position to another."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            self._pyautogui.moveTo(start_x, start_y, duration=0.2)
            self._pyautogui.drag(end_x - start_x, end_y - start_y, duration=0.5)
            agent_memory.set_last_action(f"drag({start_x},{start_y}→{end_x},{end_y})")
            return True, f"Dragged to ({end_x}, {end_y})"
        except Exception as e:
            return False, f"Drag failed: {e}"

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> tuple:
        """Scroll the mouse wheel (positive = up, negative = down)."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            if x is not None and y is not None:
                self._pyautogui.scroll(clicks, x, y)
            else:
                self._pyautogui.scroll(clicks)
            agent_memory.set_last_action(f"scroll({clicks})")
            return True, f"Scrolled {clicks}"
        except Exception as e:
            return False, f"Scroll failed: {e}"

    # ════════════════════════════════════════════════════
    # KEYBOARD ACTIONS
    # ════════════════════════════════════════════════════

    def keyboard_type(self, text: str) -> tuple:
        """Type a string of text."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            self._pyautogui.write(text, interval=0.01)
            agent_memory.set_last_action(f"type({text[:50]})")
            return True, f"Typed: {text[:50]}"
        except Exception as e:
            return False, f"Type failed: {e}"

    def keyboard_press(self, key: str) -> tuple:
        """Press a single key (e.g. 'enter', 'tab', 'escape')."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            self._pyautogui.press(key)
            agent_memory.set_last_action(f"press({key})")
            return True, f"Pressed: {key}"
        except Exception as e:
            return False, f"Key press failed: {e}"

    def keyboard_hotkey(self, *keys: str) -> tuple:
        """Press a combination of keys (e.g. Ctrl+C, Alt+Tab)."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            self._pyautogui.hotkey(*keys)
            agent_memory.set_last_action(f"hotkey({'+'.join(keys)})")
            return True, f"Pressed: {'+'.join(keys)}"
        except Exception as e:
            return False, f"Hotkey failed: {e}"

    # ════════════════════════════════════════════════════
    # CLIPBOARD ACTIONS
    # ════════════════════════════════════════════════════

    def clipboard_copy(self) -> tuple:
        """Copy selected text to clipboard."""
        if not self._pyautogui or not self._pyperclip:
            return False, "Clipboard not available"
        try:
            self._pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            text = self._pyperclip.paste()
            agent_memory.set_last_action("clipboard_copy")
            return True, f"Copied: {text[:100]}"
        except Exception as e:
            return False, f"Copy failed: {e}"

    def clipboard_paste(self, text: str) -> tuple:
        """Paste text from clipboard."""
        if not self._pyautogui or not self._pyperclip:
            return False, "Clipboard not available"
        try:
            self._pyperclip.copy(text)
            time.sleep(0.1)
            self._pyautogui.hotkey('ctrl', 'v')
            agent_memory.set_last_action("clipboard_paste")
            return True, "Pasted"
        except Exception as e:
            return False, f"Paste failed: {e}"

    # ════════════════════════════════════════════════════
    # BROWSER ACTIONS
    # ════════════════════════════════════════════════════

    def browser_navigate(self, url: str) -> tuple:
        """Navigate browser to URL."""
        if not browser_controller.is_available:
            return False, "Browser not available"
        result = browser_controller.navigate(url)
        if result:
            agent_memory.set_last_action(f"browser_navigate({url})")
            return True, f"Navigated to {url}"
        return False, "Navigation failed"

    def browser_click(self, selector: str) -> tuple:
        """Click an element by CSS selector."""
        if not browser_controller.is_available:
            return False, "Browser not available"
        result = browser_controller.click(selector)
        if result:
            agent_memory.set_last_action(f"browser_click({selector})")
            return True, f"Clicked {selector}"
        return False, f"Click failed: {selector}"

    def browser_click_text(self, text: str) -> tuple:
        """Click an element by its visible text."""
        if not browser_controller.is_available:
            return False, "Browser not available"
        result = browser_controller.click_text(text)
        if result:
            agent_memory.set_last_action(f"browser_click_text({text})")
            return True, f"Clicked '{text}'"
        return False, f"Click text failed: '{text}'"

    def browser_type(self, text: str, selector: Optional[str] = None) -> tuple:
        """Type text into the page."""
        if not browser_controller.is_available:
            return False, "Browser not available"
        result = browser_controller.type_text(text, selector)
        if result:
            agent_memory.set_last_action(f"browser_type({text[:30]})")
            return True, f"Typed into page"
        return False, "Type failed"

    def browser_screenshot(self) -> tuple:
        """Take a browser screenshot."""
        if not browser_controller.is_available:
            return False, "Browser not available"
        path = browser_controller.screenshot()
        if path:
            return True, f"Screenshot saved to {path}"
        return False, "Screenshot failed"

    def browser_get_url(self) -> tuple:
        """Get current browser URL."""
        url = browser_controller.get_current_url()
        if url:
            return True, f"Current URL: {url}"
        return False, "Could not get URL"

    def browser_get_text(self, selector: str) -> tuple:
        """Get text from a page element."""
        text = browser_controller.get_text(selector)
        if text:
            return True, text
        return False, f"Could not find text for '{selector}'"

    def browser_list_tabs(self) -> tuple:
        """List all open browser tabs."""
        tabs = browser_controller.list_tabs()
        if tabs:
            return True, "\n".join(f"[{t.id}] {t.title}: {t.url}" for t in tabs)
        return False, "No tabs found"

    def browser_new_tab(self, url: str = "about:blank") -> tuple:
        """Open a new browser tab."""
        result = browser_controller.new_tab(url)
        if result:
            return True, f"New tab opened: {url}"
        return False, "New tab failed"

    def browser_close_tab(self) -> tuple:
        """Close current browser tab."""
        result = browser_controller.close_tab()
        if result:
            return True, "Tab closed"
        return False, "Close tab failed"

    def browser_switch_tab(self, index: int) -> tuple:
        """Switch to a specific tab."""
        result = browser_controller.switch_tab(index)
        if result:
            return True, f"Switched to tab {index}"
        return False, f"Tab {index} not found"

    # ════════════════════════════════════════════════════
    # DESKTOP ACTIONS
    # ════════════════════════════════════════════════════

    def desktop_open(self, app_name: str) -> tuple:
        """Open a desktop application."""
        try:
            subprocess.Popen([app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            agent_memory.set_last_action(f"open_app({app_name})")
            return True, f"Opened {app_name}"
        except FileNotFoundError:
            # Try with shutil.which
            path = shutil.which(app_name)
            if path:
                subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
                return True, f"Opened {app_name}"
            return False, f"Application '{app_name}' not found"
        except Exception as e:
            return False, f"Failed to open {app_name}: {e}"

    def desktop_screenshot(self) -> tuple:
        """Take a desktop screenshot."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            path = f"/tmp/Diego_desktop_{int(time.time())}.png"
            screenshot = self._pyautogui.screenshot(path)
            agent_memory.set_last_screenshot(path)
            return True, f"Screenshot: {path}"
        except Exception as e:
            return False, f"Screenshot failed: {e}"

    def desktop_get_position(self) -> tuple:
        """Get current mouse position."""
        if not self._pyautogui:
            return False, "Desktop automation not available"
        try:
            x, y = self._pyautogui.position()
            return True, f"Mouse at ({x}, {y})"
        except Exception as e:
            return False, f"Position failed: {e}"

    # ════════════════════════════════════════════════════
    # STATE
    # ════════════════════════════════════════════════════

    @property
    def is_available(self) -> bool:
        return self._initialized

    def close(self) -> None:
        """Release executor resources."""
        self._initialized = False
        logger.info("Agent executor shut down")


# Global singleton
agent_executor = AgentExecutor()