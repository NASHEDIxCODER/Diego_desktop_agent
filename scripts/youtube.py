"""
YouTube WebDriver controller for Leo Desktop Assistant.

Fully lazy-initialized with singleton browser reuse.
All operations have configurable timeouts.
Never blocks the assistant at import time.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton browser state
_driver = None
_wait = None
_initialized = False
_init_lock = None  # Will be set to threading.Lock() on first use

# Configurable timeouts
PAGE_LOAD_TIMEOUT = 15
ELEMENT_TIMEOUT = 10
SCRIPT_TIMEOUT = 5


def _get_lock():
    """Get or create the initialization lock."""
    global _init_lock
    if _init_lock is None:
        import threading
        _init_lock = threading.Lock()
    return _init_lock


def _ensure_browser():
    """Lazy-initialize the browser singleton. Thread-safe."""
    global _driver, _wait, _initialized

    if _initialized and _driver is not None:
        # Check if browser is still alive
        try:
            _driver.current_url  # Lightweight check
            return True
        except Exception:
            logger.info("Browser session lost, reinitializing...")
            _close_browser()

    with _get_lock():
        if _initialized and _driver is not None:
            return True

        try:
            from selenium import webdriver
            from selenium.webdriver.support.ui import WebDriverWait

            options = webdriver.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--mute-audio")
            options.page_load_strategy = "eager"  # Don't wait for full page load

            _driver = webdriver.Chrome(options=options)
            _driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            _wait = WebDriverWait(_driver, ELEMENT_TIMEOUT)
            _initialized = True
            logger.info("YouTube browser initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize YouTube browser: %s", e)
            _initialized = False
            return False


def _close_browser():
    """Close the browser if open."""
    global _driver, _wait, _initialized
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None
        _wait = None
        _initialized = False


def youtube():
    """Open YouTube homepage using webbrowser. Returns True on success."""
    import webbrowser
    try:
        webbrowser.open("https://www.youtube.com/")
        logger.info("YouTube opened in browser")
        return True
    except Exception as e:
        logger.warning("YouTube open failed: %s", e)
        return False


def search_song(query: str) -> bool:
    """Search and play a song on YouTube. Returns True on success."""
    if not _ensure_browser():
        return False
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.keys import Keys

        if "youtube.com" not in _driver.current_url:
            _driver.get("https://www.youtube.com/")

        search_box = _wait.until(EC.element_to_be_clickable((By.NAME, "search_query")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

        # Click first video
        first_video = _wait.until(
            EC.element_to_be_clickable((By.XPATH, "(//a[@id='video-title' and @href])[1]"))
        )
        first_video.click()

        # Ensure player is ready
        _wait.until(EC.presence_of_element_located((By.CLASS_NAME, "html5-video-player")))
        logger.info("Playing: %s", query)
        return True
    except Exception as e:
        logger.warning("YouTube search failed: %s", e)
        return False


def skip_ad():
    """Skip YouTube ad if present."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script("""
            let skipBtn = document.querySelector('.ytp-ad-skip-button, .ytp-ad-skip-button-modern');
            if (skipBtn) skipBtn.click();
        """)
    except Exception:
        pass


def pause_or_play():
    """Toggle play/pause."""
    if not _ensure_browser():
        return
    try:
        from selenium.webdriver.common.by import By
        video = _driver.find_element(By.TAG_NAME, "video")
        video.click()
        _driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        logger.warning("Pause/Play error: %s", e)


def play_next_song():
    """Play next video."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script("""
            var next = document.querySelector('.ytp-next-button');
            if (next) next.click();
        """)
        _driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        logger.warning("Next video error: %s", e)


def play_previous_song():
    """Play previous video."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script("""
            var prev = document.querySelector('.ytp-prev-button');
            if (prev) prev.click();
        """)
        _driver.execute_script("document.querySelector('video').focus();")
    except Exception as e:
        logger.warning("Previous video error: %s", e)


def set_playback_speed(speed: float):
    """Set playback speed."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script(
            f"document.querySelector('video').playbackRate = {speed};"
        )
    except Exception as e:
        logger.warning("Speed error: %s", e)


def increase_speed():
    """Increase playback speed."""
    if not _ensure_browser():
        return
    try:
        current = _driver.execute_script(
            "return document.querySelector('video').playbackRate;"
        )
        new_speed = min(current + 0.25, 2.0)
        set_playback_speed(new_speed)
    except Exception as e:
        logger.warning("Increase speed error: %s", e)


def decrease_speed():
    """Decrease playback speed."""
    if not _ensure_browser():
        return
    try:
        current = _driver.execute_script(
            "return document.querySelector('video').playbackRate;"
        )
        new_speed = max(current - 0.25, 0.25)
        set_playback_speed(new_speed)
    except Exception as e:
        logger.warning("Decrease speed error: %s", e)


def set_volume(level: float):
    """Set volume 0.0–1.0."""
    if not _ensure_browser():
        return
    level = max(0.0, min(1.0, level))
    try:
        _driver.execute_script(
            f"document.querySelector('video').volume = {level};"
        )
    except Exception as e:
        logger.warning("Volume error: %s", e)


def seek_forward(seconds: int = 10):
    """Seek forward."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script(
            f"var v=document.querySelector('video'); v.currentTime += {seconds};"
        )
    except Exception:
        pass


def seek_backward(seconds: int = 10):
    """Seek backward."""
    if not _ensure_browser():
        return
    try:
        _driver.execute_script(
            f"var v=document.querySelector('video'); v.currentTime -= {seconds};"
        )
    except Exception:
        pass


def toggle_mute() -> Optional[str]:
    """Toggle mute/unmute. Returns 'muted' or 'unmuted'."""
    if not _ensure_browser():
        return None
    try:
        status = _driver.execute_script("""
            var v = document.querySelector('video');
            if (v.muted === true) {
                v.muted = false;
                return "unmuted";
            } else {
                v.muted = true;
                return "muted";
            }
        """)
        return status
    except Exception as e:
        logger.warning("Mute/Unmute error: %s", e)
        return None


def close_youtube():
    """Close the browser."""
    _close_browser()
    logger.info("YouTube closed.")


def is_active() -> bool:
    """Check if YouTube browser is active."""
    if not _initialized or _driver is None:
        return False
    try:
        _driver.current_url
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("YouTube module loaded (lazy init)")