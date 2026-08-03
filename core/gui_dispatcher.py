"""
MainThreadGUI — every GUI operation on the main thread. NO exceptions.

WHY THIS EXISTS:
The old FaceAuthPopup created ``tk.Tk()`` and ran ``mainloop()`` on a
daemon worker thread while PhotoImage objects were garbage-collected on the
asyncio executor thread. Tcl then aborted the process:

    Tcl_AsyncDelete: async handler deleted by the wrong thread
    → SIGKILL (immediately after successful face authentication)

THE RULE enforced here:
  * The hidden Tk root is created ONCE, ON THE MAIN THREAD.
  * ALL Tkinter / OpenCV-GUI creation, update and destruction — Tk(),
    Toplevel(), imshow(), waitKey(), namedWindow(), destroy(), quit(),
    update(), mainloop(), after() — is marshalled to the main thread.
  * Worker threads NEVER touch a GUI object directly. They submit callables
    through this dispatcher and (optionally) wait for completion.

HOW IT WORKS:
  * ``gui.start()`` is called from the main thread during BOOT and creates a
    hidden (withdrawn) ``tk.Tk()`` root.
  * ``gui.pump()`` is an asyncio task on the main event loop: every ~30 ms
    it drains the task queue (executing submitted callables ON THE MAIN
    THREAD) and calls ``root.update()`` so Tk events are processed without
    ever calling ``mainloop()``.
  * ``gui.submit(fn, wait=True)`` blocks a worker thread until the main
    thread has executed ``fn`` — used for window creation and, critically,
    for DESTRUCTION so callers can wait until GUI teardown has finished.
  * Headless systems: ``start()`` fails cleanly, ``available`` is False and
    callers fall back to headless behaviour.

Usage:
    from core.gui_dispatcher import gui

    gui.start()                          # main thread, during BOOT
    pump_task = asyncio.create_task(gui.pump())

    ok = gui.submit(create_window, wait=True)      # from any thread
    gui.submit_nowait(render_frame, frame)         # fire-and-forget
    gui.submit(destroy_window, wait=True)          # waits until destroyed
"""

import asyncio
import logging
import queue
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class GUIUnavailable(RuntimeError):
    """Raised when a GUI operation is requested but no GUI exists."""


class MainThreadGUI:
    """Owns the single hidden Tk root and marshals ALL GUI work to the
    main thread."""

    def __init__(self):
        self._root = None
        self._tk = None
        self._main_thread: Optional[threading.Thread] = None
        self._main_thread_ident: Optional[int] = None
        self._tasks: "queue.Queue" = queue.Queue()
        self._available = False
        self._stopping = False

    # ── Lifecycle (main thread only) ──────────────────────

    def start(self) -> bool:
        """Create the hidden Tk root. MUST be called on the main thread."""
        if self._available:
            return True
        self._main_thread = threading.current_thread()
        self._main_thread_ident = threading.get_ident()
        self._stopping = False
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()          # hidden root — popups are Toplevels
            self._tk = tk
            self._root = root
            self._available = True
            logger.info("[GUI] Main-thread GUI dispatcher started "
                        "(main thread id=%s)", self._main_thread_ident)
            return True
        except Exception as e:
            logger.warning("[GUI] Tk unavailable (%s) — running headless", e)
            self._root = None
            self._available = False
            return False

    def stop(self) -> None:
        """Destroy the root on the main thread and shut the dispatcher down."""
        self._stopping = True

        def _destroy() -> None:
            if self._root is not None:
                try:
                    self._root.destroy()
                except Exception:
                    pass
                self._root = None

        if self._available:
            try:
                self.submit(_destroy, wait=True)
            except Exception:
                pass
        self._available = False
        logger.info("[GUI] Main-thread GUI dispatcher stopped")

    # ── Thread checks ─────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def root(self):
        """The hidden Tk root. TOUCH ONLY ON THE MAIN THREAD."""
        return self._root

    @property
    def main_thread_ident(self) -> Optional[int]:
        return self._main_thread_ident

    def is_gui_thread(self) -> bool:
        """True when called from the thread that owns the Tk root."""
        if self._main_thread is None:
            return threading.current_thread() is threading.main_thread()
        return threading.current_thread() is self._main_thread

    def assert_gui_thread(self, what: str = "GUI operation") -> None:
        if not self.is_gui_thread():
            raise RuntimeError(
                f"{what} attempted on worker thread "
                f"'{threading.current_thread().name}' — forbidden. "
                f"Marshal it via gui.submit().")

    # ── Marshalling ───────────────────────────────────────

    def submit(self, fn: Callable, *args, wait: bool = True, **kwargs) -> Any:
        """Execute ``fn(*args, **kwargs)`` ON THE MAIN GUI THREAD.

        Args:
            fn:   Callable performing GUI work (create/update/destroy).
            wait: True  → block the caller until the main thread finished
                         (returns fn's result, re-raises fn's exception).
                  False → fire-and-forget (returns None immediately).
        """
        if not self._available:
            raise GUIUnavailable("GUI dispatcher not available (headless?)")
        if self.is_gui_thread():
            # Already on the main thread — execute directly, never deadlock.
            return fn(*args, **kwargs)
        if not wait:
            self._tasks.put((fn, args, kwargs, None))
            return None
        done = threading.Event()
        box: dict = {}
        self._tasks.put((fn, args, kwargs, (done, box)))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def submit_nowait(self, fn: Callable, *args, **kwargs) -> None:
        """Fire-and-forget GUI task. Silently dropped when headless."""
        try:
            self.submit(fn, *args, wait=False, **kwargs)
        except GUIUnavailable:
            pass

    # ── Main-thread pump (asyncio task on the main loop) ──

    def pump_once(self) -> None:
        """Drain queued tasks and pump Tk. MAIN THREAD ONLY."""
        while True:
            try:
                fn, args, kwargs, waiter = self._tasks.get_nowait()
            except queue.Empty:
                break
            try:
                result = fn(*args, **kwargs)
                if waiter is not None:
                    waiter[1]["result"] = result
            except Exception as e:
                if waiter is not None:
                    waiter[1]["error"] = e
                logger.debug("[GUI] task error: %s", e)
            finally:
                if waiter is not None:
                    waiter[0].set()
        if self._root is not None:
            try:
                # update() processes all pending Tk events (replaces mainloop,
                # which must never be called — the asyncio loop owns the thread).
                self._root.update()
            except Exception:
                pass

    async def pump(self, interval: float = 0.03) -> None:
        """Asyncio task: keep the GUI alive from the main event loop."""
        logger.info("[GUI] Pump running on main event loop (interval=%.0fms)",
                    interval * 1000)
        while not self._stopping:
            try:
                self.pump_once()
            except Exception as e:
                logger.debug("[GUI] pump error: %s", e)
            await asyncio.sleep(interval)


# Global singleton — the ONLY way any module may touch GUI objects.
gui = MainThreadGUI()
