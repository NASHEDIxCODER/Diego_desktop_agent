"""
StatusPopup — Tiny always-on-top floating window for Diego's runtime state.

Shows (Problem 9):
  - Current state: Waiting for wake word / Listening / Thinking / Speaking /
    Face Authentication / Conversation / Idle
  - Animated microphone (pulsing while listening)
  - Wake score
  - Face box (during face auth)
  - Conversation timer (countdown to session timeout)
  - Current transcript (what the user said)
  - Current response (what Diego is saying)
  - Tool currently executing

THREAD-SAFETY MODEL (same as FaceAuthPopup):
  * The popup owns NO thread and NO Tk root.
  * EVERY Tk operation is marshalled to the MAIN THREAD through
    ``core.gui_dispatcher.gui``.
  * ``show()`` blocks until the main thread has CREATED the window.
  * ``update()`` is fire-and-forget (~10 FPS from the runtime loop).
  * ``close(wait=True)`` blocks until the main thread has DESTROYED the
    window.

If the GUI dispatcher is unavailable (headless), show() returns False and
the runtime simply runs without the popup.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.gui_dispatcher import gui, GUIUnavailable

logger = logging.getLogger(__name__)

# ── Appearance ───────────────────────────────────────────
WIN_W, WIN_H = 300, 210
BG = "#0f1115"          # window background (near-black)
CARD = "#171a21"        # card background
FG = "#e6e9ef"          # primary text
SUB = "#9aa3b2"         # secondary text
ACCENT = "#3b82f6"      # blue accent
RED = "#ef4444"
YELLOW = "#f59e0b"
GREEN = "#22c55e"

# State → (label, color)
STATE_LABELS = {
    "BOOT": ("Booting", SUB),
    "LOAD_MODELS": ("Loading models", SUB),
    "INITIALIZE_SERVICES": ("Starting services", SUB),
    "FACE_AUTH": ("Face Authentication", YELLOW),
    "IDLE": ("Idle", SUB),
    "WAIT_WAKE": ("Waiting for wake word", SUB),
    "WAKE_DETECTED": ("Wake detected", GREEN),
    "GREETING": ("Greeting", ACCENT),
    "CONVERSATION": ("Conversation", ACCENT),
    "LISTENING": ("Listening", GREEN),
    "THINKING": ("Thinking", YELLOW),
    "EXECUTING": ("Executing", YELLOW),
    "SPEAKING": ("Speaking", ACCENT),
    "FOLLOW_UP": ("Listening for follow-up", GREEN),
    "TIMEOUT": ("Conversation timeout", SUB),
    "RECOVERING": ("Recovering", RED),
    "SHUTDOWN": ("Shutting down", SUB),
}


@dataclass
class PopupStatus:
    """A single UI update pushed by the runtime."""
    state: str = "BOOT"
    wake_score: float = 0.0
    transcript: str = ""
    response: str = ""
    tool: str = ""
    session_remaining_s: float = 0.0
    face_box: Optional[tuple] = None          # (x, y, w, h) in frame coords
    face_status: str = ""                     # "No face detected" / "Move closer" / ...
    mic_level: float = 0.0                    # 0..1 for the mic animation


class StatusPopup:
    """Main-thread-marshalled status popup. Owns NO GUI thread."""

    def __init__(self, title: str = "Diego"):
        self._title = title
        self._win = None
        self._canvas = None
        self._ok = False
        self._closed = threading.Event()
        self._destroyed = threading.Event()
        self._last_status: Optional[PopupStatus] = None
        self._drag = (0, 0)
        self._anim_t = 0.0

    # ── Public API (safe to call from ANY thread) ────────

    def show(self) -> bool:
        """Create the window ON THE MAIN THREAD. Blocks until created."""
        if self._ok:
            return True
        if not gui.available:
            logger.info("[POPUP] GUI unavailable — headless mode")
            return False
        self._closed.clear()
        self._destroyed.clear()
        try:
            ok = gui.submit(self._create_on_main, wait=True)
        except GUIUnavailable:
            return False
        except Exception as e:
            logger.warning("[POPUP] Window creation failed: %s", e)
            ok = False
        return bool(ok)

    def update(self, status: PopupStatus) -> None:
        """Push a new status (~10 FPS from the runtime loop)."""
        if not self._ok or self._closed.is_set():
            return
        self._last_status = status
        gui.submit_nowait(self._render_on_main)

    def close(self, wait: bool = True) -> None:
        """Destroy the popup ON THE MAIN THREAD."""
        self._closed.set()
        try:
            gui.submit(self._destroy_on_main, wait=True)
        except GUIUnavailable:
            self._destroyed.set()
        except Exception as e:
            logger.debug("[POPUP] destroy submit error: %s", e)
        if wait:
            self._destroyed.wait(timeout=3.0)

    @property
    def is_open(self) -> bool:
        return self._ok and not self._closed.is_set()

    # ═════════════════════════════════════════════════════
    # EVERYTHING BELOW RUNS ON THE MAIN THREAD ONLY.
    # ═════════════════════════════════════════════════════

    def _create_on_main(self) -> bool:
        """MAIN THREAD: build the Toplevel + canvas."""
        try:
            import tkinter as tk
            root = gui.root
            if root is None:
                return False

            win = tk.Toplevel(root)
            self._win = win
            win.title(self._title)
            win.overrideredirect(True)            # borderless
            win.attributes("-topmost", True)      # always on top
            try:
                win.attributes("-alpha", 0.97)
            except Exception:
                pass
            win.configure(bg=BG)
            # Bottom-right corner
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            win.geometry(f"{WIN_W}x{WIN_H}+{sw-WIN_W-24}+{sh-WIN_H-64}")
            win.protocol("WM_DELETE_WINDOW", self._destroy_on_main)

            canvas = tk.Canvas(win, width=WIN_W, height=WIN_H, bg=BG,
                               highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)
            self._canvas = canvas
            self._draw_card()

            # Dragging (since borderless)
            for w in (win, canvas):
                w.bind("<ButtonPress-1>", self._drag_start)
                w.bind("<B1-Motion>", self._drag_move)

            self._ok = True
            self._render_on_main()
            logger.info("[POPUP] Status window created on main thread")
            return True
        except Exception as e:
            logger.warning("[POPUP] Failed to create status window: %s", e)
            self._ok = False
            return False

    def _destroy_on_main(self) -> None:
        """MAIN THREAD: tear down the window. Idempotent."""
        try:
            if self._canvas is not None:
                try:
                    self._canvas.delete("all")
                except Exception:
                    pass
                self._canvas = None
            if self._win is not None:
                try:
                    self._win.destroy()
                except Exception:
                    pass
                self._win = None
        finally:
            self._ok = False
            self._closed.set()
            self._destroyed.set()
            logger.info("[POPUP] Status window destroyed on main thread")

    def _draw_card(self) -> None:
        c = self._canvas
        c.delete("card")
        r = 14
        self._round_rect(c, 4, 4, WIN_W-4, WIN_H-4, r, fill=CARD,
                         outline="#23262e", width=1, tags="card")
        c.create_text(16, 20, text="🎙  Diego", anchor="w",
                      fill=FG, font=("Segoe UI", 11, "bold"), tags="card")

    @staticmethod
    def _round_rect(c, x1, y1, x2, y2, r, **kw):
        pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
               x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return c.create_polygon(pts, smooth=True, **kw)

    def _drag_start(self, e):
        self._drag = (e.x, e.y)

    def _drag_move(self, e):
        if self._win is None:
            return
        x = self._win.winfo_x() + (e.x - self._drag[0])
        y = self._win.winfo_y() + (e.y - self._drag[1])
        self._win.geometry(f"+{x}+{y}")

    def _render_on_main(self) -> None:
        """MAIN THREAD: draw the latest status."""
        if self._closed.is_set() or self._win is None or self._canvas is None:
            return
        c = self._canvas
        st = self._last_status or PopupStatus()
        self._anim_t += 0.1

        c.delete("content")

        # ── State label + color ──
        label, color = STATE_LABELS.get(st.state, (st.state, SUB))
        c.create_text(16, 44, text=label, anchor="w", fill=color,
                      font=("Segoe UI", 10, "bold"), tags="content")

        # ── Animated microphone (pulsing while listening) ──
        mic_x, mic_y = WIN_W - 30, 44
        pulse = 0.0
        if st.state in ("LISTENING", "FOLLOW_UP", "WAIT_WAKE", "CONVERSATION"):
            pulse = (self._anim_t % 1.0) * 6.0
        mic_color = GREEN if st.state in ("LISTENING", "FOLLOW_UP") else SUB
        # Mic body
        c.create_oval(mic_x-8-pulse, mic_y-8-pulse, mic_x+8+pulse, mic_y+8+pulse,
                      outline=mic_color, width=1, tags="content")
        c.create_rectangle(mic_x-4, mic_y-6, mic_x+4, mic_y+2,
                           fill=mic_color, outline="", tags="content")
        c.create_arc(mic_x-5, mic_y-2, mic_x+5, mic_y+6, start=0, extent=180,
                     style="arc", outline=mic_color, tags="content")
        c.create_line(mic_x, mic_y+6, mic_x, mic_y+10, fill=mic_color,
                      tags="content")
        c.create_line(mic_x-4, mic_y+10, mic_x+4, mic_y+10, fill=mic_color,
                      tags="content")

        # ── Wake score (only meaningful in WAIT_WAKE) ──
        if st.state == "WAIT_WAKE" and st.wake_score > 0:
            c.create_text(16, 62, text=f"Wake: {st.wake_score:.2f}", anchor="w",
                          fill=SUB, font=("Segoe UI", 8), tags="content")

        # ── Conversation timer ──
        if st.session_remaining_s > 0:
            c.create_text(16, 76, text=f"Session: {st.session_remaining_s:.0f}s",
                          anchor="w", fill=SUB, font=("Segoe UI", 8),
                          tags="content")

        # ── Face box / status (during face auth) ──
        if st.state == "FACE_AUTH":
            c.create_text(16, 90, text=st.face_status or "Looking for you...",
                          anchor="w", fill=YELLOW, font=("Segoe UI", 9),
                          tags="content")

        # ── Current transcript ──
        if st.transcript:
            c.create_text(16, 108, text="You:", anchor="w", fill=SUB,
                          font=("Segoe UI", 8, "bold"), tags="content")
            c.create_text(16, 122, text=st.transcript[:60], anchor="w",
                          fill=FG, font=("Segoe UI", 9), width=WIN_W-32,
                          tags="content")

        # ── Current response ──
        if st.response:
            c.create_text(16, 142, text="Diego:", anchor="w", fill=SUB,
                          font=("Segoe UI", 8, "bold"), tags="content")
            c.create_text(16, 156, text=st.response[:60], anchor="w",
                          fill=ACCENT, font=("Segoe UI", 9), width=WIN_W-32,
                          tags="content")

        # ── Tool currently executing ──
        if st.tool:
            c.create_text(16, 180, text=f"⚙ {st.tool[:40]}", anchor="w",
                          fill=YELLOW, font=("Segoe UI", 8), tags="content")


# Global singleton
status_popup = StatusPopup()
