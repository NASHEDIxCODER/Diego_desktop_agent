"""
FaceAuthPopup — Floating, always-on-top face authentication window.

THREAD-SAFETY MODEL (fixes the Tcl_AsyncDelete → SIGKILL crash):

  The OLD popup created ``tk.Tk()`` and ran ``mainloop()`` on a private
  daemon thread. PhotoImage objects were then garbage-collected on whatever
  thread happened to drop the last reference (the auth executor thread),
  and Tcl killed the whole process:
      Tcl_AsyncDelete: async handler deleted by the wrong thread

  The NEW popup owns NO thread and NO Tk root:
    * EVERY Tk operation — Toplevel() creation, canvas updates, PhotoImage
      creation/disposal, destroy() — is marshalled to the MAIN THREAD
      through ``core.gui_dispatcher.gui``.
    * ``show()`` blocks until the main thread has CREATED the window.
    * ``update()`` is fire-and-forget (~30 FPS from the auth worker loop).
    * ``close(wait=True)`` blocks until the main thread has DESTROYED the
      window — callers can rely on teardown being complete before they
      continue (required before entering WAKE_LISTEN).

States & colors:
  searching  → no rectangle, "Searching for face..."
  no_face    → red rectangle,    "No face detected — please look at the camera"
  guidance   → red rectangle,    e.g. "Move closer", "Too dark", "Too blurry"
  detected   → yellow rectangle, "Face detected — hold still..."
  verified   → green rectangle,  "Identity verified — welcome <name>"

If the GUI dispatcher is unavailable (headless), show() returns False and
the caller falls back to a headless auth path.

Usage (from the auth worker thread):
    popup = FaceAuthPopup()
    if popup.show():                       # window created on main thread
        popup.update(frame_bgr, FaceStatus("detected", "Hold still...", box))
        ...
        popup.close(wait=True)             # destroyed on main thread, waited
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from core.gui_dispatcher import gui, GUIUnavailable

logger = logging.getLogger(__name__)

# ── Appearance ───────────────────────────────────────────
WIN_W, WIN_H = 420, 320
BG = "#0f1115"          # window background (near-black)
CARD = "#171a21"        # card background
FG = "#e6e9ef"          # primary text
SUB = "#9aa3b2"         # secondary text
ACCENT = "#3b82f6"      # blue accent
RED = "#ef4444"
YELLOW = "#f59e0b"
GREEN = "#22c55e"


@dataclass
class FaceStatus:
    """A single UI update pushed by the auth loop."""
    state: str                                  # searching|no_face|guidance|detected|verified
    message: str
    box: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h) in frame coords
    sub: str = ""                               # secondary line (e.g. quality detail)
    landmarks: Optional[list] = None            # [(x,y), ...] face landmark points (frame coords)
    confidence: float = -1.0                    # 0..1 recognition confidence (-1 = n/a)



class FaceAuthPopup:
    """Main-thread-marshalled face-auth popup. Owns NO GUI thread."""

    def __init__(self, title: str = "Face Authentication"):
        self._title = title
        self._win = None                # Toplevel — main thread only
        self._canvas = None             # Canvas   — main thread only
        self._tk_photo = None           # PhotoImage — created & dropped on main thread
        self._ok = False
        self._closed = threading.Event()
        self._destroyed = threading.Event()
        self._last_frame: Optional[np.ndarray] = None
        self._last_status: Optional[FaceStatus] = None
        self._close_after: Optional[float] = None
        self._drag = (0, 0)

    # ── Public API (safe to call from ANY thread) ────────

    def show(self) -> bool:
        """Create the window ON THE MAIN THREAD. Blocks until created."""
        if self._ok:
            return True
        if not gui.available:
            logger.info("[POPUP] GUI unavailable — headless auth mode")
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

    def update(self, frame_bgr: Optional[np.ndarray], status: FaceStatus) -> None:
        """Push a new frame + status (~30×/s from the auth worker loop)."""
        if not self._ok or self._closed.is_set():
            return
        if frame_bgr is not None:
            self._last_frame = frame_bgr
        self._last_status = status
        # Fire-and-forget: the render executes on the main thread.
        gui.submit_nowait(self._render_on_main)

    def close(self, wait: bool = True) -> None:
        """Destroy the popup ON THE MAIN THREAD.

        With wait=True (default) this returns ONLY after destruction has
        fully completed — the caller may safely proceed to WAKE_LISTEN.
        """
        self._closed.set()
        try:
            gui.submit(self._destroy_on_main, wait=True)
        except GUIUnavailable:
            self._destroyed.set()
        except Exception as e:
            logger.debug("[POPUP] destroy submit error: %s", e)
        if wait:
            self._destroyed.wait(timeout=3.0)

    def schedule_close(self, delay_s: float = 1.2) -> None:
        """Close the popup after a short delay (e.g. to show 'verified')."""
        self._close_after = time.time() + delay_s

    @property
    def is_open(self) -> bool:
        return self._ok and not self._closed.is_set()

    # ═════════════════════════════════════════════════════
    # EVERYTHING BELOW RUNS ON THE MAIN THREAD ONLY.
    # Never call these directly from a worker thread.
    # ═════════════════════════════════════════════════════

    def _create_on_main(self) -> bool:
        """MAIN THREAD: build the Toplevel + canvas."""
        try:
            import tkinter as tk
            from PIL import Image, ImageTk, ImageDraw
            self._Image = Image
            self._ImageTk = ImageTk
            self._ImageDraw = ImageDraw

            root = gui.root
            if root is None:
                return False

            win = tk.Toplevel(root)
            self._win = win
            win.title(self._title)
            win.overrideredirect(True)            # borderless
            win.attributes("-topmost", True)      # always on top
            try:
                win.attributes("-alpha", 0.98)
            except Exception:
                pass
            win.configure(bg=BG)
            # Center on screen
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            win.geometry(f"{WIN_W}x{WIN_H}+{(sw-WIN_W)//2}+{(sh-WIN_H)//2}")
            # If the user clicks the window's (WM) close, destroy safely here.
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
            self._render_on_main()  # first paint
            logger.info("[POPUP] Window created on main thread (id=%s)",
                        gui.main_thread_ident)
            return True
        except Exception as e:
            logger.warning("[POPUP] Failed to create window: %s", e)
            self._ok = False
            return False

    def _destroy_on_main(self) -> None:
        """MAIN THREAD: tear down the window. Idempotent."""
        try:
            self._tk_photo = None       # drop PhotoImage ON THE MAIN THREAD
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
            self._destroyed.set()       # close(wait=True) unblocks HERE
            logger.info("[POPUP] Window destroyed on main thread")

    def _draw_card(self) -> None:
        c = self._canvas
        c.delete("card")
        r = 18
        self._round_rect(c, 6, 6, WIN_W-6, WIN_H-6, r, fill=CARD,
                         outline="#23262e", width=1, tags="card")
        c.create_text(WIN_W//2, 28, text="🔒  Face Authentication",
                      fill=FG, font=("Segoe UI", 13, "bold"), tags="card")
        c.create_line(20, 46, WIN_W-20, 46, fill="#23262e", tags="card")

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
        """MAIN THREAD: draw the latest frame + status."""
        if self._closed.is_set() or self._win is None or self._canvas is None:
            return
        # Auto-close scheduling
        if self._close_after and time.time() >= self._close_after:
            self._destroy_on_main()
            return

        c = self._canvas
        status = self._last_status
        px, py, pw, ph = 20, 60, WIN_W-40, 180

        # Draw the webcam frame
        if self._last_frame is not None:
            try:
                img = self._frame_to_photo(self._last_frame, pw, ph, status)
                c.delete("preview")
                c.create_image(px+pw//2, py+ph//2, image=img, tags="preview")
                # Replacing this reference DISPOSES the previous PhotoImage
                # — and it happens right here, ON THE MAIN THREAD.
                self._tk_photo = img
            except Exception as e:
                logger.debug("[POPUP] render frame error: %s", e)
        else:
            c.delete("preview")
            c.create_rectangle(px, py, px+pw, py+ph, fill="#0a0c10",
                               outline="#23262e", tags="preview")
            c.create_text(px+pw//2, py+ph//2, text="Starting camera...",
                          fill=SUB, font=("Segoe UI", 10), tags="preview")

        # Status text
        c.delete("status")
        state = status.state if status else "searching"
        message = status.message if status else "Searching for face..."
        # TASK 6: GREEN only when a usable face is in frame; RED otherwise.
        color = {"no_face": RED, "guidance": RED, "detected": GREEN,
                 "verified": GREEN, "searching": SUB}.get(state, SUB)
        c.create_text(WIN_W//2, py+ph+22, text=message, fill=color,
                      font=("Segoe UI", 11, "bold"), tags="status", width=WIN_W-40)
        if status and status.sub:
            c.create_text(WIN_W//2, py+ph+42, text=status.sub, fill=SUB,
                          font=("Segoe UI", 9), tags="status", width=WIN_W-40)


    def _frame_to_photo(self, frame_bgr: np.ndarray, pw: int, ph: int,
                        status: Optional[FaceStatus]):
        """MAIN THREAD: BGR frame → Tk PhotoImage with box overlay."""
        import cv2 as cv
        img = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil = self._Image.fromarray(img)
        # Fit into preview box
        pil.thumbnail((pw, ph))
        # Pad to exact size (center on dark background)
        canvas_img = self._Image.new("RGB", (pw, ph), (10, 12, 16))
        ox = (pw - pil.width)//2
        oy = (ph - pil.height)//2
        canvas_img.paste(pil, (ox, oy))

        draw = self._ImageDraw.Draw(canvas_img)
        # TASK 6: GREEN box only when the face is usable (detected/verified);
        # RED box while searching / guiding. Landmarks + confidence drawn.
        if status and status.box:
            sx = pil.width / max(frame_bgr.shape[1], 1)
            sy = pil.height / max(frame_bgr.shape[0], 1)
            x, y, w, h = status.box
            bx1 = ox + int(x*sx); by1 = oy + int(y*sy)
            bx2 = ox + int((x+w)*sx); by2 = oy + int((y+h)*sy)
            col = {"no_face": (239, 68, 68), "guidance": (239, 68, 68),
                   "detected": (34, 197, 94), "verified": (34, 197, 94),
                   "searching": (154, 163, 178)}.get(status.state, (239, 68, 68))
            for i in range(3):  # thick rectangle
                draw.rectangle([bx1-i, by1-i, bx2+i, by2+i], outline=col)

            # ── Face landmarks (68-pt), scaled into preview coords ──
            if status.landmarks:
                lcol = (96, 165, 250)  # light blue
                for (lx, ly) in status.landmarks:
                    cx = ox + int(lx * sx)
                    cy = oy + int(ly * sy)
                    draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=lcol)

            # ── Confidence badge (top-left of the face box) ──
            if status.confidence >= 0.0:
                pct = f"{status.confidence * 100:.0f}%"
                draw.rectangle([bx1, by1 - 14, bx1 + 46, by1 - 2],
                               fill=(16, 18, 24))
                draw.text((bx1 + 3, by1 - 14), pct, fill=col)
        # Border around preview
        draw.rectangle([0, 0, pw-1, ph-1], outline=(35, 38, 46))

        return self._ImageTk.PhotoImage(canvas_img)



# Quick manual test (run directly: python -m auth.face_popup)
if __name__ == "__main__":
    import asyncio
    import cv2 as cv

    async def _demo():
        logging.basicConfig(level=logging.INFO)
        if not gui.start():
            print("GUI unavailable (headless?)")
            return
        pump = asyncio.create_task(gui.pump())
        p = FaceAuthPopup()
        if p.show():
            cap = cv.VideoCapture(0)
            states = ["searching", "no_face", "detected", "verified"]
            i = 0
            loop = asyncio.get_event_loop()
            try:
                while p.is_open and i < 300:
                    ret, frame = await loop.run_in_executor(None, cap.read)
                    st = states[(i//40) % len(states)]
                    box = (80, 40, 120, 120) if st in ("detected", "verified") else None
                    p.update(frame, FaceStatus(st, f"State: {st}", box))
                    i += 1
                    await asyncio.sleep(0.033)
            except KeyboardInterrupt:
                pass
            cap.release()
            p.close(wait=True)
            print("Popup destroyed — GUI cleanup complete")
        pump.cancel()
        gui.stop()

    asyncio.run(_demo())
