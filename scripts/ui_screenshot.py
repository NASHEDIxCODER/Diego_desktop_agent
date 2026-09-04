"""
Render the Diego UI offscreen at reference sizes and capture screenshots.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/ui_screenshot.py [outdir]
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402

from ui.event_bridge import EventBridge  # noqa: E402
from ui.main_window import DiegoMainWindow  # noqa: E402

SIZES = [(680, 620), (780, 720), (1100, 900)]


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "screenshots"
    os.makedirs(outdir, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    bridge = EventBridge()
    window = DiegoMainWindow(bridge=bridge, loop=None)
    window.show()

    # Populate representative content matching the reference
    bridge.emit_listening()
    bridge.emit_final("Diego, find my project which I have worked on recently")
    bridge.emit_response(
        "I found your most recent project: "
        "Diego-Voice-Assistant is located in ~/projects/Diego-Voice-Assistant. "
        "Would you like me to open it?"
    )
    bridge.emit_speaking()
    window.set_stt_latency(712)
    window.set_system_status("STT", True)
    window.set_system_status("Agent", True)
    window.set_system_status("TTS", True)
    window.set_system_status("Tools", True)

    def render_all():
        # Drive the visualizer with a representative mic level so the
        # hero core renders in its active listening state.
        window._visualizer.set_input_level(0.65)
        for w, h in SIZES:
            window.resize(w, h)
            window._visualizer.set_input_level(0.65)
            for _ in range(60):
                window._visualizer._animate()
                app.processEvents()
            pixmap = window.grab()
            path = os.path.join(outdir, f"diego_ui_{w}x{h}.png")
            pixmap.save(path)
            print(f"saved {path} ({pixmap.width()}x{pixmap.height()})")
        bridge.stop()
        app.quit()

    QTimer.singleShot(600, render_all)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())