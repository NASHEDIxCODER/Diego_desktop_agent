"""
Probe widget geometry at reference sizes to validate layout parity.

Checks:
    - visualizer size (no collapsed core)
    - left/right column proportions (~72/28)
    - transcript/response visibility and dominance
    - right panels readable (min widths)
    - footer present
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


def report(window, w, h):
    viz = window._visualizer
    tp = window._transcript_panel
    rp = window._response_panel
    vp = window._voice_state_panel
    ap = window._activity_panel
    mc = window._metrics
    ss = window._system_status
    footer = window._footer

    left_w = tp.parentWidget().width()
    right_w = vp.parentWidget().width()
    total = left_w + right_w

    print(f"\n== {w}x{h} ==")
    print(f"  visualizer:      {viz.width()}x{viz.height()}")
    print(f"  transcript:      {tp.width()}x{tp.height()} visible={tp.isVisible()}")
    print(f"  response:        {rp.width()}x{rp.height()} visible={rp.isVisible()}")
    print(f"  voice state:     {vp.width()}x{vp.height()}")
    print(f"  activity:        {ap.width()}x{ap.height()}")
    print(f"  metrics:         {mc.width()}x{mc.height()}")
    print(f"  system status:   {ss.width()}x{ss.height()}")
    print(f"  footer:          {footer.width()}x{footer.height()}")
    print(f"  columns L/R:     {left_w}/{right_w} = {left_w / total:.0%}/{right_w / total:.0%}")

    issues = []
    if viz.width() < 200 or viz.height() < 180:
        issues.append("visualizer collapsed")
    if not tp.isVisible() or not rp.isVisible():
        issues.append("transcript/response hidden")
    if rp.height() < tp.height():
        issues.append("response not dominant vs transcript")
    if right_w < 200:
        issues.append("right column too narrow")
    if ap.width() < 180:
        issues.append("activity panel too narrow")
    if footer.height() < 30:
        issues.append("footer collapsed")
    for name, panel in (("voice_state", vp), ("activity", ap),
                        ("metrics", mc), ("status", ss)):
        if panel.height() < 40:
            issues.append(f"{name} collapsed")
    if issues:
        print("  ISSUES:", "; ".join(issues))
    else:
        print("  OK")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    bridge = EventBridge()
    window = DiegoMainWindow(bridge=bridge, loop=None)
    window.show()

    bridge.emit_listening()
    bridge.emit_final("Diego, find my project which I have worked on recently")
    bridge.emit_response(
        "I found your most recent project: "
        "Diego-Voice-Assistant is located in ~/projects/Diego-Voice-Assistant. "
        "Would you like me to open it?"
    )

    def run():
        for w, h in [(680, 620), (780, 720), (1100, 900)]:
            window.resize(w, h)
            for _ in range(20):
                app.processEvents()
            report(window, w, h)
        bridge.stop()
        app.quit()

    QTimer.singleShot(400, run)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())