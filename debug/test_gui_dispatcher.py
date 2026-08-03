"""
Headless test of the main-thread GUI dispatcher.

Proves (without a real display):
  1. Callables submitted from WORKER threads execute ON THE MAIN THREAD.
  2. submit(wait=True) blocks the worker until the main thread FINISHES
     (this is what guarantees "popup destroyed" before WAKE_LISTEN).
  3. Fire-and-forget submits never run on the worker thread.
  4. gui.stop() destroys the root on the main thread.

Run:  python debug/test_gui_dispatcher.py
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.gui_dispatcher import MainThreadGUI


class StubRoot:
    """Fake Tk root for headless testing."""
    def __init__(self):
        self.updates = 0
        self.destroyed = False

    def update(self):
        self.updates += 1

    def destroy(self):
        self.destroyed = True


async def main() -> int:
    gui = MainThreadGUI()
    # Inject a stub root instead of creating a real Tk (headless CI-safe).
    gui._root = StubRoot()
    gui._available = True
    gui._main_thread = threading.current_thread()
    gui._main_thread_ident = threading.get_ident()

    pump = asyncio.create_task(gui.pump(interval=0.01))

    executed_on = {}
    order = []

    def gui_op(name):
        executed_on[name] = threading.get_ident()
        order.append(name)
        time.sleep(0.05)  # simulate slow GUI work (e.g. window destroy)
        return name

    main_ident = threading.get_ident()
    worker_errors = []

    def worker():
        try:
            # Blocking submit: must run on main thread AND wait for completion.
            r1 = gui.submit(gui_op, "create", wait=True)
            assert r1 == "create"
            order.append("worker-resumed")   # must come AFTER "create"
            # Fire-and-forget: returns immediately, still runs on main thread.
            gui.submit_nowait(gui_op, "render")
            # Waited destroy: returns only after destruction completed.
            r2 = gui.submit(gui_op, "destroy", wait=True)
            assert r2 == "destroy"
            order.append("worker-after-destroy")
        except Exception as e:
            worker_errors.append(e)

    t = threading.Thread(target=worker, name="auth-worker")
    t.start()
    deadline = time.time() + 5.0
    while t.is_alive() and time.time() < deadline:
        await asyncio.sleep(0.02)
    t.join(timeout=2.0)
    await asyncio.sleep(0.1)  # let the render task drain

    assert not worker_errors, f"worker errors: {worker_errors}"
    assert not t.is_alive(), "worker deadlocked — submit(wait=True) never returned"

    # 1. EVERY operation ran on the main thread.
    for name, ident in executed_on.items():
        assert ident == main_ident, \
            f"'{name}' ran on thread {ident}, NOT main thread {main_ident}"

    # 2. Blocking submit waited for main-thread completion.
    assert order.index("create") < order.index("worker-resumed"), \
        f"submit(wait=True) did not wait: {order}"
    assert order.index("destroy") < order.index("worker-after-destroy"), \
        f"destroy was not waited: {order}"

    # 3. Fire-and-forget also ran on main thread.
    assert "render" in executed_on, "nowait task never executed"

    # 4. gui.stop() destroys root on the main thread.
    gui.stop()
    assert gui._root is None or gui._root.destroyed, "root not destroyed"
    assert not gui.available

    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)

    print(f"Executed on main thread: {sorted(executed_on)}")
    print(f"Order: {order}")
    print("ALL GUI DISPATCHER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
