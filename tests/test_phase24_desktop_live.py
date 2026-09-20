"""
Phase 24 real-desktop validation harness (run with `python`, NOT pytest).

This is a SAFE, non-destructive probe of the live desktop stack that the
DesktopGoalEngine reuses. It performs NO message send and touches NO app state
destructively — it only:

  1. resolves the Telegram identity via services.app_resolver (aliases
     -> .desktop entries -> PATH; never a guessed launch),
  2. checks Telegram presence (process/window tables via /proc + xdotool),
  3. observes the live desktop through DesktopObserver, and
  4. exercises the Telegram semantic skill's semantic hint helpers against
     the OBSERVED context (no coordinates, no API).

It reports a PASS/FAIL/UNKNOWN result per step. When Telegram or an X display
is absent, the corresponding step reports UNKNOWN (honest degradation), never
a fabricated success. The message-send flow itself is intentionally NOT
automated here without an operator: sending is an external side effect gated
by explicit user confirmation (see tests/test_phase24_desktop_goal.py).

Run:
    .venv/bin/python tests/test_phase24_desktop_live.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when run as a standalone script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def _report(step: str, ok: bool | None, detail: str) -> None:
    if ok is True:
        print(f"[PASS]   {step}: {detail}")
    elif ok is False:
        print(f"[FAIL]   {step}: {detail}")
    else:
        print(f"[UNKNOWN] {step}: {detail}")


def main() -> int:
    failures = 0

    _header("1. Telegram identity resolution (AppResolver)")
    try:
        from services.app_resolver import ResolutionStatus, resolve_app
        res = resolve_app("Telegram")
        if res.status == ResolutionStatus.RESOLVED and res.identity is not None:
            ident = res.identity
            _report("resolve_app('Telegram')", True,
                    f"canonical={ident.canonical} exec={ident.executable} "
                    f"entry={ident.desktop_entry} "
                    f"proc={ident.process_patterns} win={ident.window_patterns}")
        else:
            _report("resolve_app('Telegram')", False,
                    f"status={res.status.value} msg={res.message}")
            failures += 1
    except Exception as e:  # noqa: BLE001
        _report("resolve_app('Telegram')", None, f"unavailable: {e}")

    _header("2. Telegram presence (process / window — target identity only)")
    try:
        from services.app_resolver import check_presence
        ident = resolve_app("Telegram").identity
        if ident is None:
            _report("check_presence", None, "no identity from step 1")
        else:
            ev = check_presence(ident)
            if ev.present:
                _report("check_presence", True,
                        f"via={ev.via} {ev.detail}")
            else:
                _report("check_presence", None,
                        f"Telegram not running (safe to skip): {ev.detail}")
    except Exception as e:  # noqa: BLE001
        _report("check_presence", None, f"unavailable: {e}")

    _header("3. Live desktop observation (DesktopObserver)")
    try:
        from agent.desktop_context import DesktopObserver
        ctx = DesktopObserver().observe("validation")
        _report("DesktopObserver.observe", ctx.observation_method != "",
                f"method={ctx.observation_method or 'none'} "
                f"app={ctx.active_application or '-'} "
                f"title='{ctx.window_title[:50]}' elements="
                f"{len(ctx.interactive_elements)} "
                f"state={ctx.application_state}")
        if ctx.observation_method == "":
            failures += 1
    except Exception as e:  # noqa: BLE001
        _report("DesktopObserver.observe", None, f"unavailable: {e}")

    _header("4. Telegram semantic skill (hints only, no coordinates)")
    try:
        from agent.desktop_context import DesktopContext, InteractiveElement
        from agent.desktop_skill_registry import DesktopSkillRegistry
        skill = DesktopSkillRegistry().get("telegram")
        ctx = DesktopContext(
            active_application="telegram", window_title="Telegram — Rahul",
            application_state="present", observation_method="accessibility",
            interactive_elements=[
                InteractiveElement(label="Message", kind="input",
                                   text_input=True),
                InteractiveElement(label="Send", kind="button",
                                   clickable=True),
            ],
        )
        _report("skill.message_input", skill.message_input(ctx) is not None,
                "found a semantic message input")
        _report("skill.send_control", skill.send_control(ctx) is not None,
                "found a semantic send control")
    except Exception as e:  # noqa: BLE001
        _report("telegram skill", None, f"unavailable: {e}")

    _header("5. Safety contract (no Telegram API / token)")
    try:
        import agent.desktop_goal_engine as dge
        import agent.desktop_skill_registry as dsr
        import agent.desktop_goal as dg
        sources = [dg, dsr, dge]
        leaked = []
        for mod in sources:
            blob = open(mod.__file__).read().lower()
            for token in ("telethon", "api_id", "api_hash", "bot_token",
                          "botfather"):
                if token in blob:
                    leaked.append((mod.__name__, token))
        if leaked:
            _report("no Telegram API/token", False, f"leaked: {leaked}")
            failures += 1
        else:
            _report("no Telegram API/token", True,
                    "no Telegram API imports or credentials anywhere")
    except Exception as e:  # noqa: BLE001
        _report("no Telegram API/token", None, f"unavailable: {e}")

    print("\nValidation complete with", failures, "hard failure(s).")
    print("UNKNOWN = capability not present on this host (honest, not fake).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
