"""
Phase 24: DesktopSkillRegistry — a generic skill boundary for desktop apps.

A SKILL is a declarative, app-scoped record of SEMANTIC capabilities. It maps
generic DesktopAction steps to semantic hints for perception + action, but it
contains NO hard-coded coordinates, NO Telegram API/token, and NO plumbing of
its own — every concrete action still goes through ``ComputerController`` and
all perception goes through ``ElementFinder`` / the accessibility/OCR tiers.

Future applications register capabilities here without touching
``DesktopGoalEngine``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.desktop_context import DesktopContext, InteractiveElement
from agent.desktop_goal import (
    ROLE_MESSAGE_INPUT,
    ROLE_SEARCH,
    DesktopAction,
)

logger = logging.getLogger(__name__)


@dataclass
class DesktopSkill:
    """Declarative capability record for ONE desktop application."""

    app: str                                   # canonical id, e.g. "telegram"
    display_name: str = ""
    # Semantic hints (words that identify roles in OBSERVED state) — generic,
    # no coordinates, no fixed selectors.
    search_words: tuple = ("search", "find", "query", "people", "contacts")
    message_input_words: tuple = ("message", "write", "type", "compose",
                                  "broadcast", "chat")
    send_words: tuple = ("send", "submit", "deliver")
    # Window-class / title markers proving the app is active (generic).
    window_markers: tuple = ()

    def contact_search_input(self, ctx: DesktopContext,
                             ) -> Optional[InteractiveElement]:
        return _match_input(ctx, self.search_words, self.message_input_words)

    def message_input(self, ctx: DesktopContext,
                      ) -> Optional[InteractiveElement]:
        return _match_input(ctx, self.message_input_words, self.search_words)

    def send_control(self, ctx: DesktopContext) -> Optional[InteractiveElement]:
        """Locate a send/submit control semantically (never coordinates)."""
        return _match_button(ctx, self.send_words)

    def supports(self, action: DesktopAction) -> bool:
        return action in _DEFAULT_CAPABILITIES

    def capabilities(self) -> List[str]:
        return sorted(a.value for a in _DEFAULT_CAPABILITIES)


# Capabilities every skill exposes (the generic desktop-goal set).
_DEFAULT_CAPABILITIES = frozenset({
    DesktopAction.OPEN_APPLICATION,
    DesktopAction.FIND_CONTACT,
    DesktopAction.VERIFY_CONTACT,
    DesktopAction.OPEN_CONVERSATION,
    DesktopAction.READ_CONVERSATION,
    DesktopAction.COMPOSE_MESSAGE,
    DesktopAction.SEND_MESSAGE,
    DesktopAction.VERIFY_SENT_MESSAGE,
})


def _match_input(ctx: DesktopContext, primary: tuple,
                 fallback: tuple) -> Optional[InteractiveElement]:
    """Semantically identify an input from OBSERVED labels (no coordinates)."""
    inputs = [e for e in ctx.inputs()]
    if not inputs:
        return None
    words = set(primary) | set(fallback)
    for e in inputs:
        hay = " ".join([e.label, e.role, e.kind]).lower()
        if any(w in hay for w in words):
            return e
    if len(inputs) == 1:
        return inputs[0]                 # exactly one text field — unambiguous
    return None


def _match_button(ctx: DesktopContext, words: tuple) -> Optional[InteractiveElement]:
    for e in ctx.buttons():
        hay = " ".join([e.label, e.role]).lower()
        if any(w in hay for w in words):
            return e
    return None


# ═══════════════════════════════════════════════════════════════════
# Telegram semantic skill (semantic hints only — NO API/token/coords)
# ═══════════════════════════════════════════════════════════════════

TELEGRAM_SKILL = DesktopSkill(
    app="telegram",
    display_name="Telegram",
    search_words=("search", "find", "people", "contacts"),
    message_input_words=("message", "write a message", "broadcast", "chat",
                         "compose"),
    send_words=("send", "submit", "deliver"),
    window_markers=("telegram",),
)


class DesktopSkillRegistry:
    """Holds registered skills and resolves the best match for a goal."""

    def __init__(self, skills: Optional[List[DesktopSkill]] = None) -> None:
        self._skills: Dict[str, DesktopSkill] = {}
        for skill in (skills or [TELEGRAM_SKILL]):
            self.register(skill)

    def register(self, skill: DesktopSkill) -> None:
        self._skills[skill.app] = skill
        logger.info("[DESKTOP-SKILL] registered skill for '%s'", skill.app)

    def get(self, app: str) -> Optional[DesktopSkill]:
        return self._skills.get((app or "").lower())

    def skill_for(self, app: str) -> Optional[DesktopSkill]:
        skill = self.get(app)
        if skill is None:
            # Fall back to a generic skill so ANY app can at least open /
            # read / compose / send through semantic perception.
            return DesktopSkill(app=app, display_name=app.title())
        return skill

    def apps(self) -> List[str]:
        return sorted(self._skills)

    def as_dict(self) -> Dict[str, Any]:
        return {app: {"display": s.display_name,
                      "capabilities": s.capabilities()}
                for app, s in self._skills.items()}


# Module-level singleton (the engine uses this).
desktop_skill_registry = DesktopSkillRegistry()


__all__ = [
    "DesktopSkill", "DesktopSkillRegistry", "TELEGRAM_SKILL",
    "desktop_skill_registry",
]
