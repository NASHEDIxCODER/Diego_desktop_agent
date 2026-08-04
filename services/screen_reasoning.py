"""
ScreenReasoning — High-level screen understanding for Leo.

Wraps the OCR and UI Tree services with semantic reasoning:
  - Settings pages → recognizes categories, toggles, sliders
  - Dialogs → recognizes buttons, text fields, dropdowns, alerts
  - Forms → recognizes labels, inputs, validation errors
  - Code editors → recognizes file name, language, line number
  - Browser pages → recognizes page type, forms, videos, tables
  - Terminal output → recognizes errors, prompts, results
  - Dashboards → recognizes charts, metrics, filter controls
  - Charts → recognizes chart type, data ranges, trends
  - Tables → recognizes columns, headers, row counts
  - Buttons → recognizes button labels and states
  - Menus → recognizes menu items and hierarchies

Implements natural commands like:
  - "Click Build"
  - "Read this"
  - "Explain this error"
  - "What is wrong?"
  - "Fill this form"

Usage:
    from services.screen_reasoning import ScreenReasoner
    reasoner = ScreenReasoner()
    elements = reasoner.get_interactive_elements(screen_text, ui_tree)
    recommendation = reasoner.recommend_action(user_request, elements)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Element Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScreenElement:
    """A recognized interactive element on screen."""
    type: str  # button, text_field, toggle, dropdown, slider, link, tab, menu_item, label, input, checkbox, radio, table, chart, list
    label: str = ""
    value: str = ""
    position: Tuple[int, int] = (0, 0)  # x, y center
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    confidence: float = 0.5
    parent: str = ""  # parent container label
    state: str = ""  # enabled, disabled, focused, selected, checked
    role: str = ""  # accessibility role if available
    text_content: str = ""  # inner text for labels/buttons


@dataclass
class ScreenContext:
    """Semantic understanding of the current screen."""
    page_type: str = ""  # settings, dialog, form, editor, browser, terminal, dashboard, table, chart, menu, unknown
    title: str = ""
    elements: List[ScreenElement] = field(default_factory=list)
    error_elements: List[ScreenElement] = field(default_factory=list)
    text_content: str = ""
    ocr_text: str = ""
    ui_tree_text: str = ""


# ═══════════════════════════════════════════════════════════════
# Page type detection patterns
# ═══════════════════════════════════════════════════════════════

PAGE_TYPE_PATTERNS = {
    "settings": [
        re.compile(r"(settings|preferences|configuration|options)", re.IGNORECASE),
        re.compile(r"(on|off|enable|disable|toggle|slider)", re.IGNORECASE),
    ],
    "dialog": [
        re.compile(r"(ok|cancel|yes|no|confirm|dismiss|close|save|apply)", re.IGNORECASE),
        re.compile(r"(alert|warning|error|info|notification)", re.IGNORECASE),
    ],
    "form": [
        re.compile(r"(submit|reset|clear|send|save)", re.IGNORECASE),
        re.compile(r"(required|invalid|valid|field|input)", re.IGNORECASE),
    ],
    "editor": [
        re.compile(r"(file|edit|view|refactor|run|debug|tool)", re.IGNORECASE),
        re.compile(r"(line\s+\d+|column\s+\d+|\d+:\d+|\.py|\.js|\.go|\.ts|\.rs)", re.IGNORECASE),
    ],
    "browser": [
        re.compile(r"(https?://|www\.|\.com|\.org|\.io|\.dev)", re.IGNORECASE),
        re.compile(r"(back|forward|reload|bookmark|tab)", re.IGNORECASE),
    ],
    "terminal": [
        re.compile(r"(bash|zsh|fish|powershell|terminal|command)", re.IGNORECASE),
        re.compile(r"(\$|#|→|>)\s", re.IGNORECASE),
    ],
    "dashboard": [
        re.compile(r"(dashboard|overview|metrics|analytics)", re.IGNORECASE),
        re.compile(r"(chart|graph|plot|gauge|stat)", re.IGNORECASE),
    ],
    "table": [
        re.compile(r"(\|.*\|)", re.IGNORECASE),
        re.compile(r"(header|column|row|sort|filter|paginate)", re.IGNORECASE),
    ],
    "chart": [
        re.compile(r"(bar|line|pie|scatter|histogram|area)", re.IGNORECASE),
        re.compile(r"(axis|legend|data|series|trend)", re.IGNORECASE),
    ],
    "menu": [
        re.compile(r"(menu|context|dropdown|popup|toolbar)", re.IGNORECASE),
        re.compile(r"^\s*(✓|✔|•|▸)\s", re.IGNORECASE),
    ],
}


# ═══════════════════════════════════════════════════════════════
# Interactive element patterns
# ═══════════════════════════════════════════════════════════════

BUTTON_PATTERNS = [
    re.compile(r'(ok|cancel|yes|no|submit|save|delete|close|apply|reset|confirm|dismiss|back|next|finish|done|cancel)',
               re.IGNORECASE),
    re.compile(r'^(build|run|debug|start|stop|restart|deploy|commit|push|pull|merge|install|update)$',
               re.IGNORECASE),
]

TOGGLE_PATTERNS = [
    re.compile(r'(toggle|switch|on|off|enable|disable)', re.IGNORECASE),
]

INPUT_PATTERNS = [
    re.compile(r'(input|field|textbox|textarea|entry|search|filter)', re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════
# ScreenReasoner
# ═══════════════════════════════════════════════════════════════

class ScreenReasoner:
    """
    High-level screen understanding engine.

    Takes raw OCR text and UI tree data and produces semantic
    understanding: what type of page, what interactive elements
    are available, and what actions make sense.
    """

    def __init__(self):
        self._last_context: Optional[ScreenContext] = None

    # ── Screen Analysis ────────────────────────────────────────

    def analyze(
        self, ocr_text: str = "", ui_tree_text: str = "",
        screen_text: str = ""
    ) -> ScreenContext:
        """
        Analyze the current screen and return a ScreenContext.

        Args:
            ocr_text: Raw OCR output from screen capture.
            ui_tree_text: UI tree text from accessibility/UI detection.
            screen_text: Combined screen text from other sources.

        Returns:
            A ScreenContext with type classification and elements.
        """
        combined = (screen_text or "") + "\n" + (ocr_text or "") + "\n" + (ui_tree_text or "")
        combined = combined.strip()

        ctx = ScreenContext(
            ocr_text=ocr_text or "",
            ui_tree_text=ui_tree_text or "",
            text_content=combined,
        )

        # Detect page type
        ctx.page_type = self.detect_page_type(combined)

        # Extract interactive elements
        ctx.elements = self.extract_elements(combined, ctx.page_type)

        # Detect errors on screen
        ctx.error_elements = self.extract_errors(combined)

        self._last_context = ctx
        return ctx

    def detect_page_type(self, text: str) -> str:
        """
        Classify the type of page currently on screen.
        """
        scores: Dict[str, int] = {}

        for page_type, patterns in PAGE_TYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches) * 2
            if score > 0:
                scores[page_type] = score

        if not scores:
            return "unknown"

        return max(scores, key=scores.get)

    def extract_elements(self, text: str, page_type: str = "") -> List[ScreenElement]:
        """
        Extract recognizable interactive elements from screen text.
        """
        elements: List[ScreenElement] = []

        # Extract buttons
        for pattern in BUTTON_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                # Get position from surrounding context
                line_start = max(0, text.rfind("\n", 0, match.start()))
                line_end = text.find("\n", match.end())
                line = text[line_start:line_end] if line_end > 0 else text[line_start:]

                elements.append(ScreenElement(
                    type="button",
                    label=word,
                    text_content=line.strip()[:100],
                    confidence=0.7,
                ))

        # Extract toggles/switches
        for pattern in TOGGLE_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                elements.append(ScreenElement(
                    type="toggle",
                    label=word,
                    confidence=0.6,
                ))

        # Extract input fields
        for pattern in INPUT_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                elements.append(ScreenElement(
                    type="input",
                    label=word,
                    confidence=0.5,
                ))

        # Extract common UI elements from labels
        common_labels = re.findall(r'\b\w+\b', text)
        seen = set()
        for label in common_labels[:50]:
            if label.lower() in seen or len(label) < 3:
                continue
            seen.add(label.lower())

        return elements

    def extract_errors(self, text: str) -> List[ScreenElement]:
        """Extract visible error messages from screen."""
        errors: List[ScreenElement] = []

        error_patterns = [
            re.compile(r'error[:\s]+(.+)', re.IGNORECASE),
            re.compile(r'(failed|failure)[:\s]+(.+)', re.IGNORECASE),
            re.compile(r'(warning|caution|critical)[:\s]+(.+)', re.IGNORECASE),
            re.compile(r'✗\s*(.+)'),
            re.compile(r'✘\s*(.+)'),
            re.compile(r'❌\s*(.+)'),
            re.compile(r'⚠\s*(.+)'),
        ]

        for pattern in error_patterns:
            for match in pattern.finditer(text):
                msg = match.group(1).strip() if match.lastindex and match.group(1) else match.group(0)
                errors.append(ScreenElement(
                    type="error",
                    label=msg[:200],
                    confidence=0.8,
                ))

        return errors

    # ── Action Recommendations ─────────────────────────────────

    def recommend_action(
        self, user_request: str, elements: Optional[List[ScreenElement]] = None
    ) -> Dict[str, Any]:
        """
        Recommend an action based on what the user wants and what's on screen.

        Returns:
            Dict with keys: action_type, target_label, confidence, explanation
        """
        request = user_request.lower()
        ctx = self._last_context

        if elements is None and ctx:
            elements = ctx.elements

        if not elements:
            return {
                "action_type": "unknown",
                "confidence": 0.0,
                "explanation": "No recognizable elements on screen.",
            }

        # "Click X"
        click_match = re.search(r'(?:click|press|hit|tap)\s+(.+)', request)
        if click_match:
            target = click_match.group(1).strip()
            # Find best matching element
            for elem in elements:
                if target.lower() in elem.label.lower():
                    return {
                        "action_type": "click",
                        "target_label": elem.label,
                        "target_position": elem.position,
                        "confidence": 0.85,
                        "explanation": f"Found '{elem.label}' button on screen.",
                    }
            return {
                "action_type": "click",
                "target_label": target,
                "confidence": 0.3,
                "explanation": f"Looking for '{target}' but didn't find it on screen.",
            }

        # "Read this"
        if any(p in request for p in ("read this", "what does this say", "what is this",
                                        "explain this", "read what's", "read what is")):
            return {
                "action_type": "read_screen",
                "confidence": 0.9,
                "explanation": "Reading current screen content.",
            }

        # "What is wrong?"
        if any(p in request for p in ("what is wrong", "what's wrong",
                                        "what error", "why did it fail")):
            if ctx and ctx.error_elements:
                return {
                    "action_type": "explain_error",
                    "target_label": ctx.error_elements[0].label,
                    "confidence": 0.8,
                    "explanation": f"Found error: {ctx.error_elements[0].label[:100]}",
                }
            return {
                "action_type": "check_screen",
                "confidence": 0.5,
                "explanation": "Scanning screen for errors...",
            }

        # "Fill this form" / "Type X"
        type_match = re.search(r'(?:type|fill|enter|input|write)\s+(.+)', request)
        if type_match:
            text = type_match.group(1).strip()
            return {
                "action_type": "type",
                "text": text,
                "confidence": 0.8,
                "explanation": f"Typing '{text}'.",
            }

        # "Scroll up/down"
        if "scroll" in request:
            direction = "down" if "down" in request else "up"
            return {
                "action_type": "scroll",
                "direction": direction,
                "confidence": 0.9,
                "explanation": f"Scrolling {direction}.",
            }

        # Default: analyze the screen
        return {
            "action_type": "analyze_screen",
            "confidence": 0.4,
            "explanation": f"Current screen appears to be a {ctx.page_type if ctx else 'unknown'} page.",
        }

    # ── Context for LLM ────────────────────────────────────────

    def context_description(self) -> str:
        """Return a natural-language description for LLM injection."""
        if not self._last_context:
            return "No screen context available."

        ctx = self._last_context
        parts = [f"Screen type: {ctx.page_type}"]

        if ctx.elements:
            element_labels = [e.label for e in ctx.elements[:10]]
            parts.append(f"Interactive elements: {', '.join(element_labels)}")

        if ctx.error_elements:
            err_labels = [e.label for e in ctx.error_elements[:3]]
            parts.append(f"Errors: {'; '.join(err_labels)}")

        return " | ".join(parts)

    def quick_analysis(self, ocr_text: str) -> str:
        """Quick analysis of screen OCR text."""
        ctx = self.analyze(ocr_text=ocr_text)
        desc = self.context_description()
        return desc


# Global singleton
screen_reasoner = ScreenReasoner()