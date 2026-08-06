"""
ScreenReasoning — High-level semantic screen understanding for Leo.

Enhanced v3: integrates with the new vision pipeline stages 8 (Semantic Reasoning).

Provides natural language understanding of the desktop:
  - Page type classification (settings, dialog, form, editor, browser, terminal, etc.)
  - Interactive element extraction (buttons, inputs, toggles, dropdowns, tabs, links)
  - Error detection (compiler errors, stack traces, status messages, warnings)
  - Natural command interpretation:
      "Click the blue Run button"
      "Open the second Chrome tab"
      "Close the popup"
      "Scroll until Build appears"
      "Click Install"
      "Read the error"
      "What changed?"
      "Summarize this screen"
      "Where is the login button?"
      "Which terminal failed?"
      "What is the compiler error?"
  - Context description for LLM prompt injection
  - Action recommendation from user intent

Usage:
    from services.screen_reasoning import screen_reasoner
    ctx = screen_reasoner.analyze(ocr_text, ui_tree_text)
    recommendation = screen_reasoner.recommend_action("click Run", ctx.elements)
"""

from __future__ import annotations

import json
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
    type: str  # button, text_field, toggle, dropdown, slider, link, tab, menu_item, label, input, checkbox, radio, table, chart, list, dialog, notification, error
    label: str = ""
    value: str = ""
    position: Tuple[int, int] = (0, 0)  # x, y center
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    confidence: float = 0.5
    parent: str = ""  # parent container label
    state: str = ""  # enabled, disabled, focused, selected, checked, active
    role: str = ""  # accessibility role if available
    text_content: str = ""  # inner text for labels/buttons
    region: str = ""  # layout region (toolbar, sidebar, editor, etc.)
    color_hint: str = ""  # "blue", "red", "green", etc. if detectable


@dataclass
class ScreenContext:
    """Semantic understanding of the current screen."""
    page_type: str = ""  # settings, dialog, form, editor, browser, terminal, dashboard, table, chart, menu, error, unknown
    title: str = ""
    elements: List[ScreenElement] = field(default_factory=list)
    error_elements: List[ScreenElement] = field(default_factory=list)
    text_content: str = ""
    ocr_text: str = ""
    ui_tree_text: str = ""
    app_type: str = ""  # from layout analyzer
    layout_regions: List[str] = field(default_factory=list)  # region names
    has_dialog: bool = False
    has_notification: bool = False
    reading_order: List[str] = field(default_factory=list)  # element labels in reading order


# ═══════════════════════════════════════════════════════════════
# Page type detection patterns
# ═══════════════════════════════════════════════════════════════

PAGE_TYPE_PATTERNS = {
    "settings": [
        re.compile(r"(settings|preferences|configuration|options|properties)", re.IGNORECASE),
        re.compile(r"(on|off|enable|disable|toggle|slider|switch|checkbox)", re.IGNORECASE),
        re.compile(r"(theme|appearance|language|region|timezone|privacy|security)", re.IGNORECASE),
    ],
    "dialog": [
        re.compile(r"(ok|cancel|yes|no|confirm|dismiss|close|save|apply|discard|delete)", re.IGNORECASE),
        re.compile(r"(alert|warning|error|info|notification|message|prompt)", re.IGNORECASE),
        re.compile(r"(are you sure|do you want|cannot|unable|confirm you)", re.IGNORECASE),
    ],
    "form": [
        re.compile(r"(submit|reset|clear|send|save|sign up|register|checkout)", re.IGNORECASE),
        re.compile(r"(required|invalid|valid|field|input|email|password|username)", re.IGNORECASE),
        re.compile(r"(\* required|\*required|mandatory)", re.IGNORECASE),
    ],
    "editor": [
        re.compile(r"(file|edit|view|refactor|run|debug|tools?|navigate)", re.IGNORECASE),
        re.compile(r"(line\s+\d+|column\s+\d+|\d+:\d+|\.py\b|\.js\b|\.go\b|\.ts\b|\.rs\b|\.java\b|\.cpp\b)", re.IGNORECASE),
        re.compile(r"(def |class |import |from |function |const |let |var )", re.IGNORECASE),
    ],
    "browser": [
        re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.io\b|\.dev\b|\.net\b)", re.IGNORECASE),
        re.compile(r"(back|forward|reload|bookmark|tab|new tab|incognito|history)", re.IGNORECASE),
        re.compile(r"(search|address bar|omnibox)", re.IGNORECASE),
    ],
    "terminal": [
        re.compile(r"(bash|zsh|fish|powershell|terminal|command|shell)", re.IGNORECASE),
        re.compile(r"(\$|#|→|>|\])\s", re.IGNORECASE),
        re.compile(r"(Error:|Traceback|Exception|Caused by:|at line)", re.IGNORECASE),
    ],
    "dashboard": [
        re.compile(r"(dashboard|overview|metrics|analytics|monitor|summary)", re.IGNORECASE),
        re.compile(r"(chart|graph|plot|gauge|stat|progress|utilization|throughput)", re.IGNORECASE),
    ],
    "table": [
        re.compile(r"(\|.*\|)", re.IGNORECASE),
        re.compile(r"(header|column|row|sort|filter|paginate|page \d+ of \d+)", re.IGNORECASE),
    ],
    "chart": [
        re.compile(r"(bar|line|pie|scatter|histogram|area)\s*(chart|graph)", re.IGNORECASE),
        re.compile(r"(axis|legend|data|series|trend|correlation|regression)", re.IGNORECASE),
    ],
    "menu": [
        re.compile(r"(menu|context|dropdown|popup|toolbar|popover)", re.IGNORECASE),
        re.compile(r"^\s*(✓|✔|•|▸|►|→)\s", re.IGNORECASE),
    ],
    "error": [
        re.compile(r"(error|exception|traceback|fail|crash|abort|killed|segfault)", re.IGNORECASE),
        re.compile(r"(cannot|could not|unable to|failed to|refused|denied|timeout)", re.IGNORECASE),
        re.compile(r"(fatal|critical|severe|panic|out of memory)", re.IGNORECASE),
    ],
}

# ── Interactive element detection patterns ────────────────────────

BUTTON_PATTERNS = [
    re.compile(r'\b(ok|cancel|yes|no|submit|save|delete|close|apply|reset|confirm|dismiss|back|next|finish|done|cancel|retry|skip|search|clear|refresh|reload|open|new|edit|copy|paste|undo|redo|send|share|export|import|download|upload|install|update|upgrade|remove|add|create|modify|enable|disable|on|off|login|logout|sign in|sign up|register|play|pause|stop|mute|merge|commit|push|pull)\b', re.IGNORECASE),
    re.compile(r'^(build|run|debug|start|restart|deploy|launch|execute|compile|test)$', re.IGNORECASE),
]

TOGGLE_PATTERNS = [
    re.compile(r'(toggle|switch|on/off|enable|disable)', re.IGNORECASE),
]

INPUT_PATTERNS = [
    re.compile(r'(input|field|textbox|textarea|entry|search|filter|query)', re.IGNORECASE),
    re.compile(r'(email|password|username|name|address|phone|zip|code)', re.IGNORECASE),
]

DROPDOWN_PATTERNS = [
    re.compile(r'(select|choose|pick|dropdown|combo)', re.IGNORECASE),
]

TAB_PATTERNS = [
    re.compile(r'\.(py|js|ts|html|css|json|yaml|yml|md|txt|java|go|rs|cpp|c|h|rb|php|sql)$', re.IGNORECASE),
]

LINK_PATTERNS = [
    re.compile(r'(https?://|www\.|ftp://|mailto:)', re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════
# ScreenReasoner
# ═══════════════════════════════════════════════════════════════

class ScreenReasoner:
    """
    High-level screen understanding engine.

    Takes raw OCR text and UI tree data and produces semantic
    understanding: what type of page, what interactive elements
    are available, what errors exist, and what actions make sense.

    Supports natural language commands about the screen:
      "Click the Run button"
      "Open the second tab"
      "Close the dialog"
      "What changed?"
      "Summarize this screen"
      "Where is the X button?"
      "What is the error?"
      "Scroll until X appears"
    """

    def __init__(self):
        self._last_context: Optional[ScreenContext] = None

    # ── Screen Analysis ────────────────────────────────────

    def analyze(
        self,
        ocr_text: str = "",
        ui_tree_text: str = "",
        screen_text: str = "",
        app_type: str = "",
    ) -> ScreenContext:
        """
        Analyze the current screen and return a ScreenContext.

        Args:
            ocr_text: Raw OCR output from screen capture.
            ui_tree_text: UI tree text from UI detection.
            screen_text: Combined screen text from other sources.
            app_type: Application type from layout analyzer.

        Returns:
            A ScreenContext with type classification and elements.
        """
        combined = (screen_text or "") + "\n" + (ocr_text or "") + "\n" + (ui_tree_text or "")
        combined = combined.strip()

        ctx = ScreenContext(
            ocr_text=ocr_text or "",
            ui_tree_text=ui_tree_text or "",
            text_content=combined,
            app_type=app_type,
        )

        # Detect page type
        ctx.page_type = self.detect_page_type(combined)

        # Extract interactive elements
        ctx.elements = self.extract_elements(combined, ctx.page_type)

        # Extract layout region hints
        ctx.layout_regions = self._extract_layout_regions(ui_tree_text)

        # Detect dialogs/notifications
        ctx.has_dialog = self._detect_dialog(combined)
        ctx.has_notification = self._detect_notification(combined)

        # Detect errors on screen
        ctx.error_elements = self.extract_errors(combined)

        # Build reading order
        ctx.reading_order = [e.label for e in ctx.elements if e.label]

        self._last_context = ctx
        return ctx

    def detect_page_type(self, text: str) -> str:
        """
        Classify the type of page currently on screen.

        Uses multi-pattern scoring — strongest match wins.
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

        Returns a list of ScreenElement objects with types, labels,
        and estimated positions.
        """
        elements: List[ScreenElement] = []
        seen_labels: set = set()

        # Extract buttons
        for pattern in BUTTON_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                if word in seen_labels:
                    continue
                seen_labels.add(word)

                # Get position from surrounding context
                line_start = max(0, text.rfind("\n", 0, match.start()))
                line_end = text.find("\n", match.end())
                line = text[line_start:line_end] if line_end > 0 else text[line_start:]

                elements.append(ScreenElement(
                    type="button",
                    label=word,
                    text_content=line.strip()[:100],
                    confidence=0.80,
                    role="button",
                ))

        # Extract toggles/switches
        for pattern in TOGGLE_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                if word in seen_labels:
                    continue
                seen_labels.add(word)
                elements.append(ScreenElement(
                    type="toggle",
                    label=word,
                    confidence=0.70,
                    role="switch",
                ))

        # Extract input fields
        for pattern in INPUT_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                if word in seen_labels:
                    continue
                seen_labels.add(word)
                elements.append(ScreenElement(
                    type="input",
                    label=word,
                    confidence=0.65,
                    role="textbox",
                ))

        # Extract dropdowns
        for pattern in DROPDOWN_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip().lower()
                if word in seen_labels:
                    continue
                seen_labels.add(word)
                elements.append(ScreenElement(
                    type="dropdown",
                    label=word,
                    confidence=0.65,
                    role="combobox",
                ))

        # Extract tabs
        for pattern in TAB_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip()
                if word in seen_labels:
                    continue
                seen_labels.add(word)
                elements.append(ScreenElement(
                    type="tab",
                    label=word,
                    confidence=0.75,
                    role="tab",
                ))

        # Extract links
        for pattern in LINK_PATTERNS:
            for match in pattern.finditer(text):
                word = match.group(0).strip()
                if word in seen_labels:
                    continue
                seen_labels.add(word)
                elements.append(ScreenElement(
                    type="link",
                    label=word,
                    confidence=0.85,
                    role="link",
                ))

        return elements

    def extract_errors(self, text: str) -> List[ScreenElement]:
        """Extract visible error messages from screen."""
        errors: List[ScreenElement] = []

        error_patterns = [
            # Standard error patterns
            re.compile(r'error[:\s]+(.+)', re.IGNORECASE),
            re.compile(r'(Error|ERROR):\s*(.+)'),
            re.compile(r'(failed|failure|cannot|could not|unable to)[:\s]+(.+)', re.IGNORECASE),
            re.compile(r'(warning|caution|critical|severe|fatal)[:\s]+(.+)', re.IGNORECASE),
            # Stack traces
            re.compile(r'Traceback\s*\(most recent call last\):.*?\n(.*?Error):\s*(.+)', re.DOTALL),
            re.compile(r'at\s+(.+?):(\d+):(\d+)'),
            # Symbol indicators
            re.compile(r'✗\s*(.+)'),
            re.compile(r'✘\s*(.+)'),
            re.compile(r'❌\s*(.+)'),
            re.compile(r'⚠\s*(.+)'),
            # Compiler errors
            re.compile(r'(\w+\.\w+):(\d+):(\d+):\s*error:\s*(.+)', re.IGNORECASE),
            re.compile(r'(.+)\.(py|js|ts|go|rs):(\d+):\s*(.+)'),
        ]

        for pattern in error_patterns:
            for match in pattern.finditer(text):
                # Get the best error message from groups
                if match.lastindex and match.lastindex >= 1:
                    # Take the last matched group (usually the message)
                    for g in range(match.lastindex, 0, -1):
                        if match.group(g) and len(match.group(g).strip()) > 3:
                            msg = match.group(g).strip()
                            break
                    else:
                        msg = match.group(0).strip()
                else:
                    msg = match.group(0).strip()

                errors.append(ScreenElement(
                    type="error",
                    label=msg[:300],
                    confidence=0.85,
                    role="alert",
                ))

        return errors

    @staticmethod
    def _extract_layout_regions(ui_tree_text: str) -> List[str]:
        """Extract layout region names from UI tree text."""
        regions: List[str] = []
        region_types = [
            "toolbar", "sidebar", "editor", "content", "status_bar",
            "navigation", "tab_bar", "menu_bar", "title_bar",
            "left_panel", "right_panel", "bottom_panel",
            "dialog", "popup", "notification",
        ]
        for line in ui_tree_text.split("\n"):
            line_lower = line.lower()
            for rt in region_types:
                if rt in line_lower and rt not in regions:
                    regions.append(rt)
        return regions

    @staticmethod
    def _detect_dialog(text: str) -> bool:
        """Check if a dialog is present on screen."""
        dialog_indicators = [
            "☐", "☑", "ok", "cancel", "yes", "no", "confirm",
            "dismiss", "alert", "warning", "notification",
        ]
        text_lower = text.lower()
        count = sum(1 for ind in dialog_indicators if ind in text_lower)
        return count >= 2

    @staticmethod
    def _detect_notification(text: str) -> bool:
        """Check if a notification is present."""
        notif_indicators = [
            "notification", "updated", "installed", "downloaded",
            "completed", "synced", "backup finished",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in notif_indicators)

    # ═══════════════════════════════════════════════════════════════
    # Natural Language Command Interpretation
    # ═══════════════════════════════════════════════════════════════

    def interpret_command(self, user_request: str) -> Dict[str, Any]:
        """
        Interpret a natural language command about the screen.

        Supported patterns:
          - "Click [element]" → action: click, target: element
          - "Open the [Nth] tab" → action: click_tab, target: N
          - "Close the [dialog/popup]" → action: close
          - "Scroll until [text] appears" → action: scroll_until
          - "Read the error" → action: read_error
          - "What changed?" → action: what_changed
          - "Summarize this screen" → action: summarize
          - "Where is the [element]?" → action: find
          - "What is the [error]?" → action: explain_error
          - "Open [app]" → action: open_app
          - "Type [text]" → action: type

        Returns:
            Dict with action_type, target, confidence, params, explanation.
        """
        request = user_request.strip()
        request_lower = request.lower()

        # "Click the blue Run button" / "Click Run" / "Click the X"
        click_match = re.search(
            r'(?:click|press|hit|tap|select)\s+(?:the\s+)?(?:(\w+)\s+)?(.+)',
            request_lower)
        if click_match:
            color = click_match.group(1)  # optional color (blue, red, green)
            target = click_match.group(2).strip()
            # Strip trailing "button", "icon", "link", "tab"
            for suffix in (" button", " icon", " link", " tab", " menu", " option"):
                if target.endswith(suffix):
                    target = target[:-len(suffix)]
            return {
                "action_type": "click",
                "target_label": target,
                "color_hint": color or "",
                "confidence": 0.85 if color else 0.80,
                "params": {"label": target},
                "explanation": f"Clicking '{target}'",
            }

        # "Open the [second/2nd/Nth] [browser] tab"
        tab_match = re.search(
            r'(?:open|switch to|go to)\s+(?:the\s+)?((\d+)(?:st|nd|rd|th)?|first|second|third|fourth|fifth|last)\s*(?:browser\s+)?tab',
            request_lower)
        if tab_match:
            ordinal = tab_match.group(1)
            # Convert ordinal to index
            ordinal_map = {"first": 0, "second": 1, "third": 2, "fourth": 3,
                           "fifth": 4, "last": -1}
            index = ordinal_map.get(ordinal.lower(), 0)
            if ordinal.isdigit():
                index = int(ordinal.rstrip("stndrdth")) - 1
            return {
                "action_type": "switch_tab",
                "target_label": str(index + 1),
                "tab_index": index,
                "confidence": 0.80,
                "params": {"index": index},
                "explanation": f"Switching to tab #{index + 1}",
            }

        # "Close the popup/dialog/window/notification"
        close_match = re.search(
            r'(?:close|dismiss|hide)\s+(?:the\s+)?(popup|dialog|window|notification|tab)',
            request_lower)
        if close_match:
            target_type = close_match.group(1)
            return {
                "action_type": "close",
                "target_label": target_type,
                "confidence": 0.90,
                "params": {"target_type": target_type},
                "explanation": f"Closing the {target_type}",
            }

        # "Scroll until [text] appears"
        scroll_match = re.search(
            r'(?:scroll|go)\s+(?:until|till|to)\s+(.+?)\s+(?:appears|is visible|shows?|comes)',
            request_lower)
        if scroll_match:
            target = scroll_match.group(1).strip()
            return {
                "action_type": "scroll_until",
                "target_label": target,
                "confidence": 0.75,
                "params": {"target_text": target},
                "explanation": f"Scrolling until '{target}' appears",
            }

        # "Read the error" / "What is the error?"
        if any(p in request_lower for p in (
            "read the error", "what is the error", "what error",
            "what's the error", "whats the error", "explain the error",
            "show me the error", "what failed", "why did it fail",
        )):
            if self._last_context and self._last_context.error_elements:
                return {
                    "action_type": "explain_error",
                    "target_label": self._last_context.error_elements[0].label,
                    "confidence": 0.90,
                    "params": {},
                    "explanation": f"Error found: {self._last_context.error_elements[0].label[:100]}",
                }
            return {
                "action_type": "check_screen",
                "confidence": 0.50,
                "params": {},
                "explanation": "Scanning screen for errors...",
            }

        # "Read this" / "Summarize this screen" / "What's on my screen?"
        if any(p in request_lower for p in (
            "read this", "what does this say", "what is this",
            "explain this", "read what's", "read what is",
            "summarize this", "what's on my screen", "whats on my screen",
            "what do you see", "describe this", "tell me what's",
        )):
            return {
                "action_type": "read_screen",
                "confidence": 0.95,
                "params": {},
                "explanation": "Reading current screen content.",
            }

        # "What changed?"
        if any(p in request_lower for p in (
            "what changed", "what's changed", "what has changed",
            "whats changed", "did anything change", "did the screen change",
        )):
            return {
                "action_type": "what_changed",
                "confidence": 0.95,
                "params": {},
                "explanation": "Checking what changed on screen.",
            }

        # "Where is the [element]?"
        find_match = re.search(
            r'(?:where is|wheres|find|locate|search for|look for)\s+(?:the\s+)?(.+)',
            request_lower)
        if find_match:
            target = find_match.group(1).strip()
            return {
                "action_type": "find_element",
                "target_label": target,
                "confidence": 0.75,
                "params": {"label": target},
                "explanation": f"Searching for '{target}' on screen",
            }

        # "Type [text]" / "Fill [text]" / "Enter [text]"
        type_match = re.search(
            r'(?:type|fill|enter|input|write)\s+(.+)',
            request_lower)
        if type_match:
            text = type_match.group(1).strip().strip('\'"')
            return {
                "action_type": "type",
                "target_label": text,
                "confidence": 0.85,
                "params": {"text": text},
                "explanation": f"Typing '{text}'",
            }

        # "Scroll up/down"
        if "scroll up" in request_lower:
            return {
                "action_type": "scroll",
                "confidence": 0.95,
                "params": {"direction": "up"},
                "explanation": "Scrolling up",
            }
        if "scroll down" in request_lower:
            return {
                "action_type": "scroll",
                "confidence": 0.95,
                "params": {"direction": "down"},
                "explanation": "Scrolling down",
            }

        # Default: analyze the screen
        return {
            "action_type": "analyze_screen",
            "confidence": 0.40,
            "params": {},
            "explanation": f"Current screen appears to be a {self._last_context.page_type if self._last_context else 'unknown'} page.",
        }

    # ═══════════════════════════════════════════════════════════════
    # Action Recommendations (legacy API, backward compatible)
    # ═══════════════════════════════════════════════════════════════

    def recommend_action(
        self, user_request: str, elements: Optional[List[ScreenElement]] = None
    ) -> Dict[str, Any]:
        """
        Recommend an action based on what the user wants and what's on screen.

        Backward compatible with the v2 API.

        Returns:
            Dict with keys: action_type, target_label, confidence, explanation
        """
        # Use the new command interpreter
        result = self.interpret_command(user_request)

        if result["confidence"] >= 0.70:
            return result

        # Fallback: search for matching elements
        request_lower = user_request.lower()
        ctx = self._last_context

        if elements is None and ctx:
            elements = ctx.elements

        if not elements:
            return {
                "action_type": "unknown",
                "confidence": 0.0,
                "explanation": "No recognizable elements on screen.",
            }

        # "Click X" — fallback search
        click_match = re.search(r'(?:click|press|hit|tap)\s+(.+)', request_lower)
        if click_match:
            target = click_match.group(1).strip()
            for elem in elements:
                if target.lower() in elem.label.lower():
                    return {
                        "action_type": "click",
                        "target_label": elem.label,
                        "target_position": elem.position,
                        "confidence": 0.85,
                        "params": {"label": elem.label},
                        "explanation": f"Found '{elem.label}' on screen.",
                    }
            return {
                "action_type": "click",
                "target_label": target,
                "confidence": 0.30,
                "params": {"label": target},
                "explanation": f"Looking for '{target}' but didn't find it.",
            }

        return result

    # ═══════════════════════════════════════════════════════════════
    # Context for LLM
    # ═══════════════════════════════════════════════════════════════

    def context_description(self) -> str:
        """Return a natural-language description for LLM injection."""
        if not self._last_context:
            return "No screen context available."

        ctx = self._last_context
        parts = [f"Screen type: {ctx.page_type}"]

        if ctx.has_dialog:
            parts.append("[DIALOG PRESENT]")
        if ctx.has_notification:
            parts.append("[NOTIFICATION PRESENT]")

        if ctx.layout_regions:
            parts.append(f"Layout regions: {', '.join(ctx.layout_regions[:8])}")

        if ctx.elements:
            # Group by type
            by_type: Dict[str, List[str]] = {}
            for e in ctx.elements[:30]:
                by_type.setdefault(e.type, []).append(e.label)

            element_parts = []
            for etype, labels in by_type.items():
                if etype in ("button", "input", "toggle", "dropdown", "tab", "link"):
                    element_parts.append(f"{etype}s: {', '.join(labels[:8])}")
            if element_parts:
                parts.append(f"Interactive: {'; '.join(element_parts)}")

        if ctx.error_elements:
            err_labels = [e.label[:100] for e in ctx.error_elements[:5]]
            parts.append(f"Errors: {' | '.join(err_labels)}")

        return " | ".join(parts)

    def compact_summary(self) -> str:
        """Ultra-compact summary for LLM context."""
        if not self._last_context:
            return ""

        ctx = self._last_context
        items = [f"page={ctx.page_type}"]

        if ctx.has_dialog:
            items.append("dialog=yes")
        if ctx.error_elements:
            items.append(f"errors={len(ctx.error_elements)}")

        buttons = [e.label for e in ctx.elements if e.type == "button"][:5]
        if buttons:
            items.append(f"buttons=[{', '.join(buttons)}]")

        tabs = [e.label for e in ctx.elements if e.type == "tab"][:3]
        if tabs:
            items.append(f"tabs=[{', '.join(tabs)}]")

        return " | ".join(items)

    def quick_analysis(self, ocr_text: str) -> str:
        """Quick analysis of screen OCR text."""
        ctx = self.analyze(ocr_text=ocr_text)
        return self.context_description()

    # ═══════════════════════════════════════════════════════════════
    # Semantic descriptions (for LLM)
    # ═══════════════════════════════════════════════════════════════

    def describe_screen(self) -> str:
        """Produce a human-readable description of the current screen."""
        if not self._last_context:
            return "I don't have any screen context yet."

        ctx = self._last_context
        parts: List[str] = []

        # Application and page type
        app_info = ""
        if ctx.app_type:
            app_info = f" {ctx.app_type}"
        parts.append(f"You're looking at a{app_info} {ctx.page_type} screen")

        # Layout
        if ctx.layout_regions:
            parts.append(f" with {', '.join(ctx.layout_regions[:4])}")

        # Elements
        if ctx.elements:
            button_count = sum(1 for e in ctx.elements if e.type == "button")
            input_count = sum(1 for e in ctx.elements if e.type == "input")
            tab_count = sum(1 for e in ctx.elements if e.type == "tab")

            element_descs = []
            if button_count:
                element_descs.append(f"{button_count} buttons")
            if input_count:
                element_descs.append(f"{input_count} text fields")
            if tab_count:
                element_descs.append(f"{tab_count} tabs")
            if element_descs:
                parts.append(f". I can see {', '.join(element_descs)}")

        # Errors
        if ctx.error_elements:
            parts.append(f". There {'is' if len(ctx.error_elements) == 1 else 'are'} "
                         f"{len(ctx.error_elements)} visible error message{'s' if len(ctx.error_elements) > 1 else ''}")

        # Dialog
        if ctx.has_dialog:
            parts.append(". A dialog is open")

        return "".join(parts) + "."

    def describe_errors(self) -> str:
        """Describe visible errors in natural language."""
        ctx = self._last_context
        if not ctx or not ctx.error_elements:
            return "I don't see any error messages on screen."

        if len(ctx.error_elements) == 1:
            return f"I see one error: {ctx.error_elements[0].label[:200]}"

        desc = f"I see {len(ctx.error_elements)} errors:\n"
        for i, err in enumerate(ctx.error_elements[:5], 1):
            desc += f"  {i}. {err.label[:150]}\n"
        return desc


# Global singleton
screen_reasoner = ScreenReasoner()