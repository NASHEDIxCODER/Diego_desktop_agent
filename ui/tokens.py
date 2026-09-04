"""
Diego UI Design Tokens — single source of truth for geometry & color.

Every widget reads from this module. No magic numbers scattered
throughout widget code. Proportions are preserved across the
supported window sizes (680x620, 780x720, 1100x900).
"""

from __future__ import annotations

# ── Colors ─────────────────────────────────────────────────────
WINDOW_BACKGROUND = "#060a12"      # near-black navy
PANEL_BACKGROUND = "#0b111c"       # glass panel base
PANEL_BACKGROUND_HI = "#0e1522"    # elevated glass panel
PANEL_BORDER = "#16283c"           # 1px subtle border
PANEL_BORDER_SOFT = "#101c2c"      # even subtler border

PRIMARY_ACCENT = "#22d3ee"         # bright cyan / electric blue
PRIMARY_ACCENT_DIM = "#0e7490"
SECONDARY_ACCENT = "#a78bfa"       # subtle violet
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"

TEXT_PRIMARY = "#e6f1ff"
TEXT_SECONDARY = "#8ba3bd"
TEXT_MUTED = "#4a5a70"

GLOW_INTENSITY = 0.35              # 0..1 — master glow strength

# ── Geometry ───────────────────────────────────────────────────
PANEL_RADIUS = 14                  # consistent corner radius (px)
CARD_RADIUS = 10
HEADER_HEIGHT = 52                 # compact header
FOOTER_HEIGHT = 32
RIGHT_COLUMN_MIN_WIDTH = 206       # right column min width (px)
RIGHT_COLUMN_STRETCH = 28          # ~28% of width
LEFT_COLUMN_STRETCH = 72           # ~72% of width
HERO_MIN_HEIGHT = 280              # voice core hero min height

SPACING_SMALL = 6
SPACING_MEDIUM = 12
SPACING_LARGE = 20
MARGIN_PANEL = 14

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY = "'Inter', 'Segoe UI', 'Ubuntu', 'DejaVu Sans', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'DejaVu Sans Mono', monospace"

FONT_XS = 9
FONT_SM = 10
FONT_MD = 12
FONT_LG = 14
FONT_XL = 17
FONT_2XL = 22
FONT_TITLE = 20

# ── Supported reference sizes ──────────────────────────────────
WINDOW_MIN = (680, 620)
WINDOW_DEFAULT = (780, 720)
WINDOW_LARGE = (1100, 900)


# ── Derived helpers ────────────────────────────────────────────
def rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha (0..1) to 'rgba(r, g, b, a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"