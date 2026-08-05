"""
Leo Desktop Assistant — Production Vision System (v3)

Backward-compatible bridge to the new vision pipeline.

The old `vision_manager` singleton still exists and delegates to:
  - services.vision_service for screen analysis
  - vision.ocr_pipeline for OCR
  - services.screen_capture for capture

New architecture:
  vision/
    frame_differencer.py  — Frame hash + motion + gating
    layout_analyzer.py    — App type detection + region segmentation
    ocr_pipeline.py       — Enhanced multi-backend OCR
    screen_memory.py      — Frame history + state comparison
    action_verifier.py    — Post-action verification + auto-retry
    perception.py         — Continuous desktop observer

Services:
  services/vision_service.py   — 10-stage vision pipeline orchestrator
  services/screen_capture.py   — mss/pyautogui capture
  services/ui_tree.py          — Structured UI element hierarchy
  services/screen_reasoning.py — Semantic reasoning + NL commands
  services/desktop_observer.py — Background desktop state monitor
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Legacy types (backward compatible)
# ═══════════════════════════════════════════════════════════════

from enum import Enum, auto


class CaptureSource(Enum):
    """Sources for vision capture (legacy, kept for backward compat)."""
    FULL_SCREEN = auto()
    ACTIVE_MONITOR = auto()
    ACTIVE_WINDOW = auto()
    SELECTED_REGION = auto()
    WEBCAM = auto()
    CLIPBOARD = auto()


class VisionModel(Enum):
    """Supported vision model backends (legacy)."""
    QWEN2_5_VL = "qwen2.5-vl"
    QWEN3_VL = "qwen3-vl"
    MOONDREAM = "moondream"
    LLAVA = "llava"
    MINICPM_V = "minicpm-v"
    OPENAI = "openai"
    GEMINI = "gemini"
    TESSERACT = "tesseract"


# Bridge to new enhanced OCR pipeline
from vision.ocr_pipeline import OCRBox, OCRResult, EnhancedOCREngine, TextClass

# Bridge to new frame differencer
from vision.frame_differencer import (
    FrameDifferencer, FrameDiffResult, GatingDecision, MotionRegion,
    frame_differencer,
)

# Bridge to new layout analyzer
from vision.layout_analyzer import (
    LayoutAnalyzer, ApplicationType, WindowLayout, RegionType, LayoutRegion,
    layout_analyzer,
)

# Bridge to new screen memory
from vision.screen_memory import (
    ScreenMemory, ScreenSnapshot,
    screen_memory,
)

# Bridge to new action verifier
from vision.action_verifier import (
    ActionVerifier, VerificationResult, VerificationStatus,
    action_verifier,
)

# Bridge to new perception
from vision.perception import PerceptionService, perception_service


# ═══════════════════════════════════════════════════════════════
# Legacy VisionManager bridge (backward compatible)
# ═══════════════════════════════════════════════════════════════

class VisionManagerBridge:
    """
    Backward-compatible bridge to the new vision pipeline.

    Delegates to services.vision_service for all screen analysis.
    Existing code that uses `vision_manager` continues to work.
    """

    def __init__(self):
        self._initialized = False
        self._config: Dict[str, Any] = {
            "ocr_enabled": True,
            "cache_enabled": True,
            "cache_ttl": 2.0,
            "auto_capture": True,
            "vision_model": "none",
        }
        self._screen_capturer = None   # lazily set
        self._ocr_engine = None        # lazily set

    @property
    def _capturer(self):
        if self._screen_capturer is None:
            try:
                from services.screen_capture import screen_capture_service
                self._screen_capturer = screen_capture_service
            except Exception:
                pass
        return self._screen_capturer

    @property
    def _ocr(self):
        if self._ocr_engine is None:
            try:
                from vision.ocr_pipeline import ocr_pipeline
                self._ocr_engine = ocr_pipeline
            except Exception:
                pass
        return self._ocr_engine

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        if self._initialized:
            return True
        if config:
            self._config.update(config)

        # Try to initialize OCR
        if self._ocr:
            self._ocr.initialize()

        self._initialized = True
        logger.info("VisionManager bridge initialized (delegating to new pipeline)")
        return True

    @property
    def is_available(self) -> bool:
        return self._initialized

    # ── Capture (delegates to screen_capture_service) ──────

    async def capture_screen(self, source=CaptureSource.FULL_SCREEN):
        try:
            from services.screen_capture import screen_capture_service
            if source == CaptureSource.ACTIVE_WINDOW:
                return await screen_capture_service.capture_active_window()
            elif source == CaptureSource.ACTIVE_MONITOR:
                return await screen_capture_service.capture_monitor(1)
            else:
                return await screen_capture_service.capture_fullscreen()
        except Exception as e:
            logger.warning("VisionManager capture failed: %s", e)
            return None

    async def capture_webcam(self):
        return None  # Webcam handled by auth subsystem

    async def capture_region(self, left: int, top: int, width: int, height: int):
        try:
            from services.screen_capture import screen_capture_service
            return await screen_capture_service.capture_region(left, top, width, height)
        except Exception:
            return None

    # ── OCR ────────────────────────────────────────────────

    async def ocr_screen(self, source=CaptureSource.ACTIVE_WINDOW) -> str:
        try:
            from services.vision_service import vision_service
            return await vision_service.ocr_only()
        except Exception as e:
            logger.debug("OCR failed: %s", e)
            return ""

    # ── Full analysis (delegates to vision_service) ────────

    async def analyze_screen(self, source=CaptureSource.FULL_SCREEN):
        try:
            from services.vision_service import vision_service
            ctx = await vision_service.force_analyze()
            # Convert to legacy VisionResult-like dict
            result = type('Result', (), {})()
            result.text = ctx.ocr_text
            result.raw_ocr = ctx.ocr_text
            result.scene_description = ctx.semantic_summary
            result.buttons = []
            result.error_messages = []
            result.latency_ms = ctx.total_time_ms
            result.confidence = 0.0
            return result
        except Exception as e:
            logger.warning("VisionManager analyze failed: %s", e)
            result = type('Result', (), {})()
            result.text = ""
            result.scene_description = ""
            result.latency_ms = 0.0
            return result

    async def analyze_webcam(self):
        result = type('Result', (), {})()
        result.text = "Webcam analysis not available"
        result.scene_description = ""
        result.latency_ms = 0.0
        return result

    async def answer_question_about_screen(self, question: str) -> str:
        try:
            from services.vision_service import vision_service
            ctx = await vision_service.force_analyze()
            if ctx.semantic_summary:
                return f"On screen: {ctx.semantic_summary}"
            return ctx.ocr_text or "I cannot answer that question about the screen."
        except Exception:
            return "I cannot answer that question about the screen."

    # ── Context-aware vision ──────────────────────────────

    def needs_vision(self, intent: str, text: str) -> Optional[CaptureSource]:
        text_lower = text.lower()
        screen_keywords = [
            "screen", "see", "look", "display", "monitor", "desktop",
            "open", "button", "menu", "dialog", "window", "tab",
            "error", "notification", "icon", "highlighted",
        ]
        if any(kw in text_lower for kw in screen_keywords):
            return CaptureSource.FULL_SCREEN
        return None

    # ── Configuration ─────────────────────────────────────

    def update_config(self, config: Dict[str, Any]) -> None:
        self._config.update(config)

    def invalidate_cache(self) -> None:
        try:
            from services.vision_service import vision_service
            vision_service.clear_cache()
        except Exception:
            pass

    # ── Diagnostics ───────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "screen_capture": "mss (via screen_capture_service)",
            "ocr_available": self._ocr.ready if self._ocr else False,
            "vision_model": "none (using OCR + heuristics)",
            "config": dict(self._config),
        }

    def close(self) -> None:
        if self._ocr:
            self._ocr.close()
        self._initialized = False
        logger.info("VisionManager bridge shut down")


# ═══════════════════════════════════════════════════════════════
# Global singletons
# ═══════════════════════════════════════════════════════════════

# Legacy singleton — delegates to new pipeline
vision_manager = VisionManagerBridge()

# New pipeline singletons
__all__ = [
    # Legacy
    "vision_manager",
    "VisionManagerBridge",
    "CaptureSource",
    "VisionModel",
    # New frame differencer
    "FrameDifferencer",
    "FrameDiffResult",
    "GatingDecision",
    "MotionRegion",
    "frame_differencer",
    # New layout analyzer
    "LayoutAnalyzer",
    "ApplicationType",
    "WindowLayout",
    "RegionType",
    "LayoutRegion",
    "layout_analyzer",
    # New OCR pipeline
    "EnhancedOCREngine",
    "OCRBox",
    "OCRResult",
    "TextClass",
    "ocr_pipeline",
    # New screen memory
    "ScreenMemory",
    "ScreenSnapshot",
    "screen_memory",
    # New action verifier
    "ActionVerifier",
    "VerificationResult",
    "VerificationStatus",
    "action_verifier",
    # Perception
    "PerceptionService",
    "perception_service",
]