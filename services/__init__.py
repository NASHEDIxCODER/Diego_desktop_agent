"""
Leo Desktop Assistant — Services Layer

New production-grade subsystems:

    SearchService      — web search (DuckDuckGo, Tavily, Playwright, trafilatura)
    ScreenCapture      — fast desktop capture (mss) with frame differencing
    VisionService      — screen understanding (OCR, UI detection, structured tree)
    UITree             — structured UI element hierarchy (data model)

All services extend BaseService (core/service.py) and are registered in the
global ServiceRegistry. The conversation engine accesses them through their
public interfaces only — no direct imports of internal modules.
"""

from services.search_service import SearchService
from services.screen_capture import ScreenCapture
from services.ui_tree import UIElement, UIWindow, UIDesktop, ElementType
from services.vision_service import VisionService

__all__ = [
    "SearchService",
    "ScreenCapture",
    "UIElement",
    "UIWindow",
    "UIDesktop",
    "ElementType",
    "VisionService",
]
