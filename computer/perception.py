"""Phase 22: perception hierarchy constants (additive, mirrors existing pipeline).

Order (highest confidence first; NEVER blind coordinate guesses):
  1. browser/dom  2. accessibility  3. native window/app state
  4. ocr/text  5. visual element detection  6. coordinate fallback (justified only)
"""
from __future__ import annotations
from enum import Enum


class PerceptionMethod(str, Enum):
    BROWSER_DOM = "browser_dom"
    ACCESSIBILITY = "accessibility"
    NATIVE_WINDOW = "native_window"
    OCR = "ocr"
    VISUAL = "visual"
    COORDINATE = "coordinate"


PERCEPTION_ORDER = (
    PerceptionMethod.BROWSER_DOM,
    PerceptionMethod.ACCESSIBILITY,
    PerceptionMethod.NATIVE_WINDOW,
    PerceptionMethod.OCR,
    PerceptionMethod.VISUAL,
    PerceptionMethod.COORDINATE,
)

METHOD_RANK = {m: i for i, m in enumerate(PERCEPTION_ORDER)}


def rank(method: str) -> int:
    try:
        return METHOD_RANK[PerceptionMethod(str(method))]
    except ValueError:
        return len(PERCEPTION_ORDER)
