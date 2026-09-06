"""
Regression test: NameError in VisionService.analyze_forensic().

`analyze_forensic()` referenced `verify_previous_action` (a parameter of
`analyze()`) without defining it, crashing both `Diego.py debug vision` and
`Diego.py inspect screen` at forensic stage 9.

The fix adds `verify_previous_action: bool = False` to the
`analyze_forensic()` signature, mirroring `analyze()`. These tests verify:

1. `analyze_forensic()` completes without NameError (default call).
2. With `verify_previous_action=False` (default), stage 9 is logged and
   skipped — no verification is attached to the context.
3. With `verify_previous_action=True`, stage 9 runs and attaches a
   VerificationResult, preserving the intended forensic behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import services.vision_service as vs
from services.vision_service import VisionService
from services.screen_capture import CaptureResult


@pytest.fixture
def service(monkeypatch):
    """A VisionService with lightweight stubs so the forensic pipeline
    runs headless (no real capture / OCR / layout / memory writes)."""
    svc = VisionService()
    svc.use_layout = False
    svc.use_ocr = False
    svc.use_memory = False

    img = np.zeros((64, 96, 3), dtype=np.uint8)
    cap = CaptureResult(image=img, width=96, height=64,
                        source_label="window:TestWindow")

    async def fake_capture_active_window():
        return cap

    monkeypatch.setattr(
        vs.screen_capture_service,
        "capture_active_window",
        fake_capture_active_window,
    )
    return svc


@pytest.mark.asyncio
async def test_analyze_forensic_no_name_error_on_default_call(service):
    """Default call must complete past stage 9 without NameError."""
    ctx, report = await service.analyze_forensic(force=True)

    assert ctx.error == ""
    assert ctx.capture is not None
    # All 10 forensic stages were logged (stage 9 included)
    stage_numbers = [s.stage for s in report.stages]
    assert 9 in stage_numbers
    # Default: verification not requested -> no verification attached
    assert ctx.verification is None
    assert "verify" not in ctx.stages_run


@pytest.mark.asyncio
async def test_analyze_forensic_verify_previous_action_runs_stage9(
    service, monkeypatch
):
    """verify_previous_action=True must run stage 9 and attach a result,
    preserving the intended forensic/action-verification behavior."""
    # Simulate a recorded action whose UI change was detected, so the
    # stage-9 success path is exercised (a no-change result intentionally
    # logs stage 9 as a forensic failure, not an error).
    monkeypatch.setattr(
        vs.screen_memory,
        "last_action_changed_ui",
        lambda: (True, "UI changed after action"),
    )

    ctx, report = await service.analyze_forensic(
        force=True, verify_previous_action=True
    )

    assert ctx.error == ""
    assert ctx.verification is not None
    assert "verify" in ctx.stages_run
    stage9 = [s for s in report.stages if s.stage == 9]
    assert stage9, "Stage 9 (Action Verification) must be present in report"
    assert "verify_previous=True" in stage9[0].input_summary
    assert stage9[0].success, (
        "Stage 9 must complete successfully when verification is requested"
    )
