import logging
import sys

sys.path.insert(0, "/home/nashedi_x_coder/Workspace/PycharmProjects/Diego_desktop_agent")
logging.basicConfig(level=logging.INFO)

from voice.calibrate_wake import _select_base_model
from voice.settings import voice_settings
from openwakeword import Model as OWWModel

# Test 1: _select_base_model resolves a model
r = _select_base_model(voice_settings.wake_phrase)
print("TEST 1 RESULT:", r)
assert r is not None, "FAIL: _select_base_model returned None"
assert r.exists(), "FAIL: model path does not exist: %s" % r
print("  PASS: base model =", r)

# Test 2: OWWModel can be constructed with onnx inference framework
print("\nTEST 2: Constructing OWWModel with inference_framework='onnx'...")
# inference_framework is an explicit named parameter of Model.__init__
# (forwarded to AudioFeatures.__init__), NOT part of **kwargs, so it does
# not cause a TypeError. We must pass it explicitly for .onnx models.
inference_framework = "onnx" if r.suffix == ".onnx" else "tflite"
oww = OWWModel(
    wakeword_models=[str(r)],
    inference_framework=inference_framework,
)
base_stem = r.stem
feats_ndx = oww.model_inputs[base_stem]
print("  PASS: model constructed, feats_ndx =", feats_ndx)

# Test 3: WakeModelManager can resolve a model path
print("\nTEST 3: WakeModelManager._resolve_model_path()...")
from voice.wake_model_manager import wake_model_manager
resolved = wake_model_manager._resolve_model_path()
print("  RESOLVED:", resolved)
assert resolved is not None, "FAIL: _resolve_model_path returned None"
print("  PASS")

print("\nALL TESTS PASSED")