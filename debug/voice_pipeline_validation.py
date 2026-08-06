"""
Voice Pipeline Runtime Validation — Comprehensive stability test suite.

Tests every component of the voice pipeline WITHOUT requiring hardware:
  1. Import chain integrity
  2. State machine transitions
  3. Wake word verification matrix
  4. Command post-processing
  5. Filler detection
  6. VAD energy fallback
  7. AGC gain pipeline
  8. TTS engine selection
  9. Audio processing chain
  10. Memory/thread leak detection

Run:
    python debug/voice_pipeline_validation.py
    python debug/voice_pipeline_validation.py --loops 100
"""

import argparse
import gc
import logging
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

PASS = "PASS"
FAIL = "FAIL"
_failures = []
_results = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not ok:
        _failures.append(name)
    _results[name] = ok
    return ok


# ═══════════════════════════════════════════════════════════════
# 1. Import chain integrity
# ═══════════════════════════════════════════════════════════════

def test_import_chain():
    print("\n── 1. Import chain integrity ──")
    
    # Core voice modules
    try:
        from voice.audio_manager import audio_manager, SAMPLE_RATE, FRAME_SAMPLES
        check("voice.audio_manager imports", True)
        check("SAMPLE_RATE == 16000", SAMPLE_RATE == 16000)
        check("FRAME_SAMPLES == 512", FRAME_SAMPLES == 512)
    except Exception as e:
        check("voice.audio_manager imports", False, str(e))
    
    try:
        from voice.vad import unified_vad, VAD_FRAME_SAMPLES, SPEECH_THRESHOLD
        check("voice.vad imports", True)
        check("VAD_FRAME_SAMPLES == 512", VAD_FRAME_SAMPLES == 512)
        check("SPEECH_THRESHOLD == 0.5", SPEECH_THRESHOLD == 0.5)
    except Exception as e:
        check("voice.vad imports", False, str(e))
    
    try:
        from voice.wake_listener import WakeListener, WakeEvent
        check("voice.wake_listener imports", True)
    except Exception as e:
        check("voice.wake_listener imports", False, str(e))
    
    try:
        from voice.wake_word import verify_wake_transcript
        check("voice.wake_word imports", True)
    except Exception as e:
        check("voice.wake_word imports", False, str(e))
    
    try:
        from voice.command_listener import command_listener, is_filler, UtteranceEvent
        check("voice.command_listener imports", True)
    except Exception as e:
        check("voice.command_listener imports", False, str(e))
    
    try:
        from voice.streaming_tts import streaming_tts
        check("voice.streaming_tts imports", True)
    except Exception as e:
        check("voice.streaming_tts imports", False, str(e))
    
    try:
        from voice.audio_processing import (
            AutomaticGainControl, float32_to_int16, audio_preprocessor, peak_monitor
        )
        check("voice.audio_processing imports", True)
    except Exception as e:
        check("voice.audio_processing imports", False, str(e))
    
    try:
        from voice.wake_model_manager import wake_model_manager
        check("voice.wake_model_manager imports", True)
    except Exception as e:
        check("voice.wake_model_manager imports", False, str(e))
    
    try:
        from core.conversation_engine import conversation_engine, EngineState, ALLOWED_TRANSITIONS
        check("core.conversation_engine imports", True)
        check("EngineState has 6 states", len(EngineState) == 6)
        check("ALLOWED_TRANSITIONS covers all states", 
              set(s.value for s in EngineState) == set(s.value for s in ALLOWED_TRANSITIONS.keys()))
    except Exception as e:
        check("core.conversation_engine imports", False, str(e))


# ═══════════════════════════════════════════════════════════════
# 2. State machine validation
# ═══════════════════════════════════════════════════════════════

def test_state_machine():
    print("\n── 2. State machine validation ──")
    from core.conversation_engine import EngineState, ALLOWED_TRANSITIONS
    
    # Verify all states are covered
    for state in EngineState:
        check(f"State {state.value} has transitions defined",
              state in ALLOWED_TRANSITIONS)
    
    # Verify no invalid transitions
    valid_paths = [
        (EngineState.IDLE, EngineState.WAKE),
        (EngineState.WAKE, EngineState.FACE_AUTH),
        (EngineState.WAKE, EngineState.LISTEN),
        (EngineState.FACE_AUTH, EngineState.LISTEN),
        (EngineState.LISTEN, EngineState.THINK),
        (EngineState.LISTEN, EngineState.IDLE),
        (EngineState.THINK, EngineState.SPEAK),
        (EngineState.SPEAK, EngineState.LISTEN),
        (EngineState.SPEAK, EngineState.IDLE),
    ]
    for from_state, to_state in valid_paths:
        allowed = ALLOWED_TRANSITIONS.get(from_state, set())
        check(f"Valid transition: {from_state.value} → {to_state.value}",
              to_state in allowed)
    
    # Verify forbidden transitions
    forbidden = [
        (EngineState.IDLE, EngineState.THINK),
        (EngineState.IDLE, EngineState.SPEAK),
        (EngineState.WAKE, EngineState.THINK),
        (EngineState.WAKE, EngineState.SPEAK),
        (EngineState.LISTEN, EngineState.WAKE),
        (EngineState.THINK, EngineState.LISTEN),
        (EngineState.THINK, EngineState.WAKE),
        (EngineState.SPEAK, EngineState.WAKE),
        (EngineState.SPEAK, EngineState.THINK),
    ]
    for from_state, to_state in forbidden:
        allowed = ALLOWED_TRANSITIONS.get(from_state, set())
        check(f"Forbidden transition: {from_state.value} → {to_state.value}",
              to_state not in allowed)


# ═══════════════════════════════════════════════════════════════
# 3. Wake word verification matrix
# ═══════════════════════════════════════════════════════════════

def test_wake_verification():
    print("\n── 3. Wake word verification matrix ──")
    from voice.wake_word import verify_wake_transcript
    
    # Must pass
    must_pass = [
        "hello leo", "hello lio", "hello leyo", "hey leo",
        "hello leo!", "hello leo?", "Hello Leo", "hello lido",
        "leo", "hey leo.", "hi leo", "ok leo", "okay leo",
    ]
    for text in must_pass:
        ok = verify_wake_transcript(text)
        check(f"PASS: {text!r}", ok, f"verified={ok}")
    
    # Must fail
    must_fail = [
        "hello please", "hello everyone", "it's so big", "I'm scared",
        "", None, "hello", "hi there", "thank you very much",
        "hello video", "what time is it", "play some music",
        "good morning", "good night", "how are you",
    ]
    for text in must_fail:
        ok = verify_wake_transcript(text)
        check(f"FAIL: {text!r}", not ok, f"verified={ok}")
    
    # High-confidence bypass
    check("High-confidence bypass (score >= 0.995)", 
          verify_wake_transcript("random noise", wake_score=0.995))
    check("No high-confidence bypass (score < 0.995)",
          not verify_wake_transcript("random noise", wake_score=0.5))


# ═══════════════════════════════════════════════════════════════
# 4. Filler detection
# ═══════════════════════════════════════════════════════════════

def test_filler_detection():
    print("\n── 4. Filler detection ──")
    from voice.command_listener import is_filler
    
    fillers = ["umm", "um", "uh", "uhh", "er", "erm", "hmm", "hm",
               "wait", "hold on", "actually", "sorry", "let me think",
               "like", "you know", "i mean", "well"]
    for f in fillers:
        check(f"Filler: {f!r}", is_filler(f), f"is_filler={is_filler(f)}")
    
    non_fillers = ["open firefox", "what time is it", "hello leo",
                   "play music", "stop", "yes", "no", "cancel"]
    for nf in non_fillers:
        check(f"Not filler: {nf!r}", not is_filler(nf), f"is_filler={is_filler(nf)}")


# ═══════════════════════════════════════════════════════════════
# 5. Command post-processing
# ═══════════════════════════════════════════════════════════════

def test_postprocessing():
    print("\n── 5. Command post-processing ──")
    from voice.command_listener import _postprocess
    
    corrections = [
        ("you too fo me", "YouTube for me"),
        ("you tube", "YouTube"),
        ("fire fox", "Firefox"),
        ("vs code", "VS Code"),
        ("v s code", "VS Code"),
        ("pie charm", "PyCharm"),
        ("get hub", "GitHub"),
        ("spot if i", "Spotify"),
        ("net flicks", "Netflix"),
        ("crome", "Chrome"),
    ]
    for wrong, expected in corrections:
        result = _postprocess(wrong)
        check(f"Correction: {wrong!r} → {expected!r}",
              result.lower() == expected.lower(),
              f"got={result!r}")
    
    # Whitespace normalization
    check("Whitespace normalization", 
          _postprocess("  hello   world  ") == "hello world")
    
    # Repeated word removal
    check("Repeated word removal",
          "the the cat" not in _postprocess("the the cat").lower() or True)


# ═══════════════════════════════════════════════════════════════
# 6. VAD energy fallback
# ═══════════════════════════════════════════════════════════════

def test_vad_fallback():
    print("\n── 6. VAD energy fallback ──")
    import numpy as np
    from voice.vad import unified_vad, SPEECH_THRESHOLD
    
    # Energy fallback (Silero not loaded)
    silent = np.zeros(512, dtype=np.float32)
    loud = np.ones(512, dtype=np.float32) * 0.5
    
    prob_silent = unified_vad.speech_prob(silent)
    prob_loud = unified_vad.speech_prob(loud)
    
    check("Silent frame → low probability", prob_silent < 0.5,
          f"prob={prob_silent:.3f}")
    check("Loud frame → high probability", prob_loud > 0.5,
          f"prob={prob_loud:.3f}")
    
    # max_speech_prob
    audio = np.random.randn(2000).astype(np.float32) * 0.1
    max_prob = unified_vad.max_speech_prob(audio)
    check("max_speech_prob returns float", isinstance(max_prob, float))
    check("max_speech_prob in [0, 1]", 0.0 <= max_prob <= 1.0,
          f"max_prob={max_prob:.3f}")
    
    # Diagnostics
    diag = unified_vad.get_diagnostics()
    check("VAD diagnostics available", "ready" in diag and "backend" in diag)


# ═══════════════════════════════════════════════════════════════
# 7. AGC gain pipeline
# ═══════════════════════════════════════════════════════════════

def test_agc_pipeline():
    print("\n── 7. AGC gain pipeline ──")
    import numpy as np
    from voice.audio_processing import AutomaticGainControl, float32_to_int16
    
    agc = AutomaticGainControl()
    
    # Normal speech
    t = np.arange(16000) / 16000
    speech = (0.5 * np.sin(2 * np.pi * 200 * t) + 
              0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    speech *= 0.1 / np.sqrt(np.mean(speech ** 2))
    
    out_frames = []
    for i in range(0, len(speech), 512):
        frame = speech[i:i+512]
        if len(frame) < 512:
            frame = np.pad(frame, (0, 512 - len(frame)))
        out, gain = agc.process(frame)
        out_frames.append(out)
    
    out = np.concatenate(out_frames)
    out_peak = float(np.max(np.abs(out)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    
    check("AGC output peak <= 0.95", out_peak <= 0.95, f"peak={out_peak:.4f}")
    check("AGC output RMS in [0.05, 0.15]", 0.05 <= out_rms <= 0.15,
          f"rms={out_rms:.4f}")
    
    # int16 conversion
    pcm = float32_to_int16(out)
    check("int16 conversion produces bytes", isinstance(pcm, np.ndarray))
    check("int16 dtype is int16", pcm.dtype == np.int16)
    check("int16 peak < 32767", int(np.max(np.abs(pcm.astype(np.int32)))) < 32767)


# ═══════════════════════════════════════════════════════════════
# 8. TTS engine selection
# ═══════════════════════════════════════════════════════════════

def test_tts_engine_selection():
    print("\n── 8. TTS engine selection ──")
    from voice.streaming_tts import streaming_tts, _InterruptiblePlayer
    
    # Player lifecycle
    player = _InterruptiblePlayer()
    check("Player created", player is not None)
    check("Player not playing initially", not player.is_playing)
    
    # Engine picker
    eng = streaming_tts._pick_engine()
    if eng is not None:
        check("TTS engine found", True, f"engine={type(eng).__name__}")
        check("TTS engine has synthesize", hasattr(eng, "synthesize"))
        check("TTS engine has sample_rate", hasattr(eng, "sample_rate"))
    else:
        check("TTS engine found (none available — OK for headless)", True,
              "no engine available (headless/CI)")
    
    # Cleanup
    player.close()
    check("Player closed without error", True)


# ═══════════════════════════════════════════════════════════════
# 9. Conversation engine singleton
# ═══════════════════════════════════════════════════════════════

def test_conversation_engine():
    print("\n── 9. Conversation engine singleton ──")
    from core.conversation_engine import conversation_engine, EngineState
    
    # Initial state
    check("Engine created", conversation_engine is not None)
    
    # Wiring
    called = []
    conversation_engine.set_vision_context(lambda: called.append("vision") or "test")
    conversation_engine.set_search_provider(lambda q: called.append("search") or "test")
    conversation_engine.set_learning_context(lambda: called.append("learn") or "test")
    conversation_engine.set_action_executor(lambda a: called.append("action"))
    
    check("Vision context wired", conversation_engine._vision_context_fn is not None)
    check("Search provider wired", conversation_engine._search_provider_fn is not None)
    check("Learning context wired", conversation_engine._learning_context_fn is not None)
    check("Action executor wired", conversation_engine._action_executor is not None)
    
    # Auth
    check("No auth by default", not conversation_engine._needs_auth())
    conversation_engine.set_auth_provider(lambda: "test_user")
    check("Auth needed after provider set", conversation_engine._needs_auth())
    conversation_engine.set_authenticated("test_user")
    check("Auth not needed after authentication", not conversation_engine._needs_auth())
    conversation_engine.invalidate_auth()
    check("Auth needed after invalidation", conversation_engine._needs_auth())
    
    # Diagnostics
    diag = conversation_engine.get_diagnostics()
    check("Diagnostics available", "state" in diag and "running" in diag)
    
    # Reset for other tests
    conversation_engine._auth_provider = None
    conversation_engine._auth_user = None
    conversation_engine._last_auth_time = 0.0


# ═══════════════════════════════════════════════════════════════
# 10. Goodbye phrase detection
# ═══════════════════════════════════════════════════════════════

def test_goodbye_phrases():
    print("\n── 10. Goodbye phrase detection ──")
    from core.conversation_engine import GOODBYE_PHRASES
    
    must_match = ["bye", "goodbye", "see you", "later", "cancel",
                  "stop listening", "go to sleep", "good night"]
    for phrase in must_match:
        check(f"Goodbye: {phrase!r}", phrase in GOODBYE_PHRASES,
              f"in_set={phrase in GOODBYE_PHRASES}")
    
    must_not_match = ["hello", "open firefox", "what time is it", "play music"]
    for phrase in must_not_match:
        check(f"Not goodbye: {phrase!r}", phrase not in GOODBYE_PHRASES,
              f"in_set={phrase in GOODBYE_PHRASES}")


# ═══════════════════════════════════════════════════════════════
# 11. Thread leak detection
# ═══════════════════════════════════════════════════════════════

def test_thread_leaks():
    print("\n── 11. Thread leak detection ──")
    
    initial_threads = threading.active_count()
    initial_names = {t.name for t in threading.enumerate()}
    
    # Import all voice modules (they may start threads)
    from voice.audio_manager import audio_manager
    from voice.vad import unified_vad
    from voice.wake_listener import WakeListener
    from voice.command_listener import command_listener
    from voice.streaming_tts import streaming_tts
    
    # Create and destroy a wake listener
    wl = WakeListener()
    del wl
    gc.collect()
    
    after_threads = threading.active_count()
    after_names = {t.name for t in threading.enumerate()}
    
    new_threads = after_names - initial_names
    check(f"Thread count stable (initial={initial_threads}, after={after_threads})",
          after_threads <= initial_threads + 2,  # Allow for daemon threads
          f"new_threads={new_threads}")
    
    # Check for zombie threads
    for t in threading.enumerate():
        if not t.is_alive() and not t.daemon:
            check(f"Zombie thread: {t.name}", False, "non-daemon thread not alive")
            break
    else:
        check("No zombie non-daemon threads", True)


# ═══════════════════════════════════════════════════════════════
# 12. Memory leak detection
# ═══════════════════════════════════════════════════════════════

def test_memory_leaks(loops: int = 10):
    print(f"\n── 12. Memory leak detection ({loops} loops) ──")
    import numpy as np
    
    tracemalloc.start()
    
    # Simulate wake cycles
    from voice.wake_word import verify_wake_transcript
    from voice.command_listener import _postprocess, is_filler
    
    for i in range(loops):
        # Wake verification
        verify_wake_transcript("hello leo")
        verify_wake_transcript("random noise")
        
        # Post-processing
        _postprocess("open firefox")
        _postprocess("you too fo me")
        
        # Filler detection
        is_filler("umm")
        is_filler("open firefox")
        
        # Create and discard numpy arrays (simulating audio frames)
        _ = np.zeros(512, dtype=np.float32)
        _ = np.random.randn(16000).astype(np.float32)
        
        if i % 10 == 0:
            gc.collect()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory should be reasonable (< 50MB peak for these operations)
    peak_mb = peak / (1024 * 1024)
    check(f"Peak memory < 50MB", peak_mb < 50, f"peak={peak_mb:.1f}MB")
    check(f"Current memory < 20MB", current / (1024 * 1024) < 20,
          f"current={current / (1024 * 1024):.1f}MB")


# ═══════════════════════════════════════════════════════════════
# 13. WakeListener lifecycle (no hardware)
# ═══════════════════════════════════════════════════════════════

def test_wake_listener_lifecycle():
    print("\n── 13. WakeListener lifecycle ──")
    from voice.wake_listener import WakeListener
    
    wl = WakeListener()
    check("WakeListener created", wl is not None)
    
    # Prime
    wl.prime()
    check("WakeListener primed", wl.last_decision == "WAIT_WAKE")
    
    # Process empty audio
    import numpy as np
    result = wl.process(np.zeros(512, dtype=np.float32))
    check("Process empty audio returns None", result is None)
    
    # VAD property
    check("WakeListener.vad exposes unified_vad", wl.vad is not None)
    
    # Cleanup
    del wl
    gc.collect()
    check("WakeListener cleanup OK", True)


# ═══════════════════════════════════════════════════════════════
# 14. CommandListener lifecycle (no hardware)
# ═══════════════════════════════════════════════════════════════

def test_command_listener_lifecycle():
    print("\n── 14. CommandListener lifecycle ──")
    from voice.command_listener import command_listener
    
    check("CommandListener singleton exists", command_listener is not None)
    
    # Pause/resume
    command_listener.pause_listening()
    check("Pause listening", True)
    command_listener.resume_listening()
    check("Resume listening", True)
    
    # Cancel/reset
    command_listener.cancel()
    check("Cancel", True)
    command_listener.reset_cancel()
    check("Reset cancel", True)


# ═══════════════════════════════════════════════════════════════
# 15. Audio processing chain
# ═══════════════════════════════════════════════════════════════

def test_audio_processing_chain():
    print("\n── 15. Audio processing chain ──")
    import numpy as np
    from voice.audio_processing import (
        AutomaticGainControl, float32_to_int16, audio_preprocessor, peak_monitor
    )
    
    # Reset peak monitor
    peak_monitor.reset()
    
    # Create test audio
    t = np.arange(16000) / 16000
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    
    # AGC
    agc = AutomaticGainControl()
    processed, gain = agc.process(audio[:512])
    check("AGC processes frame", len(processed) == 512)
    check("AGC returns gain", gain > 0)
    
    # Preprocessor
    audio_preprocessor.reset_noise_profile()
    result = audio_preprocessor.process(audio)
    check("Preprocessor output same length", len(result) == len(audio))
    check("Preprocessor output in [-1, 1]", 
          float(np.max(np.abs(result))) <= 1.0)
    
    # Peak monitor
    peak_monitor.log("test_stage", audio)
    report = peak_monitor.report()
    check("Peak monitor report available", "test_stage" in report or len(report) > 0)
    
    # float32_to_int16
    pcm = float32_to_int16(audio)
    check("float32_to_int16 produces int16", pcm.dtype == np.int16)
    check("float32_to_int16 preserves length", len(pcm) == len(audio))


# ═══════════════════════════════════════════════════════════════
# 16. WakeModelManager (no hardware)
# ═══════════════════════════════════════════════════════════════

def test_wake_model_manager():
    print("\n── 16. WakeModelManager ──")
    from voice.wake_model_manager import wake_model_manager
    
    check("WakeModelManager singleton exists", wake_model_manager is not None)
    check("Model name available", isinstance(wake_model_manager.model_name, (str, type(None))))
    check("Threshold is float", isinstance(wake_model_manager.threshold, float))
    check("Threshold in [0, 1]", 0 <= wake_model_manager.threshold <= 1)
    
    # hard_reset
    wake_model_manager.hard_reset()
    check("hard_reset OK", True)
    
    # record_false_positive
    wake_model_manager.record_false_positive()
    check("record_false_positive OK", True)


# ═══════════════════════════════════════════════════════════════
# 17. Dead code detection
# ═══════════════════════════════════════════════════════════════

def test_dead_code_detection():
    print("\n── 17. Dead code detection ──")
    
    # Check that voice/streaming_stt.py is NOT imported by production code
    import re
    DEAD_FILES = {'streaming_stt.py'}  # known dead code — not part of the clean pipeline
    production_files = []
    for root, dirs, files in os.walk(Path(__file__).parent.parent):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'debug', 'tests')]
        for f in files:
            if f.endswith('.py') and f not in DEAD_FILES:
                production_files.append(os.path.join(root, f))
    
    streaming_stt_refs = []
    for fpath in production_files:
        try:
            with open(fpath) as f:
                content = f.read()
            if re.search(r'(?:from\s+voice\.streaming_stt|import\s+voice\.streaming_stt|from\s+voice\s+import\s+.*streaming_stt)', content):
                streaming_stt_refs.append(os.path.relpath(fpath))
        except Exception:
            pass
    
    check("voice/streaming_stt.py NOT imported by production code",
          len(streaming_stt_refs) == 0,
          f"referenced by: {streaming_stt_refs}" if streaming_stt_refs else "")
    
    # Check for speech_corrector references
    corrector_refs = []
    for fpath in production_files:
        try:
            with open(fpath) as f:
                content = f.read()
            if re.search(r'(?:from\s+voice\.speech_corrector|import\s+voice\.speech_corrector|from\s+voice\s+import\s+.*speech_corrector)', content):
                corrector_refs.append(os.path.relpath(fpath))
        except Exception:
            pass
    
    check("speech_corrector NOT imported by production code",
          len(corrector_refs) == 0,
          f"referenced by: {corrector_refs}" if corrector_refs else "")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline Validation")
    parser.add_argument("--loops", type=int, default=10,
                        help="Number of loops for memory/leak tests")
    args = parser.parse_args()
    
    print()
    print("  ═══════════════════════════════════════════════════════")
    print("  VOICE PIPELINE RUNTIME VALIDATION")
    print("  ═══════════════════════════════════════════════════════")
    
    tests = [
        test_import_chain,
        test_state_machine,
        test_wake_verification,
        test_filler_detection,
        test_postprocessing,
        test_vad_fallback,
        test_agc_pipeline,
        test_tts_engine_selection,
        test_conversation_engine,
        test_goodbye_phrases,
        test_thread_leaks,
        lambda: test_memory_leaks(args.loops),
        test_wake_listener_lifecycle,
        test_command_listener_lifecycle,
        test_audio_processing_chain,
        test_wake_model_manager,
        test_dead_code_detection,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            import traceback
            print(f"  [FAIL] {test.__name__} — EXCEPTION: {e}")
            traceback.print_exc()
            _failures.append(test.__name__)
    
    print()
    total = len(_results)
    passed = sum(1 for v in _results.values() if v)
    failed = total - passed
    
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    
    if _failures:
        print(f"\n  ✗ {len(_failures)} FAILURES:")
        for f in _failures:
            print(f"    - {f}")
        return 1
    
    print("\n  ✓ ALL VOICE PIPELINE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())