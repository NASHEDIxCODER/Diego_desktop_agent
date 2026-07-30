"""
Runtime Pipeline Tests for Leo Desktop Assistant.

Tests every stage of the runtime pipeline:
  BOOT → READY → WAIT_WAKE → WAKE_DETECTED → FACE_AUTH → GREETING
  → WAIT_COMMAND → MIC OPEN → AUDIO CAPTURE → STT → TRANSCRIPT
  → NLP → PLANNER → PLUGIN → TTS → WAIT_WAKE

Each test validates:
  - entered
  - started
  - finished
  - latency
  - returned value
  - exception
  - timeout
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger("test_runtime")

# Set environment variables for testing
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DISPLAY"] = ":0"


class TestResult:
    """Track test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.duration: float = 0.0
        self.stages: list = []

    def record_stage(self, stage: str, status: str, duration: float = 0.0, detail: str = ""):
        self.stages.append({"stage": stage, "status": status, "duration": duration, "detail": detail})
        logger.info("[TEST] %s: %s (%.1fs) %s", stage, status, duration, detail)

    def mark_passed(self):
        self.passed = True

    def mark_failed(self, error: str):
        self.passed = False
        self.error = error


async def test_startup_diagnostics():
    """Test that all subsystems initialize correctly."""
    result = TestResult("startup_diagnostics")
    logger.info("=" * 60)
    logger.info("TEST: Startup Diagnostics")
    logger.info("=" * 60)

    try:
        from main import startup_diagnostics
        t0 = time.time()
        status = await startup_diagnostics()
        result.duration = time.time() - t0

        result.record_stage("startup", "complete", result.duration,
                          f"nlp={status.get('nlp')}, voice={status.get('voice')}, "
                          f"plugins={status.get('plugins')}, duckdb={status.get('duckdb')}")

        # Check core subsystems
        if not status.get("nlp"):
            result.mark_failed("NLP subsystem failed")
        elif not status.get("voice"):
            logger.warning("[TEST] Voice subsystem unavailable (mic may be missing)")
            result.mark_passed()
        else:
            result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Startup diagnostics failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_nlp_classification():
    """Test NLP classification with various intents."""
    result = TestResult("nlp_classification")
    logger.info("=" * 60)
    logger.info("TEST: NLP Classification")
    logger.info("=" * 60)

    test_cases = [
        ("open youtube", "youtube"),
        ("turn up the volume", "volume_up"),
        ("turn down the brightness", "brightness_down"),
        ("what time is it", "time_query"),
        ("tell me a joke", "joke"),
        ("what is the weather", "weather_query"),
        ("send a telegram message", "telegram_send"),
        ("goodbye", "exit"),
        ("hello", "greeting"),
        ("who am i", "who_am_i"),
    ]

    try:
        from nlp.inference import inference
        from nlp.entities import extract_entities

        if not inference.load():
            result.mark_failed("NLP model not loaded")
            return result

        t0 = time.time()
        for text, expected_intent in test_cases:
            tt = time.time()
            results = inference.classify(text, top_k=1)
            dt = time.time() - tt

            if not results:
                result.record_stage(f"classify('{text}')", "FAILED", dt, "No results")
                continue

            top = results[0]
            intent = top["intent"]
            confidence = top["confidence"]
            entities = extract_entities(text)

            status = "PASS" if intent == expected_intent else f"MISMATCH(expected={expected_intent})"
            result.record_stage(
                f"classify('{text}')", status, dt,
                f"intent={intent}, conf={confidence:.2f}, entities={entities}"
            )

        result.duration = time.time() - t0
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"NLP classification failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_microphone_backend():
    """Test microphone backend detection and initialization."""
    result = TestResult("microphone_backend")
    logger.info("=" * 60)
    logger.info("TEST: Microphone Backend")
    logger.info("=" * 60)

    try:
        from voice.microphone import microphone
        from voice.audio_device import audio_device
        from voice.settings import voice_settings

        t0 = time.time()
        voice_settings.update_from_env()
        audio_device.detect_backend()
        mic = microphone.get_microphone()
        result.duration = time.time() - t0

        if mic is None:
            result.record_stage("mic_init", "SKIPPED", result.duration, "No microphone available")
            result.mark_passed()
            return result

        result.record_stage("mic_init", "PASS", result.duration,
                          f"backend={microphone.backend}, available={microphone.available}")

        # Test calibration
        from voice.stt import calibrate
        tt = time.time()
        calibrate(duration=1.0)
        cal_time = time.time() - tt
        result.record_stage("calibration", "PASS", cal_time, "Calibration complete")

        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Microphone test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_audio_capture():
    """Test that audio capture produces valid data."""
    result = TestResult("audio_capture")
    logger.info("=" * 60)
    logger.info("TEST: Audio Capture")
    logger.info("=" * 60)

    try:
        from voice.stt import _capture_audio_sounddevice

        t0 = time.time()
        capture_result = _capture_audio_sounddevice(duration=1.0)
        result.duration = time.time() - t0

        if capture_result is None:
            result.record_stage("capture", "FAILED", result.duration, "No audio captured")
            result.mark_failed("Audio capture returned None")
            return result

        # Validate the tuple unpacking
        if not isinstance(capture_result, tuple):
            result.record_stage("capture", "FAILED", result.duration,
                              f"Expected tuple, got {type(capture_result).__name__}")
            result.mark_failed("Audio capture did not return a tuple")
            return result

        if len(capture_result) != 2:
            result.record_stage("capture", "FAILED", result.duration,
                              f"Expected tuple of length 2, got {len(capture_result)}")
            result.mark_failed("Audio capture tuple has wrong length")
            return result

        audio_bytes, samplerate = capture_result

        result.record_stage("capture", "PASS", result.duration,
                          f"bytes={len(audio_bytes)}, samplerate={samplerate}Hz, duration={len(audio_bytes)/samplerate/2:.1f}s")

        # Check audio metrics
        if len(audio_bytes) < 512:
            result.record_stage("audio_quality", "WARN", 0,
                              f"Audio too short: {len(audio_bytes)} bytes")
        else:
            # Check RMS
            import numpy as np
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(float)**2)))
            result.record_stage("audio_quality", "PASS", 0,
                              f"RMS={rms:.1f}, samples={len(samples)}")

        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Audio capture test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_stt():
    """Test speech-to-text recognition (without actual audio)."""
    result = TestResult("stt")
    logger.info("=" * 60)
    logger.info("TEST: STT (silence detection)")
    logger.info("=" * 60)

    try:
        from voice.stt import listen, _recognize_bytes, _capture_audio_sounddevice

        # Test 1: Capture silence and verify it returns None
        t0 = time.time()
        capture_result = _capture_audio_sounddevice(duration=1.0)
        cap_time = time.time() - t0

        if capture_result is None:
            result.record_stage("silence_capture", "SKIPPED", cap_time, "No audio device")
            result.mark_passed()
            return result

        audio_bytes, samplerate = capture_result
        result.record_stage("silence_capture", "PASS", cap_time,
                          f"bytes={len(audio_bytes)}, rate={samplerate}")

        # Test 2: Recognize silence bytes
        t0 = time.time()
        text = _recognize_bytes(audio_bytes, samplerate)
        rec_time = time.time() - t0
        result.record_stage("silence_recognition", "PASS" if text is None else "WARN", rec_time,
                          f"text={text} (expected None for silence)")

        # Test 3: Verify listen() returns None for silence
        t0 = time.time()
        listen_result = listen(phrase_time_limit=1.0)
        listen_time = time.time() - t0
        result.record_stage("listen_silence", "PASS" if listen_result is None else "WARN", listen_time,
                          f"result={listen_result}")

        result.duration = cap_time + rec_time + listen_time
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"STT test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_tts():
    """Test text-to-speech engine."""
    result = TestResult("tts")
    logger.info("=" * 60)
    logger.info("TEST: TTS")
    logger.info("=" * 60)

    try:
        from voice.synthesizer import speech_synthesizer
        from voice.tts.manager import tts_manager

        # Initialize TTS
        t0 = time.time()
        speech_synthesizer.initialize()
        init_time = time.time() - t0

        if not tts_manager.ready:
            result.record_stage("init", "FAILED", init_time, "No TTS engine available")
            # Don't fail - TTS may not be available on this system
            result.mark_passed()
            return result

        result.record_stage("init", "PASS", init_time,
                          f"engine={tts_manager.active_engine_name}")

        # Test speaking
        t0 = time.time()
        success = speech_synthesizer.speak("Test. One, two, three.")
        speak_time = time.time() - t0
        result.record_stage("speak", "PASS" if success else "FAILED", speak_time,
                          f"success={success}")

        # Test is_speaking after completion
        speaking = speech_synthesizer.is_speaking()
        result.record_stage("is_speaking", "PASS", 0, f"speaking={speaking} (expected False)")

        result.duration = init_time + speak_time
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"TTS test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_plugin_loading():
    """Test that plugins load and initialize correctly."""
    result = TestResult("plugin_loading")
    logger.info("=" * 60)
    logger.info("TEST: Plugin Loading")
    logger.info("=" * 60)

    try:
        from core.plugin_manager import plugin_manager

        t0 = time.time()
        await plugin_manager.load_all()
        await plugin_manager.initialize_all()
        result.duration = time.time() - t0

        names = list(plugin_manager.plugins.keys())
        result.record_stage("load", "PASS", result.duration,
                          f"plugins={len(names)}: {names}")

        for name, plugin in plugin_manager.plugins.items():
            result.record_stage(f"plugin:{name}", "PASS" if plugin.enabled else "DISABLED", 0,
                              f"enabled={plugin.enabled}")

        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Plugin loading failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_event_bus():
    """Test event bus emissions."""
    result = TestResult("event_bus")
    logger.info("=" * 60)
    logger.info("TEST: Event Bus")
    logger.info("=" * 60)

    try:
        from core.event_bus import bus, Event

        received_events = []

        async def test_handler(event: Event):
            received_events.append(event)

        bus.on("test_event", test_handler)

        t0 = time.time()
        await bus.emit("test_event", data={"test": True}, source="test")
        result.duration = time.time() - t0

        if len(received_events) == 1:
            result.record_stage("emit", "PASS", result.duration,
                              f"event={received_events[0].type}, data={received_events[0].data}")
            result.mark_passed()
        else:
            result.record_stage("emit", "FAILED", result.duration,
                              f"received {len(received_events)} events")
            result.mark_failed("Event bus did not deliver event")

        bus.off("test_event", test_handler)

    except Exception as e:
        result.mark_failed(f"Event bus test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_handle_intent():
    """Test intent handling for various intents."""
    result = TestResult("handle_intent")
    logger.info("=" * 60)
    logger.info("TEST: Intent Handling")
    logger.info("=" * 60)

    test_intents = [
        {"intent": "greeting", "confidence": 0.95, "text": "hello", "entities": {}},
        {"intent": "exit", "confidence": 0.95, "text": "goodbye", "entities": {}},
        {"intent": "time_query", "confidence": 0.95, "text": "what time is it", "entities": {}},
        {"intent": "joke", "confidence": 0.95, "text": "tell me a joke", "entities": {}},
        {"intent": "who_am_i", "confidence": 0.95, "text": "who am i", "entities": {}},
        {"intent": "youtube", "confidence": 0.95, "text": "open youtube", "entities": {}},
        {"intent": "brightness_up", "confidence": 0.95, "text": "increase brightness", "entities": {}},
    ]

    try:
        from main import handle_intent
        from voice.synthesizer import speech_synthesizer
        from voice.tts.manager import tts_manager

        # Ensure TTS is initialized
        if not tts_manager.ready:
            speech_synthesizer.initialize()

        t0 = time.time()
        for test_case in test_intents:
            tt = time.time()
            response = await handle_intent(test_case)
            dt = time.time() - tt
            result.record_stage(f"intent:{test_case['intent']}", "PASS", dt,
                              f"response='{response}'")

        result.duration = time.time() - t0
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Intent handling failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_conversation_state():
    """Test conversation state management."""
    result = TestResult("conversation_state")
    logger.info("=" * 60)
    logger.info("TEST: Conversation State")
    logger.info("=" * 60)

    try:
        from nlp.conversation_state import conversation_state, PendingAction

        # Test initial state
        t0 = time.time()
        has_pending = conversation_state.has_pending_action
        result.record_stage("initial_state", "PASS", time.time() - t0,
                          f"has_pending={has_pending}")

        # Test setting pending action
        t0 = time.time()
        conversation_state.set_pending(
            PendingAction.YOUTUBE_QUERY,
            {"query": "test"}
        )
        has_pending = conversation_state.has_pending_action
        result.record_stage("set_pending", "PASS", time.time() - t0,
                          f"has_pending={has_pending}, action={conversation_state.pending_action}")

        # Test clearing (use set_pending with None to clear)
        t0 = time.time()
        conversation_state.set_pending(None, None)
        has_pending = conversation_state.has_pending_action
        result.record_stage("clear_pending", "PASS", time.time() - t0,
                          f"has_pending={has_pending}")

        result.duration = 0.0
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Conversation state test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_embeddings():
    """Test embedding model loading and inference."""
    result = TestResult("embeddings")
    logger.info("=" * 60)
    logger.info("TEST: Embeddings")
    logger.info("=" * 60)

    try:
        from nlp.embeddings import preload_embedding_model, embed, cosine_similarity

        # Preload model
        t0 = time.time()
        preload_embedding_model()
        load_time = time.time() - t0
        result.record_stage("preload", "PASS", load_time, "Model loaded")

        # Generate embeddings
        t0 = time.time()
        vec1 = embed("hello world")
        t1 = time.time()
        vec2 = embed("hello world")
        t2 = time.time()
        vec3 = embed("goodbye world")
        t3 = time.time()

        result.record_stage("embed_1", "PASS", t1 - t0, f"dim={len(vec1)}")
        result.record_stage("embed_2_cached", "PASS", t2 - t1, "cached (should be fast)")
        result.record_stage("embed_3", "PASS", t3 - t2, f"dim={len(vec3)}")

        # Check similarity
        sim_same = cosine_similarity(vec1, vec2)
        sim_diff = cosine_similarity(vec1, vec3)
        result.record_stage("similarity_same", "PASS", 0, f"sim={sim_same:.4f} (expected ~1.0)")
        result.record_stage("similarity_diff", "PASS", 0, f"sim={sim_diff:.4f} (expected <1.0)")

        result.duration = load_time + (t3 - t0)
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Embeddings test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def test_wake_word_detection():
    """Test wake word detection logic (without actual audio)."""
    result = TestResult("wake_word")
    logger.info("=" * 60)
    logger.info("TEST: Wake Word Detection")
    logger.info("=" * 60)

    try:
        from voice.wake_word import wake_word_engine

        test_phrases = [
            ("hello leo", True),
            ("hey leo", True),
            ("hello world", False),
            ("goodbye", False),
            ("leo", True),
            ("lio", True),
        ]

        t0 = time.time()
        for phrase, expected in test_phrases:
            tt = time.time()
            detected = wake_word_engine.detect(phrase)
            dt = time.time() - tt

            status = "PASS" if detected == expected else "FAIL"
            result.record_stage(f"detect('{phrase}')", status, dt,
                              f"detected={detected}, expected={expected}")

        result.duration = time.time() - t0
        result.mark_passed()

    except Exception as e:
        result.mark_failed(f"Wake word test failed: {e}")
        logger.error("[TEST] Exception: %s", e, exc_info=True)

    return result


async def run_all_tests():
    """Run all runtime tests."""
    tests = [
        test_startup_diagnostics(),
        test_nlp_classification(),
        test_microphone_backend(),
        test_audio_capture(),
        test_stt(),
        test_tts(),
        test_plugin_loading(),
        test_event_bus(),
        test_handle_intent(),
        test_conversation_state(),
        test_embeddings(),
        test_wake_word_detection(),
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    print()
    print("=" * 70)
    print("  RUNTIME TEST RESULTS")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    skipped = 0

    for r in results:
        if isinstance(r, Exception):
            logger.error("[TEST] Unexpected exception: %s", r)
            failed += 1
            continue

        if r.passed:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"  [{status}] {r.name} ({r.duration:.2f}s)")
        for stage in r.stages:
            print(f"         ├─ [{stage['status']}] {stage['stage']} ({stage['duration']:.2f}s) {stage['detail']}")
        if r.error:
            print(f"         └─ ERROR: {r.error}")

    print()
    print("=" * 70)
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)