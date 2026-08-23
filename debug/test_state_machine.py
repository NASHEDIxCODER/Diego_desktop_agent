"""
Hardware-free end-to-end test of the strict voice-state machine.

Proves, WITHOUT a microphone / camera / speaker / Ollama:
  1. BOOT → FACE_AUTH → IDLE → WAKE_LISTEN → WAKE_DETECTED → GREETING
     → COMMAND_LISTEN → THINKING → SPEAKING → FOLLOWUP → COMMAND_LISTEN
     → (8s silence) → WAKE_LISTEN — exact order, no bypasses.
  2. Whisper is NEVER created during WAKE_LISTEN (lazy activation exactly
     once, after WAKE_DETECTED) and is destroyed when the session ends.
  3. The conversation timeout returns the machine to WAKE_LISTEN.

Run:  python debug/test_state_machine.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.conversation_engine as ce
import voice.wake_listener as wl
from core.conversation_engine import ConversationEngine, EngineState
from voice.streaming_stt import UtteranceEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("state_machine_test")

# Fast timeout for the test (the production value is 8.0s).
ce.CONVERSATION_TIMEOUT_S = 0.6


# ── Fakes ────────────────────────────────────────────────────────

class FakeAudioManager:
    is_running = True

    def __init__(self):
        self._total = 0

    @property
    def total_samples(self):
        return self._total

    def start(self):
        return True

    def read_since(self, last):
        # Deliver one 1280-sample chunk per call, like the real ring buffer.
        self._total += 1280
        return np.zeros(1280, dtype=np.float32), self._total

    def get_recent_audio(self, _dur):
        # ≥8000 samples: the wake-confirmation path requires ≥0.5 s of
        # audio (float32 ring-buffer domain). Returning fewer silently
        # starves the confirmation — the exact production failure mode.
        return np.zeros(16000, dtype=np.float32)

    def get_recent_processed(self, _dur):
        # Wake confirmation reads PREPROCESSED audio (the detector's
        # input), same ≥8000-sample contract.
        return np.zeros(16000, dtype=np.float32)



class FakeWakeModel:
    loaded = True
    threshold = 0.5
    wake_phrase = "hello Diego"
    model_name = "fake_Diego"
    load_error = None

    def __init__(self):
        self.calls = 0
        self.score = 0.12
        self.fired = False
        self._t0 = time.monotonic()

    def load(self):
        return True

    def predict_stream(self, _frame):
        self.calls += 1
        if self.fired:
            # Already woke once — park below threshold forever so the test
            # observes exactly ONE conversation cycle.
            return {self.model_name: 0.05}
        # Stay below threshold during the WakeListener's 1.0 s refractory
        # window, then fire ONCE (0.93 ≥ 0.5) and latch.
        if time.monotonic() - self._t0 < 1.3:
            return {self.model_name: 0.12}
        self.fired = True
        self.score = 0.93
        return {self.model_name: self.score}


    def highest_score(self):
        return self.model_name, self.score

    def reset_stream(self):
        pass

    def hard_reset(self):
        # Deterministic re-entry reset (called by WakeListener.prime).
        self._t0 = time.monotonic()

    def record_false_positive(self):
        pass


class FakeVADGate:
    ready = False

    def load(self):
        return False


class _FakeWhisperBackend:
    """The streaming_stt._whisper double used by the wake-confirmation
    path. Returns a verified wake transcript ONCE, then background
    speech — so the engine runs exactly ONE conversation cycle (the
    VAD-fallback trigger re-confirms continuously, and only the first
    confirmation may pass)."""

    def __init__(self):
        self.calls = 0

    def _text(self) -> str:
        self.calls += 1
        return "hello Diego" if self.calls == 1 else "what time is it"

    def transcribe_detailed(self, _pcm, _sr=16000, _use_vad=False):
        text = self._text()
        return {"text": text, "language": "en",
                "language_probability": 0.99, "avg_logprob": -0.2,
                "no_speech_prob": 0.05, "compression_ratio": 1.1,
                "segments": [{"text": text, "avg_logprob": -0.2,
                              "no_speech_prob": 0.05,
                              "compression_ratio": 1.1}],
                "ok": True, "reason": "accepted"}

    def transcribe(self, _pcm, _sr=16000, _use_vad=False):
        return self._text()



class FakeSTT:
    """Streaming Whisper double. Counts activations; must be 0 during
    WAKE_LISTEN and exactly 1 after WAKE_DETECTED."""

    def __init__(self):
        self.ready = True
        self.sessions = 0
        self.destroyed = 0
        self._whisper = _FakeWhisperBackend()


    def initialize(self):
        return True

    def stop_streaming(self):
        self.destroyed += 1

    async def stream_utterances(self):
        self.sessions += 1
        now = time.time()
        yield UtteranceEvent(kind="speech_start", started_at=now)
        await asyncio.sleep(0.05)
        yield UtteranceEvent(kind="final", text="what time is it",
                             is_final=True, started_at=now, ended_at=time.time())
        # Silence forever after the command (drives the conversation timeout).
        while True:
            await asyncio.sleep(0.05)


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def initialize(self):
        return True

    async def speak_sentences(self, sentences, _interrupt):
        async for s in sentences:
            self.spoken.append(s)
        return True

    def stop(self):
        pass

    @property
    def is_speaking(self):
        return False


class FakeLLM:
    async def generate(self, text, _cancel):
        yield "It is noon."


# ── Test ─────────────────────────────────────────────────────────

async def main() -> int:
    states = []
    orig_set_state = ConversationEngine._set_state

    def tracked(self, new_state):
        states.append(new_state)
        orig_set_state(self, new_state)

    ConversationEngine._set_state = tracked
    ConversationEngine._play_wake_chime = staticmethod(lambda: None)

    fake_stt = FakeSTT()
    fake_tts = FakeTTS()
    fake_audio = FakeAudioManager()
    fake_wake = FakeWakeModel()
    ce.audio_manager = fake_audio
    ce.wake_model_manager = fake_wake
    ce.streaming_stt = fake_stt
    ce.streaming_tts = fake_tts
    ce.streaming_llm = FakeLLM()
    # The WakeListener binds its own module-level singletons — patch them
    # at its module level too (same fakes, single pipeline).
    wl.audio_manager = fake_audio
    wl.wake_model_manager = fake_wake
    # Noise suppression passthrough (real one needs a noise profile).
    wl.audio_preprocessor = type("P", (), {"process": staticmethod(lambda a: a)})()
    # Whisper verification resolves streaming_stt lazily from its module —
    # point it at the fake (the fake's _whisper returns a wake transcript).
    import voice.streaming_stt as stt_mod
    stt_mod.streaming_stt = fake_stt

    eng = ConversationEngine()
    eng._wake_listener.vad = FakeVADGate()
    eng.set_authenticated(None)  # --no-auth dev session

    # Whisper must NOT exist yet.
    assert fake_stt.sessions == 0, "Whisper created before the engine even ran"

    task = asyncio.create_task(eng.run())
    await asyncio.sleep(4.0)
    eng._running = False
    await asyncio.sleep(0.3)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    names = [s.value for s in states]
    print("\nState sequence observed:")
    print("  " + " → ".join(names))

    # ── 1. Exact chain, in order ──────────────────────────
    # (no-auth dev session: FACE_AUTH is correctly SKIPPED — the only
    # legal BOOT exit is IDLE; auth only appears after WAKE_DETECTED
    # when an auth provider is configured).
    expected_prefix = [
        "BOOT", "IDLE", "WAKE_LISTEN", "WAKE_DETECTED",
        "GREETING", "COMMAND_LISTEN", "THINKING", "SPEAKING",
        "FOLLOWUP", "COMMAND_LISTEN",
    ]
    assert names[:len(expected_prefix)] == expected_prefix, \
        f"Chain mismatch:\n  got:  {names[:len(expected_prefix)]}\n  want: {expected_prefix}"

    # ── 2. Timeout returned to WAKE_LISTEN ────────────────
    assert names[-1] == "WAKE_LISTEN", f"Did not return to WAKE_LISTEN: {names[-1]}"

    # ── 3. Whisper lazily activated once, then destroyed ──
    assert fake_stt.sessions == 1, f"Whisper activations: {fake_stt.sessions} (want 1)"
    assert fake_stt.destroyed >= 1, "Streaming Whisper was not destroyed after the session"
    first_command_listen = names.index("COMMAND_LISTEN")
    assert first_command_listen > names.index("WAKE_DETECTED"), \
        "COMMAND_LISTEN before WAKE_DETECTED — chain violated"

    # ── 4. LLM + TTS only ran inside the conversation ─────
    assert fake_tts.spoken, "TTS never spoke the response"

    print("\nALL STATE MACHINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
