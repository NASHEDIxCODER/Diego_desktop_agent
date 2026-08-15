# RUNTIME_STABILITY_REPORT.md — Leo Desktop Assistant

Date: 2026-08-13
Scope: Full runtime error elimination task (Phases 1–17).

IMPORTANT HONESTY NOTE

The automated environment has a real microphone (`HD-Audio Generic: ALC256
Analog`) and the production entry point boots cleanly, verifies the mic,
enters WAKE, detects a wake trigger, and — after the Phase 8 fix — correctly
REJECTS false-positive transcripts instead of accepting them. However, I
cannot produce controlled *human speech* from this agent to exercise a full
`wake → command audio → VAD → STT → command processing → response → TTS →
next wake` loop and record 5 complete cycles with my own voice. Phase 16
(real hardware acceptance with 20 wake attempts) therefore remains a manual
step for the user. Everything that CAN be verified without live human speech
has been verified via unit tests and a live production boot log (reproduced
below).

---

## 1. Every runtime error discovered

| # | Error / symptom | Root cause | Affected component |
|---|---|---|---|
| 1 | `wake_score=0.996` + transcript `"I don't know who you are."` ACCEPTED (false positive). Also seen live as transcript `"you"` + score 0.998 accepted. | Tautology in `verify_wake_transcript()` high-confidence path: `w in high_words` iterates `w` FROM `high_words`, so the guard was always True and accepted ARBITRARY transcripts whenever `wake_score >= 0.995`. | `voice/wake_word.py` |
| 2 | Whisper inference could block the conversation engine / wake loop indefinitely (no timeout). | `run_in_executor(...transcribe...)` with no `wait_for`; a hung/frozen faster-whisper would stall a turn or WAKE verification forever. | `voice/command_listener.py`, `voice/wake_listener.py` |
| 3 | No structured STATE TIMEOUT / recovery: a stalled LISTEN/THINK/SPEAK state could hang without diagnostics. | `_set_state()` logged transitions but had no timeout watchdog or recovery. | `core/conversation_engine.py` |
| 4 | Wake-loop model-reload failure crashed the wake task silently (worker died, no log). | The `wake_model_manager.load()` retry was OUTSIDE the try/except; a load-raising exception escaped the coroutine. | `voice/wake_listener.py` |
| 5 | Background learner cycle errors logged at DEBUG only (`logger.debug`), essentially silent worker failures. | `_learn_loop` swallowed `_learn_one_cycle` exceptions. | `core/background_learning.py` |
| 6 | Shutdown had no hard upper timeout — a hung worker could prevent exit. | `await shutdown()` unguarded. | `leo.py` |

No deadlocks, no duplicate InputStreams, no stalled VAD OPEN/CLOSED state,
and no silent-failure in the audio ring buffer were found (audited in
`RUNTIME_AUDIT.md`).

## 2. Root cause

Per table above. The single highest-impact defect was #1 — a logical
tautology that made the wake-verification second authority accept arbitrary
speech. This is precisely the "Phase 8 Wake Verification Bug" and was
observed LIVE in the production boot log.

## 3. Affected component

`voice/wake_word.py`, `voice/wake_listener.py`, `voice/command_listener.py`,
`voice/vad.py`, `core/conversation_engine.py`, `core/background_learning.py`,
`leo.py`.

## 4. Exact fix

1. **`voice/wake_word.py`** — `w in high_words` → `w in high_distinctive`
   (and `high_distinctive` converted to a `set`). This is the minimal
   one-character-semantic fix; no architectural change.
2. **`voice/command_listener.py`** — added `_transcribe_with_timeout()` using
   `asyncio.wait_for` with `WHISPER_FINAL_TIMEOUT_S=30` and
   `WHISPER_PARTIAL_TIMEOUT_S=5`; wired into both final and partial paths.
   Added structured `[CMD-DEBUG]` traces (listener_enter, audio_frame_received,
   vad_probability, speech_started, speech_ended, endpoint, whisper_started,
   whisper_finished, transcript).
3. **`voice/wake_listener.py`** — wrapped Whisper verification in
   `asyncio.wait_for(..., VERIFY_TIMEOUT_S=30)`; wrapped the model-reload retry
   in try/except with a structured `[WORKER-CRASH]` log so the loop survives.
4. **`core/conversation_engine.py`** — added a `_state_watchdog()` task that
   logs a structured `STATE TIMEOUT` record (state/previous/elapsed/ceiling/
   thread/audio_running/active_threads) and triggers safe recovery via
   `command_listener.stop_streaming()`.
5. **`voice/vad.py`** — added VAD observability (state, last probability, last
   audio/speech timestamps) updated in `speech_prob()`.
6. **`core/background_learning.py`** — worker exception now logs structured
   `[WORKER-CRASH] worker=background_learner ...`.
7. **`leo.py`** — `shutdown()` wrapped in `asyncio.wait_for(..., timeout=15.0)`
   with an explicit `[SHUTDOWN] Hard timeout exceeded` log.

No logging was suppressed; no functionality was removed; no architecture
rewritten.

## 5. Tests added

New file `tests/test_runtime_stability.py`:
- `test_wake_accepted_valid_variants`
- `test_wake_rejected_unrelated_transcript`
- **`test_wake_to_listen_receives_fresh_audio`** (fails if the command
  listener starts but receives no fresh frames)
- `test_command_timeout_empty_silence`
- `test_command_transcription_timeout_recovery`
- `test_tts_stop_idempotent`
- `test_ring_buffer_stream_recovery`
- `test_worker_exception_not_swallowed`

## 6. Tests passed

- `tests/test_runtime_stability.py` → **8 passed**
- `tests/test_command_listener.py` → **7 passed**
- `tests/test_event_bus.py` + `tests/test_command_listener.py` → **12 passed**
- Combined (`test_command_listener` + `test_runtime_stability` + `test_event_bus`) → **20 passed**
- `tests/test_production_regression.py` (own runner) → **69 passed**
- `tests/test_response_guarantee.py` (own runner) → **26 passed**

## 7. Remaining warnings

- `pkg_resources is deprecated` — third-party import warning, not runtime.
- `[ WARN:0@...] setPreferableTarget Targets are not supported by the new graph engine` — OpenCV face-detector notice, emitted only at shutdown probe, harmless.
- OpenCV `face_recognition_models` UserWarning — cosmetic.

## 8. Remaining known issues

- **Phase 16 real-hardware acceptance (20 live wake attempts) not run** — this
  requires the user to speak into the mic; the agent cannot provide human
  speech. Unit + live-boot verification (below) confirm the pipeline mechanics.
- Root `voice/streaming_stt.py` remains a legacy duplicate STT (not wired into
  the engine); left untouched per "do not remove working components".
- Dormant `core/state_machine.py` nervous system remains unwired (documented in
  audit); not in the live path.

## 9. Performance impact

- Whisper `wait_for` adds no steady-state overhead (only an upper bound).
- CMD-DEBUG/VAD diagnostics are rate-limited (once/sec for most).
- State watchdog consumes ~1 coroutine tick/second.
- Net: negligible.

## 10. Files changed

- `RUNTIME_AUDIT.md` (created)
- `RUNTIME_STABILITY_REPORT.md` (created)
- `tests/test_runtime_stability.py` (created)
- `voice/wake_word.py` (fixed false-positive tautology)
- `voice/wake_listener.py` (verify timeout + worker crash reporting)
- `voice/command_listener.py` (inference timeouts + CMD-DEBUG diagnostics)
- `voice/vad.py` (observability)
- `core/conversation_engine.py` (state watchdog)
- `core/background_learning.py` (worker crash reporting)
- `leo.py` (shutdown hard timeout)

---

## Final runtime log (live production boot, post-fix)

Excerpt showing the fix working — the false-positive wakes are now REJECTED
instead of accepted:

```
STATE START → IDLE (2.2s)
STATE IDLE → WAKE (17.2s)
Wake trigger (score=0.999 ≥ 0.85) — verifying transcript…
[WAKE] VERDICT: transcript='you' confidence=-1.600 duration=2.50s samples=40000
[WAKE-VERIFY] REJECTED (high-confidence but transcript disagrees): wake_score=0.999 ≥ 0.995 transcript='you' — no wake-word evidence
Wake rejected (score=1.00 transcript='you') — still listening

Wake trigger (score=0.988 ≥ 0.85) — verifying transcript…
[WAKE] VERDICT: transcript='Sorry, I...' confidence=-1.035
[WAKE-VERIFY] REJECTED: text='sorry i' best='i'≈'lio' confidence=0.500 (< 0.80)
Wake rejected (score=0.99 transcript='Sorry, I...') — still listening

Wake trigger (score=1.000 ≥ 0.85) — verifying transcript…
[WAKE] VERDICT: transcript='you' confidence=-1.491
[WAKE-VERIFY] REJECTED (high-confidence but transcript disagrees): ... transcript='you'
Wake rejected (score=1.00 transcript='you') — still listening

Wake trigger (score=0.916 ≥ 0.85) — verifying transcript…
[WAKE] VERDICT: transcript='All right, dude.' confidence=-0.893
[WAKE-VERIFY] REJECTED: text='all right dude' best='all'≈'leo' confidence=0.333 (< 0.80)
Wake rejected (score=0.92 transcript='All right, dude.') — still listening
```

Zero ERROR / CRITICAL / STATE TIMEOUT / VIOLATION / WORKER-CRASH lines. Clean
shutdown (`[LEO] Shutdown complete.`).

The conversation session machinery (fresh-audio handoff, VAD, endpoint,
Whisper, response guarantee, TTS) is verified by the 20 passing unit tests and
by the earlier live boot log that showed the listener receiving fresh frames
(`[CMD] speech_started vad_prob=0.63`, `[CMD] silence_ms=…`,
`[CMD] endpoint …`) and the engine driving `THINK → SPEAK → LISTEN`.