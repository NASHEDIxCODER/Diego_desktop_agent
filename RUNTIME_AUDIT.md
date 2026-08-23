# RUNTIME_AUDIT.md — Diego Desktop Assistant Forensic Audit

Scope: entire repository, runtime entry points, voice pipeline, state machine,
threads, asyncio tasks, queues, locks, shared audio buffers, shutdown paths.

Baseline: existing `tests/test_command_listener.py` passes 7/7; `tests/test_production_regression.py`
is a standalone async runner. No automated fresh-audio handoff test exists for the
engine-level wake→command transition yet.

---

## 1. Entry points

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `main.py` / `Diego.py` `run()` | main thread → `asyncio.run(_main_async)` | argv, env | log/status | env vars, signals | module imports | none (single-threaded bootstrap) | `run()` swallows KeyboardInterrupt only | intentional env fixes; heavy imports lazy | none | keep |
| `Diego.py` `_main_async()` | 1 event loop, `run_task` + `stop_task` | signals | none | `asyncio.Event` stop | `stop.wait()` (intentional) | `asyncio.wait` FIRST_COMPLETED | cancels pending, calls `shutdown()` | clean Ctrl+C path | none | keep; add hard shutdown timeout |
| `Diego.py` `run_Diego()` | `engine_task`, `background_learner` task | none | logs | singletons | model loads via executor | engine task vs learner | finally cancels engine + stops learner | loaded models once | none | keep |

## 2. ConversationEngine (`core/conversation_engine.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `ConversationEngine` | 1 engine coroutine, `_stt_event_pump` task, `_watch_interruption` task, `_one_line_stream` producer task | wake events, STT events | spoken responses | `audio_manager`, `command_listener`, `streaming_tts`, `wake_listener` | chime `sd.play/sd.wait` in executor | `_tts_interrupt` event vs TTS player | try/finally cancels pump; no state watchdog | no per-state timeout; a hung state (Whisper/LLM/TTS) stalls the turn with no recovery; STATE TIMEOUT not structured | **Add state watchdog + STATE TIMEOUT log + recovery** |
| `EngineState` enum + `ALLOWED_TRANSITIONS` | — | — | — | — | — | — | violation logs but does NOT prevent | self-loop allowed list has no self | none | keep |

Engine state machine lacks durability: `_set_state` logs entry/exit but has no
**timeout**, **failure path**, or **recovery** enforcement. `ALLOWED_TRANSITIONS`
is checked but violations are only logged, not recovered.

## 3. WakeListener (`voice/wake_listener.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `WakeListener.wait_for_wake()` | 1 coroutine (engine's WAKE loop) | ring-buffer float32 | `WakeEvent` | `audio_manager` (read_since), `wake_model_manager`, `unified_vad`, `command_listener` (verify) | Whisper verify via executor | `_verifying` flag single-threaded | loop wraps each iter in try/except `logger.exception`, sleeps 0.1 | verification lacks timeout → hang if Whisper hangs; high score + unrelated transcript concern | **Whisper verify timeout; enforce transcript-evidence rejection (already present in wake_word.py — add tests)** |

`verify_with_whisper()` calls `command_listener._whisper.transcribe()` via executor **without timeout**. A
hanging faster-whisper inference would block the WAKE loop indefinitely.

## 4. CommandListener (`voice/command_listener.py`) — THE production STT

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `CommandListener.stream_utterances()` | async generator (driven by `_stt_event_pump`) | ring-buffer float32 | `UtteranceEvent` | `audio_manager` (read_since/total_samples), `unified_vad` | VAD + whisper via executor | `_listen_enabled`, `_drain_requested`, `_cancel` | exceptions in VAD/whisper caught inside `_WhisperTranscriber` | no `[CMD-DEBUG]` traces; whisper final/partial loop has **no timeout** → stall | **Add CMD-DEBUG traces, Whisper timeout, fresh-frame age log** |

The session boundary (`command_session_start = audio_manager.total_samples`) and drain-on-resume logic are
correct and covered by `tests/test_command_listener.py`. Missing pieces: structured
diagnostics and a Whisper inference timeout.

## 5. AudioManager (`voice/audio_manager.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `AudioManager._audio_callback` | sounddevice callback thread | mic int16/float32 | ring-buffer float32 | `RingBuffer` (deque+lock), `_vad`, AGC, hp filter | none | ring-buffer lock very short | callback try/except around normalization only | single authoritative InputStream; channel locking reasonable | none | keep |
| `RingBuffer` | thread-safe | frames | `get_recent/get_since/bytes` | `threading.Lock`, `_total_samples` | none | `total_samples` monotonic | none | peek semantics correct | none | keep |
| `start()` | executor (from engine) | devices | True/False | sounddevice | probe loops (bounded) | single init | returns False if silent | robust hardware detection | none | keep |

ONE authoritative InputStream exists. No duplicate streams. Resample/high-pass done ONCE in callback.

## 6. UnifiedVAD (`voice/vad.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `UnifiedVAD.speech_prob()` | called via executor | 512-sample frame | probability | Silero model | none | read-only model | fallback energy on any exception | no per-frame diagnostics (vad state/prob/last timestamp) | **add VAD diagnostics to command listener + callback** |

No permanent OPEN/CLOSED stuck state found; `speech_prob` always returns. Add observability.

## 7. Whisper (`_WhisperTranscriber` in command_listener.py)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `transcribe` / `transcribe_fast` | called via `run_in_executor` | PCM16 bytes | (text, logprob) | faster-whisper model | **inference can block unbounded** | none (single model) | catches Exception→("",0) | **NO timeout/recovery** | **wrap all inference in `asyncio.wait_for` + timeout** |

faster-whisper is the single highest risk blocking point. Must never block the engine indefinitely.

## 8. Wake verification (`voice/wake_word.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `verify_wake_transcript()` | sync | transcript, wake_score | bool | none | none | none | pure function | high-score+unrelated-transcript path is guarded | **already correct — add regression test** |

Current code ALREADY rejects `wake_score>=0.995` with unrelated transcript (see `core/conversation_engine.py`
verify path and `wake_word.py` lines 159-181). Confirm with tests.

## 9. StreamingTTS (`voice/streaming_tts.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `_InterruptiblePlayer._run()` | persistent `tts-playback` thread | chunk queue | speaker output | `_stream_lock`, `_stream`, events | `write()` (bounded by PortAudio) | stop protocol uses events; stream only destroyed in `close()` | exceptions recreate stream lazily | `stop()/close()` idempotent (re-arm or no-op) | `stop()` not idempotent on repeated call but harmless | keep; verify idempotency test |

`stop()`/`interrupt()` are re-entrant and `close()` is guarded. Stream destruction happens only in `close()` after worker join.

## 10. AgentBrain (`agent/brain.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `process_command` | engine coroutine | transcript | `CommandResult` | conv_memory, decision_engine, dispatcher | LLM/planner via executor | `_pipeline_timings` class attr (not per-instance) | per-stage try/except | pipeline stats class-level shared | none (not runtime-critical) | keep |

## 11. ResponseGuarantee (`core/response_guarantee.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `run_turn` | engine coroutine | process/speak fns | bool | stats counters | TTS via speak_fn | none | 3-attempt fallback; FATAL silent log | none | none | keep |

## 12. BackgroundLearner (`core/background_learning.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `_learn_loop` task | 1 task | idle state | knowledge base | files, cache_manager | httpx/URL fetch only during idle | state check via `_state` | catches cycle exceptions (debug log only) | `_learn_one_cycle` errors swallowed silently | **upgrade debug→structured WORKER/LEARN log** |

## 13. EventBus (`core/event_bus.py`)

| Component | Thread/Task | Inputs | Outputs | Shared resources | Blocking points | Race conditions | Failure handling | Symptoms | Proposed fix |
|---|---|---|---|---|---|---|---|---|---|
| `EventBus.emit` | caller coroutine | event | handler results | `_lock`, handlers | handler awaits | handler list mutation race possible | gather(return_exceptions) | none | none | keep |

## 14. RuntimeStateMachine (`core/state_machine.py`)

Unused by the live engine (the engine owns its own `EngineState`). This is a **parallel
nervous system** with `timeout_watchdog` and `recover()` but is not wired into
`run_Diego`. Document as dormant/duplicate; not changing architecture per task constraints.

## 15. StreamingSTT (`voice/streaming_stt.py`)

Legacy/duplicate STT. NOT imported by `core/conversation_engine.py` (the engine uses
`voice/command_listener.py`). Contains its own `_SileroVAD` and `_WhisperTranscriber`.
Leaving untouched (do not remove working components, do not refactor architecture).

## 16. Shutdown path (`Diego.py` shutdown(), `audio_manager.stop()`)

Order today: set `_running=False` → save session → TTS close → command listener cancel →
audio shutdown_event + stop → face auth close. Mostly correct. Missing a **hard upper
timeout** if a worker refuses to join. The TTS `close()` worker join already has 2s timeout.

## 17. Error handling summary

The codebase has ~268 bare/exception-swallowing points. Most are legitimate (best-effort
subsystem probes, mixer backends, cleanup). The runtime-critical exceptions to harden:

1. `command_listener._whisper.transcribe*` — currently no timeout (highest risk).
2. `wake_listener.verify_with_whisper` → same whisper call with no timeout.
3. `background_learning._learn_one_cycle` — silently debug-logged worker crash.
4. `conversation_engine._set_state` — no timeout/recovery watchdog.

## 18. Confirmed bug (Phase 4) root-cause summary

Log sequence:
```
Wake accepted
STATE WAKE → LISTEN
[CMD-LISTEN] Draining TTS-contaminated audio
[CMD-LISTEN] Listening started
```
then no `speech_start`/VAD/partial/endpoint/final/timeout.

Audit findings:

- The engine (production path) uses `voice/command_listener.py` whose session-boundary
  and drain logic are correct and unit-tested. The literal `[CMD-LISTEN] Draining
  TTS-contaminated audio` string is NOT in `command_listener.py` (it is in the legacy
  `voice/streaming_stt.py`); the engine logs `[CMD] session_start` / `[CMD] Listening started`.
- The only guaranteed **indefinite stall** vectors are (a) Whisper inference with no
  timeout and (b) the absence of a structured STATE TIMEOUT watchdog that could detect
  and recover a stuck turn.
- The "TTS-contaminated audio" drain marker after LISTEN suggests an earlier (pre-command-
  listener) code path whose drain consumed the fresh microphone start boundary. Current
  code establishes a clean sample-accurate boundary; the fix is to (1) enforce Whisper
  timeouts, (2) add fresh-frame diagnostics so a silent handoff is immediately visible,
  and (3) add the state watchdog.

---

## Priority fix plan (does NOT rewrite architecture)

1. Whisper inference timeout (Phase 7) — wrap all executor transcription in `asyncio.wait_for`.
2. State watchdog (Phase 3) — structured STATE TIMEOUT with previous/elapsed/queue/audio state + recovery.
3. CMD-DEBUG diagnostics (Phase 4) — listener_enter/audio_frame_received/frame_age/vad/speech/endpoint/whisper.
4. Wake verification regression tests (Phase 8) — false-positive + valid-variant matrices.
5. Worker crash structured logging (Phase 10) — `[WORKER-CRASH]` in learner + TTS + wake loops.
6. VAD diagnostics (Phase 6) — state/prob/timestamps.
7. Shutdown hard timeout (Phase 12) — bound `shutdown()`.
8. Regression tests incl. `test_wake_to_listen_receives_fresh_audio()` (Phase 15).