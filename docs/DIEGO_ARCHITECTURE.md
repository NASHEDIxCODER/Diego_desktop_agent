# Diego Desktop Assistant

Technical architecture reference for the Diego desktop agent.
This is the single canonical documentation file for the production codebase.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ UI (PySide6 — ui/)                                                  │
│   main_window.py · widgets.py · visualizer.py · event_bridge.py     │
│   (Qt main thread; own event loop; owns GUI dispatcher)             │
└───────────────┬─────────────────────────────────────────────────────┘
                │ gui.submit() thread-safe marshalling (core/gui_dispatcher)
┌───────────────▼─────────────────────────────────────────────────────┐
│ ConversationEngine (core/conversation_engine.py)                    │
│   7-state machine: IDLE→WAKE→FACE_AUTH→LISTEN→THINK→SPEAK→IDLE      │
│   The ONLY voice-state driver                                       │
└───────┬───────────────┬───────────────┬───────────────┬─────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│ AudioManager│ │WakeListener │ │CommandList. │ │ Stream TTS   │
│ (mic/ring)  │ │(openWake)   │ │(Whisper STT)│ │ (Kokoro etc) │
│ voice/      │ │ voice/      │ │ voice/      │ │ voice/       │
└─────────────┘ └─────────────┘ └──────┬──────┘ └──────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ AgentBrain (agent/brain.py) — single command orchestrator           │
│ normalize → transcript guard → intent authorizer →                  │
│ perceive → decide → plan → dispatch → verify → learn → respond      │
└───────┬──────────────────────────┬──────────────────┬───────────────┘
        │                          │                  │
        ▼                          ▼                  ▼
┌──────────────┐ ┌──────────────────────┐ ┌────────────────────────┐
│ DecisionEng. │ │ TaskRunner (closed   │ │ StreamingLLM (Ollama)  │
│ IntentAuth   │ │ loop, replan/verify) │ │ agent/streaming_llm    │
│ nlp/ core/   │ │ agent/task_state     │ │ ai/, agent/            │
└──────┬───────┘ └──────────┬───────────┘ └────────────────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐ ┌──────────────────────┐
│ActionDispatch│ │ Vision: screen, OCR, │
│ agent/       │ │ layout, a11y, verify │
│              │ │ services/, vision/   │
└──────────────┘ └──────────────────────┘

Background services (started by engine, never block boot):
  knowledge_service (indexing/retrieval/snapshot)
  background_learner (idle-time self-improvement)
  music_agent, search_service, perception_pipeline
```

---

## Startup Flow

`main.py` is the CLI entry point; `Diego.py` owns the runtime.

```
python main.py
  ├─ env fixes (ALSA/Pulse/JACK/HF offline/display)
  ├─ auto mic probe (non-fatal)
  ├─ default: ui.__main__.main()
  │    └─ Qt app starts on MAIN thread
  │         ├─ DiegoPipeline background thread → Diego.run()
  │         └─ GUI dispatcher pump timer (gui.start + Qt timer)
  └─ --headless: Diego.run() on main thread

Diego.run() → _main_async()
  ├─ signal handlers (SIGINT/SIGTERM)
  ├─ run_Diego()
  │    ├─ run_runtime_health()          (ONCE, never retries)
  │    ├─ agent_brain.initialize()      (wires planner/dispatcher/verifier/learning/perception/decision)
  │    ├─ asyncio.create_task(_warm_llm())  (non-blocking Ollama warm-up)
  │    ├─ command_router.wire(dispatcher, engine)
  │    ├─ music_agent.initialize()
  │    ├─ auth wiring: --no-auth → set_auth_disabled(); else set_auth_provider(authenticate_on_wake)
  │    ├─ background_learner.start()
  │    └─ conversation_engine.run(no_wake)
  └─ shutdown() (≤15s hard timeout; session recorder saver; engine/tts/stt/audio/faceauth close)
```

Key invariants:
- Camera opens ONLY in `FACE_AUTH` state, after a verified wake.
- There is NO startup authentication.
- `--no-auth` clears the auth provider, the only way to disable auth.
- `--no-wake` bypasses only wake; auth still runs unless `--no-auth` also given.
- If the wake model fails to load at boot, `_wake_active=False` → documented always-LISTEN wake fallback; auth stays ACTIVE.

---

## Voice Pipeline

```
Microphone
  → AudioManager (voice/audio_manager.py)
      → select/persist verified device, ring buffer, AGC/preprocessing
  → Wake (voice/wake_listener.py + wake_model_manager.py)
      → openWakeWord streaming predict + Whisper transcript verification
  → Face Auth (auth/) — only when session expired
      → live camera popup → known-encoding match → user name
  → VAD (voice/vad.py — unified, shared by wake + command)
  → STT (voice/command_listener.py)
      → streaming Whisper (partials + final endpoints + confidence)
  → normalization (nlp/command_normalizer.py + brain)
  → intent authorization (nlp/intent_authorizer.py)
  → DecisionEngine (core/decision_engine.py)
  → AgentBrain.process_command()
  → response text
  → TTS (voice/streaming_tts.py) — interruptible, pause/resume
  → next listening cycle
```

Per-turn flow is driven by `ConversationEngine._conversation_session()`:
LISTEN → (final utterance) → THINK (`agent_brain.process_command`) → SPEAK (`_think_and_speak`) → back to LISTEN. 60s silence / goodbye phrase returns to IDLE.

Interruption: while SPEAK, a `speech_start` STT event stops TTS immediately (`_watch_interruption`).

---

## Wake System

| Component | File | Role |
|---|---|---|
| `WakeListener` | voice/wake_listener.py | Primes audio buffers, runs openWakeWord predict loop, verifies with Whisper, returns `WakeEvent`. |
| `WakeModelManager` | voice/wake_model_manager.py | Loads/resolves bundled wake models + per-phrase verifier; streaming predict; threshold; false-positive/reject recording. |
| `WakeModelResolution` | voice/wake_resolver.py | Resolves model path / verifier path / phrase from `models/wake` and `_bundled_models`; `resolve_wake_model()`, `resolve_verifier()`. |
| `verify_wake_transcript` | voice/wake_word.py | Fuzzy + phonetic transcript verification against distinguished wake words. |

Runtime states:
- `IDLE` — waiting for wake (or session ended).
- `WAKE` — openWakeWord streaming; only AudioManager/VAD/wake/Whisper verification may run; LLM/TTS MUST NOT run.
- Wake fallback: if model unavailable at boot → `_wake_active=False` → engine enters always-LISTEN mode; auth remains active.
- `--no-wake` flag bypasses the WAKE state entirely (still runs FACE_AUTH unless `--no-auth`).

---

## Authentication

| Component | File | Role |
|---|---|---|
| `authenticate_on_wake` | Diego.py | Engine auth provider; runs `auth.live_auth.authenticate_live` in executor; returns name or None. NEVER raises. |
| `auth.live_auth` | auth/live_auth.py | Face detection (YuNet/HOG), landmark quality (eyes/pose/blur), encoding match, liveness checks, popup via `face_popup`. |
| `FaceAuthService` | auth/auth_service.py | Optional service wrapper: enabled, authenticated_user, needs_auth, monitor_camera. |
| `faceauth` | auth/faceauth.py | Legacy catch-all: camera enumeration, exposure, debug overlay, encoding migration. |
| `robust_auth` | auth/robust_auth.py | Alternative liveness (moire/texture/motion, head pose) — used by tests. |
| `FaceDetector` | auth/face_detector.py | Backends: YuNet ONNX, HOG, CNN; frame quality metrics. |
| `face_popup` | auth/face_popup.py | Frameless Qt popup that renders capture preview + status. |

Behavior:
- `ConversationEngine._face_auth_gate()` is the single auth entry for every path (wake, wake bypass, wake-degraded).
- Session: `AUTH_SESSION_S = 600.0` (10 min). After expiry, re-auth required.
- Failure behavior: failure/denial is logged honestly; **auth provider stays active and is NEVER silently disabled**. Only explicit `--no-auth` disables it. Session continues unauthenticated.
- Startup: NO camera access.

---

## Audio

### Input
- `voice/audio_manager.py`:
  - Enumerates input devices, probes for real analog input (rejects virtual/dead buses), auto-selects and persists verified device.
  - Ring buffer (`RingBuffer`), 16kHz resampling, AGC + spectral gate + high-pass (see `audio_processing.py`), energy threshold, frame-id freshness tracking.
  - `start()/stop()/switch_input_device()`, `get_recent_audio()`, `record_command()`, diagnostics.
- `voice/device_manager.py`:
  - Persists selection to `data/audio_devices.json` / `data/mic_selection.json`; input/output device resolution; interactive `select_microphone_interactive()`.

### Output
- `voice/streaming_tts.py`:
  - `StreamingTTS` picks engine (Kokoro → Piper → pyttsx3 → XTTS fallback chain), resamples to device, plays via `_InterruptiblePlayer`.
  - Output device can be switched at runtime; `output_device` used for wake chime too.
- `voice/tts/` — engine abstraction (`BaseTTSEngine`, `KokoroEngine`, `TTSManager`).

### Sample rates
- Mic capture resampled to 16 kHz for VAD/STT/wake.
- TTS output resampled to the selected output device's rate.

### Preprocessing
`voice/audio_processing.py`:
- `AudioPreprocessor`: high-pass, spectral gate, noise estimation, int16/float32 conversions, `is_speech` classification.
- `AutomaticGainControl`: leveler (target RMS ~0.10, band 0.08–0.12) + limiter (0.95) — never a hard clip.
- `peak_monitor` (imported by engine) for audio-level visualization.

### VAD audio path
Engine boots `voice/vad.py` `UnifiedVAD` ONCE. VAD decides:
- Wake frames speech presence.
- Partial/final utterance gating within `CommandListener` (speech evidence, hold time, silence endpoints).

### TTS playback path
`_think_and_speak()`:
1. Pause command listener (mute STT).
2. Drain stale STT events.
3. Start interruption monitor.
4. `streaming_tts.speak_sentences(sentences(), interrupt)` with 120s timeout.
5. After playback: wait 0.5s echo decay, resume listener, drain ring buffer.

---

## STT

- Primary: `voice/command_listener.py` `CommandListener` (streaming faster-whisper).
  - `stream_utterances()` yields `speech_start` / `partial` / `final` / `failure` events.
  - Partial scheduling with merging; endpointing via silence + hold; utterance min-duration 150ms.
  - Transcript validation: confidence (`avg_logprob`), language probability, repeated-hallucination rejection, filler/garbage detection.
  - `is_filler()`, `is_garbage()`, `is_low_quality_transcript()` used by engine/Brain.
- Alternative providers (not active in current production path, used by benchmarks/experiments):
  - `voice/providers/whisper_provider.py`, `sherpa_base.py`, `sherpa_providers.py`, `nemotron_provider.py`.
- `voice/streaming_stt.py` — a parallel earlier streaming implementation; kept for tests/experiments (see Legacy section).
- Session: `CommandListener.initialize()` loads the Whisper model once; destroyed/stopped after each conversation session (`stop_streaming()`); re-created on next session.

Confidence/evidence: `UtteranceEvent` carries `confidence` (avg_logprob), `audio_duration_ms`, `endpoint_reason`, `whisper_latency_ms` → forwarded to Brain for intent-gate evidence.

---

## NLP / Intent

| Module | File | Role |
|---|---|---|
| `command_normalizer` | nlp/command_normalizer.py | Canonicalizes spoken commands: app aliases, verb normalization, noise removal, follow-ups, music/search detection. |
| `intent_authorizer` | nlp/intent_authorizer.py | `authorize_intent()` → `IntentCategory` (DETERMINISTIC_COMMAND/VISION_COMMAND/SEARCH_REQUEST/CONVERSATIONAL/KNOWLEDGE_QUESTION/FOLLOW_UP/MULTI_STEP_TASK/UNCERTAIN) with confidence, actionable flag, route. Boundary BEFORE any expensive work. |
| `intent_gate` | nlp/intent_gate.py | Cheap defensive gate `evaluate_intent()` / `transcript_allows_tool_execution()`. |
| `classifier`/`inference`/`trainer`/`evaluator`/`embeddings` | nlp/ | Optional trained intent model (classifier + embeddings + metadata); used by `main.py --train/--status/--benchmark`. |
| `DecisionEngine` | core/decision_engine.py | Routes to: simple command / conversational / knowledge-base / search / vision / multi-step task / LLM. LLM is last resort. Web-search context injection for LLM path. |

Routing hierarchy (in `AgentBrain.process_command`):
1. `command_normalizer.normalize()`
2. transcript-quality guard (cheap rejection)
3. pending-confirmation continuity + task follow-up continuity
4. `intent_authorizer.authorize_intent()` — actionable/UNCERTAIN boundary
5. `intent_gate.evaluate_intent()` (defense in depth)
6. personality conversation-first
7. `DecisionEngine.decide()` (fast, no perception)
8. demand-driven perception (only if screen context needed) then re-decide
9. simple action dispatch OR planner-generated closed-loop task
10. LLM response only when nothing else resolves

---

## Agent

### Brain
`agent/brain.py` — `AgentBrain`, singleton `agent_brain`.

Entry: `process_command(text, stt_confidence, audio_duration_ms) → CommandResult`
Sole orchestrator. Also `process_goal()` (multi-task DAG), `initialize()`, `cancel_goal()`, `get_progress()`, `close()`.

### Planner
`agent/planner.py` — `AgentPlanner` (LLM plan generator). Creates plans only, never executes. `generate_plan_only()`, `process_request()`.

### ActionDispatcher
`agent/action_dispatcher.py` — `ActionDispatcher`, singleton `action_dispatcher`.
Executes: open/close/focus apps, browser navigate/search, web search, click text/type/scroll/key_press, volume/brightness, media (MPV/Spotify/YouTube/local via `services/music_agent.py`), screen reading, list windows, lock/power, run commands.
`execute(action_dict) → (result_str)`, `screen_context()`.

### TaskRunner (closed loop)
`agent/task_state.py`:
- `TaskRunner.run(request, plan, inherited)` — execute → observe → verify → replan → repeat.
- Includes: `PlanValidator` (schema + verb evidence), `IdempotencyChecker`, `LoopDetector`, `FinalStatus` enum, evidence tracking, pending-confirmation pauses for sensitive actions.
- `TaskStateStore` preserves state for follow-ups ("continue", "do the same for Chrome", "close that").

### TaskController / TaskExecutor
`agent/task_controller.py` — `AutonomousTaskController` (autonomy boundaries, evidence-first verifier, world-state gatherer). `agent/task_executor.py` — DAG execution with parallel groups, timeouts, cancel-downstream.

### Verification
`vision/action_verifier.py` — `ActionVerifier` compares pre/post screen snapshot per action type (open_app/navigate/click/type/scroll/key_press). `AgentBrain._verify()` additionally uses OS-level `pgrep -x` process checks and zombie detection for app open/close.

### Replanning
On failure with retries remaining, Brain adjusts params (`_adjust_params_for_retry`) and retries; TaskRunner re-plans from CURRENT observed state via `_plan_with_context()`, bounded by `TaskLimits`.

### Confirmation
Sensitive actions (`is_sensitive_action`) pause the task and ask the user. `Brain._handle_pending_confirmation()` resumes the SAME task on "yes"/"play it"/"do it"; cancels on "no"/"cancel". YouTube playback always asks before playing.

### Autonomy limits
- Planner actions gated by `_PLANNER_ACTION_SCHEMA` + verb evidence (`_planner_action_allowed`).
- Intent authorizer: ONLY actionable categories reach the planner/dispatcher.
- Replans capped (`TaskLimits`); LoopDetector stops repeating identical failed actions; `IdempotencyChecker` avoids redoing completed steps.

---

## Desktop Actions

Implemented in `agent/action_dispatcher.py`:

- Applications: `desktop_open`, `close_app`, `focus_app`, `switch_window`, `list_windows`, `minimize/maximize`.
- Windows/workspace: `switch_workspace`, `minimize_window`, `maximize_window`, `switch_tab`.
- Volume: `volume_up/down/set/mute` (amixer/pactl).
- Brightness: `brightness_up/down/set` (backlight + brightnessctl fallback).
- Network: not a dedicated module; browser-based web actions (`browser_navigate`, `browser_search`, `web_search`, `web_search_open_best`).
- Browser: `agent/browser.py` `BrowserController` (CDP attach/persistent Chrome) for tab control, text extraction, screenshots, type/click.
- Media: `services/music_agent.py` (MPV, Spotify via playerctl, YouTube via youtube_search + playback, local files).
- System information: `knowledge/system_info.py` reads PC snapshot (CPU/RAM/disk/GPU/temp) deterministically.
- Power: `lock_screen`, `shutdown`, `restart`.
- Screen: `read_screen` (OCR), `screenshot`.

Verification per action type: OS-level for apps/browsers, vision diff for UI actions, dispatch-result trust fallback (never trusts "Couldn't…" / known no-op strings).

---

## Vision

| Module | File | Role |
|---|---|---|
| `perception_pipeline` | services/perception_pipeline.py | Perceives window/a11y/screen into a `PerceptionContext` (`compact_summary`); OCR demand-driven. |
| `screen_capture` | services/screen_capture.py | Captures screen (mss) for vision service. |
| `vision_service` | services/vision_service.py | Orchestrates capture→OCR→layout→a11y; `analyze_forensic()`. |
| `OCR` | vision/ocr_pipeline.py | Backends: PaddleOCR / EasyOCR / Tesseract (bundled `data/tessdata`). Dedup/merge/classify boxes; `OCRResult`. |
| `a11y` | services/accessibility.py, services/ui_tree.py | AT-SPI accessibility tree traversal & element classification. |
| `layout` | vision/layout_analyzer.py | Segments screen into regions, detects application type. |
| `action_verifier` | vision/action_verifier.py | Pre/post screen diff for action verification. |
| `screen_memory` / `frame_differencer` / `forensic_logger` / `debug_overlay` | vision/ | History, gating (motion-based), forensic reports, debug overlay. |

OCR is NEVER invoked unless the request needs to read the screen (explicit vision phrases/deixis). The whole perception stage is hard-bounded at `PERCEPTION_TIMEOUT_S = 15.0`.

Live-state handling: requests about current apps/windows/processes are routed to live tools (`_is_live_state_request`), never to stale local knowledge.

---

## Knowledge

| Module | File | Role |
|---|---|---|
| `knowledge_service` | knowledge/service.py | Singleton facade: background indexing, `search()`, `context_for_llm()`, start (daemon threads, never blocks boot). |
| `policy` | knowledge/policy.py | Scanning boundary: which directories/extensions are eligible. |
| `indexer` | knowledge/indexer.py | Walks allowed dirs, indexes files. |
| `extractors` | knowledge/extractors.py | Text extraction per file type. |
| `chunker` | knowledge/chunker.py | Splits documents into chunks. |
| `embedder` | knowledge/embedder.py | Embedding generation. |
| `retriever` | knowledge/retriever.py | Retrieval over the store. |
| `store` | knowledge/store.py | Persistent store (DuckDB-backed via memory/duckdb_store). |
| `presentation` | knowledge/presentation.py | Synthesizes concise spoken answers with citations; `sanitize_spoken()` scrubs paths/scores metadata. |
| `system_info` | knowledge/system_info.py | PC snapshot collector (CPU/RAM/disk/GPU/temp); deterministic answers to "system info" queries. |
| `diagnostics` | knowledge/diagnostics.py | Live read-only diagnostics collector; answers "is Diego healthy?" etc. without executing repairs or leaking internals. |

Flow: `Brain._generate_response()` → checks system-info guard → diagnostics guard → local-knowledge (unless small-talk) → LLM with local context. Live-state requests bypass local retrieval. Small-talk never triggers local document matches.

---

## Diagnostics

- `core/runtime_health.py` — `run_runtime_health()` reports `[HEALTH] component=OK/DEGRADED/MISSING` for every production component at boot (once).
- `core/benchmark.py` — per-turn latency metrics (STT/LLM/router/turn).
- `core/metrics.py` — counters/gauges used by health/benchmark.
- `core/runtime_health` + `knowledge/diagnostics.py` expose live read-only diagnostics.
- Latency: `Brain._pipeline_timings` records per-stage latency; engine logs per-state duration; `benchmark.record_turn()`, `session_recorder` records wake/utterance/decision/turn latencies.
- Context monitor: `ai/context_monitor.py` estimates/trims LLM context (`ContextMonitor`).

---

## LLM

- `agent/streaming_llm.py` — `StreamingLLM`:
  - Ollama streaming client with sentence-prefix generation.
  - `warm_up()` warm-up (settings `LLM_WARMUP_ENABLED/TIMEOUT_S/VISION`).
  - Vision model selection (LLM_VISION_MODEL) when screen context needed.
- `ai/llm_client.py` — non-streaming client for goal decomposition / `llm_chat()`.
- Context limits handled by `ai/context_monitor.py` (`estimate_tokens`, `trim_to_fit`).
- Fallback: unavailable -> system-info/local-knowledge/personality canned responses; LLM failure -> `"I'm having trouble with that right now."` via `_generate_response`.

---

## UI

Qt PySide6 application under `ui/`:

| File | Role |
|---|---|
| `__main__.py` | Creates QApplication on main thread, starts `DiegoPipeline` background thread running `Diego.run()`, starts GUI dispatcher pump. |
| `main_window.py` | Main window: transcript, response, state indicator, voice visualizer, latency cards, audio device panel. |
| `widgets.py` | `TranscriptPanel`, `ResponsePanel`, `VoiceStatePanel`, `ActivityPanel`, `MetricsCards`, `AudioDevicePanel`, `ConnectionIndicator`, waveform, avatar. |
| `visualizer.py` | Animated voice core (states idle/listening/thinking/executing/speaking). |
| `event_bridge.py` | `EventBridge` — receives engine/STT/Brain events → emits Qt signals. `wire_all()` connects engine, STT, brain, event bus. |
| `tokens.py` / `styles.py` | Color tokens & state colors. |

Threading: Qt owns the main thread; engine runs on `DiegoPipeline` background thread; all GUI updates via thread-safe `gui.submit()` (core/gui_dispatcher). Long-running STT/LLM/TTS never runs on the GUI thread.

States shown include IDLE / WAKE / LISTEN / THINK / EXECUTE / SPEAK.

---

## State Machine

ConversationEngine 7-state machine with **no bypass**:

```
        ┌─────────────┐
        ▼             │
      IDLE ──► WAKE ──┴─► FACE_AUTH ──► LISTEN ──► THINK ──► SPEAK ──┐
        ▲                                                            │
        └────────────────────────────────────────────────────────────┘
```

Legal transitions (enforced in `ALLOWED_TRANSITIONS`, logged by `_set_state`):

| From | To |
|---|---|
| IDLE | WAKE, FACE_AUTH (wake bypass/degraded), LISTEN (auth disabled/session valid) |
| WAKE | FACE_AUTH, LISTEN (auth disabled) |
| FACE_AUTH | LISTEN |
| LISTEN | THINK, IDLE (timeout/goodbye) |
| THINK | SPEAK |
| SPEAK | LISTEN (continue conversation), IDLE (end session) |

Watchdogs (`STATE_TIMEOUTS_S`): LISTEN 75s, THINK 60s, SPEAK 120s. On timeout, the watchdog breaks the stalled stream and resets the state timer. WAKE/FACE_AUTH are intentionally unbounded.

---

## Configuration

- `config/settings.py` (`Settings` singleton):
  - Paths: `CLASSIFIER_PATH`, `METADATA_PATH`, `KNOWN_ENCODINGS_PATH`, model dirs.
  - OLLAMA: `OLLAMA_BASE_URL`, `OLLAMA_KEEP_ALIVE`, LLM_WARMUP_*.
  - Optional: HF offline flags.
- Environment overrides:
  - `DIEGO_RECORD_SESSION=1/true/yes` enables session recording.
  - Same ALSA/JACK/PulseAudio debug-suppression envs set in `main.py`/`Diego.py`.
- `voice/settings.py` — `VoiceSettings` (VAD thresholds, etc., env-overridable).
- Device persistence: `data/audio_devices.json`, `data/mic_selection.json` (written by `device_manager`).
- Model paths: `models/` for wake (bundled + custom), platform paths for Whisper (faster-whisper cache), Kokoro/TTS cache, `data/tessdata` for Tesseract.

No secrets or credentials are stored in this repository.

---

## Data / Models

| Asset | Location / Source | Used by |
|---|---|---|
| Wake models (openWakeWord bundled) | `models/wake/` (bundled) + custom phrase models | `WakeModelManager`, `wake_resolver` |
| Whisper (faster-whisper) | ~/.cache (HF cache) | `command_listener`, wake verifier |
| TTS: Kokoro / Piper / pyttsx3 / XTTS | packages + cached models | `StreamingTTS` |
| OCR: Tesseract | `data/tessdata/eng.traineddata` (bundled) | `EnhancedOCREngine` |
| OCR: PaddleOCR / EasyOCR | installed packages | `EnhancedOCREngine` |
| DuckDB | DuckDB file (persistent store) | `memory/duckdb_store`, `memory/unified_memory`, knowledge store |
| Embedding cache (MiniLM-L6-v2) | HF cache (offline) | `nlp/embeddings`, `knowledge/embedder` |

---

## Dependencies

- **Required** (runtime core): numpy, sounddevice, faster-whisper, kokoro, torch (VAD), openwakeword, PySide6 (UI), requests/httpx, duckdb, sentence-transformers, Pillow, mss (screen capture), pytesseract.
- **Optional**: sherpa-onnx providers, nemotron ASR, EasyOCR/PaddleOCR, mpv/playerctl (media), spacy (legacy).
- **Dev-only**: pytest, debug/ scripts, requirements.*-lock files.

_Note: exact versions are enforced in `requirements.txt` and `requirements.runtime-lock.txt`; `requirements.before-cuda-repair.txt` is a historical snapshot._

---

## CLI / Entry Points

| Command | Effect |
|---|---|
| `python main.py` | Start Diego (UI by default; `--headless` for CLI runtime). |
| `python main.py --help` | All flags. |
| `python main.py --train` | Train NLP classifier, exit. |
| `python main.py --status` | NLP model status. |
| `python main.py --benchmark` | NLP classification benchmarks. |
| `python main.py --select-mic` | Interactive mic selection. |
| `python main.py --audio-debug` | Live audio level visualizer. |
| `python main.py --train-wake` | Record wake phrases + train verifier. |
| `python Diego.py` | Conversational runtime (same as `main --headless`). |
| `python Diego.py --no-auth/--no-wake/--status` | Dev bypasses / subsystem status. |
| `python Diego.py debug vision` | Live vision debug overlay. |
| `python Diego.py inspect screen` | Full screen inspection report. |
| `python ui/__main__.py [--no-wake] [--no-auth]` | Start the UI directly. |
| `python -m nlp.trainer` | (dev) training utilities. |

---

## Testing

- `tests/` contains 30+ test modules covering: voice pipeline (wake, audio, command listener), UI, NLP, intent authorization, task continuation/controller, knowledge indexing, runtime health, startup auth independence, production readiness, integration fixes, response guarantee, silence UX, event bus, duckdb store, system info.
- `conftest.py` at repo root provides pytest fixtures/environment.
- `pytest.ini` configures test collection.
- Production-path tests exercise real singletons via import (e.g., `test_startup_auth_independence.py`, `test_wake_resolver.py`, `test_command_listener.py`).
- Run: `pytest -q`; compile check: `python -m compileall .`.

---

## Runtime Failure Handling

- Boot: any subsystem load failure is logged and Diego continues (runtime health reports OK/DEGRADED/MISSING).
- Audio: retried every 5s until available; if never available, engine stops.
- Wake model missing: wake DEGRADED → always-LISTEN fallback; auth unaffected.
- STT unavailable: engine speaks "My speech recognizer isn't available right now." and returns to IDLE.
- TTS failure: response_guarantee speaks a fallback; TTS timeout (120s) recovers by stopping TTS.
- Brain/perception: all hard-bounded timeouts; on failure returns canned recovery responses.
- Shutdown: hard 15s timeout; leftover tasks drained.
- LLM: warm-up fail-safe; generation failures → canned error response.

---

## Security / Read-only Boundaries

- Knowledge scanning is read-only for eligible directories (`knowledge/policy.py`); never modifies user files.
- System/diagnostic questions are answered from read-only collectors; Diego NEVER executes repair actions in response.
- Spoken responses never leak: raw paths, JSON, stack traces, retrieval scores, database rows, or internal metadata (`knowledge/presentation.sanitize_spoken`).
- Face encodings read-only; camera only in FACE_AUTH.
- `--no-auth`/`--no-wake` are dev-only bypasses; in production auth remains enabled by default.
- Auth failure is NEVER silently converted to disabled auth.
- Subprocess commands are constructed from allow-listed schemas (ActionDispatcher).

---

## Known Limitations (current)

- Speech recognition is local Whisper; noisy environments may produce low-confidence transcripts that are rejected.
- Perception/OCR is demand-driven and hard-bounded; deep UI navigation may rely on repeated small steps.
- Multi-step task execution is sequential (ready tasks executed one at a time; TaskExecutor parallel groups exist but Brain's `process_command` uses TaskRunner).
- Some legacy/parallel implementations remain in tree (see Legacy section) — they are not part of the active path.
- Wake model availability depends on bundled openWakeWord assets in `models/wake`.
- Device selection is persisted per-machine; first run may probe/auto-select.

---

## File/Module Reference

Complete module-by-module inventory of production files (line counts approximate, from AST scan).

### Top-level
- `main.py` — CLI entry; utility commands; mic probe; UI/headless dispatch.
- `Diego.py` — canonical blocking runtime entry (`Diego.run()`), auth provider, shutdown, debug/inspect subcommands.
- `compat.py` — Python 3.14 stdlib compatibility shims (audioop/imghdr/aifc stubs).
- `Listen.py` — legacy micro entry (`find_executable`) — see Legacy.

### agent/
- `action_dispatcher.py` — all desktop actions.
- `brain.py` — orchestrator.
- `browser.py` — Chrome CDP controller.
- `code_assistant.py` — code analysis (errors, tests, diffs).
- `context_composer.py` — memory ranking/compression for LLM pipe.
- `conversation_memory.py` — short-term conversation memory.
- `executor.py` — low-level executor (browser/desktop/clipboard/keyboard/mouse) legacy-ish.
- `goal_manager.py` — goal/task persistence.
- `memory.py` — step/confirmation memory for the earlier task agent.
- `personality.py` — canned responses, greetings, sentiment.
- `planner.py` — LLM plan generation.
- `proactive_agent.py` — idle proactive suggestions (EventBus hooks).
- `project_mode.py` — project context (IDE, repo, recent files).
- `streaming_llm.py` — streaming Ollama client.
- `task_continuation.py` — pending confirmation manager.
- `task_controller.py` — autonomous controller, evidence verifier.
- `task_executor.py` — DAG task execution.
- `task_state.py` — closed-loop TaskRunner/PlanValidator/LoopDetector.

### ai/
- `context_monitor.py` — LLM context estimation/trimming.
- `llm_client.py` — non-streaming LLM client (goal decomposition fallback, `llm_chat`).

### auth/
- `auth_service.py`, `encode.py`, `face_detector.py`, `face_popup.py`, `faceauth.py`, `live_auth.py`, `robust_auth.py`.

### config/
- `settings.py` — Settings singleton.

### core/
- `autonomous_reasoning.py` — context collector (editor/terminal/browser errors).
- `background_agent.py` — rule-based background observer (battery, git, clipboard).
- `background_learning.py` — idle learner.
- `background_workers.py` — periodic workers (health check, memory summarize, goals vacuum, experience optimizer, semantic linker, planner optimizer).
- `benchmark.py`, `cache_manager.py`, `command_router.py`, `conversation_engine.py`, `decision_engine.py`, `event_bus.py`, `gui_dispatcher.py`, `manual_session_recorder.py`, `metrics.py`, `plugin_base.py`, `plugin_manager.py`, `response_guarantee.py`, `runtime_health.py`, `service.py`, `startup_health.py`, `state_machine.py`, `tool_registry.py`, `tool_reliability.py`.

### desktop/
- `__init__.py` — empty legacy package (likely leftover).

### knowledge/
- `chunker.py`, `cli.py`, `diagnostics.py`, `embedder.py`, `extractors.py`, `indexer.py`, `policy.py`, `presentation.py`, `retriever.py`, `service.py`, `snapshot.py`, `store.py`, `system_info.py`.

### learning/
- `desktop_layouts.py`, `experience_db.py`, `habits.py`, `learning_engine.py`, `preferences.py`, `skill_memory.py`, `user_profile.py`.

### memory/
- `duckdb_store.py`, `semantic_memory.py`, `unified_memory.py`.

### nlp/
- `classifier.py`, `command_normalizer.py`, `confidence.py`, `context.py`, `conversation_state.py`, `embeddings.py`, `entities.py`, `evaluator.py`, `inference.py`, `intent_authorizer.py`, `intent_gate.py`, `model_metadata.py`, `normalizer.py`, `parser.py`, `tokenizer.py`, `trainer.py`.

### plugins/
- `brightness_plugin.py`, `telegram_plugin.py`, `youtube_plugin.py`, `plugin_base.py` — not imported by any production module (see Legacy).

### runtime/
- `status_popup.py` — UI status popup; not imported by current production modules.

### services/
- `accessibility.py`, `desktop_observer.py`, `desktop_state.py`, `music_agent.py`, `perception_pipeline.py`, `screen_capture.py`, `screen_reasoning.py`, `search_service.py`, `ui_tree.py`, `vision_service.py`.

### telemetry/
- `logger.py` — `setup_logging()` used by Diego.py.

### ui/
- __main__.py, event_bridge.py, main_window.py, styles.py, tokens.py, visualizer.py, widgets.py.

### vision/
- action_verifier.py, debug_overlay.py, forensic_logger.py, frame_differencer.py, layout_analyzer.py, ocr_pipeline.py, perception.py, screen_memory.py.

### voice/
- asr_fallback.py, asr_provider.py, audio_manager.py, audio_processing.py, calibrate_wake.py, command_listener.py, device_manager.py, settings.py, streaming_stt.py, streaming_tts.py, vad.py, wake_listener.py, wake_model_manager.py, wake_resolver.py, wake_word.py, tts/base.py, tts/kokoro_engine.py, tts/manager.py, providers/*.

---

## Legacy / Dead Code

The following are NOT part of the current production voice/command path, based on import analysis and caller inspection.

| Module | Reason considered legacy |
|---|---|
| `desktop/` (empty package) | No production module imports it; only a placeholder `__init__.py`. |
| `plugins/` (brightness, telegram, youtube, plugin_base, plugin_manager) | No production or test imports; `plugin_manager.py` references the `plugins` package only in a string; unknown used-for future extensibility. |
| `runtime/status_popup.py` | No production import; only referenced in docs/strings. |
| `Listen.py` | Top-level legacy entry with a single `find_executable`; never imported by prod or tests (only itself). |
| `voice/streaming_stt.py` | Parallel older STT implementation. `command_listener` is the active streaming STT. `streaming_stt` still kept for tests/experiments (tests import helpers). |
| `voice/providers/` (whisper_provider, sherpa_*, nemotron) | Provider registry exists but the active path uses `command_listener` + faster-whisper directly. Used by `debug/` benchmarks. |
| `agent/executor.py`, `agent/memory.py` | Used by older/alternative task pipeline; not called from `brain.py` or `conversation_engine.py`. `executor` still may be used by plugins/tests. |
| `vision/perception.py` | Old perception service; current path uses `services/perception_pipeline.py`. |

Debug / dev-only: entire `debug/` directory (diagnostics, benchmarks, repro scripts), `scripts/`, `datasets/`, `telemetry/` (only used by Diego.py logger), `learning/` (used by Brain/background), `memory/` (used by knowledge store).

_Note: "legacy" here means "not on the active production call graph" — not "dead strings." No code was requested to be removed.

---

## Function Reference (production modules)

For every module below, `Class.method(args) → return` plus side effects/fallback/callers are derived from source. This is the authoritative index.

_(Because 165 production modules contain ~2,000 functions, this reference is provided as a structured map of the active production call-graph, covering the core orchestrators and singletons. Remaining modules follow the same patterns described in the File/Module Reference section.)_

### core/conversation_engine.py — ConversationEngine

- `__init__()` — state, wake listener, auth session, TTS interrupt event, GUI pump.
- `set_auth_provider(fn)` — store auth callable.
- `set_auth_disabled()` — clear provider (only --no-auth path); logs.
- `set_authenticated(name)` — update session + conv_memory user name.
- `invalidate_auth()` — expire current session.
- `_needs_auth()` -> bool — provider set? name? session expired?.
- `_set_state(new, **diag)` — enforce ALLOWED_TRANSITIONS; log; store entry time/diag.
- `run(no_wake)` — boot: audio, wake model, VAD, TTS, Whisper; forever: WAKE/FACE_AUTH/(conversation) loop; watchdog.
- `_state_watchdog()` — log STATE TIMEOUT + safe recovery (stop_streaming).
- `_setup_gui()` — bind GUI dispatcher per host (Qt vs headless vs none).
- `_ensure_wake_model()` -> bool — load once; log error.
- `_run_auth()` -> Optional[str] — call provider; log error.
- `_face_auth_gate(trigger)` — auth if needed; on success greet (guarded TTS); on failure log honest, keep provider.
- `_wake_listen_loop()` -> Optional[WakeEvent].
- `_play_wake_chime()` — wave play to TTS device.
- `_conversation_session()` — LISTEN→THINK→SPEAK loop; identity/goodbye; response_guarantee.run_turn; drain stale events.
- `_stt_event_pump(stream, events)` — queue events.
- `_think_and_speak(user_text, events, canned)` -> bool — SPEAK via TTS; duplicate-text guard; interruption monitor; echo-decay.
- `_watch_interruption(events)` — speech_start during SPEAK → stop TTS.
- `_one_line_stream(text)` — abbreviation-aware sentence splitting.
- `_speak_line(text)` -> bool.
- `_speak_guarded(text)` -> bool — pause listener, speak, echo-decay, drain.
- `_speak_failure_response(reason)` -> bool.
- `_is_identity_question(text)` -> bool.
- `get_diagnostics()` -> dict — state/session/wake/auth/response_guarantee.

### agent/brain.py — AgentBrain

- `initialize()` -> bool — wire subsystems.
- `process_command(text, stt_confidence, audio_duration_ms)` -> CommandResult — full pipeline.
- `process_goal(description, context)` -> Goal — decomposition + DAG execution.
- `_decompose_goal/_build_tasks/_parse_task_json/_fallback_decompose` — goal → tasks.
- `_execute_task_graph(goal)` — sequential DAG execution + retries.
- `_execute_single_task(goal, task)` — run via planner with timeout.
- `_run_through_planner(desc, ctx)` -> (bool, str).
- `_perceive(text, include_ocr)` — bounded perception; `_perception_needed`, `_ocr_required`.
- `_decide(text, perception_ctx, search_context)` — DecisionEngine or LLM fallback.
- `_plan(text, ctx)` / `_plan_with_context(request, ctx)` — planner.
- `_dispatch_and_verify(action)` -> (ok, result) — capture→dispatch→verify→learn; retry≤2.
- `_verify(action, params, result)` -> bool — OS pgrep / vision / trust.
- `_learn(action_name, params, success, error)` — record to learning engine.
- `_run_task_loop(request, plan, inherited, approved)` -> TaskExecutionState.
- `_handle_pending_confirmation(...)` / `_register_pending_confirmation(state)`.
- `_maybe_confirm_youtube_playback(...)`.
- `_generate_response(text, ctx, result)` — system-info/diagnostics/local-knowledge/LLM cascade.
- `cancel_goal(goal_id)` / `get_progress()` / `close()`.

### agent/action_dispatcher.py — ActionDispatcher

- `execute(action)` -> str — validate schema + route to one of ~50 handlers.
- `screen_context()` -> str — current window/app/brief.
- Handlers: `_open_app`, `_close_app`, `_open_url_fallback`, `_browser_*`, `_web_search*`, `_music_action` (music_agent), `_volume_*`, `_brightness_*`, `_lock_screen`, `_power`, `_read_screen`, `_list_windows`, `_click_text`, `_locate_text_*`, `_type_text` (via executor), `_run_cmd`, `_window_action`, `_radio`.

### voice/* — key singletons
- `audio_manager` (AudioManager) — `start/stop/read_since/get_recent_audio/switch_input_device/record_command/calibrate`, diagnostics.
- `wake_listener` (WakeListener) — `prime/wait_for_wake/verify_with_whisper`.
- `wake_model_manager` (WakeModelManager) — `load/detect/predict_stream/close/reload/hard_reset`, thresholds, diagnostics.
- `command_listener` (CommandListener) — `initialize/stream_utterances/pause_listening/resume_listening/stop_streaming/cancel/ready`.
- `streaming_tts` (StreamingTTS) — `initialize/speak_sentences/stop/close/set_output_device`.
- `unified_vad` (UnifiedVAD) — `load/speech_prob/robust_speech_prob/reset_state`.

### knowledge/service.py
- `knowledge_service` — `start()`, `search(query, top_k)`, `context_for_llm(query)`.

---

## Documentation sources

This document is derived from the current source tree (AST scan, import graph, and direct reading of production modules). It is the single canonical reference; other docs in `docs/` (architecture.md, ui.md, Database.md, benchmarks) are secondary/historical.
