# Leo Desktop Assistant — Comprehensive Audit Report v2

## 1. Startup Observations (Live Run: 2026-07-28)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total startup time | ~2.5s | <2s | ⚠ |
| NLP load | 0.1s (cached) | <0.5s | ✓ |
| DuckDB connect | 0.02s | <0.1s | ✓ |
| Plugin load | 0.8s (Telegram: 0.7s) | <0.5s | ⚠ |
| Voice ready | FAILED (PyAudio missing) | <500ms | ✗ |
| Face Auth | Unavailable (cv2 import) | N/A | ⚠ |
| Vision | Unavailable | N/A | ⚠ |

## 2. Critical Runtime Crashes (PHASE 1)

### 2.1 PyAudio Not Installed — Voice Completely Dead
- **File**: `voice/stt.py:34-42`
- **Error**: `Could not find PyAudio; check installation`
- **Impact**: Voice subsystem entirely disabled. No STT, no wake word, no TTS.
- **Root Cause**: PyAudio 0.2.14 in requirements.txt but not actually installed in environment.
- **Fix**: Install PyAudio or use sounddevice/portaudio fallback.

### 2.2 Telegram Plugin — Telethon API Version Mismatch
- **File**: `plugins/telegram_plugin.py:62-63`
- **Error**: `too many values to unpack (expected 5, got 6)`
- **Impact**: Telegram client non-functional. Plugin loads but all Telegram intents fail silently.
- **Root Cause**: Telethon 1.42.0 changed return signature of internal method.
- **Fix**: Pin Telethon version or patch the unpack.

### 2.3 audioop Stub — Incomplete Implementation
- **File**: `compat.py:62-96`
- **Problem**: `audioop.lin2lin` returns input unchanged. `audioop.ratecv` returns (d, s) but speech_recognition expects real resampling.
- **Impact**: Audio processing may produce corrupted data at runtime.
- **Fix**: Use `audioop-lts` PyPI package instead of custom stub.

### 2.4 aifc Stub — Duplicate Injection
- **Files**: `compat.py:19-49` and `voice/stt.py:19` (both inject aifc)
- **Problem**: Duplicate stub injection. `voice/stt.py` imports compat but main.py already injected.
- **Impact**: Redundant code, potential for stub conflicts.

### 2.5 No Graceful Shutdown on SIGTERM/SIGINT
- **File**: `main.py:402-412`
- **Problem**: `shutdown_gracefully` catches all exceptions silently. DuckDB WAL may be corrupted.
- **Fix**: Add WAL checkpoint before close. Log all shutdown errors.

### 2.6 Plugin Crash Can Hang Main Loop
- **File**: `core/plugin_manager.py:113-139`
- **Problem**: `handle_event` uses `asyncio.wait_for` but plugin_manager.py line 128 has `asyncio` not imported at module level.
- **Impact**: `NameError: name 'asyncio' is not defined` if any plugin times out.
- **Fix**: Add `import asyncio` at top of plugin_manager.py.

## 3. Architecture Issues (PHASE 2)

### 3.1 No Voice Pipeline State Machine
- **Current**: Ad-hoc `if voice_ok` checks in main loop.
- **Required**: Deterministic state machine: BOOT → INIT → LOAD → READY → AUTH → GREETING → WAKEWORD → LISTEN → STT → NLP → PLUGIN → TTS → WAKEWORD
- **Files**: `main.py:332-399`

### 3.2 No Voice Supervisor
- **Current**: Voice components (STT, TTS) are standalone functions with global state.
- **Required**: `VoiceSupervisor` managing `MicrophoneManager`, `WakeWordEngine`, `SpeechRecognizer`, `SpeechSynthesizer`, `AudioDeviceManager`, `NoiseCalibrator`, `VoiceSettings`.

### 3.3 No Dependency Injection
- **Current**: Global singletons everywhere (`bus`, `plugin_manager`, `inference`, `store`, `llm_client`).
- **Required**: DI container for testability and isolation.

### 3.4 No Correlation IDs
- **Current**: Log messages have no request tracing.
- **Required**: Correlation ID per user query, propagated through all subsystems.

### 3.5 No Audio Backend Auto-Detection
- **File**: `voice/tts.py:49-54`
- **Problem**: Hardcoded to `paplay` (PulseAudio only).
- **Required**: Auto-detect ALSA/PulseAudio/PipeWire/JACK.

### 3.6 faceauth.py Uses Separate TTS Pipeline
- **File**: `auth/faceauth.py:23-31`
- **Problem**: Uses `pyttsx3` instead of main TTS pipeline.
- **Fix**: Use `voice/tts.speak()`.

### 3.7 No Plugin Health Checks
- **File**: `core/plugin_base.py`
- **Problem**: No `health_check()` method. No way to detect hung plugins.
- **Fix**: Add health check to BasePlugin.

### 3.8 No Plugin Dependency Validation
- **File**: `core/plugin_manager.py`
- **Problem**: `PluginMetadata.dependencies` defined but never validated.
- **Fix**: Validate dependencies before loading.

## 4. Performance Issues (PHASE 3)

### 4.1 Startup Time >2s
- **Target**: <2s
- **Current**: ~2.5s (Telegram init dominates at 0.7s)
- **Fix**: Lazy-load Telegram. Defer non-critical initialization.

### 4.2 Heavy Imports at Module Level
- **Files**: `auth/faceauth.py:4-10` — cv2, face_recognition, firebase_admin at module level.
- **Impact**: Even if disabled, these modules are loaded on import.

### 4.3 No DuckDB Connection Pooling
- **File**: `memory/duckdb_store.py:135-150`
- **Problem**: `connect()` creates new connection each time.
- **Fix**: Use persistent connection with retry.

### 4.4 No Async Database Operations
- **File**: `memory/duckdb_store.py`
- **Problem**: All DB ops are synchronous, blocking event loop.
- **Fix**: Use `asyncio.to_thread()` for DB operations.

### 4.5 TTS Speed Not Applied
- **File**: `voice/tts.py:48`
- **Problem**: `settings.TTS_SPEED` defined but never passed to TTS engine.
- **Fix**: Pass speed parameter to `tts.tts_to_file()`.

## 5. Voice Subsystem Issues

### 5.1 No Audio Device Recovery
- **File**: `voice/stt.py:70-85`
- **Problem**: If microphone fails during calibration, `voice_ok` stays False forever.
- **Fix**: Periodic retry with exponential backoff.

### 5.2 No Offline Wake Word Engine
- **Current**: Uses Google Speech API for wake word (requires internet, high latency).
- **Required**: Offline wake word engine (Porcupine, Vosk, or custom).

### 5.3 No Noise Calibration Persistence
- **File**: `voice/stt.py:70-85`
- **Problem**: Calibration runs every startup. No persistence of noise profile.
- **Fix**: Cache noise profile to disk.

### 5.4 No Voice Settings in .env
- **Current**: Only `TTS_SPEED`, `TTS_VOLUME`.
- **Required**: `VOICE_RATE`, `VOICE_VOLUME`, `VOICE_PITCH`, `VOICE_ID`.

## 6. Database Issues

### 6.1 No WAL Recovery on Startup
- **File**: `memory/duckdb_store.py:87-96`
- **Problem**: Stale WAL files prevent startup.
- **Fix**: Checkpoint on shutdown, recover on startup.

### 6.2 No Concurrent Read/Write Support
- **File**: `memory/duckdb_store.py`
- **Problem**: Single connection. Concurrent reads block writes.
- **Fix**: Read-only connections for queries, write connection for mutations.

### 6.3 No Automatic Retry on Transient Failures
- **File**: `memory/duckdb_store.py:98-133`
- **Problem**: Retry only on connection, not on individual queries.
- **Fix**: Add retry decorator for all DB operations.

## 7. Plugin System Issues

### 7.1 Missing `asyncio` Import in plugin_manager.py
- **File**: `core/plugin_manager.py:128`
- **Problem**: `asyncio.wait_for` used but `asyncio` not imported.
- **Impact**: Runtime `NameError` on plugin timeout.
- **Severity**: CRITICAL

### 7.2 No Plugin Timeout Configuration
- **File**: `core/plugin_manager.py:113`
- **Problem**: Timeout hardcoded to 30s.
- **Fix**: Make configurable via settings.

### 7.3 No Plugin Event Filtering
- **File**: `core/plugin_manager.py:124-139`
- **Problem**: All events dispatched to all plugins. No filtering.
- **Fix**: Only dispatch events that plugin subscribes to.

## 8. Logging Issues

### 8.1 No Correlation IDs
- **File**: `telemetry/logger.py`
- **Problem**: Can't trace single user request through system.
- **Fix**: Add correlation ID to log context.

### 8.2 No Latency Metrics
- **File**: `telemetry/logger.py`
- **Problem**: No timing information in logs.
- **Fix**: Add duration to each log entry.

### 8.3 No Memory Metrics
- **File**: `telemetry/logger.py`
- **Problem**: No memory usage tracking.
- **Fix**: Periodic memory snapshots.

### 8.4 No Subsystem ID in Logs
- **File**: `telemetry/logger.py`
- **Problem**: Can't filter logs by subsystem.
- **Fix**: Add subsystem field to structured logs.

## 9. Missing Features

### 9.1 No Greeting After Face Auth
- **File**: `main.py:352`
- **Current**: `speak(f"Hello {user_name}, how may I assist you?")`
- **Required**: "Authentication successful. Welcome back {name}. I am ready. How can I help you today?"

### 9.2 No Voice Settings Runtime Configuration
- **Required**: Runtime API to change voice rate, volume, pitch without restart.

### 9.3 No Microphone Hotplug Support
- **Required**: Detect microphone insertion/removal and adapt.

## 10. Test Coverage

| Component | Tests | Coverage | Required |
|-----------|-------|----------|----------|
| NLP | `tests/test_nlp.py` | Basic | ✓ |
| DuckDB | `tests/test_duckdb_store.py` | Basic | ✓ |
| Event Bus | `tests/test_event_bus.py` | Basic | ✓ |
| Trainer | `tests/test_trainer.py` | Basic | ✓ |
| Voice | None | 0% | ✗ |
| Plugins | None | 0% | ✗ |
| Auth | None | 0% | ✗ |
| Startup | None | 0% | ✗ |
| Wake Word | None | 0% | ✗ |
| Audio | None | 0% | ✗ |
| State Machine | None | 0% | ✗ |
| Voice Supervisor | None | 0% | ✗ |

## 11. Python 3.14 Compatibility Issues

### 11.1 audioop Stub Incomplete
- **File**: `compat.py:62-96`
- **Problem**: `audioop.lin2lin` returns input unchanged. `audioop.ratecv` returns (d, s) but speech_recognition expects real resampling.
- **Fix**: Use `audioop-lts` package.

### 11.2 aifc Stub Duplicate
- **Files**: `compat.py:19-49`, `voice/stt.py:19`
- **Fix**: Remove duplicate from voice/stt.py.

### 11.3 imghdr Stub for Telethon
- **File**: `compat.py:52-59`
- **Problem**: Stub returns None for all images. Telethon may fail on image processing.
- **Fix**: Use `python-imghdr` backport if available.

## 12. Audio Stack Issues

### 12.1 No ALSA/PulseAudio/PipeWire Detection
- **File**: `voice/tts.py:49-54`
- **Problem**: Hardcoded to `paplay`.
- **Fix**: Auto-detect: `aplay` (ALSA), `paplay` (PulseAudio), `pw-play` (PipeWire), `jack_play` (JACK).

### 12.2 No Harmless ALSA Warning Suppression
- **Problem**: ALSA emits `Cannot find card 'default'` warnings on systems without audio hardware.
- **Fix**: Suppress ALSA warnings via environment variable.

### 12.3 No Microphone Failure Recovery
- **File**: `voice/stt.py:108-123`
- **Problem**: If microphone fails mid-session, no recovery.
- **Fix**: Add microphone health check and automatic reconnection.

## Priority Fix Order

### PHASE 1 — Critical Runtime Crashes
1. Fix `asyncio` import missing in `plugin_manager.py` (CRITICAL)
2. Install PyAudio or add sounddevice fallback
3. Fix Telegram plugin Telethon version mismatch
4. Fix audioop stub with `audioop-lts`
5. Remove duplicate aifc stub injection
6. Add WAL checkpoint on shutdown

### PHASE 2 — Architecture
7. Implement VoiceSupervisor with state machine
8. Implement MicrophoneManager with auto-recovery
9. Implement WakeWordEngine (offline)
10. Implement SpeechRecognizer with fallback chain
11. Implement SpeechSynthesizer with multi-backend
12. Implement AudioDeviceManager (ALSA/PulseAudio/PipeWire/JACK)
13. Implement NoiseCalibrator with persistence
14. Implement VoiceSettings with runtime config
15. Add correlation IDs to logging
16. Add plugin health checks and dependency validation

### PHASE 3 — Performance
17. Lazy-load Telegram and heavy imports
18. Add DuckDB connection pooling
19. Make DB operations async
20. Apply TTS speed/volume/pitch settings

### PHASE 4 — UX
21. Implement proper greeting after face auth
22. Add voice settings to .env
23. Suppress harmless ALSA warnings
24. Add microphone hotplug support

### PHASE 5 — Testing & Documentation
25. Create voice subsystem tests
26. Create plugin system tests
27. Create startup tests
28. Create audio backend tests
29. Create state machine tests
30. Update architecture documentation