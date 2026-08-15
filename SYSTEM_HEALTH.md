# Leo Desktop Assistant v3.0 — System Health Report

**Generated:** 2026-07-29  
**Audit Tool:** `debug/audit_system.py`  
**Test Suite:** 83/83 passing, NLP 98.3% accuracy  

---

## Final Subsystem Status

| Subsystem | Status | Latency | Details |
|-----------|--------|---------|---------|
| **Startup** | ✅ READY | 1ms | StartupHealth functional |
| **NLP** | ✅ READY | 1,015ms | v2.0.0, 35 intents, 1,225 examples, 98.3% accuracy |
| **Embeddings** | ✅ READY | 5ms | all-MiniLM-L6-v2, dim=384, CUDA |
| **Voice** | ✅ READY | 324ms | Audio: pipewire ✓, Mic: sounddevice ✓ |
| **TTS** | ✅ READY | 8ms | pyttsx3 engine (fallback chain: kokoro→xtts→piper→pyttsx3) |
| **FaceAuth** | ⚠️ DISABLED | N/A | cv2 not installed — `pip install opencv-python face_recognition` |
| **Vision** | ✅ READY | 323ms | Capture: ImageMagick ✓, OCR: Tesseract ✓ |
| **DuckDB** | ✅ READY | 68ms | Connected, schema v2 |
| **Browser** | ⚠️ DEGRADED | 2,027ms | Chrome 136+ blocks CDP attach to default profile. Falls back to persistent Leo profile at `~/.leo/browser_profile` |
| **Ollama** | ✅ READY | 193ms | 3 models auto-detected: kimi-k2.6, deepseek-coder-v2, deepseek-coder |
| **Plugins** | ⚠️ DEGRADED | 4ms | 2 enabled (Brightness, YouTube), 1 disabled (Telegram: no API creds in .env) |
| **Planner** | ✅ READY | N/A | Agent planner initialized |
| **Shutdown** | ✅ READY | N/A | Graceful shutdown handlers registered |

**Summary:** 10 READY · 3 DEGRADED · 0 FAILED

---

## Fixes Applied

### Voice → READY
- Added `sounddevice` microphone backend (no PyAudio required)
- `MicrophoneManager` now tries: PyAudio → sounddevice → none
- Auto-detects PipeWire, PulseAudio, ALSA via `AudioDeviceManager`
- Microphone recovery with exponential backoff
- Health checks every 5 seconds

### Vision → READY
- Screen capture via ImageMagick `import` command (most reliable on Linux)
- OCR via Tesseract (2,771 chars extracted in test)
- Fallback chain: ImageMagick → mss → pyautogui
- Vision cache with LRU eviction and hash-based change detection

### Browser → DEGRADED (documented)
- Chrome 136+ security blocks CDP attach to default profile
- Automatic fallback to persistent Leo profile at `~/.leo/browser_profile`
- Clear logging explains why fallback occurred
- To fix: start Chrome manually with `google-chrome --remote-debugging-port=9222`

### Remaining 3 Degraded (all environment, not code)
1. **FaceAuth**: `pip install opencv-python face_recognition`
2. **Browser**: Start Chrome with `--remote-debugging-port=9222` or use persistent profile
3. **Plugins**: Add `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` to `.env`

---

## Performance Benchmarks

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| NLP Accuracy | 98.3% | >95% | ✅ |
| NLP Avg Latency | 55.57ms | <200ms | ✅ |
| Voice Init | 324ms | <1s | ✅ |
| TTS Init | 8ms | <1s | ✅ |
| Vision Init | 323ms | <1s | ✅ |
| DuckDB Init | 68ms | <1s | ✅ |
| Ollama Check | 193ms | <2s | ✅ |
| Plugin Init | 4ms | <1s | ✅ |
| Startup (parallel) | ~14s | <30s | ✅ |

---

## Architecture

```
BOOT → INIT → READY → AUTH → GREETING → WAKE → LISTEN → STT → NLP → VISION → PLANNER → EXECUTOR → VERIFY → TTS → WAKE
```

All subsystems are instrumented with timing. Performance report logged after every interaction:
```
PERF [#N] wake=Xms | face=Yms | vision=Zms | stt=Wms | nlp=Vms | plugin=Ums | total=Tms