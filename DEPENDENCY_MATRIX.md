# Diego Runtime Dependency Matrix — FINAL STATE

Generated: 2026-09-01 | Verified: 2026-09-01 19:33 UTC

| Component | Package/Module | Required Version | Installed? | Production Import? | Runtime Asset | Asset Present? | Fix |
|-----------|---------------|-----------------|------------|-------------------|---------------|----------------|-----|
| Microphone | sounddevice | >=0.4.6 | ✅ 0.5.6 | voice/audio_manager.py | ALC256 device 5 | ✅ | none |
| Screen Capture | mss | >=9.0.0 | ✅ 10.2.0 | services/screen_capture.py | — | — | none |
| ASR #1 | faster-whisper | >=1.0.0 | ✅ 1.2.1 | voice/command_listener.py, voice/streaming_stt.py, voice/providers/whisper_provider.py | qwen2.5:3b (Ollama) | ✅ | **installed** |
| ASR runtime | sherpa-onnx | >=1.10.0 | ✅ 1.13.7 | voice/providers/sherpa_providers.py | data/asr_models/*.onnx | ✅ | **installed** |
| VAD | silero-vad | >=5.1 | ✅ 6.2.1 | voice/vad.py, voice/streaming_stt.py | bundled ONNX (auto-download) | ✅ | **installed** |
| VAD runtime | torch | >=2.1.0 | ✅ 2.13.0+cpu | voice/vad.py, voice/streaming_tts.py, voice/tts/kokoro_engine.py | — | — | **installed (CPU wheel)** |
| VAD runtime | torchaudio | >=2.1.0 | ✅ 2.11.0+cpu | silero_vad (transitive) | — | — | **installed (CPU wheel)** |
| Wake Word | openwakeword | >=0.4.0 | ✅ 0.6.0 | voice/wake_model_manager.py, voice/calibrate_wake.py | models/wake/verifier.pkl | ✅ | **installed** |
| TTS #1 | kokoro | >=0.7.16 | ✅ 0.9.4 | voice/tts/kokoro_engine.py, voice/streaming_tts.py | hexgrad/Kokoro-82M (auto-download) | ✅ | **installed** |
| TTS #2 | misaki[en] | >=0.7.4 | ✅ 0.9.4 | voice/tts/kokoro_engine.py | — | — | **installed** |
| TTS #3 | pyttsx3 | >=2.90 | ✅ 2.99 | voice/streaming_tts.py | — | — | **installed (fallback)** |
| TTS #4 | TTS (Coqui) | >=0.22.0 (<3.12) | ❌ | voice/streaming_tts.py | — | — | optional (kokoro is primary) |
| TTS #5 | piper-tts | >=1.6.0 | ❌ | voice/streaming_tts.py | voice/tts/piper voice | ❌ | optional (kokoro is primary) |
| OCR #1 | pytesseract | >=0.3.10 | ✅ 0.3.13 | vision/ocr_pipeline.py, agent/action_dispatcher.py | data/tessdata/eng.traineddata | ✅ | **installed** |
| OCR system | tesseract-ocr | — | ✅ 5.5.3 (/sbin/tesseract) | pytesseract (binary) | — | — | **present** |
| OCR #2 | easyocr | >=1.7.0 | ❌ | vision/ocr_pipeline.py | — | — | optional (tesseract active) |
| OCR #3 | paddleocr | >=2.8.0 | ❌ | vision/ocr_pipeline.py | — | — | optional (tesseract active) |
| Persistence | duckdb | >=1.0.0 | ✅ 1.5.5 | memory/duckdb_store.py | data/Diego.duckdb | ✅ | **installed** |
| LLM | ollama (server) | — | ✅ | agent/streaming_llm.py | qwen2.5:3b | ✅ | none |
| Image | Pillow | >=10.0.0 | ✅ 12.3.0 | vision/ocr_pipeline.py, agent/action_dispatcher.py | — | — | **installed** |
| Image | opencv-python-headless | >=4.8.0 | ✅ 5.0.0.93 | vision/ocr_pipeline.py | — | — | **installed** |
| Web | playwright | >=1.45.0 | ✅ 1.62.0 | agent/browser.py | chromium (optional) | ? | check `playwright install chromium` |
| NLP | rapidfuzz | >=3.0.0 | ✅ 3.14.6 | nlp/*.py | — | — | **installed** |
| Search | trafilatura | >=1.12.0 | ✅ 2.2.0 | services/search_service.py | — | — | **installed** |
| Search | beautifulsoup4 | >=4.12.0 | ✅ 4.15.0 | services/search_service.py | — | — | **installed** |
| Search | lxml | >=5.0.0 | ✅ 6.1.2 | services/search_service.py | — | — | **installed** |
| Test | pytest-asyncio | >=0.23.0 | ✅ 1.4.0 | pytest.ini (asyncio_mode=auto) | — | — | **installed** |
| API | httpx | >=0.27.0 | ✅ 0.28.1 | ai/llm_client.py | — | — | none |
| API | google-generativeai | >=0.8.0 | ❌ | ai/llm_client.py | — | — | optional (API key) |
| API | firebase-admin | >=6.0.0 | ❌ | auth/* | — | — | optional |
| ASR dev | nemo-toolkit[asr] | >=2.0.0 | ❌ | voice/providers/nemotron_provider.py | — | — | DEVONLY (faster-whisper is production STT) |

## Verification Results (2026-09-01)

```
sounddevice OK 0.5.6
mss OK
faster_whisper OK 1.2.1
silero_vad OK
duckdb OK 1.5.5
torch OK 2.13.0+cpu
torchaudio OK 2.11.0+cpu
kokoro OK
openwakeword OK
pytesseract OK
sherpa_onnx OK
trafilatura OK
```

**Production module checks:**
- `[VAD] load() -> True | backend: silero_vad` ✅
- `[STREAM-TTS] Ready (primary=_KokoroSynth, fallbacks=['kokoro', 'piper', 'pyttsx3', 'xtts'])` ✅
- `[CMD-LISTEN] faster-whisper loaded (device=cpu, compute=int8)` ✅
- `[OCR] Backend: Tesseract` ✅
- `[DUCKDB] HAS_DUCKDB = True` ✅
- `[WAKE] WakeModelManager OK` ✅

**Test suite:** `318 passed, 5 skipped` (pytest-asyncio 1.4.0 required for asyncio_mode=auto)

**Runtime launch (`Diego.py --no-wake --no-auth`):**
```
[HEALTH] component=OK name=microphone package=sounddevice/0.5.6
[HEALTH] component=OK name=vad package=silero_vad/6.2.1
[HEALTH] component=OK name=stt package=faster_whisper/1.2.1
[HEALTH] component=OK name=tts package=kokoro/0.9.4
[HEALTH] component=OK name=wake package=openwakeword/0.6.0
[HEALTH] component=OK name=screen_capture package=mss/10.2.0
[HEALTH] component=OK name=ocr package=pytesseract
[HEALTH] component=OK name=llm package=ollama (5 models)
[HEALTH] component=OK name=duckdb package=duckdb/1.5.5
[HEALTH] component=OK name=search package=trafilatura/bs4
[HEALTH] component=OK name=vision package=cv2/5.0.0
[HEALTH] component=OK name=sherpa_onnx package=sherpa_onnx/1.13.7
[HEALTH] component=MISSING name=nemo_toolkit (DEVONLY — faster-whisper is production STT)

REQUIRED voice components: 4 OK, 0 degraded, 0 missing
✓ Normal voice operation READY
```

**No repeated errors:** "Whisper unavailable", "No synthesis engine available", "Silero VAD unavailable" — all absent from runtime log.