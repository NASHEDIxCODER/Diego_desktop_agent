# Leo → Conversational Desktop Agent — Transformation

## Goal
Transform Leo from a command executor into a real conversational desktop
companion (Siri / ChatGPT Voice / Gemini Live): full-duplex, streaming,
interruptible, with memory, personality, vision, and automatic action.

## Status: ✅ Core transformation complete & tested

### What was built (new modules)

| Module | Purpose | Tested |
|--------|---------|--------|
| `core/conversation_engine.py` | Full-duplex orchestrator: wake→conversation lifecycle, interruption, timeout watchdog | ✅ logic test |
| `voice/streaming_stt.py` | Streaming Silero VAD + faster-whisper, partial transcription, smart endpointing (600ms pause, filler words), interruption detection | ✅ imports/logic |
| `voice/streaming_tts.py` | Interruptible sentence-streamed TTS; Kokoro→XTTS→Piper→pyttsx3; persistent playback worker; instant abort | ✅ synthesis+playback |
| `agent/streaming_llm.py` | Token-streaming Ollama LLM → sentence segmentation → ACTION extraction; memory + personality integration | ✅ live stream |
| `agent/conversation_memory.py` | Rolling window + long-term facts + auto-summarization | ✅ recall test |
| `agent/personality.py` | Varied greetings/acks/farewells; never "How may I assist you?" | ✅ unit |
| `agent/action_dispatcher.py` | LLM ACTION → desktop ops (open app, browser, click, scroll, media) + screen context (OCR + active window) | ✅ logic |
| `auth/robust_auth.py` | Multi-frame voting, confidence averaging, head-pose estimation, anti-spoofing (texture/moiré/motion) | ✅ imports |
| `auth/face_popup.py` | Floating always-on-top Tk popup (420×320, dark, rounded, live 30 FPS preview, red/yellow/green states) | ✅ render test |
| `auth/live_auth.py` | Live auth loop: waits forever for a face, quality→guidance, 15-frame majority vote + confidence averaging | ✅ logic |
| `leo.py` | Conversational entry: BOOT → models → audio → WAIT_WAKE (NO startup auth) | ✅ status cmd |

### Modified
- `main.py` — launches conversational mode by default (`--legacy` for old loop, `--no-auth` for dev)
- `voice/settings.py` — added kokoro_voice, piper_model, conversational endpointing tunables
- `requirements.txt` — added faster-whisper, silero-vad, kokoro, misaki[en], piper-tts, sounddevice
- `README.md` — documented conversational architecture + usage

### Requirements coverage

- **Full duplex / interruption** — `streaming_stt.detect_interruption()` runs concurrently with TTS; `streaming_tts.interrupt()` aborts sounddevice stream instantly. ✅
- **Natural pauses (<600ms)** — `MIN_PAUSE_MS=600`, `ENDPOINT_SILENCE_MS=900` in streaming_stt. ✅
- **Filler words** — `is_filler()` keeps turn open for "umm/wait/hold on/actually/no". ✅
- **Streaming pipeline** — every stage streams; TTS starts on first sentence. ✅
- **Memory** — rolling context + facts; "remember my project is Leo" → "what was my project called?" answered in ~117ms. ✅
- **Personality** — varied greetings, no robotic phrases. ✅
- **Voice** — Kokoro primary (verified synthesizing), XTTS/Piper fallback, pyttsx3 emergency. ✅
- **Intelligence** — LLM emits ACTION lines → action_dispatcher executes (open VS Code, search, Spotify, etc.) without confirmation. ✅
- **Vision** — active-window title + OCR injected when the user references the screen. ✅
- **Face auth** — mandatory; robust multi-frame voting + pose + anti-spoofing; falls back to standard recognizer. ✅
- **Wake system** — "leo/hey leo/hello leo" via partials (fast); stays in conversation; sleeps on goodbye/timeout. ✅
- **Async/cancellation** — all async, cancellable, graceful shutdown. ✅

### Verified end-to-end (logic test)
```
wake partial "hey leo"   → mode: wake → conversation, greeting spoken
"open vs code"           → speaks ack + executes desktop_open(code)
"umm"                    → no response (turn stays open)
"goodbye"                → mode: conversation → wake
```

## ✅ Rework: face auth moved OFF startup (user feedback)

The original flow authenticated at startup (`BOOT → FACE AUTH → FAIL → EXIT`).
That was wrong. The corrected state machine:

```
BOOT → LOAD MODELS → INIT AUDIO → WAIT_WAKE
        ("Listening for wake word..." — nothing else, no camera/auth/greeting)
  ↓ wake word ("leo" / "hey leo" / "hello leo")
OPEN FACE AUTH POPUP → WAIT_FOR_FACE → FACE VERIFIED
  ↓
GREETING ("Welcome back, Sonu.") → conversation → GOODBYE → WAIT_WAKE
```

- **Boot never fails.** Leo always reaches WAIT_WAKE and stays alive forever.
- **Auth only after wake.** `conversation_engine._on_wake()` → `_do_auth_and_enter()` → `leo.authenticate_on_wake()` → `auth.live_auth.authenticate_live()`.
- **Popup** (`auth/face_popup.py`): floating, always-on-top, 420×320, dark, rounded, live 30 FPS preview, no OpenCV window, no terminal spam. States: searching / no-face (red) / guidance (red) / detected (yellow) / verified (green).
- **Waits forever for a face** — no "no face" timeout. Quality failures show guidance instead of failing ("Move closer", "Too dark", "Too blurry", "Look at camera", "Center your face").
- **Multi-frame verification** — 15 consecutive good frames, majority vote (8), confidence averaging.
- **Re-auth suppression** — 10-minute session (`AUTH_SESSION_S`); `invalidate_auth()` forces re-auth (logout/security/lock).
- **Auth failure NEVER terminates Leo** — denies the interaction, returns to WAIT_WAKE.

### Verified auth-on-wake flow (logic test)
```
boot                 → mode=wake (no auth, no greeting)
wake#1               → auth called ONCE → "Welcome back, Sonu." → conversation
goodbye              → wake
wake#2 (in session)  → NO auth call (suppressed) → conversation
session expired      → needs_auth again
auth denied          → Leo STAYS ALIVE in wake, "I couldn't verify your identity."
```

### Verified popup render (GUI test)
Tk window opened (always-on-top 420×320), 38 frames drawn at ~30 FPS with
red/yellow/green state overlays, closed cleanly. Headless → falls back to
non-popup auth.

### Environment notes
- Project runs in `.venv` (Python 3.14). faster-whisper, silero-vad, torch, sounddevice, kokoro all present.
- Kokoro runs on **CPU** (GPU reserved for the Ollama LLM) — avoids CUDA OOM.
- Kokoro English G2P uses espeak (misaki/phonemizer-fork) since spacy can't build on cp314.
- Piper installed as fallback; needs a voice model path via `PIPER_MODEL` to activate.

### How to run
```bash
source .venv/bin/activate
python leo.py            # conversational Leo
python leo.py --status   # check subsystems
python main.py           # same conversational mode (default)
python main.py --legacy  # old command-executor loop
```

### Known limitations / next steps
- Wake latency is Whisper-partial-based (~400ms) rather than a dedicated <150ms keyword spotter; integrating openWakeWord streaming would tighten this.
- XTTS fallback requires the `TTS` package + a GPU with free VRAM; Kokoro+CPU is the practical primary.
- LLM first-token latency depends on the local model size (qwen2.5:7b on CPU ≈ seconds); a smaller model (e.g. llama3.2:1b) improves responsiveness.
