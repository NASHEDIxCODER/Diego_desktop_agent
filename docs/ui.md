# Diego Desktop UI — Premium Voice-First HUD

The Diego Desktop UI is a PySide6-based voice-first assistant HUD that sits on top of the existing production pipeline. It is **NOT a chat application** — the primary interaction is voice.

## Design Language

- **Dark glass/HUD aesthetic**: Deep dark background with subtle glass panels
- **Neon cyan/blue primary accent**: `#22d3ee` for primary interactions
- **Restrained purple secondary**: `#a78bfa` for Diego responses
- **Thin borders, soft shadows, rounded panels**
- **Strong typographic hierarchy**: Large readable transcript/response text
- **Voice interaction is the visual centerpiece**

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER                                                          │
│  DIEGO                    [● Ready]        [─] [✕]              │
│  Voice Assistant                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────┐                          │
│                    │   STATE LABEL   │                          │
│                    └─────────────────┘                          │
│                                                                 │
│                    ╭─────────────────╮                          │
│                   ╱                   ╲                         │
│                  │   VOICE CORE        │      ┌──────────────┐  │
│                  │   VISUALIZER        │      │  ACTIVITY    │  │
│                  │   (Hero Element)    │      │  PANEL       │  │
│                   ╲                   ╱       │              │  │
│                    ╰─────────────────╯        │  Voice       │  │
│                                               │  detected    │  │
│                                               │  Thinking    │  │
│                                               │  Executing   │  │
│                                               │  Responding  │  │
│                                               └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  YOU SAID                                                       │
│  "open firefox and search..."                                   │
├─────────────────────────────────────────────────────────────────┤
│  DIEGO                                        [Speaking···]     │
│  "Opening Firefox for you."                                     │
│  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ (mini waveform)                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                │
│  │ STT │ │Agent│ │ TTS │ │Total│   METRICS                      │
│  │600ms│ │1.2s │ │2.0s │ │3.8s │                                │
│  └─────┘ └─────┘ └─────┘ └─────┘                                │
├─────────────────────────────────────────────────────────────────┤
│  STT ✓   Agent ✓   TTS ✓   Tools ✓        SYSTEM STATUS        │
└─────────────────────────────────────────────────────────────────┘
```

## Voice Core Visualizer

The central voice core is the hero element. It's a custom Qt widget with:

- **Large circular core** with radial gradient glow
- **Concentric rings** (static outer, breathing middle)
- **Radial waveform bars** (64 bars around the core)
- **State-dependent animations**
- **Real audio reactivity** (microphone RMS / TTS activity)

### Visualizer States

| State | Animation | Color |
|-------|-----------|-------|
| IDLE | Calm breathing (extremely subtle) | Gray |
| LISTENING | Live microphone waveform | Cyan |
| SPEECH_DETECTED | Stronger pulse + waveform | Green |
| THINKING | Slow processing rotation | Purple |
| EXECUTING | Progress/activity sweep | Blue |
| SPEAKING | Output waveform (TTS-driven) | Cyan |
| ERROR | Subtle error pulse | Red |

### Performance

- QTimer-based animation at ~60 FPS when active
- Timer slows to ~20 FPS when idle (subtle breathing)
- Smooth interpolation, no jitter
- No fake random waveform during active microphone use
- Low CPU usage when idle

## Audio Reactivity

The visualizer reacts to **real audio data**:

- **Input (microphone)**: Polls `AudioManager.last_frame_rms` (int16 scale) and normalizes to 0-1
- **Output (TTS)**: Uses `StreamingTTS.is_speaking` to drive speaking animation

No second audio capture pipeline is created. The UI polls the existing pipeline at ~30 FPS.

## Event Integration

The UI reuses the existing `EventBridge` (thread-safe):

| Event | UI Response |
|-------|-------------|
| LISTENING | Visualizer → LISTENING, Activity → "Voice detected" |
| PARTIAL_TRANSCRIPT | Update "YOU SAID" (italic, live) |
| FINAL_TRANSCRIPT | Replace partial with final (smooth transition) |
| THINKING | Visualizer → THINKING, Activity → "Thinking" |
| PLANNING | Visualizer → THINKING, Activity → "Planning" |
| EXECUTING | Visualizer → EXECUTING, Activity → "Executing" |
| OBSERVING | Visualizer → EXECUTING, Activity → "Observing" |
| VERIFYING | Visualizer → EXECUTING, Activity → "Verifying" |
| REPLANNING | Visualizer → THINKING, Activity → "Planning" |
| SPEAKING | Visualizer → SPEAKING, show speaking indicator |
| RESPONSE | Display Diego response prominently |
| ERROR | Show error in response panel |
| IDLE | Visualizer → IDLE, reset activity panel |
| AUDIO_LEVEL | Update visualizer input level |
| TTS_LEVEL | Update visualizer output level |

## Activity Panel

The right-side panel shows **only human-readable high-level activity**:

- Voice detected
- Transcribing
- Thinking
- Planning
- Executing
- Observing
- Verifying
- Responding

**NOT shown**: paths, JSON, scores, stack traces, database rows, embedding metadata, internal class names.

## Transcript Handling

- **Single "YOU SAID" region** (no chat bubbles)
- During speech: live partial text (italic, secondary color)
- When final: replaces partial with smooth transition
- **Never duplicates** — one region only

## Response Display

- **"DIEGO" header** with cyan accent
- **Large response text** (prominent)
- **Speaking indicator** (animated dots + mini waveform)
- Response appears immediately when available (before/during TTS)
- Remains visible after TTS completes

## Metrics

Compact latency cards:

- **STT**: Speech-to-text latency
- **Agent**: Brain/decision latency
- **TTS**: Text-to-speech latency
- **Total**: End-to-end turn latency

## System Status

Footer shows component health:

- STT ✓
- Agent ✓
- TTS ✓
- Tools ✓

## Running the UI

```bash
# Full production mode (wake + auth)
python -m ui

# Skip wake word (development)
python -m ui --no-wake

# Skip face auth (development)
python -m ui --no-auth

# Full dev mode
python -m ui --no-wake --no-auth

# UI only (no pipeline, for testing)
python -m ui --ui-only
```

## Architecture

```
ui/
├── __init__.py       # Package exports
├── __main__.py       # Entry point + audio level poller
├── event_bridge.py   # Thread-safe bridge (pipeline → Qt)
├── main_window.py    # Main HUD window
├── visualizer.py     # VoiceCoreVisualizer (hero element)
├── widgets.py        # HUD widgets (panels, metrics, status)
└── styles.py         # Dark glass/HUD theme
```

### Threading Model

- Qt event loop runs on the **main thread**
- Pipeline (asyncio) runs in a **background thread**
- Events flow via `EventBridge` (thread-safe queue + QTimer polling)
- The Qt event loop is **NEVER blocked**

### Preserved Components

The UI does **NOT modify**:

- STT (speech recognition)
- VAD (voice activity detection)
- TTS (text-to-speech)
- Brain (agent logic)
- DecisionEngine
- TaskController
- Knowledge
- Diagnostics
- Vision
- Authentication
- Wake system

Only UI and minimal event plumbing are modified.

## Testing

```bash
# Run UI tests
pytest tests/test_ui.py -q

# Run full test suite
pytest -q
```

Tests cover:
- Voice visualizer states
- Real audio level updates
- Listening/speaking animations
- Partial/final transcript handling
- Response rendering
- State transitions
- Resize behavior
- No chat composer
- UI non-blocking
- Event bridge integration