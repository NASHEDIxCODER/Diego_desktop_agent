# Diego UI — Voice-First Desktop Assistant HUD

## Design

The Diego UI is a **voice-first desktop assistant HUD**, NOT a chat application.
The primary interaction model is:

```
MICROPHONE → LIVE TRANSCRIPT → DIEGO RESPONSE PRINTED
→ DIEGO RESPONSE SPOKEN → LISTEN AGAIN
```

No mouse/keyboard interaction is required for normal operation.

## Layout

```
┌──────────────────────────────────────────────────┐
│ HEADER                                           │
│   ● DIEGO                    [Listening]  ─  ✕  │
├──────────────────────────────────────────────────┤
│ CENTER                                           │
│   Large animated waveform / audio visualizer     │
│   ● Diego is speaking  (visible during TTS)      │
├──────────────────────────────────────────────────┤
│ TRANSCRIPT AREA                                  │
│   YOU                                            │
│   "open firefox"                                 │
├──────────────────────────────────────────────────┤
│ RESPONSE AREA                                    │
│   DIEGO                                          │
│   "Opening Firefox for you."                     │
├──────────────────────────────────────────────────┤
│ FOOTER                                           │
│   Listening...  STT: 600ms Agent: 1.2s TTS: 2.1s│
└──────────────────────────────────────────────────┘
```

## Voice States

The header displays the current voice state:

| State | Meaning |
|---|---|
| IDLE | Waiting for wake word |
| LISTENING | Microphone active, capturing |
| SPEECH DETECTED | VAD detected speech onset |
| THINKING | Brain processing |
| PLANNING | Planner creating a plan |
| EXECUTING | Actions being dispatched |
| OBSERVING | Environment observation |
| VERIFYING | Action verification |
| REPLANNING | Re-planning after failure |
| SPEAKING | TTS playback active |
| ERROR | Friendly error displayed |

## Transcript Behaviour

- **LIVE TRANSCRIPT**: While the user speaks, the partial STT transcript
  updates a single live region in real time.
- **FINAL TRANSCRIPT**: When speech finalizes, the partial region is
  **replaced** (never duplicated) with the final recognized sentence.
- **DIEGO RESPONSE**: The response is printed prominently **before/during**
  TTS and remains visible after TTS completes.
- **SPEAKING INDICATOR**: Clearly shows Diego is speaking through TTS.

## Latency Metrics (footer, optional)

Small monospace diagnostics show measured latencies:
- STT latency
- Agent (Brain) latency
- TTS latency
- Total turn latency

Internal logs, planner traces, embeddings, scores, paths, JSON, and database
information are NEVER exposed in the UI.

## Event-Driven Architecture

The UI never blocks the Qt main thread. All pipeline work (STT, LLM, agent,
tools, perception, TTS) runs off-thread. Events arrive via the thread-safe
`EventBridge` (queue + Qt signal dispatch):

`LISTENING, PARTIAL_TRANSCRIPT, FINAL_TRANSCRIPT, THINKING, PLANNING,
EXECUTING, OBSERVING, VERIFYING, REPLANNING, SPEAKING, RESPONSE, ERROR, IDLE`

The bridge reuses the production ConversationEngine / Brain pipeline — there
is no second voice pipeline.

## Modules

```
ui/
├── __init__.py       # Package exports
├── __main__.py       # Entry point: python -m ui
├── event_bridge.py   # Thread-safe bridge from pipeline events to Qt signals
├── main_window.py    # Voice-first HUD main window
├── widgets.py        # VoiceStateIndicator, WaveformWidget, TranscriptLabel,
│                     # ResponseLabel, LatencyMetrics, MicIndicator
└── styles.py         # Dark futuristic theme QSS
```

## Performance

- Animations are subtle and low CPU (waveform timer only runs when active).
- When idle: calm state, no unnecessary animation.
- When listening: waveform active, mic indicator pulsing.
- When speaking: waveform/output indicator active.
- When thinking: clear processing indicator.

## Running

```bash
python -m ui                     # Full production mode (wake + auth)
python -m ui --no-wake           # Skip wake word (dev)
python -m ui --no-auth           # Skip face auth (dev)
python -m ui --no-wake --no-auth # Full dev mode
python -m ui --ui-only           # UI only, no voice pipeline (testing)
```
