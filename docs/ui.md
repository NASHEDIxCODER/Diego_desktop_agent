# Diego Desktop UI

A native PySide6 conversational interface for Diego that sits on top of the
existing production assistant pipeline.

## Launch

```bash
# Full production mode (wake word + face auth)
python -m ui

# Development mode — skip wake detection
python -m ui --no-wake

# Development mode — skip face auth
python -m ui --no-auth

# Full dev mode (both flags)
python -m ui --no-wake --no-auth

# UI-only mode (no voice pipeline — for testing the interface)
python -m ui --ui-only
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Qt Event Loop (main thread)              │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  DiegoMainWindow │◄───│        EventBridge           │  │
│  │  - transcript    │    │  (QObject + thread-safe queue)│  │
│  │  - state display │    └──────────┬───────────────────┘  │
│  │  - mic indicator │               │ Qt signals           │
│  │  - waveform      │               │                      │
│  │  - text input    │               │                      │
│  └──────────────────┘               │                      │
└──────────────────────────────────────┼──────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
        ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐
        │ ConversationEngine│  │   EventBus   │  │  Brain pipeline  │
        │  (state machine) │  │ (existing)   │  │  (process_command)│
        └─────────────────┘  └──────────────┘  └──────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Wake/VAD    STT/Whisper   TTS
     (unchanged) (unchanged)  (unchanged)
```

### Key Design Decisions

1. **No second assistant pipeline.** The UI subscribes to the existing
   `ConversationEngine` state machine and `Brain.process_command()`.
   Typed input goes through the **same** `Brain.process_command()` path
   as voice commands.

2. **Event bridge, not logic duplication.** The `EventBridge` patches
   only add *event emission* alongside existing state transitions:
   - `conversation_engine._set_state()` → emits UI state events
   - `conversation_engine._stt_event_pump()` → emits partial/final transcripts
   - `EventBus` wildcard subscription → forwards task/planning events

3. **Thread safety.** The Qt event loop runs on the main thread. The
   asyncio pipeline (ConversationEngine, Brain, voice) runs in a
   background thread. All events are marshalled through a thread-safe
   queue + QTimer polling — never direct cross-thread signal emission.

4. **Low CPU when idle.** The event bridge timer runs at 30ms intervals
   but only processes events when the queue is non-empty. The waveform
   animation timer only runs when audio is active.

## Event Types

| Event | UI Signal | Display |
|---|---|---|
| `LISTENING` | `listening` | Mic indicator pulses, state = "Listening" |
| `PARTIAL_TRANSCRIPT` | `partial_transcript` | Dashed partial bubble (updates in place) |
| `FINAL_TRANSCRIPT` | `final_transcript` | Partial bubble → user message (no duplication) |
| `THINKING` | `thinking` | Typing indicator, state = "Thinking" |
| `PLANNING` | `planning` | State = "Planning" |
| `EXECUTING` | `executing` | State = "Executing" |
| `OBSERVING` | `observing` | State = "Observing" |
| `VERIFYING` | `verifying` | State = "Verifying" |
| `REPLANNING` | `replanning` | State = "Replanning" |
| `RESPONSE` | `response` | Diego message bubble |
| `RESPONSE_CHUNK` | `response_chunk` | Streaming text in one bubble |
| `ERROR` | `error` | Red error bubble (friendly message) |
| `IDLE` | `idle` | State = "Idle" |

## Text Input

Typed commands are submitted through the **same** production path as voice:

```python
# In ui/main_window.py
result = await agent_brain.process_command(text)
```

This means typed input benefits from all existing guards:
- Intent authorization
- Transcript quality gate
- Decision engine routing
- Planner / dispatcher / verifier
- Learning engine

## Development Mode

```bash
python -m ui --no-wake --no-auth
```

This bypasses wake detection and face authentication, entering LISTEN
directly — identical to `python Diego.py --no-wake --no-auth`.

## Testing

```bash
# Run UI tests only
python -m pytest tests/test_ui.py -v

# Run full test suite
python -m pytest
```

The UI test suite covers:
- UI startup
- Event bridge thread safety
- Partial transcript update (no duplication)
- Final transcript replacement
- User message rendering
- Diego response rendering
- State changes
- Error rendering
- Typed input uses ConversationEngine/Brain
- Long-running operations do not block UI
- No duplicate transcript messages

## Production Compatibility

The UI does **not** modify:
- Wake/VAD thresholds
- Audio architecture
- Face authentication
- Vision
- Knowledge indexing
- Embeddings
- System-info subsystem
- Diagnostics subsystem
- Autonomous task controller

Only event/UI integration is added.