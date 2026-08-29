# Command Recognition & Reply Latency Optimization

## Analysis Summary

The full pipeline: WAKE → LISTEN (STT) → THINK (LLM) → SPEAK (TTS)

### Identified Bottlenecks

1. **STT final transcription: beam_size=5, best_of=5** (command_listener.py:549-560)
   - Whisper beam search with beam_size=5 is ~5x slower than greedy decoding
   - For short voice commands, greedy (beam_size=1) is sufficient and much faster
   - **Impact: 2-5x STT latency reduction**

2. **endpoint_silence_ms = 700ms** (command_listener.py:85)
   - After user stops speaking, system waits 700ms of silence before finalizing
   - 400ms is sufficient to avoid cutting off mid-word
   - **Impact: ~300ms saved per command**

3. **min_context_ms = 800ms** (command_listener.py:84)
   - Minimum 800ms of audio before Whisper processes anything
   - 500ms is enough for Whisper to produce useful output
   - **Impact: ~300ms saved per command**

4. **Perception pipeline runs on EVERY command** (brain.py:382)
   - Full screen capture + OCR + accessibility tree on every command
   - For simple commands (open app, play music), screen context is unnecessary
   - **Impact: 500ms-2s saved per simple command**

5. **3-second settle wait in verification** (brain.py:724)
   - Polls pgrep for up to 3 seconds waiting for process to appear
   - 1.5s is sufficient for most desktop apps
   - **Impact: up to 1.5s saved per app-open command**

6. **VAD runs in executor per frame** (command_listener.py:1001-1002)
   - `run_in_executor` dispatch overhead for every 32ms frame
   - Can batch VAD calls or use direct call when Silero is available
   - **Impact: reduced per-frame overhead**

## Implementation Plan

- [x] Analyze the full pipeline (STT → LLM → TTS)
- [ ] Fix 1: Reduce Whisper beam_size/best_of for final transcription
- [ ] Fix 2: Reduce endpoint_silence_ms (700→400)
- [ ] Fix 3: Reduce min_context_ms (800→500)
- [ ] Fix 4: Skip perception for simple/direct commands
- [ ] Fix 5: Reduce settle wait in verification (3s→1.5s)
- [ ] Fix 6: Optimize VAD per-frame executor overhead
- [ ] Verify changes are syntactically correct
