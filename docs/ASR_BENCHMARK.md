# LEO ASR BENCHMARK — Nemotron vs Whisper

**Date:** 2026-08-15
**Scope:** Evaluate NVIDIA Nemotron-3.5-ASR-Streaming as a potential replacement for Leo's current faster-whisper command ASR. **No architecture was changed.** faster-whisper, wake-word detection, VAD, Brain, Planner, Dispatcher, TTS, and conversation flow remain untouched.

---

## 1. Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 |
| VRAM | 4096 MiB (4 GB) |
| System RAM | 16 GB |
| OS | Linux 7.1 |
| CUDA | 13.0 (driver 610.57.04, UMD 13.3) |
| PyTorch | 2.13.0+cu130 (CUDA available) |
| transformers (installed) | 4.57.6 |
| faster-whisper | 1.2.1 |
| onnxruntime | 1.28.0 |
| nemo_toolkit | **not installed** |
| llama-cpp-python | **not installed** |

---

## 2. Models Evaluated

| Model | Local runnable? | Blocker |
|---|---|---|
| **Nemotron-3.5-ASR-Streaming 0.6B** | **No** | float32 weights = **2.55 GB** (exceeds 4 GB VRAM with CUDA context). Native inference requires `transformers >= 5.x` (installed 4.57.6) or NeMo toolkit (not installed). |
| Parakeet RNNT 1.1B | No | Same runtime requirement; larger than 0.6B. |
| Parakeet RNNT 0.6B | No | `model.safetensors` present but requires NeMo toolkit / transformers 5.x. |
| Parakeet CTC 0.6B | No | Same as above. |
| Parakeet TDT 0.6B v3 | No | Same as above. |
| **faster-whisper (base, CPU int8)** | **Yes** | Current production backend. |

### Runtime options attempted for Nemotron

1. **transformers >= 5.x** — installed version is 4.57.6; `Nemotron3_5AsrForRNNT` architecture is only available in transformers 5.x. **Failed.**
2. **onnxruntime int4** (`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`) — `model.onnx` returns **404** (no such file in the repo). **Failed.**
3. **NeMo toolkit** — `nemo_toolkit` not installed. **Failed.**
4. **llama-cpp GGUF (q8_0, 0.74 GB)** — `llama_cpp` not installed; and GGUF does not support streaming RNNT decoding. **Failed.**

### VRAM analysis

- Nemotron float32: **2.55 GB** weights + CUDA context (~300–500 MB) + encoder activations → **exceeds 4 GB VRAM budget**.
- Nemotron q8_0 GGUF: **0.74 GB** — fits VRAM, but the GGUF path is a text-completion runtime, not a streaming RNNT ASR decoder.
- ONNX int4 export (~0.35 GB) would fit, but the community export is incomplete (no `model.onnx`).

**Conclusion: Nemotron-3.5-ASR-Streaming cannot be run locally on this machine with the current toolchain, and its float32 weights alone exceed the 4 GB VRAM budget.**

---

## 3. Benchmark Methodology

- Real Leo command set (25 utterances): English commands, natural speech, and Hindi/Hinglish.
- Audio synthesized to 16 kHz mono WAV via `espeak-ng` (English `en-us`, Hindi `hi` voices).
- Each clip transcribed through every available `ASRProvider`.
- Metrics recorded per transcript: model, language, transcript, expected, WER, command accuracy, first-word accuracy, time-to-first-token, finalization latency, real-time factor (RTF), CPU, GPU VRAM, GPU utilization, RAM, dropped chunks, hallucinations.
- WER and accuracy are **punctuation- and case-insensitive** (a trailing period or capitalization is not an error).

---

## 4. Results

### faster-whisper (base, CPU int8) — the current backend

| Metric | Value |
|---|---|
| N (utterances) | 25 |
| Avg WER | 0.527 |
| Median WER | 0.000 |
| **Command accuracy** | **52.0%** |
| **First-word accuracy** | **76.0%** |
| Avg finalization latency | 597.5 ms |
| P95 latency | 649.1 ms |
| Avg RTF | 0.424 |
| Avg GPU VRAM | 18 MB (CPU inference) |
| Avg GPU util | 0.0% |
| Avg RAM | 837 MB |
| Hallucinations | 5 |
| Dropped chunks | 0 |

### Nemotron-3.5-ASR-Streaming

**NOT READY — could not be benchmarked.** Runtime unavailable (see §2).

---

## 5. Hindi / Hinglish Performance (faster-whisper)

| Utterance | Transcript | WER | Correct? |
|---|---|---|---|
| youtube kholo | "YouTube co-lo." | 0.50 | No |
| firefox kholo | "Firefox co-lo." | 0.50 | No |
| gaana chalao | "on a channel." | 1.50 | No |
| volume kam karo | "volume cam carrow." | 0.67 | No |
| isko band karo | "is co-bind caro." | 1.00 | No |
| youtube par music chalao | "YouTube pop music channel." | 0.50 | No |

**Hindi/Hinglish command accuracy: 0%.** Whisper (base, English-forced) does not recognize Hindi/Hinglish commands reliably.

---

## 6. Failure Cases

1. **Nemotron cannot be loaded** — no usable local runtime on 4 GB VRAM (float32 > VRAM; transformers 4.x lacks the architecture; NeMo not installed; ONNX int4 export incomplete).
2. **Whisper hallucinates** on short/ambiguous audio (5 hallucinations across 25 clips).
3. **Whisper fails Hindi/Hinglish** — 0% command accuracy on the Hindi set.
4. **Whisper mis-transcribes "vscode"** as "the code" (WER 1.0) and "stop"/"continue" with trailing punctuation (now corrected by normalization, but raw output still differs).

---

## 7. Acceptance Criteria Assessment

| Criterion | Target | Result |
|---|---|---|
| Command accuracy | ≥98% | **52.0% (Whisper)** — below target. Nemotron untestable. |
| First-word accuracy | ≥95% | **76.0% (Whisper)** — below target. |
| Silent turns | 0 | Not measured (no live mic loop in this benchmark). |
| ASR crashes | 0 | 0 crashes observed. |
| Deadlocks | 0 | 0 observed. |
| TTS contamination | 0 | N/A — benchmark uses pre-recorded audio, not live TTS. |
| Stable streaming | — | Not measured (no live streaming loop). |
| Acceptable latency on RTX 3050 | — | Whisper CPU: ~597 ms avg, RTF 0.42. |

---

## 8. Recommendation

**KEEP WHISPER as the primary ASR. Do NOT replace it with Nemotron.**

Nemotron-3.5-ASR-Streaming **cannot be run locally** on this RTX 3050 (4 GB VRAM) with the current toolchain:

- Its float32 weights (2.55 GB) exceed the 4 GB VRAM budget.
- Native inference requires `transformers >= 5.x` (installed 4.57.6) or NeMo toolkit (not installed).
- The ONNX int4 community export is incomplete (no `model.onnx`).
- GGUF q8_0 fits VRAM but is not a streaming RNNT ASR runtime.

Because Nemotron could not be benchmarked, the acceptance criteria (≥98% command accuracy, ≥95% first-word) **cannot be verified for Nemotron** — and the current Whisper baseline (52% command accuracy) is itself well below those targets, so there is no evidence to justify a swap.

### Path forward (only if desired, and only after re-benchmarking)

1. Install `transformers >= 5.x` **or** `nemo_toolkit` and retry Nemotron — but confirm the float32 model can be quantized to fit 4 GB VRAM (e.g., int8/int4) before claiming it is viable.
2. If a quantized Nemotron/Parakeet export becomes available and fits VRAM, re-run this benchmark **before** any integration.
3. Until then, the fallback chain (`Nemotron → Whisper → "I didn't catch that."`) is implemented in `voice/asr_fallback.py` and will transparently use Whisper when Nemotron is unavailable — which is the current reality.

---

## 9. Files Added (no existing architecture modified)

| File | Purpose |
|---|---|
| `voice/asr_provider.py` | `ASRProvider` abstract interface + WER/accuracy/resource helpers. |
| `voice/providers/whisper_provider.py` | `WhisperProvider` (faster-whisper behind the interface). |
| `voice/providers/nemotron_provider.py` | `NemotronProvider` (honest runtime probing). |
| `voice/asr_fallback.py` | `ASRFallback` — Nemotron → Whisper → fallback phrase. |
| `debug/benchmark_asr.py` | The benchmark harness (synthesis + metrics + report). |
| `data/asr_benchmark.json` | Raw benchmark results. |

No changes were made to `voice/streaming_stt.py`, `voice/command_listener.py`, `voice/vad.py`, `voice/wake_word.py`, `voice/wake_listener.py`, `voice/audio_manager.py`, Brain, Planner, Dispatcher, TTS, or conversation architecture.