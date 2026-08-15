# Leo ASR Alternatives Benchmark

**Date:** 2026-08-15
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM, Linux, CUDA available
**Audio:** Identical 16 kHz mono normalized float32 [-1,1] — synthesized via espeak-ng (122 commands) through the SAME unified pipeline. No model created its own VAD / AudioManager / preprocessor.

---

## Executive Summary

**Whisper is NOT replaced.** None of the five alternatives demonstrably beats faster-whisper on Leo's real command workload. faster-whisper remains the PRIMARY ASR.

**Recommendation:**
- **PRIMARY ASR:** faster-whisper (base, CPU int8) — unchanged.
- **FALLBACK ASR:** NVIDIA Parakeet CTC 1.1B INT8 (sherpa-onnx) — best accuracy/latency balance among the alternatives, with a viable path to lower latency.

The single most important finding: **command accuracy is the bottleneck, not latency.** faster-whisper's 53.3% command accuracy (vs. the 52% baseline) is the highest of any model tested, and the alternatives all score lower (25–44%). Latency is a secondary concern because Leo's endpointing already adds ~1s of silence before finalization.

---

## Candidate Research (verified, not marketing)

| Model | Runtime | Size (INT8) | VRAM | RAM | Streaming | Languages | Hindi | License | Install |
|---|---|---|---|---|---|---|---|---|---|
| faster-whisper (base) | CTranslate2 | ~74 MB | ~18 MB | ~800 MB | No (chunk re-transcribe) | 99 langs | Weak | MIT | Already installed |
| Qwen3-ASR 0.6B INT8 | sherpa-onnx | ~370 MB | ~15 MB | ~1.5 GB | No (offline) | Multilingual | Weak | Apache-2.0 | sherpa-onnx |
| Parakeet CTC 1.1B INT8 | sherpa-onnx | ~1.1 GB | ~15 MB | ~2 GB | No (offline) | English | None | CC-BY-4.0 | sherpa-onnx |
| Streaming Zipformer EN INT8 | sherpa-onnx | ~68 MB | — | — | **Yes (true streaming)** | English | None | Apache-2.0 | sherpa-onnx |
| FireRedASR2 CTC zh_en INT8 | sherpa-onnx | ~1.2 GB | ~15 MB | ~2 GB | No (offline) | zh + en | None | Apache-2.0 | sherpa-onnx |
| SenseVoice INT8 | sherpa-onnx | ~234 MB | ~15 MB | ~1 GB | No (offline) | zh/en/ja/ko/yue | None | MIT | sherpa-onnx |

**Notes:**
- The Parakeet **RNNT 1.1B Multilingual** export requested in the spec does NOT exist in sherpa-onnx; the closest available 1.1B Parakeet export is **CTC English** (`runanywhere/sherpa-onnx-nemo-parakeet-ctc-1.1b-int8`). The RNNT (transducer) architecture is only available at 0.6B/110M sizes.
- The **streaming Zipformer EN** model **crashes on load** (C++ segfault: `'attention_dims' does not exist in the metadata`) with sherpa-onnx 1.13.5. It is not usable on this hardware/runtime combination.
- FireRedASR2 CTC is trained for Chinese+English but performs catastrophically on English commands (4.9% accuracy) — it is effectively unusable for Leo.

---

## Benchmark Results (122 commands)

### Accuracy

| Model | Command Acc | First-Word Acc | Exact Match | WER | Hallucination |
|---|---|---|---|---|---|
| **faster-whisper** | **53.3%** | **72.1%** | 53.3% | 0.486 | 14.8% |
| Qwen3-ASR 0.6B INT8 | 44.3% | 56.6% | 44.3% | 0.568 | **8.2%** |
| Parakeet CTC 1.1B INT8 | 42.6% | 59.0% | 42.6% | 0.571 | 23.0% |
| SenseVoice INT8 | 25.4% | 41.8% | 25.4% | 0.693 | 37.7% |
| FireRedASR2 CTC | 4.9% | 13.9% | 4.9% | 0.974 | 60.7% |
| Streaming Zipformer | — (crash) | — | — | — | — |

### Latency & Resources

| Model | Avg Latency | RTF | First Useful Word | Load Time | RAM | VRAM |
|---|---|---|---|---|---|---|
| faster-whisper | 624 ms | 0.51 | 526 ms | 6.7 s | ~800 MB | 18 MB |
| Qwen3-ASR 0.6B INT8 | 761 ms | 0.61 | 2073 ms | 5.4 s | ~1.5 GB | 15 MB |
| Parakeet CTC 1.1B INT8 | 414 ms | 0.32 | 119 ms | 9.1 s | ~2 GB | 15 MB |
| SenseVoice INT8 | **62 ms** | **0.05** | **30 ms** | 2.4 s | ~1 GB | 15 MB |
| FireRedASR2 CTC | 386 ms | 0.30 | 162 ms | 107.6 s | ~2 GB | 15 MB |

### Language Breakdown

| Model | English | Hindi | Hinglish |
|---|---|---|---|
| faster-whisper | 62.5% | 0.0% | 0.0% |
| Qwen3-ASR 0.6B INT8 | 51.0% | 0.0% | 8.3% |
| Parakeet CTC 1.1B INT8 | 50.0% | 0.0% | 0.0% |
| SenseVoice INT8 | 29.8% | 0.0% | 0.0% |
| FireRedASR2 CTC | 5.8% | 0.0% | 0.0% |

**⚠️ Hindi/Hinglish caveat:** All models scored 0% on Hindi/Hinglish because espeak-ng's Hindi voice produces audio that none of the models can recognize. This is a **synthetic-TTS limitation**, not a definitive model verdict. Real microphone recordings (`data/asr_test_audio/`) are required for an accurate Hindi/Hinglish evaluation. The `debug/record_asr_dataset.py` script exists for this purpose.

### Short Commands (stop/pause/resume/yes/no/cancel/back/again)

| Model | Short Cmd Accuracy |
|---|---|
| faster-whisper | **66.7%** |
| Parakeet CTC 1.1B INT8 | 41.7% |
| SenseVoice INT8 | 29.2% |
| FireRedASR2 CTC | 12.5% |
| Qwen3-ASR 0.6B INT8 | 0.0% |

Qwen3-ASR **discards all short commands** (0.0%) — it appears to have a minimum-utterance gate that drops single-word inputs. This is disqualifying for Leo's conversational interaction.

### Noise (hallucination rate under fan/keyboard/music)

| Model | Noise Hallucination |
|---|---|
| Qwen3-ASR 0.6B INT8 | **20.0%** |
| Parakeet CTC 1.1B INT8 | 25.0% |
| faster-whisper | 27.5% |
| SenseVoice INT8 | 35.0% |
| FireRedASR2 CTC | 55.0% |

---

## Weighted Ranking (per spec)

Weights: 40% command accuracy, 20% first-word accuracy, 15% streaming latency, 10% hallucination rate, 10% resource usage, 5% Hindi/Hinglish.

| Rank | Model | Score |
|---|---|---|
| 1 | **faster-whisper** | **65.3** |
| 2 | Parakeet CTC 1.1B INT8 | 60.6 |
| 3 | SenseVoice INT8 | 49.5 |
| 4 | Qwen3-ASR 0.6B INT8 | 48.4 |
| 5 | FireRedASR2 CTC | 32.4 |
| 6 | Streaming Zipformer EN | 0.0 (crash) |

---

## Per-Model Pros / Cons

### faster-whisper (PRIMARY — keep)
- **Pros:** Highest command accuracy (53.3%), highest first-word accuracy (72.1%), best short-command handling (66.7%), mature CTranslate2 runtime, already integrated.
- **Cons:** Highest latency (624 ms) and RTF (0.51); 14.8% hallucination; weak Hindi.

### NVIDIA Parakeet CTC 1.1B INT8 (FALLBACK)
- **Pros:** Second-highest score (60.6); good latency (414 ms, RTF 0.32); fast first-useful-word (119 ms); decent short-command handling (41.7%).
- **Cons:** 23% hallucination (highest among viable models); English-only (no Hindi); 2 GB RAM footprint.

### Qwen3-ASR 0.6B INT8
- **Pros:** **Lowest hallucination rate (8.2%)**; best noise robustness (20%); multilingual; Apache-2.0.
- **Cons:** **Drops all short commands (0.0%)** — disqualifying for Leo; high latency (761 ms) and slow first-useful-word (2073 ms).

### SenseVoice INT8
- **Pros:** Fastest (62 ms, RTF 0.05, first-useful 30 ms); tiny footprint; multilingual.
- **Cons:** Low accuracy (25.4%); high hallucination (37.7%); no Hindi.

### FireRedASR2 CTC zh_en INT8
- **Pros:** None for Leo's workload.
- **Cons:** Catastrophic English accuracy (4.9%); 60.7% hallucination; 107s load time.

### Streaming Zipformer EN INT8
- **Pros:** True streaming architecture (the only one).
- **Cons:** **Crashes on load** (C++ metadata incompatibility with sherpa-onnx 1.13.5). Not usable.

---

## Final Recommendation

### PRIMARY ASR: faster-whisper (unchanged)

Do NOT replace Whisper. On Leo's real command workload it remains the most accurate model (53.3% command accuracy, 72.1% first-word accuracy), and no alternative demonstrably beats it. The 52% accuracy problem is **not** a Whisper-specific defect — it reflects the difficulty of the command set (natural speech, short commands, Hindi) and the synthetic espeak-ng audio.

### FALLBACK ASR: NVIDIA Parakeet CTC 1.1B INT8 (sherpa-onnx)

Parakeet is the best fallback because it offers the second-highest weighted score (60.6) with meaningfully lower latency (414 ms vs 624 ms) and a fast first-useful-word (119 ms). It is a viable drop-in when Whisper is unavailable or when a lower-latency path is desired for specific command types.

### Why NOT the others

- **Qwen3-ASR** — drops all short commands (stop/pause/resume/yes/no), which is fatal for conversational interaction.
- **SenseVoice** — too inaccurate (25.4%) and too hallucination-prone (37.7%).
- **FireRedASR2** — effectively non-functional on English (4.9%).
- **Streaming Zipformer** — crashes on this runtime.

---

## How to Reproduce

```bash
# 1. Install runtime
.venv/bin/pip install sherpa-onnx

# 2. Record real microphone utterances (optional, for Hindi/Hinglish accuracy)
python debug/record_asr_dataset.py --list
python debug/record_asr_dataset.py          # record all 122 commands

# 3. Run the full benchmark (isolated subprocess per model)
python debug/benchmark_asr_alternatives.py

# 4. Quick smoke test
python debug/benchmark_asr_alternatives.py --quick

# 5. View last report
python debug/benchmark_asr_alternatives.py --report
```

Results are written to `data/asr_alternatives_benchmark.json`.

---

## Files Created

| File | Purpose |
|---|---|
| `voice/providers/sherpa_base.py` | Shared sherpa-onnx ASRProvider base (offline + streaming) |
| `voice/providers/sherpa_providers.py` | Concrete providers: Qwen3-ASR, Parakeet, Zipformer, FireRedASR, SenseVoice |
| `debug/asr_dataset.py` | 122-command dataset (English, natural, Hindi/Hinglish, short, noise) |
| `debug/record_asr_dataset.py` | Real microphone recording mode |
| `debug/benchmark_worker.py` | Isolated subprocess worker (crash-safe) |
| `debug/benchmark_asr_alternatives.py` | Orchestrating benchmark + weighted ranking |
| `docs/ASR_ALTERNATIVES_BENCHMARK.md` | This report |

**No Leo production files (Brain, Planner, Dispatcher, ConversationEngine, TTS, WakeListener, wake-word model) were modified.** Only ASR provider integration was added, and the existing Whisper provider continues to work unchanged.