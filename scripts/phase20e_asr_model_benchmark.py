#!/usr/bin/env python3
"""Phase 20E — multilingual ASR model benchmark (BENCHMARK ONLY, inference-only).

Compares faster-whisper model sizes against the production baseline
(``base``, CPU/int8, en-primary + gated hi fallback) over a CONTROLLED
synthetic-audio dataset. Inference-only: NEVER executes Diego actions,
NEVER touches TaskRunner/tools/autonomy, NEVER modifies production
configuration (no voice_settings, no command_listener, no VAD, no
normalizer, no thresholds, no Whisper cache writes).

DATASET LIMITATION (stated honestly): Phase 20D stored transcripts +
metrics ONLY (``debug/phase20d_mic_results.json``) — NO raw microphone
audio was stored, by design. This harness therefore synthesizes a
controlled dataset with the OFFLINE local ``espeak-ng`` binary
(``en`` voice for English, ``hi`` voice for Hindi/Hinglish/mixed) at
16 kHz mono float32 [-1,1]. Synthetic formant speech is NOT a substitute
for real microphone speech: absolute accuracy numbers measure
model-vs-model behavior on clean synthetic audio, NOT real-world WER.
The comparison signal (which candidate preserves entities / hallucinates
less / costs how much latency+RAM) is the evidence used for TASK 13.

Candidates (faster-whisper, same decode config as production: beam=1,
temperature=0.0, no_speech_threshold=0.9, vad_filter=False):
  - baseline : Systran/faster-whisper-base          (production model)
  - small    : Systran/faster-whisper-small         (~244MB params, ~1GB RAM)
  - turbo    : mobiuslabsgmbh/faster-whisper-large-v3-turbo (~810M params,
               ~2GB RAM) — multilingual EN/HI code-switch oriented.

Rejected (documented, NOT downloaded):
  - faster-whisper-medium (~769M params, slower than turbo, weaker HI than
    turbo per public multilingual reports) — dominated by turbo.
  - faster-distil-whisper-large-v3 (EN-only distil, drops Hindi support that
    20D needs) — wrong language coverage.
  - sherpa-onnx Qwen3-ASR/Parakeet/Zipformer/FireRed/SenseVoice (HF cache
    holds refs only, weights NOT downloaded; SenseVoice/FireRed lack Hindi;
    Parakeet/Zipformer are EN-only; Qwen3-ASR 0.6B download ~1.2GB for
    unproven HI gain vs turbo) — not benchmarked, noted as future work.
  - nemotron-3.5-asr-streaming (no usable local runtime: transformers has
    no native arch, ONNX int4 placeholder returns "", NeMo/GGUF absent).

Language modes per candidate: "en", "hi" (forced, same as production
passes). "auto" run ONCE per utterance for the baseline only, to document
why auto is NOT the production solution (latency + mis-detection); NOT
used for scoring candidates.

Scoring reuses production functions IMPORT-ONLY (no behavior change):
  voice.command_listener._postprocess/_validate_transcript/is_garbage/
  is_low_quality_transcript/_is_repeated_hallucination,
  nlp.command_normalizer.command_normalizer.normalize,
  nlp.intent_authorizer.authorize_intent (authorize NEVER executes),
  agent.task_continuation.classify_confirmation.

Output: benchmarks/phase20e_asr_results.json (benchmark data only).

Usage:
  .venv/bin/python scripts/phase20e_asr_model_benchmark.py --help
  .venv/bin/python scripts/phase20e_asr_model_benchmark.py --list-only
  .venv/bin/python scripts/phase20e_asr_model_benchmark.py --models baseline,small
  .venv/bin/python scripts/phase20e_asr_model_benchmark.py --models baseline,small,turbo --reps 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "phase20e_asr_results.json"

SAMPLE_RATE = 16000

# ── Controlled synthetic dataset ──────────────────────────────
# (utterance_id, language_group, espeak_voice, speak_text, expected_keywords)
# speak_text uses Devanagari where a Hindi voice is needed; Hinglish/mixed
# rows use romanized Hindi (Latin script) with the hi voice, matching how a
# Hindi speaker's Hinglish is phonotactically shaped.
DATASET: list[tuple[str, str, str, str, list[str]]] = [
    ("en-01", "en", "en", "Open Firefox", ["firefox"]),
    ("en-02", "en", "en", "Open the terminal", ["terminal"]),
    ("en-03", "en", "en", "Play Believer on YouTube", ["believer", "youtube"]),
    ("en-04", "en", "en", "What's my RAM?", ["ram"]),
    ("hx-01", "hinglish", "hi", "Firefox kholo", ["firefox", "khol"]),
    ("hx-02", "hinglish", "hi", "YouTube pe Believer chalao",
     ["youtube", "believer", "chalao"]),
    ("hx-03", "hinglish", "hi", "Volume thoda kam karo", ["volume", "kam"]),
    ("hx-04", "hinglish", "hi", "Mera CPU kitna use ho raha hai",
     ["cpu", "kitna"]),
    ("hi-01", "hindi", "hi", "फायरफॉक्स खोलो", ["फायरफॉक्स", "खोल"]),
    ("hi-02", "hindi", "hi", "यूट्यूब पर गाना चलाओ",
     ["यूट्यूब", "गाना", "चलाओ"]),
    ("hi-03", "hindi", "hi", "आवाज़ थोड़ी कम करो", ["कम", "करो"]),
    ("mx-01", "mixed", "hi", "YouTube pe Arijit Singh ka song chalao",
     ["youtube", "arijit", "singh", "chalao"]),
    ("mx-02", "mixed", "hi", "Firefox mein GitHub kholo",
     ["firefox", "github", "khol"]),
    ("amb-01", "ambiguous", "hi", "play thodi der on youtube",
     ["play", "youtube"]),
]

# Entities whose preservation we track explicitly (TASK 6).
ENTITY_KEYS = ["firefox", "youtube", "believer", "arijit", "cpu", "ram",
               "volume", "terminal", "github"]

CANDIDATES: dict[str, dict[str, str]] = {
    "baseline": {"repo": "Systran/faster-whisper-base",
                 "size": "base (74M params, ~145MB)",
                 "langs": "multilingual (99+)",
                 "license": "MIT (Systran CTranslate2 conversion of OpenAI Whisper)"},
    "small": {"repo": "Systran/faster-whisper-small",
              "size": "small (244M params, ~460MB)",
              "langs": "multilingual (99+)",
              "license": "MIT (Systran CTranslate2 conversion of OpenAI Whisper)"},
    "turbo": {"repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
              "size": "large-v3-turbo (810M params, ~1.6GB)",
              "langs": "multilingual (99+)",
              "license": "conversion of OpenAI whisper-large-v3-turbo (MIT+Apache-2.0 parts)"},
}


def synth_clip(text: str, voice: str) -> np.ndarray:
    """Synthesize 16kHz mono float32 [-1,1] with offline espeak-ng."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        r = subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "150", "-w", wav_path, text],
            capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {r.stderr.decode()[:200]}")
        import wave
        with wave.open(wav_path, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
            sampwidth = w.getsampwidth()
            nchan = w.getnchannels()
            fr = w.getframerate()
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise RuntimeError(f"unexpected sampwidth {sampwidth}")
        if nchan > 1:
            audio = audio.reshape(-1, nchan).mean(axis=1)
        if fr != SAMPLE_RATE:
            # integer-ratio resample via linear interp (22050->16000 etc.)
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / fr))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = (audio / peak * 0.85).astype(np.float32)
        # pad/trim to 3.0s deterministic window
        target = SAMPLE_RATE * 3
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        return audio.astype(np.float32)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _gpu_used_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return 0.0


def _proc_ram_mb() -> float:
    try:
        import psutil
        import os
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        return 0.0


def load_model(repo: str, device: str, compute: str):
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(repo, device=device, compute_type=compute)
    # warmup (zeros, not timed for latency stats)
    warm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    list(model.transcribe(warm, beam_size=1, without_timestamps=True)[0])
    return model, (time.time() - t0) * 1000.0


def decode_once(model, audio: np.ndarray, language) -> tuple:
    t0 = time.time()
    segments, info = model.transcribe(
        audio, beam_size=1, language=language, temperature=0.0, best_of=1,
        condition_on_previous_text=False, compression_ratio_threshold=None,
        no_speech_threshold=0.9, vad_filter=False, without_timestamps=True)
    segs = list(segments)
    lat = (time.time() - t0) * 1000.0
    if not segs:
        return "", 0.0, lat
    text = " ".join(s.text.strip() for s in segs).strip()
    lps = [float(getattr(s, "avg_logprob", 0.0) or 0.0) for s in segs]
    avg = float(np.mean(lps)) if lps else 0.0
    lang = getattr(info, "language", None) or (language or "auto")
    return text, avg, lat, lang


def score_row(transcript: str, logprob: float, keywords: list[str]) -> dict:
    import voice.command_listener as CL
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent
    from agent.task_continuation import classify_confirmation
    post = CL._postprocess(transcript)
    valid, fail = CL._validate_transcript(post, confidence=logprob,
                                          speech_dur_ms=3000.0)
    garbage = CL.is_garbage(post) if post else True
    lowq = CL.is_low_quality_transcript(post) if post else True
    rep = CL._is_repeated_hallucination(post) if post else False
    hallucinated = bool(garbage or lowq or rep)
    normed, auth_actionable, intent_cat = "", False, ""
    try:
        normed = command_normalizer.normalize(post)
        auth = authorize_intent(normed, stt_confidence=float(logprob),
                                audio_duration_ms=3000.0)
        auth_actionable = bool(auth.actionable)
        intent_cat = str(getattr(auth, "category", ""))
    except Exception:
        pass
    tl = transcript.lower()
    kw_hits = {k: (k.lower() in tl) for k in keywords}
    ent_hits = {e: (e in tl) for e in ENTITY_KEYS if e in [k.lower() for k in keywords]}
    usable = bool(valid and not hallucinated)
    confirm = classify_confirmation(post)
    return {
        "postprocessed": post, "valid": bool(valid), "fail_reason": fail,
        "garbage": bool(garbage), "low_quality": bool(lowq),
        "repeated_hallucination": bool(rep),
        "hallucinated": hallucinated, "usable": usable,
        "normalized_command": normed, "intent_category": intent_cat,
        "actionable": auth_actionable,
        "keyword_hits": kw_hits,
        "keyword_recall": (sum(kw_hits.values()) / len(kw_hits)) if kw_hits else 0.0,
        "entity_hits": ent_hits,
        "confirmation_word": confirm,
    }


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(q / 100.0 * len(s))))
    return float(s[i])


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20E ASR benchmark (inference-only)")
    ap.add_argument("--models", default="baseline,small,turbo")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", default="en,hi")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--auto-probe", action="store_true",
                    help="also run language=None auto-detect probe on baseline only")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    if args.list_only:
        print(json.dumps({"candidates": CANDIDATES,
                          "dataset": [u[0] for u in DATASET]}, indent=2))
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in CANDIDATES:
            print(f"ERROR: unknown model {m!r} (choose from {sorted(CANDIDATES)})")
            return 2
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    import voice.command_listener as CL  # import-only; never instantiated/mutated
    assert CL.FALLBACK_MAX_PASSES == 2  # sanity: production untouched

    print("=== Phase 20E: synthesizing controlled dataset (espeak-ng, offline) ===")
    clips: dict[str, np.ndarray] = {}
    for uid, group, voice, speak, _kw in DATASET:
        clips[uid] = synth_clip(speak, voice)
        print(f"  {uid} [{group}/{voice}]: {speak!r} "
              f"({len(clips[uid]) / SAMPLE_RATE:.2f}s)")

    results: dict = {"phase": "20E",
                     "captured_at": datetime.now(timezone.utc).isoformat(),
                     "dataset_note": ("controlled synthetic espeak-ng audio "
                                      "(NOT real mic — 20D stored no raw audio); "
                                      "model-vs-model comparison only"),
                     "decode_config": {"beam_size": 1, "temperature": 0.0,
                                       "no_speech_threshold": 0.9,
                                       "vad_filter": False,
                                       "device": args.device,
                                       "compute": args.compute},
                     "models": {}}
    ram_before = _proc_ram_mb()
    for mname in models:
        repo = CANDIDATES[mname]["repo"]
        print(f"\n=== loading {mname} ({repo}) device={args.device} "
              f"compute={args.compute} ===")
        ram0, gpu0 = _proc_ram_mb(), _gpu_used_mb()
        try:
            model, load_ms = load_model(repo, args.device, args.compute)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:300]}")
            results["models"][mname] = {"load_failed": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        ram1, gpu1 = _proc_ram_mb(), _gpu_used_mb()
        print(f"  loaded in {load_ms:.0f}ms ram_delta={ram1 - ram0:.0f}MB")
        mrec: dict = {"repo": repo, "meta": CANDIDATES[mname],
                      "load_ms": round(load_ms, 1),
                      "ram_delta_mb": round(ram1 - ram0, 1),
                      "gpu_delta_mb": round(gpu1 - gpu0, 1),
                      "utterances": []}
        for uid, group, _voice, speak, kw in DATASET:
            for mode in modes:
                lats, trec = [], None
                for _ in range(max(1, args.reps)):
                    # warm rep discarded for median stability on CPU
                    text, lp, lat, det = decode_once(model, clips[uid], mode)
                    lats.append(lat)
                    trec = (text, lp, det)
                text, lp, det = trec
                # one extra first-inference sample already in lats[0]
                s = score_row(text, lp, kw)
                first_lat = lats[0]
                rep_lats = lats[1:] if len(lats) > 1 else lats
                mrec["utterances"].append({
                    "id": uid, "group": group, "prompt": speak,
                    "mode": mode, "detected_language": det,
                    "transcript": text, "avg_logprob": round(float(lp), 3),
                    "first_latency_ms": round(first_lat, 1),
                    "latencies_ms": [round(x, 1) for x in lats],
                    "median_latency_ms": round(float(statistics.median(rep_lats)), 1),
                    **s})
                print(f"  [{uid}/{mode}] {text!r} lp={lp:.2f} "
                      f"med={statistics.median(rep_lats):.0f}ms "
                      f"usable={s['usable']} kw={s['keyword_recall']:.2f}")
            if args.auto_probe and mname == "baseline":
                for uid, group, _voice, speak, kw in DATASET:
                    if uid != "en-01":
                        pass
                # single auto probe on one EN + one HX row (documents auto cost)
                for uid in ("en-03", "hx-02"):
                    _g = next(u[1] for u in DATASET if u[0] == uid)
                    _s = next(u[3] for u in DATASET if u[0] == uid)
                    text, lp, lat, det = decode_once(model, clips[uid], None)
                    mrec["utterances"].append({
                        "id": uid, "group": _g, "prompt": _s, "mode": "auto",
                        "detected_language": det, "transcript": text,
                        "avg_logprob": round(float(lp), 3),
                        "first_latency_ms": round(lat, 1),
                        "latencies_ms": [round(lat, 1)],
                        "median_latency_ms": round(lat, 1),
                        **score_row(text, lp, next(u[4] for u in DATASET if u[0] == uid))})
                    print(f"  [{uid}/auto] {text!r} detected={det} {lat:.0f}ms")
        # aggregate per-model stats (forced modes only)
        for mode in modes:
            rows = [u for u in mrec["utterances"] if u["mode"] == mode]
            if rows:
                meds = [u["median_latency_ms"] for u in rows]
                mrec[f"stats_{mode}"] = {
                    "n": len(rows),
                    "usable_rate": round(sum(1 for u in rows if u["usable"]) / len(rows), 4),
                    "keyword_recall_mean": round(float(np.mean([u["keyword_recall"] for u in rows])), 4),
                    "median_ms": round(float(statistics.median(meds)), 1),
                    "p95_ms": round(pct(meds, 95), 1),
                    "max_ms": round(max(meds), 1)}
        results["models"][mname] = mrec
        del model  # release before next candidate
    results["process_ram_mb"] = {"before": round(ram_before, 1),
                                 "after": round(_proc_ram_mb(), 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved benchmark data ONLY to {out} (no production changes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
