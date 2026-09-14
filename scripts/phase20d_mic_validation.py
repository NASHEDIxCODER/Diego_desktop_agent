#!/usr/bin/env python3
"""Phase 20D — real-microphone multilingual ASR validation (validation-only).

Guided capture through the REAL production path, read-only:

    microphone (sounddevice, default input)
      -> UnifiedVAD diagnostics (speech_prob per 512-sample frame, no changes)
      -> _WhisperTranscriber.transcribe_with_language(pcm, sr, "en") [timed]
      -> is_primary_suspicious() gate (unchanged thresholds)
      -> optional single transcribe_with_language(pcm, sr, "hi") [timed]
      -> select_best_transcript() (max 2 passes, unchanged logic)
      -> _postprocess -> _validate_transcript -> normalize_multilingual/
         command_normalizer.normalize -> authorize_intent (comparison only)
         -> classify_confirmation (context-sensitivity check)

Records for EVERY utterance (Task 3): primary transcript, primary
avg_logprob, primary validation status, fallback triggered yes/no,
fallback transcript, fallback avg_logprob, selected transcript, selected
language, fallback reason, total STT latency (decode only, never human
delay), normalized command, intent category/route, authorization result,
actionable/uncertain/rejected.

Safety / non-invasiveness:
  - NEVER executes actions, NEVER touches autonomy/dispatch/routing.
  - NEVER changes Whisper model, compute type, VAD thresholds,
    normalization, thresholds, or face-auth.
  - NEVER stores raw audio: PCM lives in memory only and is released
    after each utterance. Only transcripts + scalar metrics are written
    to debug/phase20d_mic_results.json (no WAV files).
  - "play thodi der on youtube" and all rows are transcribe+authorize
    ONLY (authorize_intent never executes).

Usage:
  .venv/bin/python scripts/phase20d_mic_validation.py            # guided live capture
  .venv/bin/python scripts/phase20d_mic_validation.py --smoke    # no-mic self-test
  .venv/bin/python scripts/phase20d_mic_validation.py --list-devices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

UTTERANCES = [
    ("en-01", "en", "Open Firefox"),
    ("en-02", "en", "Open the terminal"),
    ("en-03", "en", "Play Believer on YouTube"),
    ("en-04", "en", "What's my RAM?"),
    ("hx-01", "hinglish", "Firefox kholo"),
    ("hx-02", "hinglish", "YouTube pe Believer chalao"),
    ("hx-03", "hinglish", "Volume thoda kam karo"),
    ("hx-04", "hinglish", "Mera CPU kitna use ho raha hai"),
    ("hi-01", "hindi", "फ़ायरफ़ॉक्स खोलो"),
    ("hi-02", "hindi", "यूट्यूब पर गाना चलाओ"),
    ("hi-03", "hindi", "आवाज़ थोड़ी कम करो"),
    ("mx-01", "mixed", "YouTube pe Arijit Singh ka song chalao"),
    ("mx-02", "mixed", "Firefox mein GitHub kholo"),
    ("amb-01", "ambiguous", "play thodi der on youtube"),
]

RECORD_SECONDS = 4.0
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512

RESULTS_PATH = PROJECT_ROOT / "debug" / "phase20d_mic_results.json"


def list_devices() -> int:
    import sounddevice as sd

    print("=== Input devices ===")
    for i, d in enumerate(sd.query_devices()):
        print(f"[{i}] {d['name']} (in={d['max_input_channels']}, "
              f"default_sr={d.get('default_samplerate')})")
    print("default:", sd.default.device)
    return 0


def record_utterance(seconds: float = RECORD_SECONDS) -> np.ndarray:
    """Record mono float32 @16k from the default input. Memory only."""
    import sounddevice as sd

    n = int(seconds * SAMPLE_RATE)
    audio = sd.rec(n, samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", blocking=True)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def vad_diagnostics(audio: np.ndarray) -> dict:
    """Run the REAL UnifiedVAD over the capture (diagnostics only)."""
    from voice.vad import UnifiedVAD

    vad = UnifiedVAD()
    probs = []
    n = len(audio)
    for i in range(0, n - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        probs.append(float(vad.speech_prob(audio[i:i + FRAME_SAMPLES])))
    probs = np.asarray(probs, dtype=np.float64) if probs else np.zeros(0)
    speech_frames = int(np.sum(probs >= 0.5)) if probs.size else 0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if n else 0.0
    return {
        "frames": int(probs.size),
        "vad_mean": float(np.mean(probs)) if probs.size else 0.0,
        "vad_max": float(np.max(probs)) if probs.size else 0.0,
        "vad_speech_frames": speech_frames,
        "speech_frame_ratio": (speech_frames / probs.size) if probs.size else 0.0,
        "rms": rms,
        "speech_detected": bool(speech_frames >= 5),
    }


def process_utterance(uid: str, lang_group: str, prompt: str,
                      audio: np.ndarray, transcriber) -> dict:
    """Run the full production pipeline over in-memory audio (no exec)."""
    import voice.command_listener as CL
    from voice.audio_processing import float32_to_int16
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent
    from agent.task_continuation import classify_confirmation

    dur_ms = len(audio) / SAMPLE_RATE * 1000.0
    pcm = float32_to_int16(np.asarray(audio, dtype=np.float32)).tobytes()
    vad = vad_diagnostics(np.asarray(audio, dtype=np.float32))

    # Primary English pass (timed decode only).
    t0 = time.perf_counter()
    raw_primary, conf_primary = transcriber.transcribe_with_language(
        pcm, SAMPLE_RATE, CL.FALLBACK_PRIMARY_LANG)
    primary_latency_ms = (time.perf_counter() - t0) * 1000.0

    primary_text = CL._postprocess(raw_primary) if raw_primary else ""
    primary_valid, primary_fail = CL._validate_transcript(
        primary_text, confidence=conf_primary, speech_dur_ms=dur_ms)
    suspicious, susp_reason = CL.is_primary_suspicious(
        primary_text, conf_primary, dur_ms)

    fallback_used = False
    fb_raw, fb_conf, fb_latency_ms = "", 0.0, 0.0
    fb_text, fb_valid, fb_fail = "", False, ""
    if suspicious and CL.MULTILINGUAL_FALLBACK_ENABLED:
        fallback_used = True
        t1 = time.perf_counter()
        fb_raw, fb_conf = transcriber.transcribe_with_language(
            pcm, SAMPLE_RATE, CL.FALLBACK_SECONDARY_LANG)
        fb_latency_ms = (time.perf_counter() - t1) * 1000.0
        fb_text = CL._postprocess(fb_raw) if fb_raw else ""
        fb_valid, fb_fail = CL._validate_transcript(
            fb_text, confidence=fb_conf, speech_dur_ms=dur_ms)

    if fallback_used:
        selected_text, selected_conf, selected_lang, select_reason = \
            CL.select_best_transcript(primary_text, conf_primary,
                                      fb_text, fb_conf, dur_ms)
    else:
        selected_text, selected_conf = primary_text, conf_primary
        selected_lang, select_reason = "en", "primary_confident_no_fallback"

    try:
        normalized = command_normalizer.normalize(selected_text)
    except Exception as e:  # never crash the harness on normalize
        normalized = f"<normalize-error: {e}>"
    try:
        auth = authorize_intent(normalized, stt_confidence=selected_conf,
                                audio_duration_ms=dur_ms)
        intent = {"category": auth.category.name if hasattr(auth.category, "name")
                  else str(auth.category),
                  "route": auth.route,
                  "actionable": bool(auth.actionable),
                  "llm_allowed": bool(auth.llm_allowed),
                  "confidence": float(auth.confidence),
                  "reason": auth.reason}
    except Exception as e:
        intent = {"error": f"{e}", "actionable": False}
    try:
        confirm = classify_confirmation(selected_text)
    except Exception:
        confirm = None

    if not selected_text:
        disposition = "rejected"
    elif intent.get("actionable"):
        disposition = "actionable"
    else:
        disposition = "uncertain"

    total_latency_ms = primary_latency_ms + fb_latency_ms
    return {
        "id": uid,
        "language_group": lang_group,
        "prompt": prompt,
        "audio_duration_ms": round(dur_ms, 1),
        "vad": vad,
        "primary_transcript": primary_text,
        "primary_raw": raw_primary,
        "primary_avg_logprob": float(conf_primary),
        "primary_valid": bool(primary_valid),
        "primary_failure": primary_fail,
        "fallback_triggered": bool(fallback_used),
        "fallback_reason": (susp_reason + " -> " + select_reason) if fallback_used else susp_reason,
        "fallback_transcript": fb_text,
        "fallback_avg_logprob": float(fb_conf) if fallback_used else None,
        "fallback_valid": bool(fb_valid) if fallback_used else None,
        "selected_transcript": selected_text,
        "selected_confidence": float(selected_conf),
        "selected_language": selected_lang,
        "selection_reason": select_reason,
        "primary_latency_ms": round(primary_latency_ms, 1),
        "fallback_latency_ms": round(fb_latency_ms, 1),
        "total_stt_latency_ms": round(total_latency_ms, 1),
        "normalized_command": normalized,
        "intent": intent,
        "confirmation_word": confirm,
        "disposition": disposition,
        "executed": False,
    }


def summarize(results: list[dict]) -> dict:
    def _pct(vals):
        import statistics as st
        vals = sorted(vals)
        if not vals:
            return {"n": 0}
        return {"n": len(vals), "median": round(st.median(vals), 1),
                "p95": round(vals[max(0, int(len(vals) * 0.95) - 1)], 1),
                "min": round(vals[0], 1), "max": round(vals[-1], 1)}

    prim = [r["primary_latency_ms"] for r in results]
    fb = [r["fallback_latency_ms"] for r in results if r["fallback_triggered"]]
    tot = [r["total_stt_latency_ms"] for r in results]
    nofb = [r["total_stt_latency_ms"] for r in results if not r["fallback_triggered"]]
    wfb = [r["total_stt_latency_ms"] for r in results if r["fallback_triggered"]]
    by_lang: dict[str, dict] = {}
    for r in results:
        g = by_lang.setdefault(r["language_group"],
                               {"n": 0, "actionable": 0, "uncertain": 0, "rejected": 0,
                                "fallback_used": 0})
        g["n"] += 1
        g[r["disposition"]] += 1
        g["fallback_used"] += int(r["fallback_triggered"])
    return {
        "primary_latency_ms": _pct(prim),
        "fallback_latency_ms": _pct(fb),
        "total_latency_ms": _pct(tot),
        "total_no_fallback_ms": _pct(nofb),
        "total_with_fallback_ms": _pct(wfb),
        "fallback_rate": (sum(1 for r in results if r["fallback_triggered"])
                          / len(results)) if results else 0.0,
        "by_language_group": by_lang,
        "fallback_cases": [
            {"id": r["id"], "primary": r["primary_transcript"],
             "fallback": r["fallback_transcript"],
             "selected": r["selected_transcript"],
             "selected_language": r["selected_language"],
             "reason": r["fallback_reason"]} for r in results
            if r["fallback_triggered"]],
    }


def run_smoke() -> int:
    """No-mic self-test: synthetic voiced tone through VAD + gate logic."""
    import voice.command_listener as CL

    sr = SAMPLE_RATE
    t = np.arange(int(sr * 1.0)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    vad = vad_diagnostics(tone)
    assert vad["frames"] > 0, "VAD produced no frames"
    assert 0.0 <= vad["vad_mean"] <= 1.0
    susp, reason = CL.is_primary_suspicious("open firefox", -0.4, 2500.0)
    assert susp is False and reason == "primary_confident"
    susp2, _ = CL.is_primary_suspicious("", -0.4, 2500.0)
    assert susp2 is True
    sel = CL.select_best_transcript("open firefox", -0.4, "", -99.0, 2500.0)
    assert sel[0] == "open firefox" and sel[2] == "en"
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent
    normed = command_normalizer.normalize("open firefox")
    auth = authorize_intent(normed, stt_confidence=-0.4, audio_duration_ms=2500.0)
    assert auth.actionable is True
    print(f"SMOKE OK: vad_frames={vad['frames']} rms={vad['rms']:.1f} "
          f"normalize={normed!r} actionable={auth.actionable}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20D mic validation (read-only)")
    ap.add_argument("--smoke", action="store_true", help="no-mic self-test")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    ap.add_argument("--prep-seconds", type=float, default=6.0)
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    args = ap.parse_args()

    if args.list_devices:
        return list_devices()
    if args.smoke:
        return run_smoke()

    import voice.command_listener as CL

    print("=== Phase 20D config (read-only, NOT modified) ===")
    print(f"whisper model=base device/cpu-int8 backend={Path('data/whisper_backend.json').read_text().strip()}")
    print(f"primary_lang={CL.FALLBACK_PRIMARY_LANG} secondary={CL.FALLBACK_SECONDARY_LANG} "
          f"max_passes={CL.FALLBACK_MAX_PASSES} fallback_enabled={CL.MULTILINGUAL_FALLBACK_ENABLED}")
    print(f"suspicion gate: healthy>={CL.SUSPICIOUS_HEALTHY_CONFIDENCE} "
          f"weak<{CL.SUSPICIOUS_WEAK_CONFIDENCE} min_dur={CL.SUSPICIOUS_MIN_DURATION_MS}ms "
          f"margin={CL.LOGPROB_DECISIVE_MARGIN}")
    print("Raw audio is NEVER stored. Only transcripts + metrics are saved.")

    transcriber = CL._WhisperTranscriber()
    print("Loading faster-whisper base (unchanged production model)...")
    if not transcriber.load():
        print("ERROR: Whisper failed to load; aborting (no config changed).")
        return 2
    print(f"Whisper ready (device={transcriber._device} compute={transcriber._compute}).")

    results: list[dict] = []
    for uid, group, prompt in UTTERANCES:
        print(f"\n--- [{uid}] ({group}) Say: {prompt!r} ---")
        print(f"GET READY — speak right after the countdown ({args.prep_seconds:.0f}s prep) ...", flush=True)
        _prep = max(0.0, float(args.prep_seconds))
        _t0 = time.time()
        while time.time() - _t0 < _prep:
            _left = _prep - (time.time() - _t0)
            print(f"\r... {_left:.0f}s — say: {prompt!r}   ", end="", flush=True)
            time.sleep(0.5)
        print(f"\rRECORDING {args.seconds:.0f}s — SPEAK NOW: {prompt!r}   ", flush=True)
        audio = record_utterance(args.seconds)
        rec = process_utterance(uid, group, prompt, audio, transcriber)
        del audio  # release PCM immediately; never stored
        results.append(rec)
        print(f"primary={rec['primary_transcript']!r} ({rec['primary_avg_logprob']:.2f}, "
              f"valid={rec['primary_valid']})")
        print(f"fallback={rec['fallback_triggered']} reason={rec['fallback_reason']!r} "
              f"fb={rec['fallback_transcript']!r}")
        print(f"selected[{rec['selected_language']}]={rec['selected_transcript']!r} "
              f"total={rec['total_stt_latency_ms']}ms norm={rec['normalized_command']!r} "
              f"intent={rec['intent'].get('category')}/{rec['intent'].get('route')} "
              f"disposition={rec['disposition']}")

    summary = summarize(results)
    payload = {"phase": "20D", "captured_at": datetime.now(timezone.utc).isoformat(),
               "config_snapshot": {
                   "whisper_model": "base",
                   "primary_lang": CL.FALLBACK_PRIMARY_LANG,
                   "secondary_lang": CL.FALLBACK_SECONDARY_LANG,
                   "max_passes": CL.FALLBACK_MAX_PASSES,
                   "fallback_enabled": CL.MULTILINGUAL_FALLBACK_ENABLED,
                   "healthy_conf": CL.SUSPICIOUS_HEALTHY_CONFIDENCE,
                   "weak_conf": CL.SUSPICIOUS_WEAK_CONFIDENCE,
                   "min_dur_ms": CL.SUSPICIOUS_MIN_DURATION_MS,
                   "margin": CL.LOGPROB_DECISIVE_MARGIN},
               "utterances": results, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved transcripts+metrics ONLY to {out} (no audio stored).")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("NOTE: no actions executed; end-to-end rows are authorize-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
