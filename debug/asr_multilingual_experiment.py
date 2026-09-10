#!/usr/bin/env python3
"""
PHASE 20A — Multilingual speech-recognition controlled experiment (READ-ONLY
over the production stack).

This harness does NOT modify any production file. It:

  1. Generates controlled recordings (gTTS) for English / Hinglish / Hindi /
     mixed utterances (live microphone was proven unreliable for audits —
     see debug/asr_accent_audit_results.json hallucination records).
  2. Transcribes the SAME audio through the REAL production Whisper path
     (voice.command_listener._WhisperTranscriber.transcribe, base/int8/CPU,
     beam_size=1, best_of=1, temperature=0) under three language modes:
       A. lang_code="en"  → language="en" (production behaviour today)
       B. lang_code="hi"  → language="hi"
       C. lang_code=""    → language=None (automatic detection)
     The mode is applied by mutating the runtime settings singleton INSIDE
     this process only — no production configuration is changed.
  3. Pipes every result through the UNCHANGED downstream pipeline:
       _postprocess → _validate_transcript → command_normalizer.normalize
       → is_low_quality_transcript → authorize_intent → decision_engine
  4. Measures per-utterance: transcript, detected language + probability,
     avg_logprob, STT latency (3 reps), word count, hallucination flags,
     normalization, intent, decision path, actionable/uncertain/rejected.
  5. Simulates a dual-pass strategy latency budget (en-pass + hi-pass) from
     the already-collected per-mode latencies (no extra inference).

Results → debug/asr_multilingual_experiment_results.json

Usage:
    .venv/bin/python debug/asr_multilingual_experiment.py [--reps 3]
"""

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"

import compat  # noqa: F401,E402

import numpy as np  # noqa: E402

from voice.settings import voice_settings  # noqa: E402
from voice.command_listener import (  # noqa: E402
    _WhisperTranscriber, _postprocess, _validate_transcript,
    _is_repeated_hallucination, is_garbage, is_low_quality_transcript,
)
from nlp.command_normalizer import command_normalizer  # noqa: E402
from nlp.intent_authorizer import authorize_intent  # noqa: E402
from core.decision_engine import decision_engine  # noqa: E402

SAMPLE_RATE = 16000
AUDIO_DIR = PROJECT_ROOT / "debug" / "multilingual_audio"
RESULTS_PATH = PROJECT_ROOT / "debug" / "asr_multilingual_experiment_results.json"

# Language modes under test. "" → _whisper_language() returns None → auto.
MODES = {"A_forced_en": "en", "B_forced_hi": "hi", "C_auto": ""}


# Acoustic language we EXPECT the recording to carry (for detection accuracy
# reporting only — Hinglish is intentionally ambiguous: code-mixed).
UTTERANCES = [
    # ── English ──
    {"id": "en_open_firefox", "text": "Open Firefox", "acoustic_lang": "en",
     "tts_lang": "en", "tld": "co.in",
     "expect_any": {"firefox", "open"}},
    {"id": "en_play_believer", "text": "Play Believer on YouTube",
     "acoustic_lang": "en", "tts_lang": "en", "tld": "co.in",
     "expect_any": {"play", "believer", "youtube"}},
    {"id": "en_whats_my_ram", "text": "What's my RAM?",
     "acoustic_lang": "en", "tts_lang": "en", "tld": "co.in",
     "expect_any": {"ram", "what"}},
    # ── Hinglish (Latin script, code-mixed) ──
    {"id": "hi_firefox_kholo", "text": "Firefox kholo", "acoustic_lang": "hinglish",
     "tts_lang": "hi", "tld": "com",
     "expect_any": {"firefox", "kholo", "open"}},
    {"id": "hi_youtube_pe_music_chalao", "text": "YouTube pe music chalao",
     "acoustic_lang": "hinglish", "tts_lang": "hi", "tld": "com",
     "expect_any": {"youtube", "music", "chalao", "play"}},
    {"id": "hi_volume_thoda_kam_karo", "text": "volume thoda kam karo",
     "acoustic_lang": "hinglish", "tts_lang": "hi", "tld": "com",
     "expect_any": {"volume", "thoda", "kam", "down"}},
    {"id": "hi_mera_cpu_kitna", "text": "mera CPU kitna use ho raha hai",
     "acoustic_lang": "hinglish", "tts_lang": "hi", "tld": "com",
     "expect_any": {"cpu", "kitna", "use"}},
    # ── Hindi (Devanagari script) ──
    {"id": "dev_youtube_gana_chalao", "text": "यूट्यूब पर गाना चलाओ",
     "acoustic_lang": "hi", "tts_lang": "hi", "tld": "com",
     "expect_any": {"youtube", "यूट्यूब", "गाना", "chalao", "chalau", "gana"}},
    {"id": "dev_firefox_kholo", "text": "फ़ायरफ़ॉक्स खोलो",
     "acoustic_lang": "hi", "tts_lang": "hi", "tld": "com",
     "expect_any": {"firefox", "फ़ायरफ़ॉक्स", "kholo", "खोलो"}},
    # ── Mixed entities ──
    {"id": "mix_arijit_singh", "text": "YouTube pe Arijit Singh ka song chalao",
     "acoustic_lang": "hinglish", "tts_lang": "hi", "tld": "com",
     "expect_any": {"youtube", "arijit", "singh", "song", "chalao", "play"}},
    # ── Problematic phrase (semantics deliberately NOT assumed) ──
    {"id": "prob_thodi_der", "text": "play thodi der on youtube",
     "acoustic_lang": "hinglish", "tts_lang": "hi", "tld": "com",
     "expect_any": {"play", "thodi", "der", "youtube"}},
]


# ── Controlled-audio generation ───────────────────────────────────

def generate_audio(u: dict) -> Path:
    """Generate (once, then cache) a 16 kHz mono WAV for the utterance via gTTS."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = AUDIO_DIR / f"{u['id']}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return wav_path
    mp3_path = AUDIO_DIR / f"{u['id']}.mp3"
    from gtts import gTTS
    tts = gTTS(text=u["text"], lang=u["tts_lang"], tld=u.get("tld", "com"))
    tts.save(str(mp3_path))
    # Decode mp3 → 16 kHz mono s16 WAV (faster-whisper's expected layout).
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
         str(wav_path)],
        check=True)
    mp3_path.unlink(missing_ok=True)
    return wav_path


def load_wav_float32(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, f"{path}: rate={wf.getframerate()}"
        assert wf.getnchannels() == 1, f"{path}: channels={wf.getnchannels()}"
        pcm = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def pct(values, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(q * (len(s) - 1) + 0.5), len(s) - 1)
    return round(s[idx], 1)


# ── Hallucination heuristic ───────────────────────────────────────

def hallucination_flags(text: str, u: dict) -> dict:
    t = (text or "").strip()
    words = t.lower().translate(str.maketrans("", "", ".,!?;:'\"")).split()
    overlap = sorted(set(words) & {w.lower() for w in u["expect_any"]}) \
        if t else []
    return {
        "repeated_hallucination": bool(t) and _is_repeated_hallucination(t),
        "garbage": bool(t) and is_garbage(t),
        "no_expected_overlap": bool(t) and not overlap,
        "expected_overlap": overlap,
    }


# ── Language metadata reader (no production behaviour change) ─────

def _detect_language(transcriber, audio, lang_val):
    """Read detected language + probability from faster-whisper info.

    When language is forced, faster-whisper still returns the (forced) code
    in info.language — we record whatever the library reports.
    """
    try:
        _segments, info = transcriber._model.transcribe(
            audio, beam_size=1, language=lang_val if lang_val else None,
            temperature=0.0, best_of=1, condition_on_previous_text=False,
            compression_ratio_threshold=None, no_speech_threshold=0.9,
            vad_filter=False, without_timestamps=True)
        lang = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return lang, prob
    except Exception as e:
        return f"error:{type(e).__name__}", 0.0

# ── Main experiment ───────────────────────────────────────────────

async def main() -> None:
    reps = 3
    if "--reps" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--reps") + 1])

    print("[EXP] generating controlled recordings (gTTS) ...", flush=True)
    audio_cache = {}
    for u in UTTERANCES:
        wav = generate_audio(u)
        audio_cache[u["id"]] = load_wav_float32(wav)
        print(f"    {u['id']}: {len(audio_cache[u['id']]) / SAMPLE_RATE:.2f}s", flush=True)

    print("[EXP] loading production Whisper (base/int8/cpu) ...", flush=True)
    transcriber = _WhisperTranscriber()
    if not transcriber.load():
        print("[EXP] FATAL: production Whisper failed to load", flush=True)
        return
    orig_lang_code = voice_settings.lang_code
    print(f"[EXP] production lang_code default = {orig_lang_code!r}", flush=True)

    results = []
    try:
        for mode_name, lang_val in MODES.items():
            # Runtime-only mutation inside THIS process. Production files,
            # .env, and the global default are untouched on disk.
            voice_settings.lang_code = lang_val
            for u in UTTERANCES:
                audio = audio_cache[u["id"]]
                pcm = pcm_bytes(audio)
                dur_ms = len(audio) / SAMPLE_RATE * 1000.0
                reps_out = []
                for r in range(reps):
                    t0 = time.time()
                    text_raw, conf = transcriber.transcribe(pcm, SAMPLE_RATE)
                    latency_ms = (time.time() - t0) * 1000.0
                    detected_lang, lang_prob = _detect_language(
                        transcriber, audio, lang_val)
                    reps_out.append({
                        "rep": r,
                        "transcript_raw": text_raw,
                        "confidence_avg_logprob": round(conf, 3),
                        "stt_latency_ms": round(latency_ms, 1),
                        "detected_language": detected_lang,
                        "language_probability": (
                            round(lang_prob, 4) if lang_prob else None),
                    })
                # Use the LAST rep's transcript for downstream evaluation
                # (temperature=0 → deterministic; reps exist for latency stats).
                rep_last = reps_out[-1]
                text_raw = rep_last["transcript_raw"]
                conf = rep_last["confidence_avg_logprob"]
                text = _postprocess(text_raw or "")
                accepted, failure = _validate_transcript(
                    text, conf, dur_ms,
                    language_prob=rep_last["language_probability"])
                normalized = command_normalizer.normalize(text) if text else ""
                auth = authorize_intent(
                    text, stt_confidence=conf, audio_duration_ms=dur_ms)
                decision = None
                if normalized and accepted:
                    try:
                        d = await decision_engine.decide(normalized)
                        decision = {"path": d.path.value,
                                    "needs_llm": d.needs_llm,
                                    "action": d.action}
                    except Exception as e:
                        decision = {"error": f"{type(e).__name__}: {e}"}

                if not accepted:
                    outcome = "rejected"
                elif decision and decision.get("action"):
                    outcome = "actionable"
                else:
                    outcome = "uncertain"

                record = {
                    "mode": mode_name,
                    "mode_lang_arg": lang_val if lang_val else None,
                    "utterance_id": u["id"],
                    "utterance": u["text"],
                    "acoustic_lang": u["acoustic_lang"],
                    "audio_duration_ms": round(dur_ms, 1),
                    "word_count": len(text.split()) if text else 0,
                    "transcript_postprocessed": text,
                    "normalized": normalized,
                    "low_quality_guard": (
                        is_low_quality_transcript(normalized)
                        if normalized else True),
                    "validate_accepted": accepted,
                    "validate_failure_reason": failure,
                    "intent": {
                        "category": auth.category.value,
                        "actionable": auth.actionable,
                        "llm_allowed": auth.llm_allowed,
                        "confidence": round(auth.confidence, 3),
                        "reason": auth.reason,
                    },
                    "decision": decision,
                    "outcome": outcome,
                    "hallucination": hallucination_flags(text_raw or "", u),
                    "reps": reps_out,
                }
                results.append(record)
                med_lat = statistics.median(
                    r["stt_latency_ms"] for r in reps_out)
                print(f"[EXP] {mode_name:12s} {u['id']:26s} "
                      f"lang={rep_last['detected_language']}"
                      f"({rep_last['language_probability']}) "
                      f"text={text!r} conf={conf:.2f} "
                      f"lat={med_lat:.0f}ms → {outcome}", flush=True)
                with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, ensure_ascii=False, indent=2)
    finally:
        voice_settings.lang_code = orig_lang_code

    _summarize(results)
    print(f"[EXP] saved {len(results)} records + summary → {RESULTS_PATH}",
          flush=True)


def _summarize(results: list) -> None:
    """Aggregate latency / detection / outcome summaries and re-save."""
    summary = {"latency": {}, "detection": {}, "outcomes": {}}
    for mode_name in MODES:
        lat = [r["stt_latency_ms"] for rec in results if rec["mode"] == mode_name
               for r in rec["reps"]]
        summary["latency"][mode_name] = {
            "n": len(lat),
            "median_ms": round(statistics.median(lat), 1) if lat else 0.0,
            "p95_ms": pct(lat, 0.95),
            "mean_ms": round(statistics.fmean(lat), 1) if lat else 0.0,
        }
    # Dual-pass simulation: per utterance, median en-pass + median hi-pass.
    dual = []
    for u in UTTERANCES:
        en = [r["stt_latency_ms"] for rec in results
              if rec["mode"] == "A_forced_en" and rec["utterance_id"] == u["id"]
              for r in rec["reps"]]
        hi = [r["stt_latency_ms"] for rec in results
              if rec["mode"] == "B_forced_hi" and rec["utterance_id"] == u["id"]
              for r in rec["reps"]]
        if en and hi:
            dual.append(statistics.median(en) + statistics.median(hi))
    summary["latency"]["D_dual_pass_sim"] = {
        "n": len(dual),
        "median_ms": round(statistics.median(dual), 1) if dual else 0.0,
        "p95_ms": pct(dual, 0.95),
        "mean_ms": round(statistics.fmean(dual), 1) if dual else 0.0,
        "note": "sequential en-pass + hi-pass on the SAME audio (simulated)",
    }
    # Detection accuracy vs expected acoustic language.
    for mode_name in MODES:
        hits, total, detail = 0, 0, []
        for rec in results:
            if rec["mode"] != mode_name:
                continue
            total += 1
            det = rec["reps"][-1]["detected_language"]
            ok = (det == rec["acoustic_lang"]) or (
                rec["acoustic_lang"] == "hinglish" and det in ("hi", "en"))
            hits += ok
            detail.append({"id": rec["utterance_id"],
                           "acoustic": rec["acoustic_lang"],
                           "detected": det, "expected_match": bool(ok)})
        summary["detection"][mode_name] = {
            "correct": hits, "total": total, "detail": detail}
    # Outcome counts per mode.
    for mode_name in MODES:
        counts = {}
        for rec in results:
            if rec["mode"] == mode_name:
                counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        summary["outcomes"][mode_name] = counts
    payload = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())

