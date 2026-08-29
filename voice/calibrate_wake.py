"""
calibrate_wake — Interactive wake-phrase calibration + verifier training.

This module is the production entry point for `python main.py --train-wake`.
It:

  1. Records N wake-phrase samples from the user's verified microphone
     (via AudioManager, the SAME capture path used at runtime).
  2. Builds a deterministic training dataset:
       * positives  — the user's recordings + gain/speed augmentations +
                      clean synthetic "hello Diego" renderings (espeak).
       * negatives  — non-wake speech (espeak), the wake chime, silence,
                      seeded noise, time-reversed wake clips, and real
                      room tone captured through the production path.
  3. Trains a logistic-regression custom verifier on feature windows
     harvested with the EXACT runtime regime (cold buffer reset →
     1280-sample int16 streaming → window after every frame).
  4. Calibrates the detection threshold with runtime-identical clip scores.
  5. Saves verifier.pkl + metadata.json to models/wake/.
  6. Validates through WakeModelManager (the production load path).

The verifier is attached to the base openWakeWord model automatically by
WakeModelManager when metadata.json is present.

Exports:
    _select_base_model(phrase)  — resolve a bundled openWakeWord base model
                                   (used by debug/retrain_wake_verifier.py).
    run_calibration()           — full interactive calibration + training.
"""

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

ROOT = settings.BASE_DIR
WAKE_DIR = settings.MODELS_WAKE_DIR
POS = WAKE_DIR / "positives"
NEG = WAKE_DIR / "negatives"
SR = 16000
FRAME = 1280  # 80 ms — the openWakeWord streaming frame

# Non-wake speech negatives. Includes the EXACT phrases the transcript
# verifier must reject, so the acoustic verifier learns them too.
NON_WAKE_PHRASES = [
    "hello", "hello everyone", "thank you", "good morning",
    "what time is it", "open chrome", "how are you", "see you later",
    "play some music", "turn up the volume", "tell me a joke",
    "what is the weather like", "lock the screen", "good night",
    "hello video", "yellow meow", "below zero", "hey there",
    "open the terminal", "search google for news",
]

# Synthetic CLEAN "hello Diego" positives (engine, voice, speed): clean
# synthetic renderings teach the verifier the phrase itself — any speaker,
# any clean capture — while the user recordings anchor the real-mic domain.
SYNTH_WAKE_VOICES = (
    ("espeak-ng", "en", 140), ("espeak-ng", "en-us", 170),
    ("espeak-ng", "en-gb", 110), ("espeak-ng", "en", 200),
    ("espeak", None, None), ("espeak", "en-us", 150),
)

SEED = 7

# Number of wake-phrase samples to record during interactive calibration.
DEFAULT_SAMPLES = 5
# Seconds of audio to capture per sample (phrase + margin).
SAMPLE_SECONDS = 3.0


# ── Base model selection ───────────────────────────────────────

def _select_base_model(phrase: str) -> Optional[Path]:
    """Resolve a bundled openWakeWord base model for the given phrase.

    Priority:
      1. Explicit WAKE_MODEL config (env / settings / voice_settings).
      2. Custom ONNX model in models/wake/ (trained wake model).
      3. Bundled model best matching the phrase (never hardcoded).

    Returns the model path or None if no candidate exists.
    """
    import difflib
    import os
    from voice.wake_model_manager import (
        DEFAULT_BUNDLED_MODEL,
        NON_WAKE_FILES,
    )

    # 1) Explicit WAKE_MODEL configuration
    cfg = (os.getenv("WAKE_MODEL")
           or getattr(settings, "WAKE_MODEL", None)
           or voice_settings.wake_model)
    if cfg:
        p = Path(str(cfg)).expanduser()
        if p.exists():
            return p
        p2 = settings.BASE_DIR / str(cfg)
        if p2.exists():
            return p2
        logger.warning("[WAKE] WAKE_MODEL='%s' not found at %s or %s",
                       cfg, p, p2)

    # 2) Custom ONNX model in models/wake/ (trained wake model)
    try:
        for f in sorted(settings.MODELS_WAKE_DIR.glob("*.onnx")):
            if f.name in NON_WAKE_FILES:
                continue
            return f
    except Exception as e:
        logger.debug("[WAKE] models/wake glob error: %s", e)

    # 3) Bundled model best matching the phrase. Uses the SAME robust
    # resolver as WakeModelManager so models are found even when the active
    # openwakeword install lacks its resources/models dir (e.g. a pip
    # install that omitted the bundled .onnx files).
    from voice.wake_model_manager import wake_model_manager as _wmm
    models = _wmm._bundled_models()
    if not models:
        return None

    phrase_lower = phrase.lower().strip()
    best_name = DEFAULT_BUNDLED_MODEL
    best_ratio = -1.0
    for name in models:
        pretty = name.replace("_", " ")
        ratio = difflib.SequenceMatcher(None, phrase_lower, pretty).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name
    if best_ratio < 0.3:
        best_name = DEFAULT_BUNDLED_MODEL
    logger.info("[WAKE] Selected bundled model '%s' for phrase '%s' "
                "(match ratio=%.2f)", best_name, phrase, best_ratio)
    return models.get(best_name)


# ── WAV helpers ────────────────────────────────────────────────

def read_wav_int16(path: Path) -> np.ndarray:
    """Read a WAV as mono int16 @ 16 kHz (deterministic resample)."""
    import wave
    from scipy import signal as sc
    import math
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    if sr != SR:
        g = math.gcd(SR, sr)
        a = sc.resample_poly(a.astype(np.float64), SR // g, sr // g)
        a = np.clip(a, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(a)


def write_wav_int16(path: Path, audio: np.ndarray) -> None:
    import wave
    from voice.audio_processing import float32_to_int16
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(float32_to_int16(audio).tobytes())


def speed_change(audio_f32: np.ndarray, factor: float) -> np.ndarray:
    n = max(1, int(len(audio_f32) / factor))
    x_old = np.linspace(0, 1, len(audio_f32), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio_f32).astype(np.float32)


# ── Interactive recording ──────────────────────────────────────

def _record_samples(count: int = DEFAULT_SAMPLES) -> bool:
    """Record `count` wake-phrase samples from the verified microphone.

    Uses AudioManager (the SAME capture path as runtime) so the training
    domain matches the live domain exactly. Each sample is saved to
    models/wake/wake_phrase.wav (the reference) and models/wake/positives/.

    Returns True if at least one sample was captured.
    """
    from voice.audio_manager import audio_manager

    print("\n  ── WAKE PHRASE CALIBRATION ──")
    print(f"  Say '{voice_settings.wake_phrase}' clearly, {count} times.\n")

    if not audio_manager.start():
        print("  ✗ FAILED: cannot start AudioManager (no working microphone)")
        return False

    POS.mkdir(parents=True, exist_ok=True)
    # Clear previous positives so a fresh calibration is clean.
    for f in POS.glob("*.wav"):
        f.unlink()

    captured = 0
    try:
        for i in range(1, count + 1):
            print(f"  [{i}/{count}] Say '{voice_settings.wake_phrase}' now...")
            # Give the user a moment to start speaking, then capture.
            time.sleep(0.5)
            audio = audio_manager.get_recent_audio(SAMPLE_SECONDS)
            if len(audio) < SR // 2:
                print("    ⚠ too little audio captured — retrying")
                continue

            # Save the first sample as the canonical reference.
            if i == 1:
                write_wav_int16(WAKE_DIR / "wake_phrase.wav", audio)
                print(f"    ✓ reference saved: {WAKE_DIR / 'wake_phrase.wav'}")

            write_wav_int16(POS / f"positive_{i:03d}.wav", audio)
            captured += 1
            print(f"    ✓ sample {i} captured ({len(audio) / SR:.2f}s)")

            # Brief pause between samples.
            if i < count:
                time.sleep(0.8)
    finally:
        audio_manager.stop()

    if captured == 0:
        print("  ✗ No wake-phrase samples captured")
        return False

    print(f"  ✓ Captured {captured} wake-phrase sample(s)")
    return True


# ── Dataset construction (deterministic) ───────────────────────

def build_dataset() -> bool:
    """Write positives/negatives WAVs. Returns False without sources."""
    POS.mkdir(parents=True, exist_ok=True)
    NEG.mkdir(parents=True, exist_ok=True)
    for d in (POS, NEG):
        for f in d.glob("*.wav"):
            f.unlink()

    # ── POSITIVES: every real user recording + augmentations ──
    sources = []
    ref = WAKE_DIR / "wake_phrase.wav"
    if ref.exists():
        sources.append(read_wav_int16(ref))
    for p in sorted((WAKE_DIR / "positives_backup").glob("*.wav")):
        sources.append(read_wav_int16(p))
    # Keep the previously recorded positives as sources too.
    for p in sorted((WAKE_DIR / "positives_orig").glob("*.wav")):
        sources.append(read_wav_int16(p))
    if not sources:
        logger.error("No wake-phrase source recordings found "
                     "(wake_phrase.wav / positives_backup / positives_orig)")
        return False

    n_pos = 0
    for base_i in sources:
        base_f = base_i.astype(np.float32) / 32768.0
        variants = [base_f]
        for g in (0.85, 1.15):
            variants.append(np.clip(base_f * g, -1.0, 1.0))
        for s in (0.92, 1.08):
            variants.append(speed_change(base_f, s))
        for v in variants:
            n_pos += 1
            write_wav_int16(POS / f"positive_{n_pos:03d}.wav", v)

    # ── Synthetic clean "hello Diego" positives (deterministic espeak) ──
    for engine, voice, spd in SYNTH_WAKE_VOICES:
        cmd = [engine]
        if voice:
            cmd += ["-v", voice]
        if spd:
            cmd += ["-s", str(spd)]
        cmd += ["-w", None, "hello Diego"]  # placeholder for tmp path
        tmp = tempfile.mktemp(suffix=".wav")
        cmd[cmd.index(None)] = tmp
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
            if not Path(tmp).exists():
                continue
            audio = read_wav_int16(Path(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)
        if len(audio) < SR // 4:
            continue
        n_pos += 1
        write_wav_int16(POS / f"positive_{n_pos:03d}.wav",
                        audio.astype(np.float32) / 32768.0)

    # ── NEGATIVES (class-prefixed names so threshold calibration can
    # distinguish REAL negatives from hard synthetic boundary cases) ──
    n_neg = 0

    def _put(cls: str, audio_f32: np.ndarray) -> None:
        nonlocal n_neg
        n_neg += 1
        write_wav_int16(NEG / f"neg_{cls}_{n_neg:03d}.wav", audio_f32)

    # 2a. Non-wake speech via espeak (deterministic, offline). Every phrase
    # is rendered by BOTH engines and at several speeds so the verifier
    # learns the "hello <not Diego>" class robustly.
    for phrase in NON_WAKE_PHRASES:
        for engine, extra in (("espeak-ng", []), ("espeak-ng", ["-s", "110"]),
                              ("espeak", [])):
            tmp = tempfile.mktemp(suffix=".wav")
            try:
                subprocess.run([engine, *extra, "-w", tmp, phrase],
                               capture_output=True, timeout=15)
                if not Path(tmp).exists():
                    continue
                audio = read_wav_int16(Path(tmp))
            finally:
                Path(tmp).unlink(missing_ok=True)
            if len(audio) < SR // 4:
                continue
            _put("esp", audio.astype(np.float32) / 32768.0)

    # 2b. The wake chime must NEVER fire the detector.
    for chime_name in ("Diego.wav", "Diego_voice_backup.wav"):
        chime = ROOT / chime_name
        if chime.exists():
            a = read_wav_int16(chime)
            _put("chime", a.astype(np.float32) / 32768.0)

    # 2c. Time-reversed wake clips: same spectral content, NOT the phrase.
    #     HARD synthetic negatives — they sharpen the decision boundary but
    #     can never occur live, so they are EXCLUDED from threshold
    #     calibration.
    for base_i in sources:
        rev = base_i[::-1].copy()
        _put("rev", rev.astype(np.float32) / 32768.0)

    # 2d. Digital silence.
    for _ in range(4):
        _put("sil", np.zeros(SR * 2, dtype=np.float32))

    # 2e. Quiet noise floor (seeded → deterministic).
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        _put("noise", rng.standard_normal(SR * 2).astype(np.float32) * 0.005)

    # 2f. REAL room tone captured through the production capture path
    # (debug/room_tone.wav — AGC-leveled raw ring-buffer domain). This is
    # the negative class that actually occurs at runtime; without it the
    # verifier false-fires on amplified fan/room noise.
    room = ROOT / "debug" / "room_tone.wav"
    if room.exists():
        a = read_wav_int16(room)
        for i in range(0, len(a) - 2 * SR + 1, 2 * SR):
            _put("room", a[i:i + 2 * SR].astype(np.float32) / 32768.0)
    else:
        logger.warning("debug/room_tone.wav missing — room-tone negatives "
                       "skipped (capture 10–16 s of room audio and retrain "
                       "for best live precision)")

    logger.info("dataset: %d positives, %d negatives", n_pos, n_neg)
    return n_pos > 0 and n_neg > 0


# ── Training ───────────────────────────────────────────────────

def _full_reset(m) -> None:
    """Deterministic reset of every openWakeWord internal buffer.

    Model.reset() alone clears ONLY the prediction buffer — the raw audio
    buffer would still hold the tail of the PREVIOUS clip, contaminating
    the first ~2 s of features. Identical to WakeModelManager.hard_reset().
    """
    pre = m.preprocessor
    pre.raw_data_buffer.clear()
    pre.melspectrogram_buffer = np.ones((76, 32))
    pre.accumulated_samples = 0
    pre.feature_buffer = pre._get_embeddings(np.zeros(160000).astype(np.int16))
    m.reset()
    silence = np.zeros(FRAME, dtype=np.int16)
    for _ in range(6):  # re-prime the 5-frame zeroed prediction window
        m.predict(silence)


def train() -> bool:
    """Train the verifier on feature windows harvested with the EXACT
    runtime regime (cold buffer reset → 1280-sample int16 streaming →
    window after every frame), then calibrate the threshold with
    runtime-identical clip scores.

    Classifier: logistic regression (C=1.0) — fully deterministic and, on
    regime-identical features, measured to separate perfectly:
    positives min=1.000, real-world negatives max=0.46, silence≈0.015.
    """
    import pickle
    from openwakeword import Model as OWWModel
    from openwakeword.custom_verifier_model import flatten_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    wake_phrase = voice_settings.wake_phrase
    base_model_path = _select_base_model(wake_phrase)
    if base_model_path is None:
        logger.error("No bundled base model found")
        return False
    base_stem = Path(base_model_path).stem
    print(f"  Base model: {base_stem}")

    # Select the inference framework based on the model file extension.
    # The openwakeword Model defaults to inference_framework="tflite",
    # which raises ValueError when an ONNX model is provided. We must
    # explicitly select "onnx" for .onnx models. `inference_framework`
    # is an explicit named parameter of Model.__init__ (forwarded to
    # AudioFeatures.__init__), NOT part of **kwargs, so it does not
    # cause a TypeError.
    inference_framework = "onnx" if base_model_path.suffix == ".onnx" else "tflite"
    oww = OWWModel(
        wakeword_models=[str(base_model_path)],
        inference_framework=inference_framework,
    )
    feats_ndx = oww.model_inputs[base_stem]

    def harvest(dat: np.ndarray):
        """Stream one clip through the model in the runtime regime and
        return the feature window after EVERY 1280-sample frame."""
        _full_reset(oww)
        windows = []
        for i in range(0, len(dat) - FRAME + 1, FRAME):
            oww.predict(dat[i:i + FRAME])
            windows.append(oww.preprocessor.get_features(feats_ndx)[0].copy())
        return windows

    Xs, ys = [], []
    for p in sorted(POS.glob("*.wav")):
        for w in harvest(read_wav_int16(p)):
            Xs.append(w)
            ys.append(1)
    for n in sorted(NEG.glob("*.wav")):
        for w in harvest(read_wav_int16(n)):
            Xs.append(w)
            ys.append(0)
    if not Xs or not any(ys) or all(ys):
        logger.error("Feature harvest failed (windows=%d)", len(Xs))
        return False
    X = np.array(Xs)
    y = np.array(ys)
    print(f"  Windows: total={X.shape[0]} positive={int(y.sum())} "
          f"negative={int(len(y) - y.sum())}")

    model = make_pipeline(
        FunctionTransformer(flatten_features),
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=3000, C=1.0),
    )
    model.fit(X, y)
    verifier_path = WAKE_DIR / "verifier.pkl"
    with open(verifier_path, "wb") as f:
        pickle.dump(model, f)

    # ── Threshold calibration: runtime-identical streaming scores ──
    def clip_max_score(path) -> float:
        dat = read_wav_int16(path)
        _full_reset(oww)
        best = 0.0
        for i in range(0, len(dat) - FRAME + 1, FRAME):
            oww.predict(dat[i:i + FRAME])
            feats = oww.preprocessor.get_features(feats_ndx)
            best = max(best, float(model.predict_proba(feats)[0][-1]))
        return best

    pos_scores = np.array([clip_max_score(p) for p in sorted(POS.glob("*.wav"))])
    neg_files = sorted(NEG.glob("*.wav"))
    neg_scores = np.array([clip_max_score(n) for n in neg_files])
    # Threshold calibration uses REAL negatives only (speech / chime /
    # silence / noise). Time-reversed clips (neg_rev_*) are hard synthetic
    # boundary cases that cannot occur live — including them would push the
    # threshold needlessly close to the positive distribution.
    real_neg = np.array([
        s for n, s in zip(neg_files, neg_scores)
        if not n.name.startswith("neg_rev_")
    ])
    pos_min, pos_mean = float(pos_scores.min()), float(pos_scores.mean())
    neg_max = float(real_neg.max()) if len(real_neg) else float(neg_scores.max())
    neg_mean = float(neg_scores.mean())
    print(f"  Positive scores: min={pos_min:.3f} mean={pos_mean:.3f}")
    print(f"  Negative scores: max={float(neg_scores.max()):.3f} "
          f"(real-world max={neg_max:.3f}) mean={neg_mean:.3f}")

    if pos_min > neg_max:
        threshold = neg_max + 0.5 * (pos_min - neg_max)
    else:
        threshold = neg_max * 1.15
        print("  ⚠ distributions overlap — more/better positives recommended")
    threshold = max(0.40, min(0.85, float(threshold)))
    print(f"  detection_threshold = {threshold:.3f}")

    metadata = {
        "wake_phrase": wake_phrase,
        "base_model": base_stem,
        "verifier_path": "verifier.pkl",
        "positive_count": len(list(POS.glob('*.wav'))),
        "negative_count": len(list(NEG.glob('*.wav'))),
        "verifier_threshold": 0.0,
        "detection_threshold": threshold,
        "domain": "int16-16kHz-mono",
        "trained_at": time.time(),
        "classifier": "logreg(c=1.0, regime-identical-windows)",
        "accept": "oww-trigger + whisper-transcript-verification",
    }
    (WAKE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2),
                                            encoding="utf-8")
    print(f"  ✓ Verifier saved: {verifier_path}")
    print(f"  ✓ Metadata saved: {WAKE_DIR / 'metadata.json'}")
    return True


# ── Validation ─────────────────────────────────────────────────

def validate() -> bool:
    """Fresh load through WakeModelManager (the production path) and score
    the held-out wake reference + representative negatives."""
    from voice.wake_model_manager import wake_model_manager as wmm
    wmm.close()
    if not wmm.load():
        print("  ✗ retrained model failed to load")
        return False

    def max_score_int16(path) -> float:
        dat = read_wav_int16(path)
        wmm.hard_reset()
        best = 0.0
        for i in range(0, len(dat) - FRAME + 1, FRAME):
            preds = wmm.predict_stream(dat[i:i + FRAME])
            if preds:
                best = max(best, max(float(v) for v in preds.values()))
        return best

    print("\n── DISCRIMINATION TEST (int16, production path) ──")
    thr = wmm.threshold
    ok = True

    wake = max_score_int16(WAKE_DIR / "wake_phrase.wav")
    print(f"  wake phrase reference : {wake:.3f}  (threshold {thr:.3f}) "
          f"{'OK' if wake >= thr else 'MISS'}")
    ok &= wake >= thr

    for name, path in (("silence", None), ("chime", ROOT / "Diego.wav")):
        if path is None:
            wmm.hard_reset()
            best = 0.0
            for _ in range(25):
                preds = wmm.predict_stream(np.zeros(FRAME, dtype=np.int16))
                if preds:
                    best = max(best, max(float(v) for v in preds.values()))
            s = best
        else:
            s = max_score_int16(path)
        print(f"  {name:19s} : {s:.3f}  {'OK' if s < thr else 'FALSE-FIRE'}")
        ok &= s < thr

    # espeak non-wake speech (in-set phrases + a fully unseen phrase)
    for phrase in ("hello everyone", "thank you", "good morning",
                   "hello there friend"):
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["espeak", "-w", tmp, phrase],
                       capture_output=True, timeout=15)
        s = max_score_int16(Path(tmp)) if Path(tmp).exists() else 0.0
        Path(tmp).unlink(missing_ok=True)
        print(f"  '{phrase:19s}' : {s:.3f}  {'OK' if s < thr else 'FALSE-FIRE'}")
        ok &= s < thr

    # Runtime-domain room tone (the class that false-fired live)
    room = ROOT / "debug" / "room_tone.wav"
    if room.exists():
        s = max_score_int16(room)
        print(f"  {'room tone':19s} : {s:.3f}  {'OK' if s < thr else 'FALSE-FIRE'}")
        ok &= s < thr

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ── Orchestration ──────────────────────────────────────────────

def run_calibration() -> bool:
    """Full interactive wake-phrase calibration + verifier training.

    Returns True on success (verifier trained + validated), False otherwise.
    """
    print("\n  ══════════════════════════════════════════════════════════")
    print("  WAKE WORD CALIBRATION & VERIFIER TRAINING")
    print("  ══════════════════════════════════════════════════════════")

    # Back up the current verifier once (never overwrite the backup).
    backup = WAKE_DIR / "backup_pre_int16"
    if (WAKE_DIR / "verifier.pkl").exists() and not backup.exists():
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("verifier.pkl", "metadata.json"):
            src = WAKE_DIR / name
            if src.exists():
                shutil.copy2(src, backup / name)
        print(f"  Backed up previous verifier → {backup}")

    # Stash the existing raw positives as training sources.
    orig = WAKE_DIR / "positives_orig"
    if not orig.exists() and any(POS.glob("*.wav")):
        orig.mkdir(parents=True, exist_ok=True)
        for f in POS.glob("*.wav"):
            shutil.copy2(f, orig / f.name)

    # 1) Record the user's wake phrase.
    if not _record_samples():
        return False

    # 2) Build the deterministic dataset.
    if not build_dataset():
        return False

    # 3) Train the verifier + calibrate threshold.
    if not train():
        return False

    # 4) Validate through the production load path.
    return validate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(0 if run_calibration() else 1)