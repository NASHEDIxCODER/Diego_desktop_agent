"""
Wake-Word Calibration Utility for Leo.

Records "Hello Leo" 100 times + background negatives, then trains an
openWakeWord custom speaker verifier stored in models/wake/.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_WAKE_DIR = PROJECT_ROOT / "models" / "wake"
POSITIVES_DIR = MODELS_WAKE_DIR / "positives"
NEGATIVES_DIR = MODELS_WAKE_DIR / "negatives"

POSITIVE_COUNT = 100
NEGATIVE_CLIP_COUNT = 20
CLIP_SECONDS = 2.0
MAX_RECORD_WAIT = 8.0
SAMPLE_RATE = 16000


def _ensure_dirs() -> None:
    MODELS_WAKE_DIR.mkdir(parents=True, exist_ok=True)
    POSITIVES_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)


def _write_wav(path, audio, samplerate):
    """Write audio to a mono 16-bit WAV file.

    WAV EXPORT sink: float32 [-1, 1] input is converted to int16 here
    (and only here); int16 input passes through unchanged.
    """
    import wave
    from voice.audio_processing import float32_to_int16
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(float32_to_int16(audio).tobytes())


def _record_clip(audio_manager, timeout=MAX_RECORD_WAIT):
    """Record one speech clip from the ring buffer via VAD."""
    import time
    samples_list = []
    speech_detected = False
    silence_start = 0.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        recent = audio_manager.get_recent_audio(0.1)
        if len(recent) == 0:
            time.sleep(0.02)
            continue
        # float32 [-1, 1] → int16-scale RMS for the threshold comparison.
        rms = float(np.sqrt(np.mean(recent.astype(float) ** 2))) * 32768.0
        if rms > audio_manager.energy_threshold:
            if not speech_detected:
                speech_detected = True
            samples_list.append(recent.copy())
            silence_start = 0.0
        elif speech_detected:
            if silence_start == 0.0:
                silence_start = time.time()
            elif time.time() - silence_start > 0.8:
                break
            samples_list.append(recent.copy())
        time.sleep(0.02)
    if not speech_detected or not samples_list:
        return None
    audio = np.concatenate(samples_list)
    max_samples = int(CLIP_SECONDS * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


def _select_base_model(wake_phrase):
    """Pick the bundled openWakeWord model closest to the wake phrase.

    Returns the full path to the .onnx file (required by train_custom_verifier).
    The stem is stored in metadata for WakeModelManager verification.
    """
    import difflib
    from voice.wake_model_manager import WakeModelManager, DEFAULT_BUNDLED_MODEL
    mm = WakeModelManager()
    models = mm._bundled_models()
    if not models:
        return None
    phrase_lower = wake_phrase.lower().strip()
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
    return models.get(best_name)


def _train_verifier(wake_phrase, base_model_path):
    """Train the openWakeWord custom speaker verifier.

    Feature windows are harvested with the EXACT runtime regime (cold
    buffer reset → 1280-sample int16 streaming frames → window after every
    frame) — the same regime debug/retrain_wake_verifier.py uses and
    validates. Classifier: deterministic logistic regression (C=1.0).
    Threshold: calibrated from runtime-identical clip max-scores.
    """
    import json
    import pickle
    import time

    import scipy.io.wavfile as wavfile
    import numpy as np
    from openwakeword import Model as OWWModel
    from openwakeword.custom_verifier_model import flatten_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    outputs = [str(p) for p in sorted(POSITIVES_DIR.glob("*.wav"))]
    negatives = [str(p) for p in sorted(NEGATIVES_DIR.glob("*.wav"))]

    if not outputs:
        logger.error("No positive samples recorded — cannot train verifier")
        return False
    if not negatives:
        logger.warning("No negative samples recorded — training with positives only")

    base_model_stem = Path(base_model_path).stem if base_model_path else None
    verifier_path = MODELS_WAKE_DIR / "verifier.pkl"
    print(f"\n  Training custom verifier ({len(outputs)} positive, "
          f"{len(negatives)} negative clips)...")
    print(f"  Base model: {base_model_stem}")

    FRAME = 1280  # 80 ms @ 16 kHz — the openWakeWord streaming frame

    def _full_reset(m) -> None:
        pre = m.preprocessor
        pre.raw_data_buffer.clear()
        pre.melspectrogram_buffer = np.ones((76, 32))
        pre.accumulated_samples = 0
        pre.feature_buffer = pre._get_embeddings(
            np.zeros(160000).astype(np.int16))
        m.reset()
        silence = np.zeros(FRAME, dtype=np.int16)
        for _ in range(6):  # re-prime the 5-frame zeroed prediction window
            m.predict(silence)

    def _read_int16(path):
        sr, dat = wavfile.read(path)
        if sr != SAMPLE_RATE:
            from scipy import signal as sc
            import math
            up, down = SAMPLE_RATE, sr
            g = math.gcd(up, down)
            dat = sc.resample_poly(dat.astype(np.float64), up // g, down // g).astype(np.int16)
        if dat.dtype != np.int16:
            dat = np.clip(dat.astype(np.float64), -32768, 32767).astype(np.int16)
        return dat

    try:
        oww = OWWModel(wakeword_model_paths=[str(base_model_path)])
        feats_ndx = oww.model_inputs[base_model_stem]

        def harvest(path):
            dat = _read_int16(path)
            _full_reset(oww)
            windows = []
            for i in range(0, len(dat) - FRAME + 1, FRAME):
                oww.predict(dat[i:i + FRAME])
                windows.append(oww.preprocessor.get_features(feats_ndx)[0].copy())
            return windows

        Xs, ys = [], []
        for p in outputs:
            for w in harvest(p):
                Xs.append(w)
                ys.append(1)
        for n in negatives:
            for w in harvest(n):
                Xs.append(w)
                ys.append(0)
        if not Xs or not any(ys):
            logger.error("No features extracted — cannot train verifier")
            return False
        if not any(not t for t in ys):
            # No negatives recorded: use the room-tone tail of silence as
            # negatives is NOT possible — refuse rather than train a
            # "nonzero = wake" classifier (the original false-wake bug).
            logger.error("No negative samples — cannot train a safe verifier")
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
        with open(verifier_path, "wb") as f:
            pickle.dump(model, f)

        # ── Threshold: runtime-identical clip max-scores ──
        def _clip_max_score(path):
            dat = _read_int16(path)
            _full_reset(oww)
            best = 0.0
            for i in range(0, len(dat) - FRAME + 1, FRAME):
                oww.predict(dat[i:i + FRAME])
                feats = oww.preprocessor.get_features(feats_ndx)
                best = max(best, float(model.predict_proba(feats)[0][-1]))
            return best

        try:
            pos_scores = np.array([_clip_max_score(p) for p in outputs])
            neg_scores = np.array([_clip_max_score(n) for n in negatives])
            pos_min = float(pos_scores.min())
            pos_mean = float(pos_scores.mean())
            neg_max = float(neg_scores.max())
            if pos_min > neg_max:
                detection_threshold = float(neg_max + 0.5 * (pos_min - neg_max))
            else:
                detection_threshold = float(neg_max * 1.15)
                print("  ⚠ WARNING: positive/negative scores overlap "
                      f"(pos_min={pos_min:.3f} <= neg_max={neg_max:.3f}). "
                      "Recording more positives will improve reliability.")
            detection_threshold = max(0.40, min(0.85, detection_threshold))
            print(f"  Verifier max-scores: positives min={pos_min:.3f} mean={pos_mean:.3f}, "
                  f"negatives max={neg_max:.3f}, "
                  f"detection_threshold={detection_threshold:.3f}")
        except Exception as e:
            logger.debug("Threshold computation failed: %s", e)
            detection_threshold = 0.5
    except Exception as e:
        logger.error("Verifier training failed: %s", e, exc_info=True)
        return False

    metadata = {
        "wake_phrase": wake_phrase,
        "base_model": base_model_stem,
        "verifier_path": "verifier.pkl",
        "positive_count": len(outputs),
        "negative_count": len(negatives),
        # 0.0 → verifier fires on every frame so speaker verification is
        # always applied (even when the base model scores the phrase < 0.1).
        "verifier_threshold": 0.0,
        # Detection threshold computed from verifier score distributions.
        "detection_threshold": detection_threshold,
        "domain": "int16-16kHz-mono",
        "trained_at": time.time(),
        "classifier": "logreg(c=1.0, regime-identical-windows)",
    }
    (MODELS_WAKE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  ✓ Verifier saved to: {verifier_path}")
    print(f"  ✓ Metadata saved to: {MODELS_WAKE_DIR / 'metadata.json'}")
    return True


def run_calibration(wake_phrase=None):
    """Run the full wake-word calibration (100 positive + negatives + verifier)."""
    import time
    from voice.audio_manager import audio_manager
    from voice.settings import voice_settings

    voice_settings.update_from_env()
    phrase = wake_phrase or voice_settings.wake_phrase or "hello leo"

    print()
    print("  ═══════════════════════════════════════════════")
    print("  WAKE WORD CALIBRATION")
    print("  ═══════════════════════════════════════════════")
    print(f"  Wake phrase:    '{phrase}'")
    print(f"  Positive clips: {POSITIVE_COUNT}")
    print(f"  Negative clips: {NEGATIVE_CLIP_COUNT}")
    print("  ═══════════════════════════════════════════════")
    print()

    _ensure_dirs()
    for d in (POSITIVES_DIR, NEGATIVES_DIR):
        for f in d.glob("*.wav"):
            try:
                f.unlink()
            except Exception:
                pass

    if not audio_manager.start():
        print("  FAILED: cannot start AudioManager — check microphone")
        return False

    base_model_name = _select_base_model(phrase)
    print(f"  Using base wake model: {base_model_name}")
    print()

    # ── Phase 1: Record 100 wake phrase clips ──────────────
    print("  PHASE 1: Record the wake phrase 100 times")
    print("  Say:", repr(phrase))
    print()

    first_positive = None
    recorded = 0
    last_log = time.time()

    while recorded < POSITIVE_COUNT:
        now = time.time()
        if now - last_log > 2.0:
            print(f"  [{recorded}/{POSITIVE_COUNT}] Say '{phrase}'...")
            last_log = now

        clip = _record_clip(audio_manager)
        if clip is None:
            continue
        # clip is float32 [-1, 1] → int16-scale RMS for the loudness check.
        rms = float(np.sqrt(np.mean(clip.astype(float) ** 2))) * 32768.0
        if rms < 100:
            continue

        recorded += 1
        _write_wav(POSITIVES_DIR / f"positive_{recorded:03d}.wav", clip, SAMPLE_RATE)
        if first_positive is None:
            first_positive = clip

    print(f"  ✓ Recorded {recorded} positive clips")
    if first_positive is not None:
        _write_wav(MODELS_WAKE_DIR / "wake_phrase.wav", first_positive, SAMPLE_RATE)
        print(f"  ✓ Reference phrase saved to: {MODELS_WAKE_DIR / 'wake_phrase.wav'}")

    # ── Phase 2: Record background negatives ───────────────
    print()
    print("  PHASE 2: Record background noise (stay silent)")
    print()

    for i in range(NEGATIVE_CLIP_COUNT):
        time.sleep(0.2)
        bg = audio_manager.get_recent_audio(CLIP_SECONDS)
        if len(bg) < SAMPLE_RATE:
            bg = np.concatenate([bg, np.zeros(SAMPLE_RATE - len(bg), dtype=bg.dtype)])
        _write_wav(NEGATIVES_DIR / f"negative_{i + 1:03d}.wav",
                   bg[: int(CLIP_SECONDS * SAMPLE_RATE)], SAMPLE_RATE)
        if i % 5 == 0 or i == NEGATIVE_CLIP_COUNT - 1:
            print(f"    [{i + 1}/{NEGATIVE_CLIP_COUNT}] background clip saved")

    print(f"  ✓ Recorded {NEGATIVE_CLIP_COUNT} negative clips")

    # ── Phase 3: Train the custom verifier ─────────────────
    audio_manager.stop()
    if not _train_verifier(phrase, base_model_name):
        print("\n  ✗ Verifier training failed")
        return False

    print()
    print("  ═══════════════════════════════════════════════")
    print("  CALIBRATION COMPLETE")
    print("  ═══════════════════════════════════════════════")
    print(f"  Wake phrase:  '{phrase}'")
    print(f"  Base model:   {base_model_name}")
    print(f"  Verifier:     {MODELS_WAKE_DIR / 'verifier.pkl'}")
    print("  The verifier will be attached automatically at startup.")
    print("  ═══════════════════════════════════════════════")
    print()
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.exit(0 if run_calibration() else 1)