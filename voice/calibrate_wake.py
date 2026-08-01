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


def _write_wav(path, audio_int16, samplerate):
    """Write int16 PCM audio to a mono 16-bit WAV file."""
    import wave
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(audio_int16.astype(np.int16).tobytes())


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
        rms = float(np.sqrt(np.mean(recent.astype(float) ** 2)))
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

    The bundled openWakeWord base model (e.g. hey_mycroft) does not recognize
    "hello leo", so train_custom_verifier's default threshold=0.5 for
    positives yields ZERO features and training fails. This function extracts
    audio features with threshold=0.0 for ALL clips (positives and
    negatives), then trains the logistic-regression verifier directly.
    """
    import json
    import pickle
    import time

    import scipy.io.wavfile as wavfile
    import numpy as np
    from openwakeword import Model as OWWModel
    from openwakeword.custom_verifier_model import (
        get_reference_clip_features,
        train_verifier_model,
    )

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

    try:
        oww = OWWModel(wakeword_model_paths=[str(base_model_path)])

        # Extract features with threshold=0.0 for ALL clips so that even
        # when the base model scores "hello leo" < 0.5, we still get features.
        print("  Extracting positive features (threshold=0.0)...")
        pos_feats = []
        for p in outputs:
            sr, dat = wavfile.read(p)
            if sr != SAMPLE_RATE:
                from scipy import signal as sc
                import math
                up, down = SAMPLE_RATE, sr
                g = math.gcd(up, down)
                dat = sc.resample_poly(dat.astype(np.float64), up // g, down // g).astype(np.int16)
            feats = get_reference_clip_features(
                dat, oww, base_model_stem, threshold=0.0, N=3
            )
            if feats.shape[0] > 0 and feats.shape[1] > 0:
                pos_feats.append(feats)
        if not pos_feats:
            logger.error("No positive features extracted — cannot train verifier")
            return False
        positive_features = np.vstack(pos_feats)

        print("  Extracting negative features (threshold=0.0)...")
        neg_feats = []
        for n in negatives:
            sr, dat = wavfile.read(n)
            if sr != SAMPLE_RATE:
                from scipy import signal as sc
                import math
                up, down = SAMPLE_RATE, sr
                g = math.gcd(up, down)
                dat = sc.resample_poly(dat.astype(np.float64), up // g, down // g).astype(np.int16)
            feats = get_reference_clip_features(
                dat, oww, base_model_stem, threshold=0.0, N=1
            )
            if feats.shape[0] > 0 and feats.shape[1] > 0:
                neg_feats.append(feats)
        if neg_feats:
            negative_features = np.vstack(neg_feats)
        else:
            # Fall back to silence negatives if no negative features were extracted
            negative_features = np.zeros(
                (positive_features.shape[0] // 2 or 10,
                 positive_features.shape[1],
                 positive_features.shape[2]),
                dtype=positive_features.dtype,
            )

        print(f"  Positive features: {positive_features.shape}")
        print(f"  Negative features: {negative_features.shape}")

        lr_model = train_verifier_model(
            np.vstack((positive_features, negative_features)),
            np.array([1] * positive_features.shape[0]
                     + [0] * negative_features.shape[0]),
        )

        with open(verifier_path, "wb") as f:
            pickle.dump(lr_model, f)

        # ── Compute optimal detection threshold (inference-consistent) ──
        # Score each FULL clip via streaming predict (max score per clip),
        # exactly as inference does at runtime. This avoids the mismatch
        # between raw feature-window scores and real-time max scores.
        def _clip_max_score(path, oww_model, model_key):
            import scipy.io.wavfile as _wav
            sr, dat = _wav.read(path)
            if sr != SAMPLE_RATE:
                from scipy import signal as sc
                import math
                up, down = SAMPLE_RATE, sr
                g = math.gcd(up, down)
                dat = sc.resample_poly(dat.astype(np.float64), up // g, down // g).astype(np.int16)
            audio_float = dat.astype(np.float32) / 32768.0
            remainder = len(audio_float) % 1280
            if remainder:
                audio_float = np.pad(audio_float, (0, 1280 - remainder))
            best = 0.0
            for i in range(0, len(audio_float), 1280):
                preds = oww_model.predict(audio_float[i:i + 1280])
                best = max(best, float(preds.get(model_key, 0.0)))
            return best

        try:
            # Attach the freshly trained verifier so scoring matches runtime.
            oww_scoring = OWWModel(
                wakeword_model_paths=[str(base_model_path)],
                custom_verifier_models={base_model_stem: str(verifier_path)},
                custom_verifier_threshold=0.0,
            )
            pos_scores = [_clip_max_score(p, oww_scoring, base_model_stem) for p in outputs]
            neg_scores = [_clip_max_score(n, oww_scoring, base_model_stem) for n in negatives]
            pos_min = float(np.min(pos_scores))
            neg_max = float(np.max(neg_scores))
            pos_mean = float(np.mean(pos_scores))
            # Wake-word thresholds favor PRECISION (fewer false positives)
            # over recall: bias the cut-off toward the positive distribution
            # (65% of the way from the negative max to the positive min) so
            # ambient/non-wake audio does not spuriously trigger the assistant.
            if pos_min > neg_max:
                detection_threshold = float(neg_max + 0.65 * (pos_min - neg_max))
            else:
                # Distributions overlap (weak/mismatched data). Use a small
                # margin above the negative max so detection still works, and
                # warn that more/better training data would improve robustness.
                detection_threshold = float(neg_max * 1.15)
                print("  ⚠ WARNING: positive/negative scores overlap "
                      f"(pos_min={pos_min:.3f} <= neg_max={neg_max:.3f}). "
                      "Recording more positives will improve reliability.")
            detection_threshold = max(0.05, min(0.90, detection_threshold))
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
        "trained_at": time.time(),
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
        rms = float(np.sqrt(np.mean(clip.astype(float) ** 2)))
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
            bg = np.concatenate([bg, np.zeros(SAMPLE_RATE - len(bg), dtype=np.int16)])
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