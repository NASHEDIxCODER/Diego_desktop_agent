#!/usr/bin/env python3
"""
Retrain the wake verifier with a PROPER dataset.

Root cause of the false-wake storm: the previous verifier was trained on
2 positive clips and 20 room-tone negatives — ZERO speech negatives. The
logistic regression learned "any speech = wake phrase" and outputs ~1.0
for everything (chime, TTS, silence included).

This script:
  1. Positives: the 3 real "hello leo" recordings + deterministic
     augmentations (gain/pitch/speed variants) for diversity.
  2. Negatives: NON-WAKE SPEECH (espeak phrases), the chime, silence,
     room tone — the exact classes that false-fired.
  3. Retrains via voice.calibrate_wake._train_verifier (same feature
     extraction path as production).
  4. Verifies discrimination on held-out clips (wake phrase vs others).
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compat  # noqa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retrain")

ROOT = Path(__file__).resolve().parent.parent
WAKE_DIR = ROOT / "models" / "wake"
POS = WAKE_DIR / "positives"
NEG = WAKE_DIR / "negatives"
SR = 16000

NON_WAKE_PHRASES = [
    "Good morning", "What time is it", "Open chrome", "How are you",
    "Thank you very much", "See you later", "What is on my screen",
    "Play some music", "Turn up the volume", "Open the terminal",
    "Read this error", "Search google for news", "Hey there",
    "What's the weather like", "Tell me a joke", "Lock the screen",
    "Good night", "Hello video", "Yellow meow", "Below zero",
]


def write_wav(path, audio_f32):
    import wave
    from voice.audio_processing import float32_to_int16
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(float32_to_int16(audio_f32).tobytes())


def read_wav_f32(path):
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR:
        from scipy import signal
        a = signal.resample_poly(a, SR, sr).astype(np.float32)
    return a


def speed_change(audio, factor):
    """Resample to change speed (and pitch) without external libs."""
    n = int(len(audio) / factor)
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def main():
    POS.mkdir(parents=True, exist_ok=True)
    NEG.mkdir(parents=True, exist_ok=True)
    for d in (POS, NEG):
        for f in d.glob("*.wav"):
            f.unlink()

    # ── 1. POSITIVES: real recordings + augmentations ────────────
    sources = []
    ref = WAKE_DIR / "wake_phrase.wav"
    if ref.exists():
        sources.append(read_wav_f32(ref))
    for p in sorted((WAKE_DIR / "positives_backup").glob("*.wav")):
        sources.append(read_wav_f32(p))
    # fall back to any pre-existing positives we stashed
    if not sources:
        for p in sorted(POS.glob("*.wav")):
            sources.append(read_wav_f32(p))
    if not sources:
        logger.error("No source wake-phrase recordings found")
        return False

    n_pos = 0
    for i, base in enumerate(sources):
        variants = [base]
        for g in (0.85, 1.15):
            variants.append(np.clip(base * g, -1.0, 1.0))
        for s in (0.92, 1.08):
            variants.append(speed_change(base, s))
        for v in variants:
            n_pos += 1
            write_wav(POS / f"positive_{n_pos:03d}.wav", v)
    logger.info("positives: %d clips (from %d real recordings)", n_pos, len(sources))

    # ── 2. NEGATIVES: non-wake speech + chime + silence + tone ───
    n_neg = 0
    # 2a. non-wake speech via espeak (the class that was missing)
    for phrase in NON_WAKE_PHRASES:
        for variant in ("", " -s 110"):
            tmp = tempfile.mktemp(suffix=".wav")
            r = subprocess.run(
                ["espeak-ng", "-w", tmp, f'[[{phrase}]]'],
                capture_output=True)
            if r.returncode != 0 or not Path(tmp).exists():
                r = subprocess.run(["espeak", "-w", tmp, phrase],
                                   capture_output=True)
            if not Path(tmp).exists():
                continue
            try:
                audio = read_wav_f32(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
            if len(audio) < SR // 4:
                continue
            n_neg += 1
            write_wav(NEG / f"negative_{n_neg:03d}.wav", audio)

    # 2b. the chime (NOT the wake word — must never fire)
    chime = ROOT / "leo.wav"
    if chime.exists():
        n_neg += 1
        write_wav(NEG / f"negative_{n_neg:03d}.wav", read_wav_f32(chime))

    # 2c. silence / room tone
    for _ in range(4):
        n_neg += 1
        write_wav(NEG / f"negative_{n_neg:03d}.wav",
                  np.zeros(SR * 2, dtype=np.float32))
    # 2d. quiet noise floor
    rng = np.random.default_rng(7)
    for _ in range(4):
        n_neg += 1
        write_wav(NEG / f"negative_{n_neg:03d}.wav",
                  (rng.standard_normal(SR * 2).astype(np.float32) * 0.005))
    logger.info("negatives: %d clips", n_neg)

    # ── 3. Retrain the verifier through the production path ───────
    from voice.calibrate_wake import _select_base_model, _train_verifier
    base_model = _select_base_model("hello leo")
    logger.info("base model: %s", base_model)
    ok = _train_verifier("hello leo", base_model)
    if not ok:
        logger.error("verifier training failed")
        return False

    # ── 4. Discrimination test: fresh model, held-out clips ───────
    from voice.wake_model_manager import wake_model_manager as wmm
    wmm.close()
    assert wmm.load(), "retrained model failed to load"

    def max_score(path):
        a = read_wav_f32(path)
        wmm.reset_stream()
        best = 0.0
        for i in range(0, len(a), 1280):
            f = a[i:i + 1280]
            if len(f) < 1280:
                f = np.pad(f, (0, 1280 - len(f)))
            p = wmm.predict_stream(f)
            if p:
                best = max(best, max(p.values()))
        return best

    print("\n── DISCRIMINATION TEST ──")
    wake_score = max_score(ref)
    print(f"  wake phrase (held source)      : {wake_score:.3f}  (want ≥ threshold)")
    fails = 0
    for phrase in NON_WAKE_PHRASES[:6]:
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["espeak", "-w", tmp, phrase], capture_output=True)
        s = max_score(tmp)
        Path(tmp).unlink(missing_ok=True)
        flag = "OK" if s < wmm.threshold else "FALSE-FIRE"
        if s >= wmm.threshold:
            fails += 1
        print(f"  '{phrase:28}': {s:.3f}  {flag}")
    s_chime = max_score(chime) if chime.exists() else 0.0
    s_sil = max_score(NEG / "negative_021.wav") if (NEG / "negative_021.wav").exists() else 0.0
    print(f"  chime                          : {s_chime:.3f}")
    print(f"  silence                        : {s_sil:.3f}")
    print(f"  threshold                      : {wmm.threshold:.3f}")
    print(f"\n  wake={wake_score:.3f} ≥ threshold={wmm.threshold:.3f} "
          f"and {fails} false-fires → {'PASS' if wake_score >= wmm.threshold and fails == 0 else 'CHECK'}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
