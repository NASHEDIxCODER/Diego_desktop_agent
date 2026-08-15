"""
Diagnostic: reproduce the command-VAD audio pipeline with synthetic speech
and measure Silero's output at every stage.

Pipeline under test (mirrors voice/audio_manager.py callback):
  44100 Hz float32 (2ch) -> channel select -> AGC -> resample 16000 -> high-pass
  -> ring buffer -> 512-sample frames -> Silero VAD

Run:
    python debug/diag_vad_pipeline.py
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio_manager import AudioManager, SAMPLE_RATE, FRAME_SAMPLES
from voice.audio_processing import AutomaticGainControl
from voice.vad import unified_vad, VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def metrics(name, a):
    a = np.asarray(a)
    if a.size == 0:
        print(f"  [{name}] EMPTY")
        return
    a64 = a.astype(np.float64)
    rms = float(np.sqrt(np.mean(a64 ** 2)))
    peak = float(np.max(np.abs(a64)))
    finite = bool(np.all(np.isfinite(a64)))
    zero_ratio = float(np.mean(a64 == 0.0))
    print(
        f"  [{name}] dtype={a.dtype} shape={a.shape} "
        f"min={a64.min():.6f} max={a64.max():.6f} "
        f"rms={rms:.6f} (int16={rms*32768:.1f}) peak={peak:.6f} "
        f"zero_ratio={zero_ratio:.4f} finite={finite}"
    )


def make_speech_44k(seconds=2.0, rms_int16=3000.0, sr=44100):
    """Synthetic speech-like signal at 44.1 kHz (harmonic stack + envelope)."""
    t = np.arange(int(seconds * sr)) / sr
    sig = (0.55 * np.sin(2 * np.pi * 140 * t)
           + 0.28 * np.sin(2 * np.pi * 280 * t)
           + 0.17 * np.sin(2 * np.pi * 420 * t))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    sig = sig * env
    sig /= np.sqrt(np.mean(sig ** 2))
    return (sig * (rms_int16 / 32768.0)).astype(np.float32)


def main():
    print("=" * 70)
    print("VAD PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # 1. Load Silero and test with a KNOWN-GOOD 16k speech signal.
    print("\n[1] Silero VAD load + known-good 16k speech check")
    ok = unified_vad.load()
    print(f"  Silero ready={ok} backend={unified_vad.get_diagnostics()['backend']}")

    # Known-good: 16 kHz speech directly (no resample).
    t16 = np.arange(16000) / 16000.0
    speech16 = (0.55 * np.sin(2 * np.pi * 140 * t16)
                + 0.28 * np.sin(2 * np.pi * 280 * t16)
                + 0.17 * np.sin(2 * np.pi * 420 * t16))
    speech16 = speech16 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t16))
    speech16 /= np.sqrt(np.mean(speech16 ** 2))
    speech16 = (speech16 * 0.1).astype(np.float32)  # RMS ~0.1 float
    probs = []
    for i in range(0, len(speech16) - 512 + 1, 512):
        probs.append(unified_vad.speech_prob(speech16[i:i + 512]))
    print(f"  Known-good 16k speech: vad_probs={[round(p,3) for p in probs[:8]]}")

    # 2. Reproduce the full callback pipeline.
    print("\n[2] Full callback pipeline (44100 -> AGC -> resample -> high-pass)")
    am = AudioManager()
    am._running = True
    am._speech_channel = 0
    am._actual_sample_rate = 44100
    am._stream_channels = 2

    speech44 = make_speech_44k(seconds=2.0, rms_int16=3000.0)
    metrics("raw 44k speech (mono)", speech44)

    # Feed in 1411-sample frames (32ms @ 44.1k), 2 channels.
    frame44 = 1411
    n_frames = len(speech44) // frame44
    for i in range(n_frames):
        mono = speech44[i * frame44:(i + 1) * frame44]
        stereo = np.stack([mono, mono * 0.5], axis=1).astype(np.float32)
        am._audio_callback(stereo, frame44, None, None)

    # Read the ring buffer (post high-pass, 16k).
    ring = am._ring_buffer.get_recent(2.0)
    metrics("ring buffer (post AGC+resample+highpass)", ring)

    # 3. Feed ring-buffer frames to Silero.
    print("\n[3] Silero on ring-buffer frames")
    probs = []
    for i in range(0, len(ring) - 512 + 1, 512):
        probs.append(unified_vad.speech_prob(ring[i:i + 512]))
    print(f"  vad_probs={[round(p,3) for p in probs[:12]]}")

    # 4. Isolate each stage to find where corruption happens.
    print("\n[4] Stage isolation")
    agc = AutomaticGainControl()
    mono = speech44[:frame44]
    agc_out, gain = agc.process(mono)
    metrics("AGC output (44k)", agc_out)
    print(f"  AGC gain={gain:.4f}")

    # Resample stage in isolation.
    resampled = am._resample_to_16k(agc_out)
    metrics("resample output (16k)", resampled)
    print(f"  resample in_len={len(agc_out)} out_len={len(resampled)}")

    # High-pass stage in isolation.
    from scipy import signal as scipy_signal
    from voice.audio_processing import HIGH_PASS_CUTOFF, HIGH_PASS_ORDER
    nyquist = SAMPLE_RATE / 2
    sos = scipy_signal.butter(HIGH_PASS_ORDER, HIGH_PASS_CUTOFF / nyquist,
                              btype="highpass", output="sos")
    zi = scipy_signal.sosfilt_zi(sos) * 0
    hp, _ = scipy_signal.sosfilt(sos, resampled.astype(np.float64), zi=zi)
    hp = np.clip(hp.astype(np.float32), -1.0, 1.0)
    metrics("high-pass output (16k)", hp)

    # Silero on the isolated high-pass output.
    if len(hp) >= 512:
        p = unified_vad.speech_prob(hp[:512])
        print(f"  Silero on isolated high-pass frame: {p:.4f}")

    # 5. Check resample filter design.
    print("\n[5] Resample filter design")
    up = SAMPLE_RATE
    down = 44100
    gcd = math.gcd(up, down)
    up //= gcd
    down //= gcd
    print(f"  up={up} down={down} (44100->16000)")
    max_rate = max(up, down)
    num_taps = 2 * 10 * max_rate + 1
    print(f"  filter taps={num_taps} (input frame={frame44} samples)")

    # 6. Test: resample a LONG signal vs SHORT frame to see attenuation.
    print("\n[6] Resample long vs short (attenuation check)")
    long_sig = make_speech_44k(seconds=3.0, rms_int16=3000.0)
    long_resampled = am._resample_to_16k(long_sig)
    metrics("resample LONG (3s) output", long_resampled)
    short_resampled = am._resample_to_16k(mono)
    metrics("resample SHORT (1411) output", short_resampled)

    am._running = False
    print("\nDone.")


if __name__ == "__main__":
    main()