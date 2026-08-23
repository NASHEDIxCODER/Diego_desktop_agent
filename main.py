"""
Diego Desktop Assistant — Production Entry Point

  python main.py               # Start Diego (conversational runtime)
  python main.py --no-auth     # Start without face auth (development only)
  python main.py --train       # Train the NLP model and exit
  python main.py --status      # Show NLP model status and exit
  python main.py --benchmark   # Run NLP benchmarks and exit
  python main.py --select-mic  # Interactively select the microphone
  python main.py --audio-debug # Real-time audio level visualizer
  python main.py --train-wake  # Record wake phrases + train custom verifier

THE VOICE PIPELINE (exactly ONE implementation — core/conversation_engine.py):

  Microphone → AudioManager → VAD → openWakeWord → wake-transcript
  verification → face authentication (camera opens ONLY here) →
  streaming Whisper → LLM → tool execution (ONE dispatcher) →
  streaming TTS → continuous conversation → 60 s silence / goodbye →
  back to wake listening. Models load ONCE. Diego NEVER exits on its own.

Heavy subsystems are imported LAZILY inside each command so the runtime
entry path never inherits a broken legacy import.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT-LEVEL FIXES (must happen before any heavy imports)
# ═══════════════════════════════════════════════════════════════

# ── Suppress ALL ALSA/JACK/PulseAudio library noise ──
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_OUTPUT_FORMAT"] = "0"
os.environ["ALSA_DEBUG"] = "0"
os.environ["ALSA_DEBUG_FILE"] = "/dev/null"
os.environ["PYTTXS3_ALSA_DEBUG"] = "0"
os.environ["SPEECH_RECOGNITION_ALSA_DEBUG"] = "0"
os.environ["DISPLAY_ALSA_OUTPUT"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["PULSE_LOG_LEVEL"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"

# ── Qt Font Configuration ──
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

# ── HuggingFace offline-friendly settings ──
# The embedding model is cached locally. Normal startup must NOT contact
# HuggingFace. Only the first download (when model is not cached) may use
# the network. We set offline mode BEFORE any model imports happen.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
_hf_home = os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache/huggingface")
_model_name = "all-MiniLM-L6-v2"
_model_cache_candidates = [
    os.path.join(_hf_home, "hub", f"models--{_model_name.replace('/', '--')}", "snapshots"),
    os.path.join(_hf_home, "hub", f"models--sentence-transformers--{_model_name.replace('/', '--')}", "snapshots"),
]
_model_is_cached = any(
    os.path.isdir(p) and bool(os.listdir(p))
    for p in _model_cache_candidates
)
if _model_is_cached:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("  [HF] Embedding model cached locally — HuggingFace offline mode enabled")
else:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

# ── Disable noisy TensorFlow/Keras warnings ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Ensure DISPLAY exists ──
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

try:
    subprocess.run(["xhost", "+local:"], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
except Exception:
    pass

# ── Python 3.14 compatibility: inject removed stdlib modules ──
import compat  # noqa: F401

# ── The conversational runtime (light module-level import: Diego.py only
# imports compat/config/telemetry at module scope; the engine, models and
# audio hardware are loaded inside run()). Diego.py owns logging setup. ──
import Diego

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Utility commands (each imports its heavy dependencies lazily)
# ═══════════════════════════════════════════════════════════════

def cmd_train(args) -> None:
    """Run NLP training and exit."""
    import asyncio
    from config.settings import settings
    from nlp.trainer import trainer

    print(f"Training NLP with {args.examples} examples per intent...")
    total = asyncio.run(trainer.train(examples_per_intent=args.examples))
    print(f"✓ Training complete! {total} examples generated.")
    print(f"  Model:    {settings.CLASSIFIER_PATH}")
    print(f"  Metadata: {settings.METADATA_PATH}")


def cmd_status() -> None:
    """Show NLP model status."""
    from config.settings import settings
    from nlp.model_metadata import is_model_ready, get_model_status

    if not is_model_ready():
        print("NLP model not found. Run: python main.py --train")
        return
    status = get_model_status()
    print("Diego NLP Model Status")
    print("=" * 40)
    print(f"  Ready:        ✓")
    print(f"  Version:      {status.get('version', '?')}")
    print(f"  Embedding:    {status.get('embedding_model', '?')}")
    print(f"  Intents:      {status.get('intents', 0)}")
    print(f"  Examples:     {status.get('examples', 0)}")
    print(f"  Trained at:   {status.get('trained_at', '?')}")
    print(f"  Threshold:    {status.get('threshold', 0.75)}")
    print(f"  Model path:   {settings.CLASSIFIER_PATH}")
    print(f"  Metadata:     {settings.METADATA_PATH}")


def cmd_benchmark() -> None:
    """Run NLP benchmarks."""
    from nlp.inference import inference

    if not inference.load():
        print("NLP model not found. Run: python main.py --train")
        return
    from nlp.evaluator import evaluator
    from nlp.classifier import BUILTIN_INTENTS
    test_cases = {}
    for intent_name, examples in BUILTIN_INTENTS.items():
        if intent_name == "unknown" or not examples:
            continue
        test_cases[intent_name] = examples[:5]
    print("Running NLP benchmarks...")
    results = evaluator.evaluate_classification(test_cases)
    print(evaluator.report())


def cmd_select_mic() -> None:
    """Interactive microphone selection."""
    from voice.mic_selector import select_microphone_interactive
    select_microphone_interactive()


def cmd_train_wake() -> None:
    """Record wake phrases and train the custom verifier."""
    from voice.calibrate_wake import run_calibration
    ok = run_calibration()
    sys.exit(0 if ok else 1)


def cmd_audio_debug() -> None:
    """Real-time audio level visualizer."""
    import numpy as _np
    from voice.audio_manager import audio_manager
    from voice.audio_processing import audio_preprocessor

    def _bar(value, max_val, width=40):
        pct = min(max(int(value / max_val * width), 0), width)
        return "█" * pct + "░" * (width - pct)

    print("\n  Real-time audio debug (Ctrl+C to stop)\n")

    if not audio_manager.start():
        print("  FAILED: cannot start AudioManager")
        sys.exit(1)

    try:
        while True:
            time.sleep(0.1)
            audio = audio_manager.get_recent_audio(0.1)
            if len(audio) == 0:
                continue

            # Ring-buffer audio is float32 [-1, 1]; show int16-scale levels.
            rms = float(_np.sqrt(_np.mean(audio.astype(float) ** 2))) * 32768.0
            peak = float(_np.max(_np.abs(audio))) * 32768.0
            norm_rms = rms / 32768.0

            preproc = audio_preprocessor.get_metrics()
            noise_floor = preproc.get("noise_floor_raw", 0)
            gain = preproc.get("gain_applied", 1.0)

            # Compute noise floor dB
            nf_db = 20 * _np.log10(noise_floor / 32768.0) if noise_floor > 0 else -120.0

            # VAD state
            vad_state = audio_manager.get_diagnostics().get("vad_state", "?")

            sys.stdout.write("\r")
            sys.stdout.write(
                f" In: {_bar(rms, 32768)} {norm_rms*100:5.1f}% "
                f"| RMS={rms:6.0f} Peak={peak:6.0f} "
                f"| Noise={nf_db:6.1f}dB Gain={gain:.2f}x "
                f"| VAD={vad_state}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  Stopped")
    finally:
        audio_manager.stop()


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Diego Desktop Assistant")
    parser.add_argument("--train", action="store_true",
                        help="Train the NLP model and exit")
    parser.add_argument("--examples", type=int, default=100,
                        help="Examples per intent for training (default: 100)")
    parser.add_argument("--status", action="store_true",
                        help="Show NLP model status")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run NLP benchmarks")
    parser.add_argument("--select-mic", action="store_true",
                        help="Interactively select the best microphone")
    parser.add_argument("--audio-debug", action="store_true",
                        help="Real-time audio level visualizer")
    parser.add_argument("--train-wake", action="store_true",
                        help="Record 100 wake phrases + train custom verifier")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip face authentication (development only)")
    parser.add_argument("--record-session", action="store_true",
                        help="Record every conversation turn to logs/manual_voice_session.json")

    args = parser.parse_args()

    if args.train:
        cmd_train(args)
        sys.exit(0)
    if args.status:
        cmd_status()
        sys.exit(0)
    if args.benchmark:
        cmd_benchmark()
        sys.exit(0)
    if args.select_mic:
        cmd_select_mic()
        sys.exit(0)
    if args.audio_debug:
        cmd_audio_debug()
        sys.exit(0)
    if args.train_wake:
        cmd_train_wake()
        sys.exit(0)

    # ── Default: the conversational runtime (boots once, waits forever
    # for the wake word, never exits unless the user quits). ──
    Diego.run(no_auth=args.no_auth, record_session=args.record_session)


if __name__ == "__main__":
    main()
