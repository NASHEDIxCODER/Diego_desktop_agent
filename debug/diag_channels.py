"""
Per-channel live RMS diagnostic after mixer calibration.

Run:
    python debug/diag_channels.py

Prints per-channel RMS before and after the startup mixer calibration,
then reports which channel the pipeline selected.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio_manager import AudioManager


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    am = AudioManager()
    if not am.start():
        print("FAILED to start")
        return 1

    print("\nDevice:", am.device_name, "@", am.sample_rate, "Hz")
    print("Probe-selected speech channel:", am.speech_channel)

    # Let the live stream run and print per-channel RMS every 0.5s.
    for _ in range(6):
        time.sleep(0.5)
        chans = {k: round(v * 32768.0, 1) for k, v in am._channel_rms.items()}
        print(f"  live per-channel RMS (int16): {chans} "
              f"selected=ch{am.speech_channel}")

    am.stop()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())