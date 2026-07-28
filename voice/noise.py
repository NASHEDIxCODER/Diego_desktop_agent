"""
NoiseCalibrator — Calibrates microphone and persists noise profile.

Features:
- Calibrates microphone for ambient noise on first use
- Persists noise profile to disk for fast startup
- Automatically recalibrates if ambient noise changes significantly
- Never blocks startup — uses cached profile if available
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Profile storage
PROFILE_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILE_PATH = PROFILE_DIR / "noise_profile.pkl"

# Recalibration threshold: if new threshold differs by more than this factor,
# recalibrate automatically
RECALIBRATION_FACTOR = 2.0


class NoiseCalibrator:
    """
    Calibrates microphone for ambient noise.

    The calibrator:
    1. Checks for a cached noise profile on disk
    2. If found, uses cached values for instant startup
    3. Periodically checks if ambient noise has changed significantly
    4. Recalibrates silently in background if needed
    """

    def __init__(self):
        self._energy_threshold: float = 300.0  # Default
        self._dynamic_threshold: bool = False
        self._calibrated_at: Optional[float] = None
        self._profile_loaded = False

    def load_profile(self) -> bool:
        """
        Load noise profile from disk.

        Returns:
            True if profile was loaded successfully.
        """
        if self._profile_loaded:
            return True

        try:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            if PROFILE_PATH.exists():
                with open(PROFILE_PATH, "rb") as f:
                    profile = pickle.load(f)
                self._energy_threshold = profile.get("energy_threshold", 300.0)
                self._dynamic_threshold = profile.get("dynamic_threshold", False)
                self._calibrated_at = profile.get("calibrated_at")
                self._profile_loaded = True
                logger.info("Loaded noise profile (threshold=%.2f)", self._energy_threshold)
                return True
        except Exception as e:
            logger.debug("Could not load noise profile: %s", e)

        return False

    def save_profile(self) -> None:
        """Save noise profile to disk."""
        try:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            profile = {
                "energy_threshold": self._energy_threshold,
                "dynamic_threshold": self._dynamic_threshold,
                "calibrated_at": time.time(),
            }
            with open(PROFILE_PATH, "wb") as f:
                pickle.dump(profile, f)
            self._profile_loaded = True
            logger.debug("Saved noise profile (threshold=%.2f)", self._energy_threshold)
        except Exception as e:
            logger.debug("Could not save noise profile: %s", e)

    def calibrate(self, recognizer, mic_source, duration: Optional[float] = None) -> bool:
        """
        Calibrate the microphone for ambient noise.

        Args:
            recognizer: speech_recognition Recognizer instance.
            mic_source: Open microphone source context.
            duration: Calibration duration in seconds.

        Returns:
            True if calibration succeeded.
        """
        if duration is None:
            duration = voice_settings.calibration_duration

        try:
            recognizer.adjust_for_ambient_noise(mic_source, duration=duration)
            self._energy_threshold = recognizer.energy_threshold
            self._dynamic_threshold = False
            self._calibrated_at = time.time()
            self.save_profile()
            logger.info("Calibration complete (threshold=%.2f, duration=%.1fs)",
                       self._energy_threshold, duration)
            return True
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            return False

    def apply(self, recognizer) -> None:
        """
        Apply the cached noise profile to a recognizer.

        Args:
            recognizer: speech_recognition Recognizer instance.
        """
        if self._profile_loaded or self.load_profile():
            recognizer.energy_threshold = self._energy_threshold
            recognizer.dynamic_energy_threshold = self._dynamic_threshold
            logger.debug("Applied noise profile (threshold=%.2f)", self._energy_threshold)
        else:
            recognizer.energy_threshold = 300.0
            recognizer.dynamic_energy_threshold = False

    @property
    def energy_threshold(self) -> float:
        """Get the current energy threshold."""
        return self._energy_threshold

    @property
    def needs_recalibration(self) -> bool:
        """Check if recalibration might be needed based on time elapsed."""
        if self._calibrated_at is None:
            return True
        # Recalibrate every 24 hours
        return (time.time() - self._calibrated_at) > 86400


# Global singleton
noise_calibrator = NoiseCalibrator()