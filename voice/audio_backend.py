"""
AudioBackend — Minimal abstraction over audio capture/playback backends.

This module establishes a clean contract for audio operations without
speculating about future backends. The only concrete implementation
is SoundDeviceBackend, wrapping the existing sounddevice-based pipeline.

Design principles:
- Expose only what the current runtime actually needs.
- Do NOT expose sounddevice-specific objects through the abstraction.
- Preserve all existing device-selection, retry, and recovery behavior.
- Allow dependency injection for testing.

Public API:
    AudioBackend (abstract interface)
    SoundDeviceBackend (concrete implementation)
    create_backend() — factory that returns the best available backend
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Audio device info (backend-agnostic) ──────────────────────────────────


class AudioDeviceInfo:
    """Backend-agnostic device information."""

    def __init__(
        self,
        index: int,
        name: str,
        max_input_channels: int = 0,
        max_output_channels: int = 0,
        default_samplerate: float = 44100.0,
        is_default: bool = False,
        is_virtual: bool = False,
        hostapi: str = "",
    ):
        self.index = index
        self.name = name
        self.max_input_channels = max_input_channels
        self.max_output_channels = max_output_channels
        self.default_samplerate = default_samplerate
        self.is_default = is_default
        self.is_virtual = is_virtual
        self.hostapi = hostapi

    def __repr__(self) -> str:
        return (
            f"AudioDeviceInfo(index={self.index}, name={self.name!r}, "
            f"in={self.max_input_channels}, out={self.max_output_channels}, "
            f"sr={self.default_samplerate:.0f})"
        )


# ── Stream configuration ──────────────────────────────────────────────────


class StreamConfig:
    """Configuration for opening an audio stream."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
        samplerate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        blocksize: int = 512,
        callback=None,
    ):
        self.device_index = device_index
        self.device_name = device_name
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback

    @property
    def device(self):
        """Return device identifier (name takes precedence for capture-by-name)."""
        if self.device_name is not None:
            return self.device_name
        return self.device_index


# ── Audio stream handle ───────────────────────────────────────────────────


class AudioStream:
    """
    Wrapper around a backend-specific stream object.

    This hides the concrete stream type (e.g. sounddevice.InputStream)
    from callers while still allowing access to the underlying stream
    via the `_raw` attribute when absolutely necessary.
    """

    def __init__(self, raw_stream, backend: "AudioBackend"):
        self._raw = raw_stream
        self._backend = backend

    @property
    def active(self) -> bool:
        """Check if the stream is currently running."""
        return self._backend.is_stream_active(self._raw)

    def start(self) -> None:
        """Start the stream."""
        self._backend.start_stream(self._raw)

    def stop(self) -> None:
        """Stop the stream."""
        self._backend.stop_stream(self._raw)

    def close(self) -> None:
        """Close the stream and release resources."""
        self._backend.close_stream(self._raw)


# ── AudioBackend interface ────────────────────────────────────────────────


class AudioBackend(ABC):
    """
    Minimal audio backend interface.

    All audio operations in Diego should go through this interface.
    Concrete implementations wrap specific libraries (sounddevice, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string (e.g. 'sounddevice')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is installed and usable."""
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Return backend version string."""
        ...

    # ── Device enumeration ─────────────────────────────────────────

    @abstractmethod
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Return all audio devices."""
        ...

    @abstractmethod
    def enumerate_input_devices(self) -> List[AudioDeviceInfo]:
        """Return capture-capable devices."""
        ...

    @abstractmethod
    def enumerate_output_devices(self) -> List[AudioDeviceInfo]:
        """Return playback-capable devices."""
        ...

    @abstractmethod
    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        """Return the default input device, if any."""
        ...

    @abstractmethod
    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        """Return the default output device, if any."""
        ...

    # ── Input streams ──────────────────────────────────────────────

    @abstractmethod
    def create_input_stream(self, config: StreamConfig) -> AudioStream:
        """Create and return a capture stream."""
        ...

    # ── Output streams ─────────────────────────────────────────────

    @abstractmethod
    def create_output_stream(self, config: StreamConfig) -> AudioStream:
        """Create and return a playback stream."""
        ...

    @abstractmethod
    def play(self, audio: np.ndarray, samplerate: int,
             device_index: Optional[int] = None) -> None:
        """Play audio data synchronously (blocking until done)."""
        ...

    # ── Stream lifecycle (used by AudioStream wrapper) ─────────────

    @abstractmethod
    def is_stream_active(self, raw_stream) -> bool:
        """Check if a stream is currently running."""
        ...

    @abstractmethod
    def start_stream(self, raw_stream) -> None:
        """Start a stream."""
        ...

    @abstractmethod
    def stop_stream(self, raw_stream) -> None:
        """Stop a stream."""
        ...

    @abstractmethod
    def close_stream(self, raw_stream) -> None:
        """Close a stream and release resources."""
        ...

    # ── Health ─────────────────────────────────────────────────────

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """
        Verify backend is functional.

        Returns (ok, message).
        """
        ...

# ── SoundDeviceBackend ────────────────────────────────────────────────────


class SoundDeviceBackend(AudioBackend):
    """
    Concrete AudioBackend implementation wrapping sounddevice.

    This is a thin wrapper that translates between the AudioBackend
    interface and sounddevice calls. All existing sounddevice behavior
    (device probing, stream creation, error handling) is preserved.
    """

    def __init__(self):
        self._sd = None
        self._load_sounddevice()

    def _load_sounddevice(self) -> None:
        """Import and cache the sounddevice module."""
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            self._sd = None

    @property
    def name(self) -> str:
        return "sounddevice"

    def is_available(self) -> bool:
        return self._sd is not None

    def get_version(self) -> str:
        if self._sd is None:
            return "not installed"
        return str(getattr(self._sd, "__version__", "?"))

    # ── Device enumeration ─────────────────────────────────────────

    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        if self._sd is None:
            return []
        devices = []
        try:
            for i, d in enumerate(self._sd.query_devices()):
                devices.append(AudioDeviceInfo(
                    index=i,
                    name=d.get("name", f"Device {i}"),
                    max_input_channels=int(d.get("max_input_channels", 0)),
                    max_output_channels=int(d.get("max_output_channels", 0)),
                    default_samplerate=float(d.get("default_low_samplerate",
                                                   d.get("default_samplerate", 44100))),
                    is_default=(i == self._sd.default.device[0] or
                                i == self._sd.default.device[1]),
                    is_virtual=self._is_virtual_name(d.get("name", "")),
                    hostapi=self._get_hostapi_name(i),
                ))
        except Exception:
            pass
        return devices

    def enumerate_input_devices(self) -> List[AudioDeviceInfo]:
        return [d for d in self.enumerate_devices() if d.max_input_channels > 0]

    def enumerate_output_devices(self) -> List[AudioDeviceInfo]:
        return [d for d in self.enumerate_devices() if d.max_output_channels > 0]

    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        if self._sd is None:
            return None
        try:
            idx = self._sd.default.device[0]
            for d in self.enumerate_devices():
                if d.index == idx:
                    return d
        except Exception:
            pass
        return None

    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        if self._sd is None:
            return None
        try:
            idx = self._sd.default.device[1]
            for d in self.enumerate_devices():
                if d.index == idx:
                    return d
        except Exception:
            pass
        return None

    # ── Input streams ──────────────────────────────────────────────

    def create_input_stream(self, config: StreamConfig) -> AudioStream:
        if self._sd is None:
            raise RuntimeError("sounddevice not available")
        raw = self._sd.InputStream(
            samplerate=config.samplerate,
            device=config.device,
            channels=config.channels,
            dtype=config.dtype,
            callback=config.callback,
            blocksize=config.blocksize,
        )
        return AudioStream(raw, self)

    # ── Output streams ─────────────────────────────────────────────

    def create_output_stream(self, config: StreamConfig) -> AudioStream:
        if self._sd is None:
            raise RuntimeError("sounddevice not available")
        raw = self._sd.OutputStream(
            samplerate=config.samplerate,
            device=config.device,
            channels=config.channels,
            dtype=config.dtype,
            blocksize=config.blocksize,
        )
        return AudioStream(raw, self)

    def play(self, audio: np.ndarray, samplerate: int,
             device_index: Optional[int] = None) -> None:
        if self._sd is None:
            raise RuntimeError("sounddevice not available")
        self._sd.play(audio, samplerate, device=device_index)
        self._sd.wait()

    # ── Stream lifecycle ───────────────────────────────────────────

    def is_stream_active(self, raw_stream) -> bool:
        if self._sd is None:
            return False
        return bool(getattr(raw_stream, "active", False))

    def start_stream(self, raw_stream) -> None:
        raw_stream.start()

    def stop_stream(self, raw_stream) -> None:
        raw_stream.stop()

    def close_stream(self, raw_stream) -> None:
        raw_stream.close()

    # ── Health ─────────────────────────────────────────────────────

    def health_check(self) -> Tuple[bool, str]:
        if self._sd is None:
            return False, "sounddevice not installed"
        try:
            devices = self._sd.query_devices()
            inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
            if not inputs:
                return False, "no input devices found"
            return True, f"{len(inputs)} input(s), {len(devices)} total"
        except Exception as e:
            return False, f"device query failed: {e}"

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_virtual_name(name: str) -> bool:
        """Check if a device name looks virtual."""
        virtual = {"pipewire", "pulse", "default", "dummy", "null",
                   "monitor", "loopback", "echo", "virtual"}
        return name.strip().lower() in virtual

    def _get_hostapi_name(self, device_index: int) -> str:
        """Get host API name for a device index."""
        try:
            hostapis = self._sd.query_hostapis()
            for api in hostapis:
                if device_index in api.get("devices", []):
                    return api.get("name", "")
        except Exception:
            pass
        return ""


# ── Factory ───────────────────────────────────────────────────────────────


def create_backend(preferred: Optional[str] = None) -> Optional[AudioBackend]:
    """
    Create and return the best available audio backend.

    Args:
        preferred: Optional backend name to try first (e.g. 'sounddevice').

    Returns:
        An AudioBackend instance, or None if no backend is available.
    """
    # Currently only sounddevice is supported
    backend = SoundDeviceBackend()
    if backend.is_available():
        return backend
    return None


# ── Global default backend (lazy-loaded) ────────────────────────────────--


_default_backend: Optional[AudioBackend] = None


def get_default_backend() -> Optional[AudioBackend]:
    """Return the global default backend, creating it if necessary."""
    global _default_backend
    if _default_backend is None:
        _default_backend = create_backend()
    return _default_backend
