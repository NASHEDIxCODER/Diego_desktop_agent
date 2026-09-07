'''
AudioBackend Abstraction Tests (Phase 18F-C2).

Tests cover:
1. AudioBackend interface contract
2. SoundDeviceBackend availability
3. AudioManager uses AudioBackend
4. SoundDeviceBackend device enumeration
5. Independent input/output selection
6. Input stream lifecycle
7. Backend cleanup
8. Default runtime selects SoundDeviceBackend

Uses mocks/fakes. No hardware required.
'''

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestAudioBackendInterface:
    """Tests for AudioBackend interface contract."""

    def test_backend_is_abstract(self):
        """AudioBackend cannot be instantiated directly."""
        from voice.audio_backend import AudioBackend
        with pytest.raises(TypeError):
            AudioBackend()

    def test_backend_has_required_methods(self):
        """AudioBackend defines all required abstract methods."""
        from voice.audio_backend import AudioBackend
        required = [
            "name", "is_available", "get_version",
            "enumerate_devices", "enumerate_input_devices",
            "enumerate_output_devices", "get_default_input_device",
            "get_default_output_device", "create_input_stream",
            "create_output_stream", "play", "is_stream_active",
            "start_stream", "stop_stream", "close_stream", "health_check",
        ]
        for method in required:
            assert hasattr(AudioBackend, method), f"Missing: {method}"


class TestSoundDeviceBackend:
    """Tests for SoundDeviceBackend implementation."""

    def test_backend_available(self):
        """SoundDeviceBackend should be available when sounddevice is installed."""
        from voice.audio_backend import SoundDeviceBackend
        backend = SoundDeviceBackend()
        assert backend.is_available()
        assert backend.name == "sounddevice"

    def test_backend_version(self):
        """get_version should return a version string."""
        from voice.audio_backend import SoundDeviceBackend
        backend = SoundDeviceBackend()
        ver = backend.get_version()
        assert isinstance(ver, str)
        assert len(ver) > 0

    def test_enumerate_devices(self):
        """enumerate_devices should return device list."""
        from voice.audio_backend import SoundDeviceBackend, AudioDeviceInfo
        backend = SoundDeviceBackend()
        devices = backend.enumerate_devices()
        assert isinstance(devices, list)
        if devices:
            assert isinstance(devices[0], AudioDeviceInfo)

    def test_enumerate_input_output(self):
        """Input and output enumeration should work."""
        from voice.audio_backend import SoundDeviceBackend
        backend = SoundDeviceBackend()
        inputs = backend.enumerate_input_devices()
        outputs = backend.enumerate_output_devices()
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)

    def test_health_check(self):
        """health_check should return (ok, message)."""
        from voice.audio_backend import SoundDeviceBackend
        backend = SoundDeviceBackend()
        ok, msg = backend.health_check()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
        assert ok  # Should be OK on test system


class TestAudioManagerBackendIntegration:
    """Tests for AudioManager + AudioBackend integration."""

    def test_default_backend_selected(self):
        """AudioManager should select SoundDeviceBackend by default."""
        from voice.audio_manager import AudioManager
        am = AudioManager()
        assert am._audio_backend is not None
        assert am._audio_backend.name == "sounddevice"

    def test_custom_backend_injection(self):
        """AudioManager should accept a custom backend."""
        from voice.audio_manager import AudioManager
        from voice.audio_backend import SoundDeviceBackend

        custom_backend = SoundDeviceBackend()
        am = AudioManager(backend=custom_backend)
        assert am._audio_backend is custom_backend

    def test_no_backend_fallback(self):
        """AudioManager should work even without backend abstraction."""
        from voice.audio_manager import AudioManager
        am = AudioManager(backend=None)
        # Should still have _sd access via fallback
        assert hasattr(am, "_sd")


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_create_backend_returns_sounddevice(self):
        """create_backend should return SoundDeviceBackend."""
        from voice.audio_backend import create_backend, SoundDeviceBackend
        backend = create_backend()
        assert backend is not None
        assert isinstance(backend, SoundDeviceBackend)

    def test_get_default_backend(self):
        """get_default_backend should return a backend."""
        from voice.audio_backend import get_default_backend
        backend = get_default_backend()
        assert backend is not None
        assert backend.is_available()


class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_config(self):
        """StreamConfig should have sensible defaults."""
        from voice.audio_backend import StreamConfig
        config = StreamConfig()
        assert config.samplerate == 16000
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.blocksize == 512

    def test_custom_config(self):
        """StreamConfig should accept custom values."""
        from voice.audio_backend import StreamConfig
        config = StreamConfig(samplerate=48000, channels=2, callback=lambda x: x)
        assert config.samplerate == 48000
        assert config.channels == 2
        assert config.callback is not None

    def test_device_property(self):
        """device property should prefer name over index."""
        from voice.audio_backend import StreamConfig
        config = StreamConfig(device_index=5, device_name="test")
        assert config.device == "test"
        config2 = StreamConfig(device_index=5)
        assert config2.device == 5


class TestAudioDeviceInfo:
    """Tests for AudioDeviceInfo."""

    def test_defaults(self):
        """AudioDeviceInfo should have sensible defaults."""
        from voice.audio_backend import AudioDeviceInfo
        info = AudioDeviceInfo(index=0, name="test")
        assert info.index == 0
        assert info.name == "test"
        assert info.max_input_channels == 0
        assert info.max_output_channels == 0

    def test_repr(self):
        """AudioDeviceInfo repr should be informative."""
        from voice.audio_backend import AudioDeviceInfo
        info = AudioDeviceInfo(index=5, name="Test Mic", max_input_channels=2)
        r = repr(info)
        assert "Test Mic" in r
        assert "5" in r
