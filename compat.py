"""
Python 3.14 compatibility module for Diego Desktop Assistant.

Python 3.14 removed aifc, imghdr, and audioop from the standard library.
Several third-party dependencies (speech_recognition, telethon) still
import these modules, causing ImportError at runtime.

This module injects minimal stubs or uses backport packages before those
packages are imported.

Strategy:
  - aifc: Removed in 3.14. speech_recognition imports it for file detection.
          We delegate to the standard wave module via a minimal stub.
  - imghdr: Removed in 3.14. telethon imports it for image type detection.
            We provide a minimal stub.
  - audioop: Removed in 3.14. speech_recognition needs audio operations
            for sample rate conversion and sample width conversion.
            We implement the critical functions (lin2lin, ratecv) properly.
"""

import logging
import struct
import sys
import types as _types
import wave as _wave

logger = logging.getLogger(__name__)

__all__ = [
    "inject_aifc_stub",
    "inject_imghdr_stub",
    "inject_audioop_stub",
    "inject_all",
]


def inject_aifc_stub() -> None:
    """Inject a stub for the removed 'aifc' module, delegating to wave."""
    if "aifc" in sys.modules:
        return
    try:
        import aifc as _real_aifc  # type: ignore[import-untyped]
        return  # Real module exists
    except ImportError:
        pass

    _aifc_stub = _types.ModuleType("aifc")

    class _AifcRead:
        """Minimal aifc reader that delegates to wave."""
        def __init__(self, file):
            self._w = _wave.open(file, 'rb') if isinstance(file, str) else _wave.open(file)

        def getnchannels(self): return self._w.getnchannels()
        def getsampwidth(self): return self._w.getsampwidth()
        def getframerate(self): return self._w.getframerate()
        def getnframes(self): return self._w.getnframes()
        def readframes(self, n): return self._w.readframes(n)
        def close(self): return self._w.close()

    def _open(file, mode=None):
        return _AifcRead(file)

    _aifc_stub.open = _open
    _aifc_stub.AifcRead = _AifcRead
    sys.modules["aifc"] = _aifc_stub
    logger.debug("Injected aifc stub for Python 3.14 compatibility")


def inject_imghdr_stub() -> None:
    """Inject a stub for the removed 'imghdr' module (needed by telethon)."""
    if "imghdr" in sys.modules:
        return
    _imghdr_stub = _types.ModuleType("imghdr")
    _imghdr_stub.what = lambda f, h=None: None
    sys.modules["imghdr"] = _imghdr_stub
    logger.debug("Injected imghdr stub for Python 3.14 compatibility")


def _lin2lin(data: bytes, width: int, new_width: int) -> bytes:
    """
    Convert sample width of audio data.
    
    This is CRITICAL for speech_recognition's Google STT integration.
    speech_recognition calls this to convert audio to 16-bit before sending to Google.
    
    Args:
        data: Raw audio data
        width: Current sample width in bytes (1=8bit, 2=16bit, 4=32bit)
        new_width: Target sample width in bytes
    
    Returns:
        Converted audio data
    """
    if width == new_width:
        return data
    if width == 2 and new_width == 2:
        return data
    if width == 2 and new_width == 1:
        # 16-bit to 8-bit: take high byte of each sample
        count = len(data) // 2
        result = bytearray(count)
        for i in range(count):
            sample = struct.unpack_from('<h', data, i * 2)[0]
            result[i] = (sample >> 8) + 128
        return bytes(result)
    if width == 1 and new_width == 2:
        # 8-bit to 16-bit
        result = bytearray(len(data) * 2)
        for i, b in enumerate(data):
            sample = (b - 128) << 8
            struct.pack_into('<h', result, i * 2, sample)
        return bytes(result)
    if width == 2 and new_width == 4:
        # 16-bit to 32-bit
        result = bytearray(len(data) * 2)
        for i in range(0, len(data), 2):
            sample = struct.unpack_from('<h', data, i)[0]
            struct.pack_into('<i', result, i * 2, sample)
        return bytes(result)
    if width == 4 and new_width == 2:
        # 32-bit to 16-bit
        count = len(data) // 4
        result = bytearray(count * 2)
        for i in range(count):
            sample = struct.unpack_from('<i', data, i * 4)[0]
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample >> 16))
            struct.pack_into('<h', result, i * 2, sample)
        return bytes(result)
    
    logger.warning("audioop.lin2lin: unsupported conversion %d->%d, returning original data", width, new_width)
    return data


def _ratecv(data: bytes, width: int, nchannels: int, inrate: int, outrate: int, state: tuple, weightA: int = 1, weightB: int = 0) -> tuple:
    """
    Convert sample rate of audio data.
    
    This is CRITICAL for speech_recognition's Google STT integration.
    speech_recognition calls this to convert audio to 16kHz before sending to Google.
    
    Uses simple linear interpolation for rate conversion.
    
    Args:
        data: Raw audio data
        width: Sample width in bytes
        nchannels: Number of channels
        inrate: Input sample rate
        outrate: Output sample rate
        state: Previous state (leftover samples, or None)
        weightA: Weight for filter (unused in simple implementation)
        weightB: Weight for filter (unused in simple implementation)
    
    Returns:
        (converted_data, new_state) tuple
    """
    if inrate == outrate:
        return data, state
    
    framesize = width * nchannels
    if framesize == 0:
        return data, state
    
    # Handle state (leftover samples from previous call)
    if state is None:
        leftovers = b''
    else:
        leftovers = state if isinstance(state, bytes) else b''
    
    # Combine leftovers with new data
    all_data = leftovers + data
    
    # Number of complete input frames
    nframes = len(all_data) // framesize
    if nframes == 0:
        # Not enough data for even one frame
        return b'', all_data
    
    # Trim to complete frames
    usable = all_data[:nframes * framesize]
    leftover = all_data[nframes * framesize:]
    
    # Convert using simple linear interpolation
    if width == 2:
        # 16-bit samples
        out_frames = int(nframes * outrate / inrate)
        result = bytearray(out_frames * framesize)
        
        for i in range(out_frames):
            src_pos = i * inrate / outrate
            src_idx = int(src_pos)
            frac = src_pos - src_idx
            
            if src_idx >= nframes - 1:
                src_idx = nframes - 1
                frac = 0.0
            
            for ch in range(nchannels):
                base = (src_idx * nchannels + ch) * 2
                if base + 3 < len(usable):
                    s0 = struct.unpack_from('<h', usable, base)[0]
                    s1 = struct.unpack_from('<h', usable, base + framesize)[0]
                    sample = int(s0 + (s1 - s0) * frac)
                else:
                    sample = struct.unpack_from('<h', usable, base)[0]
                
                out_base = (i * nchannels + ch) * 2
                struct.pack_into('<h', result, out_base, sample)
        
        return bytes(result), leftover
    
    # For other widths, return original data (better than nothing)
    logger.warning("audioop.ratecv: unsupported width=%d, returning original data", width)
    return data, state


def inject_audioop_stub() -> None:
    """
    Inject a stub for the removed 'audioop' module.
    
    Uses the audioop-lts backport if available, falling back to our own
    implementation of the critical functions that speech_recognition needs.
    """
    if "audioop" in sys.modules:
        return

    # Try to use the audioop-lts backport package first
    try:
        import audioop_lts
        sys.modules["audioop"] = audioop_lts
        logger.debug("Using audioop-lts backport for Python 3.14 compatibility")
        return
    except ImportError:
        pass

    # Fallback: our own implementation with proper lin2lin and ratecv
    _audioop_stub = _types.ModuleType("audioop")

    _audioop_stub.lin2lin = _lin2lin
    _audioop_stub.ratecv = _ratecv

    def _audioop_rms(d, w):
        """Compute root-mean-square of audio fragment."""
        if w == 2 and len(d) >= 2:
            n = len(d) // 2
            if n == 0:
                return 0.0
            s = 0
            for i in range(n):
                sample = struct.unpack_from('<h', d, i * 2)[0]
                s += sample * sample
            return int((s / n) ** 0.5)
        return 0

    def _audioop_minmax(d, w):
        if w == 2 and len(d) >= 2:
            s = struct.unpack_from('<h', d, 0)[0]
            return (min(s, 0), max(s, 0))
        return (0, 0)

    _audioop_stub.rms = _audioop_rms
    _audioop_stub.minmax = _audioop_minmax
    _audioop_stub.avg = lambda d, w: 0
    _audioop_stub.avgpp = lambda d, w: 0
    _audioop_stub.max = lambda d, w: 0
    _audioop_stub.cross = lambda d, w: 0
    _audioop_stub.findfactor = lambda a, b: 0.0
    _audioop_stub.findfit = lambda a, b: (0, 0)
    _audioop_stub.findmax = lambda d, l: 0
    _audioop_stub.getsample = lambda d, w, i: 0
    _audioop_stub.tostereo = lambda d, w, lf, rf: d
    _audioop_stub.tomono = lambda d, w, lf, rf: d
    _audioop_stub.add = lambda d1, d2, w: d1
    _audioop_stub.mul = lambda d, w, f: d
    _audioop_stub.reverse = lambda d, w: d
    _audioop_stub.bias = lambda d, w, b: d
    _audioop_stub.adpcm2lin = lambda d, w, s: (d, s)
    _audioop_stub.lin2adpcm = lambda d, w, s: (d, s)
    sys.modules["audioop"] = _audioop_stub
    logger.debug("Injected audioop stub with proper lin2lin and ratecv (audioop-lts not found)")


def inject_all() -> None:
    """Inject all Python 3.14 compatibility stubs."""
    inject_aifc_stub()
    inject_imghdr_stub()
    inject_audioop_stub()


# Always inject on import
inject_all()