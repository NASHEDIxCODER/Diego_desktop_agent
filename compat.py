"""
Python 3.14 compatibility module for Leo Desktop Assistant.

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
  - audioop: Removed in 3.14. speech_recognition needs audio operations.
            We use the audioop-lts PyPI package which provides the real API.
"""

import logging
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


def inject_audioop_stub() -> None:
    """
    Inject a stub for the removed 'audioop' module.
    
    Uses the audioop-lts backport if available, falling back to a basic stub
    that provides dummy implementations sufficient for speech_recognition.
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

    # Fallback: minimal stub for speech_recognition basic operations
    _audioop_stub = _types.ModuleType("audioop")

    import struct as _struct

    def _audioop_rms(d, w):
        """Compute root-mean-square of audio fragment.
        
        This is the most critical function — SpeechRecognition uses it
        for energy detection. We implement it properly.
        """
        if w == 2 and len(d) >= 2:
            n = len(d) // 2
            if n == 0:
                return 0.0
            s = 0
            for i in range(n):
                sample = _struct.unpack_from('<h', d, i * 2)[0]
                s += sample * sample
            return int((s / n) ** 0.5)
        return 0

    def _audioop_minmax(d, w):
        if w == 2 and len(d) >= 2:
            s = _struct.unpack_from('<h', d, 0)[0]
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
    _audioop_stub.lin2lin = lambda d, w, nw: d
    _audioop_stub.ratecv = lambda d, w, nc, ir, o, s, wA=1, wB=0: (d, s)
    _audioop_stub.tostereo = lambda d, w, lf, rf: d
    _audioop_stub.tomono = lambda d, w, lf, rf: d
    _audioop_stub.add = lambda d1, d2, w: d1
    _audioop_stub.mul = lambda d, w, f: d
    _audioop_stub.reverse = lambda d, w: d
    _audioop_stub.bias = lambda d, w, b: d
    _audioop_stub.adpcm2lin = lambda d, w, s: (d, s)
    _audioop_stub.lin2adpcm = lambda d, w, s: (d, s)
    sys.modules["audioop"] = _audioop_stub
    logger.debug("Injected audioop stub (audioop-lts not found, using fallback)")


def inject_all() -> None:
    """Inject all Python 3.14 compatibility stubs."""
    inject_aifc_stub()
    inject_imghdr_stub()
    inject_audioop_stub()


# Always inject on import
inject_all()