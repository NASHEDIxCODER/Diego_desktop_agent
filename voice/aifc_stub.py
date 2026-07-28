"""
Minimal aifc stub for Python 3.14 compatibility.
Python 3.14 removed the aifc module, but speech_recognition still imports it.
This stub provides the minimal interface needed.
"""
import io
import struct
import wave
from typing import Optional


class AifcRead:
    """Minimal aifc reader that delegates to wave."""
    def __init__(self, file):
        if isinstance(file, str):
            self._wave = wave.open(file, 'rb')
        else:
            self._wave = wave.open(file)
        self._nchannels = self._wave.getnchannels()
        self._sampwidth = self._wave.getsampwidth()
        self._framerate = self._wave.getframerate()
        self._nframes = self._wave.getnframes()

    def getnchannels(self): return self._nchannels
    def getsampwidth(self): return self._sampwidth
    def getframerate(self): return self._framerate
    def getnframes(self): return self._nframes
    def readframes(self, n): return self._wave.readframes(n)
    def close(self): self._wave.close()


def open(file, mode=None):
    """Open an AIFF-C file (delegates to wave for now)."""
    return AifcRead(file)