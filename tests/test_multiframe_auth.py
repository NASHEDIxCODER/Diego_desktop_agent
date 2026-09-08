'''
Regression tests for multi-frame face authentication hardening (Phase 18F-B).

Tests cover:
1. Multiple stable frames are used for authentication.
2. One bad/outlier frame does not automatically cause rejection.
3. One matching frame does not automatically authenticate.
4. Aggregate decision remains bounded by FACE_TOLERANCE policy.
5. Empty/invalid embedding set fails safely.
6. Authentication timeout/frame limit remains bounded.
7. Existing auth bypass semantics remain unchanged.

Tests use synthetic data only - no real biometric data.
'''

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_known_encodings(tmp_path):
    import pickle
    np.random.seed(42)
    enc1 = np.random.randn(128).astype(np.float64)
    enc1 = enc1 / np.linalg.norm(enc1)
    enc2 = np.random.randn(128).astype(np.float64)
    enc2 = enc2 / np.linalg.norm(enc2)
    encodings = [enc1, enc2]
    names = ["alice", "bob"]
    enc_file = tmp_path / "Known_encodings.p"
    with open(enc_file, "wb") as f:
        pickle.dump([encodings, names], f)
    return enc_file, encodings, names


@pytest.fixture
def mock_face_box():
    from auth.face_detector import FaceBox
    return FaceBox(x=100, y=100, w=200, h=200, confidence=0.95, backend="yunet")


@pytest.fixture
def mock_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestMultiFrameAuthentication:

    def test_multiple_frames_used(self, mock_known_encodings, mock_face_box, mock_frame):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(3)]
            with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                mock_dist.return_value = np.array([0.3, 0.8])
                with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                    mock_enc.return_value = [encodings[0]]
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=4)
                    assert result == "alice"
                    assert mock_enc.call_count == 3

    def test_outlier_frame_does_not_cause_rejection(
        self, mock_known_encodings, mock_face_box, mock_frame
    ):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(4)]
            with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                mock_enc.return_value = [encodings[0]]
                with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                    call_count = [0]
                    def side_effect(known, unknown):
                        call_count[0] += 1
                        if call_count[0] == 3:
                            return np.array([0.9, 0.95])
                        return np.array([0.3, 0.8])
                    mock_dist.side_effect = side_effect
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=4)
                    assert result == "alice"

    def test_single_match_does_not_authenticate(
        self, mock_known_encodings, mock_face_box, mock_frame
    ):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(4)]
            with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                mock_enc.return_value = [encodings[0]]
                with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                    call_count = [0]
                    def side_effect(known, unknown):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            return np.array([0.3, 0.8])
                        return np.array([0.8, 0.3])
                    mock_dist.side_effect = side_effect
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=4)
                    assert result == "bob"

    def test_aggregate_bounded_by_tolerance(
        self, mock_known_encodings, mock_face_box, mock_frame
    ):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names), \
             patch("auth.faceauth.FACE_TOLERANCE", 0.6):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(3)]
            with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                mock_enc.return_value = [encodings[0]]
                with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                    mock_dist.return_value = np.array([0.7, 0.8])
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=4)
                    assert result is None

    def test_empty_embeddings_fails_safely(self):
        with patch("auth.faceauth._known_encodings", []), \
             patch("auth.faceauth._known_names", []):
            from auth.faceauth import _aggregate_and_decide
            result = _aggregate_and_decide([], time.time(), max_encode=4)
            assert result is None

    def test_frame_limit_bounded(self, mock_known_encodings, mock_face_box, mock_frame):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(10)]
            with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                mock_enc.return_value = [encodings[0]]
                with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                    mock_dist.return_value = np.array([0.3, 0.8])
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=3)
                    assert mock_enc.call_count == 3
                    assert result == "alice"

    def test_consensus_requires_majority(
        self, mock_known_encodings, mock_face_box, mock_frame
    ):
        enc_file, encodings, names = mock_known_encodings
        with patch("auth.faceauth.ENCODINGS_PATH", enc_file), \
             patch("auth.faceauth._known_encodings", encodings), \
             patch("auth.faceauth._known_names", names):
            from auth.faceauth import _aggregate_and_decide
            pairs = [(mock_frame.copy(), mock_face_box) for _ in range(4)]
            with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
                mock_enc.return_value = [encodings[0]]
                with patch("auth.faceauth.face_recognition.face_distance") as mock_dist:
                    call_count = [0]
                    def side_effect(known, unknown):
                        call_count[0] += 1
                        if call_count[0] <= 2:
                            return np.array([0.3, 0.8])
                        return np.array([0.8, 0.3])
                    mock_dist.side_effect = side_effect
                    result = _aggregate_and_decide(pairs, time.time(), max_encode=4)
                    assert result is None


class TestPreprocessingContract:

    def test_preprocess_returns_tuple(self):
        from auth.face_detector import preprocess_frame
        bright_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result = preprocess_frame(bright_frame)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        dark_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        result = preprocess_frame(dark_frame)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preprocess_preserves_bright_frame(self):
        from auth.face_detector import preprocess_frame
        bright_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        processed, quality = preprocess_frame(bright_frame)
        assert quality.brightness >= 90
        assert processed.shape == bright_frame.shape

    def test_preprocess_enhances_dark_frame(self):
        from auth.face_detector import preprocess_frame
        dark_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        processed, quality = preprocess_frame(dark_frame)
        assert quality.brightness > 50


class TestAuthBypassSemantics:

    def test_set_auth_disabled_clears_provider(self):
        from core.conversation_engine import ConversationEngine
        engine = ConversationEngine()
        engine.set_auth_provider(lambda: "test_user")
        assert engine._auth_provider is not None
        engine.set_auth_disabled()
        assert engine._auth_provider is None
        assert engine._needs_auth() is False

    def test_set_authenticated_none_does_not_bypass(self):
        from core.conversation_engine import ConversationEngine
        engine = ConversationEngine()
        engine.set_auth_provider(lambda: "test_user")
        engine.set_authenticated(None)
        assert engine._auth_provider is not None
        assert engine._needs_auth() is True


class TestEncodeSingleFrame:

    def test_encode_returns_none_on_failure(self):
        from auth.faceauth import _encode_single_frame
        from auth.face_detector import FaceBox
        face = FaceBox(x=100, y=100, w=200, h=200, confidence=0.95, backend="yunet")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Mock face_encodings to return empty list (no face found)
        with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
            mock_enc.return_value = []
            result = _encode_single_frame(frame, face)
            assert result is None

    def test_encode_returns_embedding_on_success(self):
        from auth.faceauth import _encode_single_frame
        from auth.face_detector import FaceBox
        face = FaceBox(x=100, y=100, w=200, h=200, confidence=0.95, backend="yunet")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with patch("auth.faceauth.face_recognition.face_encodings") as mock_enc:
            mock_enc.return_value = [np.random.randn(128).astype(np.float64)]
            result = _encode_single_frame(frame, face)
            assert result is not None
            assert result.shape == (128,)
            assert result.dtype == np.float64
