"""
Phase 19C -- Hinglish / mixed-language command understanding.

Text-only deterministic tests over the real production path:
    transcript -> command_normalizer.normalize -> intent_authorizer.authorize

No microphone, no Whisper, no network. Verifies that mixed Hindi/English
commands reach the SAME intents as their English equivalents, that
entities (song titles, artist names, filenames) are preserved, that
unknown mixed-language utterances do NOT become arbitrary actions, and
that low-confidence behavior and the existing YouTube confirmation flow
remain unchanged.
"""

import compat  # noqa: F401

from nlp.command_normalizer import command_normalizer
from nlp.intent_authorizer import authorize_intent
from agent.task_continuation import classify_confirmation, PendingTaskManager


def norm(t: str) -> str:
    return command_normalizer.normalize(t)


def auth(t: str, conf: float = -0.5, dur: float = 2500.0):
    return authorize_intent(t, stt_confidence=conf, audio_duration_ms=dur)


# ── 1-4. Media / YouTube play (English + Hinglish + Devanagari) ──────

def test_play_music_on_youtube_english():
    n = norm("play music on youtube")
    a = auth(n)
    assert a.category.value == "DETERMINISTIC_COMMAND"
    assert a.actionable is True
    assert "play" in n and "youtube" in n


def test_youtube_pe_music_chalao():
    n = norm("youtube pe music chalao")
    a = auth(n)
    assert a.category.value == "DETERMINISTIC_COMMAND" and a.actionable
    # Canonical English shape so the existing router matches.
    assert n.startswith("play") and "on youtube" in n


def test_youtube_par_believer_play_karo():
    n = norm("youtube par believer play karo")
    a = auth(n)
    assert a.actionable
    assert n.startswith("play") and "believer" in n and "on youtube" in n


def test_play_thodi_der_on_youtube():
    # 'thodi der' is a short-duration phrase -- treated as the SONG TARGET.
    n = norm("play thodi der on youtube")
    a = auth(n)
    assert a.actionable
    assert "thodi der" in n  # entity preserved, not translated away
    assert "on youtube" in n


# ── 5-6. Application open / close ────────────────────────────────

def test_firefox_kholo():
    n = norm("firefox kholo")
    a = auth(n)
    assert a.actionable
    assert n == "open firefox"


def test_chrome_band_karo():
    n = norm("chrome band karo")
    a = auth(n)
    assert a.actionable
    assert n.startswith("close") and "chrome" in n


# ── 7-8. Volume ─────────────────────────────────────────────────

def test_volume_thoda_kam_karo():
    n = norm("volume thoda kam karo")
    a = auth(n)
    assert a.actionable
    assert n == "volume down"


def test_volume_badhao():
    n = norm("volume badhao")
    a = auth(n)
    assert a.actionable
    assert n == "volume up"


# ── 9. Screenshot ──────────────────────────────────────────────

def test_screenshot_le_lo():
    n = norm("screen shot le lo")
    a = auth(n)
    assert a.actionable
    assert "screenshot" in n


# ── 10. System info (Hinglish) ──────────────────────────────────

def test_mera_cpu_kitna_use_ho_raha_hai():
    n = norm("mera cpu kitna use ho raha hai")
    a = auth(n)
    assert a.category.value == "DETERMINISTIC_COMMAND" and a.actionable
    # The English technical noun 'cpu' is preserved.
    assert "cpu" in n


# ── 11. Hindi-script equivalents (Devanagari) ───────────────────

def test_hindi_script_open_firefox():
    # "फ़ायरफ़ॉक्स खोलो" = "firefox kholo"
    n = norm("\u092b\u093c\u093e\u092f\u0930\u092b\u0949\u0915\u094d\u0938 \u0916\u094b\u0932\u094b")
    a = auth(n)
    assert a.actionable
    assert "firefox" in n and n.startswith("open")


def test_hindi_script_youtube_play():
    # "यूट्यूब पर गाना चलाओ" ~ "youtube par gaana chalao"
    n = norm("\u092f\u0942\u091f\u094d\u092f\u0942\u092c \u092a\u0930 \u0917\u093e\u0928\u093e \u091a\u0932\u093e\u0913")
    a = auth(n)
    assert a.actionable
    assert "youtube" in n and ("play" in n or "chalao" in n)


# ── 12. Mixed English entity inside a Hindi command ─────────────

def test_english_entity_in_hindi_command():
    n = norm("youtube pe Arijit Singh ka song chalao")
    a = auth(n)
    assert a.actionable
    # The artist name must survive the normalizer untouched.
    assert "arijit singh" in n.lower()
    assert "on youtube" in n and n.startswith("play")


# ── 13. Song title containing Hindi words ──────────────────────

def test_song_title_with_hindi_words():
    # 'thodi der' is a Hindi phrase used as a song-title-like target.
    n = norm("play thodi der on youtube")
    assert "thodi der" in n  # not invented away


# ── 14. Filename with mixed Hindi/English text ──────────────────

def test_filename_mixed_text_preserved():
    # A filename is content, not a command token. It must survive.
    fn = "mera_report_final.docx"
    n = norm(f"open {fn}")
    assert fn in n  # filename entity preserved


# ── 15. Unknown mixed-language sentence must NOT become an action ─

def test_unknown_mixed_language_no_action():
    n = norm("ज़िंदगी बहुत तमाशा है भाई")
    a = auth(n, conf=-0.5, dur=2500.0)
    # Unknown Hindi sentence with no command verb / no recognized intent must
    # NOT be an actionable command.
    assert a.actionable is False
    assert a.category.value in ("CONVERSATIONAL", "KNOWLEDGE_QUESTION",
                                "UNCERTAIN")


# ── 16. Low-confidence mixed-language follows existing safety ───

def test_low_confidence_mixed_language_safety():
    n = norm("youtube pe believer chalao")
    # Hallucination band (< -1.0) -> actionable command is downgraded.
    a = auth(n, conf=-1.1, dur=2500.0)
    assert a.category.value == "UNCERTAIN"
    assert a.actionable is False


def test_low_confidence_short_audio_mixed_language_safety():
    n = norm("firefox kholo")
    # Short audio + low confidence for an actionable intent -> UNCERTAIN.
    a = auth(n, conf=-0.7, dur=400.0)
    assert a.category.value == "UNCERTAIN"
    assert a.actionable is False


# ── 17. Existing English commands remain unchanged ─────────────

def test_english_open_firefox_unchanged():
    assert norm("open firefox") == "open firefox"
    assert auth(norm("open firefox")).actionable is True


def test_english_play_believer_on_youtube_unchanged():
    n = norm("play believer on youtube")
    assert n == "play believer on youtube"
    assert auth(n).actionable is True


def test_english_conversational_unchanged():
    n = norm("how are you today")
    a = auth(n)
    # Either CONVERSATIONAL (greeting) or KNOWLEDGE_QUESTION (starts with
    # "how"); both are non-actionable. The point is it must NOT execute.
    assert a.actionable is False
    assert a.category.value in ("CONVERSATIONAL", "KNOWLEDGE_QUESTION")


# ── 18. Existing YouTube confirmation behavior unchanged ────────

def test_yes_still_confirmation_only_when_pending():
    # 'yes' is only a confirmation when a pending task exists. Without one,
    # classify_confirmation still reports 'confirm' (the function is
    # context-free), but the Brain only ACTS on it when a pending task is
    # present -- verified in tests/test_autonomous_goal_continuation.py.
    assert classify_confirmation("yes") == "confirm"
    assert classify_confirmation("youtube pe believer chalao") is None


def test_confirmation_security_unchanged():
    # The multilingual layer must NOT accidentally turn a YouTube command
    # into a confirmation word.
    n = norm("youtube pe believer chalao")
    assert classify_confirmation(n) is None
    # And a bare 'yes' still classifies as confirm.
    assert classify_confirmation(norm("yes")) == "confirm"


# ── Idempotency / entity safety ────────────────────────────────

def test_normalization_is_idempotent():
    t = "youtube pe believer chalao"
    once = norm(t)
    twice = norm(once)
    assert once == twice


def test_url_preserved():
    n = norm("open https://example.com/path")
    assert "https://example.com/path" in n


def test_pending_task_manager_unaffected():
    # The multilingual layer must not touch the pending-task singleton.
    mgr = PendingTaskManager()
    mgr.clear()
    assert mgr.get_pending() is None
    mgr.set_pending("youtube pe believer chalao",
                    resume_step={"action": "play_media"}, ttl_s=5)
    assert mgr.get_pending() is not None
    mgr.clear()
