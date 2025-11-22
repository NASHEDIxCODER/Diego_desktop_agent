import datetime
import smtplib
import time
import difflib
import os
import subprocess
from pathlib import Path

import speech_recognition as sr
from TTS.api import TTS

from auth import faceauth
# from gemini import gemini  # uncomment when ready


# ---------- CONFIG ----------

LANG_CODE = "en-IN"
WAKE_DEVICE_INDEX = None   # or an int like 0/1/2

WAKE_VARIANTS = [
    "leo",
    "hey leo",
    "hello leo",
    "ok leo",
    "leo assistant",
    "lio",
    "rio",
]

BASE_DIR = Path(__file__).resolve().parent
VOICE_FILE = BASE_DIR / "leo.wav"


# ---------- TTS (Friday-style) ----------

def _init_tts():
    try:
        print("[TTS] Initializing Friday voice model...")
        return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    except Exception as e:
        print(f"[TTS] Failed to initialize TTS: {e}")
        return None


tts = _init_tts()


def speak(text: str):
    if not text:
        return

    if tts is None:
        print(f"[SPEAK] {text}")
        return

    try:
        tts.tts_to_file(text=text, file_path=str(VOICE_FILE))
        subprocess.run(
            ["aplay", "-q", str(VOICE_FILE)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")
        print(f"[SPEAK-FALLBACK] {text}")


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning, sir.")
    elif 12 <= hour < 18:
        speak("Good afternoon, sir.")
    else:
        speak("Good evening, sir.")
    speak("I am Leo, your assistant. How may I help you?")


# ---------- Speech & Wake ----------

def get_microphone():
    if WAKE_DEVICE_INDEX is None:
        return sr.Microphone()
    return sr.Microphone(device_index=WAKE_DEVICE_INDEX)


# global recognizer + mic
RECOGNIZER = sr.Recognizer()
MIC = get_microphone()


def init_audio_calibration():
    """
    Do a single, solid ambient noise calibration at startup
    and then fix the energy threshold (no constant re-tuning).
    """
    try:
        with MIC as source:
            print("[AUDIO] Calibrating for ambient noise...")
            RECOGNIZER.dynamic_energy_threshold = True
            RECOGNIZER.adjust_for_ambient_noise(source, duration=1.5)
            print(f"[AUDIO] Initial energy threshold: {RECOGNIZER.energy_threshold}")

        # lock the threshold in place for stability
        RECOGNIZER.dynamic_energy_threshold = False
        RECOGNIZER.energy_threshold *= 1.2  # slightly more tolerant
        print(f"[AUDIO] Fixed energy threshold: {RECOGNIZER.energy_threshold}")
    except Exception as e:
        print(f"[AUDIO] Calibration failed: {e}")
        speak("I could not calibrate the microphone properly.")


def fuzzy_match(text: str, variants, cutoff: float = 0.7) -> bool:
    text = text.lower().strip()
    for v in variants:
        v = v.lower().strip()
        if v in text:
            return True
        ratio = difflib.SequenceMatcher(None, text, v).ratio()
        if ratio >= cutoff:
            return True
    return False


def listen_for_wake_word():
    """
    Uses global RECOGNIZER + MIC.
    Doesn’t recalibrate each time, just listens in short chunks.
    Logs everything it hears so you can see what Google is actually returning.
    """
    while True:
        try:
            with MIC as source:
                print("Listening for wake word...")
                # no adjust_for_ambient_noise here – already calibrated
                audio = RECOGNIZER.listen(source, phrase_time_limit=3)
        except Exception as e:
            print(f"[ERROR] Microphone error while listening: {e}")
            speak("I cannot access the microphone right now.")
            time.sleep(1)
            continue

        try:
            print("Recognizing wake word...")
            text = RECOGNIZER.recognize_google(audio, language=LANG_CODE)
            norm = text.lower().strip()
            print(f"[WAKE] Heard: {norm!r}")

            if fuzzy_match(norm, WAKE_VARIANTS):
                print("[WAKE] Wake word detected")
                return norm

        except sr.UnknownValueError:
            print("[WAKE] Could not understand audio.")
            continue
        except sr.RequestError as e:
            print(f"[WAKE] Recognition service error: {e}")
            time.sleep(1)
            continue


def takeCommand():
    """
    Uses the same global recognizer & mic.
    Slightly longer phrase_time_limit, no re-calibration, just listen & decode.
    """
    try:
        with MIC as source:
            print("Listening for command...")
            audio = RECOGNIZER.listen(source, phrase_time_limit=7)
    except Exception as e:
        print(f"[ERROR] Microphone error in takeCommand: {e}")
        speak("I cannot access the microphone right now.")
        return None

    try:
        print("Recognizing command...")
        query = RECOGNIZER.recognize_google(audio, language=LANG_CODE)
        query = query.strip()
        print(f"[CMD] User said: {query}")
        return query
    except sr.UnknownValueError:
        print("[CMD] Could not understand audio.")
        speak("Say that again, please.")
        return None
    except sr.RequestError as e:
        print(f"[CMD] Speech recognition service error: {e}")
        speak("Network error with speech service.")
        return None


# ---------- Email ----------

def sendEmail(to, content):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login("sonu.samrat7668@gmail.com", "Sk@Samrat1")
            server.sendmail("youremail@gmail.com", to, content)
    except Exception as e:
        print(f"[EMAIL] Failed to send email: {e}")


def has_words(text: str, *words):
    text = text.lower()
    return all(w in text for w in words)


# ---------- Main ----------

if __name__ == "__main__":
    init_audio_calibration()

    wake = listen_for_wake_word()
    if fuzzy_match(wake, WAKE_VARIANTS):
        # wishMe()
        userName = faceauth.recognize_faces()
        if userName:
            speak(f"Hello {userName}, how may I assist you?")
            while True:
                query = takeCommand()
                if query is None:
                    continue
                query = query.lower()

                if has_words(query, "open", "gemini"):
                    speak("Opening Gemini. What is your query?")
                    while True:
                        question = takeCommand()
                        if question is None:
                            continue
                        print("Gemini question:", question)
                        # answer = gemini(question)
                        # speak(f"{answer} What is your next query?")
                        if "exit" in question.lower():
                            speak("Closing Gemini.")
                            break

                elif "youtube" in query:
                    from scripts.youtube import (
                        youtube,
                        search_song,
                        skip_ad,
                        stop_song,
                        play_next_song,
                        play_back_speed_i,
                        play_back_speed_d,
                    )

                    speak("Opening YouTube.")
                    youtube()
                    time.sleep(3)
                    speak("Which song do you want to listen?")
                    song = takeCommand()
                    if song is None:
                        speak("I didn't understand.")
                        continue
                    song = song.lower()

                    try:
                        search_song(song)
                        time.sleep(7)
                        skip_ad()
                        yt = listen_for_wake_word()
                    except Exception as e:
                        print("YouTube error:", e)
                        continue

                    if fuzzy_match(yt, WAKE_VARIANTS):
                        stop_song()
                        speak("How may I help you?")
                        q = takeCommand()
                        if q is None:
                            continue
                        q = q.lower()

                        if has_words(q, "next", "song"):
                            play_next_song()
                        elif has_words(q, "search", "song"):
                            speak("Which song?")
                            song = takeCommand()
                            if song is None:
                                speak("I didn't understand.")
                                break
                            search_song(song.lower())
                        elif "pause" in q:
                            stop_song()
                        elif has_words(q, "increase", "playback", "speed"):
                            play_back_speed_i()
                        elif has_words(q, "decrease", "playback", "speed"):
                            play_back_speed_d()
                        elif "exit" in q:
                            break

                elif "exit" in query:
                    speak("Okay, have a good day.")
                    break
        else:
            faceauth.Unknown_Face()
