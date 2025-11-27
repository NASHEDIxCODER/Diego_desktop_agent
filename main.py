import datetime
import smtplib
import time
import difflib
import os
import subprocess
from os import close
from pathlib import Path
import speech_recognition as sr
from TTS.api import TTS
from auth import faceauth
from scripts.conversation_llm import chat
from scripts.nlp_controller import parse
from scripts.telegram_bot import send_message, init, read_latest_message, reply_message
import asyncio


# Ensure DISPLAY exists
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

# Allow local connections so Xlib / pyautogui can work
try:
    subprocess.run(["xhost", "+local:"], check=False)
except Exception:
    pass



# from gemini import gemini  # uncomment when ready

# ---------- CONFIG ----------

LANG_CODE = "en-IN"
WAKE_DEVICE_INDEX = None  # or an int like 0/1/2

WAKE_VARIANTS = [
    "angel priya",
    "angel",
    "priya",
    "hey angel",
    "hello priya",
    "hello angel priya",
    "jon assistant",
    "hey angel priya",
    "hey angel",
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
            ["paplay", str(VOICE_FILE)],  # "-q"
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

async def handle_youtube_mode():
    from scripts.youtube import (
        youtube, search_song, skip_ad, pause_or_play, play_next_song,
        play_previous_song, increase_speed, decrease_speed,
        set_playback_speed, seek_forward, seek_backward,
        close_youtube, set_volume, toggle_mute
    )

    speak("Opening YouTube.")
    youtube()
    time.sleep(2)

    speak("Which song do you want to listen?")
    song = takeCommand()
    if not song:
        speak("I didn't get the song name.")
        return

    search_song(song)
    time.sleep(5)
    skip_ad()

    speak("YouTube is ready. Say commands like pause, next, volume, or close YouTube.")

    # ------------------------
    # YOUTUBE MODE LOOP ONLY
    # ------------------------
    while True:
        cmd = takeCommand()
        if not cmd:
            continue
        cmd = cmd.lower().strip()

        if "pause" in cmd or "play" in cmd:
            pause_or_play()
            speak("Playback toggled.")

        elif "next" in cmd:
            play_next_song()
            speak("Next song.")

        elif "previous" in cmd:
            play_previous_song()
            speak("Previous song.")

        elif "skip ad" in cmd or "skip the ad" in cmd or cmd.strip()=="skip":
            skip_ad()
            speak("Ad skipped.")

        elif "faster" in cmd or "increase speed" in cmd:
            increase_speed()
            speak("Speed increased.")

        elif "slower" in cmd or "decrease speed" in cmd:
            decrease_speed()
            speak("Speed decreased.")

        elif "speed" in cmd:
            nums = [float(s) for s in cmd.split() if s.replace(".", "").isdigit()]
            if nums:
                set_playback_speed(nums[0])
                speak(f"Speed set to {nums[0]}.")
            else:
                speak("Tell me a valid speed like 1.25 or 1.5.")

        elif "forward" in cmd:
            seek_forward(10)
            speak("Forward 10 seconds.")

        elif "rewind" in cmd or "backward" in cmd:
            seek_backward(10)
            speak("Backward 10 seconds.")

        elif "volume" in cmd:
            nums = [int(s) for s in cmd.split() if s.isdigit()]
            if nums:
                n = max(0, min(100, nums[0]))
                set_volume(n / 100)
                speak(f"Volume set to {n} percent.")
            else:
                speak("Tell me a number multiple of 10.")

        elif "mute" in cmd or "unmute" in cmd:
            status = toggle_mute()
            speak(status if status else "Unable to toggle mute.")

        # ------------------------
        # EXIT YOUTUBE ONLY HERE
        # ------------------------
        elif "exit youtube" in cmd or "close youtube" in cmd:
            speak("Closing YouTube.")
            close_youtube()
            break

        else:
            speak("I didn't understand. Try again.")

async def handle_telegram_mode(query):
    intent = parse(query)
    action = intent.get("action", "none")

    if action == "send_telegram":
        target = intent.get("target")
        message = intent.get("message")

        if not target:
            speak("Whom should I send the message to?")
            target = takeCommand()

        if not message:
            speak("What should I say?")
            message = takeCommand()

        ok, err = await send_message(target, message)
        speak("Message sent." if ok else (err or "Failed to send message."))
        return

    if action == "read_telegram":
        target = intent.get("target")
        if not target:
            speak("Whose message should I read?")
            target = takeCommand()
        msg = await read_latest_message(target)
        speak(msg or f"No messages from {target}")
        return

    if action == "reply_telegram":
        message = intent.get("message")
        if not message:
            speak("What should I reply?")
            message = takeCommand()

        ok, err = await reply_message(message)
        speak("Reply sent." if ok else (err or "Failed to send reply."))
        return

async def handle_brightness(query):
    from scripts.brightness import set_brightness
    nums = [int(s) for s in query.split() if s.isdigit()]
    if nums:
        value = max(0, min(100, nums[0]))
        set_brightness(value)
        speak(f"Brightness set to {value} percent.")
    else:
        speak("Tell me a number between 1 and 100.")



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

def has_words(text: str, *words):
    text = text.lower()
    return all(w in text for w in words)


# ---------- Main ----------

async def main():
    init_audio_calibration()

    wake = listen_for_wake_word()
    if not fuzzy_match(wake, WAKE_VARIANTS):
        return

        # wishMe()
    userName = faceauth.recognize_faces()
    if not userName:
        faceauth.Unknown_Face()
        return

    speak(f" {userName} mera naam angel priya nahi devil priya hai!")
    # speak(f"Hello {userName}, how may I assist you?")
    await init()  #telegram init
    while True:
        query = takeCommand()
        if not query:
            continue
        query = query.lower().strip()

        #youtube mode
        if "youtube" in query or "play" in query:
            await handle_youtube_mode()
            continue

                # ==================================================
                # -------------- TELEGRAM MODE ----------------------
                # ==================================================
        if "send" in query or "read" in query or "reply" in query:
            await handle_telegram_mode(query)
            continue

                # ==================================================
                # -------------- BRIGHTNESS -------------------------
                # ==================================================
        if "brightness" in query:
            await handle_brightness(query)
            continue

                # ==================================================
                # -------------- GENERAL CONVERSATION ---------------
                # ==================================================
        if "good night" in query or "exit" in query:
            speak("Goodbye, have a nice day.")
            break

                # fallback normal chat
        response = chat(query)
        speak(response)


if __name__ == "__main__":
    asyncio.run(main())


