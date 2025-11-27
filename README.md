# 🦁 Leo Desktop Assistant

Leo is a voice-controlled desktop assistant for Linux with a “Friday-style” personality, wake-word activation, face authentication, Gemini-powered conversation, YouTube automation, Telegram control, and system utilities like brightness control.

> Talk to your machine, let it talk back, send Telegrams, control YouTube, and tweak your desktop — all hands-free.

---
### Created by: nashedi_x_coder
#### updated by: dem0000n

---
## ✨ Features

- 🎙 **Always-listening wake word**
  - Listens for phrases like: `hello leo`, `lio`, `hey leo`, `leo`.
  - Uses fuzzy matching so small mispronunciations / ASR errors still trigger Leo.

- 🧑‍💻 **Face authentication**
  - Uses `auth.faceauth` before giving full access.
  - If the face is unknown, `Unknown_Face()` is triggered and the session is aborted.
  - Great for a “personal AI” that only answers its real owner.

- 🧠 **Conversational AI (Gemini)**
  - High-level chat is delegated to a Gemini-powered backend via `scripts.conversation_llm.chat`.
  - Natural language questions, casual chat, and general queries are handled by the LLM.
  - Local logic decides when to route to Gemini vs. a local action (Telegram / YouTube / brightness).

- 📲 **Telegram automation**
  - Implemented in `scripts.telegram_bot` (async).
  - Current flows:
    - **Send message** – “send telegram to Alice saying I’ll be late”
    - **Read latest** – “read telegram from Bob”
    - **Reply** – “reply on telegram” after a conversation
  - All logic is interpreted via `scripts.nlp_controller.parse` to extract:
    - `action` → `send_telegram`, `read_telegram`, `reply_telegram`
    - `target` → contact / user
    - `message` → message body

- 📺 **YouTube hands-free mode**
  - Triggered by phrases including “youtube” or “play”.
  - Uses helpers in `scripts.youtube`:
    - `youtube()` – open YouTube
    - `search_song()` – search and start a song
    - `skip_ad()`, `pause_or_play()`, `play_next_song()`, `play_previous_song()`
    - `increase_speed()`, `decrease_speed()`, `set_playback_speed()`
    - `seek_forward()`, `seek_backward()`
    - `set_volume()`, `toggle_mute()`
    - `close_youtube()`
  - Voice commands supported (examples):
    - “pause”, “play”
    - “next song”, “previous song”
    - “skip ad”, “skip”
    - “faster”, “slower”, “set speed to 1 point 5”
    - “forward 10 seconds”, “rewind 10 seconds”
    - “volume 50”
    - “mute”, “unmute”
    - “exit youtube”, “close youtube”

- 💡 **Brightness control**
  - Natural language brightness control via `scripts.brightness.set_brightness`.
  - Example commands:
    - “set brightness to 30”
    - “brightness 70”
  - Values are clamped to `0–100` to avoid invalid brightness levels.

- 🔊 **Friday-style TTS**
  - Uses [Coqui TTS](https://github.com/coqui-ai/TTS):
    - Model: `tts_models/en/ljspeech/tacotron2-DDC`
  - Synthesizes to `leo.wav` and plays back using `paplay`.
  - If TTS fails, falls back to printing text to console.

- 🎧 **Robust speech recognition**
  - Shared global `speech_recognition.Recognizer` + `Microphone`.
  - Single ambient-noise calibration at startup (`init_audio_calibration()`), then a fixed threshold:
    - Reduces lag and random re-calibration
    - More stable wake-word / command recognition
  - Wake listening loop and command listening use the same calibrated mic.

---

## Tech Stack

Language: Python 3.x (asyncio-friendly)

Speech Recognition: SpeechRecognition
 + Google Web Speech API

Text-To-Speech: Coqui TTS
 (TTS.api)

LLM / Conversation: Google Gemini (google-generativeai) via scripts.conversation_llm

Messaging: Telegram Bot API (async)

Audio Backend:

paplay (PulseAudio / PipeWire sink)

Display / Desktop:

X11 (DISPLAY, xhost +local:) for GUI automation and desktop integration

OS: Linux (tested on Arch / BlackArch-style setups)

---
### Clone & set up virtualenv
```bash

$ git clone https://github.com/your-username/leo_desktop_assistant.git
$ cd leo_desktop_assistant
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```
---
**Usage**
```bash
$ cd leo_desktop_assistant
$ source .venv/bin/activate
$ python main.py
```
---
### Roadmap Ideas

**Add more desktop controls:**

* Volume, window management, workspace switching, app launching.

* Add local LLM fallback (offline mode).

* Add hotword detection using a lightweight wake-word engine.

* Add configuration file for:

* Wake words

* Language

* TTS voice and output device

----
## License
This project is licensed under the MIT License.
