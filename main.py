"""
Leo Desktop Assistant — Modular Production-Grade Entry Point

Architecture:
  Speech → STT → Preprocessor → Intent Classifier → Entity Extractor
  → Context Manager → Plugin Router → Response → TTS

LLMs (Gemini/OpenAI/Ollama) are OPTIONAL providers used only for:
  - unknown intent
  - reasoning
  - coding
  - summarization
  - long conversations

Everything else executes locally.
"""

import argparse
import asyncio
import difflib
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure DISPLAY exists
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

try:
    subprocess.run(["xhost", "+local:"], check=False)
except Exception:
    pass

from config.settings import settings
from telemetry.logger import setup_logging

# Setup structured logging
setup_logging()

logger = logging.getLogger(__name__)

# ── Core imports ──────────────────────────────────────────
from core.event_bus import bus, Event
from core.plugin_manager import plugin_manager

# ── NLP imports ───────────────────────────────────────────
from nlp.parser import parser, parse_text
from nlp.classifier import classifier
from nlp.context import context_manager
from nlp.entities import extract_entities
from nlp.trainer import trainer, ensure_classifier, is_classifier_ready

# ── Voice imports ─────────────────────────────────────────
from voice.stt import calibrate, listen, listen_wake
from voice.tts import speak

# ── AI imports ────────────────────────────────────────────
from ai.llm_client import llm_client, llm_chat

# ── Auth imports ──────────────────────────────────────────
from auth import faceauth

# ── Wake word variants ────────────────────────────────────
WAKE_VARIANTS = [
    "hello leo",
    "leo",
    "lio",
    "hey leo",
    "hello lio",
]


def fuzzy_match(text: str, variants, cutoff: float = 0.7) -> bool:
    """Fuzzy match text against a list of variants."""
    text = text.lower().strip()
    for v in variants:
        v = v.lower().strip()
        if v in text:
            return True
        ratio = difflib.SequenceMatcher(None, text, v).ratio()
        if ratio >= cutoff:
            return True
    return False


def wish_me():
    """Greet the user based on time of day."""
    import datetime
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning, sir.")
    elif 12 <= hour < 18:
        speak("Good afternoon, sir.")
    else:
        speak("Good evening, sir.")
    speak("I am Leo, your assistant. How may I help you?")


async def handle_intent(parsed: dict) -> str:
    """
    Route a parsed intent to the appropriate handler.

    Returns a response string to speak.
    """
    intent = parsed["intent"]
    entities = parsed["entities"]
    confidence = parsed["confidence"]

    logger.info("Handling intent: %s (confidence=%.4f)", intent, confidence)

    # ── Greeting ──────────────────────────────────────────
    if intent == "greeting":
        wish_me()
        return ""

    # ── Exit ──────────────────────────────────────────────
    if intent == "exit":
        speak("Goodbye, have a nice day.")
        return "__EXIT__"

    # ── YouTube intents ───────────────────────────────────
    if intent.startswith("youtube"):
        event_type = intent  # e.g., "youtube_open", "youtube_pause"
        await bus.emit(event_type, data=entities, source="nlp")
        return ""

    # ── Telegram intents ──────────────────────────────────
    if intent.startswith("telegram"):
        event_type = intent
        await bus.emit(event_type, data=entities, source="nlp")
        return ""

    # ── Brightness intents ────────────────────────────────
    if intent.startswith("brightness"):
        event_type = intent
        await bus.emit(event_type, data=entities, source="nlp")
        return ""

    # ── Time query ────────────────────────────────────────
    if intent == "time_query":
        import datetime
        now = datetime.datetime.now()
        response = f"The time is {now.strftime('%I:%M %p')}."
        speak(response)
        return ""

    # ── Date query ────────────────────────────────────────
    if intent == "date_query":
        import datetime
        now = datetime.datetime.now()
        response = f"Today is {now.strftime('%A, %B %d, %Y')}."
        speak(response)
        return ""

    # ── Help ──────────────────────────────────────────────
    if intent == "help":
        response = ("I can control YouTube, send Telegram messages, "
                    "adjust brightness, tell the time and date, "
                    "and have conversations. What would you like to do?")
        speak(response)
        return ""

    # ── Joke ──────────────────────────────────────────────
    if intent == "joke":
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the AI break up with the database? Too many relationships!",
            "What do you call a fake noodle? An impasta!",
        ]
        import random
        speak(random.choice(jokes))
        return ""

    # ── Who am I ──────────────────────────────────────────
    if intent == "who_am_i":
        speak("You are my user. I recognize you by your face.")
        return ""

    # ── Unknown / LLM fallback ────────────────────────────
    if intent == "unknown" or parsed.get("needs_llm", False):
        response = await llm_chat(parsed["text"])
        speak(response)
        return ""

    # ── Fallback to LLM for anything else ─────────────────
    response = await llm_chat(parsed["text"])
    speak(response)
    return ""


async def main_loop():
    """Main assistant loop."""
    logger.info("Leo Desktop Assistant starting...")

    # 1. Calibrate microphone
    calibrate()

    # 2. Load or train the NLP classifier (fast if cached)
    logger.info("Loading NLP classifier...")
    ensure_classifier()

    # 3. Load plugins
    logger.info("Loading plugins...")
    await plugin_manager.load_all()
    await plugin_manager.initialize_all()
    logger.info("Plugins loaded: %s", list(plugin_manager.plugins.keys()))

    # 4. Register speak handler on event bus
    async def on_speak(event: Event):
        text = event.data.get("text", "")
        if text:
            speak(text)

    bus.on("speak", on_speak)

    # 5. Main loop
    while True:
        # Wait for wake word
        logger.info("Listening for wake word...")
        wake_text = listen_wake(phrase_time_limit=3)

        if not wake_text:
            continue

        if not fuzzy_match(wake_text, WAKE_VARIANTS):
            continue

        logger.info("Wake word detected: %s", wake_text)

        # Face authentication
        user_name = faceauth.recognize_faces()
        if not user_name:
            faceauth.Unknown_Face()
            continue

        speak(f"Hello {user_name}, how may I assist you?")

        # Command loop
        while True:
            query = listen(phrase_time_limit=7)
            if not query:
                continue

            query = query.lower().strip()
            logger.info("Command: %s", query)

            # Parse with NLP pipeline
            parsed = await parse_text(query)
            logger.debug("Parsed: %s", parsed)

            # Handle the intent
            result = await handle_intent(parsed)

            if result == "__EXIT__":
                return

            # Log to DuckDB
            try:
                from memory.duckdb_store import store
                store.add_command(
                    text=query,
                    intent=parsed["intent"],
                    confidence=parsed["confidence"],
                    response=result,
                )
            except Exception as e:
                logger.debug("Failed to log command: %s", e)


async def main():
    """Entry point."""
    try:
        await main_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        await plugin_manager.shutdown_all()
        logger.info("Leo shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leo Desktop Assistant")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the NLP classifier and exit")
    parser.add_argument("--examples", type=int, default=100,
                        help="Examples per intent for training (default: 100)")
    args = parser.parse_args()

    if args.train:
        # CLI mode: train and exit
        from telemetry.logger import setup_logging
        setup_logging("INFO")
        logging.getLogger().setLevel(logging.INFO)
        print(f"Retraining NLP with {args.examples} examples per intent...")
        total = asyncio.run(trainer.train(examples_per_intent=args.examples))
        print(f"Training complete! {total} examples generated.")
        sys.exit(0)

    # Normal startup
    asyncio.run(main())
