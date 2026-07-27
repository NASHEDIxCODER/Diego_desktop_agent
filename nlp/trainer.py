"""
Intent trainer for Leo NLP pipeline.

Generates example phrases for intents, trains the classifier,
stores in DuckDB, and persists the classifier to disk.

First-run training: ~30 seconds (model download + embedding)
Subsequent startups: < 1 second (load cached classifier)

Run manually with: python -m nlp.trainer --train
"""

import argparse
import itertools
import logging
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from memory.duckdb_store import store
from nlp.classifier import classifier, BUILTIN_INTENTS
from nlp.embeddings import embed, embed_batch

logger = logging.getLogger(__name__)

# Templates for generating example phrases per intent
TEMPLATES: Dict[str, List[str]] = {
    "greeting": ["{hello}", "{hello} leo", "{hello} there", "{hi} leo",
                  "good {time_of_day}", "{time_of_day} leo"],
    "exit": ["{goodbye}", "{goodbye} leo", "{bye} for now", "i'm {leaving}",
              "time to {leave}", "{shutdown} the system"],
    "youtube": ["{play} music on youtube", "{play} a {song}", "{open} youtube",
                 "{search} for {music}", "i want to {watch} {videos}",
                 "{start} youtube", "{play} my {playlist}"],
    "youtube_pause": ["{pause} the {video}", "{pause} the {song}", "{stop} playing",
                       "{pause} {music}", "hold on"],
    "youtube_resume": ["{resume} playing", "{continue} the {video}", "{unpause}",
                        "{play} again", "keep going"],
    "youtube_next": ["next {song}", "{skip} this {song}", "play next",
                      "next {track}", "{skip} to next"],
    "youtube_previous": ["previous {song}", "go back", "previous {track}",
                          "{rewind} the {song}", "play previous"],
    "youtube_volume_up": ["{increase} volume", "turn {up} the volume",
                           "volume {up}", "make it {louder}", "raise volume"],
    "youtube_volume_down": ["{decrease} volume", "turn {down} the volume",
                             "volume {down}", "make it {quieter}", "lower volume"],
    "youtube_mute": ["{mute} the {video}", "turn off {sound}", "{silence}", "{mute} {audio}"],
    "youtube_unmute": ["{unmute}", "turn on {sound}", "restore {sound}", "enable {audio}"],
    "youtube_seek_forward": ["{skip} forward", "fast forward", "jump forward",
                              "go forward {seconds}", "forward {seconds}"],
    "youtube_seek_backward": ["{rewind}", "go back", "jump back",
                               "rewind {seconds}", "go backward"],
    "youtube_speed_up": ["{increase} speed", "speed up", "faster", "play faster"],
    "youtube_speed_down": ["{decrease} speed", "slow down", "slower", "play slower"],
    "youtube_close": ["close youtube", "exit youtube", "stop youtube",
                       "quit youtube", "end the {video}"],
    "telegram_send": ["send a telegram", "{send} message on telegram",
                       "{message} someone on telegram", "send telegram to {name}",
                       "text on telegram"],
    "telegram_read": ["read telegram", "check telegram messages", "read my messages",
                       "check telegram", "show my telegrams"],
    "telegram_reply": ["reply on telegram", "reply to message", "reply back",
                        "respond on telegram", "send a reply"],
    "brightness_set": ["set brightness to {number}", "change brightness to {number}",
                        "adjust brightness to {number}", "brightness {number}",
                        "make screen brightness {number}"],
    "brightness_up": ["{increase} brightness", "brightness up", "brighter",
                       "more brightness", "turn up brightness"],
    "brightness_down": ["{decrease} brightness", "brightness down", "dimmer",
                         "less brightness", "turn down brightness"],
    "volume_up": ["{increase} system volume", "volume up system",
                   "turn up volume", "louder system"],
    "volume_down": ["{decrease} system volume", "volume down system",
                     "turn down volume", "quieter system"],
    "time_query": ["what time is it", "tell me the time", "current time",
                    "what's the time", "show time"],
    "date_query": ["what's the date", "what day is it", "today's date",
                    "tell me the date", "current date"],
    "weather_query": ["what's the weather", "weather forecast", "how's the weather",
                       "current weather", "tell me the weather"],
    "joke": ["tell me a joke", "make me laugh", "say something funny",
              "crack a joke", "give me a joke"],
    "news": ["what's the news", "latest news", "news headlines",
              "tell me the news", "current affairs"],
    "help": ["what can you do", "help", "show commands", "list features",
              "how can you help me"],
    "who_am_i": ["who am i", "what's my name", "tell me my name",
                  "do you know me", "who is the user"],
}

WORD_POOLS = {
    "hello": ["hello", "hey", "hi", "howdy", "greetings", "yo", "what's up"],
    "hi": ["hi", "hey", "hello", "howdy"],
    "goodbye": ["goodbye", "bye", "see you", "later", "farewell"],
    "bye": ["bye", "goodbye", "see ya", "later"],
    "leaving": ["leaving", "going", "heading out", "logging off"],
    "leave": ["leave", "go", "log off", "shut down"],
    "shutdown": ["shutdown", "power off", "turn off", "sleep"],
    "play": ["play", "start", "begin", "launch"],
    "song": ["song", "music", "video", "track", "audio"],
    "open": ["open", "launch", "start", "run"],
    "search": ["search", "find", "look for", "seek"],
    "music": ["music", "songs", "tunes", "melody", "audio"],
    "watch": ["watch", "see", "view", "listen to"],
    "videos": ["videos", "music", "content", "clips"],
    "start": ["start", "begin", "launch", "open"],
    "playlist": ["playlist", "list", "favorites", "liked songs"],
    "pause": ["pause", "stop", "hold", "freeze"],
    "video": ["video", "song", "music", "track", "clip"],
    "stop": ["stop", "halt", "end", "cease"],
    "resume": ["resume", "continue", "restart", "unpause"],
    "continue": ["continue", "carry on", "proceed", "resume"],
    "unpause": ["unpause", "resume", "continue"],
    "skip": ["skip", "next", "forward", "jump"],
    "rewind": ["rewind", "back", "backward", "previous"],
    "track": ["track", "song", "video", "clip"],
    "increase": ["increase", "raise", "boost", "turn up", "ramp up"],
    "up": ["up", "higher", "more", "increase"],
    "louder": ["louder", "higher", "more volume"],
    "decrease": ["decrease", "lower", "reduce", "turn down", "cut"],
    "down": ["down", "lower", "less", "decrease"],
    "quieter": ["quieter", "lower", "less volume"],
    "mute": ["mute", "silence", "quiet", "no sound"],
    "sound": ["sound", "audio", "volume"],
    "silence": ["silence", "mute", "quiet"],
    "audio": ["audio", "sound", "volume"],
    "unmute": ["unmute", "un silence", "restore sound"],
    "seconds": ["seconds", "sec"],
    "send": ["send", "message", "text", "forward"],
    "message": ["message", "text", "send", "telegram"],
    "name": ["john", "alice", "bob", "sarah", "mike", "emma", "david", "lisa"],
    "number": [str(i) for i in range(10, 101, 10)],
    "time_of_day": ["morning", "afternoon", "evening"],
}


class IntentTrainer:
    """Generates training data and trains the classifier."""

    def __init__(self):
        self._trained = False

    def generate_examples(self, intent_name: str, count: int = 100) -> List[str]:
        """Generate example phrases for an intent."""
        templates = TEMPLATES.get(intent_name, [intent_name])
        examples = set()
        while len(examples) < count:
            template = random.choice(templates)
            phrase = self._fill_template(template)
            if phrase:
                examples.add(phrase)
        return list(examples)[:count]

    def _fill_template(self, template: str) -> str:
        """Fill a template with random words from pools."""
        import re

        def replace_placeholder(match):
            key = match.group(1)
            pool = WORD_POOLS.get(key)
            if not pool:
                return match.group(0)
            return random.choice(pool)

        filled = re.sub(r"\{(\w+)\}", replace_placeholder, template)
        return " ".join(filled.split())

    def generate_all(self, examples_per_intent: int = 100) -> Dict[str, List[str]]:
        """Generate example phrases for all built-in intents."""
        all_examples: Dict[str, List[str]] = {}
        intent_names = [n for n in BUILTIN_INTENTS if n != "unknown"]
        total = len(intent_names)

        try:
            from tqdm import tqdm
            pbar = tqdm(intent_names, desc="Generating examples", unit="intent")
        except ImportError:
            pbar = intent_names

        for intent_name in pbar:
            examples = self.generate_examples(intent_name, examples_per_intent)
            all_examples[intent_name] = examples
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"examples": len(examples)})

        return all_examples

    async def train(self, examples_per_intent: int = 100) -> int:
        """
        Generate examples, compute embeddings, store in DuckDB,
        build classifier index, and persist to disk.

        Returns the total number of examples stored.
        """
        logger.info("Starting training with %d examples per intent", examples_per_intent)
        t0 = time.perf_counter()

        all_examples = self.generate_all(examples_per_intent)
        total = 0
        intent_names = list(all_examples.keys())

        try:
            from tqdm import tqdm
            pbar = tqdm(intent_names, desc="Training", unit="intent")
        except ImportError:
            pbar = intent_names

        for intent_name in pbar:
            examples = all_examples[intent_name]

            # Store in DuckDB
            intent_id = store.add_intent(intent_name)
            for example in examples:
                emb = embed(example)
                store.add_example(intent_id, example, emb)

            # Add to classifier
            classifier.add_intent(intent_name, examples)
            total += len(examples)

            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"examples": total})

        # Build classifier index
        classifier.build_index()

        # Persist classifier to disk
        classifier.save()

        self._trained = True
        elapsed = time.perf_counter() - t0
        logger.info("Training complete: %d examples across %d intents in %.1fs",
                    total, len(all_examples), elapsed)
        return total

    def is_trained(self) -> bool:
        return self._trained


# Global trainer
trainer = IntentTrainer()


def is_classifier_ready() -> bool:
    """Check if a saved classifier exists on disk."""
    return settings.CLASSIFIER_PATH.exists()


def ensure_classifier() -> bool:
    """
    Load or train the classifier.
    Returns True if the classifier is ready.

    This is fast on subsequent startups (< 1s) because
    the classifier is saved to disk.
    """
    if classifier.load():
        return True

    # Train synchronously (blocking startup)
    import asyncio
    logger.info("No saved classifier found. Running initial training...")
    try:
        total = asyncio.run(trainer.train(examples_per_intent=100))
        logger.info("Initial training complete: %d examples", total)
        return True
    except Exception as e:
        logger.error("Initial training failed: %s", e)
        return False


if __name__ == "__main__":
    # CLI entry point for manual retraining
    parser = argparse.ArgumentParser(description="Leo NLP Trainer")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--examples", type=int, default=100,
                        help="Examples per intent (default: 100)")
    args = parser.parse_args()

    if args.train:
        from telemetry.logger import setup_logging
        setup_logging("INFO")
        logging.getLogger().setLevel(logging.INFO)
        print("Starting Leo NLP training...")
        import asyncio
        total = asyncio.run(trainer.train(examples_per_intent=args.examples))
        print(f"Training complete! Generated {total} examples.")
    else:
        print("Usage: python -m nlp.trainer --train")