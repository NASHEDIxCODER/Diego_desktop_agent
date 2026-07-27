"""
Intent classifier for Leo NLP pipeline.

Matches user input against known intents using:
1. Semantic similarity (sentence-transformers embeddings)
2. RapidFuzz fuzzy matching as fallback
3. LLM fallback for unknown intents

Supports save/load to disk for fast startup.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings
from nlp.embeddings import embed, embed_batch, find_best_match
from nlp.normalizer import normalize
from nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)

# Built-in intents with example phrases
BUILTIN_INTENTS = {
    "greeting": [
        "hello", "hi leo", "hey leo", "good morning", "good afternoon",
        "good evening", "what's up", "yo leo", "hey there", "howdy",
        "greetings", "nice to see you", "hello there", "hiya", "hey buddy",
    ],
    "exit": [
        "goodbye", "bye", "exit", "quit", "see you later", "good night",
        "talk to you later", "later", "see ya", "take care", "bye bye",
        "i'm leaving", "shutdown", "power off", "sleep",
    ],
    "youtube": [
        "play music on youtube", "open youtube", "play a song", "play video",
        "search on youtube", "play my playlist", "start youtube",
        "play some music", "i want to watch videos", "open music video",
        "play a video on youtube", "search youtube for a song",
        "start playing music", "play the latest hits",
    ],
    "youtube_pause": [
        "pause", "pause the video", "pause the song", "stop playing",
        "pause music", "hold on", "pause playback",
    ],
    "youtube_resume": [
        "play", "resume", "resume playing", "continue playing",
        "unpause", "play the video again", "continue",
    ],
    "youtube_next": [
        "next song", "next video", "skip this song", "skip",
        "play next", "next track", "go to next", "forward",
    ],
    "youtube_previous": [
        "previous song", "previous video", "go back", "previous track",
        "play the previous one", "back", "rewind song",
    ],
    "youtube_volume_up": [
        "increase volume", "turn up the volume", "volume up",
        "louder", "make it louder", "raise volume", "higher volume",
    ],
    "youtube_volume_down": [
        "decrease volume", "turn down the volume", "volume down",
        "quieter", "lower volume", "reduce volume", "softer",
    ],
    "youtube_mute": [
        "mute", "mute the video", "turn off sound", "silence",
        "no sound", "mute the audio",
    ],
    "youtube_unmute": [
        "unmute", "unmute the video", "turn on sound",
        "restore sound", "enable audio",
    ],
    "youtube_seek_forward": [
        "skip forward", "fast forward", "jump forward",
        "skip ahead", "go forward", "forward 10 seconds",
    ],
    "youtube_seek_backward": [
        "rewind", "go back", "skip backward", "jump back",
        "rewind 10 seconds", "go backward", "back up",
    ],
    "youtube_speed_up": [
        "increase speed", "speed up", "faster", "play faster",
        "increase playback speed", "go faster",
    ],
    "youtube_speed_down": [
        "decrease speed", "slow down", "slower", "play slower",
        "decrease playback speed", "go slower",
    ],
    "youtube_close": [
        "close youtube", "exit youtube", "stop youtube",
        "close the video", "end youtube", "quit youtube",
    ],
    "telegram_send": [
        "send a telegram", "send message on telegram",
        "message someone on telegram", "send telegram to",
        "text on telegram", "send a message via telegram",
        "send a text on telegram",
    ],
    "telegram_read": [
        "read telegram", "check telegram messages",
        "read my messages", "check telegram", "read latest message",
        "show my telegrams", "what's new on telegram",
    ],
    "telegram_reply": [
        "reply on telegram", "reply to message", "reply back",
        "respond on telegram", "send a reply",
    ],
    "brightness_set": [
        "set brightness", "change brightness", "adjust brightness",
        "brightness level", "set screen brightness",
        "make screen brighter", "make screen dimmer",
    ],
    "brightness_up": [
        "increase brightness", "brightness up", "brighter",
        "more brightness", "turn up brightness", "raise brightness",
    ],
    "brightness_down": [
        "decrease brightness", "brightness down", "dimmer",
        "less brightness", "turn down brightness", "lower brightness",
    ],
    "volume_up": [
        "increase system volume", "volume up", "turn up volume",
        "louder system", "raise system volume",
    ],
    "volume_down": [
        "decrease system volume", "volume down", "turn down volume",
        "quieter system", "lower system volume",
    ],
    "time_query": [
        "what time is it", "tell me the time", "current time",
        "what's the time", "show me the time", "time now",
    ],
    "date_query": [
        "what's the date", "what day is it", "today's date",
        "tell me the date", "current date", "what day is today",
    ],
    "weather_query": [
        "what's the weather", "weather forecast", "how's the weather",
        "what's the temperature", "is it raining", "weather outside",
        "current weather", "tell me the weather",
    ],
    "open_app": [
        "open application", "launch program", "start app",
        "open browser", "launch terminal", "open settings",
    ],
    "take_note": [
        "take a note", "write this down", "remember this",
        "make a note", "save a note", "note this down",
        "i need to remember",
    ],
    "read_note": [
        "read my notes", "show notes", "open notes",
        "what did i note", "recall notes", "list notes",
    ],
    "joke": [
        "tell me a joke", "make me laugh", "say something funny",
        "tell a joke", "crack a joke", "do you know any jokes",
        "give me a joke",
    ],
    "news": [
        "what's the news", "latest news", "news headlines",
        "tell me the news", "current affairs", "what's happening",
        "any breaking news",
    ],
    "help": [
        "what can you do", "help", "show commands", "list features",
        "what are your capabilities", "how can you help me",
        "tell me what you can do", "available commands",
    ],
    "system_info": [
        "system status", "how is my computer", "system info",
        "check system", "what's my system specs",
        "cpu usage", "memory usage", "battery status",
    ],
    "who_am_i": [
        "who am i", "what's my name", "tell me my name",
        "do you know me", "who is the user",
    ],
    "unknown": [],
}


class IntentClassifier:
    """
    Classifies user input into known intents using semantic similarity.

    Supports save/load to disk so training only runs once.
    """

    def __init__(self):
        self._intents: Dict[str, List[str]] = dict(BUILTIN_INTENTS)
        self._intent_embeddings: Dict[str, np.ndarray] = {}
        self._example_embeddings: Dict[str, np.ndarray] = {}
        self._intent_names: List[str] = []
        self._all_examples: List[str] = []
        self._all_intent_labels: List[str] = []
        self._ready = False

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Serialise the classifier to disk (intents + precomputed embeddings)."""
        dest = path or settings.CLASSIFIER_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "intents": self._intents,
            "intent_names": self._intent_names,
            "all_examples": self._all_examples,
            "all_intent_labels": self._all_intent_labels,
            "ready": self._ready,
        }

        # numpy arrays — convert to lists for pickle compatibility
        if self._example_embeddings.get("all") is not None:
            state["example_embeddings_all"] = self._example_embeddings["all"]

        intent_embeds = {}
        for name, arr in self._intent_embeddings.items():
            intent_embeds[name] = arr
        state["intent_embeddings"] = intent_embeds

        with open(dest, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Classifier saved to %s (%d intents, %d examples)",
                     dest, len(self._intent_names), len(self._all_examples))

    def load(self, path: Optional[Path] = None) -> bool:
        """Load classifier state from disk. Returns True on success."""
        src = path or settings.CLASSIFIER_PATH

        if not src.exists():
            logger.info("No saved classifier found at %s", src)
            return False

        try:
            with open(src, "rb") as f:
                state = pickle.load(f)

            self._intents = state.get("intents", dict(BUILTIN_INTENTS))
            self._intent_names = state.get("intent_names", [])
            self._all_examples = state.get("all_examples", [])
            self._all_intent_labels = state.get("all_intent_labels", [])
            self._ready = state.get("ready", False)

            if "example_embeddings_all" in state:
                self._example_embeddings["all"] = state["example_embeddings_all"]

            self._intent_embeddings = state.get("intent_embeddings", {})

            logger.info("Classifier loaded from %s (%d intents, %d examples)",
                         src, len(self._intent_names), len(self._all_examples))
            return True

        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)
            return False

    # ── Intent management ─────────────────────────────────────

    def add_intent(self, name: str, examples: List[str]) -> None:
        """Add or extend a custom intent."""
        if name not in self._intents:
            self._intents[name] = []
        self._intents[name].extend(examples)
        self._ready = False

    def remove_intent(self, name: str) -> None:
        """Remove an intent."""
        self._intents.pop(name, None)
        self._ready = False

    def build_index(self) -> None:
        """Pre-compute all embeddings for fast matching."""
        self._intent_names = list(self._intents.keys())
        self._all_examples = []
        self._all_intent_labels = []

        for intent_name, examples in self._intents.items():
            for example in examples:
                if example.strip():
                    self._all_examples.append(example)
                    self._all_intent_labels.append(intent_name)

        if self._all_examples:
            self._example_embeddings["all"] = embed_batch(self._all_examples)

        # Per-intent centroid embeddings
        for intent_name, examples in self._intents.items():
            clean = [e for e in examples if e.strip()]
            if clean:
                self._intent_embeddings[intent_name] = embed_batch(clean).mean(axis=0)

        self._ready = True
        logger.info("Built index: %d intents, %d examples",
                    len(self._intent_names), len(self._all_examples))

    # ── Classification ────────────────────────────────────────

    def classify(self, text: str,
                 threshold: Optional[float] = None) -> Tuple[str, float, Dict]:
        """
        Classify text into an intent.

        Returns (intent_name, confidence, metadata).
        """
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD

        if not self._ready:
            self.build_index()

        norm_text = normalize(text)
        if not norm_text:
            return "unknown", 0.0, {}

        query_emb = embed(norm_text)

        # 1. Semantic similarity against all examples
        if self._all_examples:
            all_embs = self._example_embeddings.get("all")
            if all_embs is not None and all_embs.shape[0] > 0:
                idx, score = find_best_match(query_emb, all_embs, threshold)
                if idx >= 0:
                    intent = self._all_intent_labels[idx]
                    return intent, score, {"matched_example": self._all_examples[idx]}

        # 2. Centroid matching
        best_intent = "unknown"
        best_score = 0.0
        for intent_name, centroid in self._intent_embeddings.items():
            if centroid.shape[0] > 0:
                from nlp.embeddings import cosine_similarity
                sim = cosine_similarity(query_emb, centroid)
                if sim > best_score:
                    best_score = sim
                    best_intent = intent_name

        if best_score >= threshold:
            return best_intent, best_score, {}

        # 3. Fuzzy matching fallback
        from rapidfuzz import fuzz
        norm_text = norm_text.lower()
        for intent_name, examples in self._intents.items():
            for example in examples:
                ratio = fuzz.ratio(norm_text, example.lower())
                if ratio / 100.0 >= threshold:
                    return intent_name, ratio / 100.0, {"fuzzy_match": example}

        return "unknown", best_score, {}


# Global classifier instance
classifier = IntentClassifier()


def classify_intent(text: str) -> Tuple[str, float, Dict]:
    """Convenience function to classify intent."""
    return classifier.classify(text)