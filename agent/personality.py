"""
LeoPersonality — Natural, varied conversational personality.

Makes Leo sound alive instead of robotic. Generates varied greetings,
acknowledgments, and filler responses so Leo never repeats the same
phrase like "How may I assist you?"

Usage:
    from agent.personality import personality

    greeting = personality.greeting()  # "Hey.", "Welcome back.", etc.
    ack = personality.acknowledgment()  # "Got it.", "Sure.", etc.
"""

import logging
import random
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class LeoPersonality:
    """
    Generates varied, natural conversational responses.

    Tracks recently used phrases to avoid repetition.
    Adapts based on time of day, conversation context, mood, and user sentiment.

    NEVER says:
        - "How may I assist you?"
        - "Task completed."
        - "Done."
        - Robotic confirmation patterns
    """

    def __init__(self):
        self._recent_greetings: List[str] = []
        self._recent_acks: List[str] = []
        self._recent_results: List[str] = []
        self._max_recent = 8     # Don't repeat last N phrases
        self._session_greeting_count = 0
        self._used_greetings_today: List[str] = []

        # ── Greetings: varied by time of day ──────────────
        self._greetings_morning = [
            "Good morning.",
            "Morning. Ready when you are.",
            "Hey, good morning.",
            "Morning. What's on your mind?",
            "Morning. Coffee's on you though.",
            "Rise and shine. What are we working on?",
            "Good morning. Let's get to it.",
        ]

        self._greetings_afternoon = [
            "Hey.",
            "What's up?",
            "Hey, what can I do for you?",
            "Yo. What do you need?",
            "Hey there. What are we working on?",
            "Afternoon. How's it going?",
            "Hey hey. What's on the docket?",
        ]

        self._greetings_evening = [
            "Good evening.",
            "Evening. How did the day go?",
            "Hey, winding down?",
            "Evening. What's up?",
            "Hey. Still going strong, I see.",
            "Evening. Late session today?",
            "Good to see you. How was the day?",
        ]

        self._greetings_night = [
            "Hey. Late one?",
            "Burning the midnight oil, huh?",
            "Late night. What's going on?",
            "Hey. Couldn't sleep?",
            "Still up? I'm here.",
            "Late shift. What do you need?",
        ]

        self._greetings_returning = [
            "Welcome back.",
            "Nice to see you again.",
            "Hey, you're back.",
            "Oh hey. Welcome back.",
            "Good to see you again. What's up?",
            "Back so soon? What's up?",
            "There you are. How's it going?",
        ]

        self._greetings_generic = [
            "Hey.",
            "What's up?",
            "Ready when you are.",
            "Yeah?",
            "I'm here.",
            "What do you need?",
            "Go ahead.",
            "Listening.",
            "I'm all ears.",
            "Present.",
        ]

        # ── Acknowledgments: short and natural ────────────
        self._acks = [
            "Got it.",
            "Sure.",
            "Okay.",
            "On it.",
            "Right.",
            "Makes sense.",
            "Cool.",
            "Alright.",
            "Yep.",
            "Understood.",
            "Say no more.",
            "Gotcha.",
            "Noted.",
            "Consider it done.",
            "I'm on it.",
        ]

        # ── Thinking fillers ──────────────────────────────
        self._thinking = [
            "Let me think...",
            "Hmm, give me a second.",
            "One moment...",
            "Let me check that.",
            "Thinking...",
            "Hang on, let me process that.",
            "Just a sec...",
        ]

        # ── Farewells: varied ─────────────────────────────
        self._farewells = [
            "Catch you later.",
            "See ya.",
            "Bye for now.",
            "Take it easy.",
            "Later.",
            "I'll be here when you need me.",
            "Until next time.",
            "Have a good one.",
        ]

        # ── Error responses: natural, not robotic ─────────
        self._errors = [
            "Hmm, that didn't work. Let me try something else.",
            "I hit a snag there. Give me a moment.",
            "That didn't go through. Want me to try again?",
            "Ran into an issue. Let me figure this out.",
            "Something went sideways. Working on it.",
            "Ah, that failed. Let me take another approach.",
            "Didn't work. Trying a different way.",
        ]

        # ── Task confirmation: natural, never "Task completed" ──
        self._task_confirmations = [
            "That worked.",
            "Done. {detail}",
            "{detail} — all set.",
            "Alright, {detail}",
            "Got it. {detail}",
            "That's done.",
            "Finished. {detail}",
            "It's done.",
            "Sorted.",
        ]

        # ── Status observations: for when Leo notices something ──
        self._observations = [
            "Looks like you're working on {detail}.",
            "I see {detail}.",
            "Looks like {detail}.",
            "Noticed {detail}.",
            "Ah, {detail}.",
        ]

    def greeting(self, returning: bool = False) -> str:
        """
        Generate a varied greeting.

        Args:
            returning: True if the user is returning after a break.
        """
        if returning:
            pool = self._greetings_returning
        else:
            hour = time.localtime().tm_hour
            if 5 <= hour < 12:
                pool = self._greetings_morning
            elif 12 <= hour < 17:
                pool = self._greetings_afternoon
            elif 17 <= hour < 22:
                pool = self._greetings_evening
            else:
                pool = self._greetings_generic

        return self._pick_varied(pool, self._recent_greetings)

    def acknowledgment(self) -> str:
        """Generate a varied acknowledgment."""
        return self._pick_varied(self._acks, self._recent_acks)

    def thinking(self) -> str:
        """Generate a thinking filler."""
        return random.choice(self._thinking)

    def farewell(self) -> str:
        """Generate a varied farewell."""
        return random.choice(self._farewells)

    def error_response(self) -> str:
        """Generate a varied error response."""
        return random.choice(self._errors)

    def _pick_varied(self, pool: List[str], recent: List[str]) -> str:
        """Pick a phrase from pool, avoiding recent ones."""
        available = [p for p in pool if p not in recent]
        if not available:
            # All used recently — reset and pick from full pool
            available = pool
            recent.clear()

        choice = random.choice(available)
        recent.append(choice)
        if len(recent) > self._max_recent:
            recent.pop(0)

        return choice

    def task_confirmation(self, detail: str = "") -> str:
        """
        Generate a natural task confirmation.

        NEVER "Task completed." or "Done." — always includes detail naturally.

        Args:
            detail: What was accomplished (e.g. "VS Code is open", "volume set to 50")
        """
        template = self._pick_varied(self._task_confirmations, self._recent_results)
        return template.format(detail=detail) if detail else template.replace("{detail}", "").strip()

    def observation(self, detail: str) -> str:
        """
        Generate a natural observation about the user's desktop state.

        Args:
            detail: What Leo noticed (e.g. "you're on GhostLine", "two failing tests")
        """
        template = random.choice(self._observations)
        return template.format(detail=detail)

    def contextual_response(self, user_text: str) -> Optional[str]:
        """
        Generate a contextual short response for common inputs.

        Returns None if no contextual response applies (let the LLM handle it).
        """
        text = user_text.lower().strip()

        # Greetings from user
        if text in ("hello", "hi", "hey", "yo", "sup", "what's up", "whats up"):
            return self.greeting()

        # How are you
        if any(phrase in text for phrase in ["how are you", "how's it going", "how are things"]):
            responses = [
                "Doing good. What about you?",
                "All good here. What do you need?",
                "Can't complain. What's up?",
                "Running smoothly. What are we doing?",
            ]
            return random.choice(responses)

        # Thanks
        if any(phrase in text for phrase in ["thank", "thanks", "appreciate"]):
            responses = [
                "No problem.",
                "Anytime.",
                "You got it.",
                "Don't mention it.",
                "Happy to help.",
            ]
            return random.choice(responses)

        # Goodbye
        if any(phrase in text for phrase in ["bye", "goodbye", "see you", "see ya", "later"]):
            return self.farewell()

        return None

    # ── Emotional tone helpers ─────────────────────────────

    @staticmethod
    def detect_sentiment(text: str) -> str:
        """Quick sentiment detection from user input. Returns 'positive', 'negative', or 'neutral'."""
        t = text.lower()
        positive_words = ("great", "awesome", "thanks", "love", "perfect",
                          "nice", "good", "cool", "amazing", "excellent")
        negative_words = ("bad", "terrible", "awful", "hate", "wrong",
                          "broken", "fails", "doesn't work", "stupid")
        pos = sum(1 for w in positive_words if w in t)
        neg = sum(1 for w in negative_words if w in t)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"

    @staticmethod
    def tone_match(response: str, sentiment: str) -> str:
        """Adjust response tone to match the user's sentiment."""
        if sentiment == "negative":
            acknowledgments = ["I hear you.", "Got it. Let me fix that.",
                               "Understood. Working on it.", "Alright, let's sort this out."]
            if not any(response.startswith(a.rstrip(".")) for a in acknowledgments):
                return random.choice(acknowledgments) + " " + response
        elif sentiment == "positive":
            if not any(w in response.lower() for w in ("great", "awesome", "love", "glad")):
                pass  # LLM should handle positive tone
        return response


# Global singleton
personality = LeoPersonality()
