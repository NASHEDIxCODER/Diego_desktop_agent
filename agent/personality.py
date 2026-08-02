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
    Adapts based on time of day and conversation context.
    """

    def __init__(self):
        self._recent_greetings: List[str] = []
        self._recent_acks: List[str] = []
        self._max_recent = 5  # Don't repeat last N phrases

        # Greetings — never use "How may I assist you?"
        self._greetings_morning = [
            "Good morning.",
            "Morning! Ready when you are.",
            "Hey, good morning.",
            "Up and at 'em. What's on your mind?",
            "Morning. Coffee's on you though.",
        ]

        self._greetings_afternoon = [
            "Hey.",
            "What's up?",
            "Hey, what can I do for you?",
            "Yo. What do you need?",
            "Hey there. What are we working on?",
        ]

        self._greetings_evening = [
            "Good evening.",
            "Evening. How did the day go?",
            "Hey, winding down?",
            "Evening. What's up?",
            "Hey. Still going strong, I see.",
        ]

        self._greetings_returning = [
            "Welcome back.",
            "Nice to see you again.",
            "Hey, you're back.",
            "Oh hey. Welcome back.",
            "Good to see you again. What's up?",
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
        ]

        # Acknowledgments (short, natural)
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
        ]

        # Thinking fillers (when Leo needs a moment)
        self._thinking = [
            "Let me think...",
            "Hmm, give me a second.",
            "One moment...",
            "Let me check that.",
            "Thinking...",
        ]

        # Farewells
        self._farewells = [
            "Catch you later.",
            "See ya.",
            "Bye for now.",
            "Take it easy.",
            "Later.",
            "I'll be here when you need me.",
        ]

        # Error responses (varied, not robotic)
        self._errors = [
            "Hmm, that didn't work. Let me try something else.",
            "I hit a snag there. Give me a moment.",
            "That didn't go through. Want me to try again?",
            "Ran into an issue. Let me figure this out.",
            "Something went sideways. Working on it.",
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


# Global singleton
personality = LeoPersonality()