"""
DiegoPersonality — Natural, varied conversational personality.

Makes Diego sound like a real desktop companion, not a chatbot.
Generates varied greetings, acknowledgments, and filler responses
so Diego never repeats the same phrase.

NEVER says:
  - "How may I assist you?"
  - "Task completed."
  - "Command executed."
  - "Action successful."
  - "Processing request."
  - Robotic confirmation patterns

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


class DiegoPersonality:
    """
    Generates varied, natural conversational responses.

    Tracks recently used phrases to avoid repetition.
    Adapts based on time of day, conversation context, mood, and user sentiment.
    """

    def __init__(self):
        self._recent_greetings: List[str] = []
        self._recent_acks: List[str] = []
        self._recent_results: List[str] = []
        self._recent_observations: List[str] = []
        self._recent_clarifications: List[str] = []
        self._recent_errors: List[str] = []
        self._max_recent = 10     # Don't repeat last N phrases
        self._session_greeting_count = 0

        # ── Greetings: varied by time of day + mood ──────
        self._greetings_morning = [
            "Good morning.",
            "Morning. Ready when you are.",
            "Hey, good morning.",
            "Morning. What's on your mind?",
            "Morning. Coffee's on you though.",
            "Rise and shine. What are we working on?",
            "Good morning. Let's get to it.",
            "Morning. Got anything fun lined up?",
            "Hey. Fresh day, fresh code.",
            "Morning. What's the plan?",
            "Good morning. I'm all ears.",
            "Morning. Let's make it count.",
        ]

        self._greetings_afternoon = [
            "Hey.",
            "What's up?",
            "Hey, what can I do for you?",
            "Yo. What do you need?",
            "Hey there. What are we working on?",
            "Afternoon. How's it going?",
            "Hey hey. What's on the docket?",
            "Afternoon. Anything interesting?",
            "Hey. How can I help?",
            "What's happening?",
            "Hey. What are we up to?",
            "Afternoon. What's going on?",
        ]

        self._greetings_evening = [
            "Good evening.",
            "Evening. How did the day go?",
            "Hey, winding down?",
            "Evening. What's up?",
            "Hey. Still going strong, I see.",
            "Evening. Late session today?",
            "Good to see you. How was the day?",
            "Evening. Got something to wrap up?",
            "Hey. How's the evening treating you?",
            "Evening. What do you need?",
        ]

        self._greetings_night = [
            "Hey. Late one?",
            "Burning the midnight oil, huh?",
            "Late night. What's going on?",
            "Hey. Couldn't sleep?",
            "Still up? I'm here.",
            "Late shift. What do you need?",
            "Night owl mode. What's up?",
            "It's late. Everything okay?",
            "Hey. Working late?",
            "Still going? What's up?",
        ]

        self._greetings_returning = [
            "Welcome back.",
            "Nice to see you again.",
            "Hey, you're back.",
            "Oh hey. Welcome back.",
            "Good to see you again. What's up?",
            "Back so soon? What's up?",
            "There you are. How's it going?",
            "You're back. Everything good?",
            "Hey again. What's going on?",
            "Welcome back. Where were we?",
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
            "Here.",
            "What can I do?",
            "Hey. What's going on?",
            "I'm here. What's up?",
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
            "One sec.",
            "Will do.",
            "You got it.",
            "No problem.",
            "Alright, let me handle it.",
            "Absolutely.",
            "Sounds good.",
            "Let's do it.",
            "Working on it.",
            "Right away.",
            "Sure thing.",
            "No worries.",
            "Happy to.",
            "Easy.",
            "Done deal.",
            "Perfect.",
            "That works.",
            "Good call.",
            "Nice.",
            "Alrighty.",
            "You bet.",
            "For sure.",
            "Totally.",
            "Agreed.",
            "Makes total sense.",
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
            "Let me look...",
            "Processing...",
            "Give me a moment...",
            "Hmm...",
            "Let me see...",
            "One second...",
            "Hold on...",
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
            "Bye. Don't break anything while I'm gone.",
            "See you around.",
            "Peace.",
            "Alright, I'll be listening.",
            "Talk soon.",
            "Goodbye. I'm around if you need me.",
            "Later. I'll be right here.",
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
            "That failed. But I've got other ideas.",
            "Nope, that didn't work. Let me try plan B.",
            "Struck out on that one. Let me switch tactics.",
            "That didn't quite work. One more try.",
            "Hmm, that didn't land. Let me fix it.",
            "Not quite. Trying again.",
            "That didn't go as planned. Give me a sec.",
            "Missed that one. Let me adjust.",
        ]

        # ── Task confirmation: natural, never "Task completed" ──
        self._task_confirmations = [
            "Done. {detail}",
            "{detail} — all set.",
            "Alright, {detail}",
            "Got it. {detail}",
            "That's done.",
            "Finished. {detail}",
            "It's done.",
            "Sorted.",
            "{detail}. Done.",
            "All good. {detail}",
            "Handled. {detail}",
            "{detail} — done.",
            "There you go. {detail}",
            "{detail}. Anything else?",
            "That's sorted. {detail}",
            "{detail} — all good.",
            "Done and dusted. {detail}",
            "{detail}. Easy.",
            "All set. {detail}",
            "{detail} — that's it.",
        ]

        # ── Standalone confirmations (no detail available) ──
        # CRITICAL FIX (audit B1): when `detail` is empty, rendering a
        # detail-template with "{detail}" stripped produced malformed
        # fragments that were spoken by TTS verbatim:
        #   "{detail}. Easy."        → ". Easy."
        #   "{detail} — all set."    → "— all set."
        #   "Alright, {detail}"      → "Alright,"
        # Empty-detail confirmations must therefore come from a pool of
        # templates that are complete sentences on their own.
        self._standalone_confirmations = [
            "Done.",
            "That's done.",
            "It's done.",
            "Finished.",
            "Sorted.",
            "All set.",
            "All good.",
            "Handled.",
            "Got it.",
            "There you go.",
            "Done and dusted.",
            "Easy.",
        ]

        # ── Status observations ──────────────────────────
        self._observations = [
            "Looks like you're working on {detail}.",
            "I see {detail}.",
            "Looks like {detail}.",
            "Noticed {detail}.",
            "Ah, {detail}.",
            "I see you have {detail}.",
            "Looks like {detail} is going on.",
            "I noticed {detail}.",
            "Seems like {detail}.",
            "Oh, {detail}.",
        ]

        # ── Clarifications: one simple question ───────────
        self._clarifications = [
            "Which one?",
            "Which browser?",
            "Which folder?",
            "Which project?",
            "Which monitor?",
            "Which app?",
            "Which file?",
            "Which one did you mean?",
            "Can you be more specific?",
            "Which of those?",
            "Which one should I use?",
            "Which one are you talking about?",
        ]

        # ── Corrections: user says "that's not what I meant" ──
        self._corrections = [
            "Got it. Tell me what you wanted instead.",
            "Ah, my bad. What did you mean?",
            "Okay, let me redo that. What were you thinking?",
            "Understood. What should I do instead?",
            "Right, let me fix that. What did you want?",
            "Gotcha. What's the right way?",
            "My mistake. What should it be?",
        ]

        # ── "How are you" responses ──────────────────────
        self._how_are_you = [
            "Doing well. What's up?",
            "All good here. What do you need?",
            "Can't complain. What's up?",
            "Running smoothly. What are we doing?",
            "Pretty good. What can I help with?",
            "Solid. What's on your mind?",
            "Doing great. What's going on?",
            "All good. What are we up to?",
            "Feeling sharp. What do you need?",
            "Good, good. What's happening?",
        ]

        # ── Thanks responses ─────────────────────────────
        self._thanks = [
            "No problem.",
            "Anytime.",
            "You got it.",
            "Don't mention it.",
            "Happy to help.",
            "Glad I could help.",
            "That's what I'm here for.",
            "Sure thing.",
            "No worries at all.",
            "My pleasure.",
            "Easy. Happy to do it.",
            "Anytime, that's what I'm for.",
        ]

        # ── "What can you do" responses ──────────────────
        self._what_can_you_do = [
            "I can open apps, control music, adjust volume and brightness, search the web, read your screen, and help with your projects. Just ask.",
            "Open apps, play music, search, control your system, read screens, and help with code. What do you need?",
            "I'm your desktop companion. Open things, play things, search things, and help you work. Try me.",
        ]

        # ── "Who are you" responses ──────────────────────
        self._who_are_you = [
            "I'm Diego, your desktop assistant.",
            "Diego. Your desktop companion.",
            "I'm Diego. I live on your desktop and help you get things done.",
        ]

        # ── "Who made you" responses ─────────────────────
        self._who_made_you = [
            "I was created by Yeshu.",
            "Yeshu built me.",
            "I'm Yeshu's creation.",
        ]

        # ── "What are you working on" responses ──────────
        self._what_working_on = [
            "You're working on {detail}.",
            "Looks like {detail} is your current project.",
            "You've been on {detail}.",
            "{detail} — that's what you're working on.",
        ]

        # ── "Already running" responses ──────────────────
        self._already_running = [
            "{app} is already open.",
            "{app} is already running.",
            "Already open. {app} is up.",
            "{app} is already there.",
        ]

        # ── "Opening" responses (speak immediately) ──────
        self._opening = [
            "Opening {app}.",
            "Opening {app} now.",
            "Sure, opening {app}.",
            "On it — opening {app}.",
            "Opening {app} for you.",
            "{app} coming up.",
        ]

        # ── "Searching" responses ────────────────────────
        self._searching = [
            "Searching for {query}.",
            "Looking up {query}.",
            "Searching {query} now.",
            "On it — searching for {query}.",
            "Finding {query} for you.",
        ]

        # ── "Playing" responses ──────────────────────────
        self._playing = [
            "Playing {query}.",
            "Starting {query}.",
            "Playing {query} now.",
            "On it — playing {query}.",
            "{query} coming up.",
        ]

        # ── "Paused" responses ───────────────────────────
        self._paused = [
            "Paused.",
            "Paused the music.",
            "Stopped it.",
            "Paused. Want me to resume?",
        ]

        # ── "Resumed" responses ──────────────────────────
        self._resumed = [
            "Resumed.",
            "Playing again.",
            "Back on.",
            "Resumed the music.",
        ]

        # ── "Closed" responses ───────────────────────────
        self._closed = [
            "Closed {app}.",
            "{app} is closed.",
            "Done — closed {app}.",
            "{app} shut down.",
        ]

        # ── "Volume" responses ───────────────────────────
        self._volume_up = [
            "Volume up.",
            "Louder.",
            "Turned it up.",
            "Volume increased.",
        ]

        self._volume_down = [
            "Volume down.",
            "Quieter.",
            "Turned it down.",
            "Volume decreased.",
        ]

        self._volume_mute = [
            "Muted.",
            "Silenced.",
            "Muted the sound.",
        ]

        self._volume_set = [
            "Volume at {percent} percent.",
            "Set to {percent} percent.",
            "Volume is now {percent} percent.",
        ]

        # ── "Brightness" responses ───────────────────────
        self._brightness_up = [
            "Brightness up.",
            "Brighter.",
            "Turned it up.",
        ]

        self._brightness_down = [
            "Brightness down.",
            "Dimmer.",
            "Turned it down.",
        ]

        self._brightness_set = [
            "Brightness at {percent} percent.",
            "Set to {percent} percent.",
            "Brightness is now {percent} percent.",
        ]

        # ── "Locked" responses ───────────────────────────
        self._locked = [
            "Locked.",
            "Screen locked.",
            "Locked it up.",
        ]

        # ── "Next/Previous" responses ────────────────────
        self._next = [
            "Next track.",
            "Skipping.",
            "Next one.",
        ]

        self._previous = [
            "Previous track.",
            "Going back.",
            "Previous one.",
        ]

        # ── "Shuffle/Repeat" responses ───────────────────
        self._shuffle = [
            "Shuffled.",
            "Shuffle on.",
            "Mixed it up.",
        ]

        self._repeat = [
            "Repeat on.",
            "Repeating.",
            "Looping.",
        ]

        # ── "Time/Date" responses ────────────────────────
        self._time = [
            "It's {time}.",
            "The time is {time}.",
            "{time}.",
        ]

        self._date = [
            "Today is {date}.",
            "It's {date}.",
            "{date}.",
        ]

        # ── "Screen" responses ───────────────────────────
        self._screen = [
            "Here's what's on your screen: {detail}",
            "On your screen: {detail}",
            "You're looking at {detail}",
        ]

        # ── "Continue" responses ─────────────────────────
        self._continue = [
            "Continuing with {goal}.",
            "Picking up where we left off — {goal}.",
            "Resuming {goal}.",
            "Back to {goal}.",
        ]

        # ── "Go back" responses ──────────────────────────
        self._go_back = [
            "Going back.",
            "Undoing that.",
            "Back to before.",
            "Reverting.",
        ]

        # ── "Not what I meant" responses ─────────────────
        self._not_what_meant = [
            "Got it. Tell me what you wanted instead.",
            "Ah, my bad. What did you mean?",
            "Okay, let me redo that. What were you thinking?",
            "Understood. What should I do instead?",
        ]

        # ── "Yes/No" responses ───────────────────────────
        self._yes = [
            "Sure.",
            "Yep.",
            "Okay.",
            "Absolutely.",
            "On it.",
            "Let's do it.",
            "Right away.",
            "Gotcha.",
            "Sounds good.",
            "You bet.",
        ]

        self._no = [
            "Alright, never mind then.",
            "Okay, skipping that.",
            "Got it, not doing that.",
            "Sure, leaving it.",
        ]

    def greeting(self, returning: bool = False) -> str:
        """Generate a varied greeting."""
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
            elif 22 <= hour or hour < 3:
                pool = self._greetings_night
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

    def error_response(self, detail: str = "") -> str:
        """Generate a varied error response, optionally with detail."""
        response = self._pick_varied(self._errors, self._recent_errors)
        if detail:
            response = response.rstrip(".") + f" ({detail})."
        return response

    def clarification(self) -> str:
        """Generate a natural clarification question."""
        return self._pick_varied(self._clarifications, self._recent_clarifications)

    def correction(self) -> str:
        """Respond to a user correction naturally."""
        return random.choice(self._corrections)

    def how_are_you(self) -> str:
        """Respond to 'how are you'."""
        return random.choice(self._how_are_you)

    def thanks(self) -> str:
        """Respond to thanks."""
        return random.choice(self._thanks)

    def what_can_you_do(self) -> str:
        """Respond to 'what can you do'."""
        return random.choice(self._what_can_you_do)

    def who_are_you(self) -> str:
        """Respond to 'who are you'."""
        return random.choice(self._who_are_you)

    def who_made_you(self) -> str:
        """Respond to 'who made you'."""
        return random.choice(self._who_made_you)

    def what_working_on(self, detail: str) -> str:
        """Respond to 'what am I working on'."""
        template = random.choice(self._what_working_on)
        return template.format(detail=detail)

    def already_running(self, app: str) -> str:
        """Tell the user an app is already open."""
        template = random.choice(self._already_running)
        return template.format(app=app)

    def opening(self, app: str) -> str:
        """Speak immediately when opening an app."""
        template = random.choice(self._opening)
        return template.format(app=app)

    def searching(self, query: str) -> str:
        """Speak immediately when searching."""
        template = random.choice(self._searching)
        return template.format(query=query)

    def playing(self, query: str) -> str:
        """Speak immediately when playing media."""
        template = random.choice(self._playing)
        return template.format(query=query)

    def paused(self) -> str:
        """Confirm pause."""
        return random.choice(self._paused)

    def resumed(self) -> str:
        """Confirm resume."""
        return random.choice(self._resumed)

    def closed(self, app: str) -> str:
        """Confirm closing an app."""
        template = random.choice(self._closed)
        return template.format(app=app)

    def volume_up(self) -> str:
        return random.choice(self._volume_up)

    def volume_down(self) -> str:
        return random.choice(self._volume_down)

    def volume_mute(self) -> str:
        return random.choice(self._volume_mute)

    def volume_set(self, percent: int) -> str:
        template = random.choice(self._volume_set)
        return template.format(percent=percent)

    def brightness_up(self) -> str:
        return random.choice(self._brightness_up)

    def brightness_down(self) -> str:
        return random.choice(self._brightness_down)

    def brightness_set(self, percent: int) -> str:
        template = random.choice(self._brightness_set)
        return template.format(percent=percent)

    def locked(self) -> str:
        return random.choice(self._locked)

    def next_track(self) -> str:
        return random.choice(self._next)

    def previous_track(self) -> str:
        return random.choice(self._previous)

    def shuffle(self) -> str:
        return random.choice(self._shuffle)

    def repeat(self) -> str:
        return random.choice(self._repeat)

    def time(self, time_str: str) -> str:
        template = random.choice(self._time)
        return template.format(time=time_str)

    def date(self, date_str: str) -> str:
        template = random.choice(self._date)
        return template.format(date=date_str)

    def screen(self, detail: str) -> str:
        template = random.choice(self._screen)
        return template.format(detail=detail)

    def continue_goal(self, goal: str) -> str:
        template = random.choice(self._continue)
        return template.format(goal=goal)

    def go_back(self) -> str:
        return random.choice(self._go_back)

    def not_what_meant(self) -> str:
        return random.choice(self._not_what_meant)

    def yes(self) -> str:
        return random.choice(self._yes)

    def no(self) -> str:
        return random.choice(self._no)

    def _pick_varied(self, pool: List[str], recent: List[str]) -> str:
        """Pick a phrase from pool, avoiding recent ones."""
        available = [p for p in pool if p not in recent]
        if not available:
            available = pool
            recent.clear()

        choice = random.choice(available)
        recent.append(choice)
        if len(recent) > self._max_recent:
            recent.pop(0)

        return choice

    @staticmethod
    def _clean_fragment(text: str) -> str:
        """Safety net: normalize a rendered confirmation into a clean sentence.

        Strips dangling leading/trailing punctuation left over from a
        template whose {detail} placeholder was removed, so fragments
        like ". Easy." / "— all set." / "Alright," can never reach TTS.
        """
        cleaned = text.strip()
        # Strip leading orphan punctuation/whitespace (". Easy." → "Easy.")
        cleaned = cleaned.lstrip(".,;:—–- ")
        # Strip trailing dangling comma (no sentence ends in a comma)
        cleaned = cleaned.rstrip(",")
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def task_confirmation(self, detail: str = "") -> str:
        """
        Generate a natural task confirmation.

        NEVER "Task completed." or "Done." — always includes detail naturally.

        CRITICAL FIX (audit B1): with no detail, pick from the standalone
        confirmation pool instead of stripping "{detail}" from a
        detail-template (which produced ". Easy." / "— all set." /
        "Alright," fragments). A sanitizer is applied as a final safety
        net so no malformed fragment can ever be returned.
        """
        if detail:
            template = self._pick_varied(self._task_confirmations, self._recent_results)
            return self._clean_fragment(template.format(detail=detail))
        standalone = self._pick_varied(
            self._standalone_confirmations, self._recent_results)
        return self._clean_fragment(standalone)

    def observation(self, detail: str) -> str:
        """Generate a natural observation about the user's desktop state."""
        template = self._pick_varied(self._observations, self._recent_observations)
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

        # BLOCKER 1 FIX (2026-08-30): short greeting PHRASES ("hello dear",
        # "hey there", "hi diego", "good morning man") are conversation,
        # never commands. A greeting-lead with <= 3 words and NO action
        # verb must get a conversational greeting response instead of
        # falling through to the planner (the "Hello dear" -> get_time +
        # type_text 29s-turn bug).
        words = text.split()
        if words and len(words) <= 3:
            first = words[0].strip(",.!?")
            if first in ("hello", "hi", "hey", "yo", "sup", "hiya", "hola"):
                if not any(v in text for v in (
                        "open", "close", "play", "pause", "resume", "stop",
                        "start", "run", "search", "find", "set", "change",
                        "switch", "scroll", "click", "type", "write", "press",
                        "send", "create", "delete", "volume", "brightness",
                        "mute", "lock", "shutdown", "restart", "next",
                        "previous", "skip", "read", "show", "tell", "what",
                        "who", "when", "where", "why", "how", "which")):
                    return self.greeting()

        # How are you
        if any(phrase in text for phrase in ["how are you", "how's it going", "how are things"]):
            return self.how_are_you()

        # Thanks
        if any(phrase in text for phrase in ["thank", "thanks", "appreciate"]):
            return self.thanks()

        # Goodbye
        if any(phrase in text for phrase in ["bye", "goodbye", "see you", "see ya", "later"]):
            return self.farewell()

        # What can you do
        if any(phrase in text for phrase in ["what can you do", "what are you capable", "what do you do"]):
            return self.what_can_you_do()

        # Who are you
        if any(phrase in text for phrase in ["who are you", "what are you", "what is your name"]):
            return self.who_are_you()

        # Who made you
        if any(phrase in text for phrase in ["who made you", "who created you", "who built you"]):
            return self.who_made_you()

        # Corrections
        if any(phrase in text for phrase in ["not what i meant", "that's not what", "thats not what",
                                              "not that", "wrong one", "i meant something else"]):
            return self.correction()

        return None

    # ── Emotional tone helpers ─────────────────────────────

    @staticmethod
    def detect_sentiment(text: str) -> str:
        """Quick sentiment detection from user input. Returns 'positive', 'negative', or 'neutral'."""
        t = text.lower()
        positive_words = ("great", "awesome", "thanks", "love", "perfect",
                          "nice", "good", "cool", "amazing", "excellent",
                          "wow", "fantastic", "brilliant", "wonderful")
        negative_words = ("bad", "terrible", "awful", "hate", "wrong",
                          "broken", "fails", "doesn't work", "stupid",
                          "annoying", "frustrating", "useless", "worst")
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
            acknowledgments = [
                "I hear you.",
                "Got it. Let me fix that.",
                "Understood. Working on it.",
                "Alright, let's sort this out.",
                "That's frustrating. Let me help.",
            ]
            if not any(response.startswith(a.rstrip(".")) for a in acknowledgments):
                return random.choice(acknowledgments) + " " + response
        elif sentiment == "positive":
            # Don't modify — LLM should already be upbeat
            pass
        return response

    def proactive_comment(self, event_type: str, detail: str = "") -> Optional[str]:
        """
        Generate a proactive comment for desktop events.

        Args:
            event_type: "download_complete", "build_failure", "git_conflict",
                        "terminal_error", "low_battery", "docker_failure"
            detail: Specific detail about the event.
        """
        templates = {
            "download_complete": [
                "Looks like that download finished.",
                "Download complete, by the way.",
                "That download wrapped up.",
            ],
            "build_failure": [
                "Hey, that build didn't go through. Want me to look at it?",
                "Build failed. I can check the error if you want.",
                "Hit a build error there. Need a hand?",
            ],
            "git_conflict": [
                "There's a merge conflict. Want me to help sort it out?",
                "Git conflict in {detail}. I can take a look.",
                "Looks like a merge conflict. Need a second pair of eyes?",
            ],
            "terminal_error": [
                "Saw an error in the terminal. Everything okay?",
                "Terminal threw an error. Want me to investigate?",
                "Error in the terminal. I can help debug if you want.",
            ],
            "low_battery": [
                "Battery's getting low. Might want to plug in soon.",
                "Running low on battery. Just a heads up.",
                "Battery's at {detail}. You might want to find a charger.",
            ],
            "docker_failure": [
                "Docker seems unhappy. Want me to check the logs?",
                "Docker failed. I can inspect the container if you want.",
                "Docker issue detected. Need me to troubleshoot?",
            ],
        }
        pool = templates.get(event_type, [])
        if not pool:
            return None
        template = random.choice(pool)
        return template.format(detail=detail) if detail else template


# Global singleton
personality = DiegoPersonality()