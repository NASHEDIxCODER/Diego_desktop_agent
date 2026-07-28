"""
Generate static JSON datasets for all intents.
Replaces runtime generation with pre-computed examples.

Usage:
    python scripts/generate_datasets.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Intent-specific templates and word pools for generating varied examples
TEMPLATES = {
    "greeting": ["hello", "hi {name}", "hey {name}", "good morning", "good afternoon",
                 "good evening", "what's up", "yo {name}", "hey there", "howdy",
                 "greetings", "nice to see you", "hello there", "hiya", "hey buddy"],
    "exit": ["goodbye", "bye", "exit", "quit", "see you later", "good night",
             "talk to you later", "later", "see ya", "take care", "bye bye",
             "i'm leaving", "shutdown", "power off", "sleep"],
    "time_query": ["what time is it", "tell me the time", "current time",
                   "what's the time", "show me the time", "time now"],
    "date_query": ["what's the date", "what day is it", "today's date",
                   "tell me the date", "current date", "what day is today"],
    "open_app": ["open {app}", "launch {app}", "start {app}", "run {app}"],
    "take_note": ["take a note", "save this thought", "write a note", "remember this",
                  "note this down", "make a note", "save a reminder"],
    "read_note": ["read my notes", "show my notes", "what did I note", "get my notes",
                  "list notes", "open notes", "read the note"],
    "system_info": ["system info", "show system information", "what is my system",
                    "system status", "computer info", "show specs"],
}

WORD_POOLS = {
    "name": ["leo", "there", "buddy", "friend"],
    "app": ["calculator", "browser", "firefox", "vs code", "terminal", "settings",
            "spotify", "chrome", "file manager", "discord", "telegram", "python",
            "notepad", "gedit", "thunderbird", "libreoffice", "slack", "zoom",
            "the video player", "the music player", "the photo editor"],
}


def generate_examples_for_intent(intent: str, max_examples: int = 100) -> list:
    """Generate all unique examples for an intent, limited to max_examples."""
    templates = TEMPLATES.get(intent, [intent])
    examples: set = set()

    for t in templates:
        if "{" not in t:
            examples.add(t)
            if len(examples) >= max_examples:
                break
            continue

        # Find placeholders like {app}, {name}
        import re
        placeholders = re.findall(r'\{(\w+)\}', t)
        pools = [WORD_POOLS.get(p, [p]) for p in placeholders]

        for i, word in enumerate(pools[0] if pools else []):
            phrase = t
            for ph_name, replacement in zip(placeholders, [word] + ([""] * (len(placeholders) - 1))):
                phrase = phrase.replace('{' + ph_name + '}', replacement, 1)
            phrase = ' '.join(phrase.split())
            examples.add(phrase)
            if len(examples) >= max_examples:
                break

        if len(examples) >= max_examples:
            break

    return sorted(list(examples))[:max_examples]


def main():
    datasets_dir = pathlib.Path('datasets/intents')
    datasets_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for intent in sorted(TEMPLATES.keys()):
        examples = generate_examples_for_intent(intent, max_examples=100)
        filepath = datasets_dir / f'{intent}.json'
        filepath.write_text(json.dumps(examples, indent=2))
        print(f'{intent:30s} {len(examples):3d} examples -> {filepath}')
        total += len(examples)

    print(f'\nTotal: {total} examples across {len(TEMPLATES)} intents')


if __name__ == '__main__':
    main()