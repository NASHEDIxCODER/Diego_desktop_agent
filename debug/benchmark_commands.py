"""
Command Recognition Benchmark — 500+ spoken commands.

Tests command normalization and classification accuracy across:
  - App launching
  - Volume/brightness control
  - Music control
  - System control
  - Web/search
  - Follow-up commands
  - Noisy variants
  - Mixed Hindi-English
  - Long commands

Run:
    python debug/benchmark_commands.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)

import compat  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# Benchmark data — generated programmatically for compactness
# ═══════════════════════════════════════════════════════════════

def build_commands() -> List[Tuple[str, str, str]]:
    """Generate 500+ benchmark commands.
    Returns (spoken, expected_normalized, expected_class).
    """
    cmds: List[Tuple[str, str, str]] = []

    # ── App launching: open X (X = each app, multiple phrasing variants) ──
    apps = [
        ("firefox", "firefox", "desktop_open"),
        ("fire fox", "firefox", "desktop_open"),
        ("chrome", "chrome", "desktop_open"),
        ("crome", "chrome", "desktop_open"),
        ("google chrome", "chrome", "desktop_open"),
        ("vs code", "vscode", "desktop_open"),
        ("v s code", "vscode", "desktop_open"),
        ("visual studio code", "vscode", "desktop_open"),
        ("terminal", "terminal", "desktop_open"),
        ("spotify", "spotify", "desktop_open"),
        ("spot if i", "spotify", "desktop_open"),
        ("slack", "slack", "desktop_open"),
        ("discord", "discord", "desktop_open"),
        ("telegram", "telegram", "desktop_open"),
        ("notion", "notion", "desktop_open"),
        ("calculator", "calculator", "desktop_open"),
        ("settings", "settings", "desktop_open"),
        ("files", "files", "desktop_open"),
        ("pycharm", "pycharm", "desktop_open"),
        ("pie charm", "pycharm", "desktop_open"),
        ("github", "github", "desktop_open"),
        ("get hub", "github", "desktop_open"),
        ("obsidian", "obsidian", "desktop_open"),
        ("youtube", "youtube", "desktop_open"),
        ("you tube", "youtube", "desktop_open"),
        ("vlc", "vlc", "desktop_open"),
    ]

    open_variants = [
        "open {app}",
        "open the {app}",
        "open a {app}",
        "open an {app}",
        "launch {app}",
        "start {app}",
        "open {app} please",
        "please open {app}",
        "can you open {app}",
        "could you open {app}",
        "hey Diego open {app}",
        "ok Diego open {app}",
        "i want to open {app}",
        "i'd like to open {app}",
        "go ahead and open {app}",
        "just open {app}",
        "open {app} for me",
        "open {app} right now",
    ]
    for app, canonical, cls in apps:
        for variant in open_variants:
            spoken = variant.format(app=app)
            expected = f"open {canonical}"
            cmds.append((spoken, expected, cls))

    # ── Volume/brightness ──
    volume = [
        ("volume up", "volume up", "volume_up"),
        ("turn up the volume", "volume up", "volume_up"),
        ("increase volume", "volume up", "volume_up"),
        ("make it louder", "volume up", "volume_up"),
        ("louder", "volume up", "volume_up"),
        ("volume down", "volume down", "volume_down"),
        ("turn down the volume", "volume down", "volume_down"),
        ("decrease volume", "volume down", "volume_down"),
        ("quieter", "volume down", "volume_down"),
        ("mute", "mute", "volume_mute"),
        ("mute the volume", "mute", "volume_mute"),
        ("unmute", "unmute", "volume_mute"),
        ("set volume to 50", "set volume to 50", "volume_set"),
        ("set the volume to 50 percent", "set the volume to 50", "volume_set"),
        ("volume 50", "volume 50", "volume_set"),
        ("volume to 75", "volume to 75", "volume_set"),
        ("brightness up", "brightness up", "brightness_up"),
        ("increase brightness", "brightness up", "brightness_up"),
        ("make it brighter", "brightness up", "brightness_up"),
        ("brightness down", "brightness down", "brightness_down"),
        ("decrease brightness", "brightness down", "brightness_down"),
        ("dimmer", "brightness down", "brightness_down"),
        ("set brightness to 50", "set brightness to 50", "brightness_set"),
        ("set the brightness to 75 percent", "set the brightness to 75", "brightness_set"),
        ("brightness 25", "brightness 25", "brightness_set"),
    ]
    cmds.extend(volume)

    # ── Music control ──
    # CRITICAL FIX: The normalizer returns NATURAL commands that the
    # CommandRouter can match — NOT tag formats like "play_music:lofi".
    # The router's regex patterns match "play lofi" → play_media action.
    music = [
        ("play music", "play music", "play_media"),
        ("play some music", "play music", "play_media"),
        ("play lofi", "play lofi", "play_media"),
        ("play some lofi hip hop", "play lofi hip hop", "play_media"),
        ("play jazz", "play jazz", "play_media"),
        ("play rock", "play rock", "play_media"),
        ("play classical", "play classical", "play_media"),
        ("play pop", "play pop", "play_media"),
        ("play the beatles", "play the beatles", "play_media"),
        ("play taylor swift", "play taylor swift", "play_media"),
        ("play a song", "play a song", "play_media"),
        ("play my playlist", "play my playlist", "play_media"),
        ("pause", "pause", "music_pause"),
        ("pause the music", "pause music", "music_pause"),
        ("pause the song", "pause song", "music_pause"),
        ("stop", "stop", "music_stop"),
        ("stop the music", "stop music", "music_stop"),
        ("stop the song", "stop song", "music_stop"),
        ("resume", "resume", "music_resume"),
        ("resume the music", "resume the music", "music_resume"),
        ("continue playing", "continue", "music_resume"),
        ("next", "next", "music_next"),
        ("next song", "next", "music_next"),
        ("next track", "next", "music_next"),
        ("skip", "next", "music_next"),
        ("skip this song", "next song", "music_next"),
        ("previous", "previous", "music_previous"),
        ("previous song", "previous", "music_previous"),
        ("previous track", "previous", "music_previous"),
        ("go back", "previous", "music_previous"),
        ("shuffle", "shuffle", "music_shuffle"),
        ("repeat", "repeat", "music_repeat"),
    ]
    cmds.extend(music)

    # ── System control ──
    system = [
        ("lock screen", "lock", "lock_screen"),
        ("lock the screen", "lock", "lock_screen"),
        ("lock my computer", "lock", "lock_screen"),
        ("lock my pc", "lock", "lock_screen"),
        ("lock my laptop", "lock", "lock_screen"),
        ("shutdown", "shutdown", "shutdown"),
        ("shut down", "shutdown", "shutdown"),
        ("shut down the computer", "shutdown", "shutdown"),
        ("power off", "shutdown", "shutdown"),
        ("turn off the computer", "shutdown", "shutdown"),
        ("restart", "restart", "restart"),
        ("restart the computer", "restart", "restart"),
        ("reboot", "restart", "restart"),
        ("reboot the computer", "restart", "restart"),
        ("scroll down", "scroll down", "scroll"),
        ("scroll up", "scroll up", "scroll"),
        ("scroll down the page", "scroll down", "scroll"),
        ("scroll up the page", "scroll up", "scroll"),
        ("take a screenshot", "screenshot", "screenshot"),
        ("screenshot", "screenshot", "screenshot"),
        ("capture the screen", "screenshot", "screenshot"),
        ("what's on my screen", "what is on my screen", "read_screen"),
        ("what is on my screen", "what is on my screen", "read_screen"),
        ("read the screen", "read the screen", "read_screen"),
        ("what do you see", "what do you see", "read_screen"),
        ("what are you looking at", "what are you looking at", "read_screen"),
    ]
    cmds.extend(system)

    # ── Web/search ──
    # CRITICAL FIX: The normalizer returns NATURAL commands that the
    # CommandRouter can match — NOT tag formats like "search:cats".
    # The router's regex patterns match "search cats" → browser_search.
    search = [
        ("search for cats", "search cats", "search"),
        ("search cats", "search cats", "search"),
        ("search the web for cats", "search cats", "search"),
        ("look up cats", "search cats", "search"),
        ("find cats", "search cats", "search"),
        ("google cats", "search cats", "search"),
        ("what is a cat", "search a cat", "search"),
        ("what is the weather", "search the weather", "search"),
        ("weather in tokyo", "search tokyo", "search"),
        ("weather in london", "search london", "search"),
        ("weather in new york", "search new york", "search"),
        ("weather in paris", "search paris", "search"),
        ("weather in delhi", "search delhi", "search"),
        ("weather in mumbai", "search mumbai", "search"),
        ("who is the president", "search the president", "search"),
        ("who is elon musk", "search elon musk", "search"),
        ("who is albert einstein", "search albert einstein", "search"),
        ("how to make pasta", "search make pasta", "search"),
        ("how to cook rice", "search cook rice", "search"),
        ("how to learn python", "search learn python", "search"),
        ("how to code", "search code", "search"),
        ("news", "search news", "search"),
        ("news about technology", "search technology", "search"),
        ("news about sports", "search sports", "search"),
        ("news about politics", "search politics", "search"),
        ("news about science", "search science", "search"),
        ("news about space", "search space", "search"),
        ("news about ai", "search ai", "search"),
        ("news about crypto", "search crypto", "search"),
        ("news about stocks", "search stocks", "search"),
        ("news about movies", "search movies", "search"),
        ("news about music", "search music", "search"),
        ("news about games", "search games", "search"),
        ("news about cars", "search cars", "search"),
        ("news about food", "search food", "search"),
        ("news about health", "search health", "search"),
    ]
    cmds.extend(search)

    # ── Follow-up commands ──
    # CRITICAL FIX: "open it", "open that", "open this" are NOT swallowed
    # by the normalizer — they are pronouns that need resolution by
    # conv_memory ("open it" → "open firefox"). The normalizer deliberately
    # leaves them intact so the DecisionEngine can resolve them.
    followups = [
        ("open it", "open it", "follow_up"),
        ("open that", "open that", "follow_up"),
        ("open this", "open this", "follow_up"),
        ("launch it", "open it", "follow_up"),
        ("start it", "open it", "follow_up"),
        ("run it", "open it", "follow_up"),
        ("continue", "continue", "follow_up"),
        ("go on", "continue", "follow_up"),
        ("keep going", "continue", "follow_up"),
        ("carry on", "continue", "follow_up"),
        ("proceed", "continue", "follow_up"),
        ("go back", "previous", "follow_up"),
        ("back", "previous", "follow_up"),
        ("close that", "close that", "follow_up"),
        ("close it", "close it", "follow_up"),
        ("again", "again", "follow_up"),
        ("do it again", "again", "follow_up"),
        ("repeat", "repeat", "follow_up"),
        ("say again", "again", "follow_up"),
        ("cancel", "cancel", "follow_up"),
        ("never mind", "cancel", "follow_up"),
        ("forget it", "cancel", "follow_up"),
        ("skip", "next", "follow_up"),
        ("next", "next", "follow_up"),
        ("previous", "previous", "follow_up"),
        ("that one", "that one", "follow_up"),
        ("this one", "this one", "follow_up"),
        ("yes", "yes", "follow_up"),
        ("yeah", "yes", "follow_up"),
        ("sure", "yes", "follow_up"),
        ("ok", "yes", "follow_up"),
        ("okay", "yes", "follow_up"),
        ("go ahead", "yes", "follow_up"),
        ("do it", "yes", "follow_up"),
        ("no", "no", "follow_up"),
        ("nope", "no", "follow_up"),
        ("nah", "no", "follow_up"),
    ]
    cmds.extend(followups)

    # ── Noisy variants (programmatic: prefix noise before commands) ──
    noisy_prefixes = [
        "please ", "can you ", "could you ", "would you ", "will you ",
        "hey Diego ", "ok Diego ", "okay Diego ", "i want to ", "i need to ",
        "i'd like to ", "go ahead and ", "just ",
    ]
    base_commands = [
        ("open fire fox", "open firefox", "desktop_open"),
        ("open firefox", "open firefox", "desktop_open"),
        ("open chrome", "open chrome", "desktop_open"),
        ("open vs code", "open vscode", "desktop_open"),
        ("open terminal", "open terminal", "desktop_open"),
        ("volume up", "volume up", "volume_up"),
        ("volume down", "volume down", "volume_down"),
        ("play music", "play music", "play_media"),
        ("pause", "pause", "music_pause"),
        ("stop", "stop", "music_stop"),
        ("lock screen", "lock", "lock_screen"),
        ("shutdown", "shutdown", "shutdown"),
    ]
    for prefix in noisy_prefixes:
        for base, expected, cls in base_commands:
            cmds.append((prefix + base, expected, cls))

    # ── Mixed Hindi-English ──
    hindi = [
        ("open firefox karo", "open firefox", "desktop_open"),
        ("firefox kholo", "open firefox", "desktop_open"),
        ("firefox khol do", "open firefox", "desktop_open"),
        ("open chrome karo", "open chrome", "desktop_open"),
        ("chrome kholo", "open chrome", "desktop_open"),
        ("chrome khol do", "open chrome", "desktop_open"),
        ("open vs code karo", "open vscode", "desktop_open"),
        ("vs code kholo", "open vscode", "desktop_open"),
        ("vs code khol do", "open vscode", "desktop_open"),
        ("open terminal karo", "open terminal", "desktop_open"),
        ("terminal kholo", "open terminal", "desktop_open"),
        ("terminal khol do", "open terminal", "desktop_open"),
        ("open spotify karo", "open spotify", "desktop_open"),
        ("spotify kholo", "open spotify", "desktop_open"),
        ("spotify khol do", "open spotify", "desktop_open"),
        ("volume up karo", "volume up", "volume_up"),
        ("volume badhao", "volume up", "volume_up"),
        ("volume badha do", "volume up", "volume_up"),
        ("volume down karo", "volume down", "volume_down"),
        ("volume kam karo", "volume down", "volume_down"),
        ("volume kam kar do", "volume down", "volume_down"),
        ("mute karo", "mute", "volume_mute"),
        ("mute kar do", "mute", "volume_mute"),
        ("play music karo", "play music", "play_media"),
        ("music chalao", "play music", "play_media"),
        ("music chala do", "play music", "play_media"),
        ("play lofi karo", "play lofi", "play_media"),
        ("lofi chalao", "play lofi", "play_media"),
        ("lofi chala do", "play lofi", "play_media"),
        ("pause karo", "pause", "music_pause"),
        ("pause kar do", "pause", "music_pause"),
        ("stop karo", "stop", "music_stop"),
        ("stop kar do", "stop", "music_stop"),
        ("resume karo", "resume", "music_resume"),
        ("resume kar do", "resume", "music_resume"),
        ("next karo", "next", "music_next"),
        ("next kar do", "next", "music_next"),
        ("previous karo", "previous", "music_previous"),
        ("previous kar do", "previous", "music_previous"),
        ("lock karo", "lock", "lock_screen"),
        ("lock kar do", "lock", "lock_screen"),
        ("shutdown karo", "shutdown", "shutdown"),
        ("shutdown kar do", "shutdown", "shutdown"),
        ("restart karo", "restart", "restart"),
        ("restart kar do", "restart", "restart"),
        ("search karo", "search", "search"),
        ("search for cats karo", "search cats", "search"),
    ]
    cmds.extend(hindi)

    # ── Long commands (multi-app) ──
    long_apps = ["firefox", "chrome", "terminal", "spotify", "slack", "discord"]
    for n in range(3, len(long_apps) + 1):
        apps_str = " and ".join(f"open {a}" for a in long_apps[:n])
        expected = " and ".join(
            f"open {a.replace('fire fox', 'firefox').replace('crome', 'chrome').replace('spot if i', 'spotify')}"
            for a in long_apps[:n]
        )
        cmds.append((apps_str, expected, "multi"))

    return cmds


COMMANDS = build_commands()


# ═══════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════

PASS = 0
FAIL = 0
FAILURES: List[Tuple[str, str, str, str]] = []
START_TIME = 0.0


def run_benchmark():
    global PASS, FAIL, START_TIME

    print("=" * 70)
    print(f"  COMMAND RECOGNITION BENCHMARK — {len(COMMANDS)} commands")
    print("=" * 70)

    from nlp.command_normalizer import command_normalizer

    START_TIME = time.time()

    # Track per-category stats
    categories: Dict[str, Dict] = {}
    for spoken, expected, cls in COMMANDS:
        cat = cls
        if cat not in categories:
            categories[cat] = {"total": 0, "pass": 0, "fail": 0}

        categories[cat]["total"] += 1
        result = command_normalizer.normalize(spoken)

        if result == expected:
            PASS += 1
            categories[cat]["pass"] += 1
        else:
            FAIL += 1
            categories[cat]["fail"] += 1
            FAILURES.append((spoken, expected, result, cat))

    elapsed = time.time() - START_TIME
    accuracy = PASS / max(len(COMMANDS), 1) * 100

    # ── Summary ──
    print()
    print(f"  TOTAL:     {len(COMMANDS)} commands")
    print(f"  PASSED:    {PASS}")
    print(f"  FAILED:    {FAIL}")
    print(f"  ACCURACY:  {accuracy:.2f}%")
    print(f"  LATENCY:   {elapsed / max(len(COMMANDS), 1) * 1000:.3f} ms/command")
    print()

    # ── Per-category ──
    print("  ── BY CATEGORY ──")
    for cat, stats in sorted(categories.items()):
        pct = stats["pass"] / max(stats["total"], 1) * 100
        print(f"    {cat:<20} {stats['pass']:>4}/{stats['total']:<4} ({pct:>6.2f}%)")

    # ── Failures ──
    if FAILURES:
        print()
        print(f"  ── {len(FAILURES)} FAILURES (first 30) ──")
        for spoken, expected, result, cat in FAILURES[:30]:
            print(f"    ✗ '{spoken}'")
            print(f"      expected: '{expected}'")
            print(f"      got:      '{result}'")
            print(f"      category: {cat}")

    print()
    print("=" * 70)
    if FAIL == 0:
        print(f"  ✓ ALL {len(COMMANDS)} COMMANDS RECOGNIZED CORRECTLY")
    else:
        print(f"  ✗ {FAIL} COMMANDS FAILED")
    print("=" * 70)

    return FAIL == 0


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)