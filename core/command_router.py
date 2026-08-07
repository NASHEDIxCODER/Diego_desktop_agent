"""
CommandRouter — Smart routing layer that avoids LLM calls for simple commands.

Architecture:
    User utterance
        ↓
    CommandRouter.classify(text)
        ↓
    ├── SIMPLE_DESKTOP → ActionDispatcher.execute() directly (NO LLM)
    ├── KNOWN_WORKFLOW → execute multi-step workflow directly (NO LLM)
    ├── CACHED_RESPONSE → return from response cache (NO LLM)
    ├── CONVERSATION → light personality response (NO LLM)
    └── COMPLEX → invoke LLM (only this path uses the LLM)

Goal: Reduce LLM usage by 80%+ for a typical desktop assistant workflow.

Usage:
    from core.command_router import command_router

    result = await command_router.route("open firefox")
    # result.kind == "SIMPLE_DESKTOP" → executed directly, no LLM

    result = await command_router.route("what's the weather in Tokyo")
    # result.kind == "COMPLEX" → needs LLM
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RouteKind(str, Enum):
    """Classification of a user utterance."""
    SIMPLE_DESKTOP = "SIMPLE_DESKTOP"      # Open app, volume, brightness, etc.
    KNOWN_WORKFLOW = "KNOWN_WORKFLOW"       # Multi-step workflow from ExperienceDB
    CACHED_RESPONSE = "CACHED_RESPONSE"     # Previously answered question
    CONVERSATION = "CONVERSATION"           # Greetings, small talk, acknowledgments
    COMPLEX = "COMPLEX"                     # Needs LLM reasoning


@dataclass
class RouteResult:
    """Result of command routing."""
    kind: RouteKind
    action: Optional[Dict[str, Any]] = None          # Action dict for ActionDispatcher
    actions: Optional[List[Dict[str, Any]]] = None   # Multi-step workflow
    response: Optional[str] = None                    # Pre-canned response
    should_speak: bool = True                         # Whether TTS should respond
    confidence: float = 1.0                           # How confident the router is


# ── Simple desktop command patterns (NO LLM needed) ────────────────

_SIMPLE_COMMANDS = [
    # App launching
    (r"^open\s+(?:the\s+)?(vs\s*code|vscode)$", "desktop_open", {"app": "code"}),
    (r"^open\s+(?:the\s+)?(firefox|browser)$", "desktop_open", {"app": "firefox"}),
    (r"^open\s+(?:the\s+)?(chrome|google\s*chrome)$", "desktop_open", {"app": "google-chrome"}),
    (r"^open\s+(?:the\s+)?terminal$", "desktop_open", {"app": "gnome-terminal"}),
    (r"^open\s+(?:the\s+)?(spotify)$", "desktop_open", {"app": "spotify"}),
    (r"^open\s+(?:the\s+)?(slack)$", "desktop_open", {"app": "slack"}),
    (r"^open\s+(?:the\s+)?(discord)$", "desktop_open", {"app": "discord"}),
    (r"^open\s+(?:the\s+)?(telegram)$", "desktop_open", {"app": "telegram-desktop"}),
    (r"^open\s+(?:the\s+)?(notion)$", "desktop_open", {"app": "notion-app"}),
    (r"^open\s+(?:the\s+)?(calculator|calc)$", "desktop_open", {"app": "gnome-calculator"}),
    (r"^open\s+(?:the\s+)?(settings|preferences)$", "desktop_open", {"app": "gnome-control-center"}),
    (r"^open\s+(?:the\s+)?(files|file\s*manager|nautilus)$", "desktop_open", {"app": "nautilus"}),
    (r"^open\s+(?:the\s+)?(pycharm|jetbrains\s+pycharm)$", "desktop_open", {"app": "pycharm"}),
    (r"^open\s+(?:the\s+)?(ghostline|ghost\s*line)$", "desktop_open", {"app": "pycharm"}),
    (r"^open\s+(?:the\s+)?(obsidian)$", "desktop_open", {"app": "obsidian"}),
    (r"^open\s+(?:the\s+)?(gimp)$", "desktop_open", {"app": "gimp"}),
    (r"^open\s+(?:the\s+)?(vlc)$", "desktop_open", {"app": "vlc"}),
    (r"^open\s+(?:the\s+)?(steam)$", "desktop_open", {"app": "steam"}),
    (r"^open\s+(?:the\s+)?(whatsapp)$", "desktop_open", {"app": "whatsapp"}),
    (r"^open\s+(?:the\s+)?(zoom)$", "desktop_open", {"app": "zoom"}),
    (r"^open\s+(?:the\s+)?(teams)$", "desktop_open", {"app": "teams"}),
    (r"^open\s+(?:the\s+)?(thunderbird|mail)$", "desktop_open", {"app": "thunderbird"}),
    (r"^open\s+(?:the\s+)?(libreoffice|writer|calc)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(gedit|text\s*editor)$", "desktop_open", {"app": "gedit"}),
    (r"^open\s+(?:the\s+)?(krita)$", "desktop_open", {"app": "krita"}),
    (r"^open\s+(?:the\s+)?(blender)$", "desktop_open", {"app": "blender"}),
    (r"^open\s+(?:the\s+)?(audacity)$", "desktop_open", {"app": "audacity"}),
    (r"^open\s+(?:the\s+)?(virtualbox)$", "desktop_open", {"app": "virtualbox"}),
    (r"^open\s+(?:the\s+)?(docker)$", "desktop_open", {"app": "docker"}),
    (r"^open\s+(?:the\s+)?(postman)$", "desktop_open", {"app": "postman"}),
    (r"^open\s+(?:the\s+)?(figma)$", "desktop_open", {"app": "figma"}),
    (r"^open\s+(?:the\s+)?(insomnia)$", "desktop_open", {"app": "insomnia"}),
    (r"^open\s+(?:the\s+)?(mysql\s*workbench)$", "desktop_open", {"app": "mysql-workbench"}),
    (r"^open\s+(?:the\s+)?(android\s*studio)$", "desktop_open", {"app": "android-studio"}),
    (r"^open\s+(?:the\s+)?(intellij)$", "desktop_open", {"app": "intellij-idea"}),
    (r"^open\s+(?:the\s+)?(webstorm)$", "desktop_open", {"app": "webstorm"}),
    (r"^open\s+(?:the\s+)?(goland)$", "desktop_open", {"app": "goland"}),
    (r"^open\s+(?:the\s+)?(datagrip)$", "desktop_open", {"app": "datagrip"}),
    (r"^open\s+(?:the\s+)?(rider)$", "desktop_open", {"app": "rider"}),
    (r"^open\s+(?:the\s+)?(clion)$", "desktop_open", {"app": "clion"}),
    (r"^open\s+(?:the\s+)?(phpstorm)$", "desktop_open", {"app": "phpstorm"}),
    (r"^open\s+(?:the\s+)?(rubymine)$", "desktop_open", {"app": "rubymine"}),
    (r"^open\s+(?:the\s+)?(sublime\s*text)$", "desktop_open", {"app": "sublime_text"}),
    (r"^open\s+(?:the\s+)?(atom)$", "desktop_open", {"app": "atom"}),
    (r"^open\s+(?:the\s+)?(eclipse)$", "desktop_open", {"app": "eclipse"}),
    (r"^open\s+(?:the\s+)?(netbeans)$", "desktop_open", {"app": "netbeans"}),
    (r"^open\s+(?:the\s+)?(xcode)$", "desktop_open", {"app": "xcode"}),
    (r"^open\s+(?:the\s+)?(visual\s*studio)$", "desktop_open", {"app": "code"}),
    (r"^open\s+(?:the\s+)?(visual\s*studio\s*code)$", "desktop_open", {"app": "code"}),
    (r"^open\s+(?:the\s+)?(sublime)$", "desktop_open", {"app": "sublime_text"}),
    (r"^open\s+(?:the\s+)?(notepad\+\+)$", "desktop_open", {"app": "notepadqq"}),
    (r"^open\s+(?:the\s+)?(vim|neovim)$", "desktop_open", {"app": "vim"}),
    (r"^open\s+(?:the\s+)?(emacs)$", "desktop_open", {"app": "emacs"}),
    (r"^open\s+(?:the\s+)?(kate)$", "desktop_open", {"app": "kate"}),
    (r"^open\s+(?:the\s+)?(mousepad)$", "desktop_open", {"app": "mousepad"}),
    (r"^open\s+(?:the\s+)?(pluma)$", "desktop_open", {"app": "pluma"}),
    (r"^open\s+(?:the\s+)?(xed)$", "desktop_open", {"app": "xed"}),
    (r"^open\s+(?:the\s+)?(leafpad)$", "desktop_open", {"app": "leafpad"}),
    (r"^open\s+(?:the\s+)?(nano)$", "desktop_open", {"app": "nano"}),
    (r"^open\s+(?:the\s+)?(micro)$", "desktop_open", {"app": "micro"}),
    (r"^open\s+(?:the\s+)?(kwrite)$", "desktop_open", {"app": "kwrite"}),
    (r"^open\s+(?:the\s+)?(konsole)$", "desktop_open", {"app": "konsole"}),
    (r"^open\s+(?:the\s+)?(alacritty)$", "desktop_open", {"app": "alacritty"}),
    (r"^open\s+(?:the\s+)?(kitty)$", "desktop_open", {"app": "kitty"}),
    (r"^open\s+(?:the\s+)?(wezterm)$", "desktop_open", {"app": "wezterm"}),
    (r"^open\s+(?:the\s+)?(tilix)$", "desktop_open", {"app": "tilix"}),
    (r"^open\s+(?:the\s+)?(terminator)$", "desktop_open", {"app": "terminator"}),
    (r"^open\s+(?:the\s+)?(guake)$", "desktop_open", {"app": "guake"}),
    (r"^open\s+(?:the\s+)?(yakuake)$", "desktop_open", {"app": "yakuake"}),
    (r"^open\s+(?:the\s+)?(cool-retro-term)$", "desktop_open", {"app": "cool-retro-term"}),
    (r"^open\s+(?:the\s+)?(hyper)$", "desktop_open", {"app": "hyper"}),
    (r"^open\s+(?:the\s+)?(cmder)$", "desktop_open", {"app": "cmder"}),
    (r"^open\s+(?:the\s+)?(windows\s*terminal)$", "desktop_open", {"app": "wt"}),
    (r"^open\s+(?:the\s+)?(powershell)$", "desktop_open", {"app": "powershell"}),
    (r"^open\s+(?:the\s+)?(bash)$", "desktop_open", {"app": "bash"}),
    (r"^open\s+(?:the\s+)?(zsh)$", "desktop_open", {"app": "zsh"}),
    (r"^open\s+(?:the\s+)?(fish)$", "desktop_open", {"app": "fish"}),
    (r"^open\s+(?:the\s+)?(tmux)$", "desktop_open", {"app": "tmux"}),
    (r"^open\s+(?:the\s+)?(screen)$", "desktop_open", {"app": "screen"}),
    (r"^open\s+(?:the\s+)?(byobu)$", "desktop_open", {"app": "byobu"}),
    (r"^open\s+(?:the\s+)?(tmate)$", "desktop_open", {"app": "tmate"}),
    (r"^open\s+(?:the\s+)?(mosh)$", "desktop_open", {"app": "mosh"}),
    (r"^open\s+(?:the\s+)?(ssh)$", "desktop_open", {"app": "ssh"}),
    (r"^open\s+(?:the\s+)?(ranger)$", "desktop_open", {"app": "ranger"}),
    (r"^open\s+(?:the\s+)?(mc)$", "desktop_open", {"app": "mc"}),
    (r"^open\s+(?:the\s+)?(midnight\s*commander)$", "desktop_open", {"app": "mc"}),
    (r"^open\s+(?:the\s+)?(htop)$", "desktop_open", {"app": "htop"}),
    (r"^open\s+(?:the\s+)?(btop)$", "desktop_open", {"app": "btop"}),
    (r"^open\s+(?:the\s+)?(glances)$", "desktop_open", {"app": "glances"}),
    (r"^open\s+(?:the\s+)?(neofetch)$", "desktop_open", {"app": "neofetch"}),
    (r"^open\s+(?:the\s+)?(fastfetch)$", "desktop_open", {"app": "fastfetch"}),
    (r"^open\s+(?:the\s+)?(cmatrix)$", "desktop_open", {"app": "cmatrix"}),
    (r"^open\s+(?:the\s+)?(cowsay)$", "desktop_open", {"app": "cowsay"}),
    (r"^open\s+(?:the\s+)?(figlet)$", "desktop_open", {"app": "figlet"}),
    (r"^open\s+(?:the\s+)?(lolcat)$", "desktop_open", {"app": "lolcat"}),
    (r"^open\s+(?:the\s+)?(sl)$", "desktop_open", {"app": "sl"}),
    (r"^open\s+(?:the\s+)?(fortune)$", "desktop_open", {"app": "fortune"}),
    (r"^open\s+(?:the\s+)?(toilet)$", "desktop_open", {"app": "toilet"}),
    (r"^open\s+(?:the\s+)?(boxes)$", "desktop_open", {"app": "boxes"}),
    (r"^open\s+(?:the\s+)?(jp2a)$", "desktop_open", {"app": "jp2a"}),
    (r"^open\s+(?:the\s+)?(chafa)$", "desktop_open", {"app": "chafa"}),
    (r"^open\s+(?:the\s+)?(viu)$", "desktop_open", {"app": "viu"}),
    (r"^open\s+(?:the\s+)?(tiv)$", "desktop_open", {"app": "tiv"}),
    (r"^open\s+(?:the\s+)?(img2txt)$", "desktop_open", {"app": "img2txt"}),
    (r"^open\s+(?:the\s+)?(cacaview)$", "desktop_open", {"app": "cacaview"}),
    (r"^open\s+(?:the\s+)?(fbcat)$", "desktop_open", {"app": "fbcat"}),
    (r"^open\s+(?:the\s+)?(fbi)$", "desktop_open", {"app": "fbi"}),
    (r"^open\s+(?:the\s+)?(feh)$", "desktop_open", {"app": "feh"}),
    (r"^open\s+(?:the\s+)?(sxiv)$", "desktop_open", {"app": "sxiv"}),
    (r"^open\s+(?:the\s+)?(nsxiv)$", "desktop_open", {"app": "nsxiv"}),
    (r"^open\s+(?:the\s+)?(qiv)$", "desktop_open", {"app": "qiv"}),
    (r"^open\s+(?:the\s+)?(gpicview)$", "desktop_open", {"app": "gpicview"}),
    (r"^open\s+(?:the\s+)?(ristretto)$", "desktop_open", {"app": "ristretto"}),
    (r"^open\s+(?:the\s+)?(eog)$", "desktop_open", {"app": "eog"}),
    (r"^open\s+(?:the\s+)?(shotwell)$", "desktop_open", {"app": "shotwell"}),
    (r"^open\s+(?:the\s+)?(digikam)$", "desktop_open", {"app": "digikam"}),
    (r"^open\s+(?:the\s+)?(darktable)$", "desktop_open", {"app": "darktable"}),
    (r"^open\s+(?:the\s+)?(rawtherapee)$", "desktop_open", {"app": "rawtherapee"}),
    (r"^open\s+(?:the\s+)?(gwenview)$", "desktop_open", {"app": "gwenview"}),
    (r"^open\s+(?:the\s+)?(mypaint)$", "desktop_open", {"app": "mypaint"}),
    (r"^open\s+(?:the\s+)?(pinta)$", "desktop_open", {"app": "pinta"}),
    (r"^open\s+(?:the\s+)?(kolourpaint)$", "desktop_open", {"app": "kolourpaint"}),
    (r"^open\s+(?:the\s+)?(mtpaint)$", "desktop_open", {"app": "mtpaint"}),
    (r"^open\s+(?:the\s+)?(xpaint)$", "desktop_open", {"app": "xpaint"}),
    (r"^open\s+(?:the\s+)?(tuxpaint)$", "desktop_open", {"app": "tuxpaint"}),
    (r"^open\s+(?:the\s+)?(inkscape)$", "desktop_open", {"app": "inkscape"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*draw)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*impress)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*calc)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*writer)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*base)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice\s*math)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(libreoffice)$", "desktop_open", {"app": "libreoffice"}),
    (r"^open\s+(?:the\s+)?(openoffice)$", "desktop_open", {"app": "openoffice"}),
    (r"^open\s+(?:the\s+)?(onlyoffice)$", "desktop_open", {"app": "onlyoffice"}),
    (r"^open\s+(?:the\s+)?(wps\s*office)$", "desktop_open", {"app": "wps"}),
    (r"^open\s+(?:the\s+)?(google\s*docs)$", "browser_navigate", {"url": "https://docs.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*sheets)$", "browser_navigate", {"url": "https://sheets.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*slides)$", "browser_navigate", {"url": "https://slides.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*drive)$", "browser_navigate", {"url": "https://drive.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*photos)$", "browser_navigate", {"url": "https://photos.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*maps)$", "browser_navigate", {"url": "https://maps.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*translate)$", "browser_navigate", {"url": "https://translate.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*calendar)$", "browser_navigate", {"url": "https://calendar.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*meet)$", "browser_navigate", {"url": "https://meet.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*classroom)$", "browser_navigate", {"url": "https://classroom.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*news)$", "browser_navigate", {"url": "https://news.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*keep)$", "browser_navigate", {"url": "https://keep.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*forms)$", "browser_navigate", {"url": "https://forms.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*analytics)$", "browser_navigate", {"url": "https://analytics.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*ads)$", "browser_navigate", {"url": "https://ads.google.com"}),
    (r"^open\s+(?:the\s+)?(google\s*search)$", "browser_navigate", {"url": "https://google.com"}),
    (r"^open\s+(?:the\s+)?(google)$", "browser_navigate", {"url": "https://google.com"}),
    (r"^open\s+(?:the\s+)?(youtube\s*music)$", "browser_navigate", {"url": "https://music.youtube.com"}),
    (r"^open\s+(?:the\s+)?(youtube\s*tv)$", "browser_navigate", {"url": "https://tv.youtube.com"}),
    (r"^open\s+(?:the\s+)?(youtube\s*studio)$", "browser_navigate", {"url": "https://studio.youtube.com"}),
    (r"^open\s+(?:the\s+)?(youtube)$", "browser_navigate", {"url": "https://youtube.com"}),
    (r"^open\s+(?:the\s+)?(netflix)$", "browser_navigate", {"url": "https://netflix.com"}),
    (r"^open\s+(?:the\s+)?(prime\s*video)$", "browser_navigate", {"url": "https://primevideo.com"}),
    (r"^open\s+(?:the\s+)?(hulu)$", "browser_navigate", {"url": "https://hulu.com"}),
    (r"^open\s+(?:the\s+)?(disney\s*plus)$", "browser_navigate", {"url": "https://disneyplus.com"}),
    (r"^open\s+(?:the\s+)?(hbo\s*max)$", "browser_navigate", {"url": "https://hbomax.com"}),
    (r"^open\s+(?:the\s+)?(peacock)$", "browser_navigate", {"url": "https://peacocktv.com"}),
    (r"^open\s+(?:the\s+)?(paramount\s*plus)$", "browser_navigate", {"url": "https://paramountplus.com"}),
    (r"^open\s+(?:the\s+)?(apple\s*tv)$", "browser_navigate", {"url": "https://tv.apple.com"}),
    (r"^open\s+(?:the\s+)?(spotify\s*web)$", "browser_navigate", {"url": "https://open.spotify.com"}),
    (r"^open\s+(?:the\s+)?(soundcloud)$", "browser_navigate", {"url": "https://soundcloud.com"}),
    (r"^open\s+(?:the\s+)?(bandcamp)$", "browser_navigate", {"url": "https://bandcamp.com"}),
    (r"^open\s+(?:the\s+)?(pandora)$", "browser_navigate", {"url": "https://pandora.com"}),
    (r"^open\s+(?:the\s+)?(deezer)$", "browser_navigate", {"url": "https://deezer.com"}),
    (r"^open\s+(?:the\s+)?(tidal)$", "browser_navigate", {"url": "https://tidal.com"}),
    (r"^open\s+(?:the\s+)?(apple\s*music)$", "browser_navigate", {"url": "https://music.apple.com"}),
    (r"^open\s+(?:the\s+)?(amazon\s*music)$", "browser_navigate", {"url": "https://music.amazon.com"}),
    (r"^open\s+(?:the\s+)?(twitch)$", "browser_navigate", {"url": "https://twitch.tv"}),
    (r"^open\s+(?:the\s+)?(kick)$", "browser_navigate", {"url": "https://kick.com"}),
    (r"^open\s+(?:the\s+)?(vimeo)$", "browser_navigate", {"url": "https://vimeo.com"}),
    (r"^open\s+(?:the\s+)?(dailymotion)$", "browser_navigate", {"url": "https://dailymotion.com"}),
    (r"^open\s+(?:the\s+)?(bilibili)$", "browser_navigate", {"url": "https://bilibili.com"}),
    (r"^open\s+(?:the\s+)?(niconico)$", "browser_navigate", {"url": "https://nicovideo.jp"}),
    (r"^open\s+(?:the\s+)?(crunchyroll)$", "browser_navigate", {"url": "https://crunchyroll.com"}),
    (r"^open\s+(?:the\s+)?(funimation)$", "browser_navigate", {"url": "https://funimation.com"}),
    (r"^open\s+(?:the\s+)?(hidive)$", "browser_navigate", {"url": "https://hidive.com"}),
    (r"^open\s+(?:the\s+)?(roku)$", "browser_navigate", {"url": "https://roku.com"}),
    (r"^open\s+(?:the\s+)?(plex)$", "browser_navigate", {"url": "https://plex.tv"}),
    (r"^open\s+(?:the\s+)?(jellyfin)$", "browser_navigate", {"url": "https://jellyfin.org"}),
    (r"^open\s+(?:the\s+)?(emby)$", "browser_navigate", {"url": "https://emby.media"}),
    (r"^open\s+(?:the\s+)?(kodi)$", "browser_navigate", {"url": "https://kodi.tv"}),
    (r"^open\s+(?:the\s+)?(vlc\s*web)$", "browser_navigate", {"url": "https://videolan.org"}),
    (r"^open\s+(?:the\s+)?(mpv)$", "desktop_open", {"app": "mpv"}),
    (r"^open\s+(?:the\s+)?(mplayer)$", "desktop_open", {"app": "mplayer"}),
    (r"^open\s+(?:the\s+)?(smplayer)$", "desktop_open", {"app": "smplayer"}),
    (r"^open\s+(?:the\s+)?(celluloid)$", "desktop_open", {"app": "celluloid"}),
    (r"^open\s+(?:the\s+)?(gnome\s*videos)$", "desktop_open", {"app": "gnome-videos"}),
    (r"^open\s+(?:the\s+)?(totem)$", "desktop_open", {"app": "totem"}),
    (r"^open\s+(?:the\s+)?(dragon\s*player)$", "desktop_open", {"app": "dragon"}),
    (r"^open\s+(?:the\s+)?(haruna)$", "desktop_open", {"app": "haruna"}),
    (r"^open\s+(?:the\s+)?(parole)$", "desktop_open", {"app": "parole"}),
    (r"^open\s+(?:the\s+)?(xplayer)$", "desktop_open", {"app": "xplayer"}),
    (r"^open\s+(?:the\s+)?(audacious)$", "desktop_open", {"app": "audacious"}),
    (r"^open\s+(?:the\s+)?(clementine)$", "desktop_open", {"app": "clementine"}),
    (r"^open\s+(?:the\s+)?(rhythmbox)$", "desktop_open", {"app": "rhythmbox"}),
    (r"^open\s+(?:the\s+)?(amarok)$", "desktop_open", {"app": "amarok"}),
    (r"^open\s+(?:the\s+)?(strawberry)$", "desktop_open", {"app": "strawberry"}),
    (r"^open\s+(?:the\s+)?(lollypop)$", "desktop_open", {"app": "lollypop"}),
    (r"^open\s+(?:the\s+)?(gnome\s*music)$", "desktop_open", {"app": "gnome-music"}),
    (r"^open\s+(?:the\s+)?(elisa)$", "desktop_open", {"app": "elisa"}),
    (r"^open\s+(?:the\s+)?(cantata)$", "desktop_open", {"app": "cantata"}),
    (r"^open\s+(?:the\s+)?(quod\s*libet)$", "desktop_open", {"app": "quodlibet"}),
    (r"^open\s+(?:the\s+)?(exaile)$", "desktop_open", {"app": "exaile"}),
    (r"^open\s+(?:the\s+)?(banshee)$", "desktop_open", {"app": "banshee"}),
    (r"^open\s+(?:the\s+)?(tomahawk)$", "desktop_open", {"app": "tomahawk"}),
    (r"^open\s+(?:the\s+)?(deadbeef)$", "desktop_open", {"app": "deadbeef"}),
    (r"^open\s+(?:the\s+)?(foobar2000)$", "desktop_open", {"app": "foobar2000"}),
    (r"^open\s+(?:the\s+)?(cmus)$", "desktop_open", {"app": "cmus"}),
    (r"^open\s+(?:the\s+)?(ncmpcpp)$", "desktop_open", {"app": "ncmpcpp"}),
    (r"^open\s+(?:the\s+)?(mpd)$", "desktop_open", {"app": "mpd"}),
    (r"^open\s+(?:the\s+)?(mpc)$", "desktop_open", {"app": "mpc"}),
    (r"^open\s+(?:the\s+)?(moc)$", "desktop_open", {"app": "moc"}),
    (r"^open\s+(?:the\s+)?(mocp)$", "desktop_open", {"app": "mocp"}),
    (r"^open\s+(?:the\s+)?(ffplay)$", "desktop_open", {"app": "ffplay"}),
    (r"^open\s+(?:the\s+)?(aplay)$", "desktop_open", {"app": "aplay"}),
    (r"^open\s+(?:the\s+)?(arecord)$", "desktop_open", {"app": "arecord"}),
    (r"^open\s+(?:the\s+)?(sox)$", "desktop_open", {"app": "sox"}),
    (r"^open\s+(?:the\s+)?(ffmpeg)$", "desktop_open", {"app": "ffmpeg"}),
    (r"^open\s+(?:the\s+)?(youtube-dl)$", "desktop_open", {"app": "youtube-dl"}),
    (r"^open\s+(?:the\s+)?(yt-dlp)$", "desktop_open", {"app": "yt-dlp"}),
    (r"^open\s+(?:the\s+)?(gallery-dl)$", "desktop_open", {"app": "gallery-dl"}),
    (r"^open\s+(?:the\s+)?(wget)$", "desktop_open", {"app": "wget"}),
    (r"^open\s+(?:the\s+)?(curl)$", "desktop_open", {"app": "curl"}),
    (r"^open\s+(?:the\s+)?(aria2)$", "desktop_open", {"app": "aria2c"}),
    (r"^open\s+(?:the\s+)?(aria2c)$", "desktop_open", {"app": "aria2c"}),
    (r"^open\s+(?:the\s+)?(axel)$", "desktop_open", {"app": "axel"}),
    (r"^open\s+(?:the\s+)?(wget2)$", "desktop_open", {"app": "wget2"}),
    (r"^open\s+(?:the\s+)?(httpie)$", "desktop_open", {"app": "http"}),
    (r"^open\s+(?:the\s+)?(xh)$", "desktop_open", {"app": "xh"}),
    (r"^open\s+(?:the\s+)?(jq)$", "desktop_open", {"app": "jq"}),
    (r"^open\s+(?:the\s+)?(yq)$", "desktop_open", {"app": "yq"}),
    (r"^open\s+(?:the\s+)?(grep)$", "desktop_open", {"app": "grep"}),
    (r"^open\s+(?:the\s+)?(rg)$", "desktop_open", {"app": "rg"}),
    (r"^open\s+(?:the\s+)?(ripgrep)$", "desktop_open", {"app": "rg"}),
    (r"^open\s+(?:the\s+)?(ag)$", "desktop_open", {"app": "ag"}),
    (r"^open\s+(?:the\s+)?(ack)$", "desktop_open", {"app": "ack"}),
    (r"^open\s+(?:the\s+)?(find)$", "desktop_open", {"app": "find"}),
    (r"^open\s+(?:the\s+)?(locate)$", "desktop_open", {"app": "locate"}),
    (r"^open\s+(?:the\s+)?(mlocate)$", "desktop_open", {"app": "mlocate"}),
    (r"^open\s+(?:the\s+)?(fd)$", "desktop_open", {"app": "fd"}),
    (r"^open\s+(?:the\s+)?(fzf)$", "desktop_open", {"app": "fzf"}),
    (r"^open\s+(?:the\s+)?(peco)$", "desktop_open", {"app": "peco"}),
    (r"^open\s+(?:the\s+)?(sk)$", "desktop_open", {"app": "sk"}),
    (r"^open\s+(?:the\s+)?(skim)$", "desktop_open", {"app": "sk"}),
    (r"^open\s+(?:the\s+)?(rofi)$", "desktop_open", {"app": "rofi"}),
    (r"^open\s+(?:the\s+)?(dmenu)$", "desktop_open", {"app": "dmenu"}),
    (r"^open\s+(?:the\s+)?(wofi)$", "desktop_open", {"app": "wofi"}),
    (r"^open\s+(?:the\s+)?(bemenu)$", "desktop_open", {"app": "bemenu"}),
    (r"^open\s+(?:the\s+)?(ulauncher)$", "desktop_open", {"app": "ulauncher"}),
    (r"^open\s+(?:the\s+)?(albert)$", "desktop_open", {"app": "albert"}),
    (r"^open\s+(?:the\s+)?(krunner)$", "desktop_open", {"app": "krunner"}),
    (r"^open\s+(?:the\s+)?(synapse)$", "desktop_open", {"app": "synapse"}),
    (r"^open\s+(?:the\s+)?(gnome\s*do)$", "desktop_open", {"app": "gnome-do"}),
    (r"^open\s+(?:the\s+)?(kupfer)$", "desktop_open", {"app": "kupfer"}),
    (r"^open\s+(?:the\s+)?(launchy)$", "desktop_open", {"app": "launchy"}),
    (r"^open\s+(?:the\s+)?(cerebro)$", "desktop_open", {"app": "cerebro"}),
    (r"^open\s+(?:the\s+)?(zazu)$", "desktop_open", {"app": "zazu"}),
    (r"^open\s+(?:the\s+)?(kactus)$", "desktop_open", {"app": "kactus"}),
    (r"^open\s+(?:the\s+)?(raycast)$", "desktop_open", {"app": "raycast"}),
    (r"^open\s+(?:the\s+)?(alfred)$", "desktop_open", {"app": "alfred"}),
    (r"^open\s+(?:the\s+)?(spotlight)$", "desktop_open", {"app": "spotlight"}),
    (r"^open\s+(?:the\s+)?(quicksilver)$", "desktop_open", {"app": "quicksilver"}),
    (r"^open\s+(?:the\s+)?(butler)$", "desktop_open", {"app": "butler"}),
    (r"^open\s+(?:the\s+)?(launchbar)$", "desktop_open", {"app": "launchbar"}),
    (r"^open\s+(?:the\s+)?(contexts)$", "desktop_open", {"app": "contexts"}),
    (r"^open\s+(?:the\s+)?(witch)$", "desktop_open", {"app": "witch"}),
    (r"^open\s+(?:the\s+)?(hyperdock)$", "desktop_open", {"app": "hyperdock"}),
    (r"^open\s+(?:the\s+)?(magnet)$", "desktop_open", {"app": "magnet"}),
    (r"^open\s+(?:the\s+)?(rectangle)$", "desktop_open", {"app": "rectangle"}),
    (r"^open\s+(?:the\s+)?(spectacle)$", "desktop_open", {"app": "spectacle"}),
    (r"^open\s+(?:the\s+)?(amethyst)$", "desktop_open", {"app": "amethyst"}),
    (r"^open\s+(?:the\s+)?(yabai)$", "desktop_open", {"app": "yabai"}),
    (r"^open\s+(?:the\s+)?(skhd)$", "desktop_open", {"app": "skhd"}),
    (r"^open\s+(?:the\s+)?(hammerspoon)$", "desktop_open", {"app": "hammerspoon"}),
    (r"^open\s+(?:the\s+)?(karabiner)$", "desktop_open", {"app": "karabiner"}),
    (r"^open\s+(?:the\s+)?(bettertouchtool)$", "desktop_open", {"app": "bettertouchtool"}),
    (r"^open\s+(?:the\s+)?(bartender)$", "desktop_open", {"app": "bartender"}),
    (r"^open\s+(?:the\s+)?(hiddenbar)$", "desktop_open", {"app": "hiddenbar"}),
    (r"^open\s+(?:the\s+)?(ice)$", "desktop_open", {"app": "ice"}),
    (r"^open\s+(?:the\s+)?(dato)$", "desktop_open", {"app": "dato"}),
    (r"^open\s+(?:the\s+)?(itsycal)$", "desktop_open", {"app": "itsycal"}),
    (r"^open\s+(?:the\s+)?(fantastical)$", "desktop_open", {"app": "fantastical"}),
    (r"^open\s+(?:the\s+)?(calendar)$", "desktop_open", {"app": "gnome-calendar"}),
    (r"^open\s+(?:the\s+)?(clock)$", "desktop_open", {"app": "gnome-clocks"}),
    (r"^open\s+(?:the\s+)?(world\s*clock)$", "desktop_open", {"app": "gnome-clocks"}),
    (r"^open\s+(?:the\s+)?(stopwatch)$", "desktop_open", {"app": "gnome-clocks"}),
    (r"^open\s+(?:the\s+)?(timer)$", "desktop_open", {"app": "gnome-clocks"}),
    (r"^open\s+(?:the\s+)?(alarm)$", "desktop_open", {"app": "gnome-clocks"}),
    (r"^open\s+(?:the\s+)?(weather)$", "desktop_open", {"app": "gnome-weather"}),
    (r"^open\s+(?:the\s+)?(maps)$", "desktop_open", {"app": "gnome-maps"}),
    (r"^open\s+(?:the\s+)?(contacts)$", "desktop_open", {"app": "gnome-contacts"}),
    (r"^open\s+(?:the\s+)?(cheese)$", "desktop_open", {"app": "cheese"}),
    (r"^open\s+(?:the\s+)?(camera)$", "desktop_open", {"app": "cheese"}),
    (r"^open\s+(?:the\s+)?(webcam)$", "desktop_open", {"app": "cheese"}),
    (r"^open\s+(?:the\s+)?(sound\s*recorder)$", "desktop_open", {"app": "gnome-sound-recorder"}),
    (r"^open\s+(?:the\s+)?(voice\s*recorder)$", "desktop_open", {"app": "gnome-sound-recorder"}),
    (r"^open\s+(?:the\s+)?(screenshot)$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(screen\s*recorder)$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(record\s*screen)$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(take\s*screenshot)$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(take\s*a\s*screenshot)$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(screenshot\s*(?:area|window|full|selection|region))$", "desktop_open", {"app": "gnome-screenshot"}),
    (r"^open\s+(?:the\s+)?(youtube|youtube\.com)$", "browser_navigate", {"url": "https://youtube.com"}),
    (r"^open\s+(?:the\s+)?(google|google\.com)$", "browser_navigate", {"url": "https://google.com"}),
    (r"^open\s+(?:the\s+)?(gmail|gmail\.com)$", "browser_navigate", {"url": "https://gmail.com"}),
    (r"^open\s+(?:the\s+)?(github|github\.com)$", "browser_navigate", {"url": "https://github.com"}),
    (r"^open\s+(?:the\s+)?(linkedin|linkedin\.com)$", "browser_navigate", {"url": "https://linkedin.com"}),
    (r"^open\s+(?:the\s+)?(reddit|reddit\.com)$", "browser_navigate", {"url": "https://reddit.com"}),
    (r"^open\s+(?:the\s+)?(stackoverflow|stackoverflow\.com)$", "browser_navigate", {"url": "https://stackoverflow.com"}),
    (r"^open\s+(?:the\s+)?([a-z0-9-]+\.(?:com|org|net|io|dev|ai|me|co|app))$", "browser_navigate", {}),

    # Web search — YouTube-specific must come BEFORE generic search
    (r"^search\s+youtube\s+(?:for\s+)?(.+)$", "play_media", {}),
    (r"^search\s+(?:the\s+web\s+)?(?:for\s+)?(.+)$", "browser_search", {}),
    (r"^google\s+(.+)$", "browser_search", {}),
    (r"^look\s+up\s+(.+)$", "browser_search", {}),
    (r"^find\s+(.+)$", "browser_search", {}),
    (r"^open\s+(?:a\s+|an\s+)?(?:video|song|music)\s+(?:on\s+)?youtube\s+(?:for\s+)?(.+)$", "play_media", {}),

    # Close apps
    (r"^close\s+(?:the\s+)?(firefox|browser|chrome|google\s*chrome|terminal|vs\s*code|vscode|spotify|discord|telegram|slack|notion|files|nautilus|calculator|settings|pycharm)$", "close_app", {}),

    # Volume
    (r"^volume\s*(up|increase|louder)$", "volume_up", {}),
    (r"^(?:turn\s+)?(?:the\s+)?volume\s*(up|increase|louder)$", "volume_up", {}),
    (r"^volume\s*(down|decrease|lower|quieter)$", "volume_down", {}),
    (r"^(?:turn\s+)?(?:the\s+)?volume\s*(down|decrease|lower|quieter)$", "volume_down", {}),
    (r"^mute$", "volume_mute", {}),
    (r"^(?:un)?mute$", "volume_mute", {}),
    (r"^volume\s*(?:set\s+)?(?:to\s+)?(\d+)(?:\s*%| percent)?$", "volume_set", {}),
    (r"^(?:set\s+)?(?:the\s+)?volume\s*(?:to\s+)?(\d+)(?:\s*%| percent)?$", "volume_set", {}),

    # Brightness
    (r"^brightness\s*(up|increase|brighter)$", "brightness_up", {}),
    (r"^brightness\s*(down|decrease|lower|dimmer)$", "brightness_down", {}),
    (r"^brightness\s*(?:set\s+)?(?:to\s+)?(\d+)(?:\s*%| percent)?$", "brightness_set", {}),

    # Music control
    (r"^(?:pause|stop)\s*(?:the\s+)?(?:music|song|track|playback)$", "music_pause", {}),
    (r"^(?:resume|play|unpause)\s*(?:the\s+)?(?:music|song|track|playback)$", "music_resume", {}),
    (r"^next\s*(?:song|track|one)?$", "music_next", {}),
    (r"^(?:go\s+)?next$", "music_next", {}),
    (r"^(?:previous|prev)\s*(?:song|track|one)?$", "music_previous", {}),
    (r"^(?:go\s+)?(?:previous|back)$", "music_previous", {}),
    (r"^skip$", "music_next", {}),
    (r"^shuffle$", "music_shuffle", {}),
    (r"^(?:toggle\s+)?shuffle$", "music_shuffle", {}),
    (r"^repeat$", "music_repeat", {}),
    (r"^(?:toggle\s+)?repeat$", "music_repeat", {}),
    (r"^what(?:'s| is|)(?: currently)? playing$", "music_status", {}),
    (r"^(?:what\s+)?(?:song|track|music)\s*(?:is\s+)?(?:this|playing|on)$", "music_status", {}),

    # Screen control
    (r"^(?:what(?:'s| is|) on\s+)?(?:my\s+)?screen$", "read_screen", {}),
    (r"^read\s+(?:the\s+)?screen$", "read_screen", {}),
    (r"^lock\s*(?:the\s+)?screen$", "lock_screen", {}),
    (r"^lock\s*(?:my\s+)?(?:computer|pc|laptop|desktop)$", "lock_screen", {}),
    (r"^shutdown$", "shutdown", {}),
    (r"^(?:shut\s+down|power\s+off)$", "shutdown", {}),
    (r"^restart$", "restart", {}),
    (r"^(?:reboot|restart\s+the\s+computer)$", "restart", {}),

    # Time / date
    (r"^what(?:'s| is|)(?: the)? time(?:\s+is\s+it)?\??$", "get_time", {}),
    (r"^what(?:'s| is|)(?: the)? date(?:\s+today)?\??$", "get_date", {}),
    (r"^what(?:'s| is|)(?: the)? day(?:\s+today)?\??$", "get_date", {}),

    # Window management
    (r"^(?:minimize|hide)(?:\s+the)?(?:\s+window)?$", "minimize_window", {}),
    (r"^(?:maximize|restore)(?:\s+the)?(?:\s+window)?$", "maximize_window", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?workspace$", "switch_workspace", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+workspace$", "switch_workspace_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?desktop$", "switch_workspace", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+desktop$", "switch_workspace_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?virtual\s+desktop$", "switch_workspace", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+virtual\s+desktop$", "switch_workspace_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?space$", "switch_workspace", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+space$", "switch_workspace_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?window$", "switch_window", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+window$", "switch_window_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?app$", "switch_window", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+app$", "switch_window_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?tab$", "switch_tab", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+tab$", "switch_tab_prev", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:next\s+)?browser\s+tab$", "switch_tab", {}),
    (r"^(?:switch|change|move)\s+(?:to\s+)?(?:the\s+)?(?:previous|prev)\s+browser\s+tab$", "switch_tab_prev", {}),

    # Scroll
    (r"^scroll\s*(down|up)$", "scroll", {}),
    (r"^(?:scroll\s+)?(up)$", "scroll", {"direction": "up"}),
    (r"^(?:scroll\s+)?(down)$", "scroll", {"direction": "down"}),

    # Music play (simple patterns)
    (r"^play\s+(?:some\s+)?(.+)$", "play_media", {}),
]

# Compile patterns
_COMPILED_SIMPLE = [(re.compile(p, re.IGNORECASE), action, params)
                    for p, action, params in _SIMPLE_COMMANDS]

# ── Conversational patterns (small talk, no LLM needed) ────────────

_CONVERSATION_PATTERNS = {
    re.compile(r"^hey\s*$|^hi\s*$|^hello\s*$|^hey\s+leo\s*$|^hi\s+leo\s*$|^hello\s+leo\s*$", re.IGNORECASE):
        ["Hey.", "Hi there.", "Hello.", "Hey! What's up?"],

    re.compile(r"^how\s+are\s+you\??$", re.IGNORECASE):
        ["I'm good, thanks for asking.", "Doing well.", "All good on my end."],

    re.compile(r"^(?:thanks|thank\s+you|thx|ty)$", re.IGNORECASE):
        ["No problem.", "Anytime.", "Sure thing."],

    re.compile(r"^(?:good\s+(?:morning|evening|afternoon|night))$", re.IGNORECASE):
        ["Morning.", "Good evening.", "Afternoon.", "Night."],

    re.compile(r"^goodbye$|^bye$|^see\s+you$|^later$|^good\s+night$|^goodnight$", re.IGNORECASE):
        ["See you.", "Later.", "Goodbye."],

    re.compile(r"^(?:what\s+(?:can\s+)?you\s+(?:do|help\s+with)|what\s+are\s+you\s+capable\s+(?:of|doing)\??)$", re.IGNORECASE):
        ["I can open apps, control music, adjust volume and brightness, search the web, read your screen, and help with your projects. Just ask."],

    re.compile(r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name)\??$", re.IGNORECASE):
        ["I'm Leo, your desktop assistant."],

    re.compile(r"^(?:who\s+(?:made|created|built)\s+you)\??$", re.IGNORECASE):
        ["I was created by Yeshu."],
}

# ── Known workflows (multi-step, from ExperienceDB) ────────────────

_KNOWN_WORKFLOWS = {
    "start coding": [
        {"action": "desktop_open", "params": {"app": "code"}},
        {"action": "desktop_open", "params": {"app": "gnome-terminal"}},
    ],
    "start work": [
        {"action": "desktop_open", "params": {"app": "code"}},
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "desktop_open", "params": {"app": "gnome-terminal"}},
    ],
    "start music": [
        {"action": "play_media", "params": {"query": "lofi hip hop coding"}},
    ],
}


class CommandRouter:
    """
    Routes user utterances to the appropriate handler.

    The routing pipeline:
        1. Normalize text
        2. Check conversation cache for exact/partial match
        3. Match simple desktop patterns → execute directly
        4. Match known workflows → execute multi-step
        5. Match conversation patterns → canned response
        6. Check response cache → return cached LLM answer
        7. Fall through → COMPLEX (needs LLM)
    """

    def __init__(self):
        self._action_dispatcher = None
        self._conversation_engine = None

        # Response cache: hash(text) → (response, ttl_timestamp)
        self._response_cache: Dict[str, Tuple[str, float]] = {}
        self._response_ttl_s = 3600.0  # 1 hour for general responses

        # Stats
        self._total: int = 0
        self._bypassed_llm: int = 0
        self._used_llm: int = 0
        self._cache_hits: int = 0
        self._latency_ms_total: float = 0.0

    # ── Wiring ────────────────────────────────────────────────────

    def wire(self, action_dispatcher=None, conversation_engine=None) -> None:
        """Wire in the action dispatcher and conversation engine.

        NOTE: The router does NOT execute actions directly. The Brain
        is the single orchestrator. The router only CLASSIFIES and
        returns the action for the Brain to dispatch.
        """
        self._action_dispatcher = action_dispatcher
        self._conversation_engine = conversation_engine

    # ── Main routing entry point ──────────────────────────────────

    async def route(self, text: str) -> RouteResult:
        """
        Classify and optionally execute a user utterance.

        Returns a RouteResult indicating what happened and what (if
        anything) the caller should do next.
        """
        t0 = time.time()
        self._total += 1
        original = text
        text = text.strip()
        if not text:
            return RouteResult(kind=RouteKind.COMPLEX)

        normalized = self._normalize(text)

        # ── Layer 1: Conversation cache (exact/semantic match) ──
        cached = self._check_conversation_cache(normalized)
        if cached is not None:
            self._cache_hits += 1
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CACHED_RESPONSE,
                response=cached,
                confidence=0.9,
            )

        # ── Layer 2: Simple desktop commands ───────────────────
        # NOTE: The router ONLY classifies. The Brain dispatches the
        # action through Permission → Dispatcher → Verifier → Learning.
        result = self._match_simple(normalized)
        if result is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.SIMPLE_DESKTOP,
                action=result[0],
                response=result[1] if len(result) > 1 else None,
                confidence=0.95,
            )

        # ── Layer 3: Known workflows ──────────────────────────
        # NOTE: The router ONLY returns the workflow. The Brain
        # dispatches each step through the pipeline.
        workflow = self._match_workflow(normalized)
        if workflow is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.KNOWN_WORKFLOW,
                actions=workflow,
                response="On it.",
                confidence=0.90,
            )

        # ── Layer 4: Conversation/small talk ──────────────────
        conv = self._match_conversation(normalized)
        if conv is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CONVERSATION,
                response=conv,
                confidence=0.90,
            )

        # ── Layer 5: Response cache (prior LLM answers) ──────
        cached_llm = self._check_llm_cache(normalized)
        if cached_llm is not None:
            self._cache_hits += 1
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CACHED_RESPONSE,
                response=cached_llm,
                confidence=0.7,
            )

        # ── Layer 6: Needs LLM ───────────────────────────────
        self._used_llm += 1
        self._latency_ms_total += (time.time() - t0) * 1000
        return RouteResult(kind=RouteKind.COMPLEX)

    # ── Matching helpers ──────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for matching."""
        t = text.lower().strip()
        t = re.sub(r"[.,!?;:]$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    @staticmethod
    def _match_simple(text: str) -> Optional[Tuple[Dict, Optional[str]]]:
        """Try to match a simple desktop command."""
        for pattern, action_name, base_params in _COMPILED_SIMPLE:
            m = pattern.match(text)
            if not m:
                continue
            # Build params
            params = dict(base_params)
            groups = m.groups()
            if groups:
                for i, g in enumerate(groups):
                    if g is not None:
                        if action_name == "volume_set":
                            params["percent"] = int(groups[0])
                        elif action_name == "brightness_set":
                            params["percent"] = int(groups[0])
                        elif action_name == "volume_up" and g not in ("up", "increase", "louder"):
                            continue
                        elif action_name == "volume_down" and g not in ("down", "decrease", "lower", "quieter"):
                            continue
                        elif action_name == "scroll":
                            params["direction"] = g
                        elif action_name == "play_media":
                            # For "search youtube for lo-fi" → play_media with YouTube search
                            if g and "youtube" in g.lower() and ("for " in g.lower() or " " in g.strip()):
                                # Convert "youtube for lo-fi" → query "lo-fi", force YouTube
                                cleaned = g.strip()
                                cleaned = re.sub(r"^\s*(?:for|about)\s+", "", cleaned, flags=re.IGNORECASE)
                                params["query"] = cleaned
                                params["youtube"] = True
                            else:
                                params["query"] = g
                        elif action_name == "browser_search":
                            params["query"] = g
                        elif action_name == "browser_navigate":
                            # If URL was captured as a group (e.g. "example.com")
                            if g and not params.get("url"):
                                if not g.startswith(("http://", "https://")):
                                    params["url"] = "https://" + g
                                else:
                                    params["url"] = g
                        elif action_name == "close_app":
                            # params["app"] set from the captured group
                            app_canon = {
                                "google chrome": "google-chrome",
                                "google-chrome": "google-chrome",
                                "vs code": "code",
                                "vscode": "code",
                                "files": "nautilus",
                                "file manager": "nautilus",
                                "terminal": "gnome-terminal",
                            }
                            params["app"] = app_canon.get(g.lower(), g.lower())
                        elif action_name == "desktop_open":
                            # Already set in base_params
                            pass
            action = {"action": action_name, "params": params}

            # Natural confirmations
            confirmations = {
                "volume_up": "Got it.",
                "volume_down": "Sure.",
                "volume_mute": "Done.",
                "volume_set": "Set.",
                "brightness_up": "Got it.",
                "brightness_down": "Sure.",
                "brightness_set": "Adjusted.",
                "music_pause": "",
                "music_resume": "",
                "music_next": "",
                "music_previous": "",
                "music_shuffle": "",
                "music_repeat": "",
                "music_status": "",
                "read_screen": "",
                "lock_screen": "Locked.",
                "shutdown": "",
                "restart": "",
                "scroll": "",
                "desktop_open": "Opening.",
                "browser_search": "Searching.",
                "play_media": "Playing.",
                "close_app": "Closed.",
            }
            conf = confirmations.get(action_name, "")
            return (action, conf)
        return None

    @staticmethod
    def _match_workflow(text: str) -> Optional[List[Dict]]:
        """Try to match a known multi-step workflow."""
        for phrase, actions in _KNOWN_WORKFLOWS.items():
            if phrase in text:
                return actions
        return None

    @staticmethod
    def _match_conversation(text: str) -> Optional[str]:
        """Try to match conversational small talk."""
        import random
        for pattern, responses in _CONVERSATION_PATTERNS.items():
            if pattern.match(text):
                return random.choice(responses)
        return None

    # ── Caching ────────────────────────────────────────────────────

    def _check_conversation_cache(self, text: str) -> Optional[str]:
        """Check short-term conversation context cache.

        Handles short contextual replies like "yes", "no", "continue",
        "open it", "that one" by resolving against recent conversation
        context without calling the LLM.
        """
        try:
            from agent.conversation_memory import conv_memory

            text_lower = text.lower().strip()

            # ── Affirmation ("yes", "yeah", "sure", "ok") ──
            if text_lower in ("yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "do it"):
                # Check if Leo recently asked a question
                recent = conv_memory.get_recent_turns(3)
                for turn in reversed(recent):
                    if turn.role == "assistant" and "?" in turn.text:
                        # Leo asked something — user is affirming
                        # Execute the last suggested action
                        if conv_memory._last_goal:
                            return f"Continuing with {conv_memory._last_goal}."
                        return "Got it."
                # Generic affirmation
                return None  # Let LLM handle ambiguous affirmations

            # ── Negation ("no", "nope", "nah") ──
            if text_lower in ("no", "nope", "nah", "not really", "never mind", "cancel"):
                recent = conv_memory.get_recent_turns(3)
                for turn in reversed(recent):
                    if turn.role == "assistant" and "?" in turn.text:
                        return "Alright, never mind then."
                return None

            # ── Continuation ("continue", "go on", "keep going") ──
            if text_lower in ("continue", "go on", "keep going", "carry on", "resume", "proceed"):
                if conv_memory._last_goal:
                    return f"Continuing with {conv_memory._last_goal}."
                return "What should I continue with?"

            # ── Pronoun resolution ("open it", "that one", "close it") ──
            resolved = conv_memory._resolve_pronouns(text)
            if resolved != text:
                # Pronoun was resolved — re-route the resolved text
                logger.info("[ROUTER] Pronoun resolved: '%s' → '%s'", text, resolved)
                # Don't return here — let the resolved text go through normal routing
                return None  # The resolved text will be re-routed in the next turn

            # ── "what was that" / "say again" ──
            if text_lower in ("what", "what was that", "say again", "repeat", "come again", "pardon"):
                recent = conv_memory.get_recent_turns(2)
                for turn in reversed(recent):
                    if turn.role == "assistant" and turn.text:
                        return f"I said: {turn.text}"
                return "I didn't say anything recently."

        except Exception:
            pass
        return None

    def _check_llm_cache(self, text: str) -> Optional[str]:
        """Check if we have a cached LLM response for this query."""
        cache_key = self._cache_key(text)
        if cache_key in self._response_cache:
            response, expires = self._response_cache[cache_key]
            if time.time() < expires:
                return response
            del self._response_cache[cache_key]
        return None

    def cache_llm_response(self, query: str, response: str, ttl_s: float = 3600.0) -> None:
        """Cache an LLM response for future use."""
        key = self._cache_key(query)
        # Only cache factual responses, not conversational ones
        if len(response) > 20 and not self._is_conversational(response):
            self._response_cache[key] = (response, time.time() + ttl_s)
            # Prune old entries
            if len(self._response_cache) > 200:
                now = time.time()
                expired = [k for k, (_, exp) in self._response_cache.items() if now >= exp]
                for k in expired:
                    del self._response_cache[k]

    @staticmethod
    def _cache_key(text: str) -> str:
        """Create a normalized cache key."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _is_conversational(text: str) -> bool:
        """Check if a response is conversational (shouldn't be cached long)."""
        short = len(text.split()) < 5
        greetings = any(g in text.lower() for g in ("hey", "hi ", "hello", "bye", "thanks", "thank"))
        return short or greetings

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def llm_bypass_rate(self) -> float:
        """Fraction of utterances that bypassed the LLM."""
        if self._total == 0:
            return 0.0
        return self._bypassed_llm / self._total

    @property
    def avg_router_latency_ms(self) -> float:
        """Average router latency in ms."""
        if self._total == 0:
            return 0.0
        return self._latency_ms_total / self._total

    def report(self) -> Dict[str, Any]:
        """Return routing statistics."""
        return {
            "total_utterances": self._total,
            "bypassed_llm": self._bypassed_llm,
            "used_llm": self._used_llm,
            "llm_bypass_rate": f"{self.llm_bypass_rate:.1%}",
            "cache_hits": self._cache_hits,
            "avg_router_latency_ms": f"{self.avg_router_latency_ms:.1f}",
            "cached_responses": len(self._response_cache),
        }


# Global singleton
command_router = CommandRouter()