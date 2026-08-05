"""
SpeechCorrector — Local speech correction layer for Leo.

Builds a dynamic dictionary of known terms from the user's desktop
environment and uses fuzzy matching to correct common Whisper mistakes.
NEVER invokes an LLM — all corrections are local and deterministic.

Correction sources (scanned once at init, refreshable):
  - Installed applications (.desktop files)
  - Project directories and git repositories
  - Browser bookmarks (Chrome, Firefox, Brave, Edge)
  - Music providers and media apps
  - Common folders (Documents, Downloads, Desktop, etc.)
  - Desktop commands and system actions
  - Custom user terms from settings

Fuzzy matching uses RapidFuzz when available (difflib fallback).
Corrections are applied ONLY when confidence is high (≥ threshold).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Fuzzy matching backend ─────────────────────────────────────
try:
    from rapidfuzz import fuzz as _rf_fuzz, process as _rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _rf_fuzz = None
    _rf_process = None
    _HAS_RAPIDFUZZ = False
    logger.debug("[CORRECTOR] rapidfuzz not installed — using difflib fallback")

import difflib


def _fuzzy_ratio(a: str, b: str) -> float:
    """Fuzzy similarity 0..1."""
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.ratio(a.lower(), b.lower()) / 100.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _fuzzy_partial_ratio(a: str, b: str) -> float:
    """Best substring similarity 0..1."""
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.partial_ratio(a.lower(), b.lower()) / 100.0
    # difflib fallback: find best matching substring
    a_lower, b_lower = a.lower(), b.lower()
    if len(a_lower) > len(b_lower):
        a_lower, b_lower = b_lower, a_lower
    best = 0.0
    for i in range(len(b_lower) - len(a_lower) + 1):
        ratio = difflib.SequenceMatcher(None, a_lower, b_lower[i:i + len(a_lower)]).ratio()
        if ratio > best:
            best = ratio
    return best


# ── Known term sources ─────────────────────────────────────────

# Common desktop applications (fallback when .desktop scanning fails)
_FALLBACK_APPS = {
    "firefox", "chrome", "chromium", "brave", "edge", "safari", "opera",
    "terminal", "gnome terminal", "konsole", "alacritty", "kitty", "wezterm",
    "files", "nautilus", "dolphin", "thunar", "pcmanfm", "file explorer",
    "settings", "system settings", "gnome settings", "control panel",
    "calculator", "calendar", "notepad", "gedit", "kate", "nano", "vim", "neovim",
    "vscode", "visual studio code", "code", "pycharm", "intellij", "goland",
    "webstorm", "phpstorm", "clion", "rider", "datagrip", "android studio",
    "sublime", "sublime text", "atom", "brackets", "eclipse", "netbeans",
    "spotify", "rhythmbox", "clementine", "amarok", "audacious", "vlc",
    "mpv", "totem", "kodi", "plex", "jellyfin",
    "slack", "discord", "telegram", "signal", "whatsapp", "messenger",
    "zoom", "teams", "skype", "google meet",
    "libreoffice", "libreoffice writer", "libreoffice calc", "libreoffice impress",
    "onlyoffice", "wps office", "openoffice",
    "gimp", "inkscape", "krita", "blender", "darktable", "digikam",
    "thunderbird", "evolution", "geary", "mailspring",
    "docker", "podman", "virtualbox", "vmware", "qemu",
    "obs", "obs studio", "kdenlive", "shotcut", "openshot", "audacity",
    "steam", "lutris", "heroic", "bottles", "wine",
    "bitwarden", "keepassxc", "authy",
    "timeshift", "deja dup", "backups",
    "htop", "btop", "bashtop", "glances", "neofetch", "fastfetch",
    "system monitor", "task manager", "disk usage analyzer", "baobab",
    "gnome disks", "gparted",
    "software center", "gnome software", "discover", "pamac", "synaptic",
    "update manager",
    "network manager", "bluetooth manager", "power manager",
    "screenshot", "flameshot", "spectacle", "shutter",
    "screen recorder", "peek", "kooha",
    "clipboard manager", "copyq", "diodon", "parcellite",
    "notes", "obsidian", "joplin", "logseq", "notion", "evernote",
    "todo", "todoist", "ticktick", "gnome todo",
    "password manager", "1password", "lastpass",
    "rss reader", "feedly", "newsboat", "liferea",
    "torrent", "transmission", "qbittorrent", "deluge",
    "ftp", "filezilla",
    "remote desktop", "remmina", "vinagre", "anydesk", "teamviewer",
    "postman", "insomnia", "bruno",
    "dbeaver", "pgadmin", "mysql workbench", "sqlitebrowser",
    "wireshark", "nmap", "zenmap",
    "gitkraken", "sourcetree", "gitg", "qgit",
    "figma", "penpot", "lunacy",
    "discord canary", "discord ptb",
    "caprine", "ferdium", "rambox", "franz",
    "whatsapp desktop", "telegram desktop",
    "youtube", "youtube music", "netflix", "prime video", "disney plus",
    "hotstar", "hulu", "hbo max", "apple tv", "peacock", "paramount plus",
    "twitch", "kick",
    "google chrome", "mozilla firefox", "microsoft edge", "brave browser",
    "opera browser", "vivaldi", "waterfox", "librewolf", "tor browser",
    "google", "google drive", "google docs", "google sheets", "google slides",
    "google calendar", "google keep", "google photos", "google maps",
    "gmail", "outlook", "yahoo mail", "protonmail",
    "github", "gitlab", "bitbucket", "azure devops",
    "stack overflow", "reddit", "twitter", "x", "facebook", "instagram",
    "linkedin", "pinterest", "tumblr", "snapchat", "tiktok",
    "amazon", "flipkart", "ebay", "aliexpress", "shopify",
    "swiggy", "zomato", "ubereats", "doordash", "grubhub",
    "uber", "ola", "lyft", "rapido",
    "airbnb", "booking.com", "makemytrip", "expedia",
    "paypal", "google pay", "phonepe", "paytm", "venmo", "cash app",
    "notion calendar", "cron calendar", "google tasks",
    "linear", "jira", "trello", "asana", "monday", "clickup", "basecamp",
    "confluence", "notion wiki", "slite",
    "figma", "sketch", "adobe xd", "invision", "zeplin",
    "canva", "piktochart", "visme",
    "miro", "mural", "lucidchart", "draw.io", "excalidraw",
    "vercel", "netlify", "heroku", "railway", "render", "fly.io",
    "aws", "amazon web services", "google cloud", "azure", "digitalocean",
    "cloudflare", "fastly", "akamai",
    "stripe", "lemonsqueezy", "paddle", "gumroad",
    "sentry", "datadog", "new relic", "grafana", "prometheus",
    "launchdarkly", "split.io", "flagsmith",
    "twilio", "sendgrid", "mailgun", "resend", "plunk",
    "openai", "chatgpt", "claude", "anthropic", "gemini", "bard",
    "copilot", "github copilot", "cursor", "windsurf", "cline",
    "hugging face", "replicate", "together ai", "groq", "perplexity",
    "ollama", "lm studio", "jan ai", "gpt4all",
    "langchain", "llamaindex", "crewai", "autogen",
    "stable diffusion", "midjourney", "dall-e", "comfyui",
    "python", "javascript", "typescript", "rust", "go", "golang",
    "java", "kotlin", "swift", "c++", "c#", "ruby", "php", "scala",
    "elixir", "clojure", "haskell", "lua", "zig", "nim", "crystal",
    "react", "vue", "angular", "svelte", "next.js", "nuxt", "remix",
    "node.js", "deno", "bun",
    "django", "flask", "fastapi", "spring", "rails", "laravel",
    "express", "nest.js", "gin", "echo", "fiber",
    "postgresql", "mysql", "mariadb", "sqlite", "mongodb", "redis",
    "cassandra", "dynamodb", "supabase", "firebase", "planetscale",
    "neo4j", "elasticsearch", "meilisearch", "typesense",
    "kafka", "rabbitmq", "nats", "pulsar", "sqs", "pubsub",
    "docker compose", "kubernetes", "helm", "terraform", "pulumi",
    "ansible", "chef", "puppet", "saltstack",
    "nginx", "apache", "caddy", "traefik", "haproxy", "envoy",
    "github actions", "gitlab ci", "circleci", "jenkins", "drone",
    "argocd", "flux", "spinnaker",
    "prometheus", "grafana", "loki", "tempo", "mimir",
    "jaeger", "zipkin", "opentelemetry",
    "istio", "linkerd", "consul", "vault", "nomad",
    "linux", "ubuntu", "debian", "fedora", "arch", "manjaro", "pop os",
    "elementary os", "zorin os", "linux mint", "kali", "parrot",
    "raspberry pi", "arduino", "esp32",
    "bash", "zsh", "fish", "powershell", "nushell",
    "git", "make", "cmake", "bazel", "gradle", "maven", "sbt",
    "npm", "yarn", "pnpm", "bun", "pip", "cargo", "go mod",
    "homebrew", "chocolatey", "scoop", "winget", "snap", "flatpak", "appimage",
    "systemd", "journald", "cron", "systemd timers",
    "ssh", "scp", "rsync", "curl", "wget", "httpie",
    "tmux", "screen", "byobu",
    "fzf", "ripgrep", "fd", "bat", "exa", "eza", "lsd", "delta",
    "jq", "yq", "fx", "gron",
    "tldr", "cheat", "navi",
    "lazygit", "lazydocker", "lazynpm",
    "yazi", "ranger", "nnn", "lf", "vifm",
    "zoxide", "autojump", "fasd",
    "starship", "powerlevel10k", "oh my zsh", "oh my posh",
    "kitty terminal", "wezterm terminal", "alacritty terminal",
    "i3", "sway", "hyprland", "bspwm", "awesome", "qtile", "dwm",
    "gnome", "kde", "plasma", "xfce", "cinnamon", "mate", "budgie",
    "pantheon", "deepin", "lxqt", "lxde", "enlightenment",
    "wayland", "x11", "xorg",
    "pipewire", "pulseaudio", "jack", "alsa",
    "bluetooth", "wifi", "ethernet", "vpn", "wireguard", "openvpn",
    "tailscale", "zerotier", "nebula",
    "ufw", "firewalld", "iptables", "nftables",
    "selinux", "apparmor",
    "flatpak", "snap", "appimage", "nix", "guix",
    "distrobox", "toolbox", "podman compose",
    "activitywatch", "rescuetime", "toggl", "clockify",
    "anki", "remnote", "roam research", "mem.ai",
    "zotero", "mendeley", "paperpile", "readwise",
    "calibre", "kobo", "kindle",
    "signal desktop", "element", "matrix", "session", "wire",
    "threema", "briar", "jami", "tox",
    "monero", "bitcoin", "ethereum", "metamask", "phantom",
    "ledger live", "trezor suite", "exodus",
    "raspberry pi imager", "balena etcher", "ventoy", "rufus",
    "cpu-x", "gpu-z", "hardinfo", "lshw",
    "stacer", "bleachbit", "sweeper",
    "warp", "tabby", "hyper", "terminator", "tilix", "guake",
    "yakuake", "tilda", "cool retro term",
    "ulauncher", "albert", "rofi", "wofi", "dmenu", "fuzzel",
    "picom", "compton", "dunst", "mako", "waybar", "polybar",
    "eww", "conky", "ags",
    "sddm", "gdm", "lightdm", "ly", "lemurs",
    "grub", "systemd-boot", "refind", "clover",
    "ventura", "sonoma", "sequoia", "macos", "windows 11", "windows 10",
}

# Common folder names
_COMMON_FOLDERS = {
    "desktop", "documents", "downloads", "music", "pictures", "videos",
    "templates", "public", "home", "root", "opt", "var", "etc", "tmp",
    "usr", "bin", "sbin", "lib", "lib64", "share", "local", "include",
    "src", "build", "dist", "target", "node_modules", "vendor",
    ".config", ".local", ".cache", ".ssh", ".gnupg", ".docker",
    ".git", ".github", ".vscode", ".idea", ".cursor",
    "projects", "workspace", "dev", "development", "code", "repos",
    "screenshots", "wallpapers", "icons", "fonts", "themes",
    "backups", "archives", "iso", "mount", "media",
    "snap", "flatpak", "appimages",
    "logs", "crashes", "dumps", "traces",
    "python", "node", "rust", "go", "java", "ruby",
    "venv", ".venv", "env", ".env", "virtualenv",
    "docker", "containers", "volumes", "compose",
    "kubernetes", "k8s", "helm", "charts",
    "terraform", "ansible", "puppet", "chef",
    "notebooks", "datasets", "models", "checkpoints",
    "plugins", "extensions", "addons",
    "config", "configuration", "settings",
    "data", "database", "db", "storage",
    "assets", "static", "public", "resources",
    "tests", "spec", "e2e", "integration", "unit",
    "docs", "documentation", "wiki", "readme",
    "scripts", "tools", "utils", "helpers",
    "components", "modules", "packages", "libs",
    "services", "api", "graphql", "rest", "grpc",
    "frontend", "backend", "client", "server",
    "mobile", "desktop", "web", "cli",
    "staging", "production", "development", "testing",
    "main", "master", "develop", "feature", "release", "hotfix",
    "v1", "v2", "v3", "latest", "stable", "nightly", "canary",
    "trunk", "branch", "tag", "fork",
    "origin", "upstream", "downstream",
    "issue", "pr", "pull request", "merge request",
    "commit", "push", "pull", "fetch", "rebase", "squash",
    "stash", "cherry pick", "bisect", "blame",
    "worktree", "submodule", "subtree",
    "hook", "action", "workflow", "pipeline", "job",
    "artifact", "cache", "secret", "variable",
    "deploy", "rollback", "scale", "restart",
    "monitor", "alert", "incident", "oncall",
    "sprint", "backlog", "epic", "story", "task", "bug",
    "retro", "standup", "planning", "review",
    "okr", "kpi", "metric", "dashboard",
    "roadmap", "milestone", "timeline", "deadline",
    "onboarding", "handbook", "playbook", "runbook",
    "changelog", "release notes", "migration", "upgrade",
    "deprecation", "sunset", "eol", "lts",
    "license", "contributing", "code of conduct", "security",
    "privacy", "terms", "cookies", "gdpr",
    "sla", "slo", "sli", "error budget",
    "pagerduty", "opsgenie", "victorops",
    "statuspage", "incident.io", "firehydrant",
    "splunk", "sumologic", "logz.io", "papertrail",
    "honeycomb", "lightstep", "signoz", "hyperdx",
    "mixpanel", "amplitude", "posthog", "heap", "segment",
    "launchdarkly", "optimizely", "growthbook",
    "intercom", "zendesk", "helpscout", "crisp", "tawk",
    "hubspot", "salesforce", "pipedrive", "close",
    "stripe", "paddle", "chargebee", "recurly",
    "auth0", "clerk", "kinde", "workos", "supabase auth",
    "s3", "r2", "b2", "wasabi", "minio",
    "cloudinary", "imgix", "uploadthing",
    "vercel", "netlify", "cloudflare pages", "deno deploy",
    "fly.io", "railway", "render", "koyeb", "porter",
    "planetscale", "neon", "turso", "xata", "nile",
    "upstash", "convex", "liveblocks", "replicache",
    "resend", "plunk", "loops", "customer.io", "braze",
    "novu", "courier", "knock", "magicbell",
    "sanity", "strapi", "contentful", "prismic", "hygraph",
    "tina", "decap cms", "netlify cms", "payload",
    "nextra", "docusaurus", "vitepress", "mdbook", "gitbook",
    "storybook", "ladle", "histoire",
    "playwright", "cypress", "vitest", "jest", "mocha",
    "testing library", "react testing library", "enzyme",
    "chromatic", "percy", "argos", "lost pixel",
    "eslint", "prettier", "biome", "oxc", "oxlint",
    "husky", "lint-staged", "commitlint", "commitizen",
    "changesets", "semantic release", "standard version",
    "renovate", "dependabot", "snyk", "socket",
    "turbo", "nx", "lerna", "rush", "bazel", "pants",
    "pnpm workspaces", "yarn workspaces", "npm workspaces",
    "vite", "webpack", "rollup", "esbuild", "swc", "parcel",
    "turbopack", "rspack", "farm",
    "tailwind", "unocss", "windicss", "styled components",
    "emotion", "vanilla extract", "panda css", "stylex",
    "radix ui", "shadcn ui", "headless ui", "ark ui",
    "chakra ui", "mantine", "mui", "ant design", "arco design",
    "next ui", "tremor", "flowbite", "daisyui", "preline",
    "tanstack query", "swr", "rtk query", "apollo client",
    "urql", "relay", "graphql request",
    "tanstack router", "react router", "wouter", "type route",
    "tanstack table", "ag grid", "glide data grid",
    "react hook form", "formik", "zod", "yup", "valibot", "arktype",
    "framer motion", "gsap", "react spring", "motion one",
    "d3", "echarts", "recharts", "nivo", "visx", "tremor",
    "lexical", "tiptap", "slate", "prosemirror", "quill",
    "monaco", "codemirror", "ace", "prism", "shiki",
    "pdf.js", "react pdf", "docx", "xlsx", "pptx",
    "exceljs", "sheetjs", "handsontable",
    "video.js", "plyr", "hls.js", "dash.js",
    "leaflet", "mapbox", "maplibre", "google maps js",
    "three.js", "babylon.js", "react three fiber",
    "pixi.js", "phaser", "konva", "fabric.js",
    "tensorflow.js", "onnx runtime", "transformers.js",
    "mediapipe", "ml5.js", "brain.js",
    "socket.io", "ws", "sse", "webrtc", "peerjs",
    "livekit", "agora", "twilio video", "daily",
    "ably", "pusher", "centrifugo", "socket.io admin",
    "temporal", "inngest", "trigger.dev", "windmill",
    "prefect", "dagster", "airflow", "luigi",
    "dbt", "airbyte", "fivetran", "stitch", "meltano",
    "snowflake", "bigquery", "redshift", "athena", "trino",
    "duckdb", "clickhouse", "druid", "pinot", "starrocks",
    "datalake", "iceberg", "delta lake", "hudi",
    "kubeflow", "mlflow", "bentoml", "seldon", "ray",
    "weights & biases", "neptune", "comet", "aim",
    "label studio", "prodigy", "argilla", "rubrix",
    "langfuse", "langsmith", "helicone", "athina",
    "chroma", "pinecone", "weaviate", "qdrant", "milvus",
    "llamaindex", "langchain", "haystack", "txtai",
    "vllm", "tgi", "triton", "torchserve", "bentoml",
    "modal", "banana", "replicate", "huggingface spaces",
    "gradio", "streamlit", "dash", "panel", "nicegui",
    "solara", "shiny", "taipy", "mesop",
    "reflex", "pynecone", "fasthtml", "htmx",
    "alpine.js", "petite vue", "stimulus", "hotwire",
    "liveview", "livewire", "blazor", "unicorn",
    "wasm", "webassembly", "webgpu", "webgl", "webxr",
    "pwa", "service worker", "workbox", "vite pwa",
    "tauri", "electron", "nw.js", "neutralino",
    "react native", "expo", "flutter", "ionic", "capacitor",
    "kotlin multiplatform", "compose multiplatform", "swift ui",
    "jetpack compose", "flutter flow", "draftbit",
    "appwrite", "supabase", "firebase", "nhost", "nhot",
    "pocketbase", "directus", "strapi", "payload cms",
    "baserow", "nocodb", "airtable", "retool", "budibase",
    "n8n", "make", "zapier", "ifttt", "huginn",
    "home assistant", "openhab", "homebridge", "scrypted",
    "frigate", "blue iris", "zoneminder", "shinobi",
    "jellyfin", "plex", "emby", "kodi", "infuse",
    "sonarr", "radarr", "lidarr", "prowlarr", "bazarr",
    "sabnzbd", "nzbget", "transmission", "qbittorrent", "deluge",
    "pihole", "adguard", "blocky", "technitium",
    "traefik", "nginx proxy manager", "caddy", "swag",
    "portainer", "yacht", "dockge", "casaos", "umbrel",
    "unraid", "truenas", "openmediavault", "synology", "qnap",
    "proxmox", "esxi", "hyper-v", "xcp-ng", "xen",
    "pfsense", "opnsense", "openwrt", "dd-wrt", "freshtomato",
    "tailscale", "headscale", "netbird", "netmaker", "innernet",
    "vaultwarden", "bitwarden", "passbolt", "padloc",
    "nextcloud", "owncloud", "seafile", "syncthing", "resilio",
    "immich", "photoprism", "piwigo", "lychee", "chevereto",
    "paperless", "paperless-ngx", "docspell", "teedy",
    "changedetection", "huginn", "n8n", "node-red",
    "uptime kuma", "gatus", "statping", "vigil",
    "plausible", "umami", "matomo", "fathom", "pirsch",
    "ntfy", "gotify", "pushover", "pushbullet", "slack",
    "grafana", "prometheus", "alertmanager", "loki", "tempo",
    "authentik", "authelia", "keycloak", "ory", "zitadel",
    "outline", "bookstack", "wikijs", "dokuwiki", "trilium",
    "affine", "appflowy", "siyuan", "anytype",
    "nocodb", "baserow", "grist", "rowy",
    "focalboard", "plane", "huly", "taiga", "openproject",
    "mattermost", "rocket.chat", "zulip", "matrix", "element",
    "jitsi", "bigbluebutton", "galene", "livekit",
    "mastodon", "pleroma", "misskey", "pixelfed", "peertube",
    "lemmy", "kbin", "discourse", "flarum", "nodebb",
    "ghost", "wordpress", "drupal", "joomla", "typo3",
    "hugo", "jekyll", "gatsby", "next.js", "nuxt", "sveltekit",
    "astro", "eleventy", "zola", "pelican", "nikola",
    "mkdocs", "sphinx", "doxygen", "jsdoc", "typedoc",
    "swagger", "openapi", "asyncapi", "graphql schema",
    "postman", "insomnia", "hoppscotch", "bruno", "httpie",
    "k6", "jmeter", "locust", "artillery", "wrk", "hey",
    "chaos mesh", "gremlin", "chaos monkey", "litmus",
    "trivy", "grype", "snyk", "dependency track", "fossa",
    "sonarqube", "codeql", "semgrep", "bearer", "checkov",
    "tfsec", "terrascan", "kics", "bridgecrew",
    "falco", "tetragon", "tracee", "cilium",
    "kyverno", "opa", "gatekeeper", "jsPolicy",
    "cert manager", "external secrets", "sealed secrets", "sops",
    "crossplane", "terraform", "pulumi", "cdk", "cdk8s", "cdktf",
    "atlantis", "digger", "env0", "spacelift", "scalr",
    "backstage", "port", "configure8", "opslevel", "cortex",
    "openfeature", "flagd", "go feature flag", "unleash",
    "keda", "karpenter", "cluster autoscaler", "goldilocks",
    "descheduler", "vpa", "addon resizer",
    "robusta", "komodor", "groundcover", "epsagon",
    "pixie", "parca", "pyroscope", "grafana phlare",
    "opencost", "kubecost", "cast ai", "spot.io",
    "teleport", "strongdm", "boundary", "pomerium",
    "infisical", "doppler", "vault", "aws secrets manager",
    "stepzen", "hasura", "graphql mesh", "apollo router",
    "wundergraph", "grafbase", "tailcall", "async graphql",
    "encore", "ampt", "nitric", "shuttle", "napptive",
    "acorn", "devtron", "shipa", "qovery", "coherence",
    "mirrord", "telepresence", "garden", "tilt", "skaffold",
    "devspace", "okteto", "loft", "vcluster", "kluctl",
    "kustomize", "helm", "jsonnet", "cue", "dhall", "nickel",
    "timoni", "flux", "argocd", "carvel", "werf",
    "dagger", "earthly", "depot", "buildkit", "buildx",
    "ko", "jib", "pack", "buildpacks", "source to image",
    "cosign", "sigstore", "notary", "notation", "witness",
    "kyverno", "connaisseur", "ratify", "trust policy",
    "spiffe", "spire", "cert-manager", "step-ca", "smallstep",
    "dex", "ory hydra", "keycloak", "authentik", "casdoor",
    "logto", "supertokens", "clerk", "workos", "propelauth",
    "descope", "frontegg", "userfront", "authgear",
    "rowy", "saltcorn", "reify", "bildr", "bubble",
    "glide", "adalo", "flutterflow", "draftbit", "thunkable",
    "outsystems", "mendix", "power apps", "appian", "pega",
    "servicenow", "salesforce", "dynamics 365", "oracle apex",
    "sap build", "zoho creator", "quickbase", "kissflow",
    "creatio", "nintex", "kintone", "smartsheet",
    "airtable", "notion", "coda", "fibery", "clay",
    "whimsical", "tldraw", "excalidraw", "miro", "figjam",
    "loom", "mmhmm", "cleanShot", "screen studio", "capcut",
    "descript", "riverside", "restream", "streamyard",
    "obsidian", "logseq", "roam", "notion", "craft", "bear",
    "things", "todoist", "ticktick", "omnifocus", "reminders",
    "fantastical", "cron", "cal.com", "calendly", "savvycal",
    "superhuman", "mimestream", "spark", "canary", "newton",
    "arc", "sigmaos", "sidekick", "station", "biscuit",
    "raycast", "alfred", "launchbar", "quicksilver", "ueli",
    "rectangle", "magnet", "moom", "divvy", "amethyst", "yabai",
    "karabiner", "bettertouchtool", "hammerspoon", "skhd",
    "hazel", "dropover", "yoink", "unclutter", "default folder x",
    "popclip", "dash", "devdocs", "zeal", "velocity",
    "iterm2", "warp", "kitty", "alacritty", "wezterm", "rio",
    "tmux", "zellij", "byobu", "screen", "mosh",
    "fish", "zsh", "nushell", "elvish", "ion", "oil",
    "atuin", "mcfly", "zoxide", "fzf", "skim",
    "bat", "exa", "eza", "lsd", "dust", "duf", "bottom", "btm",
    "procs", "ripgrep", "fd", "sd", "delta", "difftastic",
    "jq", "yq", "xq", "dasel", "gron", "fx", "jid", "jiq",
    "httpie", "xh", "curlie", "xplr", "broot", "gitui",
    "lazygit", "lazydocker", "lazynpm", "lazyjj",
    "neovim", "helix", "kakoune", "emacs", "doom emacs", "spacemacs",
    "vscode", "zed", "lapce", "pulsar", "lite-xl", "cudatext",
    "micro", "nano", "amp", "slap", "xi", "gnu emacs",
    "tree-sitter", "lsp", "dap", "nvim-tree", "telescope",
    "harpoon", "which-key", "undotree", "gitsigns", "diffview",
    "nvim-cmp", "copilot.lua", "codeium", "supermaven", "tabnine",
    "oil.nvim", "neo-tree", "nvim-tree", "mini.files", "dirbuf",
    "toggleterm", "floaterm", "neoterm", "vim-test", "neotest",
    "overseer", "asyncrun", "quickrun", "code_runner",
    "markdown-preview", "glow", "mdcat", "rich-cli",
    "lazydocker", "dry", "dive", "ctop", "lens", "k9s",
    "kubectx", "kubens", "kube-ps1", "kubecolor", "kubie",
    "stern", "kail", "kubetail", "kubespy", "popeye",
    "kubeconform", "kubeval", "pluto", "nova", "kube-no-trouble",
    "kube-score", "polaris", "kube-bench", "kube-hunter",
    "kube-linter", "datree", "kubescape", "trivy-operator",
    "botkube", "kubewatch", "kured", "node-problem-detector",
    "awscli", "aws-vault", "granted", "saml2aws", "okta-aws",
    "gcloud", "gsutil", "az", "doctl", "doppler", "infisical",
    "terraform", "terragrunt", "terramate", "atmos", "terraspace",
    "pulumi", "cdktf", "wing", "pluto", "bicep", "pkl",
    "ansible", "molecule", "ansible-lint", "ansible-navigator",
    "packer", "vagrant", "veertu", "utm", "lima", "colima",
    "orbstack", "rancher desktop", "podman desktop", "finch",
    "minikube", "kind", "k3d", "microk8s", "k0s", "k3s",
    "talos", "flatcar", "bottlerocket", "kubeos", "rancheros",
    "longhorn", "rook", "ceph", "openebs", "mayastor", "seaweedfs",
    "minio", "garage", "seaweedfs", "juicefs", "alluxio",
    "velero", "kasten", "trilio", "cloudcasa", "kanister",
    "cert-manager", "external-dns", "external-secrets", "vault",
    "ingress-nginx", "contour", "emissary", "gloo", "kong",
    "cilium", "calico", "flannel", "weave", "multus", "antrea",
    "istio", "linkerd", "consul", "kuma", "traefik mesh",
    "jaeger", "tempo", "zipkin", "opentelemetry", "signoz",
    "fluentd", "fluentbit", "vector", "logstash", "filebeat",
    "elasticsearch", "opensearch", "quickwit", "loki", "meilisearch",
    "grafana", "kibana", "opensearch dashboards", "redash", "metabase",
    "argo workflows", "argo events", "tekton", "keptn", "keel",
    "flux", "argocd", "jenkins x", "spinnaker", "harness",
    "knative", "openfaas", "fission", "kubeless", "nuclio",
    "dapr", "keda", "wasmcloud", "spin", "fermyon",
    "crossplane", "upbound", "terraform operator", "ack", "ack",
    "strimzi", "rabbitmq operator", "postgres operator", "zalando",
    "cloudnative-pg", "stackgres", "percona operator", "mysql operator",
    "victoria metrics", "thanos", "cortex", "mimir", "grafana mimir",
    "tempo", "pyroscope", "phlare", "parca", "pixie",
    "opencost", "kubecost", "castai", "spot", "stormforge",
    "vcluster", "loft", "devtron", "shipa", "portainer",
    "kubeapps", "glasskube", "kubepak", "timoni", "werf",
    "devspace", "garden", "tilt", "skaffold", "okteto",
    "telepresence", "mirrord", "gefyra", "kubefwd", "ktunnel",
    "kubevpn", "vpnkit", "docker desktop", "rancher desktop",
    "orbstack", "colima", "lima", "finch", "podman machine",
    "multipass", "microk8s", "minikube", "kind", "k3d",
    "vagrant", "virtualbox", "vmware fusion", "parallels", "utm",
    "qemu", "libvirt", "virt-manager", "gnome boxes", "quickemu",
    "distrobox", "toolbox", "devbox", "devenv", "flox",
    "nix", "nixos", "nix-darwin", "home-manager", "nixpkgs",
    "guix", "guix system", "guix home", "guix pack",
    "brew", "linuxbrew", "macports", "fink", "pkgsrc",
    "conda", "mamba", "micromamba", "pixi", "rattler",
    "pip", "pipx", "poetry", "pdm", "hatch", "rye", "uv",
    "npm", "yarn", "pnpm", "bun", "deno", "volta", "fnm", "nvm",
    "cargo", "rustup", "go", "dotnet", "sdkman", "jabba",
    "asdf", "mise", "proto", "vfox", "aqua", "rtx",
    "direnv", "dotenv", "envchain", "envkey", "infisical",
    "starship", "oh-my-posh", "powerlevel10k", "spaceship", "pure",
    "nerd fonts", "powerline", "font awesome", "material icons",
    "catppuccin", "tokyo night", "nord", "dracula", "gruvbox",
    "rose pine", "everforest", "kanagawa", "onedark", "monokai",
    "solarized", "ayu", "palenight", "night owl", "github theme",
    "zellij", "tmux", "screen", "byobu", "mtm", "dvtm",
    "wezterm", "kitty", "alacritty", "foot", "ghostty", "rio",
    "warp", "iterm2", "hyper", "tabby", "terminus", "extraterm",
    "windows terminal", "conemu", "cmder", "mintty", "msys2",
    "putty", "mobaxterm", "securecrt", "royal tsx", "termius",
    "blink", "shelly", "a-shell", "ish", "termux", "juicessh",
    "code-server", "openvscode-server", "gitpod", "github codespaces",
    "coder", "cdr", "devpod", "envd", "gitpod flex",
    "replit", "stackblitz", "codesandbox", "glitch", "codepen",
    "jsfiddle", "playcode", "runjs", "quokkajs", "observable",
    "deepnote", "hex", "noteable", "colab", "kaggle", "sagemaker",
    "databricks", "snowflake", "bigquery", "redshift", "duckdb",
    "motherduck", "chdb", "libsql", "turso", "rqlite", "dqlite",
    "sqlite", "postgres", "mysql", "mariadb", "cockroachdb",
    "yugabyte", "tidb", "oceanbase", "vitess", "spanner",
    "alloydb", "aurora", "rds", "cloud sql", "azure sql",
    "supabase", "neon", "planetscale", "xata", "nile", "convex",
    "firebase", "appwrite", "pocketbase", "nhost", "directus",
    "payload", "strapi", "sanity", "contentful", "hygraph",
    "prismic", "storyblok", "kontent", "buttercms", "agility",
    "builder.io", "plasmic", "makeswift", "instant", "tina",
    "keystatic", "decap cms", "netlify cms", "prose", "cloudcannon",
    "siteleaf", "forestry", "tina cms", "spina cms", "statamic",
    "craft cms", "expressionengine", "modx", "processwire", "bolt",
    "grav", "getgrav", "pico", "bludit", "automad", "typo3",
    "neos", "contao", "typo3", "drupal", "joomla", "wordpress",
    "shopify", "bigcommerce", "woocommerce", "magento", "prestashop",
    "saleor", "medusa", "vendure", "swell", "commercetools",
    "elastic path", "fabric", "nopcommerce", "virto commerce",
    "orocommerce", "sylius", "solidus", "spree", "shuup",
    "sentry", "datadog", "newrelic", "dynatrace", "appdynamics",
    "instana", "honeycomb", "lightstep", "jaeger", "tempo",
    "signoz", "hyperdx", "highlight", "logrocket", "fullstory",
    "hotjar", "mouseflow", "crazyegg", "luckyorange", "clarity",
    "posthog", "mixpanel", "amplitude", "heap", "pendo", "indicative",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature", "flagd", "go-feature-flag", "flipt",
    "pagerduty", "opsgenie", "victorops", "splunk on-call",
    "incident.io", "firehydrant", "rootly", "incident labs",
    "blameless", "jeli", "transposit", "cortex", "opslevel",
    "statuspage", "hund", "instatus", "betterstack", "checkly",
    "pingdom", "uptime", "site24x7", "datadog synthetics",
    "splunk", "sumologic", "logz.io", "papertrail", "loggly",
    "datadog logs", "newrelic logs", "elastic", "opensearch",
    "meilisearch", "typesense", "algolia", "elasticsearch",
    "pinecone", "weaviate", "qdrant", "chroma", "milvus",
    "redis", "dragonfly", "keydb", "garnet", "valkey",
    "upstash", "momento", "readySet", "polyScale",
    "cloudflare", "fastly", "akamai", "bunny", "keycdn",
    "vercel", "netlify", "cloudflare pages", "deno deploy",
    "fly.io", "railway", "render", "koyeb", "porter",
    "heroku", "digitalocean", "linode", "vultr", "hetzner",
    "ovh", "scaleway", "upcloud", "exoscale", "civo",
    "aws", "gcp", "azure", "oracle cloud", "ibm cloud",
    "alibaba cloud", "tencent cloud", "huawei cloud", "baidu cloud",
    "openstack", "cloudstack", "opennebula", "apache cloudstack",
    "maas", "metal", "equinix", "packet", "phoenixnap",
    "leaseweb", "ovhcloud", "soyoustart", "kimsufi", "buyvm",
    "netcup", "contabo", "strato", "ionos", "hostinger",
    "namecheap", "godaddy", "cloudflare registrar", "porkbun",
    "hover", "gandi", "iwantmyname", "dnsimple", "dnsmadeeasy",
    "route53", "cloudflare dns", "google domains", "azure dns",
    "bunny dns", "ns1", "constellix", "dyn", "ultradns",
    "vercel domains", "netlify domains", "framer domains",
    "carrd", "framer", "webflow", "squarespace", "wix",
    "weebly", "strikingly", "duda", "jimdo", "site123",
    "wordpress.com", "ghost.org", "medium", "substack", "beehiiv",
    "convertkit", "mailchimp", "klaviyo", "drip", "customer.io",
    "sendgrid", "mailgun", "postmark", "resend", "plunk",
    "loops", "buttondown", "curated", "revue", "mailbrew",
    "hey", "fastmail", "protonmail", "tuta", "skiff", "mailbox.org",
    "zoho mail", "mxroute", "migadu", "purelymail", "forwardemail",
    "simplelogin", "anonaddy", "duckduckgo email", "firefox relay",
    "1password", "bitwarden", "dashlane", "nordpass", "keeper",
    "roboform", "sticky password", "zoho vault", "keeper",
    "proton pass", "heylogin", "passbolt", "padloc", "spectre",
    "lesspass", "masterpassword", "hashpass", "pwgen", "diceware",
    "yubikey", "solokey", "nitrokey", "onlykey", "trezor",
    "ledger", "keepkey", "bitbox", "coldcard", "passport",
    "seedsigner", "specter", "sparrow", "electrum", "wasabi",
    "samourai", "bluewallet", "muun", "breez", "phoenix",
    "zeus", "blink", "wallet of satoshi", "alby", "getalby",
    "nostr", "damus", "amethyst", "primal", "snort", "iris",
    "coracle", "nostrudel", "yakihonne", "highlighter", "zapstream",
    "bluesky", "threads", "mastodon", "pixelfed", "loops",
    "signal", "telegram", "whatsapp", "matrix", "element",
    "session", "simpleX", "briar", "jami", "keet", "wire",
    "threema", "status", "berty", "delta chat", "cwtch",
    "discord", "slack", "teams", "google chat", "mattermost",
    "rocket.chat", "zulip", "twist", "flock", "ryver",
    "basecamp", "asana", "monday", "clickup", "linear",
    "height", "plane", "huly", "taiga", "openproject",
    "jira", "confluence", "notion", "coda", "fibery",
    "airtable", "smartsheet", "quip", "dropbox paper", "slite",
    "almanac", "tettra", "guru", "bloomfire", "document360",
    "gitbook", "readme", "archbee", "mintlify", "fern",
    "redocly", "bump.sh", "stoplight", "swaggerhub", "postman",
    "insomnia", "hoppscotch", "bruno", "yaak", "httpie",
    "graphql playground", "altair", "graphiql", "apollo studio",
    "hasura console", "dgraph ratel", "arangodb webui", "neo4j browser",
    "redis insight", "mongo compass", "dbeaver", "tableplus",
    "datagrip", "navicat", "sequel ace", "sequel pro", "pgadmin",
    "phpmyadmin", "adminer", "sqlitebrowser", "db browser for sqlite",
    "beekeeper studio", "heidisql", "dbschema", "dbvisualizer",
    "valentina studio", "razorsql", "sqlectron", "dbgate",
    "azure data studio", "mysql workbench", "oracle sql developer",
    "robo 3t", "studio 3t", "nosqlbooster", "mongochef",
    "redis commander", "medis", "another redis desktop manager",
    "kafdrop", "kafka ui", "redpanda console", "akhq", "kpow",
    "conduktor", "offset explorer", "kcat", "kafkacat",
    "rabbitmq management", "nats dashboard", "natsboard",
    "elasticvue", "dejavu", "cerebro", "elasticsearch head",
    "opensearch dashboards", "kibana", "grafana", "chronograf",
    "prometheus", "alertmanager", "thanos", "cortex", "mimir",
    "victoriametrics", "vmui", "grafana explore", "grafana loki",
    "jaeger ui", "zipkin ui", "tempo query", "signoz frontend",
    "hyperdx", "highlight", "logrocket", "fullstory", "hotjar",
    "posthog", "mixpanel", "amplitude", "heap", "pendo",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature flagd", "flipt", "go-feature-flag",
    "dagger", "earthly", "depot", "buildkit", "buildx",
    "ko", "jib", "pack", "buildpacks", "source to image",
    "cosign", "sigstore", "notary", "notation", "witness",
    "kyverno", "connaisseur", "ratify", "trust policy",
    "spiffe", "spire", "cert-manager", "step-ca", "smallstep",
    "dex", "ory hydra", "keycloak", "authentik", "casdoor",
    "logto", "supertokens", "clerk", "workos", "propelauth",
    "descope", "frontegg", "userfront", "authgear",
    "rowy", "saltcorn", "reify", "bildr", "bubble",
    "glide", "adalo", "flutterflow", "draftbit", "thunkable",
    "outsystems", "mendix", "power apps", "appian", "pega",
    "servicenow", "salesforce", "dynamics 365", "oracle apex",
    "sap build", "zoho creator", "quickbase", "kissflow",
    "creatio", "nintex", "kintone", "smartsheet",
    "airtable", "notion", "coda", "fibery", "clay",
    "whimsical", "tldraw", "excalidraw", "miro", "figjam",
    "loom", "mmhmm", "cleanShot", "screen studio", "capcut",
    "descript", "riverside", "restream", "streamyard",
    "obsidian", "logseq", "roam", "notion", "craft", "bear",
    "things", "todoist", "ticktick", "omnifocus", "reminders",
    "fantastical", "cron", "cal.com", "calendly", "savvycal",
    "superhuman", "mimestream", "spark", "canary", "newton",
    "arc", "sigmaos", "sidekick", "station", "biscuit",
    "raycast", "alfred", "launchbar", "quicksilver", "ueli",
    "rectangle", "magnet", "moom", "divvy", "amethyst", "yabai",
    "karabiner", "bettertouchtool", "hammerspoon", "skhd",
    "hazel", "dropover", "yoink", "unclutter", "default folder x",
    "popclip", "dash", "devdocs", "zeal", "velocity",
    "iterm2", "warp", "kitty", "alacritty", "wezterm", "rio",
    "tmux", "zellij", "byobu", "screen", "mosh",
    "fish", "zsh", "nushell", "elvish", "ion", "oil",
    "atuin", "mcfly", "zoxide", "fzf", "skim",
    "bat", "exa", "eza", "lsd", "dust", "duf", "bottom", "btm",
    "procs", "ripgrep", "fd", "sd", "delta", "difftastic",
    "jq", "yq", "xq", "dasel", "gron", "fx", "jid", "jiq",
    "httpie", "xh", "curlie", "xplr", "broot", "gitui",
    "lazygit", "lazydocker", "lazynpm", "lazyjj",
    "neovim", "helix", "kakoune", "emacs", "doom emacs", "spacemacs",
    "vscode", "zed", "lapce", "pulsar", "lite-xl", "cudatext",
    "micro", "nano", "amp", "slap", "xi", "gnu emacs",
    "tree-sitter", "lsp", "dap", "nvim-tree", "telescope",
    "harpoon", "which-key", "undotree", "gitsigns", "diffview",
    "nvim-cmp", "copilot.lua", "codeium", "supermaven", "tabnine",
    "oil.nvim", "neo-tree", "nvim-tree", "mini.files", "dirbuf",
    "toggleterm", "floaterm", "neoterm", "vim-test", "neotest",
    "overseer", "asyncrun", "quickrun", "code_runner",
    "markdown-preview", "glow", "mdcat", "rich-cli",
    "lazydocker", "dry", "dive", "ctop", "lens", "k9s",
    "kubectx", "kubens", "kube-ps1", "kubecolor", "kubie",
    "stern", "kail", "kubetail", "kubespy", "popeye",
    "kubeconform", "kubeval", "pluto", "nova", "kube-no-trouble",
    "kube-score", "polaris", "kube-bench", "kube-hunter",
    "kube-linter", "datree", "kubescape", "trivy-operator",
    "botkube", "kubewatch", "kured", "node-problem-detector",
    "awscli", "aws-vault", "granted", "saml2aws", "okta-aws",
    "gcloud", "gsutil", "az", "doctl", "doppler", "infisical",
    "terraform", "terragrunt", "terramate", "atmos", "terraspace",
    "pulumi", "cdktf", "wing", "pluto", "bicep", "pkl",
    "ansible", "molecule", "ansible-lint", "ansible-navigator",
    "packer", "vagrant", "veertu", "utm", "lima", "colima",
    "orbstack", "rancher desktop", "podman desktop", "finch",
    "minikube", "kind", "k3d", "microk8s", "k0s", "k3s",
    "talos", "flatcar", "bottlerocket", "kubeos", "rancheros",
    "longhorn", "rook", "ceph", "openebs", "mayastor", "seaweedfs",
    "minio", "garage", "seaweedfs", "juicefs", "alluxio",
    "velero", "kasten", "trilio", "cloudcasa", "kanister",
    "cert-manager", "external-dns", "external-secrets", "vault",
    "ingress-nginx", "contour", "emissary", "gloo", "kong",
    "cilium", "calico", "flannel", "weave", "multus", "antrea",
    "istio", "linkerd", "consul", "kuma", "traefik mesh",
    "jaeger", "tempo", "zipkin", "opentelemetry", "signoz",
    "fluentd", "fluentbit", "vector", "logstash", "filebeat",
    "elasticsearch", "opensearch", "quickwit", "loki", "meilisearch",
    "grafana", "kibana", "opensearch dashboards", "redash", "metabase",
    "argo workflows", "argo events", "tekton", "keptn", "keel",
    "flux", "argocd", "jenkins x", "spinnaker", "harness",
    "knative", "openfaas", "fission", "kubeless", "nuclio",
    "dapr", "keda", "wasmcloud", "spin", "fermyon",
    "crossplane", "upbound", "terraform operator", "ack", "ack",
    "strimzi", "rabbitmq operator", "postgres operator", "zalando",
    "cloudnative-pg", "stackgres", "percona operator", "mysql operator",
    "victoria metrics", "thanos", "cortex", "mimir", "grafana mimir",
    "tempo", "pyroscope", "phlare", "parca", "pixie",
    "opencost", "kubecost", "castai", "spot", "stormforge",
    "vcluster", "loft", "devtron", "shipa", "portainer",
    "kubeapps", "glasskube", "kubepak", "timoni", "werf",
    "devspace", "garden", "tilt", "skaffold", "okteto",
    "telepresence", "mirrord", "gefyra", "kubefwd", "ktunnel",
    "kubevpn", "vpnkit", "docker desktop", "rancher desktop",
    "orbstack", "colima", "lima", "finch", "podman machine",
    "multipass", "microk8s", "minikube", "kind", "k3d",
    "vagrant", "virtualbox", "vmware fusion", "parallels", "utm",
    "qemu", "libvirt", "virt-manager", "gnome boxes", "quickemu",
    "distrobox", "toolbox", "devbox", "devenv", "flox",
    "nix", "nixos", "nix-darwin", "home-manager", "nixpkgs",
    "guix", "guix system", "guix home", "guix pack",
    "brew", "linuxbrew", "macports", "fink", "pkgsrc",
    "conda", "mamba", "micromamba", "pixi", "rattler",
    "pip", "pipx", "poetry", "pdm", "hatch", "rye", "uv",
    "npm", "yarn", "pnpm", "bun", "deno", "volta", "fnm", "nvm",
    "cargo", "rustup", "go", "dotnet", "sdkman", "jabba",
    "asdf", "mise", "proto", "vfox", "aqua", "rtx",
    "direnv", "dotenv", "envchain", "envkey", "infisical",
    "starship", "oh-my-posh", "powerlevel10k", "spaceship", "pure",
    "nerd fonts", "powerline", "font awesome", "material icons",
    "catppuccin", "tokyo night", "nord", "dracula", "gruvbox",
    "rose pine", "everforest", "kanagawa", "onedark", "monokai",
    "solarized", "ayu", "palenight", "night owl", "github theme",
    "zellij", "tmux", "screen", "byobu", "mtm", "dvtm",
    "wezterm", "kitty", "alacritty", "foot", "ghostty", "rio",
    "warp", "iterm2", "hyper", "tabby", "terminus", "extraterm",
    "windows terminal", "conemu", "cmder", "mintty", "msys2",
    "putty", "mobaxterm", "securecrt", "royal tsx", "termius",
    "blink", "shelly", "a-shell", "ish", "termux", "juicessh",
    "code-server", "openvscode-server", "gitpod", "github codespaces",
    "coder", "cdr", "devpod", "envd", "gitpod flex",
    "replit", "stackblitz", "codesandbox", "glitch", "codepen",
    "jsfiddle", "playcode", "runjs", "quokkajs", "observable",
    "deepnote", "hex", "noteable", "colab", "kaggle", "sagemaker",
    "databricks", "snowflake", "bigquery", "redshift", "duckdb",
    "motherduck", "chdb", "libsql", "turso", "rqlite", "dqlite",
    "sqlite", "postgres", "mysql", "mariadb", "cockroachdb",
    "yugabyte", "tidb", "oceanbase", "vitess", "spanner",
    "alloydb", "aurora", "rds", "cloud sql", "azure sql",
    "supabase", "neon", "planetscale", "xata", "nile", "convex",
    "firebase", "appwrite", "pocketbase", "nhost", "directus",
    "payload", "strapi", "sanity", "contentful", "hygraph",
    "prismic", "storyblok", "kontent", "buttercms", "agility",
    "builder.io", "plasmic", "makeswift", "instant", "tina",
    "keystatic", "decap cms", "netlify cms", "prose", "cloudcannon",
    "siteleaf", "forestry", "tina cms", "spina cms", "statamic",
    "craft cms", "expressionengine", "modx", "processwire", "bolt",
    "grav", "getgrav", "pico", "bludit", "automad", "typo3",
    "neos", "contao", "typo3", "drupal", "joomla", "wordpress",
    "shopify", "bigcommerce", "woocommerce", "magento", "prestashop",
    "saleor", "medusa", "vendure", "swell", "commercetools",
    "elastic path", "fabric", "nopcommerce", "virto commerce",
    "orocommerce", "sylius", "solidus", "spree", "shuup",
    "sentry", "datadog", "newrelic", "dynatrace", "appdynamics",
    "instana", "honeycomb", "lightstep", "jaeger", "tempo",
    "signoz", "hyperdx", "highlight", "logrocket", "fullstory",
    "hotjar", "mouseflow", "crazyegg", "luckyorange", "clarity",
    "posthog", "mixpanel", "amplitude", "heap", "pendo", "indicative",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature", "flagd", "go-feature-flag", "flipt",
    "pagerduty", "opsgenie", "victorops", "splunk on-call",
    "incident.io", "firehydrant", "rootly", "incident labs",
    "blameless", "jeli", "transposit", "cortex", "opslevel",
    "statuspage", "hund", "instatus", "betterstack", "checkly",
    "pingdom", "uptime", "site24x7", "datadog synthetics",
    "splunk", "sumologic", "logz.io", "papertrail", "loggly",
    "datadog logs", "newrelic logs", "elastic", "opensearch",
    "meilisearch", "typesense", "algolia", "elasticsearch",
    "pinecone", "weaviate", "qdrant", "chroma", "milvus",
    "redis", "dragonfly", "keydb", "garnet", "valkey",
    "upstash", "momento", "readySet", "polyScale",
    "cloudflare", "fastly", "akamai", "bunny", "keycdn",
    "vercel", "netlify", "cloudflare pages", "deno deploy",
    "fly.io", "railway", "render", "koyeb", "porter",
    "heroku", "digitalocean", "linode", "vultr", "hetzner",
    "ovh", "scaleway", "upcloud", "exoscale", "civo",
    "aws", "gcp", "azure", "oracle cloud", "ibm cloud",
    "alibaba cloud", "tencent cloud", "huawei cloud", "baidu cloud",
    "openstack", "cloudstack", "opennebula", "apache cloudstack",
    "maas", "metal", "equinix", "packet", "phoenixnap",
    "leaseweb", "ovhcloud", "soyoustart", "kimsufi", "buyvm",
    "netcup", "contabo", "strato", "ionos", "hostinger",
    "namecheap", "godaddy", "cloudflare registrar", "porkbun",
    "hover", "gandi", "iwantmyname", "dnsimple", "dnsmadeeasy",
    "route53", "cloudflare dns", "google domains", "azure dns",
    "bunny dns", "ns1", "constellix", "dyn", "ultradns",
    "vercel domains", "netlify domains", "framer domains",
    "carrd", "framer", "webflow", "squarespace", "wix",
    "weebly", "strikingly", "duda", "jimdo", "site123",
    "wordpress.com", "ghost.org", "medium", "substack", "beehiiv",
    "convertkit", "mailchimp", "klaviyo", "drip", "customer.io",
    "sendgrid", "mailgun", "postmark", "resend", "plunk",
    "loops", "buttondown", "curated", "revue", "mailbrew",
    "hey", "fastmail", "protonmail", "tuta", "skiff", "mailbox.org",
    "zoho mail", "mxroute", "migadu", "purelymail", "forwardemail",
    "simplelogin", "anonaddy", "duckduckgo email", "firefox relay",
    "1password", "bitwarden", "dashlane", "nordpass", "keeper",
    "roboform", "sticky password", "zoho vault", "keeper",
    "proton pass", "heylogin", "passbolt", "padloc", "spectre",
    "lesspass", "masterpassword", "hashpass", "pwgen", "diceware",
    "yubikey", "solokey", "nitrokey", "onlykey", "trezor",
    "ledger", "keepkey", "bitbox", "coldcard", "passport",
    "seedsigner", "specter", "sparrow", "electrum", "wasabi",
    "samourai", "bluewallet", "muun", "breez", "phoenix",
    "zeus", "blink", "wallet of satoshi", "alby", "getalby",
    "nostr", "damus", "amethyst", "primal", "snort", "iris",
    "coracle", "nostrudel", "yakihonne", "highlighter", "zapstream",
    "bluesky", "threads", "mastodon", "pixelfed", "loops",
    "signal", "telegram", "whatsapp", "matrix", "element",
    "session", "simpleX", "briar", "jami", "keet", "wire",
    "threema", "status", "berty", "delta chat", "cwtch",
    "discord", "slack", "teams", "google chat", "mattermost",
    "rocket.chat", "zulip", "twist", "flock", "ryver",
    "basecamp", "asana", "monday", "clickup", "linear",
    "height", "plane", "huly", "taiga", "openproject",
    "jira", "confluence", "notion", "coda", "fibery",
    "airtable", "smartsheet", "quip", "dropbox paper", "slite",
    "almanac", "tettra", "guru", "bloomfire", "document360",
    "gitbook", "readme", "archbee", "mintlify", "fern",
    "redocly", "bump.sh", "stoplight", "swaggerhub", "postman",
    "insomnia", "hoppscotch", "bruno", "yaak", "httpie",
    "graphql playground", "altair", "graphiql", "apollo studio",
    "hasura console", "dgraph ratel", "arangodb webui", "neo4j browser",
    "redis insight", "mongo compass", "dbeaver", "tableplus",
    "datagrip", "navicat", "sequel ace", "sequel pro", "pgadmin",
    "phpmyadmin", "adminer", "sqlitebrowser", "db browser for sqlite",
    "beekeeper studio", "heidisql", "dbschema", "dbvisualizer",
    "valentina studio", "razorsql", "sqlectron", "dbgate",
    "azure data studio", "mysql workbench", "oracle sql developer",
    "robo 3t", "studio 3t", "nosqlbooster", "mongochef",
    "redis commander", "medis", "another redis desktop manager",
    "kafdrop", "kafka ui", "redpanda console", "akhq", "kpow",
    "conduktor", "offset explorer", "kcat", "kafkacat",
    "rabbitmq management", "nats dashboard", "natsboard",
    "elasticvue", "dejavu", "cerebro", "elasticsearch head",
    "opensearch dashboards", "kibana", "grafana", "chronograf",
    "prometheus", "alertmanager", "thanos", "cortex", "mimir",
    "victoriametrics", "vmui", "grafana explore", "grafana loki",
    "jaeger ui", "zipkin ui", "tempo query", "signoz frontend",
    "hyperdx", "highlight", "logrocket", "fullstory", "hotjar",
    "posthog", "mixpanel", "amplitude", "heap", "pendo",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature flagd", "flipt", "go-feature-flag",
}

# Desktop commands / actions
_DESKTOP_COMMANDS = {
    "open", "close", "start", "stop", "launch", "quit", "exit",
    "play", "pause", "resume", "next", "previous", "skip", "rewind",
    "volume up", "volume down", "mute", "unmute",
    "brightness up", "brightness down",
    "screenshot", "screen shot", "screen record",
    "lock", "unlock", "shutdown", "restart", "sleep", "hibernate",
    "log out", "sign out", "sign in", "log in",
    "copy", "paste", "cut", "delete", "undo", "redo",
    "save", "save as", "print", "export", "import",
    "refresh", "reload", "back", "forward",
    "zoom in", "zoom out", "full screen", "minimize", "maximize", "restore",
    "new tab", "close tab", "new window", "close window",
    "switch tab", "switch window", "go to", "navigate to",
    "scroll up", "scroll down", "page up", "page down",
    "search", "find", "replace", "select all",
    "take a note", "create a reminder", "set a timer", "set an alarm",
    "what time is it", "what day is it", "what is the date",
    "tell me a joke", "how are you", "what can you do",
    "help", "settings", "preferences", "configure",
    "update", "upgrade", "install", "uninstall",
    "check for updates", "system update",
    "show desktop", "hide window", "move window", "resize window",
    "split screen", "tile window", "snap window",
    "focus", "alt tab", "switch to", "bring to front",
    "send to back", "always on top",
    "dark mode", "light mode", "night light", "blue light filter",
    "do not disturb", "focus mode", "airplane mode",
    "wifi on", "wifi off", "bluetooth on", "bluetooth off",
    "battery status", "power mode", "performance mode", "battery saver",
    "connect to", "disconnect from", "pair", "unpair",
    "cast", "screen share", "present", "mirror screen",
    "record meeting", "start recording", "stop recording",
    "take a picture", "take a photo", "record video",
    "scan", "scan document", "scan qr code",
    "translate", "define", "spell", "pronounce",
    "calculate", "convert", "measure",
    "set volume to", "set brightness to",
    "play music", "pause music", "next song", "previous song",
    "shuffle", "repeat", "like", "dislike", "add to playlist",
    "what song is this", "who is this artist",
    "play some music", "play something", "recommend music",
    "read my email", "check email", "send email", "compose email",
    "read messages", "send message", "reply", "forward",
    "call", "hang up", "answer", "decline",
    "show calendar", "add event", "create meeting", "schedule",
    "show reminders", "add reminder", "mark as done",
    "show notes", "create note", "edit note", "delete note",
    "open file", "open folder", "show in folder", "reveal in finder",
    "move to trash", "empty trash", "restore",
    "rename", "duplicate", "make a copy", "move to",
    "compress", "extract", "zip", "unzip",
    "share", "airdrop", "send to", "upload", "download",
    "open with", "set default app", "get info", "properties",
    "run", "execute", "debug", "build", "compile", "test",
    "deploy", "publish", "release", "rollback",
    "commit", "push", "pull", "fetch", "merge", "rebase",
    "create branch", "switch branch", "delete branch",
    "create pr", "create pull request", "review pr",
    "open repo", "clone", "fork",
    "start server", "stop server", "restart server",
    "check status", "show logs", "tail logs",
    "run tests", "run lint", "format code",
    "open terminal", "new terminal", "split terminal",
    "clear", "history", "repeat last command",
    "list files", "show current directory", "change directory",
    "create file", "create directory", "remove file", "remove directory",
    "find file", "search in files", "grep",
    "show processes", "kill process", "show ports",
    "show disk usage", "show memory usage", "show cpu usage",
    "show network", "show ip", "ping", "traceroute",
    "ssh into", "connect to server", "disconnect",
    "start docker", "stop docker", "docker compose up", "docker compose down",
    "list containers", "list images", "remove container", "remove image",
    "start vm", "stop vm", "list vms",
    "edit file", "open in editor", "open in vscode", "open in pycharm",
    "open in browser", "open url", "go to website",
    "google", "search for", "look up", "find information about",
    "what is", "who is", "how to", "why is", "when is", "where is",
    "tell me about", "explain", "summarize", "elaborate",
    "write code", "fix bug", "refactor", "optimize",
    "add feature", "remove feature", "update dependency",
    "review code", "explain code", "document code",
    "generate", "create", "make", "build", "scaffold",
    "analyze", "inspect", "examine", "investigate",
    "monitor", "watch", "track", "observe",
    "compare", "diff", "contrast",
    "merge", "combine", "join", "split", "separate",
    "filter", "sort", "group", "organize",
    "increase", "decrease", "maximize", "minimize",
    "enable", "disable", "toggle", "switch",
    "show", "hide", "reveal", "conceal",
    "expand", "collapse", "fold", "unfold",
    "pin", "unpin", "favorite", "bookmark",
    "sync", "backup", "restore", "reset",
    "import", "export", "convert", "transform",
    "validate", "verify", "check", "confirm",
    "schedule", "plan", "organize", "arrange",
    "assign", "delegate", "transfer", "reassign",
    "approve", "reject", "accept", "decline",
    "subscribe", "unsubscribe", "follow", "unfollow",
    "like", "dislike", "upvote", "downvote",
    "comment", "reply", "respond", "answer",
    "post", "publish", "share", "broadcast",
    "notify", "alert", "warn", "inform",
    "remind", "prompt", "nudge", "ping",
    "greet", "welcome", "introduce", "present",
    "thank", "apologize", "congratulate", "encourage",
    "ask", "question", "inquire", "request",
    "suggest", "recommend", "propose", "advise",
    "agree", "disagree", "confirm", "deny",
    "allow", "deny", "permit", "forbid",
    "grant", "revoke", "authorize", "unauthorize",
    "encrypt", "decrypt", "encode", "decode",
    "compress", "decompress", "pack", "unpack",
    "serialize", "deserialize", "marshal", "unmarshal",
    "normalize", "denormalize", "sanitize", "escape",
    "interpolate", "extrapolate", "predict", "forecast",
    "train", "evaluate", "infer", "predict",
    "classify", "cluster", "regress", "embed",
    "tokenize", "detokenize", "encode", "decode",
    "preprocess", "postprocess", "augment", "synthesize",
    "fine-tune", "distill", "quantize", "prune",
    "benchmark", "profile", "trace", "instrument",
    "log", "metric", "trace", "span",
    "sample", "aggregate", "rollup", "downsample",
    "alert", "page", "escalate", "resolve",
    "acknowledge", "snooze", "silence", "unacknowledge",
    "triage", "diagnose", "troubleshoot", "remediate",
    "mitigate", "prevent", "detect", "respond",
    "recover", "restore", "failover", "fallback",
    "scale up", "scale down", "scale out", "scale in",
    "provision", "deprovision", "allocate", "deallocate",
    "reserve", "release", "acquire", "relinquish",
    "bootstrap", "initialize", "configure", "teardown",
    "warm up", "cool down", "prime", "flush",
    "hydrate", "dehydrate", "populate", "depopulate",
    "seed", "migrate", "rollback", "reset",
    "snapshot", "clone", "fork", "branch",
    "archive", "purge", "clean", "garbage collect",
    "compact", "vacuum", "reindex", "rebuild",
    "rotate", "cycle", "refresh", "renew",
    "expire", "invalidate", "evict", "purge",
    "throttle", "rate limit", "debounce", "batch",
    "queue", "dequeue", "enqueue", "publish",
    "subscribe", "unsubscribe", "bind", "unbind",
    "register", "deregister", "discover", "forget",
    "resolve", "lookup", "query", "mutate",
    "read", "write", "update", "delete", "patch",
    "get", "post", "put", "delete", "head", "options",
    "create", "read", "update", "delete", "list",
    "upsert", "insert", "select", "join", "union",
    "begin", "commit", "rollback", "savepoint",
    "lock", "unlock", "acquire", "release",
    "open", "close", "connect", "disconnect",
    "bind", "listen", "accept", "dial",
    "send", "receive", "broadcast", "multicast",
    "stream", "buffer", "flush", "drain",
    "pipe", "tee", "split", "merge",
    "map", "reduce", "filter", "scan",
    "fold", "unfold", "zip", "unzip",
    "take", "skip", "limit", "offset",
    "first", "last", "nth", "slice",
    "reverse", "shuffle", "sort", "unique",
    "group by", "order by", "having", "where",
    "inner join", "left join", "right join", "full join",
    "cross join", "natural join", "self join", "anti join",
    "union all", "intersect", "except", "minus",
    "with", "recursive", "window", "over",
    "partition by", "rows between", "range between",
    "lag", "lead", "first value", "last value", "nth value",
    "rank", "dense rank", "row number", "ntile",
    "sum", "count", "avg", "min", "max",
    "stddev", "variance", "percentile", "median",
    "corr", "covar", "regr", "cume dist",
    "rollup", "cube", "grouping sets", "pivot",
    "explain", "analyze", "vacuum", "reindex",
    "cluster", "replicate", "shard", "partition",
    "distribute", "broadcast", "coalesce", "repartition",
    "cache", "persist", "unpersist", "checkpoint",
    "checkpoint", "save", "load", "export", "import",
    "write", "read", "format", "mode",
    "overwrite", "append", "ignore", "error",
    "parquet", "orc", "avro", "json", "csv", "text",
    "delta", "iceberg", "hudi", "hive",
    "jdbc", "odbc", "mongodb", "cassandra", "redis",
    "kafka", "kinesis", "pulsar", "eventhub",
    "s3", "gcs", "abs", "adls", "minio",
    "hdfs", "nfs", "smb", "ftp", "sftp",
    "http", "https", "grpc", "thrift", "avro",
    "protobuf", "flatbuffers", "capn proto", "messagepack",
    "borsh", "bincode", "cbor", "ion", "flexbuffers",
    "base64", "hex", "url", "html", "xml", "yaml", "toml",
    "markdown", "rst", "asciidoc", "latex", "tex",
    "pdf", "docx", "xlsx", "pptx", "odt", "ods", "odp",
    "epub", "mobi", "azw", "djvu", "cbr", "cbz",
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp",
    "svg", "eps", "ai", "psd", "xcf", "sketch",
    "mp3", "wav", "flac", "aac", "ogg", "opus", "wma",
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "zst",
    "iso", "dmg", "vhd", "vmdk", "qcow2", "raw",
    "deb", "rpm", "apk", "ipa", "msi", "appimage",
    "snap", "flatpak", "nix", "guix", "brew",
    "docker", "oci", "podman", "containerd", "lxc", "lxd",
    "wasm", "wasi", "wasi preview 2", "component model",
    "wit", "world", "interface", "export", "import",
    "spin", "wasmtime", "wasmedge", "wasmer", "wazero",
    "javy", "wizer", "wasm-opt", "wasm-pack", "cargo wasi",
    "componentize", "wasm compose", "wac", "wasi-virt",
    "wasi-http", "wasi-keyvalue", "wasi-messaging", "wasi-sql",
    "wasi-blob", "wasi-config", "wasi-runtime-config",
    "wasi-logging", "wasi-cli", "wasi-clocks", "wasi-random",
    "wasi-filesystem", "wasi-sockets", "wasi-http-outgoing",
    "bytecode alliance", "wasmcloud", "fermyon", "cosmonic",
    "dapr", "keda", "wasmcloud", "spin", "fermyon",
    "crossplane", "upbound", "terraform operator", "ack",
    "strimzi", "rabbitmq operator", "postgres operator", "zalando",
    "cloudnative-pg", "stackgres", "percona operator", "mysql operator",
    "victoria metrics", "thanos", "cortex", "mimir", "grafana mimir",
    "tempo", "pyroscope", "phlare", "parca", "pixie",
    "opencost", "kubecost", "castai", "spot", "stormforge",
    "vcluster", "loft", "devtron", "shipa", "portainer",
    "kubeapps", "glasskube", "kubepak", "timoni", "werf",
    "devspace", "garden", "tilt", "skaffold", "okteto",
    "telepresence", "mirrord", "gefyra", "kubefwd", "ktunnel",
    "kubevpn", "vpnkit", "docker desktop", "rancher desktop",
    "orbstack", "colima", "lima", "finch", "podman machine",
    "multipass", "microk8s", "minikube", "kind", "k3d",
    "vagrant", "virtualbox", "vmware fusion", "parallels", "utm",
    "qemu", "libvirt", "virt-manager", "gnome boxes", "quickemu",
    "distrobox", "toolbox", "devbox", "devenv", "flox",
    "nix", "nixos", "nix-darwin", "home-manager", "nixpkgs",
    "guix", "guix system", "guix home", "guix pack",
    "brew", "linuxbrew", "macports", "fink", "pkgsrc",
    "conda", "mamba", "micromamba", "pixi", "rattler",
    "pip", "pipx", "poetry", "pdm", "hatch", "rye", "uv",
    "npm", "yarn", "pnpm", "bun", "deno", "volta", "fnm", "nvm",
    "cargo", "rustup", "go", "dotnet", "sdkman", "jabba",
    "asdf", "mise", "proto", "vfox", "aqua", "rtx",
    "direnv", "dotenv", "envchain", "envkey", "infisical",
    "starship", "oh-my-posh", "powerlevel10k", "spaceship", "pure",
    "nerd fonts", "powerline", "font awesome", "material icons",
    "catppuccin", "tokyo night", "nord", "dracula", "gruvbox",
    "rose pine", "everforest", "kanagawa", "onedark", "monokai",
    "solarized", "ayu", "palenight", "night owl", "github theme",
    "zellij", "tmux", "screen", "byobu", "mtm", "dvtm",
    "wezterm", "kitty", "alacritty", "foot", "ghostty", "rio",
    "warp", "iterm2", "hyper", "tabby", "terminus", "extraterm",
    "windows terminal", "conemu", "cmder", "mintty", "msys2",
    "putty", "mobaxterm", "securecrt", "royal tsx", "termius",
    "blink", "shelly", "a-shell", "ish", "termux", "juicessh",
    "code-server", "openvscode-server", "gitpod", "github codespaces",
    "coder", "cdr", "devpod", "envd", "gitpod flex",
    "replit", "stackblitz", "codesandbox", "glitch", "codepen",
    "jsfiddle", "playcode", "runjs", "quokkajs", "observable",
    "deepnote", "hex", "noteable", "colab", "kaggle", "sagemaker",
    "databricks", "snowflake", "bigquery", "redshift", "duckdb",
    "motherduck", "chdb", "libsql", "turso", "rqlite", "dqlite",
    "sqlite", "postgres", "mysql", "mariadb", "cockroachdb",
    "yugabyte", "tidb", "oceanbase", "vitess", "spanner",
    "alloydb", "aurora", "rds", "cloud sql", "azure sql",
    "supabase", "neon", "planetscale", "xata", "nile", "convex",
    "firebase", "appwrite", "pocketbase", "nhost", "directus",
    "payload", "strapi", "sanity", "contentful", "hygraph",
    "prismic", "storyblok", "kontent", "buttercms", "agility",
    "builder.io", "plasmic", "makeswift", "instant", "tina",
    "keystatic", "decap cms", "netlify cms", "prose", "cloudcannon",
    "siteleaf", "forestry", "tina cms", "spina cms", "statamic",
    "craft cms", "expressionengine", "modx", "processwire", "bolt",
    "grav", "getgrav", "pico", "bludit", "automad", "typo3",
    "neos", "contao", "typo3", "drupal", "joomla", "wordpress",
    "shopify", "bigcommerce", "woocommerce", "magento", "prestashop",
    "saleor", "medusa", "vendure", "swell", "commercetools",
    "elastic path", "fabric", "nopcommerce", "virto commerce",
    "orocommerce", "sylius", "solidus", "spree", "shuup",
    "sentry", "datadog", "newrelic", "dynatrace", "appdynamics",
    "instana", "honeycomb", "lightstep", "jaeger", "tempo",
    "signoz", "hyperdx", "highlight", "logrocket", "fullstory",
    "hotjar", "mouseflow", "crazyegg", "luckyorange", "clarity",
    "posthog", "mixpanel", "amplitude", "heap", "pendo", "indicative",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature", "flagd", "go-feature-flag", "flipt",
    "pagerduty", "opsgenie", "victorops", "splunk on-call",
    "incident.io", "firehydrant", "rootly", "incident labs",
    "blameless", "jeli", "transposit", "cortex", "opslevel",
    "statuspage", "hund", "instatus", "betterstack", "checkly",
    "pingdom", "uptime", "site24x7", "datadog synthetics",
    "splunk", "sumologic", "logz.io", "papertrail", "loggly",
    "datadog logs", "newrelic logs", "elastic", "opensearch",
    "meilisearch", "typesense", "algolia", "elasticsearch",
    "pinecone", "weaviate", "qdrant", "chroma", "milvus",
    "redis", "dragonfly", "keydb", "garnet", "valkey",
    "upstash", "momento", "readySet", "polyScale",
    "cloudflare", "fastly", "akamai", "bunny", "keycdn",
    "vercel", "netlify", "cloudflare pages", "deno deploy",
    "fly.io", "railway", "render", "koyeb", "porter",
    "heroku", "digitalocean", "linode", "vultr", "hetzner",
    "ovh", "scaleway", "upcloud", "exoscale", "civo",
    "aws", "gcp", "azure", "oracle cloud", "ibm cloud",
    "alibaba cloud", "tencent cloud", "huawei cloud", "baidu cloud",
    "openstack", "cloudstack", "opennebula", "apache cloudstack",
    "maas", "metal", "equinix", "packet", "phoenixnap",
    "leaseweb", "ovhcloud", "soyoustart", "kimsufi", "buyvm",
    "netcup", "contabo", "strato", "ionos", "hostinger",
    "namecheap", "godaddy", "cloudflare registrar", "porkbun",
    "hover", "gandi", "iwantmyname", "dnsimple", "dnsmadeeasy",
    "route53", "cloudflare dns", "google domains", "azure dns",
    "bunny dns", "ns1", "constellix", "dyn", "ultradns",
    "vercel domains", "netlify domains", "framer domains",
    "carrd", "framer", "webflow", "squarespace", "wix",
    "weebly", "strikingly", "duda", "jimdo", "site123",
    "wordpress.com", "ghost.org", "medium", "substack", "beehiiv",
    "convertkit", "mailchimp", "klaviyo", "drip", "customer.io",
    "sendgrid", "mailgun", "postmark", "resend", "plunk",
    "loops", "buttondown", "curated", "revue", "mailbrew",
    "hey", "fastmail", "protonmail", "tuta", "skiff", "mailbox.org",
    "zoho mail", "mxroute", "migadu", "purelymail", "forwardemail",
    "simplelogin", "anonaddy", "duckduckgo email", "firefox relay",
    "1password", "bitwarden", "dashlane", "nordpass", "keeper",
    "roboform", "sticky password", "zoho vault", "keeper",
    "proton pass", "heylogin", "passbolt", "padloc", "spectre",
    "lesspass", "masterpassword", "hashpass", "pwgen", "diceware",
    "yubikey", "solokey", "nitrokey", "onlykey", "trezor",
    "ledger", "keepkey", "bitbox", "coldcard", "passport",
    "seedsigner", "specter", "sparrow", "electrum", "wasabi",
    "samourai", "bluewallet", "muun", "breez", "phoenix",
    "zeus", "blink", "wallet of satoshi", "alby", "getalby",
    "nostr", "damus", "amethyst", "primal", "snort", "iris",
    "coracle", "nostrudel", "yakihonne", "highlighter", "zapstream",
    "bluesky", "threads", "mastodon", "pixelfed", "loops",
    "signal", "telegram", "whatsapp", "matrix", "element",
    "session", "simpleX", "briar", "jami", "keet", "wire",
    "threema", "status", "berty", "delta chat", "cwtch",
    "discord", "slack", "teams", "google chat", "mattermost",
    "rocket.chat", "zulip", "twist", "flock", "ryver",
    "basecamp", "asana", "monday", "clickup", "linear",
    "height", "plane", "huly", "taiga", "openproject",
    "jira", "confluence", "notion", "coda", "fibery",
    "airtable", "smartsheet", "quip", "dropbox paper", "slite",
    "almanac", "tettra", "guru", "bloomfire", "document360",
    "gitbook", "readme", "archbee", "mintlify", "fern",
    "redocly", "bump.sh", "stoplight", "swaggerhub", "postman",
    "insomnia", "hoppscotch", "bruno", "yaak", "httpie",
    "graphql playground", "altair", "graphiql", "apollo studio",
    "hasura console", "dgraph ratel", "arangodb webui", "neo4j browser",
    "redis insight", "mongo compass", "dbeaver", "tableplus",
    "datagrip", "navicat", "sequel ace", "sequel pro", "pgadmin",
    "phpmyadmin", "adminer", "sqlitebrowser", "db browser for sqlite",
    "beekeeper studio", "heidisql", "dbschema", "dbvisualizer",
    "valentina studio", "razorsql", "sqlectron", "dbgate",
    "azure data studio", "mysql workbench", "oracle sql developer",
    "robo 3t", "studio 3t", "nosqlbooster", "mongochef",
    "redis commander", "medis", "another redis desktop manager",
    "kafdrop", "kafka ui", "redpanda console", "akhq", "kpow",
    "conduktor", "offset explorer", "kcat", "kafkacat",
    "rabbitmq management", "nats dashboard", "natsboard",
    "elasticvue", "dejavu", "cerebro", "elasticsearch head",
    "opensearch dashboards", "kibana", "grafana", "chronograf",
    "prometheus", "alertmanager", "thanos", "cortex", "mimir",
    "victoriametrics", "vmui", "grafana explore", "grafana loki",
    "jaeger ui", "zipkin ui", "tempo query", "signoz frontend",
    "hyperdx", "highlight", "logrocket", "fullstory", "hotjar",
    "posthog", "mixpanel", "amplitude", "heap", "pendo",
    "launchdarkly", "split", "flagsmith", "growthbook", "unleash",
    "configcat", "devcycle", "hypertune", "statsig", "eppo",
    "openfeature flagd", "flipt", "go-feature-flag",
}

# Music providers
_MUSIC_PROVIDERS = {
    "spotify", "apple music", "youtube music", "amazon music",
    "tidal", "deezer", "pandora", "soundcloud", "bandcamp",
    "qobuz", "napster", "iheartradio", "tunein", "siriusxm",
    "gaana", "jiosaavn", "wynk", "hungama",
    "audiomack", "mixcloud", "beatport", "traxsource",
    "last.fm", "discogs", "musicbrainz", "genius",
    "shazam", "soundhound", "musixmatch",
    "mpv", "vlc", "rhythmbox", "clementine", "amarok",
    "audacious", "cmus", "moc", "mpd", "ncmpcpp",
    "foobar2000", "musicbee", "mediamonkey", "winamp",
    "aimp", "dopamine", "strawberry", "cantata",
    "elisa", "sayonara", "quod libet", "exaile",
    "lollypop", "gnome music", "amberol", "tauon",
    "nuclear", "funkwhale", "koel", "navidrome",
    "airsonic", "subsonic", "ampache", "mstream",
    "jellyfin", "plex", "emby", "kodi",
    "plexamp", "prism", "feishin", "sonixd",
    "sublime music", "tempo", "harmonoid", "museeks",
}

# ── Correction confidence thresholds ────────────────────────────
MIN_CORRECTION_CONFIDENCE = 0.90   # Must be ≥ this to apply a correction (raised from 0.85)
MIN_PARTIAL_CONFIDENCE = 0.85      # Lower threshold for partial/substring matches (raised from 0.80)
MAX_EDIT_DISTANCE = 3              # Max Levenshtein distance for short words

# ── "Do no harm" — common English words that must NEVER be corrected ──
# These are everyday words that the corrector should never replace, even if
# a fuzzy match exists in the dictionary. The corrector exists to fix Whisper
# mistakes like "pie charm" → "PyCharm", not to rewrite valid English.
_SAFE_COMMON_WORDS: Set[str] = {
    # Articles, prepositions, conjunctions
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "from",
    "with", "by", "as", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "has", "have", "had", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must",
    "and", "or", "but", "not", "no", "yes", "if", "then", "else",
    "when", "where", "why", "how", "what", "who", "whom", "which",
    "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "them", "we", "us", "you", "your", "my", "our", "their",
    "me", "him", "her", "his", "hers", "mine", "ours", "yours",
    "some", "any", "all", "both", "each", "every", "few", "many",
    "more", "most", "much", "such", "only", "own", "same", "so",
    "than", "too", "very", "just", "now", "then", "here", "there",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "everywhere", "nowhere", "somewhere",
    "always", "never", "sometimes", "often", "usually", "rarely",
    "already", "also", "even", "ever", "still", "yet", "ago",
    "almost", "enough", "hardly", "nearly", "quite", "rather",
    "really", "scarcely", "almost", "enough", "rather",

    # Common verbs
    "go", "went", "gone", "going", "come", "came", "coming",
    "get", "got", "gotten", "getting", "make", "made", "making",
    "take", "took", "taken", "taking", "give", "gave", "given",
    "put", "set", "let", "run", "ran", "running", "walk", "talk",
    "say", "said", "saying", "tell", "told", "telling", "ask",
    "see", "saw", "seen", "seeing", "look", "looking", "hear",
    "heard", "hearing", "listen", "feel", "felt", "feeling",
    "think", "thought", "thinking", "know", "knew", "known",
    "want", "need", "like", "love", "hate", "try", "trying",
    "work", "working", "play", "playing", "read", "reading",
    "write", "wrote", "written", "writing", "call", "called",
    "show", "showed", "shown", "showing", "find", "found",
    "keep", "kept", "keeping", "hold", "held", "holding",
    "bring", "brought", "bringing", "leave", "left", "leaving",
    "start", "started", "starting", "stop", "stopped", "stopping",
    "begin", "began", "begun", "beginning", "end", "ended",
    "open", "opened", "opening", "close", "closed", "closing",
    "turn", "turned", "turning", "move", "moved", "moving",
    "change", "changed", "changing", "help", "helped", "helping",
    "use", "used", "using", "check", "checked", "checking",
    "send", "sent", "sending", "receive", "received",
    "buy", "bought", "sell", "sold", "pay", "paid",
    "eat", "ate", "drink", "drank", "sleep", "slept",
    "live", "lived", "living", "die", "died", "dying",
    "wait", "waited", "waiting", "hope", "hoped", "hoping",
    "believe", "believed", "remember", "forget", "forgot",
    "understand", "understood", "learn", "learned", "teach",
    "build", "built", "building", "break", "broke", "broken",
    "cut", "fall", "fell", "fallen", "grow", "grew", "grown",
    "sit", "sat", "stand", "stood", "meet", "met",
    "speak", "spoke", "spoken", "sing", "sang", "sung",
    "win", "won", "lose", "lost", "choose", "chose", "chosen",
    "drive", "drove", "driven", "fly", "flew", "flown",
    "swim", "swam", "swum", "draw", "drew", "drawn",
    "throw", "threw", "thrown", "catch", "caught",
    "fight", "fought", "buy", "bought", "bring", "brought",
    "seek", "sought", "teach", "taught", "think", "thought",

    # Common nouns / time / directions
    "time", "day", "night", "week", "month", "year", "hour",
    "minute", "second", "morning", "afternoon", "evening",
    "today", "tomorrow", "yesterday", "now", "later", "soon",
    "thing", "things", "stuff", "people", "person", "man", "woman",
    "child", "kid", "friend", "family", "home", "house", "room",
    "door", "window", "table", "chair", "bed", "car", "book",
    "phone", "computer", "screen", "keyboard", "mouse",
    "water", "food", "money", "work", "job", "school", "office",
    "world", "life", "hand", "head", "eye", "ear", "mouth",
    "back", "front", "side", "top", "bottom", "left", "right",
    "north", "south", "east", "west", "center", "middle",
    "inside", "outside", "above", "below", "between", "behind",
    "way", "place", "part", "end", "beginning", "half", "whole",
    "name", "number", "word", "line", "page", "letter", "color",
    "size", "type", "kind", "sort", "form", "group", "set",
    "list", "file", "folder", "directory", "path", "link",
    "code", "data", "text", "image", "video", "audio", "music",
    "song", "sound", "voice", "noise", "silence", "light", "dark",
    "question", "answer", "problem", "solution", "idea", "thought",
    "message", "email", "mail", "call", "chat", "news", "story",
    "game", "movie", "show", "picture", "photo", "map", "app",
    "website", "site", "page", "tab", "window", "menu", "button",
    "search", "result", "history", "bookmark", "favorite",
    "password", "account", "user", "profile", "settings",
    "note", "reminder", "timer", "alarm", "calendar", "event",
    "task", "project", "plan", "goal", "step", "action",
    "error", "warning", "info", "debug", "log", "trace",
    "test", "check", "verify", "validate", "review",
    "request", "response", "input", "output", "result",
    "status", "state", "mode", "level", "value", "key",
    "version", "update", "upgrade", "patch", "fix", "bug",
    "feature", "release", "deploy", "build", "run", "start",
    "stop", "pause", "resume", "restart", "shutdown", "reboot",
    "install", "uninstall", "setup", "configure", "enable", "disable",
    "add", "remove", "delete", "create", "edit", "save", "load",
    "import", "export", "copy", "paste", "cut", "undo", "redo",
    "select", "deselect", "clear", "reset", "refresh", "reload",
    "zoom", "scroll", "swipe", "drag", "drop", "click", "tap",
    "press", "hold", "release", "type", "enter", "escape",
    "previous", "next", "first", "last", "back", "forward",
    "up", "down", "left", "right", "home", "end",
    "volume", "brightness", "contrast", "saturation",
    "mute", "unmute", "loud", "quiet", "soft", "hard",
    "fast", "slow", "high", "low", "big", "small",
    "good", "bad", "nice", "great", "fine", "okay", "ok",
    "new", "old", "young", "long", "short", "wide", "narrow",
    "hot", "cold", "warm", "cool", "wet", "dry", "clean", "dirty",
    "happy", "sad", "angry", "tired", "sick", "well", "better",
    "best", "worst", "worse", "easy", "hard", "simple", "complex",
    "true", "false", "right", "wrong", "correct", "incorrect",
    "full", "empty", "open", "closed", "free", "busy",
    "available", "unavailable", "online", "offline",
    "active", "inactive", "enabled", "disabled",
    "visible", "hidden", "shown", "locked", "unlocked",
    "public", "private", "shared", "personal",
    "local", "remote", "internal", "external",
    "current", "previous", "next", "last", "latest",
    "first", "second", "third", "fourth", "fifth",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "thousand", "million",
    "weather", "temperature", "humidity", "pressure", "wind",
    "rain", "snow", "sun", "cloud", "storm", "thunder",
    "today", "tonight", "tomorrow", "weekend", "holiday",
    "birthday", "anniversary", "meeting", "appointment",
    "breakfast", "lunch", "dinner", "coffee", "tea", "water",
    "hello", "hi", "hey", "bye", "goodbye", "thanks", "thank",
    "please", "sorry", "excuse", "welcome", "congratulations",
    "morning", "afternoon", "evening", "night", "goodnight",
    "goodbye", "farewell", "see you", "later", "take care",
    "how are you", "what's up", "how's it going",
    "joke", "story", "fact", "quote", "tip", "trick",
    "help", "support", "assistance", "guide", "tutorial",
    "documentation", "manual", "reference", "example",
    "sample", "template", "pattern", "snippet",
    "function", "method", "class", "object", "variable",
    "parameter", "argument", "return", "value", "type",
    "string", "number", "boolean", "array", "list", "map",
    "set", "queue", "stack", "tree", "graph", "node",
    "api", "endpoint", "route", "controller", "service",
    "model", "view", "component", "module", "package",
    "library", "framework", "platform", "tool", "utility",
    "database", "table", "column", "row", "index", "query",
    "server", "client", "host", "port", "protocol", "socket",
    "request", "response", "header", "body", "payload",
    "token", "session", "cookie", "cache", "storage",
    "cloud", "container", "image", "volume", "network",
    "firewall", "proxy", "gateway", "load balancer",
    "monitor", "alert", "metric", "dashboard", "report",
    "backup", "restore", "recovery", "failover", "redundancy",
    "encryption", "decryption", "hash", "signature", "certificate",
    "authentication", "authorization", "permission", "role",
    "policy", "rule", "condition", "action", "effect",
    "resource", "quota", "limit", "threshold", "budget",
    "cost", "price", "fee", "charge", "payment", "invoice",
    "subscription", "plan", "tier", "level", "package",
    "trial", "demo", "preview", "beta", "alpha", "stable",
    "production", "staging", "development", "testing", "sandbox",
    "environment", "workspace", "repository", "branch", "commit",
    "merge", "pull", "push", "fetch", "clone", "fork",
    "issue", "ticket", "bug", "feature", "enhancement",
    "sprint", "milestone", "release", "version", "tag",
    "pipeline", "workflow", "job", "stage", "step",
    "artifact", "dependency", "package", "module", "library",
    "compiler", "interpreter", "runtime", "engine", "virtual machine",
    "container", "orchestrator", "scheduler", "dispatcher",
    "queue", "topic", "stream", "event", "message",
    "producer", "consumer", "publisher", "subscriber",
    "source", "sink", "filter", "transform", "aggregate",
    "batch", "real-time", "streaming", "offline", "online",
    "synchronous", "asynchronous", "blocking", "non-blocking",
    "parallel", "concurrent", "sequential", "distributed",
    "centralized", "decentralized", "federated", "hybrid",
    "monolithic", "microservices", "serverless", "edge",
    "rest", "graphql", "grpc", "websocket", "sse",
    "json", "xml", "yaml", "toml", "csv", "protobuf",
    "http", "https", "tcp", "udp", "dns", "dhcp",
    "ip", "mac", "lan", "wan", "vpn", "vlan",
    "ssh", "ftp", "sftp", "smtp", "imap", "pop3",
    "sql", "nosql", "newsql", "acid", "base", "cap",
    "crud", "rest", "soap", "rpc", "mqtt", "amqp",
    "jwt", "oauth", "saml", "openid", "ldap", "kerberos",
    "aes", "rsa", "sha", "md5", "bcrypt", "scrypt",
    "tls", "ssl", "https", "hsts", "csp", "cors",
    "xss", "csrf", "sqli", "ddos", "mitm", "phishing",
    "agile", "scrum", "kanban", "waterfall", "devops",
    "ci", "cd", "cicd", "gitops", "devsecops", "mlops",
    "sre", "platform", "infrastructure", "architecture",
    "design", "pattern", "principle", "practice", "standard",
    "convention", "guideline", "best practice", "anti-pattern",
    "solid", "dry", "kiss", "yagni", "tdd", "bdd", "ddd",
    "mvc", "mvvm", "mvp", "flux", "redux", "bloc",
    "singleton", "factory", "builder", "prototype", "adapter",
    "bridge", "composite", "decorator", "facade", "flyweight",
    "proxy", "chain", "command", "interpreter", "iterator",
    "mediator", "memento", "observer", "state", "strategy",
    "template", "visitor", "null object", "specification",
    "repository", "unit of work", "service locator", "dependency injection",
    "event sourcing", "cqrs", "saga", "circuit breaker", "bulkhead",
    "retry", "timeout", "fallback", "cache aside", "read through",
    "write through", "write behind", "refresh ahead",
    "sharding", "partitioning", "replication", "clustering",
    "load balancing", "round robin", "least connections", "ip hash",
    "consistent hashing", "rendezvous hashing", "jump hash",
    "bloom filter", "hyperloglog", "count-min sketch", "t-digest",
    "raft", "paxos", "zab", "viewstamped replication", "chain replication",
    "two phase commit", "three phase commit", "saga", "outbox",
    "change data capture", "event sourcing", "materialized view",
    "oltp", "olap", "htap", "etl", "elt", "data warehouse",
    "data lake", "lakehouse", "data mesh", "data fabric",
    "batch processing", "stream processing", "complex event processing",
    "mapreduce", "spark", "flink", "beam", "kafka streams",
    "lambda", "kappa", "delta", "iceberg", "hudi",
    "parquet", "orc", "avro", "thrift", "protobuf", "flatbuffers",
    "snappy", "zstd", "lz4", "gzip", "bzip2", "xz",
    "columnar", "row-based", "hybrid", "in-memory", "disk-based",
    "btree", "lsm", "hash", "bitmap", "inverted", "full-text",
    "vector", "graph", "spatial", "temporal", "time-series",
    "relational", "document", "key-value", "wide-column", "object",
    "hierarchical", "network", "multi-model", "polymorphic",
    "acid", "base", "cap", "pacelc", "acid 2.0",
    "serializable", "repeatable read", "read committed", "read uncommitted",
    "snapshot isolation", "cursor stability", "optimistic", "pessimistic",
    "mvcc", "2pl", "to", "occ", "silo", "calvin",
    "b+tree", "lsm tree", "fractal tree", "mass tree", "bw tree",
    "write-ahead log", "redo log", "undo log", "binlog", "changelog",
    "checkpoint", "snapshot", "backup", "restore", "point-in-time recovery",
    "full backup", "incremental", "differential", "continuous",
    "rpo", "rto", "mtbf", "mttr", "mttd", "mtta",
    "sla", "slo", "sli", "error budget", "toil",
    "availability", "durability", "consistency", "partition tolerance",
    "latency", "throughput", "bandwidth", "jitter", "packet loss",
    "tail latency", "p50", "p95", "p99", "p999", "p9999",
    "mean", "median", "percentile", "histogram", "distribution",
    "apdex", "csat", "nps", "ces", "customer effort score",
    "kpi", "okr", "mbo", "smart", "balanced scorecard",
    "north star", "north star metric", "aarrr", "pirate metrics",
    "cac", "ltv", "churn", "retention", "engagement",
    "conversion", "acquisition", "activation", "revenue", "referral",
    "mrr", "arr", "arpu", "arppu", "cogs", "gross margin",
    "burn rate", "runway", "valuation", "cap table", "esop",
    "seed", "series a", "series b", "series c", "ipo", "exit",
    "angel", "vc", "pe", "accelerator", "incubator", "bootstrapped",
    "saas", "paas", "iaas", "faas", "baas", "caas", "daas",
    "b2b", "b2c", "b2b2c", "d2c", "c2c", "p2p",
    "marketplace", "platform", "ecosystem", "network effect",
    "flywheel", "moat", "competitive advantage", "value proposition",
    "product market fit", "product led growth", "sales led growth",
    "community led growth", "developer led growth", "partner led growth",
    "land and expand", "bottom up", "top down", "hybrid",
    "freemium", "free trial", "usage based", "seat based",
    "perpetual", "subscription", "consumption", "outcome based",
    "open core", "open source", "source available", "proprietary",
    "copyleft", "permissive", "apache", "mit", "gpl", "bsd",
    "agpl", "lgpl", "mpl", "epl", "unlicense", "cc0",
    "patent", "trademark", "copyright", "trade secret", "ip",
    "nda", "msa", "sow", "sla", "eula", "tos", "privacy policy",
    "gdpr", "ccpa", "hipaa", "soc2", "iso27001", "pci dss",
    "fedramp", "itar", "ear", "ofac", "fisma", "nist",
    "zero trust", "defense in depth", "least privilege", "need to know",
    "separation of duties", "dual control", "split knowledge",
    "data at rest", "data in transit", "data in use", "data in motion",
    "pii", "phi", "pci", "cdi", "intellectual property",
    "dlp", "siem", "soar", "xdr", "edr", "ndr", "mdr",
    "ids", "ips", "waf", "rasp", "runtime protection",
    "sast", "dast", "iast", "sca", "container scanning",
    "threat modeling", "attack surface", "vulnerability", "exploit",
    "cve", "cvss", "cwe", "owasp", "mitre", "attack",
    "kill chain", "diamond model", "mitre att&ck", "cyber kill chain",
    "reconnaissance", "weaponization", "delivery", "exploitation",
    "installation", "command and control", "actions on objectives",
    "initial access", "execution", "persistence", "privilege escalation",
    "defense evasion", "credential access", "discovery", "lateral movement",
    "collection", "exfiltration", "impact", "command and control",
    "phishing", "spear phishing", "whaling", "smishing", "vishing",
    "malware", "ransomware", "spyware", "adware", "rootkit", "bootkit",
    "trojan", "worm", "virus", "botnet", "backdoor", "rat",
    "keylogger", "screen scraper", "cryptominer", "wiper", "bricker",
    "zero day", "n day", "exploit", "payload", "shellcode",
    "buffer overflow", "heap overflow", "stack overflow", "integer overflow",
    "use after free", "double free", "null pointer dereference",
    "race condition", "time of check time of use", "toctou",
    "sql injection", "cross site scripting", "cross site request forgery",
    "server side request forgery", "xml external entity", "xxe",
    "insecure deserialization", "path traversal", "file inclusion",
    "command injection", "code injection", "template injection",
    "ldap injection", "xpath injection", "smtp injection", "imap injection",
    "header injection", "cookie injection", "crlf injection",
    "host header injection", "request smuggling", "response splitting",
    "cache poisoning", "dns poisoning", "arp spoofing", "dhcp spoofing",
    "session hijacking", "session fixation", "session replay",
    "man in the middle", "man in the browser", "evil twin",
    "rogue access point", "pineapple", "wifi deauth", "karma attack",
    "bluejacking", "bluesnarfing", "bluebugging", "car whisperer",
    "rfid cloning", "nfc relay", "side channel", "timing attack",
    "power analysis", "electromagnetic", "acoustic", "thermal",
    "fault injection", "voltage glitching", "clock glitching",
    "laser fault injection", "electromagnetic fault injection",
    "microarchitectural", "spectre", "meltdown", "foreshadow", "zombieload",
    "rowhammer", "rampage", "throwhammer", "drammer", "clkscrew",
    "plundervolt", "sgx", "trustzone", "secure enclave", "tpm",
    "hsm", "smart card", "sim card", "esim", "secure element",
    "tee", "confidential computing", "homomorphic encryption",
    "secure multi-party computation", "differential privacy",
    "federated learning", "split learning", "vertical federated learning",
    "zero knowledge proof", "zk snark", "zk stark", "bulletproofs",
    "ring signature", "group signature", "blind signature", "threshold signature",
    "multi sig", "mpc wallet", "hardware wallet", "cold storage",
    "hot wallet", "warm wallet", "custodial", "non-custodial",
    "self custody", "social recovery", "shamir backup", "seed phrase",
    "bip39", "bip32", "bip44", "bip84", "slip39",
    "hd wallet", "deterministic", "hierarchical deterministic",
    "mnemonic", "entropy", "checksum", "passphrase", "plausible deniability",
    "bitcoin", "ethereum", "solana", "cardano", "polkadot", "cosmos",
    "avalanche", "near", "flow", "algorand", "tezos", "stellar",
    "ripple", "litecoin", "dogecoin", "monero", "zcash", "dash",
    "defi", "nft", "dao", "dapp", "dex", "cex", "amm",
    "liquidity pool", "yield farming", "staking", "lending", "borrowing",
    "collateral", "liquidation", "oracle", "price feed", "keeper",
    "flash loan", "sandwich attack", "front running", "mev", "pbs",
    "layer 1", "layer 2", "rollup", "zk rollup", "optimistic rollup",
    "sidechain", "plasma", "state channel", "validium", "volition",
    "sharding", "danksharding", "proto-danksharding", "eip-4844",
    "blob", "data availability", "data availability sampling",
    "consensus", "proof of work", "proof of stake", "delegated proof of stake",
    "proof of authority", "proof of history", "proof of space", "proof of burn",
    "proof of capacity", "proof of elapsed time", "proof of importance",
    "byzantine fault tolerance", "practical byzantine fault tolerance",
    "tendermint", "hotstuff", "grandpa", "casper", "gasper",
    "nakamoto consensus", "longest chain", "heaviest chain", "ghost",
    "finality", "probabilistic finality", "absolute finality", "economic finality",
    "slashing", "staking", "delegation", "unbonding", "withdrawal",
    "validator", "miner", "staker", "delegator", "nominator",
    "full node", "light node", "archive node", "pruned node", "rpc node",
    "genesis", "block", "transaction", "receipt", "log", "event",
    "gas", "gas price", "gas limit", "base fee", "priority fee", "tip",
    "nonce", "signature", "public key", "private key", "address",
    "smart contract", "solidity", "vyper", "rust", "move", "cairo",
    "evm", "wasm", "ewasm", "move vm", "cairo vm", "fuel vm",
    "erc20", "erc721", "erc1155", "erc4626", "erc4337",
    "account abstraction", "smart account", "paymaster", "bundler",
    "eip", "erc", "rip", "bip", "sip", "cip", "pip",
    "hard fork", "soft fork", "contentious fork", "chain split",
    "upgrade", "migration", "regenesis", "state sync", "snapshot",
    "bridge", "wormhole", "layerzero", "chainlink", "ccip",
    "interoperability", "cross chain", "multi chain", "omni chain",
    "ibc", "xcm", "ics", "ica", "icq", "packet forward middleware",
    "relayer", "light client", "trusted bridge", "trustless bridge",
    "wrapped token", "synthetic asset", "stablecoin", "algorithmic stablecoin",
    "cdp", "vault", "collateralized debt position", "minting", "burning",
    "rebasing", "seigniorage", "fractional reserve", "overcollateralized",
    "dai", "usdc", "usdt", "frax", "lusd", "mim", "ust", "usdn",
    "curve", "uniswap", "sushiswap", "balancer", "bancor", "cowswap",
    "1inch", "paraswap", "matcha", "0x", "hashflow", "airswap",
    "aave", "compound", "maker", "liquity", "euler", "morpho",
    "yearn", "convex", "stakewise", "lido", "rocket pool", "frax",
    "synthetix", "uma", "ribbon", "hegic", "opyn", "dopex",
    "dydx", "perpetual protocol", "gmx", "gains network", "mux",
    "pendle", "notional", "element", "sense", "swivel", "yield",
    "ens", "unstoppable domains", "space id", "bonfida", "sid",
    "lens", "farcaster", "orb", "deso", "bitclout", "diamond",
    "friend tech", "stars arena", "post tech", "new bitcoin city",
    "worldcoin", "galxe", "rabbithole", "layer3", "questn",
    "safe", "gnosis safe", "argent", "sequence", "tor.us", "magic",
    "rainbow", "metamask", "phantom", "backpack", "keplr", "leap",
    "trust wallet", "exodus", "blockchain.com", "coinbase wallet",
    "zerion", "zapper", "debank", "rotki", "coinstats", "delta",
    "coingecko", "coinmarketcap", "defillama", "dune", "nansen",
    "messari", "the block", "coindesk", "decrypt", "blockworks",
    "bankless", "the defiant", "milk road", "cryptopragmatist",
    "a16z", "paradigm", "pantera", "polychain", "multicoin",
    "framework", "variant", "union square", "placeholder", "1confirmation",
    "dragonfly", "electric capital", "fabric ventures", "semantic ventures",
    "coinfund", "blockchain capital", "digital currency group", "galaxy",
    "wintermute", "jump", "gts", "cumberland", "b2c2", "folkvang",
    "amber", "alameda", "three arrows", "genesis", "blockfi", "celsius",
    "ftx", "binance", "coinbase", "kraken", "gemini", "bitstamp",
    "okx", "bybit", "kucoin", "gate", "huobi", "bitfinex",
    "deribit", "bitmex", "bitget", "mexc", "bingx", "phemex",
    "uniswap", "pancakeswap", "raydium", "orca", "jupiter", "trader joe",
    "spookyswap", "quickswap", "velodrome", "aerodrome", "camelot",
    "thena", "chronos", "ramses", "zyberswap", "solidly", "equalizer",
    "curve", "balancer", "saddle", "platypus", "wombat", "mav",
    "maverick", "ambient", "carbon", "bancor", "integral", "hashflow",
    "odos", "1inch", "paraswap", "cowswap", "matcha", "kyberswap",
    "openocean", "slingshot", "bebop", "dodo", "woofi", "zigzag",
    "across", "stargate", "synapse", "hop", "connext", "multichain",
    "celer", "wormhole", "layerzero", "chainlink ccip", "axelar",
    "thorchain", "maya", "chainflip", "squid", "bungee", "jumper",
    "lifi", "socket", "debridge", "orbiter", "rhino", "transit",
    "portal", "allbridge", "satellite", "nomad", "harmony bridge",
    "polygon bridge", "arbitrum bridge", "optimism bridge", "base bridge",
    "zksync bridge", "starknet bridge", "scroll bridge", "linea bridge",
    "mantle bridge", "mode bridge", "blast bridge", "zora bridge",
    "ethereum", "bitcoin", "solana", "polygon", "arbitrum", "optimism",
    "base", "zksync", "starknet", "scroll", "linea", "mantle",
    "avalanche", "fantom", "bsc", "gnosis", "celo", "moonbeam",
    "moonriver", "astar", "shiden", "acala", "karura", "phala",
    "crab", "darwinia", "khala", "bifrost", "interlay", "kintsugi",
    "centrifuge", "altair", "hydradx", "basilisk", "calamari", "manta",
    "zeitgeist", "subsocial", "polkadot", "kusama", "rococo", "westend",
    "cosmos hub", "osmosis", "juno", "evmos", "injective", "sei",
    "neutron", "stride", "quicksilver", "persistence", "akash",
    "sentinel", "regen", "ixo", "likecoin", "desmos", "bitsong",
    "comdex", "cheqd", "stargaze", "omniflix", "vidulum", "chihuahua",
    "teritori", "passage", "aura", "kujira", "mars", "astroport",
    "levana", "eris", "backbone labs", "apollo", "steak", "pryzm",
    "noble", "dydx", "celestia", "berachain", "monad", "megaeth",
    "eigenlayer", "restaking", "liquid restaking", "lrt", "avs",
    "eigenpod", "eigenda", "altlayer", "omni", "lagrange", "witness chain",
    "hyperlane", "polyhedra", "babylon", "lombard", "solayer", "jito",
    "sanctum", "margin fi", "kamino", "meteora", "orca whirlpool",
    "drift", "zeta", "parcl", "tensor", "magic eden", "blur",
    "opensea", "looksrare", "x2y2", "sudoswap", "nftx", "nftperp",
    "benddao", "jpegd", "nftfi", "arcade", "gondi", "blend",
    "sharky", "frakt", "rain", "tensorians", "mad lads", "claynosaurz",
    "degods", "y00ts", "abc", "goblintown", "pudgy penguins", "lil pudgys",
    "bored ape", "mutant ape", "cryptopunks", "meebits", "cool cats",
    "doodles", "clone x", "moonbirds", "proof", "azuki", "bean",
    "milady", "redacted", "rektguy", "rekt drinks", "rekt",
    "world of women", "wow", "invisible friends", "veefriends",
    "art blocks", "fidenza", "ringers", "chromie squiggle", "gazers",
    "lost poetics", "memes by 6529", "the currency", "damien hirst",
    "pak", "beeple", "xcopy", "hackatao", "fewocious", "deekay",
    "refik anadol", "snowfro", "tyler hobbs", "dmitri cherniak",
    "matt deslauriers", "emily xie", "ix shells", "william mapan",
    "larva labs", "yuga labs", "proof collective", "wenew", "dapper labs",
    "nbatopshot", "nfl all day", "ufc strike", "laliga", "sorare",
    "axie infinity", "stepn", "gmt", "gst", "move to earn",
    "play to earn", "p2e", "m2e", "x to earn", "create to earn",
    "sandbox", "decentraland", "voxels", "somnium space", "nft worlds",
    "worldwide webb", "treeverse", "nifty island", "highstreet",
    "illuvium", "big time", "guild of guardians", "ember sword",
    "phantom galaxies", "star atlas", "aurory", "genopets", "pegaxy",
    "defi kingdoms", "crabada", "cryptounicorns", "cryptoblades",
    "splinterlands", "gods unchained", "skyweaver", "parallel",
    "shrapnel", "off the grid", "deadrop", "dr disrepect", "midnight society",
    "wildcard", "ev.io", "elixir", "synergy land", "undead blocks",
    "thetan arena", "heroes of mavia", "blocklords", "serum", "star atlas",
    "raini", "my neighbor alice", "alien worlds", "uplift", "prospectors",
    "farsite", "influence", "dark forest", "conquest", "primodium",
    "curio", "treaty", "sovryn", "babylon", "stacking dao", "pstake",
    "ankr", "stader", "stafi", "lido", "rocket pool", "frax ether",
    "swell", "stakewise", "tranchess", "asymetrix", "unagii",
    "conic", "clever", "stakedao", "paladin", "sturdy", "gravita",
    "prisma", "raft", "lybra", "eigenpie", "renzo", "kelp dao",
    "ether fi", "puffer", "bedrock", "stakestone", "karak", "symbiotic",
    "mitosis", "allora", "nillion", "movement", "initia", "babylon",
    "saga", "dymension", "avail", "0g", "og labs", "fuel", "sui",
    "aptos", "linera", "radix", "kaspa", "ergo", "nervos",
    "iron fish", "alephium", "spacemesh", "chia", "mass", "quai",
    "aleo", "aztec", "midnight", "namada", "penumbra", "nym",
    "hopr", "sentinel", "mysterium", "orchid", "nyx", "firo",
    "pirate chain", "arrr", "dero", "secret network", "oasis",
    "pha", "automata", "obscuro", "ten", "sapphire", "opacity",
    "lit", "nucypher", "threshold", "keep", "enigma", "partisia",
    "arpa", "coti", "findora", "zama", "tfhe", "fhe", "concrete",
    "openfhe", "helib", "seal", "palisade", "lattigo", "tfhe-rs",
    "concrete ml", "zama ai", "inco", "fhenix", "mind network",
    "sight ai", "privanet", "blindnet", "cover", "nexus mutual",
    "insurace", "etherisc", "uno re", "nsure", "bridge mutual",
    "risk harbor", "solace", "cozy", "armor", "bright union",
    "degis", "insurace", "tidal", "umbrella", "polkacover",
    "chainlink", "band protocol", "tellor", "api3", "umbrella network",
    "dIA", "pyth", "switchboard", "redstone", "chronicle", "supra",
    "witnet", "razor", "bridge oracle", "nest", "dos", "umbrella",
    "gelato", "keep3r", "chainlink automation", "openzeppelin defender",
    "autotask", "hal", "pocket", "ankr", "infura", "alchemy",
    "quicknode", "moralis", "blast api", "llamanodes", "drpc",
    "publicnode", "1rpc", "omniatech", "chainstack", "blockdaemon",
    "figment", "chorus one", "stakefish", "p2p", "everstake",
    "allnodes", "staked", "stake capital", "stakin", "infstones",
    "rockx", "dsrv", "hashquark", "forbole", "b harvest", "certus one",
    "stakewith us", "stake zone", "stake baby", "smart stake",
    "stake service", "stake facilities", "stake capital", "stake.fish",
    "stakewise", "stake dao", "stakeborg", "stakehound", "stakewise",
    "lido", "rocket pool", "frax", "stader", "ankr", "pstake",
    "stafi", "tranchess", "stakewise", "swell", "ether fi", "renzo",
    "puffer", "kelp", "eigenpie", "bedrock", "stakestone", "karak",
    "symbiotic", "mitosis", "babylon", "lombard", "solayer", "jito",
    "sanctum", "picasso", "composable", "centauri", "hyperspace",
    "tanssi", "polkadot parachain", "kusama parachain", "parachain",
    "relay chain", "collator", "xcmp", "hrmp", "dmp", "ump",
    "gavin wood", "vitalik buterin", "charles hoskinson", "anatoly yakovenko",
    "raj gokal", "mo shaikh", "avery ching", "illia polosukhin",
    "alex skidanov", "mustafa al-bassam", "zaki manian", "sunny aggarwal",
    "zhuoxun yin", "sergey nazarov", "stant kulechov", "robert leshner",
    "hayden adams", "andre cronje", "daniel sesta", "michael egorov",
    "sam kazemian", "kain warwick", "0xsifu", "cobie", "dcf god",
    "gcr", "light", "hsaka", "inversebrah", "ansem", "blknoiz06",
    "pentoshi", "cantering clark", "trader xo", "crypto cred", "teddy",
    "donalt", "crypto capo", "il capo of crypto", "crypto banter",
    "coin bureau", "lark davis", "bitboy", "altcoin daily", "ivan on tech",
    "aantonop", "andreas antonopoulos", "nic carter", "lynn alden",
    "peter mccormack", "marty bent", "parker lewis", "gigi", "dergigi",
    "saifedean ammous", "stephan livera", "preston pysh", "robert breedlove",
    "michael saylor", "jack dorsey", "elon musk", "cathie wood",
    "ray dalio", "paul tudor jones", "stanley druckenmiller", "bill miller",
    "tim draper", "barry silbert", "brian armstrong", "changpeng zhao",
    "sam bankman-fried", "do kwon", "su zhu", "kyle davies",
    "alex mashinsky", "mashinsky", "celcius", "blockfi", "genesis",
    "gemini", "winklevoss", "tyler winklevoss", "cameron winklevoss",
    "barry silbert", "dcg", "grayscale", "genesis trading", "foundry",
    "luno", "coindesk", "tradeblock", "coindesk indices", "coindesk tv",
    "consensus", "invest", "ethereal", "fluidity", "crypto compare",
    "kaiko", "skew", "laevitas", "glassnode", "coinmetrics", "messari",
    "the block", "delphi digital", "hashed", "1kx", "dragonfly",
    "multicoin", "polychain", "paradigm", "a16z crypto", "variant",
    "electric", "framework", "pantera", "placeholder", "union square",
    "usv", "collaborative fund", "boost vc", "chapter one", "founders fund",
    "sequoia", "benchmark", "accel", "index", "lightspeed", "greylock",
    "bessemer", "insight", "general catalyst", "khosla", "kleiner perkins",
    "andreessen horowitz", "a16z", "y combinator", "techstars", "500 startups",
    "angelpad", "alchemist", "south park commons", "homebrew", "first round",
    "true ventures", "floodgate", "felicis", "redpoint", "battery",
    "matrix", "nea", "ivp", "meritech", "tcv", "ga", "insight",
    "thoma bravo", "vista", "silver lake", "tpg", "kkr", "blackstone",
    "carlyle", "apollo", "ares", "oaktree", "fortress", "cerberus",
    "goldman sachs", "morgan stanley", "jpmorgan", "citigroup", "bank of america",
    "wells fargo", "ubs", "credit suisse", "deutsche bank", "barclays",
    "hsbc", "bnp paribas", "societe generale", "santander", "bbva",
    "ing", "rabobank", "abn amro", "nordea", "danske bank",
    "swedbank", "seb", "dnb", "handelsbanken", "op financial",
    "blackrock", "vanguard", "state street", "fidelity", "charles schwab",
    "td ameritrade", "e-trade", "robinhood", "webull", "public",
    "sofi", "betterment", "wealthfront", "acorns", "stash",
    "m1 finance", "titan", "personal capital", "empower", "sigfig",
    "futureadvisor", "blooom", "ellevest", "lincoln", "prudential",
    "metlife", "aig", "allianz", "axa", "generali", "zurich",
    "chubb", "travelers", "liberty mutual", "progressive", "geico",
    "state farm", "allstate", "nationwide", "usaa", "farmers",
    "american family", "erie", "auto-owners", "cincinnati", "hartford",
    "berkshire hathaway", "geico", "general re", "national indemnity",
    "applied underwriters", "bi berkshire", "berkley", "arch", "everest re",
    "munich re", "swiss re", "hannover re", "scor", "partnerre",
    "transatlantic", "axis", "validus", "renaissancere", "aspen",
    "lancashire", "hiscox", "beazley", "catlin", "kiln",
    "amlin", "canopius", "novae", "brit", "chubb europe",
    "allianz global corporate", "axa xl", "aig europe", "zurich insurance",
    "generali global corporate", "hdI", "talon", "mapfre", "qbe",
    "iag", "suncorp", "allianz australia", "qbe australia", "insurance australia",
    "sompo", "tokio marine", "mitsui sumitomo", "aioi nissay dowa", "nipponkoa",
    "samsung fire", "hyundai marine", "dongbu", "kb insurance", "db insurance",
    "meritz", "hanwha", "lotte", "mg", "heungkuk", "korean re",
    "korean reinsurance", "china life", "ping an", "china pacific", "picc",
    "china reinsurance", "taiping", "people's insurance", "china united",
    "sunshine insurance", "huatai", "dajia", "evergrande", "anbang",
    "fosun", "zhong an", "ant group", "tencent", "alibaba", "baidu",
    "jd.com", "pinduoduo", "meituan", "didi", "xiaomi", "oppo",
    "vivo", "huawei", "zte", "lenovo", "tsmc", "foxconn",
    "samsung", "lg", "sk hynix", "hyundai", "kia", "hybe",
    "kakao", "naver", "coupang", "krafton", "ncsoft", "nexon",
    "netmarble", "pearl abyss", "smilegate", "com2us", "gamevil",
    "wemade", "weMade", "gravity", "webzen", "neowiz", "devsisters",
    "shift up", "project moon", "round8", "nat games", "haegin",
    "supercell", "rovio", "king", "zynga", "niantic", "scopely",
    "jam city", "pocket gems", "machine zone", "mz", "epic games",
    "unity", "roblox", "electronic arts", "activision blizzard", "take-two",
    "ubisoft", "square enix", "capcom", "bandai namco", "sega",
    "konami", "koei tecmo", "fromsoftware", "platinumgames", "kojima productions",
    "cd projekt red", "techland", "11 bit studios", "people can fly",
    "bloober team", "flying wild hog", "ci games", "reikon games",
    "warhorse studios", "bohemia interactive", "wube software", "larian studios",
    "remedy", "supergiant games", "thatgamecompany", "giant squid", "annapurna",
    "devolver digital", "raw fury", "team17", "curve digital", "humble games",
    "chucklefish", "concernedape", "eric barone", "toby fox", "lucas pope",
    "jonathan blow", "phil fish", "derek yu", "edmund mcmillen", "tommy refenes",
    "tomorrow corporation", "zachtronics", "amanita design", "ustwo", "simogo",
    "playdead", "campo santo", "fullbright", "the chinese room", "frictional games",
    "red barrels", "bloober team", "tangentlemen", "ice-pick lodge", "tale of tales",
    "thatgamecompany", "giant sparrow", "funomena", "ko-op mode", "glitch factory",
    "heart machine", "drinkbox studios", "capybara games", "klei entertainment",
    "double fine", "obsidian", "inexile", "larian", "owlcat", "harebrained schemes",
    "beamdog", "overhaul games", "nightdive studios", "aspyr", "feral interactive",
    "virtual programming", "codeweavers", "transgaming", "playonlinux", "lutris",
    "proton", "wine", "dxvk", "vkd3d", "dxvk-nvapi", "mangohud",
    "goverlay", "gamescope", "steam deck", "steam os", "chimeraos", "holoiso",
    "bazzite", "nobara", "garuda", "pop os", "ubuntu gamepack", "drauger os",
    "lakka", "batocera", "retropie", "recalbox", "emuelec", "amberelec",
    "jelos", "arkos", "muos", "minui", "onion os", "garlic os",
    "allium", "koriki", "gammaos", "daijisho", "pegasus", "emulationstation",
    "retroarch", "dolphin", "pcsx2", "rpcs3", "xenia", "cemu",
    "yuzu", "ryujinx", "citra", "melonDS", "desmume", "drastic",
    "ppsspp", "vita3k", "flycast", "redream", "duckstation", "epsxe",
    "mupen64plus", "project64", "snes9x", "bsnes", "mesen", "fceux",
    "nestopia", "genesis plus gx", "kega fusion", "picodrive", "blastem",
    "mame", "finalburn neo", "fbneo", "demul", "supermodel", "model 2",
    "teknoParrot", "sega model 3", "naomi", "atomiswave", "cave", "pgm",
    "cps1", "cps2", "cps3", "neogeo", "neogeo cd", "pc engine",
    "turbografx", "supergrafx", "wonderswan", "game gear", "master system",
    "gameboy", "gameboy color", "gameboy advance", "nintendo ds", "nintendo 3ds",
    "nintendo switch", "wii", "wii u", "gamecube", "nintendo 64",
    "super nintendo", "nes", "playstation", "playstation 2", "playstation 3",
    "playstation 4", "playstation 5", "psp", "ps vita", "xbox",
    "xbox 360", "xbox one", "xbox series x", "xbox series s", "dreamcast",
    "saturn", "sega cd", "32x", "atari 2600", "atari 7800", "atari jaguar",
    "3do", "cdi", "neo geo pocket", "ngage", "game.com", "virtual boy",
    "vectrex", "intellivision", "colecovision", "magnavox odyssey", "fairchild channel f",
    "commodore 64", "amiga", "atari st", "zx spectrum", "amstrad cpc",
    "msx", "msx2", "pc-88", "pc-98", "x68000", "fm towns",
    "apple ii", "macintosh", "windows 95", "windows 98", "windows xp",
    "ms-dos", "pc dos", "dr dos", "freeDOS", "os/2", "beos",
    "amigaos", "morphos", "aros", "risc os", "haiku", "syllable",
    "reactos", "kolibriOS", "menuetOS", "templeOS", "serenityOS", "toaruOS",
    "plan 9", "inferno", "9front", "harvey", "jehanne", "akaros",
    "barrelfish", "helenos", "fiasco", "l4", "sel4", "pistachio",
    "okl4", "nova", "genode", "fuchsia", "zircon", "minix",
    "redox", "theseus", "tock", "hubris", "aero", "asterinas",
    "linux", "freebsd", "openbsd", "netbsd", "dragonfly bsd", "illumos",
    "smartos", "omnios", "openindiana", "tribblix", "dilos", "openSXCE",
    "solaris", "aix", "hp-ux", "irix", "tru64", "unixware",
    "openserver", "xenix", "venix", "coherent", "minix", "qnx",
    "vxworks", "integrity", "lynxos", "pikeos", "threadx", "freertos",
    "zephyr", "nuttx", "mbed", "riot", "contiki", "tinyos",
    "embox", "rtems", "ecos", "mynewt", "apache nffs", "arm mbed",
    "azure rtos", "threadx", "embos", "micrium", "ucos", "freertos",
    "safeRTOS", "openRTOS", "ti-rtos", "dsp/bios", "sys/bios", "pruss",
    "android", "ios", "ipados", "watchos", "tvos", "visionos",
    "macos", "windows", "linux", "chromeos", "fuchsia", "harmonyos",
    "hyperos", "coloros", "originos", "funtouch os", "magic os", "one ui",
    "miui", "emui", "realme ui", "oxygenos", "nothing os", "zenui",
    "myux", "xperia ui", "lg ux", "samsung experience", "touchwiz", "grace ux",
    "sense", "htc sense", "flyme", "smartisan os", "nubia ui", "redmagic os",
    "blackberry 10", "bb10", "playbook os", "webos", "lg webos", "palm os",
    "symbian", "series 60", "series 80", "uiq", "moe", "bada",
    "tizen", "kaios", "sailfish os", "ubuntu touch", "plasma mobile", "postmarketos",
    "mobian", "droidian", "ubports", "lomiri", "phosh", "sxmo",
    "pureos", "librem 5", "pinephone", "pinetab", "pinebook pro", "rockpro64",
    "raspberry pi", "raspberry pi 5", "raspberry pi 4", "raspberry pi zero", "orange pi",
    "banana pi", "odroid", "rock pi", "nano pi", "beaglebone", "arduino",
    "esp32", "esp8266", "rp2040", "rp2350", "stm32", "nrf52",
    "nrf53", "nrf91", "cc2650", "cc1350", "msp430", "pic",
    "avr", "attiny", "atmega", "samd", "samd21", "samd51",
    "nrf52840", "nrf5340", "nrf9160", "nrf7002", "da1469x", "da1470x",
    "efr32", "efm32", "mg24", "bg24", "xg27", "xg28",
    "psoc", "psoc 6", "psoc 4", "psoc 5", "traveo", "xmc",
    "aurix", "tricore", "rh850", "rl78", "rx", "sh",
    "h8", "superh", "m32r", "m16c", "r8c", "v850",
    "propeller", "parallax", "basic stamp", "picaxe", "microbit", "calliope",
    "circuit playground", "adafruit", "sparkfun", "seeed", "dfrobot", "pololu",
    "sparkfun", "adafruit", "pimoroni", "waveshare", "keyestudio", "elecrow",
    "makerfocus", "sunfounder", "freenove", "elegoo", "creality", "bambu lab",
    "prusa", "voron", "ratrig", "annex", "vzbot", "switchwire",
    "ender", "cr-10", "anycubic", "phrozen", "elegoo", "uniformation",
    "formlabs", "ultimaker", "makerbot", "lulzbot", "prusa", "bambu",
    "snapmaker", "artillery", "sovol", "comgrow", "longer", "flashforge",
    "qidi", "mingda", "tronxy", "geeetech", "anet", "tevo",
    "jgaurora", "wanhao", "monoprice", "xyzprinting", "da vinci", "robo",
    "printrbot", "kossel", "rostock", "delta", "corexy", "h-bot",
    "cartesian", "scara", "polar", "belt", "conveyor", "infinite z",
    "idex", "tool changer", "mmu", "ams", "palette", "mosaic",
    "eraser", "enraged rabbit carrot feeder", "ercf", "voron ercf", "tradrack",
    "box turtle", "nightowl", "happy hare", "filamentalist", "filametrix",
    "klipper", "marlin", "repetier", "smoothieware", "grbl", "fluidnc",
    "octoprint", "mainsail", "fluidd", "klipperscreen", "moonraker", "crowsnest",
    "obico", "spaghetti detective", "octoeverywhere", "simplyprint", "astroprint",
    "polar cloud", "creality cloud", "bambu handy", "bambu studio", "orca slicer",
    "prusa slicer", "super slicer", "cura", "ideaMaker", "simplify3d",
    "slic3r", "kisslicer", "skeinforge", "mattercontrol", "repetier host",
    "pronterface", "printrun", "octoprint", "botqueue", "repetier server",
    "3d printer", "filament", "pla", "abs", "petg", "tpu",
    "tpe", "nylon", "polycarbonate", "asa", "hips", "pva",
    "bvoh", "woodfill", "metalfill", "carbon fiber", "glass fiber", "kevlar",
    "peek", "pek", "ultem", "pei", "pp", "pom",
    "delrin", "acetal", "acrylic", "pmma", "polypropylene", "polyethylene",
    "hdpe", "ldpe", "ptfe", "teflon", "pctfe", "pvdf",
    "etfe", "fep", "pfa", "ectfe", "mfa", "thv",
    "silicone", "rubber", "latex", "neoprene", "viton", "epdm",
    "nitrile", "buna", "sbr", "natural rubber", "polyurethane", "pu",
    "eva", "foam", "sponge", "cork", "felt", "fabric",
    "leather", "suede", "canvas", "denim", "nylon", "polyester",
    "cotton", "wool", "silk", "linen", "hemp", "bamboo",
    "carbon", "kevlar", "fiberglass", "aramid", "spectra", "dyneema",
    "uhmwpe", "vectran", "technora", "twaron", "nomex", "zylon",
    "boron", "silicon carbide", "alumina", "zirconia", "tungsten carbide", "diamond",
    "cbn", "cubic boron nitride", "pcd", "polycrystalline diamond", "ceramic", "cermet",
    "aluminum", "steel", "stainless steel", "titanium", "magnesium", "copper",
    "brass", "bronze", "zinc", "nickel", "chrome", "chromium",
    "cobalt", "tungsten", "molybdenum", "vanadium", "manganese", "silicon",
    "germanium", "gallium", "indium", "tin", "lead", "bismuth",
    "antimony", "tellurium", "selenium", "arsenic", "cadmium", "mercury",
    "gold", "silver", "platinum", "palladium", "rhodium", "iridium",
    "osmium", "ruthenium", "rhenium", "hafnium", "tantalum", "niobium",
    "zirconium", "yttrium", "scandium", "lanthanum", "cerium", "praseodymium",
    "neodymium", "promethium", "samarium", "europium", "gadolinium", "terbium",
    "dysprosium", "holmium", "erbium", "thulium", "ytterbium", "lutetium",
    "actinium", "thorium", "protactinium", "uranium", "neptunium", "plutonium",
    "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
    "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium",
    "bohrium", "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium",
    "nihonium", "flerovium", "moscovium", "livermorium", "tennessine", "oganesson",
    "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
    "nitrogen", "oxygen", "fluorine", "neon", "sodium", "magnesium",
    "aluminum", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
    "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium",
    "manganese", "iron", "cobalt", "nickel", "copper", "zinc",
    "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum",
    "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium",
    "indium", "tin", "antimony", "tellurium", "iodine", "xenon",
    "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
    "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium",
    "holmium", "erbium", "thulium", "ytterbium", "lutetium", "hafnium",
    "tantalum", "tungsten", "rhenium", "osmium", "iridium", "platinum",
    "gold", "mercury", "thallium", "lead", "bismuth", "polonium",
    "astatine", "radon", "francium", "radium", "actinium", "thorium",
    "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium",
    "berkelium", "californium", "einsteinium", "fermium", "mendelevium", "nobelium",
    "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium", "hassium",
    "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
    "moscovium", "livermorium", "tennessine", "oganesson",
}


class SpeechCorrector:
    """Local speech correction layer.

    Builds a dynamic dictionary from the user's desktop environment
    and corrects Whisper mistakes using fuzzy matching. NEVER invokes
    an LLM — all corrections are local and deterministic.
    """

    def __init__(self):
        self._terms: Set[str] = set()
        self._term_list: List[str] = []
        self._loaded = False
        self._load_time: float = 0.0
        self._stats = {
            "total_corrections": 0,
            "corrections_applied": 0,
            "corrections_skipped": 0,
            "terms_loaded": 0,
        }

    # ── Public API ──────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def term_count(self) -> int:
        return len(self._terms)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def initialize(self) -> bool:
        """Scan the system and build the correction dictionary.

        This is intentionally synchronous — it runs once at boot in an
        executor thread. Returns True if terms were loaded.
        """
        if self._loaded:
            return True

        t0 = time.time()
        terms: Set[str] = set()

        # 1. Installed applications (.desktop files)
        terms.update(self._scan_desktop_apps())

        # 2. Project directories (common locations)
        terms.update(self._scan_projects())

        # 3. Git repositories
        terms.update(self._scan_git_repos())

        # 4. Browser bookmarks
        terms.update(self._scan_bookmarks())

        # 5. Common folders in home directory
        terms.update(self._scan_folders())

        # 6. Fallback terms (always available)
        terms.update(_FALLBACK_APPS)
        terms.update(_COMMON_FOLDERS)
        terms.update(_DESKTOP_COMMANDS)
        terms.update(_MUSIC_PROVIDERS)

        # Normalize: lowercase, strip, dedupe
        normalized: Set[str] = set()
        for t in terms:
            t = t.strip().lower()
            if t and len(t) >= 2:
                normalized.add(t)
                # Also add capitalized version for proper nouns
                if not t[0].isupper():
                    normalized.add(t.title())

        self._terms = normalized
        self._term_list = sorted(normalized, key=len, reverse=True)
        self._loaded = True
        self._load_time = time.time() - t0
        self._stats["terms_loaded"] = len(self._terms)

        logger.info("[CORRECTOR] Loaded %d correction terms in %.1fs "
                     "(apps=%d projects=%d repos=%d bookmarks=%d folders=%d)",
                     len(self._terms), self._load_time,
                     len(self._scan_desktop_apps()),
                     len(self._scan_projects()),
                     len(self._scan_git_repos()),
                     len(self._scan_bookmarks()),
                     len(self._scan_folders()))
        return True

    def refresh(self) -> None:
        """Re-scan the system for new terms (call periodically)."""
        self._loaded = False
        self._terms.clear()
        self._term_list.clear()
        self.initialize()

    def correct(self, text: str) -> Tuple[str, List[dict]]:
        """Apply fuzzy corrections to a transcript.

        Returns (corrected_text, correction_log). Each correction log
        entry is a dict with: word, corrected_to, confidence, method.

        Only corrects when confidence is high (≥ MIN_CORRECTION_CONFIDENCE).
        NEVER invokes an LLM.
        """
        if not text or not self._loaded:
            return text, []

        self._stats["total_corrections"] += 1
        corrections: List[dict] = []
        words = text.split()
        corrected_words: List[str] = []

        for word in words:
            corrected, conf, method = self._find_best_match(word)
            if corrected and conf >= MIN_CORRECTION_CONFIDENCE:
                corrected_words.append(corrected)
                corrections.append({
                    "word": word,
                    "corrected_to": corrected,
                    "confidence": round(conf, 3),
                    "method": method,
                })
                self._stats["corrections_applied"] += 1
            else:
                corrected_words.append(word)
                if corrected and conf < MIN_CORRECTION_CONFIDENCE:
                    self._stats["corrections_skipped"] += 1

        result = " ".join(corrected_words)
        if corrections:
            logger.info("[CORRECTOR] Applied %d corrections: %s → %s",
                        len(corrections), text, result)
            for c in corrections:
                logger.debug("[CORRECTOR]   '%s' → '%s' (%.3f, %s)",
                             c["word"], c["corrected_to"],
                             c["confidence"], c["method"])

        return result, corrections

    def correct_phrase(self, text: str) -> Tuple[str, List[dict]]:
        """Correct multi-word phrases in addition to individual words.

        Tries to match the full phrase and sub-phrases against known terms
        before falling back to word-by-word correction.
        """
        if not text or not self._loaded:
            return text, []

        self._stats["total_corrections"] += 1
        corrections: List[dict] = []
        result = text

        # Try full-phrase match first
        best_match, best_conf, best_method = self._find_best_phrase_match(text.lower())
        if best_match and best_conf >= MIN_CORRECTION_CONFIDENCE:
            corrections.append({
                "word": text,
                "corrected_to": best_match,
                "confidence": round(best_conf, 3),
                "method": f"phrase_{best_method}",
            })
            self._stats["corrections_applied"] += 1
            return best_match, corrections

        # Try sub-phrase matches (sliding window of 2-4 words)
        words = text.split()
        if len(words) >= 2:
            for window in range(min(4, len(words)), 1, -1):
                for i in range(len(words) - window + 1):
                    sub = " ".join(words[i:i + window])
                    best_match, best_conf, best_method = self._find_best_phrase_match(sub.lower())
                    if best_match and best_conf >= MIN_CORRECTION_CONFIDENCE:
                        words[i:i + window] = [best_match]
                        corrections.append({
                            "word": sub,
                            "corrected_to": best_match,
                            "confidence": round(best_conf, 3),
                            "method": f"subphrase_{best_method}",
                        })
                        self._stats["corrections_applied"] += 1
                        result = " ".join(words)
                        # Recurse to handle remaining words
                        return self.correct_phrase(result)

        # Fall back to word-by-word correction
        return self.correct(text)

    # ── Internal: system scanning ──────────────────────────

    @staticmethod
    def _scan_desktop_apps() -> Set[str]:
        """Scan .desktop files for installed application names."""
        apps: Set[str] = set()
        desktop_dirs = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local" / "share" / "applications",
            Path("/var/lib/flatpak/exports/share/applications"),
            Path.home() / ".local" / "share" / "flatpak" / "exports" / "share" / "applications",
            Path("/var/lib/snapd/desktop/applications"),
        ]
        for d in desktop_dirs:
            if not d.exists():
                continue
            try:
                for f in d.glob("*.desktop"):
                    try:
                        name = SpeechCorrector._parse_desktop_name(f)
                        if name:
                            apps.add(name.lower())
                    except Exception:
                        pass
            except PermissionError:
                pass
        return apps

    @staticmethod
    def _parse_desktop_name(path: Path) -> Optional[str]:
        """Extract the application Name from a .desktop file."""
        try:
            # .desktop files may have non-UTF-8 content; read as bytes
            raw = path.read_bytes()
            # Try UTF-8 first, fall back to latin-1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="replace")
            for line in text.splitlines():
                if line.startswith("Name="):
                    name = line[5:].strip()
                    if name and not name.startswith("("):
                        return name
        except Exception:
            pass
        return None

    @staticmethod
    def _scan_projects() -> Set[str]:
        """Scan common project directories for project names."""
        projects: Set[str] = set()
        search_dirs = [
            Path.home() / "projects",
            Path.home() / "Projects",
            Path.home() / "dev",
            Path.home() / "Dev",
            Path.home() / "development",
            Path.home() / "Development",
            Path.home() / "code",
            Path.home() / "Code",
            Path.home() / "workspace",
            Path.home() / "Workspace",
            Path.home() / "src",
            Path.home() / "Src",
            Path.home() / "repos",
            Path.home() / "Repos",
            Path.home() / "git",
            Path.home() / "Git",
            Path.home() / "PycharmProjects",
            Path.home() / "IdeaProjects",
            Path.home() / "GoProjects",
            Path.home() / "rust",
            Path.home() / "Rust",
        ]
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            try:
                for child in d.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        projects.add(child.name)
            except PermissionError:
                pass
        return projects

    @staticmethod
    def _scan_git_repos() -> Set[str]:
        """Find git repository names in common locations."""
        repos: Set[str] = set()
        search_dirs = [
            Path.home() / "projects",
            Path.home() / "Projects",
            Path.home() / "dev",
            Path.home() / "Dev",
            Path.home() / "development",
            Path.home() / "Development",
            Path.home() / "code",
            Path.home() / "Code",
            Path.home() / "workspace",
            Path.home() / "Workspace",
            Path.home() / "src",
            Path.home() / "Src",
            Path.home() / "repos",
            Path.home() / "Repos",
            Path.home() / "git",
            Path.home() / "Git",
            Path.home() / "PycharmProjects",
            Path.home() / "IdeaProjects",
        ]
        for d in search_dirs:
            if not d.exists() or not d.is_dir():
                continue
            try:
                for child in d.iterdir():
                    if child.is_dir() and (child / ".git").exists():
                        repos.add(child.name)
                        # Also try to get the remote origin name
                        try:
                            remote = SpeechCorrector._get_git_remote_name(child)
                            if remote:
                                repos.add(remote)
                        except Exception:
                            pass
            except PermissionError:
                pass
        return repos

    @staticmethod
    def _get_git_remote_name(repo_path: Path) -> Optional[str]:
        """Extract the repo name from git remote origin URL."""
        try:
            config = repo_path / ".git" / "config"
            if not config.exists():
                return None
            text = config.read_text(errors="replace")
            in_remote = False
            for line in text.splitlines():
                if line.strip() == '[remote "origin"]':
                    in_remote = True
                elif in_remote and line.strip().startswith("url ="):
                    url = line.split("=", 1)[1].strip()
                    # Extract repo name from URL
                    name = url.rstrip("/").split("/")[-1]
                    if name.endswith(".git"):
                        name = name[:-4]
                    return name
                elif in_remote and line.strip().startswith("["):
                    in_remote = False
        except Exception:
            pass
        return None

    @staticmethod
    def _scan_bookmarks() -> Set[str]:
        """Scan browser bookmark files for bookmark names."""
        bookmarks: Set[str] = set()
        browsers = [
            ("google-chrome", Path.home() / ".config" / "google-chrome"),
            ("chromium", Path.home() / ".config" / "chromium"),
            ("brave", Path.home() / ".config" / "BraveSoftware" / "Brave-Browser"),
            ("edge", Path.home() / ".config" / "microsoft-edge"),
            ("firefox", Path.home() / ".mozilla" / "firefox"),
            ("opera", Path.home() / ".config" / "opera"),
            ("vivaldi", Path.home() / ".config" / "vivaldi"),
        ]
        for _browser_name, base in browsers:
            if not base.exists():
                continue
            try:
                for bm_path in base.rglob("Bookmarks"):
                    if bm_path.is_file():
                        try:
                            data = json.loads(bm_path.read_text(errors="replace"))
                            bookmarks.update(
                                SpeechCorrector._extract_bookmark_names(data))
                        except (json.JSONDecodeError, Exception):
                            pass
            except PermissionError:
                pass
        return bookmarks

    @staticmethod
    def _extract_bookmark_names(node: dict) -> Set[str]:
        """Recursively extract bookmark names from Chrome-format JSON."""
        names: Set[str] = set()
        if isinstance(node, dict):
            if node.get("type") == "url" and "name" in node:
                name = node["name"].strip()
                if name and len(name) >= 2:
                    names.add(name)
            for key in ("children", "roots"):
                if key in node:
                    child = node[key]
                    if isinstance(child, list):
                        for c in child:
                            names.update(SpeechCorrector._extract_bookmark_names(c))
                    elif isinstance(child, dict):
                        for c in child.values():
                            names.update(SpeechCorrector._extract_bookmark_names(c))
        return names

    @staticmethod
    def _scan_folders() -> Set[str]:
        """Scan common folders in the home directory."""
        folders: Set[str] = set()
        home = Path.home()
        common = [
            "Desktop", "Documents", "Downloads", "Music", "Pictures",
            "Videos", "Templates", "Public", "Projects", "workspace",
            "dev", "code", "src", "repos", "git",
            "screenshots", "wallpapers", "backups", "notes",
            "books", "pdfs", "ebooks", "audiobooks",
            "podcasts", "recordings", "screencasts",
            "scripts", "tools", "bin", ".local/bin",
            "configs", "dotfiles", "env",
        ]
        for name in common:
            p = home / name
            if p.exists() and p.is_dir():
                folders.add(name.lower())
        return folders

    # ── Internal: fuzzy matching ───────────────────────────

    def _find_best_match(self, word: str) -> Tuple[Optional[str], float, str]:
        """Find the best fuzzy match for a single word.

        Returns (corrected_word, confidence, method).
        """
        word_lower = word.lower().strip()
        if len(word_lower) < 2:
            return None, 0.0, ""

        # 0. "Do no harm" — never correct common English words
        if word_lower in _SAFE_COMMON_WORDS:
            return None, 1.0, "safe_word"

        # 1. Exact match — no correction needed
        if word_lower in self._terms:
            return None, 1.0, "exact"

        best_match = None
        best_conf = 0.0
        best_method = ""

        # 2. Check against all known terms
        for term in self._term_list:
            # Skip terms that are too different in length
            if abs(len(term) - len(word_lower)) > MAX_EDIT_DISTANCE:
                continue

            # Full ratio
            ratio = _fuzzy_ratio(word_lower, term)
            if ratio > best_conf:
                best_conf = ratio
                best_match = term
                best_method = "fuzzy_full"

            # Partial ratio (for sub-word matches)
            if len(word_lower) >= 3 and len(term) >= 3:
                partial = _fuzzy_partial_ratio(word_lower, term)
                if partial > best_conf and partial >= MIN_PARTIAL_CONFIDENCE:
                    best_conf = partial
                    best_match = term
                    best_method = "fuzzy_partial"

        # 3. Special handling for very short words (2-3 chars)
        if len(word_lower) <= 3 and best_conf < MIN_CORRECTION_CONFIDENCE:
            # For short words, also check if it's a substring of a known term
            for term in self._term_list:
                if word_lower in term and len(term) <= len(word_lower) + 3:
                    conf = len(word_lower) / len(term)
                    if conf > best_conf:
                        best_conf = conf
                        best_match = term
                        best_method = "substring"

        if best_match and best_conf >= MIN_CORRECTION_CONFIDENCE:
            # Preserve original casing
            if word[0].isupper():
                best_match = best_match[0].upper() + best_match[1:]
            elif word.isupper():
                best_match = best_match.upper()
            return best_match, best_conf, best_method

        return None, best_conf, best_method

    def _find_best_phrase_match(self, phrase: str) -> Tuple[Optional[str], float, str]:
        """Find the best fuzzy match for a multi-word phrase.

        Returns (corrected_phrase, confidence, method).
        """
        if len(phrase) < 3:
            return None, 0.0, ""

        # 0. "Do no harm" — never correct phrases made entirely of safe words
        phrase_words = set(phrase.lower().split())
        if phrase_words.issubset(_SAFE_COMMON_WORDS):
            return None, 1.0, "safe_phrase"

        # 1. Exact match
        if phrase in self._terms:
            return None, 1.0, "exact"

        best_match = None
        best_conf = 0.0
        best_method = ""

        # 2. Check against all known terms
        for term in self._term_list:
            if " " not in term:
                continue  # Skip single-word terms for phrase matching

            # Length check
            len_diff = abs(len(term) - len(phrase))
            if len_diff > max(5, len(phrase) // 2):
                continue

            ratio = _fuzzy_ratio(phrase, term)
            if ratio > best_conf:
                best_conf = ratio
                best_match = term
                best_method = "fuzzy_phrase"

            # Also try partial ratio
            partial = _fuzzy_partial_ratio(phrase, term)
            if partial > best_conf and partial >= MIN_PARTIAL_CONFIDENCE:
                best_conf = partial
                best_match = term
                best_method = "partial_phrase"

        if best_match and best_conf >= MIN_CORRECTION_CONFIDENCE:
            return best_match, best_conf, best_method

        return None, best_conf, best_method


# Global singleton
speech_corrector = SpeechCorrector()