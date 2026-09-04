"""
wake_resolver — THE canonical wake-model resolver for Diego.

ONE discovery logic used by BOTH:
  - the runtime loader (voice/wake_model_manager.WakeModelManager.load)
  - the startup health probe (core/runtime_health.py)

This guarantees: same candidate list → same selected model → same
diagnostics. The historical bug was that the health probe only checked
`import openwakeword` + the existence of models/wake/verifier.pkl while
the runtime loader resolved a base ONNX model with its own (different)
logic — so health reported READY while the runtime reported
"Wake model not found (no candidate)".

Resolution priority (identical for health and runtime):
  1. Explicit configuration: WAKE_MODEL env → settings.WAKE_MODEL →
     voice_settings.wake_model (tried as-given, then relative to BASE_DIR)
  2. Custom trained model: models/wake/*.onnx (non-resource files only)
  3. Bundled model recorded in models/wake/metadata.json ("base_model")
  4. Bundled model best matching the configured wake phrase

Bundled-model directories are discovered WITHOUT requiring a successful
`import openwakeword` at resolution time (a transient import failure
during concurrent startup previously produced an empty candidate list →
"no candidate"). Candidate dirs:
  a. the active openwakeword package's resources/models (if importable
     or already imported)
  b. the USER site-packages openwakeword resources/models
     (site.getusersitepackages() — pip --user installs live here and are
     NOT returned by site.getsitepackages())
  c. every site-packages dir from site.getsitepackages()
  d. project-local models/wake/bundled/
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Files in the openWakeWord resources dir that are not wake-word models.
NON_WAKE_FILES = {"melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"}

# Bundled fallback used only when no model matches the wake phrase.
DEFAULT_BUNDLED_MODEL = "hey_marvin"

# Minimum difflib similarity for phrase→model matching; below this the
# DEFAULT_BUNDLED_MODEL is used as a generic verifier base.
MIN_PHRASE_MATCH_RATIO = 0.3


@dataclass
class WakeModelResolution:
    """Structured result of one wake-model resolution.

    Shared verbatim by the health probe and the runtime loader so both
    always report the same selected model and the same diagnostics.
    """
    path: Optional[Path] = None
    source: str = ""                      # which priority step selected it
    candidates: List[Tuple[str, str]] = field(default_factory=list)
    verifier_path: Optional[Path] = None
    wake_phrase: str = ""
    base_model_requested: str = ""        # metadata.json base_model, if any

    @property
    def found(self) -> bool:
        return self.path is not None and self.path.exists()

    def diagnostics(self) -> dict:
        """Serializable diagnostics — identical for health and runtime."""
        return {
            "found": self.found,
            "path": str(self.path) if self.path else None,
            "source": self.source,
            "verifier_path": str(self.verifier_path) if self.verifier_path else None,
            "wake_phrase": self.wake_phrase,
            "base_model_requested": self.base_model_requested,
            "candidates": [f"{src}:{p}" for src, p in self.candidates],
        }

    def reason(self) -> str:
        """Human-readable reason for the resolution outcome."""
        if self.found:
            return f"resolved {self.path} (source={self.source})"
        if self.candidates:
            return ("no wake model found; tried: "
                    + "; ".join(f"{src}={p}" for src, p in self.candidates))
        return ("no wake model candidate found (no configuration, no custom "
                "model, no bundled models discoverable)")


def _models_wake_dir() -> Path:
    from config.settings import settings
    return settings.MODELS_WAKE_DIR


def _load_verifier_metadata() -> dict:
    meta_path = _models_wake_dir() / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("[WAKE-RESOLVER] metadata read failed: %s", e)
        return {}


def _bundled_model_dirs() -> List[Path]:
    """All candidate directories that may contain bundled openWakeWord models.

    Does NOT raise and does NOT depend on `import openwakeword`
    succeeding: the user-site and getsitepackages paths cover pip --user
    installs even when the import transiently fails during startup.
    """
    dirs: List[Path] = []

    # a) Active openwakeword package (only if already imported or importable)
    #    Newer openwakeword ships models under `resources/models`; older
    #    releases used `models` directly — BOTH are candidates so the
    #    resolver works across package layout changes.
    oww = sys.modules.get("openwakeword")
    oww_pkg_dir: Optional[Path] = None
    if oww is not None and getattr(oww, "__file__", None):
        oww_pkg_dir = Path(oww.__file__).parent
    else:
        try:
            import openwakeword as _oww  # noqa: F401
            oww_pkg_dir = Path(_oww.__file__).parent
        except Exception as e:
            logger.debug("[WAKE-RESOLVER] openwakeword import unavailable: %s", e)
    if oww_pkg_dir is not None:
        dirs.append(oww_pkg_dir / "resources" / "models")
        dirs.append(oww_pkg_dir / "models")
        # openwakeword.utils may expose a configurable MODEL_PATH
        try:
            from openwakeword.utils import MODEL_PATH as _OWW_MODEL_PATH
            dirs.append(Path(str(_OWW_MODEL_PATH)))
        except Exception as e:
            logger.debug("[WAKE-RESOLVER] openwakeword.utils.MODEL_PATH "
                         "unavailable: %s", e)

    # b) USER site-packages (pip --user installs — the common case for
    #    this project; site.getsitepackages() does NOT include it).
    try:
        user_site = site.getusersitepackages()
        if user_site:
            dirs.append(Path(user_site) / "openwakeword" / "resources" / "models")
    except Exception as e:
        logger.debug("[WAKE-RESOLVER] usersite lookup failed: %s", e)

    # c) System site-packages list
    try:
        for sp in site.getsitepackages():
            dirs.append(Path(sp) / "openwakeword" / "resources" / "models")
    except Exception as e:
        logger.debug("[WAKE-RESOLVER] sitepackages lookup failed: %s", e)

    # d) Project-local bundled models directory
    dirs.append(_models_wake_dir() / "bundled")

    # Deduplicate, keep order
    out: List[Path] = []
    seen: set = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def bundled_models() -> Dict[str, Path]:
    """{model_stem: Path} for all bundled openWakeWord models found."""
    out: Dict[str, Path] = {}
    for d in _bundled_model_dirs():
        try:
            if not d.exists():
                continue
            for f in sorted(d.glob("*.onnx")):
                if f.name in NON_WAKE_FILES:
                    continue
                out.setdefault(f.stem, f)
        except Exception as e:
            logger.debug("[WAKE-RESOLVER] glob error in %s: %s", d, e)
    return out


def _select_bundled_for_phrase(phrase: str) -> Tuple[Optional[str], float]:
    """Best bundled model stem for a wake phrase, or (None, -1.0)."""
    models = bundled_models()
    if not models:
        return None, -1.0
    phrase_lower = (phrase or "").lower().strip()
    best_name = DEFAULT_BUNDLED_MODEL
    best_ratio = -1.0
    for name in models:
        pretty = name.replace("_", " ")
        ratio = difflib.SequenceMatcher(None, phrase_lower, pretty).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name
    if best_ratio < MIN_PHRASE_MATCH_RATIO:
        best_name = DEFAULT_BUNDLED_MODEL
    return best_name, best_ratio


def resolve_verifier(base_model_name: str) -> Optional[Path]:
    """Resolve a verifier valid for the given base model, or None.

    Same policy as the runtime loader: the verifier attaches only when it
    was trained on the selected base model (metadata.json base_model).
    """
    meta = _load_verifier_metadata()
    if meta.get("base_model") and meta["base_model"] != base_model_name:
        return None

    wake_dir = _models_wake_dir()
    named = meta.get("verifier_path")
    candidates: List[Path] = []
    if named:
        candidates.append(wake_dir / str(named))
    candidates.append(wake_dir / "verifier.pkl")
    candidates.append(wake_dir / "verifier.joblib")
    try:
        candidates += sorted(wake_dir.glob("verifier.*"))
    except Exception:
        pass

    seen: set = set()
    for c in candidates:
        key = str(c)
        if key in seen or not c.exists():
            continue
        seen.add(key)
        return c
    return None


def resolve_wake_model(wake_phrase: str = "") -> WakeModelResolution:
    """THE canonical wake-model resolution.

    Used by BOTH the health probe and the runtime loader. Never raises.
    """
    from config.settings import settings
    from voice.settings import voice_settings

    phrase = wake_phrase or voice_settings.wake_phrase or settings.WAKE_PHRASE
    res = WakeModelResolution(wake_phrase=phrase)
    wake_dir = _models_wake_dir()

    # ── 1) Explicit configuration ─────────────────────────────────
    cfg = (os.getenv("WAKE_MODEL")
           or getattr(settings, "WAKE_MODEL", None)
           or voice_settings.wake_model)
    if cfg:
        p = Path(str(cfg)).expanduser()
        res.candidates.append(("WAKE_MODEL(as-given)", str(p)))
        if p.exists():
            res.path, res.source = p, "WAKE_MODEL"
            res.verifier_path = resolve_verifier(p.stem)
            return res
        p2 = settings.BASE_DIR / str(cfg)
        res.candidates.append(("WAKE_MODEL(base_dir)", str(p2)))
        if p2.exists():
            res.path, res.source = p2, "WAKE_MODEL(base_dir)"
            res.verifier_path = resolve_verifier(p2.stem)
            return res
        logger.warning("[WAKE-RESOLVER] WAKE_MODEL='%s' not found at %s or %s",
                       cfg, p, p2)

    # ── 2) Custom trained model in models/wake/ ────────────────────
    try:
        for f in sorted(wake_dir.glob("*.onnx")):
            if f.name in NON_WAKE_FILES:
                continue
            res.candidates.append(("custom models/wake", str(f)))
            res.path, res.source = f, "custom models/wake"
            res.verifier_path = resolve_verifier(f.stem)
            return res
    except Exception as e:
        logger.debug("[WAKE-RESOLVER] models/wake glob error: %s", e)

    # ── 3) Bundled model recorded in verifier metadata ────────────
    meta = _load_verifier_metadata()
    base = meta.get("base_model")
    if base:
        res.base_model_requested = str(base)
        p = bundled_models().get(str(base))
        res.candidates.append(("metadata.base_model",
                               str(p) if p else f"{base} (not found)"))
        if p is not None:
            res.path, res.source = p, "metadata.base_model"
            res.verifier_path = resolve_verifier(str(base))
            return res

    # ── 4) Bundled model best matching the wake phrase ────────────
    stem, ratio = _select_bundled_for_phrase(phrase)
    if stem is not None:
        p = bundled_models().get(stem)
        res.candidates.append(("phrase-match", str(p) if p else f"{stem} (not found)"))
        if p is not None:
            res.path, res.source = p, f"phrase-match(ratio={ratio:.2f})"
            res.verifier_path = resolve_verifier(stem)
            return res

    return res