"""Phase 21E — ISOLATED PROTOTYPE (NOT wired to production Brain).

Role-based reasoning-model router: selects which local Ollama model to use
for each reasoning phase based on task complexity, phase, context size, and
a latency budget. Preserves the existing ``BaseReasoningModel`` interface;
``for_phase`` returns an ``OllamaReasoningModel`` pinned to a model name.

Isolation contract (do NOT violate without a new phase):
  - Nothing imports this module from production code (Brain, TaskRunner,
    agent/, core/).
  - Constructing the router never changes global settings, env vars, or
    the shared Ollama client cache.
  - 21E evidence says: DO NOT route planning to a smaller model. The only
    conditionally-safe route found was 3B for diagnose/reflect; planning
    and revise stay on the production model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

from ai.reasoning_model import BaseReasoningModel, OllamaReasoningModel

ReasoningPhase = Literal["plan", "revise", "diagnose", "reflect"]

# 21E-measured defaults (single local run; medians, see
# benchmarks/reasoning_model_dataset.json + reasoning_21e_results_*.json):
#   plan(3B) p50 ~7.4s  | revise(3B) ~5.2s | diagnose(3B) ~6.8s | reflect(3B) ~6.6s
#   plan(1.5B) p50 ~8.8s (worse + unsafe) | plan(0.5B) p50 ~5.8s (unsafe garbage)
# Smaller models FAILED planning quality/safety, so they are never selected
# for plan/revise here.
_DEFAULT_MODEL_BY_PHASE: Dict[str, str] = {
    "plan": "qwen2.5:3b",      # production reasoning model — never downgrade
    "revise": "qwen2.5:3b",    # revise decides keep/replan — never downgrade
    "diagnose": "qwen2.5:3b",  # safe default; 1.5B-class allowed only w/ strict gate
    "reflect": "qwen2.5:3b",   # deferred already; 1.5B-class allowed only w/ strict gate
}

# Minimum bar a fast-model diagnose/reflect proposal must clear, else the
# caller must fall back to the strong model. Mirrors the 10-criterion
# structure of the 21E scorer (observable structured output only).
_FALLBACK_CHECKS = (
    "valid_json",
    "known_actions_only",
    "params_schema_valid",
    "no_shell_or_exec_action",
)


@dataclass
class ReasoningModelRouter:
    """Select a pinned ``OllamaReasoningModel`` per reasoning phase.

    Args:
        strong_model: model for plan/revise (default: production 3B).
        fast_model: candidate for diagnose/reflect overflow only.
        allow_fast_diagnose_reflect: keep False unless a future phase proves
            a fast model clears the fallback checks on the fixed dataset.
        latency_budget_s: advisory only — never skips verification or
            authorization; used only to prefer the fast model for reflect.
        strong_model_max_ctx_tokens: above this, force strong model.
    """

    strong_model: str = "qwen2.5:3b"
    fast_model: str = "qwen2.5:1.5b"
    allow_fast_diagnose_reflect: bool = False
    latency_budget_s: float = 8.0
    strong_model_max_ctx_tokens: int = 3000
    _cache: Dict[str, BaseReasoningModel] = field(default_factory=dict, init=False)

    def for_phase(
        self,
        phase: ReasoningPhase,
        *,
        goal_complexity: Literal["simple", "difficult"] = "simple",
        context_tokens: int = 0,
        latency_budget_s: float | None = None,
    ) -> BaseReasoningModel:
        """Return the model for ``phase``. Never returns a fast model for
        plan/revise; fast diagnose/reflect only when explicitly allowed AND
        the goal is simple AND context fits AND budget is tight."""
        _ = goal_complexity  # difficult goals always stay on strong model
        budget = self.latency_budget_s if latency_budget_s is None else latency_budget_s
        use_fast = (
            self.allow_fast_diagnose_reflect
            and phase in ("diagnose", "reflect")
            and goal_complexity == "simple"
            and 0 <= context_tokens <= self.strong_model_max_ctx_tokens
            and budget <= 8.0
        )
        name = self.fast_model if use_fast else _DEFAULT_MODEL_BY_PHASE[phase].replace(
            "qwen2.5:3b", self.strong_model
        )
        if phase in ("plan", "revise"):
            name = self.strong_model
        if name not in self._cache:
            model = OllamaReasoningModel(model=name)
            model._checked = True  # pinned: no discovery, no shared-cache mutation
            self._cache[name] = model
        return self._cache[name]

    def fallback_checks(self) -> tuple:
        """Quality gate names a fast-model proposal must satisfy."""
        return _FALLBACK_CHECKS

    async def aclose(self) -> None:
        for m in self._cache.values():
            close = getattr(m, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 — prototype must never raise on close
                    pass
        self._cache.clear()
