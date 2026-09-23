"""Phase 22 computer-use foundation package (additive; existing systems untouched)."""
from computer.action_result import ActionOutcome, ActionResult
from computer.action_policy import ActionRisk, assess_risk, confirmation_required

__all__ = [
    "ActionOutcome", "ActionResult", "ActionRisk", "assess_risk",
    "confirmation_required",
]


def __getattr__(name):
    """Lazy re-exports: heavy modules import their backends only on use."""
    if name == "ComputerController":
        from computer.computer_controller import ComputerController
        return ComputerController
    if name == "computer_controller":
        from computer.computer_controller import computer_controller
        return computer_controller
    if name == "ComputerState":
        from computer.screen_state import ComputerState
        return ComputerState
    if name == "find_element":
        from computer.element_finder import find_element
        return find_element
    if name == "ElementMatch":
        from computer.element_finder import ElementMatch
        return ElementMatch
    raise AttributeError(f"computer has no attribute '{name}'")

