from typing import Literal, TypedDict

# The six reasons stopping.decide() can return (see stopping.py).
StopReason = Literal["continue", "target_reached", "voi_exhausted", "plateau", "budget", "no_feasible"]


class AgentState(TypedDict):
    # Rule 1 (the pointer): everything else is looked up in the DB through it. Set once at the start.
    inquiry_id: str
    # Rules 2 + 3: written by the planner each round (overwritten). The edge sends one researcher per
    # ID, and each ID is handed to that researcher. IDs, not wording: the DB stays the source of truth.
    question_ids: list[str]
    # Rule 2: written by check_stop (overwritten). The edge after check_stop routes on it:
    # "continue" -> planner, "no_feasible" -> interview, any other stop -> tasks.
    stop_reason: StopReason


class ResearcherPayload(TypedDict):
    # What one Send hands one researcher. A researcher started by Send gets ONLY this, not AgentState.
    question_id: str
