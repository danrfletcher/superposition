"""The stopping rule. Code decides when research stops; the team only decides what to research.

A research round stops at the FIRST of:
  target_reached   P(top option hits every metric target) >= charter target_probability
  voi_exhausted    no open DATA question has value-per-cost >= voi_threshold
                   (what's left either can't change the decision or needs the user: action/value queues)
  plateau          the top option's expected utility moved < plateau_epsilon over the last
                   plateau_window resolved questions this round (diminishing returns, measured)
  budget           max_questions_per_round resolved this round (a hard guard, not the plan)
  no_feasible      no option survives the hard constraints: back to the user (charter or options)
"""
from __future__ import annotations

from dataclasses import dataclass

from .charter import Charter
from .scoring import Evaluation


@dataclass
class Decision:
    stop: bool
    reason: str               # one of the codes above, or 'continue'
    detail: str
    next_questions: list[dict]  # open data questions ranked by value-per-cost (empty when stopping)


def rank_data_questions(ev: Evaluation, questions: list[dict]) -> list[dict]:
    """Open data questions with their value-per-cost. Unlinked questions score 0: link them to a
    hypothesis or they never get researched."""
    out = []
    for q in questions:
        if q["kind"] != "data" or q["status"] != "open":
            continue
        v = ev.voi.get(q.get("hypothesis_id") or "", 0.0)
        out.append({**q, "voi": v, "value_per_cost": v / q["cost"], "p_flip": ev.p_flip.get(q.get("hypothesis_id") or "", 0.0)})
    return sorted(out, key=lambda q: -q["value_per_cost"])


def decide(charter: Charter, ev: Evaluation, questions: list[dict], round_snapshots: list[dict],
           resolved_this_round: int) -> Decision:
    s = charter.stopping
    top = ev.top
    if top is None:
        return Decision(True, "no_feasible", "no option survives the hard constraints", [])
    if top.p_targets >= s.target_probability:
        return Decision(True, "target_reached",
                        f"{top.name}: P(all targets) {top.p_targets:.2f} >= {s.target_probability}", [])
    ranked = rank_data_questions(ev, questions)
    if not ranked or ranked[0]["value_per_cost"] < s.voi_threshold:
        best = f"{ranked[0]['value_per_cost']:.4f}" if ranked else "none open"
        return Decision(True, "voi_exhausted", f"best data question value-per-cost {best} < {s.voi_threshold}", [])
    if resolved_this_round >= s.max_questions_per_round:
        return Decision(True, "budget", f"{resolved_this_round} questions resolved this round", [])
    window = [x["top_utility"] for x in round_snapshots[-(s.plateau_window + 1):] if x["top_utility"] is not None]
    if len(window) > s.plateau_window and max(window) - min(window) < s.plateau_epsilon:
        return Decision(True, "plateau",
                        f"top utility moved {max(window) - min(window):.4f} over {s.plateau_window} questions", [])
    return Decision(False, "continue",
                    f"best question {ranked[0]['id']} value-per-cost {ranked[0]['value_per_cost']:.4f}", ranked)
