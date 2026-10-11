"""Superposition MCP server: how agents (AI agents) and people use the system.

  start(title, life_area, question)   open an inquiry; the interview begins        [needs graph core]
  answer(inquiry_id, text)             answer the interviewer / a value question  [needs graph core]
  submit_result(inquiry_id, task_id, result, outcome)   report a task the user ran; resumes research
  status(inquiry_id)                   where it is, top option, open tasks, why it stopped
  get_menu(inquiry_id)                 the current ranking as a short options menu (any time)

The decision model lives in Superposition's own database; these tools never take an LLM's word
for a number: scores come from scoring.py, stopping from stopping.py.
"""
from __future__ import annotations

import functools
import json

from mcp.server.mcpserver import MCPServer

from .scoring import evaluate
from .store import Store

mcp = MCPServer("superposition")
NOT_YET = "not built yet: the LangGraph core (interviewer, research round) is in progress"


def tool(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except (ValueError, KeyError) as e:
            return f"refused: {e}"
    return mcp.tool()(w)


def _eval(s: Store, iid: str):
    return evaluate(s.charter(iid), s.options(iid), s.hypotheses(iid), s.estimates(iid))


@tool
def start(title: str, life_area: str, question: str) -> str:
    """Open a new inquiry for a big question in any life area. Superposition interviews the user to
    build the charter (objectives, measurable targets, constraints), then researches."""
    return NOT_YET


@tool
def answer(inquiry_id: str, text: str) -> str:
    """Pass the user's answer to the interviewer or to an open value question."""
    return NOT_YET


@tool
def submit_result(inquiry_id: str, task_id: str, result: str, outcome: str = "inconclusive") -> str:
    """Report the result of a task the user ran. outcome: 'true' or 'false' if it settled the task's
    hypothesis, else 'inconclusive'."""
    if outcome not in ("true", "false", "inconclusive"):
        raise ValueError("outcome must be true, false or inconclusive")
    s = Store()
    s.complete_task(inquiry_id, task_id, result, None if outcome == "inconclusive" else outcome == "true")
    return f"recorded {task_id}; research resumes when the graph core is built ({NOT_YET})"


@tool
def status(inquiry_id: str) -> str:
    """Where an inquiry is: status, round, top option, open tasks and open questions."""
    s = Store()
    i = s.inquiry(inquiry_id)
    out = {"title": i["title"], "status": i["status"], "round": i["round"],
           "open_tasks": [{k: t[k] for k in ("id", "action", "deadline", "decision_rule")} for t in s.tasks(inquiry_id)],
           "open_value_questions": [q["text"] for q in s.questions(inquiry_id, "value")],
           "last_stop": (s.snapshots(inquiry_id) or [{}])[-1].get("cause")}
    try:
        top = _eval(s, inquiry_id).top
        out["top_option"] = top and {"name": top.name, "p_targets": round(top.p_targets, 2)}
    except KeyError:
        out["top_option"] = None
    return json.dumps(out, indent=1)


@tool
def get_menu(inquiry_id: str, size: int = 3) -> str:
    """The current best options, each with its expected score, chance of hitting every target and
    the open tasks that could change it. Works at any point, not only at the end."""
    s = Store()
    ev = _eval(s, inquiry_id)
    ch = s.charter(inquiry_id)
    units = {m.id: f"{m.name} ({m.unit})" for m in ch.metrics}
    menu = [{"option": o.name, "expected_score": round(o.mean, 3), "p_all_targets": round(o.p_targets, 2),
             "feasible": o.feasible, "expected": {units[k]: round(v, 1) for k, v in o.metric_means.items()}}
            for o in ev.ranking[:size]]
    return json.dumps({"charter_version": ch.version, "menu": menu,
                       "not_ranked_missing_estimates": ev.incomplete,
                       "tasks_that_could_change_it": [t["action"] for t in s.tasks(inquiry_id)]}, indent=1)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
