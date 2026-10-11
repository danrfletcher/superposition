# Superposition

**A strategiser for big, open-ended decisions in any area of life.** Career moves, business bets,
relocation, health: questions where you don't yet know which questions to ask.

Most research tools give you a report, and a report on a big question tends to leave you with more
open questions than you started with. Superposition takes a different approach. It keeps every
option open, each with a probability, and runs research rounds until the evidence can't change the
decision any more. What it hands back is **tasks**: experiments only you can run, each with a
decision rule agreed in advance. Your results feed the next round, and the plan converges.

## How it works

1. **Charter.** An interview turns a vague goal into a fixed shape that's the same for every
   question: objectives, measurable proxy metrics with targets (the system proposes them, the user
   approves them), hard constraints, fixed inputs, risk attitude and stopping settings. Together,
   these are the optimisation function.
2. **Research rounds.** A planner and parallel researchers work a shared pool of questions,
   recording sourced evidence and probability-weighted hypotheses. A separate verifier checks each
   finding before it counts. Every question is sorted into one of three queues:
   - **data:** research can settle it;
   - **action:** only the user can, by running a task;
   - **value:** a trade-off only the user can decide.
3. **Tasks and the menu.** Action questions become tasks with pre-set decision rules ("if the reply
   rate is under 2% after 150 messages, drop this option"). The user's results settle hypotheses and
   start the next round. A ranked menu of options is available at any point.

## Code decides when research stops, not an agent

Research ends at the first of these (`stopping.py`):

| Reason | Meaning |
|---|---|
| `target_reached` | The top option's probability of hitting every target passes the charter's threshold |
| `voi_exhausted` | No open data question's **expected value of information** is worth its cost |
| `plateau` | The top option's expected utility has stopped moving over the last few questions |
| `budget` | A hard cap per round |
| `no_feasible` | No option survives the hard constraints, so it goes back to the user |

Options are scored by Monte Carlo simulation over low / likely / high estimates (`scoring.py`).
Estimates can be conditional on a hypothesis. Each hypothesis is drawn once per sample and shared
across options, so options that depend on the same unknown move together. Expected value of perfect
information is computed per hypothesis: a question whose answer can't change the top option scores
exactly zero and is never researched.

## Design guarantees

- **Facts and hypotheses stay separate.** A hypothesis becomes a fact only through verified evidence
  or the user's own result.
- **Calibration is tracked from the first task.** Each task records the probability given
  beforehand, so the system's forecasts earn a Brier score over time.
- **Backtests can't use hindsight.** A charter can set an evidence cutoff, and evidence dated after
  it is refused.
- **Model-agnostic workers.** LLM work runs through a pluggable worker launcher on named model
  tiers (`cheap`, `synthesis`, `strong`), not hard-coded models.
- **Usable by other agents** through an MCP server (`status`, `get_menu`, `submit_result`,
  `start`, `answer`).

## Status

| Part | State |
|---|---|
| Charter schema, decision-model store, scoring + VOI, stopping rule | Built and tested |
| MCP server (`status`, `get_menu`, `submit_result`) | Working |
| LangGraph core (interviewer, research round, verifier, menu writer) | In progress |

## Development

```bash
uv sync
uv run pytest
```

Data lives in `~/.local/share/superposition/` (override with `SUPERPOSITION_DATA`).
