"""The charter: the global shape every Superposition question fills in.

The shape is the same for every life area. The interview's job is to work out what each field
means for THIS question and to get the facts behind it from the user. Metrics are proposed by the
system and approved by the user; the charter cannot drive research until every metric is approved.

Together, metrics (with weights and targets) and hard constraints ARE the optimisation function.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Objective(BaseModel):
    """What the user actually wants, in their own words. Not measurable on its own."""
    id: str
    statement: str


class Metric(BaseModel):
    """A measurable stand-in (proxy) for one objective.

    Normalised score: 0 at `floor` (worst acceptable), 1 at `target`. Linear, may exceed [0, 1].
    """
    id: str
    objective_id: str
    name: str
    unit: str
    direction: Literal["maximize", "minimize"]
    floor: float = Field(description="worst acceptable value: scores 0")
    target: float = Field(description="the value that counts as success: scores 1")
    weight: float = Field(gt=0, description="relative importance; normalised across metrics")
    rationale: str = ""
    approved: bool = False          # proposed by the system, approved by the user

    @model_validator(mode="after")
    def _ordered(self):
        if self.floor == self.target:
            raise ValueError(f"metric {self.id}: floor and target must differ")
        if self.direction == "maximize" and self.target < self.floor:
            raise ValueError(f"metric {self.id}: maximize needs target > floor")
        if self.direction == "minimize" and self.target > self.floor:
            raise ValueError(f"metric {self.id}: minimize needs target < floor")
        return self

    def normalise(self, value: float) -> float:
        return (value - self.floor) / (self.target - self.floor)

    def hits(self, value: float) -> bool:
        return value >= self.target if self.direction == "maximize" else value <= self.target


class Constraint(BaseModel):
    """A hard limit. An option that breaks it (with probability above `tolerance`) is excluded."""
    id: str
    statement: str
    metric_id: Optional[str] = Field(None, description="metric it bounds; None = checked by judgement")
    op: Optional[Literal["<=", ">="]] = None
    bound: Optional[float] = None
    tolerance: float = Field(0.1, ge=0, le=1, description="max acceptable probability of breaking it")

    @model_validator(mode="after")
    def _complete(self):
        if self.metric_id and (self.op is None or self.bound is None):
            raise ValueError(f"constraint {self.id}: a metric constraint needs op and bound")
        return self

    def holds(self, value: float) -> bool:
        assert self.bound is not None, f"constraint {self.id} has no bound"
        return value <= self.bound if self.op == "<=" else value >= self.bound


class FixedInput(BaseModel):
    """A fact the system accepts without researching (skills, savings, current status)."""
    id: str
    name: str
    value: str
    as_of: str = ""


class Stopping(BaseModel):
    """When research stops. Enforced by code (stopping.py), never by an agent's discretion."""
    target_probability: float = Field(0.8, gt=0, le=1,
        description="stop when P(top option hits every metric target) reaches this")
    voi_threshold: float = Field(0.01, ge=0,
        description="stop when the best open data question's value-per-cost falls below this")
    plateau_window: int = Field(5, ge=2, description="resolved questions to look back over")
    plateau_epsilon: float = Field(0.01, ge=0,
        description="stop when the top option's expected utility moved less than this over the window")
    max_questions_per_round: int = Field(40, ge=1, description="hard budget per round")
    samples: int = Field(4000, ge=200, description="Monte Carlo samples")


class Charter(BaseModel):
    question: str
    life_area: str
    version: int = 1
    objectives: list[Objective]
    metrics: list[Metric]
    constraints: list[Constraint] = []
    fixed_inputs: list[FixedInput] = []
    priority_notes: str = Field("", description="what wins when objectives conflict, in the user's words")
    risk_attitude: Literal["averse", "neutral", "seeking"] = "neutral"
    horizon: str = ""
    evidence_cutoff: Optional[str] = Field(None,
        description="ISO date; research may only use evidence dated before it (backtests)")
    stopping: Stopping = Field(default_factory=Stopping)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @field_validator("objectives")
    @classmethod
    def _some_objectives(cls, v):
        if not v:
            raise ValueError("a charter needs at least one objective")
        return v

    @model_validator(mode="after")
    def _references(self):
        obj = {o.id for o in self.objectives}
        met = {m.id for m in self.metrics}
        for m in self.metrics:
            if m.objective_id not in obj:
                raise ValueError(f"metric {m.id} points at unknown objective {m.objective_id}")
        for o in obj:
            if not any(m.objective_id == o for m in self.metrics):
                raise ValueError(f"objective {o} has no metric: it can't be optimised")
        for c in self.constraints:
            if c.metric_id and c.metric_id not in met:
                raise ValueError(f"constraint {c.id} points at unknown metric {c.metric_id}")
        ids = [x.id for x in [*self.objectives, *self.metrics, *self.constraints, *self.fixed_inputs]]
        if len(ids) != len(set(ids)):
            raise ValueError("ids must be unique across objectives, metrics, constraints and inputs")
        return self

    @property
    def approved(self) -> bool:
        """Research may only run on a fully approved charter."""
        return bool(self.metrics) and all(m.approved for m in self.metrics)

    def weights(self) -> dict[str, float]:
        total = sum(m.weight for m in self.metrics)
        return {m.id: m.weight / total for m in self.metrics}

    def metric(self, mid: str) -> Metric:
        return next(m for m in self.metrics if m.id == mid)

    def next_version(self, **changes) -> "Charter":
        """Changing the charter is deliberate: a new version, metrics re-approved if they changed."""
        data = self.model_dump()
        data.update(changes)
        data["version"] = self.version + 1
        data["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return Charter.model_validate(data)
