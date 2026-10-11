"""Score options against the charter by Monte Carlo, and value each open question.

Utility of an option in one sample = sum_m weight_m * normalised(value_m), where value_m is drawn
from the option's triangular (low, likely, high) estimate. An estimate may be CONDITIONAL on one
hypothesis (a 'when true' and a 'when false' estimate); that hypothesis is drawn once per sample and
shared across all options, so options that depend on the same uncertainty move together.

Value of information (VOI) of a hypothesis = expected value of perfect information:
    EVPI(h) = E_h[ max_o E[U_o | h] ] - max_o E[U_o]
i.e. how much better the best choice gets, on average, if we learn h before choosing. It is zero
when learning h could never change which option is top. That is the whole point: research only
what could change the decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .charter import Charter

SEED = 7  # fixed: same model in, same numbers out (common random numbers across options)


@dataclass
class OptionScore:
    option_id: str
    name: str
    mean: float               # expected utility
    sd: float
    score: float              # risk-adjusted (what ranking uses)
    p_targets: float          # P(every metric hits its target)
    p_constraints: float      # P(every metric constraint holds)
    feasible: bool
    metric_means: dict[str, float] = field(default_factory=dict)


@dataclass
class Evaluation:
    ranking: list[OptionScore]                 # feasible first, by score
    incomplete: dict[str, list[str]]           # option -> metrics with no estimate (not ranked)
    voi: dict[str, float]                      # hypothesis id -> EVPI in utility units
    p_flip: dict[str, float]                   # hypothesis id -> P(learning it changes the top option)

    @property
    def top(self) -> OptionScore | None:
        return next((o for o in self.ranking if o.feasible), None)


def _draw(rng, low, likely, high, n):
    if high <= low:
        return np.full(n, likely, dtype=float)
    return rng.triangular(low, min(max(likely, low), high), high, n)


def evaluate(charter: Charter, options: list[dict], hypotheses: list[dict], estimates: list[dict]) -> Evaluation:
    n = charter.stopping.samples
    rng = np.random.default_rng(SEED)
    hyp = {h["id"]: h for h in hypotheses}
    # one shared draw per hypothesis; settled ones are constant
    hdraw = {hid: (np.full(n, h["status"] == "true") if h["status"] != "open"
                   else rng.random(n) < h["probability"]) for hid, h in sorted(hyp.items())}
    w = charter.weights()
    est: dict[tuple[str, str], list[dict]] = {}
    for e in estimates:
        est.setdefault((e["option_id"], e["metric_id"]), []).append(e)

    utils, scores, incomplete, values = {}, [], {}, {}
    for o in options:
        oid = o["id"]
        missing, u, vals = [], np.zeros(n), {}
        for m in charter.metrics:
            es = est.get((oid, m.id), [])
            uncond = [e for e in es if not e["hypothesis_id"]]
            cond = [e for e in es if e["hypothesis_id"]]
            hids = {e["hypothesis_id"] for e in cond}
            if len(hids) > 1:
                raise ValueError(f"{oid}/{m.id}: conditional on more than one hypothesis {sorted(hids)}")
            if cond:
                hid = hids.pop()
                if hid not in hyp:
                    raise ValueError(f"{oid}/{m.id}: unknown hypothesis {hid}")
                br = {bool(e["when_true"]): e for e in cond}
                if set(br) != {True, False}:
                    raise ValueError(f"{oid}/{m.id}: conditional on {hid} needs both a true and a false estimate")
                t, f = (_draw(rng, b["low"], b["likely"], b["high"], n) for b in (br[True], br[False]))
                v = np.where(hdraw[hid], t, f)
            elif uncond:
                e = uncond[0]
                v = _draw(rng, e["low"], e["likely"], e["high"], n)
            else:
                missing.append(m.id)
                continue
            vals[m.id] = v
            u += w[m.id] * m.normalise(v)
        if missing:
            incomplete[oid] = missing
            continue
        utils[oid], values[oid] = u, vals
        hits = np.ones(n, dtype=bool)
        for m in charter.metrics:
            hits &= (vals[m.id] >= m.target) if m.direction == "maximize" else (vals[m.id] <= m.target)
        ok = np.ones(n, dtype=bool)
        for c in charter.constraints:
            if c.metric_id:
                ok &= (vals[c.metric_id] <= c.bound) if c.op == "<=" else (vals[c.metric_id] >= c.bound)
        p_ok = float(ok.mean())
        feasible = all((1 - (np.asarray((vals[c.metric_id] <= c.bound) if c.op == "<=" else
                                        (vals[c.metric_id] >= c.bound))).mean()) <= c.tolerance
                       for c in charter.constraints if c.metric_id)
        mean, sd = float(u.mean()), float(u.std())
        k = {"averse": -0.5, "neutral": 0.0, "seeking": 0.5}[charter.risk_attitude]
        scores.append(OptionScore(oid, o["name"], mean, sd, mean + k * sd, float(hits.mean()), p_ok, feasible,
                                  {mid: float(v.mean()) for mid, v in vals.items()}))

    scores.sort(key=lambda s: (not s.feasible, -s.score))
    feas = [s.option_id for s in scores if s.feasible]
    voi, flip = {}, {}
    if feas:
        k = {"averse": -0.5, "neutral": 0.0, "seeking": 0.5}[charter.risk_attitude]

        def score_of(oid, mask):
            x = utils[oid][mask]
            return float(x.mean() + k * x.std()) if x.size else -np.inf

        base = max(score_of(o, np.ones(n, dtype=bool)) for o in feas)
        top = max(feas, key=lambda o: score_of(o, np.ones(n, dtype=bool)))
        for hid, h in hyp.items():
            if h["status"] != "open":
                continue
            d = hdraw[hid]
            p = float(d.mean())
            if p in (0.0, 1.0):
                voi[hid], flip[hid] = 0.0, 0.0
                continue
            best_t = max(feas, key=lambda o: score_of(o, d))
            best_f = max(feas, key=lambda o: score_of(o, ~d))
            voi[hid] = max(0.0, p * score_of(best_t, d) + (1 - p) * score_of(best_f, ~d) - base)
            flip[hid] = p * (best_t != top) + (1 - p) * (best_f != top)
    return Evaluation(scores, incomplete, voi, flip)
