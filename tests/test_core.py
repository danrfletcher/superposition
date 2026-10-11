"""Core guarantees: the charter validates, facts need evidence, VOI targets decision-relevant
questions only, and the stopping rule fires on each of its conditions."""
import pytest

from superposition.charter import Charter
from superposition.scoring import evaluate
from superposition.stopping import decide
from superposition.store import Store


def charter(**over):
    base = dict(
        question="Which job path?", life_area="career",
        objectives=[{"id": "o1", "statement": "earn well"}, {"id": "o2", "statement": "get to the US"}],
        metrics=[
            {"id": "pay", "objective_id": "o1", "name": "salary", "unit": "kGBP", "direction": "maximize",
             "floor": 40, "target": 100, "weight": 1, "approved": True},
            {"id": "us", "objective_id": "o2", "name": "months to US", "unit": "months", "direction": "minimize",
             "floor": 60, "target": 18, "weight": 1, "approved": True},
        ],
        constraints=[{"id": "c1", "statement": "income >= 45k", "metric_id": "pay", "op": ">=", "bound": 45,
                      "tolerance": 0.2}],
        stopping={"samples": 3000},
    )
    base.update(over)
    return Charter.model_validate(base)


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.create_inquiry("i1", "test", "career")
    s.save_charter("i1", charter())
    return s


# ------------------------------------------------------------------ charter
def test_objective_without_metric_rejected():
    with pytest.raises(ValueError, match="has no metric"):
        charter(objectives=[{"id": "o1", "statement": "a"}, {"id": "o2", "statement": "b"},
                            {"id": "o3", "statement": "c"}])


def test_backwards_target_rejected():
    with pytest.raises(ValueError, match="maximize needs"):
        charter(metrics=[{"id": "pay", "objective_id": "o1", "name": "s", "unit": "k", "direction": "maximize",
                          "floor": 100, "target": 40, "weight": 1},
                         {"id": "us", "objective_id": "o2", "name": "m", "unit": "m", "direction": "minimize",
                          "floor": 60, "target": 18, "weight": 1}])


def test_unapproved_metric_blocks_research():
    c = charter()
    assert c.approved
    c2 = c.next_version(metrics=[{**m.model_dump(), "approved": False} for m in c.metrics])
    assert c2.version == 2 and not c2.approved


# ------------------------------------------------------------------ store
def test_settling_needs_verified_evidence(store):
    store.add_hypothesis("i1", "h1", "visa sponsorship is common for FDEs", 0.5)
    unverified = store.add_evidence("i1", "a blog says so", "https://x", hypothesis_id="h1")
    with pytest.raises(ValueError, match="verified evidence"):
        store.settle("i1", "h1", True, "blog", evidence_id=unverified)
    ok = store.add_evidence("i1", "USCIS data", "https://uscis", hypothesis_id="h1", verified=True)
    store.settle("i1", "h1", True, "USCIS", evidence_id=ok)
    assert store.hypotheses("i1")[0]["status"] == "true"


def test_evidence_cutoff_enforced(tmp_path):
    s = Store(str(tmp_path / "b.db"))
    s.create_inquiry("b", "backtest", "career")
    s.save_charter("b", charter(evidence_cutoff="2026-03-01"))
    s.add_evidence("b", "old", "src", dated="2026-02-01")
    with pytest.raises(ValueError, match="after the cutoff"):
        s.add_evidence("b", "hindsight", "src", dated="2026-06-01")


def test_task_must_come_from_action_question_and_records_prediction(store):
    store.add_hypothesis("i1", "h1", "cold outreach converts", 0.3)
    store.add_question("i1", "q1", "is it decidable online?", "data", "h1")
    with pytest.raises(ValueError, match="action question"):
        store.add_task("i1", "t1", "q1", "do it", "x", "2026-11-01", "rule")
    store.add_question("i1", "q2", "run a sprint", "action", "h1")
    store.add_task("i1", "t1", "q2", "send 150 messages", "reply rate", "2026-11-01", "<2% -> drop")
    store.complete_task("i1", "t1", "1.1% reply rate", outcome=False)
    assert store.hypotheses("i1")[0]["status"] == "false"
    cal = store.calibration("i1")
    assert cal["n"] == 1 and abs(cal["brier"] - 0.09) < 1e-9


# ------------------------------------------------------------------ scoring + VOI
def two_options(store, p=0.5):
    store.add_option("i1", "fde", "FDE job")
    store.add_option("i1", "con", "Consultancy")
    store.add_hypothesis("i1", "h_out", "cold outreach lands clients within runway", p)
    store.add_hypothesis("i1", "h_irrelevant", "AWS cert matters", 0.5)
    # FDE: solid, unconditional
    store.add_estimate("i1", "fde", "pay", 80, 90, 100)
    store.add_estimate("i1", "fde", "us", 20, 30, 40)
    # Consultancy: great if outreach works, poor if not
    store.add_estimate("i1", "con", "pay", 120, 150, 180, hypothesis_id="h_out", when_true=True)
    store.add_estimate("i1", "con", "pay", 46, 50, 60, hypothesis_id="h_out", when_true=False)
    store.add_estimate("i1", "con", "us", 30, 40, 50)


def test_voi_targets_only_decision_relevant_hypotheses(store):
    two_options(store)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    assert ev.voi["h_out"] > 0.01           # could flip FDE vs consultancy
    assert ev.voi["h_irrelevant"] == 0.0    # no estimate depends on it: never worth researching
    assert 0 < ev.p_flip["h_out"] < 1


def test_settled_hypothesis_has_no_value(store):
    two_options(store)
    e = store.add_evidence("i1", "data", "src", hypothesis_id="h_out", verified=True)
    store.settle("i1", "h_out", False, "data", evidence_id=e)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    assert "h_out" not in ev.voi
    assert ev.top.option_id == "fde"


def test_constraint_excludes_option(store):
    store.add_option("i1", "broke", "Unpaid sabbatical")
    store.add_estimate("i1", "broke", "pay", 0, 5, 10)
    store.add_estimate("i1", "broke", "us", 1, 2, 3)
    two_options(store)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    broke = next(o for o in ev.ranking if o.option_id == "broke")
    assert not broke.feasible and ev.ranking[-1].option_id == "broke"


def test_missing_estimate_is_reported_not_ranked(store):
    store.add_option("i1", "half", "Half-estimated")
    store.add_estimate("i1", "half", "pay", 50, 60, 70)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    assert ev.incomplete == {"half": ["us"]}


# ------------------------------------------------------------------ stopping rule
def ev_and_qs(store, p=0.5):
    two_options(store, p)
    store.add_question("i1", "q_out", "what reply rates do cold emails get?", "data", "h_out", cost=1)
    store.add_question("i1", "q_aws", "does the AWS cert matter?", "data", "h_irrelevant", cost=1)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    return ev, store.questions("i1")


def test_continue_picks_the_decision_relevant_question(store):
    ev, qs = ev_and_qs(store)
    d = decide(store.charter("i1"), ev, qs, [], 0)
    assert not d.stop and d.next_questions[0]["id"] == "q_out"
    assert d.next_questions[-1]["id"] == "q_aws" and d.next_questions[-1]["voi"] == 0


def test_stops_when_voi_exhausted(store):
    ev, qs = ev_and_qs(store)
    d = decide(store.charter("i1"), ev, [q for q in qs if q["id"] == "q_aws"], [], 0)
    assert d.stop and d.reason == "voi_exhausted"


def test_stops_on_target(store):
    ev_and_qs(store)
    easy = charter(stopping={"samples": 3000, "target_probability": 0.8})
    easy = easy.next_version(metrics=[{**m.model_dump(), "target": 60 if m.id == "pay" else 45} for m in easy.metrics])
    ev = evaluate(easy, store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    assert ev.top.p_targets >= 0.8
    assert decide(easy, ev, store.questions("i1"), [], 0).reason == "target_reached"


def test_stops_on_plateau(store):
    ev, qs = ev_and_qs(store)
    flat = [{"top_utility": 0.5 + i * 0.001} for i in range(6)]
    assert decide(store.charter("i1"), ev, qs, flat, 6).reason == "plateau"
    moving = [{"top_utility": 0.5 + i * 0.05} for i in range(6)]
    assert not decide(store.charter("i1"), ev, qs, moving, 6).stop


def test_stops_on_budget(store):
    ev, qs = ev_and_qs(store)
    assert decide(store.charter("i1"), ev, qs, [], 40).reason == "budget"


def test_no_feasible_option(store):
    store.add_option("i1", "broke", "Unpaid sabbatical")
    store.add_estimate("i1", "broke", "pay", 0, 5, 10)
    store.add_estimate("i1", "broke", "us", 1, 2, 3)
    ev = evaluate(store.charter("i1"), store.options("i1"), store.hypotheses("i1"), store.estimates("i1"))
    assert decide(store.charter("i1"), ev, [], [], 0).reason == "no_feasible"
