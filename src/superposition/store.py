"""The decision model: one SQLite file per installation, outliving any orchestration framework.

An *inquiry* is one Superposition question (e.g. "optimal career path"). It holds:

  charters     versioned charter JSON (the optimisation function)
  options      the candidate plans being ranked
  hypotheses   uncertain claims with a probability; FACTS are hypotheses settled by evidence
  estimates    option x metric as low/likely/high, optionally conditional on a hypothesis
  questions    three queues: data (research settles it), action (the user runs a task), value (the user decides)
  tasks        action questions turned into experiments with a pre-set decision rule
  evidence     sourced findings, each verified or not, linked to the hypothesis it bears on
  snapshots    the top option's expected utility after each change (plateau detection)
  predictions  probability given before a task vs what happened (calibration = the eval)

Facts and hypotheses stay apart: status 'open' carries a probability; 'true'/'false' are settled
and only a verified piece of evidence (or the user) may settle one.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .charter import Charter

DATA = os.path.expanduser(os.environ.get("SUPERPOSITION_DATA", "~/.local/share/superposition"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS inquiries (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, life_area TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'interview',      -- interview|research|awaiting_user|done|frozen
  round INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS charters (
  inquiry_id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (inquiry_id, version));
CREATE TABLE IF NOT EXISTS options (
  inquiry_id TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL, description TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'live',            -- live|dropped
  PRIMARY KEY (inquiry_id, id));
CREATE TABLE IF NOT EXISTS hypotheses (
  inquiry_id TEXT NOT NULL, id TEXT NOT NULL, statement TEXT NOT NULL,
  probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
  status TEXT NOT NULL DEFAULT 'open',            -- open|true|false
  basis TEXT DEFAULT '', updated_at TEXT NOT NULL,
  PRIMARY KEY (inquiry_id, id));
CREATE TABLE IF NOT EXISTS estimates (
  inquiry_id TEXT NOT NULL, option_id TEXT NOT NULL, metric_id TEXT NOT NULL,
  hypothesis_id TEXT NOT NULL DEFAULT '',         -- '' = unconditional
  when_true INTEGER NOT NULL DEFAULT 1,           -- which branch this estimate is for
  low REAL NOT NULL, likely REAL NOT NULL, high REAL NOT NULL, source TEXT DEFAULT '',
  CHECK (low <= likely AND likely <= high),
  PRIMARY KEY (inquiry_id, option_id, metric_id, hypothesis_id, when_true));
CREATE TABLE IF NOT EXISTS questions (
  inquiry_id TEXT NOT NULL, id TEXT NOT NULL, text TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('data','action','value')),
  hypothesis_id TEXT DEFAULT '', cost REAL NOT NULL DEFAULT 1 CHECK (cost > 0),
  status TEXT NOT NULL DEFAULT 'open',            -- open|in_progress|answered|unanswerable|dropped
  answer TEXT DEFAULT '', raised_by TEXT DEFAULT '', round INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (inquiry_id, id));
CREATE TABLE IF NOT EXISTS tasks (
  inquiry_id TEXT NOT NULL, id TEXT NOT NULL, question_id TEXT NOT NULL, action TEXT NOT NULL,
  collect TEXT NOT NULL, deadline TEXT NOT NULL, decision_rule TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',            -- open|done|skipped
  result TEXT DEFAULT '', PRIMARY KEY (inquiry_id, id));
CREATE TABLE IF NOT EXISTS evidence (
  inquiry_id TEXT NOT NULL, id INTEGER PRIMARY KEY AUTOINCREMENT, hypothesis_id TEXT DEFAULT '',
  claim TEXT NOT NULL, source TEXT NOT NULL, dated TEXT DEFAULT '', verified INTEGER NOT NULL DEFAULT 0,
  verifier_note TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
  inquiry_id TEXT NOT NULL, seq INTEGER NOT NULL, round INTEGER NOT NULL, top_option TEXT,
  top_utility REAL, p_targets REAL, cause TEXT, created_at TEXT NOT NULL,
  PRIMARY KEY (inquiry_id, seq));
CREATE TABLE IF NOT EXISTS predictions (
  inquiry_id TEXT NOT NULL, hypothesis_id TEXT NOT NULL, task_id TEXT NOT NULL,
  predicted REAL NOT NULL, outcome INTEGER, created_at TEXT NOT NULL,
  PRIMARY KEY (inquiry_id, task_id));
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(DATA, "superposition.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        try:
            yield self.db
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def rows(self, sql, *a):
        return [dict(r) for r in self.db.execute(sql, a)]

    # ---------------------------------------------------------------- inquiries + charters
    def create_inquiry(self, iid: str, title: str, life_area: str) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO inquiries (id,title,life_area,created_at,updated_at) VALUES (?,?,?,?,?)",
                       (iid, title, life_area, now(), now()))

    def inquiry(self, iid: str) -> dict:
        r = self.rows("SELECT * FROM inquiries WHERE id=?", iid)
        if not r:
            raise KeyError(f"no inquiry {iid!r}")
        return r[0]

    def set_status(self, iid: str, status: str, round_: int | None = None) -> None:
        with self.tx() as db:
            if round_ is None:
                db.execute("UPDATE inquiries SET status=?, updated_at=? WHERE id=?", (status, now(), iid))
            else:
                db.execute("UPDATE inquiries SET status=?, round=?, updated_at=? WHERE id=?",
                           (status, round_, now(), iid))

    def save_charter(self, iid: str, charter: Charter) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO charters VALUES (?,?,?,?)",
                       (iid, charter.version, charter.model_dump_json(), charter.created_at))

    def charter(self, iid: str) -> Charter:
        r = self.rows("SELECT body FROM charters WHERE inquiry_id=? ORDER BY version DESC LIMIT 1", iid)
        if not r:
            raise KeyError(f"inquiry {iid!r} has no charter yet")
        return Charter.model_validate_json(r[0]["body"])

    # ---------------------------------------------------------------- model contents
    def add_option(self, iid, oid, name, description=""):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO options (inquiry_id,id,name,description) VALUES (?,?,?,?)",
                       (iid, oid, name, description))

    def add_hypothesis(self, iid, hid, statement, probability, basis=""):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO hypotheses (inquiry_id,id,statement,probability,basis,updated_at)"
                       " VALUES (?,?,?,?,?,?)", (iid, hid, statement, probability, basis, now()))

    def update_probability(self, iid, hid, probability, basis):
        with self.tx() as db:
            db.execute("UPDATE hypotheses SET probability=?, basis=?, updated_at=? WHERE inquiry_id=? AND id=?"
                       " AND status='open'", (probability, basis, now(), iid, hid))

    def settle(self, iid, hid, truth: bool, basis: str, evidence_id: int | None = None, by_user=False):
        """Turn a hypothesis into a fact. Needs a VERIFIED piece of evidence, or the user."""
        if not by_user:
            ok = evidence_id is not None and self.rows(
                "SELECT 1 FROM evidence WHERE inquiry_id=? AND id=? AND hypothesis_id=? AND verified=1",
                iid, evidence_id, hid)
            if not ok:
                raise ValueError(f"{hid}: settling needs verified evidence about it (or the user)")
        with self.tx() as db:
            db.execute("UPDATE hypotheses SET status=?, probability=?, basis=?, updated_at=?"
                       " WHERE inquiry_id=? AND id=?",
                       ("true" if truth else "false", 1.0 if truth else 0.0, basis, now(), iid, hid))

    def add_estimate(self, iid, oid, mid, low, likely, high, hypothesis_id="", when_true=True, source=""):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO estimates VALUES (?,?,?,?,?,?,?,?,?)",
                       (iid, oid, mid, hypothesis_id, int(when_true), low, likely, high, source))

    def add_question(self, iid, qid, text, kind, hypothesis_id="", cost=1.0, raised_by="", round_=0):
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO questions (inquiry_id,id,text,kind,hypothesis_id,cost,raised_by,round)"
                       " VALUES (?,?,?,?,?,?,?,?)", (iid, qid, text, kind, hypothesis_id, cost, raised_by, round_))

    def close_question(self, iid, qid, status, answer=""):
        with self.tx() as db:
            db.execute("UPDATE questions SET status=?, answer=? WHERE inquiry_id=? AND id=?",
                       (status, answer, iid, qid))

    def add_evidence(self, iid, claim, source, hypothesis_id="", dated="", verified=False, verifier_note="") -> int:
        ch = self.charter(iid)
        if ch.evidence_cutoff and dated and dated >= ch.evidence_cutoff:
            raise ValueError(f"evidence dated {dated} is after the cutoff {ch.evidence_cutoff}")
        with self.tx() as db:
            cur = db.execute("INSERT INTO evidence (inquiry_id,hypothesis_id,claim,source,dated,verified,"
                             "verifier_note,created_at) VALUES (?,?,?,?,?,?,?,?)",
                             (iid, hypothesis_id, claim, source, dated, int(verified), verifier_note, now()))
            return int(cur.lastrowid or 0)

    def add_task(self, iid, tid, question_id, action, collect, deadline, decision_rule):
        q = self.rows("SELECT * FROM questions WHERE inquiry_id=? AND id=?", iid, question_id)
        if not q or q[0]["kind"] != "action":
            raise ValueError(f"task {tid}: must come from an action question")
        hid = q[0]["hypothesis_id"]
        with self.tx() as db:
            db.execute("INSERT INTO tasks (inquiry_id,id,question_id,action,collect,deadline,decision_rule)"
                       " VALUES (?,?,?,?,?,?,?)", (iid, tid, question_id, action, collect, deadline, decision_rule))
            db.execute("UPDATE questions SET status='in_progress' WHERE inquiry_id=? AND id=?", (iid, question_id))
            if hid:
                p = self.rows("SELECT probability FROM hypotheses WHERE inquiry_id=? AND id=?", iid, hid)
                if p:
                    db.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,NULL,?)",
                               (iid, hid, tid, p[0]["probability"], now()))

    def complete_task(self, iid, tid, result: str, outcome: bool | None):
        """The user's result. `outcome` = did the linked hypothesis turn out true (None = inconclusive)."""
        t = self.rows("SELECT * FROM tasks WHERE inquiry_id=? AND id=?", iid, tid)
        if not t:
            raise KeyError(f"no task {tid}")
        q = self.rows("SELECT * FROM questions WHERE inquiry_id=? AND id=?", iid, t[0]["question_id"])[0]
        with self.tx() as db:
            db.execute("UPDATE tasks SET status='done', result=? WHERE inquiry_id=? AND id=?", (result, iid, tid))
            db.execute("UPDATE questions SET status='answered', answer=? WHERE inquiry_id=? AND id=?",
                       (result, iid, q["id"]))
            if outcome is not None:
                db.execute("UPDATE predictions SET outcome=? WHERE inquiry_id=? AND task_id=?", (int(outcome), iid, tid))
        if outcome is not None and q["hypothesis_id"]:
            self.settle(iid, q["hypothesis_id"], outcome, f"task {tid}: {result}", by_user=True)

    def snapshot(self, iid, round_, top_option, top_utility, p_targets, cause):
        seq = (self.rows("SELECT COALESCE(MAX(seq),0) s FROM snapshots WHERE inquiry_id=?", iid)[0]["s"]) + 1
        with self.tx() as db:
            db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?)",
                       (iid, seq, round_, top_option, top_utility, p_targets, cause, now()))

    # ---------------------------------------------------------------- reads
    def options(self, iid, live_only=True):
        return self.rows("SELECT * FROM options WHERE inquiry_id=?" + (" AND status='live'" if live_only else ""), iid)

    def hypotheses(self, iid):
        return self.rows("SELECT * FROM hypotheses WHERE inquiry_id=?", iid)

    def estimates(self, iid):
        return self.rows("SELECT * FROM estimates WHERE inquiry_id=?", iid)

    def questions(self, iid, kind=None, status: str | None = "open"):
        sql, a = "SELECT * FROM questions WHERE inquiry_id=?", [iid]
        if kind:
            sql += " AND kind=?"; a.append(kind)
        if status:
            sql += " AND status=?"; a.append(status)
        return self.rows(sql, *a)

    def tasks(self, iid, status="open"):
        return self.rows("SELECT * FROM tasks WHERE inquiry_id=? AND status=?", iid, status)

    def snapshots(self, iid):
        return self.rows("SELECT * FROM snapshots WHERE inquiry_id=? ORDER BY seq", iid)

    def calibration(self, iid=None):
        """Brier score over settled predictions (lower is better; 0.25 = coin-flip)."""
        sql = "SELECT predicted, outcome FROM predictions WHERE outcome IS NOT NULL"
        rows = self.rows(sql + (" AND inquiry_id=?" if iid else ""), *([iid] if iid else []))
        if not rows:
            return {"n": 0, "brier": None}
        return {"n": len(rows), "brier": sum((r["predicted"] - r["outcome"]) ** 2 for r in rows) / len(rows)}

    def dump(self, iid) -> str:
        return json.dumps({
            "inquiry": self.inquiry(iid), "charter": json.loads(self.charter(iid).model_dump_json()),
            "options": self.options(iid, False), "hypotheses": self.hypotheses(iid),
            "estimates": self.estimates(iid), "questions": self.questions(iid, status=None),
            "tasks": self.rows("SELECT * FROM tasks WHERE inquiry_id=?", iid)}, indent=1)
