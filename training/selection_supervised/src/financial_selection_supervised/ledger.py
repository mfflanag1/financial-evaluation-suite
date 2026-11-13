"""Private, transactional attempt accounting across the launcher and native processes.

Only identifiers, hashes and bounded enum outcomes are recorded. Neither financial
answers nor raw exceptions/provider messages belong in this control ledger.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

LEDGER_ENV = "FINANCIAL_SELECTION_SUPERVISOR_LEDGER"


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    max_attempts: int = 8
    startup_seconds: float = 300
    total_seconds: float = 900
    finalize_seconds: float = 30

    def __post_init__(self):
        if type(self.max_attempts) is not int or not 2 <= self.max_attempts <= 64:
            raise ValueError("max_attempts must be an integer from 2 through 64")
        for name in ("startup_seconds", "total_seconds", "finalize_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Supervisor deadlines must be positive finite numbers")
        if not self.startup_seconds <= self.total_seconds <= 3600 or self.finalize_seconds > 60:
            raise ValueError("Unsupported supervisor deadlines")


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Existing absolute private ledger required")
        if self.path.stat().st_mode & 0o077:
            raise ValueError("Ledger permissions must exclude other users")

    @classmethod
    def create(cls, root, limits, config_sha256, sources_sha256):
        root = Path(root)
        if not root.is_absolute() or root.exists():
            raise ValueError("Supervisor root must be a fresh absolute directory")
        root.mkdir(mode=0o700, parents=True)
        path = root / "run.sqlite3"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with sqlite3.connect(path) as conn:
            conn.executescript('''
              CREATE TABLE run(id INTEGER PRIMARY KEY CHECK(id=1), scope TEXT NOT NULL,
                started REAL NOT NULL, max_attempts INTEGER NOT NULL, startup REAL NOT NULL,
                total REAL NOT NULL, finalize REAL NOT NULL, closed INTEGER NOT NULL DEFAULT 0,
                reason TEXT, config_sha256 TEXT NOT NULL, sources_sha256 TEXT NOT NULL, native_finished REAL);
              CREATE TABLE attempts(id TEXT PRIMARY KEY, reserved REAL NOT NULL,
                request_id TEXT UNIQUE, status TEXT NOT NULL, ended REAL);
              CREATE TABLE worker(pid INTEGER PRIMARY KEY, created REAL NOT NULL,
                booted REAL NOT NULL, ready REAL);
              CREATE TABLE components(name TEXT PRIMARY KEY, pid INTEGER NOT NULL, created REAL NOT NULL);
            ''')
            conn.execute("INSERT INTO run VALUES(1,?,?,?,?,?,?,0,NULL,?,?,NULL)", (
                uuid4().hex, time.monotonic(), limits.max_attempts, limits.startup_seconds,
                limits.total_seconds, limits.finalize_seconds, config_sha256, sources_sha256))
        return cls(path)

    @classmethod
    def from_env(cls):
        return cls(os.environ[LEDGER_ENV])

    @contextmanager
    def transaction(self, timeout=10):
        conn = sqlite3.connect(self.path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def reserve(self):
        with self.transaction() as conn:
            run = conn.execute("SELECT * FROM run").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if run["closed"] or count >= run["max_attempts"] or time.monotonic() - run["started"] >= run["total"]:
                raise BudgetExhausted("Whole-run admission budget exhausted")
            attempt = uuid4().hex
            conn.execute("INSERT INTO attempts VALUES(?,?,NULL,'reserved',NULL)", (attempt, time.monotonic()))
            return attempt

    def bind_request(self, attempt, request_id):
        if not isinstance(request_id, str) or len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
            raise ValueError("Unexpected native request identity")
        with self.transaction() as conn:
            row = conn.execute("SELECT request_id,status FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None or row["status"] != "reserved" or row["request_id"] is not None:
                raise RuntimeError("Request must bind once to a reserved attempt")
            conn.execute("UPDATE attempts SET request_id=? WHERE id=?", (request_id, attempt))

    def finish(self, attempt, status):
        if status not in {"returned", "failed", "cancelled"}:
            raise ValueError("Unknown attempt outcome")
        with self.transaction() as conn:
            cursor = conn.execute("UPDATE attempts SET status=?,ended=? WHERE id=? AND status='reserved'",
                                  (status, time.monotonic(), attempt))
            if cursor.rowcount != 1:
                raise RuntimeError("Attempt finalization is missing or duplicated")

    def worker_boot(self, pid, created):
        with self.transaction() as conn:
            if conn.execute("SELECT COUNT(*) FROM worker").fetchone()[0]:
                raise RuntimeError("Worker replacement/restart is forbidden")
            conn.execute("INSERT INTO worker VALUES(?,?,?,NULL)", (pid, created, time.monotonic()))

    def worker_ready(self, pid):
        with self.transaction() as conn:
            cursor = conn.execute("UPDATE worker SET ready=? WHERE pid=? AND ready IS NULL", (time.monotonic(), pid))
            if cursor.rowcount != 1:
                raise RuntimeError("Worker readiness needs a unique recorded boot")

    def component(self, name, pid, created):
        if name not in {"launcher", "orchestrator", "pool", "trainer", "inference"}:
            raise ValueError("Unexpected launcher component")
        with self.transaction() as conn:
            conn.execute("INSERT INTO components VALUES(?,?,?)", (name, pid, created))

    def native_finished(self):
        with self.transaction(timeout=.1) as conn:
            cursor = conn.execute("UPDATE run SET closed=1,native_finished=? WHERE native_finished IS NULL",
                                  (time.monotonic(),))
            if cursor.rowcount != 1:
                raise RuntimeError("Native completion was already recorded")

    def close(self, reason):
        if reason not in {"completed", "native_failed", "attempt_budget", "time_budget", "startup_timeout", "worker_died", "interrupted", "supervisor_failed", "cleanup_failed"}:
            raise ValueError("Unknown supervisor stop reason")
        with self.transaction(timeout=0) as conn:
            conn.execute("UPDATE run SET closed=1, reason=COALESCE(reason,?)", (reason,))

    def snapshot(self):
        with self.transaction(timeout=.1) as conn:
            return {"run": dict(conn.execute("SELECT * FROM run").fetchone()),
                    "attempts": [dict(r) for r in conn.execute("SELECT * FROM attempts ORDER BY reserved")],
                    "workers": [dict(r) for r in conn.execute("SELECT * FROM worker")],
                    "components": [dict(r) for r in conn.execute("SELECT * FROM components")]}

    def report(self):
        data = self.snapshot()
        counts = {status: sum(a["status"] == status for a in data["attempts"])
                  for status in ("reserved", "returned", "failed", "cancelled")}
        return {"schema": "selection-supervisor-v1", "scope": data["run"]["scope"],
                "reason": data["run"]["reason"], "attempts": len(data["attempts"]),
                "outcomes": counts, "accounting_complete": counts["reserved"] == 0,
                "worker_boots": len(data["workers"]),
                "worker_ready": sum(w["ready"] is not None for w in data["workers"]),
                "config_sha256": data["run"]["config_sha256"],
                "sources_sha256": data["run"]["sources_sha256"],
                "native_components_finished": data["run"]["native_finished"] is not None,
                "optimizer_validated": False}
