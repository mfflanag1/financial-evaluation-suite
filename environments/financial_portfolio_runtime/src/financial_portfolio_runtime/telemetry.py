"""Trusted host observations, without prompt text, credentials or exception bodies."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from financial_portfolio_operations.telemetry import capture_requests


_ACTIVE = None
INFO_KEY = "portfolio_runtime"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def prompt_binding(messages):
    return {"initial_messages_sha256": digest(messages),
            "system_contents_sha256": digest([m["content"] for m in messages if m["role"] == "system"]),
            "first_user_content_sha256": digest(next(m["content"] for m in messages if m["role"] == "user"))}


class RuntimeJournal:
    def __init__(self, path):
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._lock = threading.Lock()
        self.bindings = {}
        self.run_id = uuid4().hex

    def emit(self, trace_id, event, **details):
        record = {"schema_version": 1, "run_id": self.run_id, "trace_id": trace_id, "event": event,
                  "monotonic_ns": time.monotonic_ns(), **details}
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        with self._lock:
            view = memoryview(payload)
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    raise OSError("Runtime journal write made no progress")
                view = view[written:]
            os.fsync(self._fd)
        return record

    def bind(self, trace, messages):
        binding = prompt_binding(messages)
        if trace.id in self.bindings:
            raise ValueError("Runtime harness does not resume or relaunch an attempt")
        self.bindings[trace.id] = {**binding, "requests": 0}
        trace.info.setdefault(INFO_KEY, {}).update(prompt_binding=binding, capture_id=self.run_id)
        self.emit(trace.id, "prompt_bound", **binding)

    def verify(self, trace_id, body):
        from verifiers.v1.errors import HarnessError

        binding = self.bindings.get(trace_id)
        if binding is None:
            raise HarnessError("No trusted runtime prompt binding")
        messages = body.get("messages", [])
        try:
            systems = [m["content"] for m in messages if m["role"] == "system"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            # Tool/assistant metadata is relevant on later turns; initial messages
            # must consist exactly of the bound system and user messages.
            actual = digest(messages)
            valid = (digest(systems) == binding["system_contents_sha256"]
                     and digest(user) == binding["first_user_content_sha256"]
                     and (binding["requests"] > 0 or actual == binding["initial_messages_sha256"]))
        except (KeyError, TypeError, StopIteration, ValueError):
            actual, valid = None, False
        self.emit(trace_id, "outbound_prompt_checked", request_index=binding["requests"] + 1,
                  messages_sha256=actual, valid=valid)
        if not valid:
            raise HarnessError("Outbound prompt differs from trusted runtime binding")
        binding["requests"] += 1

    def close(self):
        os.close(self._fd)


class BoundaryClient:
    def __init__(self, client, journal):
        self.client, self.journal = client, journal

    async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
        self.journal.verify(session_id, body)
        return await self.client.get_response(dialect, body, sampling, session_id=session_id,
                                              turn=turn, headers=headers)

    async def relay(self, dialect, body, session_id=None, headers=None):
        self.journal.verify(session_id, body)
        return await self.client.relay(dialect, body, session_id=session_id, headers=headers)

    async def relay_aux(self, *args, **kwargs):
        raise RuntimeError("Auxiliary model calls are unsupported by this runtime")

    async def close(self):
        return await self.client.close()


def active_journal():
    if _ACTIVE is None:
        raise RuntimeError("Runtime harness requires capture_runtime on the trusted host")
    return _ACTIVE


def record(trace, event, **details):
    row = active_journal().emit(trace.id, event, **details)
    trace.info.setdefault(INFO_KEY, {}).setdefault("events", []).append(row)
    return row


@contextmanager
def capture_runtime(directory):
    """One scoped native run, two append-only journals, no installed-source edits."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("Runtime capture contexts cannot be nested")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    journal = RuntimeJournal(directory / "lifecycle.jsonl")
    try:
        with capture_requests(directory / "provider"):
            from verifiers.v1.interception import server
            previous = server.resolve_client
            server.resolve_client = lambda config: BoundaryClient(previous(config), journal)
            _ACTIVE = journal
            try:
                yield journal
            finally:
                _ACTIVE = None
                server.resolve_client = previous
    finally:
        journal.close()
