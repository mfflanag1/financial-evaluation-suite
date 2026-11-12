"""Opt-in, host-only request journal for the pinned native evaluation client.

The journal never serializes requests, responses, headers, exception messages, or
endpoint configuration. A flushed start precedes each provider await. Cancellation
is recorded synchronously before it propagates; a hard crash leaves a pending start.
These observations do not establish whether a provider continued work after cancel.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4


SCHEMA_VERSION = 1
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cached_input_tokens", "reasoning_tokens", "cost")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/-]{1,256}\Z")
_ACTIVE = None


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("request journal identifiers must be bounded identifier strings")
    return value


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _finite_nonnegative(value):
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _usage(value):
    """Allowlist finite provider accounting numbers; never serialize an arbitrary object."""
    if value is None:
        return None
    result = {}
    for name in USAGE_FIELDS:
        item = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        valid = (_finite_nonnegative(item)
                 if name == "cost" else type(item) is int and item >= 0)
        result[name] = item if valid else None
    return result


def _observed_stream_usage(parser):
    """Retain already received chat usage even if the stream is later interrupted."""
    raw = getattr(parser, "usage", None)
    if not isinstance(raw, dict):
        return None
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    details = raw.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    prompt = raw.get("prompt_tokens")
    if type(cached) is not int or cached < 0:
        cached = None
    if type(prompt) is int and prompt >= (cached or 0):
        prompt -= cached or 0
    else:
        prompt = None
    return _usage({"prompt_tokens": prompt, "cached_input_tokens": cached,
                   "completion_tokens": raw.get("completion_tokens"),
                   "reasoning_tokens": reasoning, "cost": raw.get("cost")})


class RequestJournal:
    """One fresh private directory; one process owns its append-only JSONL file."""

    def __init__(self, directory):
        self.directory = Path(directory).absolute()
        # Existing directories, including symlinks, are rejected: never append a new
        # condition's records to a prior run or write through a solver-chosen path.
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(self.directory, 0o700)
        self.path = self.directory / "requests.jsonl"
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.run_id = uuid4().hex
        self._lock = threading.Lock()
        self._identities = {}
        self._indices = {}
        self._closed = False

    def register(self, *, trace_id, task_id, model):
        identity = {"trace_id": _identifier(trace_id), "task_id": _identifier(task_id),
                    "model": _identifier(model)}
        with self._lock:
            if self._closed:
                raise RuntimeError("request journal is closed")
            previous = self._identities.get(trace_id)
            if previous is not None and previous != identity:
                raise ValueError("trace identity changed within a request journal")
            self._identities[trace_id] = identity

    def _append(self, record):
        payload = (json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if self._closed:
                raise RuntimeError("request journal is closed")
            # All writers in this process share the lock. Partial writes are retried
            # only at the filesystem boundary, never at the model boundary.
            view = memoryview(payload)
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    raise OSError("request journal write made no progress")
                view = view[written:]
            os.fsync(self._fd)

    def start(self, trace_id, *, transport):
        if transport not in ("response", "stream"):
            raise ValueError("unsupported request transport")
        with self._lock:
            identity = self._identities.get(trace_id)
            if identity is None:
                raise RuntimeError("provider request has no registered operations trace")
            index = self._indices.get(trace_id, 0) + 1
            self._indices[trace_id] = index
        common = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                  "request_id": uuid4().hex, **identity, "request_index": index,
                  "transport": transport}
        started = time.monotonic_ns()
        self._append({**common, "event": "request_started", "at_utc": _utc_now(),
                      "monotonic_ns": started})
        return _Request(self, common, started)

    def close(self):
        with self._lock:
            if not self._closed:
                os.close(self._fd)
                self._closed = True


class _Request:
    def __init__(self, journal, common, started):
        self.journal, self.common, self.started = journal, common, started
        self.finished = False

    def finish(self, outcome, *, usage=None, error=None):
        if self.finished:
            return
        if outcome not in ("success", "error", "cancelled"):
            raise ValueError("unsupported request outcome")
        ended = time.monotonic_ns()
        status = getattr(error, "status_code", None)
        status = status if type(status) is int and 100 <= status <= 599 else None
        # Exception text can include provider bodies, credentials, and URLs. Keep
        # only a bounded class identifier and the numeric HTTP status.
        name = type(error).__name__ if error is not None else None
        name = name if name is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", name) else None
        self.journal._append({**self.common, "event": "request_terminal", "at_utc": _utc_now(),
                              "monotonic_ns": ended, "duration_seconds": (ended - self.started) / 1e9,
                              "outcome": outcome, "usage": _usage(usage),
                              "usage_is_final": outcome == "success",
                              "exception_type": name, "http_status": status})
        self.finished = True


class TelemetryClient:
    """Transparent native-client decorator; each upstream invocation occurs once."""

    def __init__(self, client, journal):
        self.client, self.journal = client, journal

    async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
        request = self.journal.start(session_id, transport="response")
        try:
            response = await self.client.get_response(dialect, body, sampling,
                                                     session_id=session_id, turn=turn, headers=headers)
        except asyncio.CancelledError as error:
            request.finish("cancelled", error=error)
            raise
        except BaseException as error:
            request.finish("error", error=error)
            raise
        request.finish("success", usage=response.usage)
        return response

    async def relay(self, dialect, body, session_id=None, headers=None):
        from verifiers.v1.clients.client import RelayReply
        from verifiers.v1.errors import ProviderError
        from verifiers.v1.interception.server import is_sse_done_event

        request = self.journal.start(session_id, transport="stream")
        try:
            reply = await self.client.relay(dialect, body, session_id=session_id, headers=headers)
        except asyncio.CancelledError as error:
            request.finish("cancelled", error=error)
            raise
        except BaseException as error:
            request.finish("error", error=error)
            raise
        parser = dialect.stream_parser()

        async def chunks():
            terminal = False
            try:
                async for chunk in reply.chunks:
                    if any(line.startswith(b"data:") for line in chunk.splitlines()):
                        terminal |= dialect.is_terminal_event(chunk)
                        if parser.on_done is not None and is_sse_done_event(chunk):
                            parser.on_done()
                        parser.feed(chunk)
                    yield chunk  # The native collector sees exactly the original bytes.
                if not terminal:
                    raise ProviderError("upstream stream ended before its terminal event")
                response = parser.finish()
                request.finish("success", usage=response.usage)
            except (asyncio.CancelledError, GeneratorExit) as error:
                request.finish("cancelled", usage=_observed_stream_usage(parser), error=error)
                raise
            except BaseException as error:
                request.finish("error", usage=_observed_stream_usage(parser), error=error)
                raise

        async def close():
            # The native collector closes replies in finally, including when its
            # parser rejects data before consuming the wrapped iterator to the end.
            if not request.finished:
                request.finish("cancelled", usage=_observed_stream_usage(parser))
            await reply.close()

        return RelayReply(content_type=reply.content_type, chunks=chunks(), close=close)

    async def relay_aux(self, *args, **kwargs):
        raise RuntimeError("auxiliary provider routes are unsupported by the operations harness")

    async def close(self):
        await self.client.close()


def active_journal():
    if _ACTIVE is None:
        raise RuntimeError("operations harness requires capture_requests on the trusted host")
    return _ACTIVE


@contextmanager
def capture_requests(directory):
    """Wrap a whole native run in-process; restore the resolver on every exit.

    The pinned upstream API lacks a client-wrapper registration hook. This narrowly
    scoped replacement intercepts only client construction inside this explicit
    context. Calls for unregistered traces fail before contacting a provider.
    """
    global _ACTIVE
    if version("verifiers") != "0.4.1":
        raise RuntimeError("operations telemetry requires verifiers==0.4.1")
    if _ACTIVE is not None:
        raise RuntimeError("request capture contexts cannot be nested")
    from verifiers.v1.interception import server
    original = server.resolve_client
    journal = RequestJournal(directory)

    def resolve(config):
        if getattr(config, "type", None) != "eval":
            raise ValueError("operations telemetry supports eval clients only")
        return TelemetryClient(original(config), journal)

    _ACTIVE = journal
    server.resolve_client = resolve
    try:
        yield journal
    finally:
        server.resolve_client = original
        _ACTIVE = None
        journal.close()


_COMMON = {"schema_version", "run_id", "request_id", "trace_id", "task_id", "model",
           "request_index", "transport", "event", "at_utc", "monotonic_ns"}
_TERMINAL = {"duration_seconds", "outcome", "usage", "usage_is_final", "exception_type", "http_status"}


def _valid_record(record):
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != SCHEMA_VERSION:
        return False
    event = record.get("event")
    if event not in ("request_started", "request_terminal"):
        return False
    if set(record) != (_COMMON if event == "request_started" else _COMMON | _TERMINAL):
        return False
    for field in ("run_id", "request_id", "trace_id", "task_id", "model"):
        if not isinstance(record[field], str) or not _IDENTIFIER.fullmatch(record[field]):
            return False
    if type(record["request_index"]) is not int or record["request_index"] < 1:
        return False
    if type(record["monotonic_ns"]) is not int or record["monotonic_ns"] < 0:
        return False
    if record["transport"] not in ("stream", "response"):
        return False
    try:
        stamp = datetime.fromisoformat(record["at_utc"])
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            return False
    except (TypeError, ValueError):
        return False
    if event == "request_terminal":
        duration = record["duration_seconds"]
        if not _finite_nonnegative(duration):
            return False
        if record["outcome"] not in ("success", "error", "cancelled"):
            return False
        if type(record["usage_is_final"]) is not bool or record["usage_is_final"] != (record["outcome"] == "success"):
            return False
        usage = record["usage"]
        if usage is not None and (not isinstance(usage, dict) or set(usage) != set(USAGE_FIELDS)
                                  or usage != _usage(usage)):
            return False
        status = record["http_status"]
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            return False
        error = record["exception_type"]
        if error is not None and (not isinstance(error, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", error)):
            return False
    return True


def read_journal(path):
    """Return valid records plus line-numbered corruption diagnostics, never raw bad data."""
    path = Path(path)
    path = path / "requests.jsonl" if path.is_dir() else path
    records, errors = [], []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not line.endswith(b"\n") or not _valid_record(record):
                    raise ValueError("invalid journal record")
            except (ValueError, UnicodeDecodeError):
                errors.append({"line": line_number, "kind": "invalid_record"})
            else:
                records.append(record)
    return {"records": records, "errors": errors}


def summarize_journal(path):
    """Reconcile starts/terminals without converting unknown usage to zero."""
    loaded = read_journal(path)
    starts, terminals, issues = {}, {}, list(loaded["errors"])
    indices, run_ids = set(), set()
    for record in loaded["records"]:
        run_ids.add(record["run_id"])
        key = record["request_id"]
        target = starts if record["event"] == "request_started" else terminals
        if key in target:
            issues.append({"request_id": key, "kind": "duplicate_start" if target is starts else "duplicate_terminal"})
        else:
            target[key] = record
        if target is starts:
            index = (record["trace_id"], record["request_index"])
            if index in indices:
                issues.append({"request_id": key, "kind": "duplicate_request_index"})
            indices.add(index)
        if target is terminals and key not in starts:
            issues.append({"request_id": key, "kind": "terminal_without_prior_start"})
    if len(run_ids) > 1:
        issues.append({"kind": "mixed_run_ids"})
    matched = []
    for key, terminal in terminals.items():
        start = starts.get(key)
        if start is None:
            issues.append({"request_id": key, "kind": "orphan_terminal"})
            continue
        common = _COMMON - {"event", "at_utc", "monotonic_ns"}
        elapsed = (terminal["monotonic_ns"] - start["monotonic_ns"]) / 1e9
        if any(start[field] != terminal[field] for field in common) or elapsed < 0 or not math.isclose(elapsed, terminal["duration_seconds"], abs_tol=1e-9):
            issues.append({"request_id": key, "kind": "identity_or_timing_mismatch"})
            continue
        matched.append(terminal)
    pending = [record for key, record in starts.items() if key not in terminals]
    incomplete = sum(not record["usage_is_final"] for record in matched)
    known, missing = {}, {}
    for field in USAGE_FIELDS:
        values = [record["usage"][field] for record in matched
                  if record["usage"] is not None and record["usage"][field] is not None]
        known[field] = sum(values) if values else None
        missing[field] = len(starts) - len(values)
    return {"schema_version": SCHEMA_VERSION, "request_starts": len(starts),
            "matched_terminals": len(matched),
            "outcomes": {name: sum(record["outcome"] == name for record in matched)
                         for name in ("success", "error", "cancelled")},
            "pending_requests": pending, "pending_count": len(pending), "issues": issues,
            "integrity_ok": not issues, "known_usage": known, "missing_usage_by_field": missing,
            "nonfinal_usage_requests": incomplete,
            "usage_complete": bool(starts) and not issues and not pending and not incomplete and not any(missing.values()),
            "cost_complete": bool(starts) and not issues and not pending and not incomplete and missing["cost"] == 0}
