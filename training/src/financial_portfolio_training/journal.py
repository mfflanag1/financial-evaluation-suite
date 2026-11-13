"""Worker-owned observations; responses and training arrays pass through unchanged."""

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np

INFO_KEY = "portfolio_training"
_OWNERS = {}
_SESSIONS = {}
_PREVIOUS_RESOLVER = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def prompt_binding(messages):
    return {"initial_messages_sha256": digest(messages),
            "system_sha256": digest([m["content"] for m in messages if m["role"] == "system"]),
            "user_sha256": digest(next(m["content"] for m in messages if m["role"] == "user"))}


def _array_binding(value):
    values = np.asarray(value)
    return {"dtype": values.dtype.str, "shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def completion_summary(ids, probabilities, support, versions):
    """Bind sampled arrays exactly; validation never replaces the native arrays."""
    ids, probabilities = np.asarray(ids), np.asarray(probabilities)
    if (ids.ndim != 1 or probabilities.ndim != 1 or len(ids) != len(probabilities)
            or not np.issubdtype(ids.dtype, np.integer) or not np.all(np.isfinite(probabilities))):
        raise ValueError("Invalid completion token/logprob alignment")
    if support is not None:
        counts, support_ids, replayed = map(np.asarray, (support.counts, support.ids, support.replayed))
        if (counts.ndim != 1 or support_ids.ndim != 1 or replayed.ndim != 1
                or len(counts) != len(ids) or len(replayed) != len(ids)
                or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)
                or not np.issubdtype(support_ids.dtype, np.integer) or np.any(support_ids < 0)
                or replayed.dtype != np.bool_ or int(counts.sum()) != len(support_ids)
                or (support.logprobs is not None and (np.asarray(support.logprobs).ndim != 1
                    or len(support.logprobs) != len(support_ids) or not np.all(np.isfinite(support.logprobs))))):
            raise ValueError("Invalid sampler support alignment")
    if versions is not None:
        previous = 0
        for span in versions:
            if span.start != previous or not previous < span.end <= len(ids):
                raise ValueError("Invalid policy version spans")
            previous = span.end
        if previous != len(ids):
            raise ValueError("Incomplete policy version spans")
    return {"completion_tokens": len(ids), "completion_ids": _array_binding(ids),
            "completion_logprobs": _array_binding(probabilities),
            "sampler_support_sha256": None if support is None else digest({
                name: None if getattr(support, name) is None else _array_binding(getattr(support, name))
                for name in ("ids", "counts", "replayed", "logprobs")}),
            "policy_versions_sha256": None if versions is None else digest([
                span.model_dump(mode="json") for span in versions])}


def token_summary(response):
    """Validate shape without coercing, copying or replacing the native response."""
    tokens = response.tokens
    if tokens is None:
        raise ValueError("Training response has no native tokens")
    return {"prompt_tokens": len(tokens.prompt_ids), "prompt_ids": _array_binding(tokens.prompt_ids),
            **completion_summary(tokens.completion_ids,
            tokens.completion_logprobs, tokens.sampler_support, tokens.weight_versions)}


@dataclass
class Attempt:
    trace_id: str
    episode_id: str
    episode_attempt: int
    task_id: str
    task_sha256: str
    expected_prompt: dict
    model: str
    events: list = field(default_factory=list)
    prompt: dict | None = None
    requests: list = field(default_factory=list)
    tools: dict = field(default_factory=dict)
    terminal: bool = False


class WorkerJournal:
    def __init__(self, root):
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("Training journal root must be absolute and host-private")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("Training journal root cannot be a symlink")
        self.worker_id = uuid4().hex
        self.directory = root / self.worker_id
        self.directory.mkdir(mode=0o700)
        self._fd = os.open(self.directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._lock = threading.Lock()
        self.attempts = {}
        self.completed_ids = set()
        self.closed = False

    def emit(self, attempt, event, **details):
        if self.closed:
            raise RuntimeError("Training journal is closed")
        row = {"schema_version": 1, "worker_id": self.worker_id, "trace_id": attempt.trace_id,
               "episode_id": attempt.episode_id, "episode_attempt": attempt.episode_attempt,
               "task_id": attempt.task_id, "event": event, "monotonic_ns": time.monotonic_ns(), **details}
        payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        with self._lock:
            view = memoryview(payload)
            while view:
                count = os.write(self._fd, view)
                if count <= 0:
                    raise OSError("Journal write made no progress")
                view = view[count:]
            os.fsync(self._fd)
        attempt.events.append(row)
        return row

    def register(self, trace, *, episode_id, episode_attempt, expected_messages, model):
        if trace.id in self.attempts or trace.id in self.completed_ids or trace.id in _SESSIONS:
            raise ValueError("Trace identity has already been registered")
        attempt = Attempt(trace.id, episode_id, episode_attempt, trace.task.data.id,
                          digest(trace.task.data.model_dump(mode="json")), prompt_binding(expected_messages), model)
        self.attempts[trace.id] = attempt
        _SESSIONS[trace.id] = self
        trace.info[INFO_KEY] = {"protocol_id": "portfolio-training-v1", "worker_id": self.worker_id,
                               "task_sha256": attempt.task_sha256}
        self.emit(attempt, "attempt_started", task_sha256=attempt.task_sha256)
        return attempt

    def bind(self, trace, messages):
        attempt = self.attempts[trace.id]
        if attempt.prompt is not None:
            raise ValueError("A training trace cannot relaunch or resume")
        binding = prompt_binding(messages)
        if binding != attempt.expected_prompt:
            raise ValueError("Harness prompt differs from its registered task")
        attempt.prompt = binding
        trace.info[INFO_KEY]["prompt_binding"] = dict(attempt.prompt)
        self.emit(attempt, "prompt_bound", **attempt.prompt)

    def verify_prompt(self, attempt, body):
        messages = body.get("messages", [])
        try:
            actual = prompt_binding(messages)
            valid = (body.get("model") == attempt.model and attempt.prompt is not None
                     and actual["system_sha256"] == attempt.prompt["system_sha256"]
                     and actual["user_sha256"] == attempt.prompt["user_sha256"]
                     and (attempt.requests or actual == attempt.prompt))
        except (KeyError, TypeError, ValueError, StopIteration):
            actual, valid = {}, False
        self.emit(attempt, "prompt_checked", valid=bool(valid), messages_sha256=actual.get("initial_messages_sha256"))
        if not valid:
            raise ValueError("Outbound training prompt differs from its host binding")

    def finish(self, trace, *, admitted, reason):
        attempt = self.attempts[trace.id]
        if attempt.terminal:
            raise ValueError("Attempt admission already recorded")
        self.emit(attempt, "attempt_terminal", admitted=admitted, reason=reason)
        attempt.terminal = True
        _SESSIONS.pop(trace.id, None)
        self.completed_ids.add(trace.id)
        del self.attempts[trace.id]

    def close(self):
        if self.closed:
            return
        for attempt in self.attempts.values():
            if not attempt.terminal:
                self.emit(attempt, "attempt_terminal", admitted=False, reason="worker_stopped_before_admission")
                attempt.terminal = True
            _SESSIONS.pop(attempt.trace_id, None)
        os.close(self._fd)
        self.closed = True


def owner(trace_id):
    journal = _SESSIONS.get(trace_id)
    if journal is None or journal.closed:
        raise RuntimeError("Unknown or completed training session")
    return journal, journal.attempts[trace_id]


def record(trace, event, **details):
    journal, attempt = owner(trace.id)
    return journal.emit(attempt, event, **details)


class ObservedTrainClient:
    def __init__(self, client):
        self.client = client

    async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
        journal, attempt = owner(session_id)
        journal.verify_prompt(attempt, body)
        request = {"index": len(attempt.requests) + 1, "outcome": "pending"}
        attempt.requests.append(request)
        journal.emit(attempt, "request_started", request_index=request["index"])
        try:
            response = await self.client.get_response(dialect, body, sampling, session_id=session_id,
                                                      turn=turn, headers=headers)
            summary = token_summary(response)
        except BaseException as error:
            request["outcome"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
            journal.emit(attempt, "request_terminal", request_index=request["index"],
                         outcome=request["outcome"], exception_type=type(error).__name__)
            raise
        request.update(outcome="success", **summary)
        journal.emit(attempt, "request_terminal", request_index=request["index"], outcome="success", **summary)
        return response

    async def relay(self, *args, **kwargs):
        raise RuntimeError("Training requires renderer responses, not eval relay")

    async def relay_aux(self, *args, **kwargs):
        raise RuntimeError("Auxiliary model requests have no training session binding")

    async def close(self):
        await self.client.close()


def install(journal):
    """Install once per process, route each owned session to its exact worker."""
    global _PREVIOUS_RESOLVER
    from verifiers.v1.interception import server
    from verifiers.v1.configs.client import TrainClientConfig
    if journal.worker_id in _OWNERS:
        raise ValueError("Worker journal already installed")
    if not _OWNERS:
        previous = server.resolve_client
        def resolve(config):
            if not isinstance(config, TrainClientConfig):
                raise ValueError("Training workers require the native TrainClientConfig")
            return ObservedTrainClient(previous(config))
        _PREVIOUS_RESOLVER = previous
        server.resolve_client = resolve
    _OWNERS[journal.worker_id] = journal


def uninstall(journal):
    global _PREVIOUS_RESOLVER
    from verifiers.v1.interception import server
    _OWNERS.pop(journal.worker_id, None)
    if not _OWNERS and _PREVIOUS_RESOLVER is not None:
        server.resolve_client = _PREVIOUS_RESOLVER
        _PREVIOUS_RESOLVER = None
    journal.close()
