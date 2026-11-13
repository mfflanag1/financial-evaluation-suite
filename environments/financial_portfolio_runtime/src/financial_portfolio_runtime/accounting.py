"""Conservative successor accounting; stored native rewards are never rewritten."""

import json
import math
from pathlib import Path

from financial_research.accounting import _loads, _record_entry

from .telemetry import INFO_KEY


def classify(record, source="record"):
    result = _record_entry(record, source)
    traces = record.get("traces", [record]) if isinstance(record, dict) else []
    if len(traces) != 1 or not isinstance(traces[0], dict):
        return result
    trace = traces[0]
    runtime = trace.get("info", {}).get(INFO_KEY)
    if not isinstance(runtime, dict):
        return {**result, "status": "unknown", "reason": "runtime_observations_missing", "reward": None}
    events = runtime.get("events", [])
    terminals = [event for event in events if event.get("event") == "harness_terminal"]
    health = [event for event in events if event.get("event") == "runtime_health"]
    tools = runtime.get("tools", {})
    pending = [key for key, value in tools.items() if value.get("status") in ("gating", "allowed")]
    result["unanswered_tool_calls"] = len(pending)
    if result["status"] in ("malformed", "infrastructure_error", "validation_only"):
        return result
    if health and health[-1].get("alive") is False:
        return {**result, "status": "infrastructure_error", "reason": "runtime_died", "reward": None}
    if len(health) != 1 or health[0].get("alive") is not True:
        return {**result, "status": "unknown", "reason": "runtime_health_unresolved", "reward": None}
    if len(terminals) != 1:
        return {**result, "status": "unknown", "reason": "harness_terminal_missing_or_duplicated", "reward": None}
    terminal = terminals[0]
    if terminal.get("outcome") != "returned":
        return {**result, "status": "unknown", "reason": "harness_execution_interrupted", "reward": None}
    if terminal.get("exit_code") != 0 and trace.get("stop_condition") is None:
        return {**result, "status": "unknown", "reason": "unclassified_harness_exit", "reward": None}
    if pending:
        return {**result, "status": "unknown", "reason": "tool_execution_unresolved", "reward": None}
    return result


def report_path(path, *, expected, journal_dir=None):
    """Only bound schedule and journal evidence can support an aggregate score."""
    if type(expected) is not int or expected < 1:
        raise ValueError("Expected attempts must be a positive integer")
    path = Path(path)
    if path.is_dir():
        path = path / "traces.jsonl"
    entries, records = [], []
    for index, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = _loads(line)
            records.append(record)
            entry = classify(record, f"traces.jsonl:{index}")
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            entry = {"status": "malformed", "reason": "invalid_runtime_record", "reward": None}
        entries.append(entry)
    identities = [entry.get("record_id") for entry in entries]
    trace_ids = [identity for entry in entries for identity in entry.get("trace_ids", [])]
    duplicate = (len({identity for identity in identities if identity is not None}) != len(identities)
                 or len(set(trace_ids)) != len(trace_ids))
    from .reconciliation import reconcile
    binding = reconcile(records, expected=expected, journal_dir=journal_dir)
    for entry in entries:
        entry["binding_issues"] = sorted(set(binding["issues"] + [issue for trace_id in entry.get("trace_ids", [])
                                                        for issue in binding["per_trace"].get(trace_id, [])]))
    usable = (len(entries) == expected and not duplicate
              and not binding["issues"]
              and all(e["status"] in ("passed", "candidate_failure") and not e["binding_issues"] for e in entries))
    return {"schema_version": 1, "accounting_protocol": "portfolio-runtime-v1", "expected_attempts": expected,
            "observed_attempts": len(entries), "missing_attempts": max(0, expected - len(entries)),
            "duplicate_or_missing_identity": duplicate, "performance_usable": usable,
            "integrity_issues": binding["issues"], "request_accounting": binding["request_accounting"],
            "run_binding_sha256": binding.get("binding_sha256"),
            "mean_reward": math.fsum(e["reward"] for e in entries) / expected if usable else None,
            "counts": {status: sum(e["status"] == status for e in entries)
                       for status in ("passed", "candidate_failure", "infrastructure_error", "unknown", "malformed")},
            "entries": entries}
