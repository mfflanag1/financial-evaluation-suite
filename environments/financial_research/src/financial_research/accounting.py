"""Lossless accounting for saved Verifiers v1 rollout records.

Native ``traces.jsonl`` stores one Episode per line, with agent Trace objects in
``traces``. A standalone Trace JSON is also accepted. Native ``vf-validate``
``results.jsonl`` records are model-free checks, never model performance.

Directories include result JSONL files outside configs, artifacts and logs. A
sibling ``attempt_manifest.json`` can declare ``{"expected_attempts": N}``.
Pass ``expected=N`` instead when the run has no predeclared manifest. An overall
reward mean requires a known denominator and every attempt to be resolved.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


OUTCOME_KEY = "financial_research_outcome"
STATUSES = ("passed", "candidate_failure", "infrastructure_error", "unknown",
            "malformed", "validation_only", "missing")
# These are the explicit infrastructure boundaries in Verifiers v0.4.1.
# TaskError, generic TimeoutError, and unknown errors remain unclassified.
INFRASTRUCTURE_ERRORS = frozenset({"ProviderError", "SandboxError", "HarnessError",
                                  "ToolsetError", "EnvError", "InterceptionError", "TunnelError"})
EXCLUDED_DIRECTORIES = frozenset({"configs", "artifacts", "logs"})
METADATA_FILES = frozenset({"summary.json", "attempt_manifest.json"})


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def _loads(payload: bytes | str) -> Any:
    return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _entry(source: str, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"source": source, "status": status, "reason": reason, "phase": None,
            "reward": None, "reproduction_verified": False, "trace_count": 0,
            "dropped": False, "error_types": [], "trace_ids": [], **extra}


def _errors(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if any(not isinstance(error, dict) or not isinstance(error.get("type"), str)
           or not error["type"] for error in value):
        return None
    return [error["type"] for error in value]


def _error_classification(types: list[str]) -> tuple[str, str]:
    if types and all(error_type in INFRASTRUCTURE_ERRORS for error_type in types):
        return "infrastructure_error", "native_infrastructure_error"
    return "unknown", "unclassified_native_error"


def _reward(trace: dict[str, Any]) -> tuple[float | None, str | None]:
    rewards = trace.get("rewards")
    if rewards is None or rewards == {}:
        return None, "reward_missing"
    if not isinstance(rewards, dict):
        return None, "malformed_rewards"
    values = []
    for name, reward in rewards.items():
        if not isinstance(name, str) or not isinstance(reward, dict):
            return None, "reward_unscored" if reward is None else "malformed_rewards"
        score, weight = reward.get("score"), reward.get("weight", 1.0)
        if not _finite(score) or not _finite(weight):
            return None, "nonfinite_or_invalid_reward"
        value = score * weight
        if not _finite(value):
            return None, "nonfinite_or_invalid_reward"
        values.append(value)
    try:
        total = math.fsum(values)
    except OverflowError:
        return None, "nonfinite_or_invalid_reward"
    if not _finite(total) or not 0 <= total <= 1:
        return None, "reward_outside_pilot_range"
    return total, None


def _outcome(trace: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    info = trace.get("info", {})
    if not isinstance(info, dict):
        return None, "malformed_trace_info"
    primary = info.get(OUTCOME_KEY)
    diagnostics = info.get("scoring_diagnostics", {})
    fallback = diagnostics.get(OUTCOME_KEY) if isinstance(diagnostics, dict) else None
    if primary is not None and fallback is not None and primary != fallback:
        return None, "conflicting_outcome_metadata"
    outcome = primary if primary is not None else fallback
    if outcome is None:
        return None, "outcome_metadata_missing"
    if not isinstance(outcome, dict):
        return None, "malformed_outcome_metadata"
    if (outcome.get("status") not in STATUSES[:3]
            or not isinstance(outcome.get("reason"), str) or not outcome["reason"]
            or not isinstance(outcome.get("phase"), str) or not outcome["phase"]
            or type(outcome.get("reproduction_verified")) is not bool):
        return None, "malformed_outcome_metadata"
    return outcome, None


def _trace_entry(trace: Any, source: str) -> dict[str, Any]:
    if not isinstance(trace, dict) or not isinstance(trace.get("task"), dict):
        return _entry(source, "malformed", "not_a_trace")
    if not isinstance(trace.get("id"), str) or not trace["id"]:
        return _entry(source, "malformed", "trace_id_missing")
    result = _entry(source, "unknown", "outcome_metadata_missing", trace_count=1,
                    record_id=trace["id"], record_kind="trace", trace_ids=[trace["id"]],
                    dropped=trace.get("dropped") is not None)
    if type(trace.get("version")) is not int or trace["version"] != 1:
        return {**result, "reason": "unsupported_trace_schema"}
    types = _errors(trace.get("errors", []))
    if types is None:
        return {**result, "status": "malformed", "reason": "malformed_native_errors"}
    result["error_types"] = types
    outcome, outcome_error = _outcome(trace)
    if outcome is not None:
        result.update(phase=outcome["phase"], reproduction_verified=outcome["reproduction_verified"])
        if outcome["status"] == "infrastructure_error":
            return {**result, "status": "infrastructure_error", "reason": outcome["reason"]}
    # An earlier successful hook cannot conceal a later native runtime failure.
    if types:
        status, reason = _error_classification(types)
        return {**result, "status": status, "reason": reason}
    if trace.get("is_timeout") is True:
        return {**result, "reason": "unclassified_native_timeout"}
    if result["dropped"]:
        return {**result, "reason": "native_dropped_trace"}
    if outcome_error:
        return {**result, "reason": outcome_error}
    assert outcome is not None
    if trace.get("is_completed") is not True or trace.get("ok") is not True:
        return {**result, "reason": "native_trace_not_completed_successfully"}
    reward, reward_error = _reward(trace)
    if reward_error:
        return {**result, "reason": reward_error}
    if outcome["status"] == "candidate_failure":
        if reward != 0:
            return {**result, "reason": "candidate_failure_nonzero_reward"}
    elif not outcome["reproduction_verified"]:
        return {**result, "reason": "passed_without_verified_reproduction"}
    return {**result, "status": outcome["status"], "reason": outcome["reason"], "reward": reward}


def _record_entry(record: Any, source: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        return _entry(source, "malformed", "result_record_is_not_an_object")
    # vf-validate records are neither a Trace nor a model rollout. In particular,
    # a valid noop has reward zero; treating that as model failure is incorrect.
    if record.get("mode") in ("setup", "gold", "noop", "all") and "valid" in record:
        if (record.get("valid") is not None and type(record["valid"]) is not bool
                or not isinstance(record.get("reason"), str)):
            return _entry(source, "malformed", "malformed_validation_record")
        return _entry(source, "validation_only", "model_free_validation_record", phase="validation",
                      validation_reason=record.get("reason"), validation_valid=record.get("valid"))
    if "traces" not in record:
        return _trace_entry(record, source)
    traces = record["traces"]
    if not isinstance(traces, list):
        return _entry(source, "malformed", "episode_traces_is_not_a_list")
    types = _errors(record.get("errors", []))
    if types is None:
        return _entry(source, "malformed", "malformed_episode_errors", trace_count=len(traces))
    if len(traces) == 1:
        result = _trace_entry(traces[0], source)
    else:
        result = _entry(source, "unknown", "episode_has_no_trace" if not traces else "multiple_agent_traces",
                        trace_count=len(traces),
                        trace_ids=[trace["id"] for trace in traces
                                   if isinstance(trace, dict) and isinstance(trace.get("id"), str)])
    if isinstance(record.get("id"), str) and record["id"]:
        result["record_id"] = record["id"]
        result["record_kind"] = "episode"
    else:
        return {**result, "status": "malformed", "reason": "episode_id_missing", "reward": None}
    if types:
        status, reason = _error_classification(types)
        return {**result, "status": status, "reason": reason, "reward": None,
                "error_types": result["error_types"] + types}
    if record.get("is_timeout") is True:
        return {**result, "status": "unknown", "reason": "unclassified_episode_timeout", "reward": None}
    if record.get("ok") is not True and result["status"] in ("passed", "candidate_failure"):
        return {**result, "status": "unknown", "reason": "native_episode_not_ok", "reward": None}
    return result


def _read_entries(path: Path, source: str) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        entries = []
        try:
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    location = f"{source}:{line_number}"
                    try:
                        record = _loads(line)
                    except (ValueError, UnicodeError, RecursionError):
                        entries.append(_entry(location, "malformed", "invalid_json"))
                    else:
                        entries.append(_record_entry(record, location))
        except OSError:
            entries.append(_entry(source, "unknown", "result_file_unreadable"))
        return entries
    try:
        payload = path.read_bytes()
    except OSError:
        return [_entry(source, "unknown", "result_file_unreadable")]
    try:
        data = _loads(payload)
    except (ValueError, UnicodeError, RecursionError):
        return [_entry(source, "malformed", "invalid_json")]
    records = data if isinstance(data, list) else [data]
    return [_record_entry(record, f"{source}:{index + 1}") for index, record in enumerate(records)]


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in (".json", ".jsonl"):
            raise ValueError("Expected a JSON/JSONL result file or run directory")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [file for file in path.rglob("*.jsonl")
             if not (set(file.relative_to(path).parts[:-1]) & EXCLUDED_DIRECTORIES)]
    # Standalone trace JSON files are supported in a flat result directory. Do
    # not descend into saved task data or candidate artifacts looking for JSON.
    files.extend(file for file in path.glob("*.json") if file.name not in METADATA_FILES)
    return sorted(set(files))


def _manifest_expected(root: Path, issues: list[dict[str, str]]) -> int | None:
    direct = root / "attempt_manifest.json"
    manifests = [direct] if direct.exists() else sorted(
        path for path in root.rglob("attempt_manifest.json")
        if not (set(path.relative_to(root).parts[:-1]) & EXCLUDED_DIRECTORIES)
    )
    if not manifests:
        return None
    totals = []
    for manifest in manifests:
        source = str(manifest.relative_to(root))
        try:
            data = _loads(manifest.read_bytes())
            count = data.get("expected_attempts") if isinstance(data, dict) else None
            if type(count) is not int or count < 0:
                raise ValueError("expected_attempts must be a nonnegative integer")
            totals.append(count)
        except (OSError, ValueError, UnicodeError, RecursionError):
            issues.append({"code": "invalid_attempt_manifest", "source": source})
    return sum(totals) if len(totals) == len(manifests) else None


def report_path(input_path: str | Path, expected: int | None = None) -> dict[str, Any]:
    """Account for every saved entry, including errors and unrecorded attempts.

    ``passed`` means scoring completed with verified reproduction, not full
    financial credit. Candidate failures require an explicit authoritative
    classification and a recorded zero reward. Unclassified errors never
    become candidate zeros. Duplicate entries stay visible and block an overall
    performance claim. No model call, runtime operation or private task lookup
    occurs here.
    """
    if expected is not None and (type(expected) is not int or expected < 0):
        raise ValueError("expected must be a nonnegative integer")
    path = Path(input_path)
    files = _result_files(path)
    root = path if path.is_dir() else path.parent
    issues: list[dict[str, str]] = []
    manifest_expected = _manifest_expected(root, issues)
    if expected is not None and manifest_expected is not None and expected != manifest_expected:
        issues.append({"code": "conflicting_expected_counts", "source": "expected/attempt_manifest.json"})
    expected_count = max(count for count in (expected, manifest_expected) if count is not None) \
        if expected is not None or manifest_expected is not None else None
    entries = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for file in files:
        source = str(file.relative_to(root))
        for entry in _read_entries(file, source):
            identities = [("trace", trace_id) for trace_id in entry["trace_ids"]]
            if entry.get("record_kind") == "episode":
                identities.append(("episode", entry["record_id"]))
            for identity in identities:
                if identity in seen:
                    earlier = seen[identity]
                    if earlier is entry:
                        continue
                    for duplicate in (earlier, entry):
                        duplicate.update(status="unknown", reason="duplicate_record_id", reward=None)
                    issue = {"code": "duplicate_record_id", "source": entry["source"]}
                    if issue not in issues:
                        issues.append(issue)
                else:
                    seen[identity] = entry
            entries.append(entry)
    if not files:
        issues.append({"code": "no_result_files", "source": "."})
    if expected_count is None:
        issues.append({"code": "expected_attempts_unknown", "source": "."})
    elif len(entries) > expected_count:
        issues.append({"code": "more_records_than_expected", "source": "."})
    missing = max(0, expected_count - len(entries)) if expected_count is not None else 0
    counts = {status: 0 for status in STATUSES}
    counts.update(Counter(entry["status"] for entry in entries))
    counts["missing"] = missing
    denominator = len(entries) + missing
    resolved = counts["passed"] + counts["candidate_failure"]
    rewards = [entry["reward"] for entry in entries if entry["reward"] is not None]
    full_credit = sum(entry["status"] == "passed" and entry["reward"] >= 1 - 1e-9 for entry in entries)
    usable = bool(denominator and not issues and resolved == denominator)
    validation_counts = Counter(entry.get("validation_reason", "unknown")
                                for entry in entries if entry["status"] == "validation_only")
    return {
        "schema_version": 1,
        "expected_attempts": expected_count,
        "recorded_entries": len(entries),
        "denominator": denominator,
        "counts": counts,
        "recorded_trace_count": sum(entry["trace_count"] for entry in entries),
        "dropped_trace_entries": sum(entry["dropped"] for entry in entries),
        "scored_attempts": len(rewards),
        "full_credit_attempts": full_credit,
        "reproduction_verified_attempts": sum(entry["reproduction_verified"] for entry in entries
                                               if entry["status"] in ("passed", "candidate_failure")),
        "performance_usable": usable,
        "comparison_ready": usable,
        "mean_reward": math.fsum(rewards) / denominator if usable else None,
        "mean_reward_scored_only": math.fsum(rewards) / len(rewards) if rewards else None,
        "resolved_coverage": resolved / denominator if denominator else None,
        "recorded_coverage": min(1.0, len(entries) / expected_count) if expected_count else None,
        "confirmed_full_credit_lower_bound": full_credit / denominator if denominator else None,
        "validation_counts": dict(validation_counts),
        "issues": issues,
        "attempts": entries,
        "interpretation": {
            "passed": "Scoring completed after verified reproduction; financial reward may be partial or zero.",
            "mean_reward_scored_only": "Conditional diagnostic over resolved scored entries; not an overall performance estimate.",
            "confirmed_full_credit_lower_bound": "Confirmed full-credit entries divided by every saved entry plus missing planned attempts; not a model performance estimate when performance_usable is false.",
            "coverage": "Without an expected count or attempt manifest, absent planned attempts cannot be detected.",
        },
    }
