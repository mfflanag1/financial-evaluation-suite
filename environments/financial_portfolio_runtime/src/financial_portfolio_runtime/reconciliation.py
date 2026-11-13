"""Join native attempts to a predeclared schedule and independent host journals."""

from collections import Counter
import hashlib
from pathlib import Path

from financial_portfolio_operations.telemetry import read_journal, summarize_journal
from financial_research.accounting import _loads

from .telemetry import INFO_KEY, prompt_binding


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def tool_states(events):
    """Reconstruct the summary from ordered host observations, failing on gaps."""
    states = {}
    for event in events:
        kind = event["event"]
        if kind not in ("tool_gate_started", "tool_gate_terminal", "tool_result_observed"):
            continue
        occurrence = event["occurrence"]
        if kind == "tool_gate_started":
            if occurrence in states:
                raise ValueError("Duplicate tool gate")
            states[occurrence] = {"status": "gating", "tool_name": event["tool_name"],
                                  "tool_call_id": occurrence.partition(":")[2]}
        elif kind == "tool_gate_terminal":
            if states[occurrence]["status"] != "gating" or event["status"] not in ("allowed", "gate_replied"):
                raise ValueError("Invalid tool gate transition")
            states[occurrence]["status"] = event["status"]
        else:
            if states[occurrence]["status"] != "allowed":
                raise ValueError("Result without an allowed tool")
            states[occurrence]["status"] = "result_observed"
    return states


def reconcile(records, *, expected, journal_dir):
    """A local consistency check, not a cryptographic attestation of the evaluator."""
    issues = []
    if journal_dir is None:
        return {"issues": ["run_binding_missing"], "per_trace": {}, "request_accounting": None}
    root = Path(journal_dir)
    try:
        binding = _loads((root / "run_binding.json").read_bytes())
        config_bytes = (root / "bound.json").read_bytes()
        config = _loads(config_bytes)
        lifecycle = []
        for line in (root / "lifecycle.jsonl").read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete lifecycle record")
            lifecycle.append(_loads(line))
        loaded = read_journal(root / "provider")
        request_accounting = summarize_journal(root / "provider")
        if (binding["schema_version"] != 1 or type(binding["schema_version"]) is not int
                or binding["protocol_id"] != "portfolio-runtime-v1" or binding["dry_run"] is not False
                or type(binding["expected_attempts"]) is not int or binding["expected_attempts"] != expected
                or binding["config_sha256"] != sha(config_bytes)
                or binding["model"] != config["model"]
                or not isinstance(binding["capture_id"], str) or not binding["capture_id"]
                or not isinstance(binding["provider_run_id"], str) or not binding["provider_run_id"]):
            issues.append("invalid_run_binding")
        schedule = binding["schedule"]
        planned = [row["task_id"] for row in schedule]
        if (len(schedule) != expected or len(set(planned)) != expected
                or any(not isinstance(task_id, str) or not task_id for task_id in planned)):
            issues.append("invalid_planned_schedule")
        by_task = {row["task_id"]: row for row in schedule}
        traces = [record["traces"][0] if "traces" in record else record for record in records]
        actual = [trace["task"]["data"]["id"] for trace in traces]
        if Counter(actual) != Counter(planned):
            issues.append("task_schedule_mismatch")
        trace_ids = [trace["id"] for trace in traces]
        if len(trace_ids) != len(set(trace_ids)):
            issues.append("duplicate_trace_id")
        if any(row["trace_id"] not in trace_ids or row.get("run_id") != binding["capture_id"] for row in lifecycle):
            issues.append("foreign_lifecycle_records")
        if loaded["errors"] or request_accounting["issues"] or request_accounting["pending_count"]:
            issues.append("provider_journal_unreconciled")
        if any(row["trace_id"] not in trace_ids or row["run_id"] != binding["provider_run_id"] for row in loaded["records"]):
            issues.append("foreign_provider_records")
        per_trace = {}
        for native, trace in zip(records, traces):
            local = []
            trace_id, data = trace["id"], trace["task"]["data"]
            row = by_task.get(data["id"])
            if row is None:
                per_trace[trace_id] = ["unexpected_task"]
                continue
            if "traces" in native and native.get("task", {}).get("data", {}).get("id") != data["id"]:
                local.append("episode_trace_task_mismatch")
            if ({name: sha(value.encode()) for name, value in data["public_files"].items()} != row["public_sha256"]
                    or sha(data["prompt"].encode()) != row["prompt_sha256"]
                    or sha(data["system_prompt"].encode()) != row["system_prompt_sha256"]):
                local.append("task_input_binding_mismatch")
            if (data["image"] != binding["docker_image_id"]
                    or trace["agent"]["runtime"]["image"] != binding["docker_image_id"]):
                local.append("native_image_binding_mismatch")
            info = trace["info"][INFO_KEY]
            events = [event for event in lifecycle if event["trace_id"] == trace_id]
            host_events = [event for event in events if event["event"] not in ("prompt_bound", "outbound_prompt_checked")]
            if host_events != info["events"] or info.get("capture_id") != binding["capture_id"]:
                local.append("lifecycle_trace_mismatch")
            if tool_states(host_events) != info.get("tools", {}):
                local.append("tool_state_lifecycle_mismatch")
            prompt_rows = [event for event in events if event["event"] == "prompt_bound"]
            fields = set(row["outbound_prompt"])
            if (len(prompt_rows) != 1 or info.get("prompt_binding") != row["outbound_prompt"]
                    or {name: prompt_rows[0].get(name) for name in fields} != row["outbound_prompt"]):
                local.append("prompt_binding_mismatch")
            checks = [event for event in events if event["event"] == "outbound_prompt_checked"]
            if (not checks or any(event.get("valid") is not True for event in checks)
                    or [event.get("request_index") for event in checks] != list(range(1, len(checks) + 1))
                    or checks[0].get("messages_sha256") != row["outbound_prompt"]["initial_messages_sha256"]):
                local.append("outbound_prompt_checks_invalid")
            # Saved native calls have a graph rather than raw request bodies. The
            # first two committed nodes must reproduce the bound initial prompt.
            if trace.get("calls"):
                initial = [node["message"] for node in trace["nodes"][:2]]
                if prompt_binding(initial) != row["outbound_prompt"]:
                    local.append("native_initial_prompt_mismatch")
            if any(call.get("model") != binding["model"] or call.get("sampling") != binding["sampling"]
                   for call in trace.get("calls", [])):
                local.append("native_model_or_sampling_mismatch")
            provider = [event for event in loaded["records"] if event["trace_id"] == trace_id]
            starts = [event for event in provider if event["event"] == "request_started"]
            successes = [event for event in provider if event["event"] == "request_terminal" and event["outcome"] == "success"]
            if (len(starts) != len(checks)
                    or [event["request_index"] for event in starts] != list(range(1, len(checks) + 1))
                    or any(event["task_id"] != data["id"] or event["model"] != binding["model"] for event in provider)
                    or len(trace.get("calls", [])) > len(starts)
                    or len(successes) > len(trace.get("calls", []))):
                local.append("provider_trace_correspondence_mismatch")
            per_trace[trace_id] = local
        return {"issues": sorted(set(issues)), "per_trace": per_trace, "request_accounting": request_accounting,
                "binding_sha256": sha((root / "run_binding.json").read_bytes())}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration, RecursionError):
        return {"issues": ["invalid_or_incomplete_bound_evidence"], "per_trace": {}, "request_accounting": None}
