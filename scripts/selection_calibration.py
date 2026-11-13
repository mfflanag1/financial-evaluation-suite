"""Offline preparation and fail-closed accounting for five selection attempts.

No command in this module invokes inference, Docker, an optimizer, or a network
client. Reports verify native records against a preregistered local binding;
local files are not authenticated provider receipts or proof of execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import stat
import sys
import tomllib
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_selection/calibration_v1"
MANIFEST = CONFIG / "manifest.json"
CONDITIONS = ("mixed", "positive", "negative", "unsupported", "sparse")
PACKAGES = ("financial_selection", "financial_research", "financial_portfolio_staged")
METRICS = frozenset(("candidate_failure", "reproduction_verified", "packets_scored",
                    "exact_success_fraction", "all_packets_correct", "current_exact_success",
                    "fresh_exact_success_fraction"))
INFO_KEY = "financial_selection_outcome"
MAX_FILE_BYTES = 64 * 1024 * 1024
IMAGE_PLACEHOLDER = "sha256:" + "0" * 64


class CalibrationError(ValueError):
    """Invalid protocol, binding, or local record; no performance aggregate."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def object_digest(value):
    return digest(canonical(value))


def read_bytes(path, limit=MAX_FILE_BYTES):
    path = Path(path)
    # Reject links at every existing ancestor, including links into valid trees.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise CalibrationError("symlink_input")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise CalibrationError("invalid_file")
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise CalibrationError("oversized_file")
    return data


def loads(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CalibrationError("duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise CalibrationError("nonfinite_json")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise CalibrationError("nonfinite_json")
        return result

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise CalibrationError("invalid_json") from exc


def read_json(path):
    return loads(read_bytes(path))


def _payload_paths():
    paths = {Path(__file__), ROOT / "tests/test_selection_calibration.py",
             ROOT / "requirements/runtime-constraints.txt",
             ROOT / "environments/financial_selection/CONTRACT.md",
             ROOT / "environments/financial_selection/VALIDATION.md"}
    for package in PACKAGES:
        base = ROOT / "environments" / package
        paths.add(base / "pyproject.toml")
        paths.update(p for p in (base / "src").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.update(p for p in CONFIG.rglob("*") if p.is_file() and p != MANIFEST
                 and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return sorted(paths)


def build_manifest():
    return dict(protocol_id="selection-calibration-v1", schema=1,
                schedule=read_json(CONFIG / "schedule.json"),
                sha256={p.relative_to(ROOT).as_posix(): digest(read_bytes(p)) for p in _payload_paths()})


def verify_manifest():
    frozen = read_json(MANIFEST)
    if frozen != build_manifest():
        raise CalibrationError("frozen_payload_changed")
    return digest(read_bytes(MANIFEST))


def verify_native_installation():
    """Require pinned installed resources, not a source-tree path overlay."""
    if sys.version_info < (3, 12) or importlib.metadata.version("verifiers") != "0.4.1":
        raise CalibrationError("native_version_mismatch")
    frozen = read_json(CONFIG / "native_sources.json")
    for package, expected in frozen.items():
        module = importlib.import_module(package)
        base = Path(module.__file__).resolve().parent
        actual = {p.relative_to(base).as_posix(): digest(read_bytes(p))
                  for p in base.rglob("*.py") if "__pycache__" not in p.parts}
        if actual != expected:
            raise CalibrationError("native_source_changed")
    for package in PACKAGES:
        module = importlib.import_module(package)
        installed = Path(module.__file__).resolve().parent
        source = ROOT / "environments" / package / "src" / package
        if installed == source.resolve() or ROOT in installed.parents:
            raise CalibrationError("noneditable_install_required")
        expected = {p.relative_to(source).as_posix(): digest(read_bytes(p))
                    for p in source.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        actual = {p.relative_to(installed).as_posix(): digest(read_bytes(p))
                  for p in installed.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if actual != expected:
            raise CalibrationError("installed_package_changed")


def _validate_identity(model, endpoint, image):
    if (not isinstance(model, str) or not model.strip() or model == "INSTRUCT_MODEL_ID"
            or len(model) > 256 or any(ord(c) < 32 for c in model)):
        raise CalibrationError("model_identity_required")
    address = urlsplit(endpoint)
    if (address.scheme not in {"http", "https"} or not address.hostname
            or address.username or address.password or address.query or address.fragment
            or any(ord(c) < 32 for c in endpoint)):
        raise CalibrationError("invalid_endpoint")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or image == IMAGE_PLACEHOLDER:
        raise CalibrationError("immutable_local_image_required")


def _render(condition, output, plan_id, model, endpoint, image):
    source = read_bytes(CONFIG / (condition + ".toml")).decode()
    replacements = {"INSTRUCT_MODEL_ID": model, "https://endpoint.invalid/v1": endpoint,
                    "CALIBRATION_OUTPUT_DIRECTORY": str(output / "results"),
                    "selection-calibration-" + condition: "selection-calibration-" + plan_id + "-" + condition,
                    IMAGE_PLACEHOLDER: image}
    for old, new in replacements.items():
        source = source.replace(json.dumps(old), json.dumps(new))
    return source


def prepare(output, *, model, endpoint, image):
    """Bind current packets only; no final or fresh packet is generated here."""
    manifest_hash = verify_manifest()
    verify_native_installation()
    _validate_identity(model, endpoint, image)
    output = Path(output).absolute()
    if ROOT == output or ROOT in output.resolve().parents or output.exists():
        raise CalibrationError("new_external_directory_required")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise CalibrationError("symlink_output")
    from verifiers.v1.configs.cli.run import RunnerConfig
    from verifiers.v1.configs.agent import resolve_agent
    from verifiers.v1.trace import TraceTask
    from verifiers.v1.utils.loaders import load_taskset
    from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
    from financial_selection.contract import input_digest
    from financial_selection.prime_adapter import PROMPT

    plan_id = uuid.uuid4().hex
    attempts, rendered = [], {}
    for condition in CONDITIONS:
        text = _render(condition, output, plan_id, model, endpoint, image)
        config = RunnerConfig.model_validate(tomllib.loads(text))
        tasks = list(load_taskset(config.env.taskset))
        if len(tasks) != 1 or not config.dry_run:
            raise CalibrationError("unexpected_schedule")
        task = tasks[0]
        agent = resolve_agent(config.env.agent, model=config.model, client=config.client,
                              sampling=config.sampling)
        wire_task = TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash)
        rendered[condition + ".toml"] = text
        attempts.append(dict(attempt_id=condition + "-1", condition=condition, split="train", index=0,
                             task_id=task.data.id, task_sha256=object_digest(wire_task.model_dump(mode="json", exclude_none=True)),
                             public_input_sha256=input_digest(task.data.public_files),
                             contract_sha256=digest(task.data.public_files["contract.json"].encode()),
                             config_sha256=digest(text.encode()),
                             resolved_config_sha256=object_digest(config.model_dump(mode="json")),
                             agent_sha256=object_digest(agent.model_dump(mode="json", exclude_none=True)),
                             run_name=config.run.name, result_dir=condition))
    if len({row["task_id"] for row in attempts}) != 5:
        raise CalibrationError("nonunique_task_identity")
    binding = dict(schema=1, protocol_id="selection-calibration-v1", manifest_sha256=manifest_hash,
                   plan_id=plan_id, model_class="instruct", model=model, endpoint=endpoint,
                   image=image, expected_attempts=5, split="train", final_selected=False,
                   retries=0, automatic_expansion=False,
                   task_prompt_sha256=digest(PROMPT.encode()),
                   harness_system_prompt_sha256=digest((BASH_SYSTEM_PROMPT + " " + EDIT_SYSTEM_PROMPT).encode()),
                   attempts=attempts)
    # Write only after every task and configuration has validated. The key and
    # public/fresh packets are deliberately absent from the preparation bundle.
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, content in rendered.items():
        (output / name).write_text(content)
    (output / "binding.json").write_bytes(canonical(binding) + b"\n")
    return dict(binding_sha256=digest(read_bytes(output / "binding.json")), plan_id=plan_id,
                expected_attempts=5, dry_run=True, model_calls=0)


def _binding(path, expected_hash):
    payload = read_bytes(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or "") or digest(payload) != expected_hash:
        raise CalibrationError("binding_digest_mismatch")
    binding = loads(payload)
    if (binding.get("schema") != 1 or binding.get("protocol_id") != "selection-calibration-v1"
            or binding.get("manifest_sha256") != verify_manifest()
            or binding.get("model_class") != "instruct" or binding.get("split") != "train"
            or binding.get("final_selected") is not False or binding.get("retries") != 0
            or binding.get("automatic_expansion") is not False or binding.get("expected_attempts") != 5):
        raise CalibrationError("invalid_binding")
    _validate_identity(binding.get("model"), binding.get("endpoint"), binding.get("image"))
    rows = binding.get("attempts", [])
    if (len(rows) != 5 or [r.get("condition") for r in rows] != list(CONDITIONS)
            or len({r.get("task_id") for r in rows}) != 5
            or any(r.get("attempt_id") != c + "-1" or r.get("result_dir") != c
                   or r.get("split") != "train" or type(r.get("index")) is not int or r["index"] != 0
                   for c, r in zip(CONDITIONS, rows))):
        raise CalibrationError("invalid_schedule")
    return binding


def _finite(value):
    # Every Python integer is finite; avoid converting adversarial large values
    # to float here. Reward/metric domain checks below reject out-of-range ints.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _score(trace):
    """Mirror the pinned task's persisted-record admission; no component inference."""
    if (trace.get("ok") is not True or trace.get("is_completed") is not True
            or trace.get("errors") != [] or trace.get("is_timeout") is not False
            or trace.get("dropped") or trace.get("is_format_violation")):
        raise CalibrationError("infrastructure_or_incomplete")
    info = trace.get("info", {})
    outcome = info.get(INFO_KEY, {})
    if outcome != info.get("scoring_diagnostics", {}).get(INFO_KEY) or outcome.get("status") not in {"scored", "candidate_failure"}:
        raise CalibrationError("unresolved_outcome")
    rewards, metrics = trace.get("rewards", {}), trace.get("metrics", {})
    reward = rewards.get("selection_reward")
    if (set(rewards) != {"selection_reward"} or not isinstance(reward, dict)
            or type(reward.get("weight")) not in (int, float) or reward["weight"] != 1
            or not _finite(reward.get("score")) or not 0 <= reward["score"] <= 1
            or not METRICS <= set(metrics) or any(not _finite(metrics[n]) for n in METRICS)):
        raise CalibrationError("incomplete_scoring")
    value = reward["score"]
    m = metrics
    if "reward" in trace and trace["reward"] != value:
        raise CalibrationError("inconsistent_reward")
    if outcome["status"] == "candidate_failure":
        consistent = (value == 0 and outcome.get("reproduction_verified") is False
                      and m["candidate_failure"] == 1 and all(m[n] == 0 for n in METRICS - {"candidate_failure"}))
    else:
        consistent = (outcome.get("reproduction_verified") is True and m["candidate_failure"] == 0
                      and m["reproduction_verified"] == 1 and m["packets_scored"] == 3
                      and m["current_exact_success"] in (0, 1) and m["fresh_exact_success_fraction"] in (0, .5, 1)
                      and m["exact_success_fraction"] == (m["current_exact_success"] + 2*m["fresh_exact_success_fraction"])/3
                      and m["all_packets_correct"] == float(m["exact_success_fraction"] == 1)
                      and value >= m["exact_success_fraction"]
                      and (value == 1) == (m["all_packets_correct"] == 1))
    if not consistent:
        raise CalibrationError("inconsistent_scoring")
    return dict(status=outcome["status"], reward=value, **{n: m[n] for n in sorted(METRICS)})


def _record(record, row, binding):
    if (record.get("ok") is not True or record.get("is_timeout") is not False or record.get("errors") != []
            or not isinstance(record.get("id"), str) or not record["id"]
            or len(record.get("traces", [])) != 1):
        raise CalibrationError("unresolved_episode")
    trace = record["traces"][0]
    run = record.get("run", {})
    if (run.get("type") != "eval" or run.get("name") != row["run_name"]
            or not isinstance(run.get("id"), str) or not run["id"] or trace.get("run") != run
            or not isinstance(trace.get("id"), str) or not trace["id"]):
        raise CalibrationError("unbound_run")
    for task in (record.get("task", {}), trace.get("task", {})):
        if object_digest(task) != row["task_sha256"] or task.get("data", {}).get("id") != row["task_id"]:
            raise CalibrationError("unbound_task")
    agent = trace.get("agent", {})
    if (object_digest(agent.get("config")) != row["agent_sha256"]
            or trace.get("verifiers", {}).get("version") != "0.4.1"):
        raise CalibrationError("unbound_agent")
    runtime = agent.get("runtime", {})
    if (runtime.get("type") != "docker" or runtime.get("image") != binding["image"]
            or runtime.get("allow") != [] or runtime.get("workdir") != "/workspace"
            or runtime.get("borrowed") is not False or not isinstance(runtime.get("id"), str)
            or not runtime["id"] or runtime.get("mounts") or runtime.get("checkpoint")):
        raise CalibrationError("unbound_runtime")
    calls = trace.get("calls")
    if not isinstance(calls, list) or not 0 < len(calls) <= 32:
        raise CalibrationError("missing_model_call_evidence")
    _prompt_paths(trace, binding)
    for call in calls:
        sampling = call.get("sampling", {})
        if (call.get("model") != binding["model"] or call.get("error") or call.get("policy")
                or call.get("endpoint") != "/chat/completions"
                or sampling.get("temperature") != 0.2 or sampling.get("reasoning_effort") is not None
                or type(sampling.get("max_tokens")) is not int or not 0 < sampling["max_tokens"] <= 8192):
            raise CalibrationError("unbound_model_call")
    result = _score(trace)
    result.update(episode_id=record["id"], trace_id=trace["id"], run_id=run["id"], runtime_id=runtime["id"])
    return result


def _text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, dict) and part.get("type") == "text"
                                      and isinstance(part.get("text"), str) for part in value):
        return "\n".join(part["text"] for part in value)
    raise CalibrationError("nontext_prompt")


def _prompt_paths(trace, binding):
    """Join every call to its acyclic recorded prefix, not an orphan prompt node."""
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not 3 <= len(nodes) <= 4096:
        raise CalibrationError("missing_prompt_evidence")
    sampled = set()
    for index, node in enumerate(nodes):
        parent = node.get("parent")
        if parent is not None and (type(parent) is not int or not 0 <= parent < index):
            raise CalibrationError("invalid_prompt_ancestry")
        if node.get("sampled") is True:
            sampled.add(index)
    seen = set()
    for call in trace["calls"]:
        target = call.get("node")
        if type(target) is not int or target not in sampled or target in seen:
            raise CalibrationError("unbound_call_node")
        seen.add(target)
        path = []
        while target is not None:
            path.append(nodes[target])
            target = nodes[target].get("parent")
        path.reverse()
        if (len(path) < 3 or path[0]["message"].get("role") != "system"
                or path[1]["message"].get("role") != "user"
                or path[0].get("sampled") is True or path[1].get("sampled") is True
                or path[-1]["message"].get("role") != "assistant"
                or any(n["message"].get("role") in {"user", "system"} for n in path[2:])):
            raise CalibrationError("unexpected_prompt_path")
        for node, expected in ((path[0], binding["harness_system_prompt_sha256"]),
                               (path[1], binding["task_prompt_sha256"])):
            if digest(_text_content(node["message"].get("content")).encode()) != expected:
                raise CalibrationError("outbound_prompt_changed")
    if sampled != seen:
        raise CalibrationError("unaccounted_sampled_node")


def report(directory, *, binding_sha256):
    """All five planned attempts are retained; any unresolved row blocks means."""
    directory = Path(directory)
    binding = _binding(directory / "binding.json", binding_sha256)
    rows, issues, identities = [], [], set()
    results = directory / "results"
    if results.exists():
        if results.is_symlink() or any(p.name not in CONDITIONS for p in results.iterdir()):
            issues.append("unplanned_result_directory")
        # Refuse extra attempts hidden in alternate trace files or nested folders.
        expected = {results / c / "traces.jsonl" for c in CONDITIONS}
        if any(p not in expected for p in results.rglob("traces*.jsonl")):
            issues.append("unplanned_trace_file")
    for row in binding["attempts"]:
        result = dict(attempt_id=row["attempt_id"], condition=row["condition"], status="unresolved",
                      reward=None, performance_usable=False)
        try:
            config_path = directory / (row["condition"] + ".toml")
            if digest(read_bytes(config_path)) != row["config_sha256"]:
                raise CalibrationError("prepared_config_changed")
            actual = read_json(results / row["result_dir"] / "configs/resolved/run.json")
            if object_digest(actual) != row["resolved_config_sha256"]:
                raise CalibrationError("unbound_resolved_config")
            data = read_bytes(results / row["result_dir"] / "traces.jsonl")
            records = [loads(line) for line in data.splitlines() if line.strip()]
            if len(records) != 1:
                raise CalibrationError("missing_or_duplicate_attempt")
            scored = _record(records[0], row, binding)
            ids = {(kind, scored[kind]) for kind in ("episode_id", "trace_id", "run_id", "runtime_id")}
            if identities & ids:
                raise CalibrationError("duplicate_identity")
            identities.update(ids)
            result.update(scored, performance_usable=True)
        except (CalibrationError, FileNotFoundError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            result["reason"] = str(exc) if isinstance(exc, CalibrationError) else "missing_or_malformed_record"
        rows.append(result)
    usable = sum(row["performance_usable"] for row in rows)
    complete = usable == 5 and not issues
    return dict(protocol_id="selection-calibration-v1", binding_sha256=binding_sha256,
                model=binding["model"], planned_attempts=5, resolved_attempts=usable,
                unresolved_attempts=5-usable, accounting_complete=complete, issues=issues, attempts=rows,
                mean_reward=sum(row["reward"] for row in rows)/5 if complete else None,
                mean_exact_success_fraction=sum(row["exact_success_fraction"] for row in rows)/5 if complete else None,
                all_packets_correct_count=sum(row["all_packets_correct"] for row in rows) if complete else None,
                reproduction_verified_count=sum(row["reproduction_verified"] for row in rows) if complete else None,
                candidate_failure_count=sum(row.get("status") == "candidate_failure" for row in rows),
                components=None,
                interpretation="Five train calibration attempts; no estimate of generalization or investment performance.",
                evidence_scope="Local native records joined to a retained preparation digest; not authenticated provider receipts.",
                baseline={"scope": "Published outcome-blind cash control on 100 different public development fixtures; not paired model results.",
                          "source": "environments/financial_selection/VALIDATION.md",
                          "command": "python -B -m financial_selection.baseline",
                          "n": 100, "full_policy_success_count": 0,
                          "rounded_mean_reward": 0.67454,
                          "rounded_condition_mean_reward": {"mixed": .54, "positive": .525,
                              "negative": .825, "unsupported": .74625, "sparse": .73646}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--write", action="store_true")
    commands.add_parser("check")
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--endpoint", required=True)
    prep.add_argument("--image", required=True)
    account = commands.add_parser("report")
    account.add_argument("directory")
    account.add_argument("--binding-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if not args.write:
            parser.error("freeze requires --write; it never replaces an existing manifest")
        with MANIFEST.open("x") as target:
            json.dump(build_manifest(), target, indent=2, sort_keys=True)
            target.write("\n")
        result = {"manifest_sha256": verify_manifest()}
    elif args.command == "check":
        result = {"manifest_sha256": verify_manifest(), "expected_attempts": 5, "model_calls": 0}
    elif args.command == "prepare":
        result = prepare(args.output, model=args.model, endpoint=args.endpoint, image=args.image)
    else:
        result = report(args.directory, binding_sha256=args.binding_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("accounting_complete", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CalibrationError, FileNotFoundError) as error:
        print("calibration rejected: " + str(error), file=sys.stderr)
        raise SystemExit(2)
