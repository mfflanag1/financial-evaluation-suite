"""Check a narrow, trusted-local one-step optimizer observation bundle.

This reads JSON and hashes files; it never runs training or deserializes a
checkpoint. Consistency is not authentication of the machine or collector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys

SCHEMA = "selection-optimizer-evidence-v1"
PRIME_RL_COMMIT = "c225aaa45d2160380b2c9c160d072b12af5c785f"
PINNED_SOURCES = {
    "prime_rl/orchestrator/algo/grpo.py": "3f125f2d41b1daa5dbd448e48ff13b1fad9138e00402f558b34caeac3881cb4f",
    "prime_rl/trainer/rl/train.py": "36e2b9a24b7b21f0c0e9508aa8d801bdad41280c083776fd1fa92a70053b42ec",
    "prime_rl/trainer/ckpt.py": "6f91936debd86573340cb77bceb3b13a20c1d213adc0f1d8a8c83e44e1e672a3",
    "prime_rl/orchestrator/watcher.py": "b8a5a432d1e97ec38a60019551f006b9db3021cce37e255e228776c7ca8590db",
    "prime_rl/orchestrator/ckpt.py": "ad3ef409cd302e8bfe26bfa67e7d968d7d67435b933fea6b65770cb4bae7c558",
}
RECORDS = {"runtime", "cohort", "before", "after", "checkpoint", "reload", "receiver", "inference"}
COMPONENTS = {"model", "optimizers", "scheduler", "progress"}


class EvidenceError(ValueError):
    """The bundle cannot support this validation gate."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise EvidenceError("Non-finite JSON number: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=invalid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")
    return value


def _text(value):
    require(isinstance(value, str) and bool(value.strip()), "Missing nonempty text")
    return value


def _integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer observation")
    return value


def _finite(value):
    require(type(value) in (int, float), "Invalid numeric observation")
    try:
        require(math.isfinite(value), "Non-finite observation")
    except OverflowError as exc:
        raise EvidenceError("Numeric observation overflow") from exc
    return value


def _f32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise EvidenceError("Float32 observation overflow") from exc


def _path(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid artifact path")
    parts = PurePosixPath(name).parts
    require(not name.startswith("/") and all(part not in (".", "..") for part in parts)
            and str(PurePosixPath(name)) == name, "Artifact path is not canonical and relative")
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), "Symlink artifact refused")
    require(path.is_file(), "Missing regular artifact: " + name)
    return path


def _inventory(value, allow_empty=False):
    require(isinstance(value, dict) and (value or allow_empty), "Empty or invalid state inventory")
    for key, checksum in value.items():
        _text(key)
        _sha(checksum)
    return value


def _state(record):
    state = record["state"]
    require(isinstance(state, dict) and set(state) == COMPONENTS, "Incomplete native AppState")
    for component, values in state.items():
        _inventory(values, allow_empty=component == "optimizers")
    counters = record["optimizer_steps"]
    require(isinstance(counters, dict) and counters, "Missing per-parameter optimizer counters")
    for key, value in counters.items():
        _text(key)
        _integer(value)
    progress = record["progress"]
    require(set(progress) == {"step", "total_tokens", "total_samples"}, "Incomplete native Progress")
    for value in progress.values():
        _integer(value)
    require(state["progress"] == {key: digest(value) for key, value in progress.items()},
            "Progress inventory differs from observed counters")
    return state, counters, progress


def _sample(record, policy_version):
    ids, logprobs, versions = (record[key] for key in ("token_ids", "logprobs", "weight_versions"))
    require(isinstance(ids, list) and ids and isinstance(logprobs, list)
            and isinstance(versions, list) and len(ids) == len(logprobs) == len(versions),
            "Missing or misaligned sampled token evidence")
    for token, logprob, version in zip(ids, logprobs, versions):
        _integer(token)
        require(_finite(logprob) <= 0, "Positive log probability")
        require(type(version) is int and version == policy_version, "Stale or mixed token policy version")
    requests = record["provider_request_ids"]
    require(isinstance(requests, list) and requests and len(set(requests)) == len(requests),
            "Missing or duplicate provider request identities")
    for request in requests:
        _text(request)
    _sha(record["prompt_sha256"])
    require(record["origin"] == "policy", "Model-free or scripted sample refused")
    return len(ids)


def _trace_arrays(trace, member):
    """Reconcile actual native call nodes, including compact sampled-token streams."""
    streams = {key: [] for key in ("token_ids", "logprobs", "token_advantages", "weight_versions")}
    seen = set()
    for call in trace["calls"]:
        index = _integer(call["node"])
        require(index < len(trace["nodes"]) and index not in seen and call["error"] is None,
                "Invalid or duplicate native call node")
        seen.add(index)
        node = trace["nodes"][index]
        ids, mask = node["token_ids"], node["mask"]
        require(node["sampled"] is True and len(ids) == len(mask) and all(type(v) is bool for v in mask),
                "Malformed native sampled mask")
        sampled = [token for token, flag in zip(ids, mask) if flag]
        require(len(node["logprobs"]) == len(node["advantages"]) == len(sampled),
                "Native compact sampled streams do not align")
        versions, previous = [], 0
        for span in node["weight_versions"]:
            require(_integer(span["start"]) == previous and previous < _integer(span["end"]) <= len(sampled)
                    and span["version"] == "0", "Native token version coverage differs")
            versions.extend([0] * (span["end"] - previous))
            previous = span["end"]
        require(previous == len(sampled), "Native token versions are incomplete")
        for key, values in (("token_ids", sampled), ("logprobs", node["logprobs"]),
                            ("token_advantages", node["advantages"]), ("weight_versions", versions)):
            streams[key].extend(values)
    require(all(not any(node["mask"]) or index in seen for index, node in enumerate(trace["nodes"])),
            "Native sampled node has no matching model call")
    for key, values in streams.items():
        require(digest(values) == digest(member[key]), "Native trace arrays differ from cohort: " + key)


def inspect_bundle(bundle_path, expected_plan_path):
    """Return a consistency report; expected plan must be retained independently before launch."""
    result = {"schema": SCHEMA, "status": "rejected", "errors": [],
              "gpu_execution_independently_verified": False, "training_readiness_claim": False}
    try:
        _inspect(Path(bundle_path), Path(expected_plan_path), result)
    except (EvidenceError, KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
        result["errors"] = [str(exc) or type(exc).__name__]
    return result


def _inspect(bundle_path, expected_plan_path, result):
    root = bundle_path.parent
    require(not root.is_symlink() and not bundle_path.is_symlink(), "Symlink bundle refused")
    bundle, expected = read_json(bundle_path), read_json(expected_plan_path)
    require(bundle["schema"] == SCHEMA and expected["schema"] == SCHEMA, "Unsupported schema")
    require(bundle["binding"] == expected, "Run plan differs from independent expected plan")
    kind = bundle["kind"]
    require(kind in {"observed", "fixture"}, "Unknown evidence kind")
    _text(expected["run_id"])
    _text(expected["model_id"])
    _text(expected["model_revision"])
    _text(expected["model_path"])
    require(expected["update_scope"] in {"lora", "full"}, "Unknown optimizer update scope")
    _sha(expected["initial_weights_sha256"])
    for key in ("config_sha256", "sources_sha256", "collector_sha256", "task_sha256"):
        _sha(expected[key])
    require(expected["prime_rl_commit"] == PRIME_RL_COMMIT, "Wrong native trainer commit")
    require(type(expected["world_size"]) is int and expected["world_size"] == 1,
            "v1 requires one trainer rank")
    require(type(expected["step"]) is int and expected["step"] == 1, "v1 requires first optimizer step")
    artifacts = _inventory(bundle["artifacts"])
    for name, checksum in artifacts.items():
        with _path(root, name).open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum,
                    "Artifact digest differs: " + name)

    def artifact(name):
        require(name in artifacts, "Unbound artifact: " + str(name))
        return read_json(_path(root, name))

    for key in ("config", "sources", "collector"):
        require(artifacts.get(bundle[key]) == expected[key + "_sha256"], "Wrong " + key + " binding")
    sources = artifact(bundle["sources"])
    for name, checksum in PINNED_SOURCES.items():
        require(sources["native_sources"].get(name) == checksum, "Unsupported source binding: " + name)
    config = artifact(bundle["config"])
    orch, trainer = config["orchestrator"], config["trainer"]
    sources = orch["train"]["source"]
    algos = [orch["train"]["algo"]] + [source["algo"] for source in sources]
    require(len(sources) == 1, "v1 requires one environment")
    for algo in algos:
        require(algo["type"] == "grpo" and algo["length_penalty"] is None
                and algo["length_weighted_baseline"] is False and algo["loss_aggregation"] == "token",
                "v1 requires unshaped, unweighted, token-aggregated native GRPO")
    require(trainer["data"]["fake"] is None and trainer["resume"] is None
            and trainer["model"]["init_from"] is None, "Fake data or resume is outside this gate")
    require(type(trainer["max_steps"]) is int and trainer["max_steps"] == 1,
            "Configuration does not cap optimizer at one step")
    require(_finite(trainer["optim"]["max_norm"]) > 0 and trainer["ckpt"]["skip_optimizer"] is False,
            "Gradient norm and complete optimizer checkpoint are required")
    require(trainer["model"]["name"] == expected["model_path"], "Wrong configured model snapshot path")
    require((trainer["model"]["lora"] is not None) == (expected["update_scope"] == "lora"),
            "Configured optimizer update scope differs from plan")

    require(set(bundle["records"]) == RECORDS, "Missing or extra stage record")
    names = list(bundle["records"].values())
    require(len(set(names)) == len(names), "Reused stage record")
    records = {stage: artifact(name) for stage, name in bundle["records"].items()}
    for stage, row in records.items():
        require(row["stage"] == stage and row["run_id"] == expected["run_id"]
                and row["binding_sha256"] == digest(expected) and row["kind"] == kind,
                "Stale, fixture, or wrong-run stage: " + stage)
        _integer(row["monotonic_ns"], 1)
    times = {name: row["monotonic_ns"] for name, row in records.items()}
    require(len(set(times.values())) == len(times), "Duplicate stage timestamp")
    for earlier, later in (("runtime", "cohort"), ("cohort", "before"), ("before", "after"),
                           ("after", "checkpoint"), ("checkpoint", "reload"),
                           ("receiver", "inference")):
        require(times[earlier] < times[later], "Impossible stage ordering")
    optimizer_time = _integer(records["after"]["optimizer_step_monotonic_ns"], 1)
    require(times["before"] < optimizer_time <= times["after"] and optimizer_time < times["receiver"],
            "Receiver apply does not follow the actual optimizer update")
    runtime = records["runtime"]
    require(runtime["backend"] == "cuda" and _integer(runtime["world_size"], 1) == 1,
            "Missing CUDA single-rank runtime observation")
    for name in ("device_uuid", "cuda_version", "torch_version", "trainer_process_id"):
        _text(runtime[name])
    require(runtime["model_id"] == expected["model_id"]
            and runtime["model_revision"] == expected["model_revision"]
            and runtime["model_path"] == expected["model_path"]
            and runtime["initial_weights_sha256"] == expected["initial_weights_sha256"], "Wrong loaded model")

    cohort = records["cohort"]
    _text(cohort["cohort_id"])
    require(cohort["task_sha256"] == expected["task_sha256"], "Wrong cohort task")
    members = cohort["members"]
    require(isinstance(members, list) and len(members) >= 2, "Need policy siblings for one task")
    trace_ids, request_ids, rewards = [], [], []
    token_count = 0
    for member in members:
        trace_ids.append(_text(member["trace_id"]))
        require(member["task_sha256"] == expected["task_sha256"]
                and member["admission"] == "admitted" and member["outcome"] == "resolved",
                "Wrong-task or unresolved cohort member")
        require(artifacts.get(member["native_trace"]) == _sha(member["native_trace_sha256"])
                and artifacts.get(member["admission_journal"]) == _sha(member["admission_journal_sha256"]),
                "Missing raw trace/admission evidence")
        trace = artifact(member["native_trace"])
        require(trace["id"] == member["trace_id"] and trace["ok"] is True
                and trace["is_completed"] is True and trace["is_timeout"] is False
                and trace["dropped"] is None and digest(trace["task"]["data"]) == expected["task_sha256"],
                "Raw native trace does not match the resolved same-task sibling")
        admission = trace["info"]["selection_training"]
        require(admission["admission"]["admitted"] is True and not admission.get("model_free_reference", False),
                "Raw native trace was not admitted policy data")
        journal = artifact(member["admission_journal"])
        events = [row for row in journal if row["trace_id"] == member["trace_id"]]
        terminals = [row for row in events if row["event"] == "attempt_terminal"]
        scores = [row for row in events if row["event"] == "scoring_complete"]
        starts = [row for row in events if row["event"] == "attempt_started"]
        require(len(starts) == len(terminals) == len(scores) == 1
                and starts[0]["task_sha256"] == expected["task_sha256"]
                and terminals[0]["admitted"] is True and scores[0]["reward"] == member["reward"],
                "Raw worker admission/scoring journal does not reconcile")
        require(trace["rewards"] and all(value is not None for value in trace["rewards"].values())
                and math.isclose(math.fsum(_finite(value["value"]) for value in trace["rewards"].values()),
                                 _finite(member["reward"]), rel_tol=1e-12, abs_tol=1e-12),
                "Native financial reward differs from cohort reward")
        token_count += _sample(member, 0)
        _trace_arrays(trace, member)
        responses = [artifact(name) for name in member["provider_responses"]]
        require(responses and all(response["trace_id"] == member["trace_id"] for response in responses)
                and [response["provider_request_id"] for response in responses] == member["provider_request_ids"]
                and responses[0]["prompt_sha256"] == member["prompt_sha256"],
                "Raw provider responses do not join the native sibling")
        for field in ("token_ids", "logprobs", "weight_versions"):
            combined = [value for response in responses for value in response[field]]
            require(digest(combined) == digest(member[field]), "Raw provider arrays differ from cohort: " + field)
        for response in responses:
            _text(response["model"])
            _sha(response["prompt_sha256"])
            require(_integer(response["monotonic_ns"], 1) < times["cohort"],
                    "Provider response does not precede its cohort")
        request_ids.extend(member["provider_request_ids"])
        reward = _finite(member["reward"])
        require(0 <= reward <= 1, "Invalid selection reward")
        rewards.append(_f32(reward))
    require(len(set(trace_ids)) == len(trace_ids) and len(set(request_ids)) == len(request_ids),
            "Duplicate sibling trace or provider request")
    require(len(set(rewards)) > 1, "No varying genuine financial rewards")
    baseline = _f32(math.fsum(rewards) / len(rewards))
    for member, reward in zip(members, rewards):
        advantage = _finite(member["advantage"])
        require(math.isclose(advantage, _f32(reward - baseline), rel_tol=1e-6, abs_tol=1e-7),
                "Advantage differs from pinned native GRPO reward-minus-mean")
        token_advantages = member["token_advantages"]
        require(len(token_advantages) == len(member["token_ids"])
                and all(math.isclose(_finite(value), advantage, rel_tol=1e-6, abs_tol=1e-7)
                        for value in token_advantages), "Advantage transport differs from cohort")
    require(any(abs(member["advantage"]) > 1e-7 for member in members), "All advantages are zero")

    before, after = records["before"], records["after"]
    pre_state, pre_steps, pre_progress = _state(before)
    post_state, post_steps, post_progress = _state(after)
    for row in (before, after):
        require(row["cohort_id"] == cohort["cohort_id"] and _integer(row["step"], 1) == 1
                and row["trace_ids"] == trace_ids and _integer(row["batch_sampled_tokens"], 1) == token_count,
                "Optimizer batch does not join the observed cohort")
        _sha(row["batch_sha256"])
    require(before["batch_sha256"] == after["batch_sha256"], "Optimizer batch changed")
    batch = artifact("raw/trainer-batch.json")
    require(isinstance(batch, list) and len(batch) == len(members)
            and digest(batch) == before["batch_sha256"], "Raw trainer batch differs from optimizer observation")
    batch_by_trace = {row["trace_id"]: row for row in batch}
    require(len(batch_by_trace) == len(batch) and set(batch_by_trace) == set(trace_ids),
            "Raw trainer batch does not uniquely join the cohort")
    for member in members:
        row = batch_by_trace[member["trace_id"]]
        for field in ("token_ids", "logprobs", "token_advantages"):
            require(digest(row[field]) == digest(member[field]),
                    "Raw trainer batch stream differs from cohort: " + field)
    micro_batches = _integer(after["batch_micro_batches"], 1)
    require(_integer(before["batch_micro_batches"], 1) == micro_batches, "Micro-batch count changed")
    require(set(pre_steps) == set(post_steps) and all(pre_steps[key] == 0 and post_steps[key] == 1
            for key in pre_steps), "Optimizer counters did not advance exactly once")
    require(set(pre_state["model"]) == set(post_state["model"])
            and pre_state["model"] != post_state["model"] and post_state["optimizers"]
            and pre_state["optimizers"] != post_state["optimizers"], "No model/optimizer state delta")
    gradient = after["gradient"]
    _finite(after["loss"])
    require(_finite(after["parameter_delta_l2"]) > 0, "Missing finite nonzero parameter delta")
    require(_finite(gradient["l2_norm"]) > 0 and _integer(gradient["total_elements"], 1)
            == _integer(gradient["finite_elements"], 1), "Missing, zero, or non-finite gradients")
    require(pre_progress["step"] == post_progress["step"] == 1
            and post_progress["total_tokens"] > pre_progress["total_tokens"]
            and post_progress["total_samples"] - pre_progress["total_samples"] == micro_batches,
            "Native Progress does not reflect the completed batch")

    checkpoint = records["checkpoint"]
    require(_integer(checkpoint["step"], 1) == 1 and checkpoint["state"] == post_state,
            "Checkpoint is not post-update state")
    inventory = _inventory(checkpoint["files"])
    prefix = "checkpoints/step_1/"
    require(all(name.startswith(prefix) for name in inventory), "Wrong checkpoint step directory")
    required = {prefix + "trainer/.metadata", prefix + "orchestrator/progress.pt"}
    require(required <= set(inventory) and any(name.endswith(".distcp") for name in inventory),
            "Missing native DCP or orchestrator checkpoint files")
    actual = {str(path.relative_to(root)) for path in (root / prefix).rglob("*") if path.is_file()}
    require(actual == set(inventory), "Checkpoint inventory does not cover every file")
    require(all(artifacts.get(name) == checksum for name, checksum in inventory.items()),
            "Checkpoint file not bound to artifact bytes")
    reload = records["reload"]
    _state(reload)
    require(reload["process_id"] != runtime["trainer_process_id"] and _text(reload["process_id"]),
            "Reload was not a fresh process")
    require(reload["method"] == "CheckpointManager.load_from_path"
            and reload["checkpoint_files"] == inventory and _integer(reload["optimizer_updates"]) == 0
            and reload["state"] == post_state and reload["optimizer_steps"] == post_steps
            and reload["progress"] == post_progress, "Fresh load-only full-state equality is unproven")
    initialization = reload["optimizer_initialization"]
    calls = _integer(reload["optimizer_initialization_calls"])
    require(isinstance(initialization, list) and calls == len(initialization) and calls <= 1,
            "Unexpected load-only optimizer initialization calls")
    for witness in initialization:
        rates = witness["learning_rates"]
        require(witness["method"] == "torch.distributed.checkpoint.state_dict._init_optim_state"
                and witness["optimizer_state_empty_before"] is True and isinstance(rates, list) and rates
                and all(_finite(rate) == 0.0 for rate in rates), "Unsafe native optimizer state initialization")
        require(_integer(witness["gradient_total_elements"], 1)
                == _integer(witness["gradient_zero_finite_elements"], 1),
                "Initialization gradients were not all finite zero")
        initial_parameters = _inventory(witness["model_parameters_before"])
        require(initial_parameters == _inventory(witness["model_parameters_after"])
                and set(initial_parameters) <= set(post_state["model"]),
                "Native optimizer initialization changed model parameter bytes")
        if expected["update_scope"] == "lora":
            require(set(after["trainable_state"]) <= set(initial_parameters),
                    "Initialization witness omitted trainable parameters")
        require(times["checkpoint"] < _integer(witness["started_monotonic_ns"], 1)
                < _integer(witness["finished_monotonic_ns"], 1) < times["reload"],
                "Initialization was not observed inside the fresh load-only process")
    require(reload["orchestrator_state_sha256"] == checkpoint["orchestrator_state_sha256"],
            "Orchestrator state did not survive reload")
    _sha(reload["orchestrator_state_sha256"])
    receiver, inference = records["receiver"], records["inference"]
    require(receiver["hook"] == "WeightWatcher.on_update" and _integer(receiver["applied_version"], 1) == 1,
            "Receiver did not apply the updated model")
    if expected["update_scope"] == "lora":
        trainable = _inventory(after["trainable_state"])
        require(all(post_state["model"].get(name) == checksum for name, checksum in trainable.items())
                and any(pre_state["model"].get(name) != checksum for name, checksum in trainable.items()),
                "Trainable adapter inventory does not join changed native model state")
        published = _inventory(receiver["published_files"])
        require(receiver["transport"] == "filesystem_lora" and receiver["applied_files"] == published
                and receiver["published_state"] == trainable
                and all(artifacts.get(name) == checksum for name, checksum in published.items()),
                "Applied adapter files do not join the trainer's published adapter state")
        applied = _inventory(receiver["applied_copy_files"])
        published_prefix, applied_prefix = "published/step_1/", "applied/step_1/"
        require(all(name.startswith(published_prefix) for name in published)
                and all(name.startswith(applied_prefix) for name in applied)
                and all(artifacts.get(name) == checksum for name, checksum in applied.items())
                and {name.removeprefix(published_prefix): checksum for name, checksum in published.items()}
                == {name.removeprefix(applied_prefix): checksum for name, checksum in applied.items()},
                "Retained applied adapter bytes differ from the published adapter inventory")
    else:
        require(receiver["model_state_sha256"] == digest(post_state["model"]),
                "Receiver model-state observation differs from trainer")
    _text(receiver["receiver_id"])
    require(inference["receiver_id"] == receiver["receiver_id"]
            and inference["model_id"] == expected["model_id"], "Inference is not from the updated receiver")
    _sample(inference, 1)
    require(not set(inference["provider_request_ids"]) & set(request_ids), "Reused pre-update inference")
    paths = inference["provider_responses"]
    require(isinstance(paths, list) and len(paths) == 1, "Expected one raw post-update probe response")
    response = artifact(paths[0])
    require(response["trace_id"] == "post-update-probe"
            and [response["provider_request_id"]] == inference["provider_request_ids"]
            and response["prompt_sha256"] == inference["prompt_sha256"]
            and response["model"] in {expected["model_id"], expected["model_path"]},
            "Raw post-update response does not join the updated inference")
    require(times["receiver"] < _integer(response["monotonic_ns"], 1) <= times["inference"],
            "Raw post-update response was not observed after receiver application")
    for field in ("token_ids", "logprobs", "weight_versions"):
        require(digest(response[field]) == digest(inference[field]),
                "Raw post-update response differs from inference: " + field)
    result.update(status="fixture_consistent" if kind == "fixture" else "trusted_observations_consistent",
                  evidence_kind=kind, run_id=expected["run_id"], cohort_size=len(members),
                  optimizer_steps=1, update_scope=expected["update_scope"], errors=[])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect_bundle(args.bundle, args.expected_plan)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return {"trusted_observations_consistent": 0, "fixture_consistent": 3}.get(report["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
