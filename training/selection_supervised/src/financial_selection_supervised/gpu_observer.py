"""Scoped observations of the pinned, single-rank LoRA training path.

These records describe a trusted local process, not remote execution attestation.
No model, reward, advantage, optimizer result, or native source is replaced. An
unsupported or missing observation fails closed. All artifacts remain private.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
from functools import wraps
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

PLAN_ENV = "FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"
EVIDENCE_ENV = "FINANCIAL_SELECTION_OPTIMIZER_EVIDENCE_DIR"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular private artifact")
    return path


def _plain(value):
    """Convert observed metadata, refusing nonfinite and opaque Python objects."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite native observation")
        return value
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unsupported observed state type: " + type(value).__name__)


def flatten(value, prefix=""):
    """Hash every leaf with unambiguous JSON-pointer names, including empty containers."""
    if isinstance(value, dict) and value:
        result = {}
        for key, item in value.items():
            name = str(key).replace("~", "~0").replace("/", "~1")
            result.update(flatten(item, prefix + "/" + name if prefix else name))
        return result
    if isinstance(value, (list, tuple)) and value:
        result = {}
        for index, item in enumerate(value):
            result.update(flatten(item, prefix + "/" + str(index) if prefix else str(index)))
        return result
    return {prefix or "$": tensor_sha(value) if hasattr(value, "detach") else digest(_plain(value))}


def local_tensor(value):
    value = value.detach()
    if hasattr(value, "to_local"):
        # The contract is one trainer rank; a shard here is the complete tensor.
        value = value.to_local()
    return value.cpu().contiguous()


def tensor_sha(value):
    import torch
    value = local_tensor(value)
    metadata = {"dtype": str(value.dtype), "shape": list(value.shape)}
    checksum = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    # Byte view supports bfloat16 without a lossy float conversion.
    checksum.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def optimizer_steps(model, optimizer):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            step = optimizer.state.get(parameter, {}).get("step", 0)
            step = step.item() if hasattr(step, "item") else step
            require(type(step) in (int, float) and math.isfinite(step) and step >= 0 and int(step) == step,
                    "Invalid native optimizer step counter")
            result[name] = int(step)
    require(result, "No trainable parameters")
    return result


def snapshot(model, optimizer, scheduler, progress, *, initialized):
    """Use AppState after initialization; do not materialize Adam moments before step one."""
    from prime_rl.trainer.ckpt import AppState, get_model_state_dict
    if initialized:
        state = AppState(model, [optimizer], scheduler, progress).state_dict()
    else:
        # DCP get_state_dict can lazily initialize an optimizer. Observing the
        # initial state must not itself call optimizer.step or create moments.
        state = {"model": get_model_state_dict(model), "optimizers": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "progress": asdict(progress)}
    return {"state": {key: flatten(value) for key, value in state.items()},
            "optimizer_steps": optimizer_steps(model, optimizer), "progress": asdict(progress)}


def gradient_observation(model):
    import torch
    total = finite = 0
    squared = 0.0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        require(parameter.grad is not None, "Trainable parameter has no observed gradient")
        gradient = local_tensor(parameter.grad)
        total += gradient.numel()
        finite += int(torch.isfinite(gradient).sum().item())
        squared += float(gradient.double().square().sum().item())
    require(total and total == finite and math.isfinite(squared), "Missing or nonfinite gradients")
    return {"l2_norm": math.sqrt(squared), "total_elements": total, "finite_elements": finite}


class PatchScope:
    def __init__(self):
        self.patches = []

    def patch(self, owner, name, replacement):
        previous = getattr(owner, name)
        self.patches.append((owner, name, previous, replacement))
        setattr(owner, name, replacement)
        return previous

    def close(self):
        for owner, name, previous, replacement in reversed(self.patches):
            require(getattr(owner, name) is replacement, "Another owner replaced an active observer hook")
            setattr(owner, name, previous)
        self.patches.clear()


class Store:
    def __init__(self, plan_path, root):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve() == self.root,
                "Evidence root must be canonical and absolute")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        from .optimizer_evidence import read_json
        self.plan = read_json(_regular(plan_path))
        require(self.plan["world_size"] == 1 and self.plan["step"] == 1
                and self.plan["update_scope"] == "lora", "Only one-rank, first-step LoRA observation is supported")
        self.binding = digest(self.plan)
        require(file_sha(__file__) == self.plan["collector_sha256"], "Observer differs from prelaunch collector binding")
        for name, field in (("config.json", "config_sha256"), ("sources.json", "sources_sha256")):
            require(file_sha(_regular(self.root / name)) == self.plan[field], "Prelaunch artifact binding changed")

    def write(self, name, value, *, raw=False):
        path = self.root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Invalid private artifact name")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(path.parent.resolve().is_relative_to(self.root), "Artifact escaped evidence root")
        payload = value if raw else json.dumps(value, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return name

    def record(self, stage, **details):
        row = {"stage": stage, "run_id": self.plan["run_id"], "kind": "observed",
               "binding_sha256": self.binding, "monotonic_ns": time.monotonic_ns(), **details}
        self.write("records/" + stage + ".json", row)
        return row

    def read(self, name):
        return json.loads(_regular(self.root / name).read_text())


def model_manifest(directory, model_id, revision):
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve() == directory, "Model path must be canonical")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Materialize model snapshot files before observation")
        if path.is_file():
            inventory[str(path.relative_to(directory))] = file_sha(path)
    require("config.json" in inventory and any(name.endswith(".safetensors") for name in inventory),
            "Missing native model config or safetensors weights")
    return {"model_id": model_id, "model_revision": revision, "files": inventory}


def prepare_plan(config_path, model_id, model_revision, evidence_dir, run_id):
    """Bind actual local model bytes and exact resolved config before any execution."""
    from importlib.resources import files
    from prime_rl.configs.rl import RLConfig
    from financial_selection_training.task import FinancialSelectionTrainingTaskset
    from .contract import verify_sources
    from .optimizer_evidence import SCHEMA, PRIME_RL_COMMIT
    verify_sources()
    require(isinstance(run_id, str) and run_id.strip() and isinstance(model_id, str) and model_id.strip()
            and isinstance(model_revision, str) and model_revision.strip(), "Missing prelaunch model/run identity")
    config = RLConfig.model_validate_json(_regular(config_path).read_text())
    resolved = config.model_dump(mode="json")
    require(config.max_steps == 1 and len(config.orchestrator.train.source) == 1
            and config.trainer.model.lora is not None, "Plan requires the reviewed one-step LoRA contract")
    source = config.orchestrator.train.source[0]
    tasks = list(FinancialSelectionTrainingTaskset(source.env.taskset).load())
    require(len(tasks) == 1, "Plan must bind exactly one native task")
    manifest = model_manifest(config.trainer.model.name, model_id, model_revision)
    root = Path(evidence_dir)
    require(root.is_absolute() and root.resolve() == root and not root.exists(), "Plan output must be a fresh canonical private directory")
    root.mkdir(mode=0o700, parents=True)
    payloads = {
        "config.json": resolved,
        "initial_model_manifest.json": manifest,
        "sources.json": {**json.loads(files("financial_selection_training").joinpath("source_bindings.json").read_text()),
                         "supervised": json.loads(files("financial_selection_supervised").joinpath("source_bindings.json").read_text())},
    }
    for name, payload in payloads.items():
        with (root / name).open("x") as stream:
            os.chmod(root / name, 0o600)
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    with (root / "collector.py").open("xb") as stream:
        os.chmod(root / "collector.py", 0o600)
        stream.write(Path(__file__).read_bytes())
    plan = {"schema": SCHEMA, "run_id": run_id, "model_id": model_id,
            "model_revision": model_revision, "model_path": config.trainer.model.name,
            "update_scope": "lora", "initial_weights_sha256": digest(manifest),
            "prime_rl_commit": PRIME_RL_COMMIT, "world_size": 1, "step": 1,
            "task_sha256": digest(tasks[0].data.model_dump(mode="json")),
            **{field: file_sha(root / name) for name, field in (
                ("config.json", "config_sha256"), ("sources.json", "sources_sha256"), ("collector.py", "collector_sha256"))}}
    with (root / "plan.json").open("x") as stream:
        os.chmod(root / "plan.json", 0o600)
        stream.write(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return root / "plan.json"


def _store(plan_path=None, evidence_dir=None):
    from .contract import verify_sources
    verify_sources()
    return Store(plan_path or os.environ[PLAN_ENV], evidence_dir or os.environ[EVIDENCE_ENV])


def response_observation(response, body, trace_id):
    tokens = response.tokens
    require(tokens is not None and response.id, "Native provider omitted token evidence or request ID")
    ids, logprobs = _plain(tokens.completion_ids), _plain(tokens.completion_logprobs)
    require(ids and len(ids) == len(logprobs), "Native token/logprob lengths differ")
    versions = []
    for span in tokens.weight_versions or []:
        require(span.start == len(versions) and span.end > span.start and str(span.version).isdigit(),
                "Malformed native policy version span")
        versions.extend([int(span.version)] * (span.end - span.start))
    require(len(versions) == len(ids), "Missing native policy version coverage")
    require(all(type(token) is int and token >= 0 for token in ids)
            and all(type(probability) in (int, float) and math.isfinite(probability) and probability <= 0
                    for probability in logprobs), "Invalid native sampled arrays")
    return {"trace_id": trace_id, "provider_request_id": response.id,
            "prompt_sha256": digest(body), "model": response.model,
            "token_ids": ids, "logprobs": logprobs, "weight_versions": versions,
            "monotonic_ns": time.monotonic_ns()}


def install_worker_observer(*, plan_path=None, evidence_dir=None):
    store, scope = _store(plan_path, evidence_dir), PatchScope()
    from financial_selection_training.journal import ObservedTrainClient
    original = ObservedTrainClient.get_response

    @wraps(original)
    async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
        response = await original(self, dialect, body, sampling, session_id, turn, headers)
        require(isinstance(session_id, str) and session_id, "Missing trusted trace identity")
        row = response_observation(response, body, session_id)
        store.write("responses/" + digest(session_id) + "/" + str(row["monotonic_ns"]) + ".json", row)
        return response
    scope.patch(ObservedTrainClient, "get_response", get_response)
    return scope


def _journal_rows(root, trace_id):
    rows = []
    for path in Path(root).glob("*/events.jsonl"):
        for line in _regular(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("trace_id") == trace_id:
                rows.append(row)
    require(rows, "Missing native admission journal")
    return rows


def _cohort(store, traces, journal_root):
    import numpy as np
    from prime_rl.orchestrator.algo.routing import trainable_nodes
    members = []
    for trace in traces:
        require(digest(trace.task.data.model_dump(mode="json")) == store.plan["task_sha256"],
                "Cohort changed task")
        nodes = trainable_nodes(trace)
        ids = [int(token) for node in nodes for token in np.asarray(node.token_ids)[np.asarray(node.mask)]]
        probabilities = [float(value) for node in nodes for value in node.logprobs]
        advantages = [float(value) for node in nodes for value in node.advantages]
        require(ids and len(ids) == len(probabilities) == len(advantages)
                and all(math.isfinite(value) for value in probabilities + advantages),
                "Native graph omits finite sampled arrays")
        paths = sorted((store.root / "responses" / digest(trace.id)).glob("*.json"),
                       key=lambda path: int(path.stem))
        responses = [json.loads(_regular(path).read_text()) for path in paths]
        require(responses and ids == [token for row in responses for token in row["token_ids"]]
                and np.array_equal(np.asarray(probabilities, dtype=np.float32),
                                   np.asarray([value for row in responses for value in row["logprobs"]], dtype=np.float32)),
                "Native graph differs from observed provider responses")
        # Preserve native JSON streams at full precision. Validate numeric arrays
        # above before the native serializer's nonfinite-to-null conversion.
        trace_path = store.write("raw/trace-" + digest(trace.id) + ".json", trace.to_json(float_decimals=None), raw=True)
        journal_path = store.write("raw/journal-" + digest(trace.id) + ".json", _journal_rows(journal_root, trace.id))
        members.append({"trace_id": trace.id, "task_sha256": store.plan["task_sha256"],
                        "admission": "admitted", "outcome": "resolved", "reward": float(trace.reward),
                        "advantage": advantages[0], "token_advantages": advantages,
                        "token_ids": ids, "logprobs": probabilities,
                        "weight_versions": [version for row in responses for version in row["weight_versions"]],
                        "provider_request_ids": [row["provider_request_id"] for row in responses],
                        "provider_responses": [str(path.relative_to(store.root)) for path in paths],
                        "prompt_sha256": responses[0]["prompt_sha256"], "origin": "policy",
                        "native_trace": trace_path, "native_trace_sha256": file_sha(store.root / trace_path),
                        "admission_journal": journal_path, "admission_journal_sha256": file_sha(store.root / journal_path)})
    row = {"stage": "cohort", "run_id": store.plan["run_id"], "kind": "observed",
           "binding_sha256": store.binding, "monotonic_ns": time.monotonic_ns(),
           "cohort_id": digest([member["trace_id"] for member in members]),
           "task_sha256": store.plan["task_sha256"], "members": members}
    store.write("cohorts/" + row["cohort_id"] + ".json", row)


def _copy_files(store, source, prefix, *, broadcast=False):
    result = {}
    for path in sorted(Path(source).rglob("*")):
        require(not path.is_symlink(), "Symlink in native output refused")
        if path.is_file():
            if broadcast and path.name in {".sender_ready", ".receiver_ready", ".started", ".finished"}:
                continue
            name = prefix + "/" + str(path.relative_to(source))
            destination = store.root / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Copy streaming: complete checkpoints can be larger than host RAM.
            with path.open("rb") as src, destination.open("xb") as dst:
                os.chmod(destination, 0o600)
                shutil.copyfileobj(src, dst)
            result[name] = file_sha(destination)
    require(result, "Empty native output inventory")
    return result


def _saved_adapter(directory):
    from safetensors.torch import load_file
    files = sorted(Path(directory).glob("*adapter*.safetensors"))
    require(len(files) == 1, "Expected one native safetensors LoRA adapter")
    return load_file(str(files[0]), device="cpu")


def canonical_adapter_inventory(adapter, full_inventory):
    result = {}
    for name, value in adapter.items():
        candidate = name.removeprefix("base_model.model.")
        if candidate.endswith((".lora_A.weight", ".lora_B.weight")):
            candidate = candidate.removesuffix(".weight")
        require(candidate in full_inventory and candidate not in result,
                "Unsupported adapter-to-native parameter mapping")
        checksum = tensor_sha(value)
        require(checksum == full_inventory[candidate], "Saved adapter differs from native post-update parameter")
        result[candidate] = checksum
    return result


def install_orchestrator_observer(config, *, plan_path=None, evidence_dir=None):
    store, scope = _store(plan_path, evidence_dir), PatchScope()
    require(digest(config.model_dump(mode="json")) == digest(store.read("config.json")["orchestrator"]),
            "Orchestrator component differs from prelaunch config")
    from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
    from prime_rl.orchestrator.algo.base import iter_trainable_traces
    from prime_rl.orchestrator.watcher import WeightWatcher
    from prime_rl.orchestrator.ckpt import CheckpointManager
    original_score = GRPOAlgorithm.score_group
    clients = []
    journal_root = config.train.source[0].env.journal_root

    @wraps(original_score)
    async def score_group(self, episodes):
        await original_score(self, episodes)
        if not clients:
            clients.append(self.clients)
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if traces:
            _cohort(store, traces, journal_root)
    scope.patch(GRPOAlgorithm, "score_group", score_group)
    original_init = WeightWatcher.__init__

    @wraps(original_init)
    def watcher_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        async def observe(step):
            if step != 1:
                return
            require(clients, "Post-update inference client was not observed during native scoring")
            require(self.policy.version == 1, "Watcher callback preceded policy application")
            publication = store.read("publication.json")
            files = _copy_files(store, self.receiver.step_dir(step), "applied/step_1", broadcast=True)
            # Same bytes, different evidence filenames. Reject mismatched adapters.
            by_leaf = lambda values: {name.split("/", 2)[2]: sha for name, sha in values.items()}
            require(by_leaf(files) == by_leaf(publication["published_files"]), "Applied files differ from trainer publication")
            receiver_id = digest({"pid": os.getpid(), "type": type(self.receiver).__name__, "model": self.receiver.model_name})
            store.record("receiver", hook="WeightWatcher.on_update", applied_version=step,
                         transport="filesystem_lora", receiver_id=receiver_id,
                         published_files=publication["published_files"], applied_files=publication["published_files"],
                         applied_copy_files=files, published_state=publication["published_state"],
                         engine_tensor_equality_observed=False)
            from verifiers.v1.clients import resolve_client
            from verifiers.v1.dialects import ChatDialect
            from verifiers.v1.configs.sampling import SamplingConfig
            client = resolve_client(clients[0].train_client)
            body = {"model": clients[0].model_name,
                    "messages": [{"role": "user", "content": "Reply with the single word ready."}]}
            try:
                response = await asyncio.wait_for(client.get_response(
                    ChatDialect(), body, SamplingConfig(max_tokens=8, temperature=0, top_p=1),
                    session_id="observer-" + uuid4().hex), timeout=60)
            finally:
                await client.close()
            raw = response_observation(response, body, "post-update-probe")
            path = store.write("raw/post-update-response.json", raw)
            store.record("inference", token_ids=raw["token_ids"], logprobs=raw["logprobs"],
                         weight_versions=raw["weight_versions"], provider_request_ids=[raw["provider_request_id"]],
                         provider_responses=[path], prompt_sha256=raw["prompt_sha256"], origin="policy",
                         receiver_id=receiver_id, model_id=store.plan["model_id"])
        self.on_update(observe)
    scope.patch(WeightWatcher, "__init__", watcher_init)
    original_save = CheckpointManager.save_state

    @wraps(original_save)
    def save_state(self, state, step):
        result = original_save(self, state, step)
        require(step == 1, "Unexpected orchestrator checkpoint step")
        store.write("orchestrator-state.json", {"state_sha256": digest(flatten(state)), "step": step})
        return result
    scope.patch(CheckpointManager, "save_state", save_state)
    return scope


def _batch_observation(store, micro_batches):
    sequences = []
    for batch in micro_batches:
        require(batch["trace_ids"] is not None, "Synthetic or unidentified trainer batch")
        ids = local_tensor(batch["input_ids"]).reshape(-1).tolist()
        mask = local_tensor(batch["loss_mask"]).reshape(-1).tolist()
        probabilities = local_tensor(batch["inference_logprobs"]).reshape(-1).tolist()
        advantages = local_tensor(batch["advantages"]).reshape(-1).tolist()
        offset = 0
        for trace_id, length in zip(batch["trace_ids"], batch["sequence_lengths"], strict=True):
            positions = [index for index in range(offset, offset + length) if mask[index]]
            if positions:
                require(trace_id, "Sampled dummy batch cannot be observed")
                sequences.append({"trace_id": trace_id, "token_ids": [ids[i] for i in positions],
                                  "logprobs": [probabilities[i] for i in positions],
                                  "token_advantages": [advantages[i] for i in positions]})
            offset += length
    trace_ids = [row["trace_id"] for row in sequences]
    require(len(trace_ids) == len(set(trace_ids)), "v1 observation requires one branch per trace")
    candidates = [json.loads(path.read_text()) for path in (store.root / "cohorts").glob("*.json")]
    matches = [row for row in candidates if set(trace_ids) == {member["trace_id"] for member in row["members"]}]
    require(len(matches) == 1, "Trainer batch does not uniquely join one observed cohort")
    cohort = matches[0]
    by_id = {row["trace_id"]: row for row in sequences}
    for member in cohort["members"]:
        for field in ("token_ids", "logprobs", "token_advantages"):
            require(by_id[member["trace_id"]][field] == member[field], "Trainer token stream differs from native cohort: " + field)
    store.write("records/cohort.json", cohort)
    store.write("raw/trainer-batch.json", sequences)
    return {"cohort_id": cohort["cohort_id"], "trace_ids": [member["trace_id"] for member in cohort["members"]],
            "step": 1, "batch_sampled_tokens": sum(len(row["token_ids"]) for row in sequences),
            "batch_micro_batches": len(micro_batches), "batch_sha256": digest(sequences)}


def install_trainer_observer(config, *, plan_path=None, evidence_dir=None):
    store, scope = _store(plan_path, evidence_dir), PatchScope()
    require(digest(config.model_dump(mode="json")) == digest(store.read("config.json")["trainer"]),
            "Trainer component differs from prelaunch config")
    import torch
    train = importlib.import_module("prime_rl.trainer.rl.train")
    from prime_rl.trainer.ckpt import CheckpointManager
    from prime_rl.transports.weights.filesystem import FileSystemWeightSender
    require(config.max_steps == 1 and config.resume is None and config.model.init_from is None
            and config.data.fake is None and config.model.lora is not None and config.model.offload == "none"
            and config.optim.type == "adamw" and config.ckpt is not None and not config.ckpt.skip_optimizer,
            "Unsupported observer trainer configuration")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Observer requires one trainer rank")
    require(str(config.model.name) == store.plan["model_path"], "Trainer model path differs from plan")
    observed_manifest = model_manifest(config.model.name, store.plan["model_id"], store.plan["model_revision"])
    require(digest(observed_manifest) == store.plan["initial_weights_sha256"]
            and observed_manifest == store.read("initial_model_manifest.json"), "Initial model snapshot changed before load")
    state = {"losses": [], "optimizer_updates": 0}
    original_model, original_optimizer = train.setup_model, train.setup_optimizer
    original_scheduler, original_progress = train.setup_scheduler, train.Progress

    def setup_model(*args, **kwargs):
        model = original_model(*args, **kwargs)
        state["model"] = model
        require(str(torch.__version__).split("+")[0] == "2.14.0", "Observer requires pinned Torch 2.14.0")
        require(torch.cuda.is_available(), "No CUDA execution observed")
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = getattr(device, "uuid", None)
        require(uuid, "CUDA device UUID unavailable")
        store.record("runtime", backend="cuda", world_size=torch.distributed.get_world_size(),
                     device_uuid=str(uuid), cuda_version=str(torch.version.cuda), torch_version=str(torch.__version__),
                     trainer_process_id=str(os.getpid()), **{key: store.plan[key] for key in
                     ("model_id", "model_path", "model_revision", "initial_weights_sha256")})
        return model
    scope.patch(train, "setup_model", setup_model)

    def setup_optimizer(*args, **kwargs):
        optimizer, manager = original_optimizer(*args, **kwargs)
        require(manager is None, "Gradient offload is outside observer scope")
        state["optimizer"] = optimizer
        original_step = optimizer.step

        @wraps(original_step)
        def step(*step_args, **step_kwargs):
            require(state["optimizer_updates"] == 0 and "batch" in state, "Unexpected or unbound optimizer step")
            model = state["model"]
            before_weights = {name: local_tensor(value).clone() for name, value in model.named_parameters() if value.requires_grad}
            store.record("before", **snapshot(model, optimizer, state["scheduler"], state["progress"], initialized=False), **state["batch"])
            state["gradient"] = gradient_observation(model)
            result = state["scheduled_step"](*step_args, **step_kwargs)
            torch.cuda.synchronize()
            state["optimizer_step_monotonic_ns"] = time.monotonic_ns()
            state["optimizer_updates"] += 1
            squared = sum(float((local_tensor(value).double() - before_weights[name].double()).square().sum().item())
                          for name, value in model.named_parameters() if value.requires_grad)
            require(math.isfinite(squared), "Nonfinite trainable parameter delta")
            state["parameter_delta_l2"] = math.sqrt(squared)
            return result
        state["step_observer"] = step
        return optimizer, manager
    scope.patch(train, "setup_optimizer", setup_optimizer)

    def setup_scheduler(*args, **kwargs):
        state["scheduler"] = original_scheduler(*args, **kwargs)
        # Native PyTorch LRScheduler wraps the bound optimizer method itself.
        # Install our observation afterward and delegate through that wrapper.
        state["scheduled_step"] = state["optimizer"].step
        scope.patch(state["optimizer"], "step", wraps(state["scheduled_step"])(state["step_observer"]))
        return state["scheduler"]
    scope.patch(train, "setup_scheduler", setup_scheduler)

    def progress(*args, **kwargs):
        state["progress"] = original_progress(*args, **kwargs)
        return state["progress"]
    scope.patch(train, "Progress", progress)
    original_batch = train.DataLoader.get_batch

    def get_batch(self, *args, **kwargs):
        result = original_batch(self, *args, **kwargs)
        require("batch" not in state, "Observer allows one optimizer batch")
        state["batch"] = _batch_observation(store, result)
        return result
    scope.patch(train.DataLoader, "get_batch", get_batch)
    original_loss = train.compute_loss

    def compute_loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        value = float(result[0].detach().item())
        require(math.isfinite(value), "Nonfinite native training loss")
        state["losses"].append(value)
        return result
    scope.patch(train, "compute_loss", compute_loss)
    original_broadcast = FileSystemWeightSender._broadcast

    def broadcast(self, model, step, step_dir):
        result = original_broadcast(self, model, step, step_dir)
        if step == 1:
            from prime_rl.trainer.ckpt import get_model_state_dict
            full = flatten(get_model_state_dict(model))
            published = canonical_adapter_inventory(_saved_adapter(step_dir), full)
            store.write("publication.json", {"published_state": published,
                        "published_files": _copy_files(store, step_dir, "published/step_1", broadcast=True)})
        return result
    scope.patch(FileSystemWeightSender, "_broadcast", broadcast)
    original_save = CheckpointManager.save_to_path

    def save_to_path(self, path, model, optimizers, scheduler, progress, dataloader=None):
        require(len(optimizers) == 1 and state["optimizer_updates"] == 1, "Checkpoint did not follow exactly one optimizer update")
        after = snapshot(model, optimizers[0], scheduler, progress, initialized=True)
        trainable = {name: after["state"]["model"][name] for name, value in model.named_parameters() if value.requires_grad}
        require(state["losses"], "No observed native loss")
        store.record("after", **after, **state["batch"], gradient=state["gradient"],
                     parameter_delta_l2=state["parameter_delta_l2"], loss=math.fsum(state["losses"]),
                     optimizer_step_monotonic_ns=state["optimizer_step_monotonic_ns"], trainable_state=trainable)
        result = original_save(self, path, model, optimizers, scheduler, progress, dataloader)
        # Orchestrator checkpoint may complete later. Finalization copies both.
        store.write("trainer-checkpoint-complete.json", {"path": str(Path(path).parent),
                    "state": after["state"], "monotonic_ns": time.monotonic_ns()})
        return result
    scope.patch(CheckpointManager, "save_to_path", save_to_path)
    return scope


def finalize_checkpoint(store, checkpoint):
    checkpoint = Path(checkpoint)
    require(checkpoint.name == "step_1", "Only first-step checkpoint is supported")
    require((checkpoint / "trainer/.metadata").is_file() and (checkpoint / "orchestrator/progress.pt").is_file(),
            "Incomplete native trainer/orchestrator checkpoint")
    complete = store.read("trainer-checkpoint-complete.json")
    require(Path(complete["path"]) == checkpoint, "Wrong native checkpoint path")
    files = _copy_files(store, checkpoint, "checkpoints/step_1")
    state = store.read("orchestrator-state.json")
    return store.record("checkpoint", step=1, state=complete["state"], files=files,
                        orchestrator_state_sha256=state["state_sha256"])


def guard_native_optimizer_initialization(model, optimizer):
    """Permit only DCP's one zero-gradient, zero-lr Adam state allocation call.

    Native Torch 2.14 invokes step to allocate empty Adam moments before load.
    That invocation is recorded separately from a learning update. Every model
    parameter must remain byte-identical, and any further step is forbidden.
    """
    import torch
    original_step = optimizer.step
    witnesses = []

    def step(*args, **kwargs):
        require(not witnesses and not optimizer.state, "Load-only optimizer was already initialized")
        require((not args or args == (None,)) and set(kwargs) <= {"closure"}
                and kwargs.get("closure") is None, "Load-only initialization cannot execute a closure")
        rates = [group["lr"] for group in optimizer.param_groups]
        require(rates and all(type(rate) in (int, float) and rate == 0.0 for rate in rates),
                "Native load initialization requires every learning rate to be exactly zero")
        named = list(model.named_parameters())
        trainable = {id(parameter) for _, parameter in named if parameter.requires_grad}
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        require(trainable and optimized == trainable, "Load-only optimizer parameter coverage differs")
        total = zero_finite = 0
        for _, parameter in named:
            if not parameter.requires_grad:
                continue
            require(parameter.grad is not None, "Native load initialization requires zero gradients")
            gradient = local_tensor(parameter.grad)
            total += gradient.numel()
            if bool(torch.isfinite(gradient).all().item()) and int(torch.count_nonzero(gradient).item()) == 0:
                zero_finite += gradient.numel()
        require(total > 0 and total == zero_finite, "Native load initialization has nonzero or nonfinite gradients")
        before = {name: tensor_sha(parameter) for name, parameter in named}
        started = time.monotonic_ns()
        result = original_step(*args, **kwargs)
        after = {name: tensor_sha(parameter) for name, parameter in named}
        require(before == after, "Native load initialization changed model parameter bytes")
        witnesses.append({"method": "torch.distributed.checkpoint.state_dict._init_optim_state",
                          "learning_rates": rates, "optimizer_state_empty_before": True,
                          "gradient_total_elements": total, "gradient_zero_finite_elements": zero_finite,
                          "model_parameters_before": before, "model_parameters_after": after,
                          "started_monotonic_ns": started, "finished_monotonic_ns": time.monotonic_ns()})
        return result
    optimizer.step = step
    return witnesses


def checkpoint_load(config_path, plan_path, evidence_dir, checkpoint):
    """Fresh native model/optimizer/scheduler construction and load; no trainer loop."""
    from datetime import timedelta
    import torch
    from prime_rl.configs.rl import RLConfig
    from prime_rl.trainer import ckpt
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.moe_runtime import setup_moe_comm_env
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.optim import setup_optimizer
    from prime_rl.trainer.scheduler import setup_scheduler
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.trainer.lora import get_lora_state
    from prime_rl.orchestrator.ckpt import CheckpointManager as OrchestratorCheckpointManager
    from prime_rl.configs.orchestrator import CheckpointConfig as OrchestratorCheckpointConfig
    store = _store(plan_path, evidence_dir)
    require(str(torch.__version__).split("+")[0] == "2.14.0", "Load-only requires pinned Torch 2.14.0")
    require(file_sha(_regular(config_path)) == store.plan["config_sha256"], "Load-only config differs from prelaunch binding")
    config = RLConfig.model_validate_json(_regular(config_path).read_text()).trainer
    require(config.model.lora is not None and config.model.offload == "none" and config.optim.type == "adamw"
            and config.ckpt is not None and not config.ckpt.skip_optimizer, "Unsupported native load-only configuration")
    before = store.read("records/runtime.json")
    require(str(os.getpid()) != before["trainer_process_id"], "Checkpoint load must use a fresh process")
    checkpoint_record = finalize_checkpoint(store, checkpoint)
    setup_moe_comm_env(config.model)
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=False)
    try:
        require(torch.distributed.get_world_size() == 1, "Load-only process requires one rank")
        torch.set_float32_matmul_precision(config.matmul_precision)
        resolve_ep(config.model)
        dims = get_parallel_dims(config.model)
        model = setup_model(config.model, dims, loading_from_checkpoint_later=False)
        optimizer, manager = setup_optimizer(config.optim, list(model.named_parameters()), dims,
                                              offload=config.model.offload, model=model)
        require(manager is None, "Unexpected load-only gradient manager")
        scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
        get_lora_state().reset_adapter_parameters()
        progress = ckpt.Progress()
        initialization = guard_native_optimizer_initialization(model, optimizer)
        manager = ckpt.CheckpointManager(config.output_dir, config.ckpt)
        manager.load_from_path(Path(checkpoint) / "trainer", model, [optimizer], scheduler, progress)
        actual = snapshot(model, optimizer, scheduler, progress, initialized=True)
        expected = store.read("records/after.json")
        require(all(actual[key] == expected[key] for key in ("state", "optimizer_steps", "progress")),
                "Fresh native checkpoint load differs from post-update AppState")
        orchestrator = OrchestratorCheckpointManager(config.output_dir, OrchestratorCheckpointConfig())
        orchestrator_state = orchestrator.read_state(1, Path(checkpoint) / "orchestrator")
        store.record("reload", **actual, process_id=str(os.getpid()), method="CheckpointManager.load_from_path",
                     optimizer_updates=0, optimizer_initialization_calls=len(initialization),
                     optimizer_initialization=initialization, checkpoint_files=checkpoint_record["files"],
                     orchestrator_state_sha256=digest(flatten(orchestrator_state)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    assemble_bundle(store)


def assemble_bundle(store):
    from .optimizer_evidence import RECORDS, SCHEMA
    names = {stage: "records/" + stage + ".json" for stage in RECORDS}
    require(all((store.root / path).is_file() for path in names.values()), "Some required observations are missing")
    for name, field in (("config.json", "config_sha256"), ("sources.json", "sources_sha256"), ("collector.py", "collector_sha256")):
        require(file_sha(_regular(store.root / name)) == store.plan[field], "Prelaunch artifact binding changed")
    artifacts = {}
    for path in sorted(store.root.rglob("*")):
        require(not path.is_symlink(), "Symlink evidence refused")
        if path.is_file() and path.name != "bundle.json":
            artifacts[str(path.relative_to(store.root))] = file_sha(path)
    store.write("bundle.json", {"schema": SCHEMA, "kind": "observed", "binding": store.plan,
                               "config": "config.json", "sources": "sources.json", "collector": "collector.py",
                               "records": names, "artifacts": artifacts})


def component_main(role, config_path):
    from prime_rl.utils.config import cli
    if role == "trainer":
        from prime_rl.configs.trainer import TrainerConfig
        from prime_rl.trainer.rl.train import train
        cls = TrainerConfig
    elif role == "orchestrator":
        from prime_rl.configs.orchestrator import OrchestratorConfig
        from prime_rl.orchestrator.orchestrator import run_orchestrator
        cls = OrchestratorConfig
    else:
        raise ValueError("Unknown observer role")
    sys.argv = [role, "@", str(config_path)]
    config = cli(cls)
    scope = (install_trainer_observer if role == "trainer" else install_orchestrator_observer)(config)
    try:
        return train(config) if role == "trainer" else asyncio.run(run_orchestrator(config))
    finally:
        scope.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "@":
        component_main("trainer", argv[1])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-plan")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    load = commands.add_parser("checkpoint-load")
    load.add_argument("--config", type=Path, required=True)
    load.add_argument("--plan", type=Path, required=True)
    load.add_argument("--evidence-dir", type=Path, required=True)
    load.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-plan":
        print(prepare_plan(args.config, args.model_id, args.model_revision, args.evidence_dir, args.run_id))
    else:
        checkpoint_load(args.config, args.plan, args.evidence_dir, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
