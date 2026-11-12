"""Trusted local numerical reference controls for the fixed development schedule.

Only the frozen standalone reference executes in a public-input container. The
evaluator key and generated private episode stay on the trusted host. These are
financial reference controls with public execution and structural checkpoint
checks, not a native agent/harness validation or a model evaluation.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4


DEVELOPMENT_SEEDS = tuple(range(128, 160))
PUBLIC_FILES = {"prices.csv", "features.csv", "securities.csv", "contract.json"}
METRICS = {"schema_valid", "audit_accuracy", "trial_accuracy", "decision_correct",
           "economic_quality"}
PUBLIC_METRICS = METRICS - {"economic_quality"}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
    path.chmod(0o600)


def _validate_seeds(seeds) -> tuple[int, ...]:
    seeds = tuple(seeds)
    if any(type(seed) is not int for seed in seeds) or seeds != DEVELOPMENT_SEEDS:
        raise ValueError("Reference logging requires development seeds 128 through 159, exactly once in order")
    return seeds


def _checked_score(score: dict) -> dict:
    if not isinstance(score, dict) or set(score) != {"reward", "metrics"}:
        raise ValueError("Unexpected reference score schema")
    metrics = score["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != METRICS:
        raise ValueError("Unexpected reference metric schema")
    for value in [score["reward"], *metrics.values()]:
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Reference scores must be finite numbers between zero and one")
    if any(metrics[name] != 1 for name in PUBLIC_METRICS):
        raise ValueError("Reference solution is not publicly perfect")
    expected = (0.25 * metrics["audit_accuracy"] + 0.35 * metrics["trial_accuracy"]
                + 0.2 * metrics["decision_correct"] + 0.2 * metrics["economic_quality"])
    if score["reward"] != expected:
        raise ValueError("Reference score differs from the frozen reward formula")
    return score


def _verify_source_hashes(repo_root: Path, hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("A nonempty frozen source hash map is required")
    for relative, expected in hashes.items():
        path = (repo_root / relative).resolve()
        if not path.is_relative_to(repo_root) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid frozen source binding")
        if _sha(path.read_bytes()) != expected:
            raise ValueError(f"Frozen source differs: {relative}")


def _dependencies(repo_root: Path):
    names = ("financial_portfolio.generator", "financial_research.verifier",
             "financial_portfolio_staged.validation", "financial_portfolio_clarified.prime_adapter")
    modules = tuple(importlib.import_module(name) for name in names)
    for name, module in zip(names, modules):
        package, filename = name.split(".")
        expected = repo_root / "environments" / package / "src" / package / f"{filename}.py"
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError(f"Imported module is not the bound local source: {name}")
    return modules


@contextmanager
def _evaluator_key(key: str):
    previous = os.environ.get("FINANCIAL_RESEARCH_KEY")
    os.environ["FINANCIAL_RESEARCH_KEY"] = key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FINANCIAL_RESEARCH_KEY", None)
        else:
            os.environ["FINANCIAL_RESEARCH_KEY"] = previous


def _container_reference(workspace: Path, image_id: str) -> dict:
    """Execute a frozen trusted source, with public files as its only mount."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("A Docker image ID pin is required")
    name = "portfolio-reference-" + uuid4().hex
    # No inherited credential environment is passed to the Docker client or
    # container. The mount contains only the four public inputs and reference.
    environment = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    command = ["docker", "run", "--name", name, "--rm", "--pull", "never",
               "--network", "none", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--memory", "256m",
               "--memory-swap", "256m", "--pids-limit", "32", "--cpus", "1",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
               "--mount", f"type=bind,src={workspace},dst=/workspace",
               "--workdir", "/workspace", image_id, "python", "-I", "-B", "-c",
               "import resource,runpy; "
               "resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456)); "
               "runpy.run_path('research.py',run_name='__main__')"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, env=environment, capture_output=True, timeout=30,
                                check=False)
        if result.returncode:
            raise RuntimeError("Standalone reference container failed")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Standalone reference container exceeded 30 seconds") from None
    finally:
        # A timed-out Docker client does not necessarily stop its container.
        # Successful --rm execution also makes this harmlessly idempotent.
        cleanup = subprocess.run(["docker", "rm", "--force", name], env=environment,
                                 capture_output=True, timeout=15, check=False)
        if cleanup.returncode and b"No such container" not in cleanup.stderr:
            raise RuntimeError("Unable to confirm reference container cleanup")
    return {"kind": "standalone_public_reference_container", "image_id": image_id,
            "network": "none", "wall_seconds": time.monotonic() - started,
            "wall_timeout_seconds": 30, "address_space_limit_bytes": 268435456,
            "returncode": result.returncode, "container_cleanup_confirmed": True,
            "native_agent_harness_gold_validation": False}


def generate_reference_scores(
    output_dir: Path, *, repo_root: Path, protocol_manifest: Path,
    baseline_preflight: Path, key_file: Path, seeds=DEVELOPMENT_SEEDS,
) -> dict:
    """Persist all 32 dev reference scores, refusing overwrite or changed inputs.

    Run serially on a trusted host. Output must be a new directory outside the
    repository. An interrupted run remains incomplete and must not be resumed or
    overwritten. No model API is contacted and no reserved-test episode is built.
    ``references.json`` exists only after every control passes.
    """
    seeds = _validate_seeds(seeds)
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    protocol_manifest, baseline_preflight = Path(protocol_manifest), Path(baseline_preflight)
    if output_dir.is_relative_to(repo_root):
        raise ValueError("Reference artifacts must remain outside the repository")
    if output_dir.exists():
        raise FileExistsError("Reference output directory already exists")
    protocol = _load_json(protocol_manifest)
    baseline = _load_json(baseline_preflight)
    _verify_source_hashes(repo_root, protocol["sha256"])
    _verify_source_hashes(repo_root, baseline["source_files_sha256"])
    key = Path(key_file).read_text(encoding="utf-8").strip()
    if len(key) < 32 or _sha(key.encode()) != baseline["evaluator_key_sha256"]:
        raise ValueError("Evaluator key fingerprint differs from the frozen development batch")
    schedule = baseline["episode_schedule"]
    if len(schedule) != 32 or tuple(row["seed"] for row in schedule) != seeds:
        raise ValueError("Baseline must bind the complete ordered development schedule")
    if len({row["task_id"] for row in schedule}) != 32:
        raise ValueError("Duplicate task IDs in baseline schedule")
    generator, verifier, validation, clarified = _dependencies(repo_root)
    source_path = repo_root / "environments/financial_portfolio/src/financial_portfolio/reference.py"
    source = source_path.read_bytes()
    image_id = baseline["docker_image"]["image_id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Baseline is missing its pinned Docker image ID")
    # Bind actual source content, not only a mutable package version or Git label.
    binding = {"protocol_id": protocol["protocol_id"],
               "protocol_manifest_sha256": _sha(protocol_manifest.read_bytes()),
               "baseline_preflight_sha256": _sha(baseline_preflight.read_bytes()),
               "evaluator_key_sha256": baseline["evaluator_key_sha256"],
               "source_files_sha256": protocol["sha256"],
               "baseline_source_commit": baseline["repository_commit"],
               "reference_source_sha256": _sha(source), "docker_image_id": image_id}
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    _exclusive_write(output_dir / "binding.json", _json_bytes(binding))
    records, seen = [], set()
    with _evaluator_key(key):
        for seed, expected in zip(seeds, schedule):
            episode = generator.build_episode(seed, split="dev")
            task_id, public_files = episode["id"], episode["public_files"]
            if task_id in seen:
                raise ValueError("Duplicate generated task ID")
            seen.add(task_id)
            if task_id != expected["task_id"] or set(public_files) != PUBLIC_FILES:
                raise ValueError("Generated task differs from frozen development schedule")
            public_hashes = {name: _sha(content.encode()) for name, content in public_files.items()}
            system_prompt = clarified.clarified_prompt(public_files)
            if (public_hashes != expected["public_sha256"]
                    or _sha(episode["prompt"].encode()) != expected["prompt_sha256"]
                    or _sha(system_prompt.encode()) != expected["system_prompt_sha256"]
                    or generator.FAMILIES[seed % 4] != expected["family"]):
                raise ValueError("Generated public inputs or prompts differ from frozen baseline")
            workspace = output_dir / f"dev-{seed}"
            workspace.mkdir(mode=0o700)
            for name, content in public_files.items():
                _exclusive_write(workspace / name, content.encode())
            _exclusive_write(workspace / "research.py", source)
            execution = _container_reference(workspace, image_id)
            if {path.name for path in workspace.iterdir()} != PUBLIC_FILES | {"research.py", "policy.json"}:
                raise ValueError("Standalone reference wrote unexpected artifacts")
            if any(_sha((workspace / name).read_bytes()) != digest
                   for name, digest in {**public_hashes, "research.py": _sha(source)}.items()):
                raise ValueError("Standalone reference changed its source or public inputs")
            policy_path = workspace / "policy.json"
            if not policy_path.is_file() or policy_path.is_symlink() or policy_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Standalone reference produced an invalid policy artifact")
            policy_path.chmod(0o600)
            policy = _load_json(policy_path)
            score = _checked_score(verifier.score_submission(episode, policy))
            gold_score = _checked_score(verifier.score_submission(
                episode, verifier.reference_submission(episode)))
            if score != gold_score:
                raise ValueError("Public reference score differs from trusted gold score")
            _exclusive_write(workspace / "ingestion.json", _json_bytes(
                validation.inventory_for_public_files(public_files)))
            _exclusive_write(workspace / "trials.json", _json_bytes(policy["trials"]))
            artifact_names = ("research.py", "ingestion.json", "trials.json", "policy.json")
            payloads = {f"/workspace/{name}": (workspace / name).read_bytes() for name in artifact_names}
            checkpoints = {stage: validation.validate_stage(stage, public_files, payloads)
                           for stage in validation.STAGES}
            if not all(result["valid"] is True for result in checkpoints.values()):
                raise ValueError("Standalone reference failed a structural checkpoint")
            record = {"task_id": task_id, "seed": seed, "split": "dev", "family": expected["family"],
                      "public_sha256": public_hashes, "prompt_sha256": expected["prompt_sha256"],
                      "system_prompt_sha256": expected["system_prompt_sha256"],
                      "input_fingerprint_sha256": _sha(_json_bytes(public_hashes)),
                      "artifact_sha256": {name: _sha((workspace / name).read_bytes()) for name in artifact_names},
                      "public_reference_score": score, "trusted_gold_score": gold_score,
                      "attainable_financial_score_ceiling": score["reward"],
                      "public_perfect": True, "decision": policy["decision"], "selected": policy["selected"],
                      "checkpoint_checks": checkpoints, "execution": execution}
            _exclusive_write(workspace / "reference_score.json", _json_bytes(record))
            records.append(record)
    _verify_source_hashes(repo_root, protocol["sha256"])
    _verify_source_hashes(repo_root, baseline["source_files_sha256"])
    result = {"schema_version": 1, "status": "complete", "binding": binding,
              "split": "dev", "seeds": list(seeds), "reference_count": len(records),
              "model_calls": 0, "reserved_test_episodes_generated": 0,
              "native_agent_harness_gold_validation": False,
              "ceiling_basis": "The required public choice is unique. Imperfect public answers disable economic credit and score below 0.8; publicly perfect answers score 0.8 plus 0.2 times the unchanged economic quality. These are raw scorer ceilings, not normalized model scores.",
              "records": records}
    # Private episodes and hidden utility tables are intentionally never saved.
    # Fail closed before completion if the actual evaluator credential escaped.
    if any(key.encode() in path.read_bytes() for path in output_dir.rglob("*") if path.is_file()):
        raise ValueError("Reference outputs failed credential-value scan")
    _exclusive_write(output_dir / "references.json", _json_bytes(result))
    return result


def validate_reference_report(path: Path, *, repo_root: Path, protocol_manifest: Path) -> dict:
    """Check a complete trusted report and its source/artifact bindings, read only.

    The caller must additionally match the evaluator-key fingerprint and task
    schedule to its own live preflight. This checks saved control evidence; it
    does not claim to rerun numerical scoring or validate a model trajectory.
    """
    path, repo_root = Path(path).resolve(), Path(repo_root).resolve()
    report, protocol = _load_json(path), _load_json(Path(protocol_manifest))
    if path.is_relative_to(repo_root):
        raise ValueError("Reference evidence must remain outside the repository")
    if (report.get("status") != "complete" or type(report.get("reference_count")) is not int
            or report["reference_count"] != 32 or report.get("split") != "dev"
            or type(report.get("model_calls")) is not int or report["model_calls"] != 0
            or type(report.get("reserved_test_episodes_generated")) is not int
            or report["reserved_test_episodes_generated"] != 0
            or report.get("native_agent_harness_gold_validation") is not False):
        raise ValueError("Reference report is not a complete development control")
    _validate_seeds(report["seeds"])
    binding = report["binding"]
    if (binding["protocol_id"] != protocol["protocol_id"]
            or binding["protocol_manifest_sha256"] != _sha(Path(protocol_manifest).read_bytes())
            or binding["source_files_sha256"] != protocol["sha256"]
            or _load_json(path.parent / "binding.json") != binding):
        raise ValueError("Reference report is bound to a different protocol")
    _verify_source_hashes(repo_root, binding["source_files_sha256"])
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", binding["docker_image_id"]):
        raise ValueError("Reference image is not pinned")
    for name in ("evaluator_key_sha256", "baseline_preflight_sha256", "reference_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", binding[name]):
            raise ValueError("Invalid reference binding digest")
    records = report["records"]
    if (not isinstance(records, list) or len(records) != 32
            or len({row["task_id"] for row in records}) != 32):
        raise ValueError("Reference report must contain 32 distinct tasks")
    _validate_seeds(row["seed"] for row in records)
    artifacts = {"research.py", "ingestion.json", "trials.json", "policy.json"}
    for row in records:
        workspace = path.parent / f"dev-{row['seed']}"
        score = _checked_score(row["public_reference_score"])
        gold = _checked_score(row["trusted_gold_score"])
        if (score != gold or row["attainable_financial_score_ceiling"] != score["reward"]
                or row["public_perfect"] is not True or row["split"] != "dev"
                or _load_json(workspace / "reference_score.json") != row):
            raise ValueError("Inconsistent reference score record")
        if (set(row["public_sha256"]) != PUBLIC_FILES or set(row["artifact_sha256"]) != artifacts
                or row["input_fingerprint_sha256"] != _sha(_json_bytes(row["public_sha256"]))
                or row["artifact_sha256"]["research.py"] != binding["reference_source_sha256"]):
            raise ValueError("Inconsistent reference artifact inventory")
        for name, expected in {**row["public_sha256"], **row["artifact_sha256"]}.items():
            artifact = workspace / name
            if artifact.is_symlink() or _sha(artifact.read_bytes()) != expected:
                raise ValueError("Reference artifact hash mismatch")
        execution = row["execution"]
        if (execution["kind"] != "standalone_public_reference_container"
                or execution["image_id"] != binding["docker_image_id"]
                or execution["network"] != "none" or type(execution["returncode"]) is not int
                or execution["returncode"] != 0 or execution["container_cleanup_confirmed"] is not True
                or type(execution["wall_timeout_seconds"]) is not int or execution["wall_timeout_seconds"] != 30
                or type(execution["address_space_limit_bytes"]) is not int
                or execution["address_space_limit_bytes"] != 268435456
                or execution["native_agent_harness_gold_validation"] is not False
                or set(row["checkpoint_checks"]) != {"ingestion", "trials", "submission"}
                or not all(item["valid"] is True for item in row["checkpoint_checks"].values())):
            raise ValueError("Reference execution or structural checks are incomplete")
    return report
