"""Fresh-key reference controls without an earlier private run or saved answers."""

from importlib.metadata import version
from importlib.resources import files
import json
from pathlib import Path
import re
import secrets
import subprocess

from financial_portfolio_operations.references import (
    PUBLIC_FILES, _checked_score, _container_reference, _evaluator_key,
    _exclusive_write, _json_bytes, _load_json, _sha, _verify_source_hashes,
)


def verify_installed_sources(manifest):
    """A clean checkout is not proof that the running installation matches it."""
    dependencies = {name: version(name) for name in manifest["packages"]}
    if dependencies != manifest["packages"]:
        raise ValueError("Installed dependencies differ from runtime protocol")
    for relative, expected in manifest["sha256"].items():
        if relative.startswith("environments/") and "/src/" in relative:
            resource = relative.split("/src/", 1)[1]
            package, remainder = resource.split("/", 1)
            if _sha(files(package).joinpath(remainder).read_bytes()) != expected:
                raise ValueError(f"Imported package resource differs from protocol: {package}/{remainder}")
    return dependencies


def inspect_image(image):
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                            text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError("Docker image must already be installed; bootstrap never pulls images")
    metadata = json.loads(result.stdout)
    if len(metadata) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", metadata[0]["Id"]):
        raise ValueError("Invalid Docker image identity")
    return {"image_id": metadata[0]["Id"], "repo_digests": metadata[0].get("RepoDigests", [])}


def bootstrap(output_dir, *, repo_root, protocol_manifest, image="python:3.12-slim",
              seed_start=4, num_tasks=4):
    """Generate a private key, public-only reproduction controls and bound config.

    The fresh key determines new synthetic realizations. These controls demonstrate
    software reproducibility, not unseen-distribution generalization or model skill.
    Output is exclusive and must be outside the checkout; interrupted preparations
    remain incomplete. Only training seeds are supported by this release.
    """
    from financial_portfolio.generator import build_episode, FAMILIES, SPLITS
    from financial_portfolio_clarified.prime_adapter import clarified_prompt
    from financial_portfolio_staged.validation import inventory_for_public_files, validate_stage, STAGES
    from financial_research.verifier import score_submission, reference_submission

    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    protocol_manifest = Path(protocol_manifest).resolve()
    if output_dir.is_relative_to(repo_root):
        raise ValueError("Private bootstrap output must be outside the repository")
    if output_dir.exists():
        raise FileExistsError("Bootstrap requires a fresh output directory")
    if (type(seed_start) is not int or type(num_tasks) is not int or not 1 <= num_tasks <= 16
            or seed_start < SPLITS["train"][0] or seed_start + num_tasks > SPLITS["train"][1]):
        raise ValueError("Bootstrap requires 1–16 contiguous training seeds")
    manifest = _load_json(protocol_manifest)
    if manifest.get("protocol_id") != "portfolio-runtime-v1":
        raise ValueError("Incorrect runtime protocol")
    _verify_source_hashes(repo_root, manifest["sha256"])
    dependencies = verify_installed_sources(manifest)
    docker = inspect_image(image)
    source = (repo_root / "environments/financial_portfolio/src/financial_portfolio/reference.py").read_bytes()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    key = secrets.token_hex(32)
    _exclusive_write(output_dir / "evaluator.key", (key + "\n").encode())
    binding = {"protocol_id": manifest["protocol_id"], "protocol_sha256": _sha(protocol_manifest.read_bytes()),
               "source_files_sha256": manifest["sha256"], "installed_distributions": dependencies,
               "evaluator_key_sha256": _sha(key.encode()), "docker_image": docker,
               "split": "train", "seeds": list(range(seed_start, seed_start + num_tasks)),
               "reference_source_sha256": _sha(source)}
    _exclusive_write(output_dir / "binding.json", _json_bytes(binding))
    records = []
    with _evaluator_key(key):
        for seed in binding["seeds"]:
            episode = build_episode(seed, split="train")
            public = episode["public_files"]
            workspace = output_dir / f"train-{seed}"
            workspace.mkdir(mode=0o700)
            for name, content in public.items():
                _exclusive_write(workspace / name, content.encode())
            _exclusive_write(workspace / "research.py", source)
            public_hashes = {name: _sha(content.encode()) for name, content in public.items()}
            execution = _container_reference(workspace, docker["image_id"])
            if {p.name for p in workspace.iterdir()} != PUBLIC_FILES | {"research.py", "policy.json"}:
                raise ValueError("Reference wrote unexpected artifacts")
            if any(_sha((workspace / name).read_bytes()) != digest
                   for name, digest in {**public_hashes, "research.py": _sha(source)}.items()):
                raise ValueError("Reference changed its source or input data")
            policy = _load_json(workspace / "policy.json")
            score = _checked_score(score_submission(episode, policy))
            gold = _checked_score(score_submission(episode, reference_submission(episode)))
            if score != gold:
                raise ValueError("Public reference differs from trusted financial score")
            _exclusive_write(workspace / "ingestion.json", _json_bytes(inventory_for_public_files(public)))
            _exclusive_write(workspace / "trials.json", _json_bytes(policy["trials"]))
            names = ("research.py", "ingestion.json", "trials.json", "policy.json")
            payloads = {f"/workspace/{name}": (workspace / name).read_bytes() for name in names}
            stages = {stage: validate_stage(stage, public, payloads) for stage in STAGES}
            if any(value["valid"] is not True for value in stages.values()):
                raise ValueError("Reference failed a structural checkpoint")
            for path in workspace.iterdir():
                path.chmod(0o600)
            records.append({"seed": seed, "task_id": episode["id"], "family": FAMILIES[seed % 4],
                            "public_sha256": public_hashes, "prompt_sha256": _sha(episode["prompt"].encode()),
                            "system_prompt_sha256": _sha(clarified_prompt(public).encode()),
                            "artifact_sha256": {name: _sha((workspace / name).read_bytes()) for name in names},
                            "reference_score": score, "trusted_gold_score": gold,
                            "checkpoint_checks": stages, "execution": execution})
    _verify_source_hashes(repo_root, manifest["sha256"])
    verify_installed_sources(manifest)
    report = {"schema_version": 1, "status": "complete", "binding": binding,
              "reference_count": len(records), "records": records, "model_calls": 0,
              "native_agent_harness_validation": False, "optimizer_step_verified": False}
    for path in output_dir.rglob("*"):
        if path.is_file() and path.name != "evaluator.key" and key.encode() in path.read_bytes():
            raise ValueError("Bootstrap artifacts failed credential-value scan")
    _exclusive_write(output_dir / "references.json", _json_bytes(report))
    return report


def validate_bootstrap(path, *, repo_root, protocol_manifest):
    """Bind saved controls to current source, private key and every public artifact."""
    path, repo_root = Path(path).resolve(), Path(repo_root).resolve()
    if path.is_relative_to(repo_root):
        raise ValueError("Bootstrap report must be outside the repository")
    report, manifest = _load_json(path), _load_json(Path(protocol_manifest))
    binding = report["binding"]
    if (report.get("status") != "complete" or binding["protocol_id"] != "portfolio-runtime-v1"
            or binding["protocol_sha256"] != _sha(Path(protocol_manifest).read_bytes())
            or binding["source_files_sha256"] != manifest["sha256"]
            or _load_json(path.parent / "binding.json") != binding):
        raise ValueError("Bootstrap binding differs from the runtime protocol")
    _verify_source_hashes(repo_root, manifest["sha256"])
    verify_installed_sources(manifest)
    key = (path.parent / "evaluator.key").read_text().strip()
    if len(key) != 64 or _sha(key.encode()) != binding["evaluator_key_sha256"]:
        raise ValueError("Bootstrap evaluator key changed")
    if (type(report.get("reference_count")) is not int or report["reference_count"] != len(binding["seeds"])
            or len(report["records"]) != report["reference_count"]
            or [r["seed"] for r in report["records"]] != binding["seeds"]
            or len({r["task_id"] for r in report["records"]}) != report["reference_count"]):
        raise ValueError("Invalid bootstrap schedule")
    for row in report["records"]:
        if _checked_score(row["reference_score"]) != _checked_score(row["trusted_gold_score"]):
            raise ValueError("Reference financial scores differ")
        if (set(row["public_sha256"]) != PUBLIC_FILES
                or set(row["artifact_sha256"]) != {"research.py", "ingestion.json", "trials.json", "policy.json"}):
            raise ValueError("Incomplete bootstrap artifacts")
        for name, expected in {**row["public_sha256"], **row["artifact_sha256"]}.items():
            file = path.parent / f"train-{row['seed']}" / name
            if file.is_symlink() or _sha(file.read_bytes()) != expected:
                raise ValueError("Bootstrap artifact changed")
    return report
