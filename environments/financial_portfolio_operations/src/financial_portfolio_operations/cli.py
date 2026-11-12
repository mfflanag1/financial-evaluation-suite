"""Local reference controls and an explicit instrumented native-run entry point."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tomllib
from urllib.parse import urlsplit


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path):
    raw = path.read_bytes()
    return tomllib.loads(raw.decode()) if path.suffix == ".toml" else json.loads(raw)


def validate_runtime_sources(repo, manifest):
    """A checked checkout cannot stand in for a different imported installation."""
    for package, expected in manifest["packages"].items():
        if version(package) != expected:
            raise ValueError(f"Installed dependency version differs: {package}")
    for relative in manifest["sha256"]:
        if relative.startswith("environments/") and "/src/" in relative and relative.endswith(".py"):
            module_name = relative.split("/src/", 1)[1][:-3].replace("/", ".")
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve() != (repo / relative).resolve():
                raise ValueError(f"Imported source differs from frozen repository: {module_name}")


def validate_reference_schedule(tasks, references):
    """Bind the controls to the actual ordered native taskset, before any model call."""
    tasks = list(tasks)
    records = references["records"]
    if len(tasks) != 32 or len(records) != 32 or len({task.data.id for task in tasks}) != 32:
        raise ValueError("Reference and native schedules must contain 32 distinct tasks")
    families = ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal")
    schedule = []
    for seed, task, record in zip(range(128, 160), tasks, records):
        data = task.data
        row = {"seed": seed, "task_id": data.id, "family": families[seed % 4],
               "public_sha256": {name: hashlib.sha256(value.encode()).hexdigest()
                                 for name, value in data.public_files.items()},
               "prompt_sha256": hashlib.sha256(data.prompt.encode()).hexdigest(),
               "system_prompt_sha256": hashlib.sha256(data.system_prompt.encode()).hexdigest()}
        if any(record.get(name) != value for name, value in row.items()):
            raise ValueError("Numerical reference differs from the live task schedule")
        schedule.append({**row, "attempts": 2})
    return schedule


def validate_run_config(config, template, repo_root):
    """Permit endpoint/run bindings, never changes to the frozen task or budgets."""
    actual, expected = deepcopy(config), deepcopy(template)
    for item in (actual, expected):
        for key in ("model", "output_dir", "run", "dry_run"):
            item.pop(key, None)
        item["client"].pop("base_url", None)
    # JSON equality is type-aware here; bool and integer budgets are not interchangeable.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Only model/endpoint/run bindings may differ from the frozen template")
    if type(config.get("dry_run")) is not bool:
        raise ValueError("dry_run must be explicit and boolean")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("A model binding is required")
    endpoint = urlsplit(config["client"]["base_url"])
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("Invalid model endpoint")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Endpoint must not contain credentials, query parameters or fragments")
    output = Path(config["output_dir"])
    if not output.is_absolute() or output.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("Run output must be an absolute path outside the repository")
    run = config.get("run")
    if not isinstance(run, dict) or set(run) != {"name", "dir"}:
        raise ValueError("Bind run name and directory explicitly")
    if any(not isinstance(value, str) or not value or Path(value).name != value
           or value in (".", "..") for value in run.values()):
        raise ValueError("Run labels must be simple directory names")
    if not config["dry_run"]:
        if config["model"] != "moonshotai/kimi-k2-0905" or endpoint.hostname == "endpoint.invalid":
            raise ValueError("Live binding must retain the calibrated instruct model and use a real endpoint")
    return output / run["dir"]


def run_instrumented(args):
    from .references import validate_reference_report
    from .telemetry import capture_requests
    from verifiers.v1.cli.run import main as native_main

    protocol = args.protocol.resolve()
    repo = protocol.parents[3]
    manifest = json.loads(protocol.read_text())
    if manifest.get("protocol_id") != "portfolio-operational-v2":
        raise ValueError("Wrong operational protocol")
    for relative, expected in manifest["sha256"].items():
        path = repo / relative
        if path.is_symlink() or _digest(path) != expected:
            raise ValueError("Frozen protocol source changed")
    validate_runtime_sources(repo, manifest)
    config = _read_config(args.config)
    template = _read_config(protocol.with_name("development.toml"))
    output = validate_run_config(config, template, repo)
    journal_dir = args.journal_dir.resolve()
    if journal_dir.is_relative_to(repo):
        raise ValueError("Journal must be outside the repository")
    if output.exists():
        raise FileExistsError("Use a fresh output directory, including after a dry run")
    references = None
    schedule = None
    if not config["dry_run"]:
        if args.reference_report is None:
            raise ValueError("A complete numerical reference report is required")
        references = validate_reference_report(args.reference_report, repo_root=repo,
                                               protocol_manifest=protocol)
        key = os.environ.get("FINANCIAL_RESEARCH_KEY", "")
        if not key or hashlib.sha256(key.encode()).hexdigest() != references["binding"]["evaluator_key_sha256"]:
            raise ValueError("Evaluator key differs from the bound reference controls")
        from financial_portfolio_clarified.prime_adapter import FinancialPortfolioClarifiedTaskset
        from verifiers.v1.configs.cli.run import RunnerConfig
        resolved = RunnerConfig.model_validate(config)
        schedule = validate_reference_schedule(
            FinancialPortfolioClarifiedTaskset(resolved.env.taskset).load(), references)
        image = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", config["env"]["agent"]["runtime"]["image"]],
            timeout=30))[0]
        if image["Id"] != references["binding"]["docker_image_id"]:
            raise ValueError("Runtime image differs from reference controls")
    with capture_requests(journal_dir):
        # Run from the exact snapshot whose hash is recorded, not a mutable caller file.
        bound = journal_dir / "bound.json"
        with bound.open("x") as handle:
            json.dump(config, handle, indent=2, allow_nan=False)
            handle.write("\n")
        binding = {"protocol_id": manifest["protocol_id"],
                   "protocol_manifest_sha256": _digest(protocol),
                   "bound_config_sha256": _digest(bound),
                   "reference_report_sha256": _digest(args.reference_report) if references else None,
                   "expected_attempts": 64, "dry_run": config["dry_run"],
                   "episode_schedule": schedule,
                   "repository_commit": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                   "worktree_dirty": bool(subprocess.check_output(
                       ["git", "status", "--porcelain"], cwd=repo)),
                   "installed_distributions": {name: version(name) for name in manifest["packages"]}}
        with (journal_dir / "run_binding.json").open("x") as handle:
            json.dump(binding, handle, indent=2)
            handle.write("\n")
        native_main(["@", str(bound)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("references", help="Run all 32 model-free development reference controls")
    for name in ("out", "repo", "protocol", "baseline-preflight", "key-file"):
        reference.add_argument("--" + name, type=Path, required=True)
    requests = commands.add_parser("requests", help="Summarize a trusted-host request journal")
    requests.add_argument("--journal", type=Path, required=True)
    requests.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="Run the bound native config inside trusted-host request capture")
    for name in ("config", "protocol", "journal-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--reference-report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "references":
        from .references import generate_reference_scores
        report = generate_reference_scores(args.out, repo_root=args.repo,
                                           protocol_manifest=args.protocol,
                                           baseline_preflight=args.baseline_preflight,
                                           key_file=args.key_file)
        print(json.dumps({"status": report["status"], "reference_count": report["reference_count"],
                          "report": str(args.out / "references.json")}))
    elif args.command == "requests":
        from .telemetry import summarize_journal
        report = summarize_journal(args.journal)
        with args.out.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Request accounting saved to {args.out}")
    else:
        run_instrumented(args)


if __name__ == "__main__":
    main()
