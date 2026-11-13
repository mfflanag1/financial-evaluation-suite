"""Explicit fresh bootstrap, native run and conservative reporting commands."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

from financial_portfolio_operations.references import _evaluator_key, _exclusive_write, _json_bytes, _sha


def run(config_path, *, bootstrap_path, repo_root, protocol_manifest, journal_dir):
    from verifiers.v1.cli.run import main as native_main
    from verifiers.v1.configs.cli.run import RunnerConfig
    from .bootstrap import inspect_image, validate_bootstrap
    from .prime_adapter import FinancialPortfolioRuntimeTaskset
    from .telemetry import capture_runtime
    from .binding import build_run_binding

    report = validate_bootstrap(bootstrap_path, repo_root=repo_root, protocol_manifest=protocol_manifest)
    payload = Path(config_path).read_bytes()
    config = tomllib.loads(payload.decode()) if Path(config_path).suffix == ".toml" else json.loads(payload)
    # Native config resolution narrows plugin dictionaries in place. Preserve the
    # caller's exact JSON-compatible config for the recorded run binding.
    resolved = RunnerConfig.model_validate(deepcopy(config))
    agent, taskset = resolved.env.agent, resolved.env.taskset
    if (agent.model is not None or agent.client is not None or agent.sampling is not None
            or agent.scoring.runtime is not None or agent.scoring.env is not None or agent.scoring.network is not None):
        raise ValueError("Runtime v1 forbids agent inference and scoring runtime/environment/network overrides")
    if (taskset.id != "financial-portfolio-runtime" or agent.harness.id != "financial-portfolio-runtime"
            or taskset.split != "train" or taskset.family is not None
            or list(range(taskset.seed_start, taskset.seed_start + taskset.num_tasks)) != report["binding"]["seeds"]):
        raise ValueError("Run task schedule differs from the bootstrap")
    if (config.get("max_concurrent") != 1 or config.get("num_rollouts") != 1
            or resolved.env.retries.max_retries != 0 or agent.retries.max_retries != 0
            or agent.scoring.retries.max_retries != 0
            or agent.runtime.type != "docker" or agent.scoring.mode != "isolated"
            or agent.runtime.allow or agent.runtime.block not in ([], ["*"]) or agent.harness.forward_env
            or agent.network_notice is not True
            or config.get("select") != {"shuffle": False, "limit": taskset.num_tasks}):
        raise ValueError("Runtime v1 requires one attempt per task, serial Docker isolation and zero retries")
    if type(config.get("dry_run")) is not bool:
        raise ValueError("dry_run must be explicit")
    client = config["client"]
    endpoint = urlsplit(client["base_url"])
    if (client.get("type") != "eval" or endpoint.scheme not in ("http", "https") or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("Invalid model endpoint")
    if not config["dry_run"] and (endpoint.hostname == "endpoint.invalid" or config["model"] == "INSTRUCT_MODEL_ID"):
        raise ValueError("Bind a real model and endpoint before live evaluation")
    output = Path(config["output_dir"]).resolve()
    if not Path(config["output_dir"]).is_absolute() or output.is_relative_to(Path(repo_root).resolve()):
        raise ValueError("Output must be an absolute path outside the repository")
    run_dir = config.get("run", {}).get("dir")
    if not isinstance(run_dir, str) or run_dir in ("", ".", "..") or Path(run_dir).name != run_dir:
        raise ValueError("Run directory must be one simple name")
    if (output / run_dir).exists():
        raise FileExistsError("Use a fresh run directory; interrupted attempts must not be overwritten")
    journal_dir = Path(journal_dir).resolve()
    if journal_dir.is_relative_to(Path(repo_root).resolve()):
        raise ValueError("Host journals must remain outside the repository")
    expected_image = report["binding"]["docker_image"]["image_id"]
    if any(inspect_image(image)["image_id"] != expected_image
           for image in (agent.runtime.image, taskset.task.image)):
        raise ValueError("Solver or verifier image differs from bootstrap")
    # Resolve mutable tags before native launch, including the verifier task image.
    config["env"]["agent"]["runtime"]["image"] = expected_image
    config["env"]["taskset"].setdefault("task", {})["image"] = expected_image
    resolved = RunnerConfig.model_validate(deepcopy(config))
    taskset = resolved.env.taskset
    key = (Path(bootstrap_path).parent / "evaluator.key").read_text().strip()
    with _evaluator_key(key):
        tasks = list(FinancialPortfolioRuntimeTaskset(taskset).load())
        for task, reference in zip(tasks, report["records"]):
            data = task.data
            if (data.id != reference["task_id"]
                    or {name: _sha(value.encode()) for name, value in data.public_files.items()} != reference["public_sha256"]
                    or _sha(data.prompt.encode()) != reference["prompt_sha256"]
                    or _sha(data.system_prompt.encode()) != reference["system_prompt_sha256"]):
                raise ValueError("Native task differs from bootstrap reference")
        with capture_runtime(journal_dir):
            bound = journal_dir / "bound.json"
            _exclusive_write(bound, _json_bytes(config))
            _exclusive_write(journal_dir / "run_binding.json", _json_bytes(build_run_binding(
                config, tasks, config_sha256=_sha(bound.read_bytes()),
                bootstrap_sha256=_sha(Path(bootstrap_path).read_bytes()),
                protocol_sha256=_sha(Path(protocol_manifest).read_bytes()))))
            native_main(["@", str(bound)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("bootstrap", help="Create a fresh key and public reference controls")
    for name in ("out", "repo", "protocol"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--image", default="python:3.12-slim")
    prepare.add_argument("--seed-start", type=int, default=4)
    prepare.add_argument("--num-tasks", type=int, default=4)
    execute = commands.add_parser("run", help="Run a bound config with host prompt and request capture")
    for name in ("config", "bootstrap", "repo", "protocol", "journal-dir"):
        execute.add_argument("--" + name, type=Path, required=True)
    report_parser = commands.add_parser("report", help="Separate resolved financial outcomes from interruptions")
    report_parser.add_argument("--run", type=Path, required=True)
    report_parser.add_argument("--expected", type=int, required=True)
    report_parser.add_argument("--journal-dir", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        from .bootstrap import bootstrap
        report = bootstrap(args.out, repo_root=args.repo, protocol_manifest=args.protocol,
                           image=args.image, seed_start=args.seed_start, num_tasks=args.num_tasks)
        print(json.dumps({"status": report["status"], "reference_count": report["reference_count"]}))
    elif args.command == "run":
        run(args.config, bootstrap_path=args.bootstrap, repo_root=args.repo,
            protocol_manifest=args.protocol, journal_dir=args.journal_dir)
    else:
        from .accounting import report_path
        _exclusive_write(args.out, _json_bytes(report_path(args.run, expected=args.expected, journal_dir=args.journal_dir)))


if __name__ == "__main__":
    main()
