"""One bounded local native launch. This is execution evidence, not learning evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .contract import KEY_ENV, OFFLINE_ENV, digest, make_config, sanitized_environment, validate_config, verify_sources
from .ledger import LEDGER_ENV, Ledger, Limits
from .processes import alive, cleanup_containers, collect_tree, stop_tree

KEY_FD_ENV = "FINANCIAL_SELECTION_SUPERVISOR_KEY_FD"


def watchdog_reason(snapshot, now, is_alive=alive):
    run, workers, attempts = snapshot["run"], snapshot["workers"], snapshot["attempts"]
    if now - run["started"] >= run["total"]:
        return "time_budget"
    if run["native_finished"] is not None:
        return "native_failed" if now - run["native_finished"] >= run["finalize"] else None
    if any(not is_alive(w["pid"], w["created"]) for w in workers):
        return "worker_died"
    if (not any(w["ready"] is not None for w in workers) or not attempts) and now - run["started"] >= run["startup"]:
        return "startup_timeout"
    if len(attempts) >= run["max_attempts"] and all(a["status"] != "reserved" for a in attempts):
        return "attempt_budget"
    return None


def process_group_class(key):
    from prime_rl.utils.process import ProcessGroup
    from .hooks import COMPONENT_SHA_ENV
    import psutil

    class SupervisedProcessGroup(ProcessGroup):
        def wait(self, *names):
            if names != ("trainer", "orchestrator"):
                raise ValueError("Unexpected native completion group")
            result = super().wait(*names)
            Ledger.from_env().native_finished()
            return result

        def start(self, name, cmd, env, log_path):
            command = list(map(str, cmd))
            clean = {**sanitized_environment(env, key), **OFFLINE_ENV}
            if len(command) == 3 and command[0] in {"orchestrator", "env-worker-pool", "inference"} and command[1] == "@":
                kind = {"orchestrator": "orchestrator", "env-worker-pool": "pool", "inference": "inference"}[command[0]]
                if kind in {"orchestrator", "pool"}:
                    clean[KEY_ENV] = key
                    clean[COMPONENT_SHA_ENV] = hashlib.sha256(Path(command[2]).read_bytes()).hexdigest()
                    command = [sys.executable, "-I", "-B", "-m", "financial_selection_supervised", "component", kind, command[2]]
                expected_name = name == kind or (kind == "pool" and name == "train env worker pool selection-training-contract")
                if not expected_name:
                    raise ValueError("Unexpected component name")
            elif (name == "trainer" and command[0] == "torchrun"
                  and command[-4:-1] == ["-m", "prime_rl.trainer.rl.train", "@"]):
                kind = "trainer"
                command.insert(1, "--max-restarts=0")
                if clean.get("FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"):
                    command[-3] = "financial_selection_supervised.gpu_observer"
            else:
                raise ValueError("Launcher attempted an unreviewed component")
            super().start(name, command, clean, log_path)
            process = self.processes[-1]
            Ledger.from_env().component(kind, process.pid, psutil.Process(process.pid).create_time())
    return SupervisedProcessGroup


def child_launch(path):
    import psutil
    verify_sources()
    ledger = Ledger.from_env()
    raw = json.loads(Path(path).read_text())
    if digest(raw) != ledger.snapshot()["run"]["config_sha256"]:
        raise RuntimeError("Launcher configuration changed after supervisor admission")
    config = validate_config(raw, ledger.path.parent)
    fd = int(os.environ.pop(KEY_FD_ENV))
    try:
        key = os.read(fd, 65).decode("ascii")
    finally:
        os.close(fd)
    if re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise RuntimeError("Private generation key transfer failed")
    ledger.component("launcher", os.getpid(), psutil.Process().create_time())
    import prime_rl.entrypoints.rl as native
    native.ProcessGroup = process_group_class(key)
    # The launcher pre-download step sees offline settings and an existing local
    # model. Only orchestrator task selection and the trusted pool receive the key.
    native.rl(config)


def optimizer_inputs(plan_path, resolved_config):
    """Read and bind the four prelaunch artifacts before any native process starts."""
    from importlib.resources import files
    from .optimizer_evidence import read_json
    from .gpu_observer import digest as evidence_digest
    plan_path = Path(plan_path)
    if not plan_path.is_absolute() or plan_path.resolve() != plan_path or plan_path.is_symlink():
        raise ValueError("Optimizer plan must be an absolute canonical private file")
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("Optimizer plan must be an object")
    inputs = {}
    for name in ("config.json", "sources.json", "collector.py", "initial_model_manifest.json"):
        path = plan_path.parent / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("Missing or invalid prelaunch optimizer artifact")
        inputs[name] = path.read_bytes()
    for name, field in (("config.json", "config_sha256"), ("sources.json", "sources_sha256"), ("collector.py", "collector_sha256")):
        if hashlib.sha256(inputs[name]).hexdigest() != plan.get(field):
            raise ValueError("Prelaunch optimizer artifact changed")
    expected_sources = {**json.loads(files("financial_selection_training").joinpath("source_bindings.json").read_text()),
                        "supervised": json.loads(files("financial_selection_supervised").joinpath("source_bindings.json").read_text())}
    if (json.loads(inputs["sources.json"]) != expected_sources
            or inputs["collector.py"] != files("financial_selection_supervised").joinpath("gpu_observer.py").read_bytes()):
        raise ValueError("Prelaunch optimizer artifacts do not bind installed sources")
    resolved = resolved_config.model_dump(mode="json")
    planned = json.loads(inputs["config.json"])
    if (json.dumps(planned, sort_keys=True) != json.dumps(resolved, sort_keys=True)
            or plan.get("model_path") != resolved_config.trainer.model.name):
        raise ValueError("Prelaunch optimizer configuration differs from this launch")
    if evidence_digest(json.loads(inputs["initial_model_manifest.json"])) != plan.get("initial_weights_sha256"):
        raise ValueError("Prelaunch model manifest changed")
    return plan, inputs


def supervise(raw, root, limits, optimizer_plan=None):
    old_umask = os.umask(0o077)
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    ledger = None
    process = None
    tracked = {}
    clean_tree = False
    clean_docker = False
    reason = "supervisor_failed"
    report = None
    scope = None
    control_ok = True
    try:
        sources_hash = verify_sources()
        resolved_config = validate_config(raw, root)
        plan_data, plan_inputs = (None, {}) if optimizer_plan is None else optimizer_inputs(optimizer_plan, resolved_config)
        key = os.environ.get(KEY_ENV, "")
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Set a fresh 32-byte hexadecimal generation key in the designated environment variable")
        if plan_data is not None and (key.lower() in json.dumps(plan_data).lower()
                                      or any(key.encode() in payload.lower() for payload in plan_inputs.values())):
            raise ValueError("Generation key cannot appear in optimizer plan")
        # Require the chosen image before launch; this command never pulls it.
        image = raw["orchestrator"]["train"]["source"][0]["env"]["agent"]["runtime"]["image"]
        subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=10)
        ledger = Ledger.create(root, limits, digest(raw), sources_hash)
        snapshot = ledger.snapshot()
        scope = snapshot["run"]["scope"]
        path = Path(root) / "config.json"
        path.write_text(json.dumps(raw, indent=2) + "\n")
        clean_environment = sanitized_environment(os.environ, key)
        for name in list(clean_environment):
            if name.startswith(("FINANCIAL_SELECTION_SUPERVISOR_", "FINANCIAL_SELECTION_OPTIMIZER_")):
                del clean_environment[name]
        environment = {**clean_environment, **OFFLINE_ENV,
                       LEDGER_ENV: str(ledger.path), "VF_RUN_ID": snapshot["run"]["scope"]}
        if plan_data is not None:
            evidence = Path(root) / "optimizer-evidence"
            evidence.mkdir(mode=0o700)
            for name, payload in plan_inputs.items():
                (evidence / name).write_bytes(payload)
            copied_plan = Path(root) / "optimizer-plan.json"
            copied_plan.write_text(json.dumps(plan_data, indent=2) + "\n")
            environment["FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"] = str(copied_plan)
            environment["FINANCIAL_SELECTION_OPTIMIZER_EVIDENCE_DIR"] = str(evidence)
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, key.encode("ascii"))
            os.close(write_fd)
            write_fd = None
            environment[KEY_FD_ENV] = str(read_fd)
            with (Path(root) / "launcher.log").open("w") as output:
                process = subprocess.Popen([sys.executable, "-I", "-B", "-m", "financial_selection_supervised", "child", str(path)],
                                           env=environment, stdout=output, stderr=output, start_new_session=True,
                                           pass_fds=(read_fd,))
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
        while True:
            collect_tree(process.pid, tracked)
            snapshot = ledger.snapshot()
            for item in snapshot["workers"] + snapshot["components"]:
                tracked[item["pid"]] = item["created"]
            status = process.poll()
            if status is not None:
                reason = "completed" if (status == 0 and snapshot["run"]["native_finished"] is not None
                                         and snapshot["attempts"] and any(w["ready"] is not None for w in snapshot["workers"])) else "native_failed"
                break
            reason = watchdog_reason(snapshot, time.monotonic())
            if reason:
                break
            time.sleep(.05)
    except (KeyboardInterrupt, SystemExit):
        reason = "interrupted"
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if ledger is not None:
                try:
                    ledger.close(reason or "supervisor_failed")  # stop admission before signals
                except Exception:
                    control_ok = False
            # A corrupt/locked control ledger must never prevent process teardown.
            if process is not None:
                try:
                    clean_tree = stop_tree(process, tracked, limits.finalize_seconds)
                except Exception:
                    clean_tree = False
            if ledger is not None:
                if scope is not None:
                    try:
                        cleanup_containers(scope)
                        clean_docker = True
                    except Exception:
                        clean_docker = False
                try:
                    report = ledger.report()
                except Exception:
                    control_ok = False
                    report = {"schema": "selection-supervisor-v1", "scope": scope,
                              "reason": reason or "supervisor_failed", "attempts": None,
                              "outcomes": None, "accounting_complete": False,
                              "optimizer_validated": False}
                report["reason"] = report["reason"] or reason or "supervisor_failed"
                report["control_ledger_complete"] = control_ok
                report["process_cleanup_complete"] = clean_tree
                report["container_cleanup_complete"] = clean_docker
                report["success"] = (reason == "completed" and control_ok and clean_tree
                                     and clean_docker and report["accounting_complete"])
                (Path(root) / "supervisor-report.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            os.umask(old_umask)
            signal.signal(signal.SIGTERM, previous_term)
    if report is None:
        raise RuntimeError("Supervisor stopped before a run could be created")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--image", required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("check", "run"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--root", type=Path, required=True)
        if name == "run":
            command.add_argument("--execute", action="store_true", required=True)
            command.add_argument("--optimizer-plan", type=Path)
            command.add_argument("--max-attempts", type=int, default=8)
            command.add_argument("--startup-seconds", type=float, default=300)
            command.add_argument("--total-seconds", type=float, default=900)
            command.add_argument("--finalize-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        raw = make_config(args.model, args.image, args.root)
        validate_config(raw, args.root)
        if args.out.exists():
            raise ValueError("Prepared configuration output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(raw, handle, indent=2)
        print(json.dumps({"prepared": True, "config_sha256": digest(raw)}))
    elif args.command == "check":
        sources = verify_sources()
        raw = json.loads(args.config.read_text())
        validate_config(raw, args.root)
        print(json.dumps({"configuration_valid": True, "config_sha256": digest(raw), "sources_sha256": sources,
                          "execution_performed": False, "optimizer_validated": False}))
    else:
        report = supervise(json.loads(args.config.read_text()), args.root,
                           Limits(args.max_attempts, args.startup_seconds, args.total_seconds, args.finalize_seconds), args.optimizer_plan)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["success"] else 1)
