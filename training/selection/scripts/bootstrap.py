#!/usr/bin/env python3
"""Build a separate selection-training CPU installation from verified sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[3]
DEPENDENCIES = ROOT / "training/dependencies"
FORBIDDEN = {"financial-portfolio-training", "financial-portfolio", "torch", "torchvision", "torchaudio", "vllm", "financial-portfolio-runtime",
             "financial-portfolio-operations", "financial-portfolio-staged", "financial-portfolio-clarified"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def verify_source(path: Path, pin: dict) -> None:
    if capture(["git", "rev-parse", "HEAD"], path) != pin["commit"]:
        raise ValueError("native source commit differs from the pinned contract")
    if capture(["git", "status", "--porcelain", "--untracked-files=all"], path):
        raise ValueError("native source checkout must be clean")
    ignored = capture(["git", "ls-files", "--others", "--ignored", "--exclude-standard"], path)
    for name in ignored.splitlines():
        parts = Path(name).parts
        if parts[0] not in {"src", "verifiers", "renderers", "packages"}:
            continue
        if "__pycache__" in parts or name == "renderers/_version.py":
            continue
        raise ValueError("ignored file could contaminate a native package build")


def validate_registry_lock(path: Path) -> set[str]:
    names: set[str] = set()
    hashed = False
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("    --hash=sha256:"):
            if not re.fullmatch(r"    --hash=sha256:[0-9a-f]{64}", line.removesuffix(" \\")):
                raise ValueError("invalid registry hash")
            hashed = True
            continue
        if names and not hashed:
            raise ValueError("registry package has no hash")
        match = re.fullmatch(r"([a-z0-9-]+)==([A-Za-z0-9.+]+) \\", line)
        if not match or match[1] in names or match[1] in FORBIDDEN:
            raise ValueError("registry lock must contain unique pinned CPU packages only")
        names.add(match[1])
        hashed = False
    if not names or not hashed:
        raise ValueError("empty or incomplete registry lock")
    return names


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        entries = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError("wheel must contain one distribution")
        metadata = BytesParser().parsebytes(archive.read(entries[0]))
    return metadata["Name"], metadata["Version"]


# This process imports exclusively installed wheels, with no source path insertion.
CHECK = r'''
import hashlib, importlib.metadata as md, importlib.util, json, pathlib, sys, sysconfig, zipfile
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert sys.flags.isolated and sys.version_info[:2] == (3, 12)
site = pathlib.Path(sysconfig.get_path("purelib")).resolve()
for name in payload["forbidden"]:
    try: md.distribution(name)
    except md.PackageNotFoundError: pass
    else: raise AssertionError("excluded distribution installed: " + name)
resources = 0
for wheel in payload["wheels"]:
    dist = md.distribution(wheel["name"])
    assert dist.version == wheel["version"], wheel["name"]
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    assert not direct.get("dir_info", {}).get("editable", False)
    with zipfile.ZipFile(wheel["path"]) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or ".dist-info/" in entry.filename: continue
            unresolved = pathlib.Path(dist.locate_file(entry.filename))
            assert not unresolved.is_symlink()
            target = unresolved.resolve()
            assert target.is_relative_to(site)
            assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(archive.read(entry)).digest(), entry.filename
            resources += 1
import verifiers.v1 as vf
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.delta import DeltaStreamer
from prime_rl.env.worker import EnvWorker
from prime_rl.configs.rl import RLConfig
from financial_selection_training import FinancialSelectionTrainingEnv, FinancialSelectionTrainingTaskset, FinancialSelectionTrainingHarness
import financial_selection_training as plugin
from financial_selection_training.environment import verify_native_sources
verify_native_sources()
bindings = json.loads((pathlib.Path(plugin.__file__).parent / "source_bindings.json").read_text())
for relative, expected in bindings["native_sources"].items():
    assert hashlib.sha256((site / relative).read_bytes()).hexdigest() == expected, relative
assert hasattr(vf, "SamplerSupport") and importlib.util.find_spec("torch") is None
assert not any(name.startswith(("financial_portfolio_training", "financial_portfolio_runtime", "financial_portfolio_operations", "financial_portfolio_staged", "financial_portfolio_clarified")) for name in sys.modules)
config = RLConfig.model_validate(payload["training_contract"])
env = vf.load_environment(config.orchestrator.train.source[0].env)
assert type(env) is FinancialSelectionTrainingEnv
assert type(env.taskset) is FinancialSelectionTrainingTaskset
assert type(env._harnesses["agent"]) is FinancialSelectionTrainingHarness
classes = [vf.Env, TrainClientConfig, DeltaStreamer, EnvWorker, RLConfig, FinancialSelectionTrainingEnv, FinancialSelectionTrainingTaskset, FinancialSelectionTrainingHarness]
print(json.dumps({"status": "passed", "installed_resource_count": resources, "native_binding_count": len(bindings["native_sources"]), "financial_binding_count": len(bindings["financial_sources"]), "adapter_binding_count": len(bindings["adapter_sources"]), "native_plugin_count": 3, "imports": [x.__name__ for x in classes], "torch_installed": False, "training_performed": False}))
'''


def bootstrap(args: argparse.Namespace) -> dict:
    out = args.out.resolve()
    if out.is_relative_to(ROOT) or ROOT.is_relative_to(out) or out.exists():
        raise ValueError("output must be a new directory outside the checkout")
    pins = json.loads((DEPENDENCIES / "sources.json").read_text())
    sources = {name: getattr(args, name + "_source").resolve() for name in pins["sources"]}
    for name, pin in pins["sources"].items():
        verify_source(sources[name], pin)
        if "gitlink" in pin:
            entry = capture(["git", "ls-tree", "HEAD", pin["gitlink"]], sources["prime_rl"])
            if entry.split()[:3] != ["160000", "commit", pin["commit"]]:
                raise ValueError("Prime RL gitlink differs from pinned source")
    if capture([args.python, "-I", "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"]) != "3.12":
        raise ValueError("Python 3.12 is required")
    registry = DEPENDENCIES / "cpu-requirements.txt"
    registry_names = validate_registry_lock(registry)
    out.mkdir(mode=0o700, parents=True)
    wheels_dir = out / "wheels"
    wheels_dir.mkdir()
    uv = [args.uv, "--no-config"] + (["--offline"] if args.offline else [])

    def run(command: list[str], log: str) -> None:
        with (out / log).open("w") as handle:
            subprocess.run(command, cwd=out, stdout=handle, stderr=subprocess.STDOUT, check=True)

    build_sources = [sources["prime_rl"], sources["prime_rl"] / "packages/prime-rl-configs",
                     sources["verifiers"], sources["renderers"], sources["pydantic_config"],
                     ROOT / "environments/financial_research", ROOT / "environments/financial_selection",
                     ROOT / "training/selection"]
    expected = {name: version for pin in pins["sources"].values() for name, version in pin["distributions"].items()}
    expected.update({"financial-research": "0.2.0", "financial-selection": "0.1.0", "financial-selection-training": "0.1.0"})
    for i, source in enumerate(build_sources):
        run(uv + ["build", "--wheel", "--python", args.python, "--build-constraints",
                  str(DEPENDENCIES / "build-constraints.txt"), "--out-dir", str(wheels_dir), str(source)], f"build-{i}.log")
    wheels = []
    for path in sorted(wheels_dir.glob("*.whl")):
        name, version = wheel_metadata(path)
        if expected.pop(name, None) != version:
            raise ValueError("built wheel metadata differs from native version pin")
        wheels.append({"name": name, "version": version, "path": str(path), "sha256": digest(path)})
    if expected:
        raise ValueError("missing native wheel")
    requirements = out / "resolved.txt"
    requirements.write_text(registry.read_text() + "\n" + "\n".join(
        f"{Path(w['path']).as_uri()} --hash=sha256:{w['sha256']}" for w in wheels) + "\n")
    run(uv + ["venv", "--python", args.python, str(out / "venv")], "venv.log")
    python = str(out / "venv/bin/python")
    run(uv + ["pip", "install", "--python", python, "--only-binary", ":all:",
              "--require-hashes", "--requirement", str(requirements)], "install.log")
    run(uv + ["pip", "check", "--python", python], "dependency-check.log")
    payload = out / "inventory.json"
    contract = ROOT / "training/selection/configs/training_v1/contract.json"
    payload.write_text(json.dumps({"wheels": wheels, "forbidden": sorted(FORBIDDEN),
                                  "training_contract": json.loads(contract.read_text())}, indent=2) + "\n")
    clean_env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}}
    clean_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", WANDB_DISABLED="true")
    completed = subprocess.run([python, "-I", "-B", "-c", CHECK, str(payload)], cwd=out,
                               env=clean_env, text=True, capture_output=True, check=True)
    imports = json.loads(completed.stdout)
    report = {"schema_version": 1, "status": "passed", "scope": pins["scope"],
              "source_commits": {k: v["commit"] for k, v in pins["sources"].items()},
              "registry_packages": len(registry_names), "registry_lock_sha256": digest(registry),
              "build_constraints_sha256": digest(DEPENDENCIES / "build-constraints.txt"),
              "training_contract_sha256": digest(contract),
              "wheels": [{k: v for k, v in w.items() if k != "path"} for w in wheels],
              "dependency_check_passed": True, "installed_checks": imports, "training_performed": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--offline", action="store_true")
    for name in ("prime-rl", "verifiers", "renderers", "pydantic-config"):
        parser.add_argument(f"--{name}-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args), indent=2))


if __name__ == "__main__":
    main()
