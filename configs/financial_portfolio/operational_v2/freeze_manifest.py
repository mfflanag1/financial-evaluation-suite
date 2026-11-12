"""Bind a new operational protocol without changing any frozen predecessor."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
PACKAGE = ROOT / "environments/financial_portfolio_operations"
MANIFEST = CONFIG / "manifest.json"
PREDECESSOR = ROOT / "configs/financial_portfolio/clarified_calibration_v1/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest():
    spec = importlib.util.spec_from_file_location("operational_predecessor", PREDECESSOR.with_name("freeze_manifest.py"))
    predecessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predecessor)
    frozen = json.loads(PREDECESSOR.read_text())
    if frozen != predecessor.build_manifest():
        raise ValueError("Frozen calibration source changed")
    files = dict(frozen["sha256"])
    files[PREDECESSOR.relative_to(ROOT).as_posix()] = digest(PREDECESSOR)
    payload = {PACKAGE / "pyproject.toml", PACKAGE / "README.md"}
    payload.update((PACKAGE / "src").rglob("*.py"))
    payload.update(CONFIG.glob("*"))
    payload.update((ROOT / "tests").glob("test_portfolio_operations_*.py"))
    payload.discard(MANIFEST)
    for path in payload:
        if path.name == "__pycache__":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Protocol payload must contain regular files")
        files[path.relative_to(ROOT).as_posix()] = digest(path)
    policy = json.loads((CONFIG / "policy.json").read_text())
    return {
        "protocol_id": "portfolio-operational-v2",
        "predecessor_manifest_sha256": digest(PREDECESSOR),
        "packages": {**frozen["packages"], "financial-portfolio-operations": "0.1.0",
                     "openai": "2.54.0"},
        "schedule": policy["schedule"],
        "financial_scorer_changed": False,
        "public_instructions_changed": False,
        "prior_results_changed": False,
        "local_checks_are_model_performance": False,
        "sha256": dict(sorted(files.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_manifest()
    if args.write:
        with MANIFEST.open("x") as handle:
            handle.write(json.dumps(expected, indent=2) + "\n")
        print("Operational protocol manifest written.")
    else:
        if json.loads(MANIFEST.read_text()) != expected:
            raise SystemExit("Operational protocol manifest does not match its payload")
        print("Operational protocol manifest verified.")


if __name__ == "__main__":
    main()
