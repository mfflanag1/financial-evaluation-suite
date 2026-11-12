"""Create or verify the prerequisite protocol manifest before live execution."""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
PACKAGE = ROOT / "environments/financial_prerequisites"
MANIFEST = CONFIG / "manifest.json"
PREDECESSOR = ROOT / "configs/financial_portfolio/staged_v1/manifest.json"
CASE_IDS = [
    "prerequisite-schema-v1",
    "prerequisite-calculation-v1",
    "prerequisite-repair-v1",
    "prerequisite-coverage-v1",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_files(directory):
    return {path for path in directory.rglob("*") if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def verify_predecessors():
    calibration = ROOT / "configs/financial_portfolio/calibration_v1/manifest.json"
    staged = json.loads(PREDECESSOR.read_text())
    if staged["predecessor_manifest_sha256"] != digest(calibration):
        raise ValueError("Frozen predecessor manifest changed")
    for path in (calibration, PREDECESSOR):
        for relative, expected in json.loads(path.read_text())["sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Frozen predecessor file changed: {relative}")


def build_manifest():
    verify_predecessors()
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]
    if (project["name"], project["version"]) != ("financial-prerequisites", "0.1.0"):
        raise ValueError("Unexpected prerequisite package version")
    for relative in ("README.md", "src/financial_prerequisites/__init__.py",
                     "src/financial_prerequisites/cases.py",
                     "src/financial_prerequisites/prime_adapter.py"):
        if not (PACKAGE / relative).is_file():
            raise ValueError(f"Incomplete prerequisite package: {relative}")
    files = payload_files(PACKAGE) | payload_files(CONFIG)
    files.discard(MANIFEST)
    files.add(ROOT / "tests/test_prerequisite_protocol.py")
    if any(path.is_symlink() for path in files):
        raise ValueError("Manifest payload must contain regular repository files")
    return {
        "protocol_id": "financial-prerequisites-v1",
        "packages": {"financial-prerequisites": "0.1.0",
                     "financial-portfolio-staged": "0.4.0",
                     "financial-portfolio": "0.3.0",
                     "financial-research": "0.2.0", "verifiers": "0.4.1"},
        "predecessor_manifest_sha256": digest(PREDECESSOR),
        "comparison": {"case_ids": CASE_IDS, "model_count": 2,
                       "attempts_per_task_per_model": 1,
                       "expected_attempts_per_model": 4, "expected_attempts": 8},
        "advancement": {"required_passes": 4, "required_attempts": 4,
                        "saved_policy_correct": True,
                        "source_replays": ["original", "alternate"],
                        "allowed_missing_unknown_or_infrastructure": 0,
                        "portfolio_split": "train", "portfolio_seed_start": 4,
                        "portfolio_num_tasks": 4},
        "sha256": {path.relative_to(ROOT).as_posix(): digest(path)
                   for path in sorted(files)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Freeze a new manifest; never overwrite")
    mode.add_argument("--check", action="store_true", help="Verify exact manifest and file inventory")
    args = parser.parse_args()
    expected = build_manifest()
    if args.write:
        with MANIFEST.open("x") as handle:
            handle.write(json.dumps(expected, indent=2) + "\n")
        print("Prerequisite protocol manifest written.")
    else:
        if json.loads(MANIFEST.read_text()) != expected:
            raise SystemExit("Prerequisite manifest does not match its frozen payload.")
        print("Prerequisite protocol manifest verified.")


if __name__ == "__main__":
    main()
