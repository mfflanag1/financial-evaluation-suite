"""Freeze or verify the separate split-convention clarification condition."""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
PACKAGE = ROOT / "environments/financial_portfolio_clarified"
MANIFEST = CONFIG / "manifest.json"
STAGED = ROOT / "configs/financial_portfolio/staged_v1/manifest.json"
PREREQUISITE = ROOT / "configs/financial_prerequisites/prerequisite_v1/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_files(directory):
    return {path for path in directory.rglob("*") if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def verify_predecessors():
    calibration = ROOT / "configs/financial_portfolio/calibration_v1/manifest.json"
    for successor, predecessor in ((STAGED, calibration), (PREREQUISITE, STAGED)):
        manifest = json.loads(successor.read_text())
        if manifest["predecessor_manifest_sha256"] != digest(predecessor):
            raise ValueError(f"Frozen predecessor manifest changed: {predecessor.relative_to(ROOT)}")
    for path in (calibration, STAGED, PREREQUISITE):
        for relative, expected in json.loads(path.read_text())["sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Frozen predecessor file changed: {relative}")


def build_manifest():
    verify_predecessors()
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]
    if (project["name"], project["version"]) != ("financial-portfolio-clarified", "0.5.0"):
        raise ValueError("Unexpected clarification package version")
    for relative in ("README.md", "src/financial_portfolio_clarified/__init__.py",
                     "src/financial_portfolio_clarified/prime_adapter.py"):
        if not (PACKAGE / relative).is_file():
            raise ValueError(f"Incomplete clarification package: {relative}")
    files = payload_files(PACKAGE) | payload_files(CONFIG)
    files.discard(MANIFEST)
    files.update(ROOT.glob("tests/test_clarified*.py"))
    if not (ROOT / "tests/test_clarified_protocol.py").is_file():
        raise ValueError("Missing clarification protocol tests")
    if any(path.is_symlink() for path in files):
        raise ValueError("Manifest payload must contain regular repository files")
    return {
        "protocol_id": "portfolio-clarified-v1",
        "packages": {"financial-portfolio-clarified": "0.5.0",
                     "financial-portfolio-staged": "0.4.0",
                     "financial-portfolio": "0.3.0", "financial-research": "0.2.0",
                     "verifiers": "0.4.1"},
        "predecessor_manifest_sha256": digest(STAGED),
        "prerequisite_manifest_sha256": digest(PREREQUISITE),
        "comparison": {
            "design": "descriptive_historical_pair",
            "baseline_protocol_id": "portfolio-staged-v1",
            "baseline_source_commit": "339eb6677a40fb989fe1aea80e8b0678bdf02f21",
            "model": "moonshotai/kimi-k2-0905",
            "split": "train", "seed_start": 4, "num_tasks": 4,
            "attempts_per_task": 1, "expected_attempts": 4,
            "eligible_attempts": "all_scheduled_attempts",
            "changed_solver_input": "appended_split_convention_system_paragraph",
            "metric_order": ["attempt_accounting", "checkpoint_completion",
                             "source_reproduction", "full_credit", "financial_reward",
                             "audit_accuracy", "trial_accuracy", "decision_quality",
                             "split_sensitive_return_diagnostic",
                             "split_direction_evidence"],
            "split_diagnostic": {
                "role": "primary_objective_diagnostic_of_targeted_clarification",
                "requests_per_episode": [
                    {"entity_id": "E02", "day": 36, "field": "forward_return"},
                    {"entity_id": "E05", "day": 74, "field": "forward_return"},
                ],
                "expected_observations_per_condition": 8,
                "expected_source": "saved_public_prices_and_securities_csv",
                "ticker_mapping": "inclusive_validity_intervals_at_each_day",
                "adjusted_close": "close * shares_per_original",
                "forward_return": "adjusted_close(d+1) / adjusted_close(d) - 1",
                "absolute_tolerance": 1e-8,
                "historical_correct_counts_by_seed": [0, 0, 2, 0],
                "historical_total_correct": 2,
                "candidate_source_execution": False,
                "source_reading": "secondary_evidence_only",
            },
            "repair_or_rerun": False,
            "large_calibration_included": False,
        },
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
        print("Clarification protocol manifest written.")
    else:
        if json.loads(MANIFEST.read_text()) != expected:
            raise SystemExit("Clarification manifest does not match its frozen payload.")
        print("Clarification protocol manifest verified.")


if __name__ == "__main__":
    main()
