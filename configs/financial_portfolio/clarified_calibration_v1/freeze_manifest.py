"""Freeze or verify the 64-attempt clarified development calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
MANIFEST = CONFIG / "manifest.json"
CALIBRATION = ROOT / "configs/financial_portfolio/calibration_v1/manifest.json"
STAGED = ROOT / "configs/financial_portfolio/staged_v1/manifest.json"
PREREQUISITE = ROOT / "configs/financial_prerequisites/prerequisite_v1/manifest.json"
CLARIFIED = ROOT / "configs/financial_portfolio/clarified_v1/manifest.json"
TEST = ROOT / "tests/test_portfolio_clarified_calibration_protocol.py"
FAMILIES = ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_predecessors():
    """Check the complete frozen source chain and return its bound payload."""
    for successor, predecessor, key in (
            (STAGED, CALIBRATION, "predecessor_manifest_sha256"),
            (PREREQUISITE, STAGED, "predecessor_manifest_sha256"),
            (CLARIFIED, STAGED, "predecessor_manifest_sha256"),
            (CLARIFIED, PREREQUISITE, "prerequisite_manifest_sha256")):
        if json.loads(successor.read_text())[key] != digest(predecessor):
            raise ValueError(f"Frozen predecessor manifest changed: {predecessor.relative_to(ROOT)}")
    files = {}
    for manifest_path in (CALIBRATION, STAGED, PREREQUISITE, CLARIFIED):
        for relative, expected in json.loads(manifest_path.read_text())["sha256"].items():
            path = ROOT / relative
            if path.is_symlink() or digest(path) != expected:
                raise ValueError(f"Frozen predecessor file changed: {relative}")
            files[relative] = expected
        files[manifest_path.relative_to(ROOT).as_posix()] = digest(manifest_path)
    return files


def build_manifest():
    files = verify_predecessors()
    config = tomllib.loads((CONFIG / "development.toml").read_text())
    if config["env"]["taskset"] != {"id": "financial-portfolio-clarified", "split": "dev",
                                     "seed_start": 128, "num_tasks": 32}:
        raise ValueError("Unexpected calibration task schedule")
    if config["num_rollouts"] != 2 or config["select"] != {"shuffle": False, "limit": 32}:
        raise ValueError("Unexpected calibration attempt schedule")
    payload = {path for path in CONFIG.rglob("*") if path.is_file()
               and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    payload.discard(MANIFEST)
    payload.add(TEST)
    for path in payload:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Manifest payload must contain regular repository files")
        files[path.relative_to(ROOT).as_posix()] = digest(path)
    return {
        "protocol_id": "portfolio-clarified-calibration-v1",
        "packages": json.loads(CLARIFIED.read_text())["packages"],
        "predecessor_manifest_sha256": digest(CLARIFIED),
        "schedule_predecessor_manifest_sha256": digest(CALIBRATION),
        "model": "moonshotai/kimi-k2-0905",
        "schedule": {"split": "dev", "seed_start": 128, "num_tasks": 32,
                     "attempts_per_task": 2, "expected_attempts": 64,
                     "episodes_per_family": 8, "attempts_per_family": 16,
                     "family_rotation": list(FAMILIES),
                     "reserved_test_seeds": [160, 191], "reserved_test_selected": False},
        "analysis": {"eligible_attempts": "all_scheduled_attempts",
                     "aggregation_unit": "episode_mean_of_both_attempts",
                     "episode_mean_distribution": ["mean", "median", "minimum", "maximum", "quartiles"],
                     "repeat_stability": ["absolute_reward_gap", "both_one_neither_full_credit",
                                          "both_one_neither_reproduction",
                                          "decision_and_selection_agreement_on_two_valid_scored_policies"],
                     "unresolved_attempts_block_aggregate": True,
                     "candidate_failure_reward": 0,
                     "continuous_reward_pass_at_k": False,
                     "repair_or_replacement_rerun": False,
                     "pilot_before_after_comparison": False},
        "sha256": dict(sorted(files.items())),
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
        print("Clarified calibration manifest written.")
    else:
        if json.loads(MANIFEST.read_text()) != expected:
            raise SystemExit("Clarified calibration manifest does not match its frozen payload.")
        print("Clarified calibration manifest verified.")


if __name__ == "__main__":
    main()
