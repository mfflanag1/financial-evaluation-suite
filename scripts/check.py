"""Run explicit verification tiers and emit a machine-readable local report.

Native checks fail on missing optional dependencies. Docker checks additionally
enable every opt-in container suite and forbid skipped tests.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCKER_FLAGS = (
    "RUN_WORKFLOW_DOCKER", "RUN_RESEARCH_DOCKER_TESTS", "RUN_STAGED_DOCKER_TESTS",
    "RUN_PREREQUISITE_DOCKER_TESTS", "RUN_RUNTIME_DOCKER_TESTS",
)


class ReleaseResult(unittest.TextTestResult):
    """Count successful methods separately from failing/skipped subtest events."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_methods = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successful_methods.append(test.id())


def run_module(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-B", "-m", module, *args], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("core", "native", "docker"), default="core")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--report", type=Path, help="Write a JSON test summary outside the checkout")
    args = parser.parse_args()
    if args.report and args.report.resolve().is_relative_to(ROOT):
        parser.error("Keep run reports outside the checkout")
    if args.report and (args.report.exists() or args.report.is_symlink()):
        parser.error("Report destination already exists")
    if args.tier != "core":
        if sys.version_info[:2] != (3, 12):
            parser.error("Native release validation is pinned to Python 3.12")
        try:
            version = importlib.metadata.version("verifiers")
        except importlib.metadata.PackageNotFoundError:
            parser.error("Install the local packages and pinned native dependencies first")
        if version != "0.4.1":
            parser.error("Native release validation requires verifiers==0.4.1")
    if args.tier == "docker":
        subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True)
    for flag in DOCKER_FLAGS:
        if args.tier == "docker":
            os.environ[flag] = "1"
        else:
            os.environ.pop(flag, None)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "private_evaluator"))
    for action in ("validate", "controls"):
        subprocess.run([sys.executable, "-B", "private_evaluator/grade.py", action], check=True)
    run_module("scoring_v2.grade", "validate")
    run_module("scoring_v3.grade", "validate")
    run_module("quality.inventory", "--check")
    run_module("quality.discrete", "--check")
    run_module("distribution.inventory", "--check", "distribution/source_inventory.json")
    subprocess.run([sys.executable, "-B", "scripts/catalog.py", "--check"], check=True)
    if args.tier != "core":
        subprocess.run([sys.executable, "-B", "scripts/check_installed.py"], check=True)
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReleaseResult).run(tests)
    skips = [{"test": test.id(), "reason": reason} for test, reason in result.skipped]
    disallowed = skips if args.tier == "docker" else (
        [row for row in skips if not any(flag in row["reason"] for flag in DOCKER_FLAGS)
         and row["reason"] != "Opt-in Docker integration"] if args.tier == "native" else [])
    passed = result.wasSuccessful() and result.testsRun > 0 and not disallowed
    report = {
        "schema_version": 1, "tier": args.tier, "pattern": args.pattern,
        "python": sys.version.split()[0], "tests_run": result.testsRun,
        "passed": len(result.successful_methods),
        "counting_unit": "test methods for tests_run/passed; recorded events for failures/errors/skips",
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors], "skipped": skips,
        "expected_failures": [test.id() for test, _ in result.expectedFailures],
        "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
        "disallowed_skips": disallowed, "successful": passed,
        "claims": {"model_performance_measured": False, "optimizer_step_validated": False},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
