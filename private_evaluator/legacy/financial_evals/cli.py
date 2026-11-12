"""Run with python -m financial_evals.cli from the evaluation project."""

import argparse
import json
from pathlib import Path

from .core import ContractError, export_actor, grade, load_case, read_json


def check_controls(root, case_id, controls):
    if (not isinstance(controls, list) or not all(isinstance(c, dict) for c in controls)
            or sum(c.get("expected_automated_pass") is True for c in controls) < 1
            or sum(c.get("expected_automated_pass") is False for c in controls) < 3):
        raise ContractError(f"Need at least one passing and three failing controls: {case_id}")
    results = []
    for control in controls:
        expected = control.get("expected_failures")
        if not isinstance(expected, dict):
            raise ContractError(f"Control needs explicit expected_failures: {case_id}/{control.get('name')}")
        actual = grade(root, case_id, control["response"])
        failures = {f["id"]: f["failures"] for f in actual["fields"] if f["failures"]}
        if not actual["fields"]:
            failures = {"__response__": actual["critical_errors"]}
        matches = (actual["automated_pass"] is control["expected_automated_pass"] and failures == expected)
        results.append({"case_id": case_id, "control": control["name"],
                        "expected": control["expected_automated_pass"], "actual": actual["automated_pass"],
                        "matches": matches, "failures": failures,
                        "critical_errors": actual["critical_errors"]})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("controls")
    exp = sub.add_parser("export")
    exp.add_argument("case_id")
    exp.add_argument("destination", type=Path)
    score = sub.add_parser("grade")
    score.add_argument("case_id")
    score.add_argument("response", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "export":
            result = export_actor(root, args.case_id, args.destination)
        elif args.command == "grade":
            load_case(root, args.case_id)  # Broken evaluator data is not a solver failure.
            try:
                response = read_json(args.response)
            except (ValueError, UnicodeError, RecursionError) as exc:
                result = {"case_id": args.case_id, "automated_pass": False,
                          "critical_errors": ["invalid_response_json"], "fields": [],
                          "response_error": type(exc).__name__}
            else:
                result = grade(root, args.case_id, response)
        else:
            cases = sorted((root / "cases").glob("*.json"))
            if not cases:
                raise ContractError("No implemented case files")
            result = []
            for path in cases:
                case, _, _ = load_case(root, path.stem)
                if args.command == "validate":
                    result.append({"case_id": case["id"], "valid": True,
                                   "review_status": case.get("review_status", {})})
                else:
                    controls = read_json(root / "controls" / path.name)
                    result.extend(check_controls(root, path.stem, controls))
            if args.command == "controls" and not all(r["matches"] for r in result):
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
        print(json.dumps(result, indent=2))
    except (ContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Contract error: {exc}\n")


if __name__ == "__main__":
    main()
