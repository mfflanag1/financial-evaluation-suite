"""Model-free export, scoring, controls and failure accounting."""

import argparse
import json
from pathlib import Path

from financial_research.accounting import report_path
from financial_research.cli import read_policy
from financial_research.verifier import reference_submission, score_submission

from .generator import FAMILIES, SPLITS, build_episode, resolve_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--split", choices=SPLITS, default="train")
    export.add_argument("--family", choices=FAMILIES)
    export.add_argument("--out", required=True, type=Path)
    grade = commands.add_parser("grade")
    grade.add_argument("task_id")
    grade.add_argument("policy", type=Path)
    check = commands.add_parser("self-check")
    check.add_argument("--count", type=int, default=4)
    report = commands.add_parser("report")
    report.add_argument("input", type=Path)
    report.add_argument("--expected", type=int)
    args = parser.parse_args()
    if args.command == "export":
        episode = build_episode(args.seed, args.split, args.family)
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            parser.error("output directory must be new or empty")
        args.out.mkdir(parents=True, exist_ok=True)
        for name, content in episode["public_files"].items():
            (args.out / name).write_text(content, encoding="utf-8")
        print(json.dumps({"id": episode["id"], "prompt": episode["prompt"]}, indent=2))
    elif args.command == "grade":
        print(json.dumps(score_submission(resolve_episode(args.task_id), read_policy(args.policy)), indent=2))
    elif args.command == "report":
        result = report_path(args.input, expected=args.expected)
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["performance_usable"]:
            raise SystemExit(2)
    else:
        if not 1 <= args.count <= 128:
            parser.error("count must be between 1 and 128")
        results = []
        for seed in range(args.count):
            episode = build_episode(seed)
            submission = reference_submission(episode)
            score = score_submission(episode, submission)
            if not all(score["metrics"][metric] == 1 for metric in ("audit_accuracy", "trial_accuracy", "decision_correct")):
                raise RuntimeError("reference control failed")
            if score_submission(episode, {})["reward"] != 0:
                raise RuntimeError("negative control failed")
            results.append({"id": episode["id"], "family": FAMILIES[seed % 4],
                            "selected": submission["selected"], "reference_reward": score["reward"],
                            **episode["private"]["diagnostics"]})
        print(json.dumps({"kind": "model-free reference controls", "cases": results}, indent=2))


if __name__ == "__main__":
    main()
