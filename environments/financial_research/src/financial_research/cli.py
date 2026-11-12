"""Local public export and trusted grading, without model credentials."""

import argparse
import json
import os
from pathlib import Path
import secrets

from .generator import FAMILIES, SPLITS, build_episode, resolve_episode
from .verifier import reference_submission, score_submission


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON number")


def read_policy(path):
    with Path(path).open("rb") as stream:
        content = stream.read(2 * 1024 * 1024 + 1)
    if len(content) > 2 * 1024 * 1024:
        return None
    try:
        return json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    key = sub.add_parser("init-key", help="Create an evaluator-only key file; never prints its contents")
    key.add_argument("--out", type=Path, required=True)
    export = sub.add_parser("export", help="Write public inputs to a new or empty directory")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--split", choices=SPLITS, default="train")
    export.add_argument("--family", choices=FAMILIES)
    export.add_argument("--out", type=Path, required=True)
    grade = sub.add_parser("grade", help="Grade a structured policy in the trusted host")
    grade.add_argument("task_id")
    grade.add_argument("policy", type=Path)
    check = sub.add_parser("self-check", help="Run model-free reference and negative controls")
    check.add_argument("--count", type=int, default=12)
    report = sub.add_parser("report", help="Account for every saved rollout, including failed and missing attempts")
    report.add_argument("input", type=Path)
    report.add_argument("--expected", type=int)
    args = parser.parse_args()
    if args.command == "init-key":
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_hex(32) + "\n")
        print("Evaluator key created. Keep it outside solver inputs and the repository.")
    elif args.command == "export":
        episode = build_episode(args.seed, args.split, args.family)
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            parser.error("output directory must be new or empty")
        args.out.mkdir(parents=True, exist_ok=True)
        for name, content in episode["public_files"].items():
            (args.out / name).write_text(content)
        print(json.dumps({"id": episode["id"], "prompt": episode["prompt"]}, indent=2))
    elif args.command == "grade":
        print(json.dumps(score_submission(resolve_episode(args.task_id), read_policy(args.policy)), indent=2))
    elif args.command == "report":
        from .accounting import report_path
        result = report_path(args.input, expected=args.expected)
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["performance_usable"]:
            raise SystemExit(2)
    else:
        if not 1 <= args.count <= 128:
            parser.error("count must be between 1 and 128")
        report = []
        for seed in range(args.count):
            episode = build_episode(seed)
            submission = reference_submission(episode)
            reference = score_submission(episode, submission)
            assert reference["metrics"]["audit_accuracy"] == 1
            assert reference["metrics"]["trial_accuracy"] == 1
            assert reference["metrics"]["decision_correct"] == 1
            assert score_submission(episode, {})["reward"] == 0
            report.append({"id": episode["id"], "decision": submission["decision"],
                           "reference_reward": reference["reward"]})
        print(json.dumps({"kind": "model-free reference controls", "cases": report}, indent=2))


if __name__ == "__main__":
    main()
