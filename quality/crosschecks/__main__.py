"""Read-only CLI for selected public-contract crosschecks."""
import argparse
import json

from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derive-only", action="store_true", help="Do not import or compare primary arithmetic audit modules")
    parser.add_argument("--check", action="store_true", help="Print only a compact result summary")
    args = parser.parse_args()
    report = run(compare_primary=not args.derive_only)
    displayed = {key: report[key] for key in ("version", "status", "case_count", "output_count")} if args.check else report
    print(json.dumps(displayed, indent=2, sort_keys=True, allow_nan=False))
    return int(report["status"] == "primary_audit_disagreement")


if __name__ == "__main__":
    raise SystemExit(main())
