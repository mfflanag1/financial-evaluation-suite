"""Export reviewed source evidence with the matching v3 public contracts.

This is a separate packet format. It leaves the original source-packet export
unchanged and resolves all inputs from the checkout containing this module.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

from .export import packet_contents as frozen_packet_contents
from .inventory import CASE_ROOT, DEFAULT_ROOT, DistributionError, digest, json_bytes, safe_file

FORMAT = "source-contract-v3"
MANIFEST = "corrected_packet_manifest.json"


def _current_contents() -> dict[str, bytes]:
    """Fresh-process implementation; callers use packet_contents, not this helper."""
    from scoring_v3 import constraints, source
    from scoring_v3.common import ROOT, VERSION, binding, verify

    if Path(ROOT).resolve() != DEFAULT_ROOT.resolve():
        raise DistributionError("scorer_repository_mismatch")
    try:
        verify()
        scorer_binding = binding()
    except (OSError, ValueError, KeyError, TypeError):
        raise DistributionError("scorer_binding_invalid") from None
    frozen = frozen_packet_contents(DEFAULT_ROOT, "source-packet")
    original_manifest = json.loads(frozen["packet_manifest.json"])
    case_ids = original_manifest["included_case_ids"]
    files = {path: data for path, data in frozen.items()
             if path.startswith(CASE_ROOT + "/") or path == "source_inventory.json"}
    for cid in case_ids:
        relative = f"{CASE_ROOT}/{cid}/case.json"
        question = constraints.annotate(json.loads(frozen[relative]), cid)
        question["scoring_corrections"]["source_citation_rule"] = source.DUPLICATE_RULE
        question.update(scorer_version=VERSION, scorer_binding=scorer_binding)
        files[relative] = json_bytes(question)
    availability = json.loads(frozen["case_availability.json"])
    availability["mode"] = FORMAT
    for case in availability["cases"]:
        if case["included"]:
            case["reason"] = "reviewed_sources_and_versioned_public_contract"
    files["case_availability.json"] = json_bytes(availability)
    files["README.txt"] = (
        "Corrected source contracts and reviewed evidence\n\n"
        f"Format: {FORMAT}\nScorer: {VERSION}\nBinding: {scorer_binding}\n\n"
        "Each questions/legacy/source_grounded/cases/<id>/case.json is the v3 public contract.\n"
        "Give the solver that contract, sources.json and the source files in the same case directory.\n"
        "Follow its response_format, including integer selectors and citation requirements.\n"
        "Source excerpts and sources.json retain their exact reviewed bytes.\n"
        "Question contracts add versioned scoring instructions to the frozen public questions.\n"
        "This format is distinct from the original frozen source-packet export.\n"
        "Grade the response separately on a trusted host with the matching scorer binding:\n"
        "  python3 -B -m scoring_v3.grade grade <case-id> response.json\n\n"
        "This is a seven-case source subset, not the full suite or a standalone grader.\n"
        "See case_availability.json for included and omitted cases.\n"
        "Source attribution, publication dates and publisher-policy references are in source_inventory.json.\n"
        "Eligibility applies only to exact reviewed excerpts, not complete linked documents or legal clearance.\n"
        "The project remains unlicensed; source permissions do not license authored questions or code.\n"
        "No answers, evaluator keys, controls or future interaction states are included.\n"
        "Answers are already exposed in the public repository; this is not an unseen benchmark.\n"
        "Keep the full checkout and evaluator artifacts outside the solver environment.\n"
    ).encode()
    # Detect a changing scorer rather than stamping a mixture of versions.
    if binding() != scorer_binding:
        raise DistributionError("scorer_changed_during_export")
    files[MANIFEST] = json_bytes({
        "schema_version": 1,
        "packet_type": FORMAT,
        "scorer_version": VERSION,
        "scorer_binding": scorer_binding,
        "source_policy_id": original_manifest["policy_id"],
        "source_policy_sha256": original_manifest["policy_sha256"],
        "frozen_source_packet_manifest_sha256": digest(frozen["packet_manifest.json"]),
        "included_case_ids": case_ids,
        "files": {path: {"sha256": digest(data), "bytes": len(data)}
                  for path, data in sorted(files.items())},
    })
    return files


def packet_contents() -> dict[str, bytes]:
    """Build in a fresh interpreter so cached scorer modules cannot mislabel rules.

    A library caller may have imported an older scorer before changing checkout
    revisions. Reloading selected modules is insufficient: their dependencies
    also cache rules. Both source verification and contract construction therefore
    happen in one isolated interpreter using this checkout explicitly.
    """
    script = (
        "import base64,json,sys; sys.path.insert(0,sys.argv[1]); "
        "from distribution.corrected import _current_contents; "
        "print(json.dumps({p:base64.b64encode(b).decode('ascii') "
        "for p,b in _current_contents().items()}))"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(DEFAULT_ROOT.resolve())],
            stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=60,
        )
        payload = json.loads(process.stdout)
        return {path: base64.b64decode(data, validate=True) for path, data in payload.items()}
    except (subprocess.SubprocessError, OSError, ValueError, TypeError, AttributeError):
        raise DistributionError("fresh_packet_build_failed") from None


def export_packet(output_dir: Path) -> dict:
    """Create a new external directory, writing its manifest only after inputs."""
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        raise DistributionError("output_must_be_new")
    output = output.resolve()
    if output.is_relative_to(DEFAULT_ROOT.resolve()):
        raise DistributionError("output_must_be_outside_repository")
    if not output.parent.is_dir():
        raise DistributionError("output_parent_unavailable")
    files = packet_contents()
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise DistributionError("output_must_be_new") from None
    for relative in sorted(files, key=lambda p: (p == MANIFEST, p)):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(files[relative])
    return verify_packet(output)


def verify_packet(packet_dir: Path) -> dict:
    """Reconstruct expected bytes; never trust a candidate-supplied manifest."""
    packet = Path(packet_dir)
    if packet.is_symlink() or not packet.is_dir():
        raise DistributionError("packet_directory_unavailable")
    packet = packet.resolve()
    expected = packet_contents()
    directories = {parent.as_posix() for relative in expected
                   for parent in Path(relative).parents if parent != Path(".")}
    found = set()
    for path in packet.rglob("*"):
        if path.is_symlink():
            raise DistributionError("symlink_not_allowed")
        relative = path.relative_to(packet).as_posix()
        if path.is_file():
            found.add(relative)
        elif path.is_dir():
            if relative not in directories:
                raise DistributionError("packet_membership_mismatch")
        else:
            raise DistributionError("special_file_not_allowed")
    if found != set(expected):
        raise DistributionError("packet_membership_mismatch")
    for relative, data in expected.items():
        if safe_file(packet, relative).read_bytes() != data:
            raise DistributionError("packet_bytes_mismatch")
    manifest = json.loads(expected[MANIFEST])
    return {
        "status": "ok", "packet_type": FORMAT, "files": len(expected),
        "included_case_ids": manifest["included_case_ids"],
        "scorer_version": manifest["scorer_version"],
        "scorer_binding": manifest["scorer_binding"],
        "packet_manifest_sha256": digest(expected[MANIFEST]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("packet", type=Path)
    args = parser.parse_args(argv)
    try:
        result = export_packet(args.out) if args.command == "create" else verify_packet(args.packet)
    except (DistributionError, OSError, ValueError, KeyError, TypeError):
        # A failed source/scorer verification can contain local paths or content.
        print(json.dumps({"status": "blocked", "code": "corrected_packet_verification_failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
