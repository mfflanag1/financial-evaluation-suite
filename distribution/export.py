"""Export link metadata or explicitly reviewed solver source packets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import (
    CASE_ROOT, DEFAULT_ROOT, POLICY_PATH, DistributionError, build_inventory,
    digest, json_bytes, read_json, safe_file,
)

MODES = ("links", "source-packet")


def packet_contents(repo_root: Path, mode: str = "links") -> dict[str, bytes]:
    """Build an allowlisted packet in memory; never export evaluator state."""
    if mode not in MODES:
        raise DistributionError("unsupported_export_mode")
    root = Path(repo_root).resolve()
    inventory = build_inventory(root)
    policy = read_json(root, POLICY_PATH)
    files = {}
    included = []
    if mode == "source-packet":
        for path, expected in sorted(policy["packet_files"].items()):
            # Restrict the policy to public question packets even if it is edited.
            if not path.startswith(CASE_ROOT + "/"):
                raise DistributionError("non_public_packet_file")
            data = safe_file(root, path).read_bytes()
            if digest(data) != expected:
                raise DistributionError("packet_file_binding_mismatch")
            files[path] = data
        included = [c["id"] for c in inventory["cases"] if c["packet_eligible"]]
    files["source_inventory.json"] = json_bytes(inventory)
    files["case_availability.json"] = json_bytes({
        "schema_version": 1,
        "mode": mode,
        "included_case_ids": included,
        "cases": [{
            "id": c["id"],
            "included": c["id"] in included,
            "source_ids": c["source_ids"],
            "reason": ("reviewed_question_and_source_bytes" if c["id"] in included
                       else "metadata_only_mode" if mode == "links"
                       else "source_rights_review_pending"),
            "blocked_source_ids": c["blocked_source_ids"],
        } for c in inventory["cases"]],
    })
    files["README.txt"] = (
        "Source packet export\n\n"
        f"Mode: {mode}\n"
        "This is a subset of the source-grounded track, not the complete evaluation suite.\n"
        "Links mode contains metadata only: no questions, evidence text or answers.\n"
        "Source-packet mode includes only exact reviewed question and evidence bytes.\n"
        "See case_availability.json for included and omitted cases.\n"
        "A source URL is a locator, not a license or a guaranteed historical snapshot.\n"
        "The exporter does not download or reconstruct omitted source text.\n"
        "Evidence is an excerpt, not necessarily the complete original document.\n"
        "Source attribution, publication dates and publisher-policy references are in source_inventory.json.\n"
        "Publisher-policy eligibility is limited to the reviewed excerpts, not legal clearance.\n"
        "No evaluator keys, answers, later interaction states or reserved-case packets are included.\n"
        "Answers and other evaluator materials exist in the public repository; these cases are exposed.\n"
        "An answer-free export does not create an unseen benchmark or erase existing public copies.\n"
        "Use an isolated solver directory; grade separately using the labelled corrected scorer.\n"
    ).encode()
    files["packet_manifest.json"] = json_bytes({
        "schema_version": 1,
        "packet_type": "source_distribution",
        "mode": mode,
        "policy_id": inventory["policy_id"],
        "policy_sha256": inventory["policy_sha256"],
        "included_case_ids": included,
        "files": {path: {"sha256": digest(data), "bytes": len(data)} for path, data in sorted(files.items())},
    })
    return files


def export_packet(repo_root: Path, output_dir: Path, mode: str = "links") -> dict:
    """Write only to a new directory outside the source checkout.

    A complete packet has a manifest written last. A failed write leaves an
    incomplete directory for inspection; it is never reused or overwritten.
    """
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        raise DistributionError("output_must_be_new")
    output = output.resolve()
    if output.is_relative_to(root):
        raise DistributionError("output_must_be_outside_repository")
    if not output.parent.is_dir():
        raise DistributionError("output_parent_unavailable")
    files = packet_contents(root, mode)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise DistributionError("output_must_be_new") from None
    for relative in sorted(files, key=lambda p: (p == "packet_manifest.json", p)):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(files[relative])
    return verify_packet(root, output)


def verify_packet(repo_root: Path, packet_dir: Path) -> dict:
    """Rebuild expected bytes; reject extras, missing files, symlinks or edits."""
    root = Path(repo_root).resolve()
    packet = Path(packet_dir)
    if packet.is_symlink() or not packet.is_dir():
        raise DistributionError("packet_directory_unavailable")
    packet = packet.resolve()
    manifest = read_json(packet, "packet_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("mode") not in MODES:
        raise DistributionError("invalid_packet_manifest")
    expected = packet_contents(root, manifest["mode"])
    expected_directories = {parent.as_posix() for relative in expected
                            for parent in Path(relative).parents if parent != Path(".")}
    found = set()
    for path in packet.rglob("*"):
        if path.is_symlink():
            raise DistributionError("symlink_not_allowed")
        if path.is_file():
            found.add(path.relative_to(packet).as_posix())
        elif path.is_dir():
            if path.relative_to(packet).as_posix() not in expected_directories:
                raise DistributionError("packet_membership_mismatch")
        else:
            raise DistributionError("special_file_not_allowed")
    if found != set(expected):
        raise DistributionError("packet_membership_mismatch")
    for relative, data in expected.items():
        if safe_file(packet, relative).read_bytes() != data:
            raise DistributionError("packet_bytes_mismatch")
    return {
        "status": "ok",
        "mode": manifest["mode"],
        "files": len(expected),
        "included_case_ids": manifest["included_case_ids"],
        "packet_manifest_sha256": digest(expected["packet_manifest.json"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--mode", choices=MODES, default="links")
    verify = sub.add_parser("verify")
    verify.add_argument("packet", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (export_packet(args.repo, args.out, args.mode) if args.command == "create"
                  else verify_packet(args.repo, args.packet))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (DistributionError, OSError) as exc:
        code = str(exc) if isinstance(exc, DistributionError) else "filesystem_error"
        print(json.dumps({"status": "blocked", "code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
