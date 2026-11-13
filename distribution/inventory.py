"""Inventory source evidence without including source text or evaluator answers."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

SOURCE_ROOT = "private_evaluator/legacy/source_grounded/sources"
CASE_ROOT = "questions/legacy/source_grounded/cases"
POLICY_PATH = "distribution/source_policy.json"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class DistributionError(ValueError):
    """A fixed diagnostic code; never includes source content or credentials."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def safe_file(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks, including symlinked parent directories."""
    if not isinstance(relative, str) or "\\" in relative:
        raise DistributionError("unsafe_path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {".", ".."} for p in parts.parts):
        raise DistributionError("unsafe_path")
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise DistributionError("symlink_not_allowed")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise DistributionError("required_file_unavailable")
    return path


def read_json(root: Path, relative: str):
    try:
        return json.loads(safe_file(root, relative).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DistributionError("invalid_json") from None


def _token(value: object) -> str:
    if not isinstance(value, str) or TOKEN.fullmatch(value) is None:
        raise DistributionError("invalid_identifier")
    return value


def _url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
        raise DistributionError("invalid_source_url")
    try:
        url = urlsplit(value)
        valid = url.scheme == "https" and url.hostname and not url.username and not url.password
    except ValueError:
        valid = False
    if not valid:
        raise DistributionError("invalid_source_url")
    return value


def _files(root: Path, relative: str) -> list[str]:
    base = root / relative
    if not base.is_dir() or base.is_symlink():
        raise DistributionError("source_directory_unavailable")
    output = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise DistributionError("symlink_not_allowed")
        if path.is_file():
            output.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise DistributionError("special_file_not_allowed")
    return output


def publisher_policies(value: object) -> dict:
    """Allow only reviewed public terms metadata, never arbitrary policy fields."""
    fields = {"basis_summary", "observed_on", "review_method", "reviewed_scope", "url"}
    if not isinstance(value, dict) or not value:
        raise DistributionError("invalid_publisher_policy")
    result = {}
    for key, item in value.items():
        _token(key)
        if (not isinstance(item, dict) or set(item) != fields
                or any(not isinstance(v, str) or not v or len(v) > 4096 for v in item.values())):
            raise DistributionError("invalid_publisher_policy")
        try:
            observed = date.fromisoformat(item["observed_on"])
        except ValueError:
            raise DistributionError("invalid_publisher_policy") from None
        if observed.isoformat() != item["observed_on"]:
            raise DistributionError("invalid_publisher_policy")
        result[key] = {name: item[name] for name in sorted(fields)}
        result[key]["url"] = _url(item["url"])
    return result


def build_inventory(repo_root: Path = DEFAULT_ROOT) -> dict:
    """Return deterministic source/file/case accounting; fail on stale bindings.

    The policy is a reviewed repository artifact, not an external access control.
    It must be reviewed alongside code changes. No network requests occur here.
    """
    root = Path(repo_root).resolve()
    policy = read_json(root, POLICY_PATH)
    if not isinstance(policy, dict) or type(policy.get("schema_version")) is not int or policy.get("schema_version") != 1 or policy.get("default") != "link_only":
        raise DistributionError("invalid_policy")
    decisions = policy.get("sources")
    packet_files = policy.get("packet_files")
    if not isinstance(decisions, dict) or not isinstance(packet_files, dict):
        raise DistributionError("invalid_policy")
    policy_id = _token(policy.get("policy_id"))
    publishers = publisher_policies(policy.get("publisher_policies"))
    found_files = _files(root, SOURCE_ROOT)
    manifests = [p for p in found_files if p.endswith("/manifest.json")]
    sources = {}
    expected_source_files = set()
    for relative in manifests:
        manifest = read_json(root, relative)
        if not isinstance(manifest, dict):
            raise DistributionError("invalid_source_manifest")
        sid = _token(manifest.get("id"))
        if sid in sources or relative != f"{SOURCE_ROOT}/{sid}/manifest.json":
            raise DistributionError("duplicate_or_misplaced_source")
        evidence_path = f"{SOURCE_ROOT}/{sid}/evidence.txt"
        expected_source_files.update([relative, evidence_path])
        evidence = safe_file(root, evidence_path).read_bytes()
        evidence_hash = digest(evidence)
        if (manifest.get("raw_path") != f"sources/{sid}/evidence.txt"
                or manifest.get("text_path") != f"sources/{sid}/evidence.txt"
                or manifest.get("raw_sha256") != evidence_hash
                or manifest.get("text_sha256") != evidence_hash):
            raise DistributionError("source_evidence_binding_mismatch")
        decision = decisions.get(sid)
        if not isinstance(decision, dict):
            raise DistributionError("source_policy_review_required")
        if (decision.get("manifest_sha256") != digest(safe_file(root, relative).read_bytes())
                or decision.get("evidence_sha256") != evidence_hash
                or decision.get("url") != manifest.get("url")):
            raise DistributionError("source_policy_binding_mismatch")
        disposition = decision.get("disposition")
        if disposition not in {"eligible_exact_excerpt", "link_only"}:
            raise DistributionError("invalid_source_disposition")
        if type(decision.get("legacy_restriction_flag")) is not bool:
            raise DistributionError("invalid_policy")
        terms_key = decision.get("terms_key")
        if terms_key is not None and not isinstance(terms_key, str):
            raise DistributionError("invalid_publisher_policy")
        if disposition == "eligible_exact_excerpt" and terms_key not in publishers:
            raise DistributionError("missing_publisher_policy")
        if disposition == "link_only" and terms_key is not None:
            raise DistributionError("invalid_publisher_policy")
        sources[sid] = {
            "id": sid,
            "url": _url(manifest["url"]),
            "manifest_path": relative,
            "manifest_sha256": decision["manifest_sha256"],
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_hash,
            "evidence_bytes": len(evidence),
            "disposition": disposition,
            "reason_code": _token(decision.get("reason_code")),
            "legacy_restriction_flag": decision["legacy_restriction_flag"],
            "publisher_policy_key": terms_key,
            "published_at": manifest.get("published_at"),
            "available_by": manifest.get("available_by"),
            "public_mirrors": [],
            "affected_cases": [],
            "related_case_files": [],
        }
    if set(sources) != set(decisions):
        raise DistributionError("source_policy_membership_mismatch")
    if set(found_files) != expected_source_files:
        raise DistributionError("unmapped_source_file")
    cases = []
    expected_case_files = set()
    case_files = _files(root, CASE_ROOT)
    for relative in [p for p in case_files if p.endswith("/case.json")]:
        case = read_json(root, relative)
        if not isinstance(case, dict):
            raise DistributionError("invalid_case")
        cid = _token(case.get("id"))
        folder = f"{CASE_ROOT}/{cid}"
        if relative != folder + "/case.json":
            raise DistributionError("misplaced_case")
        ids = case.get("source_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(s, str) for s in ids) or len(set(ids)) != len(ids):
            raise DistributionError("invalid_case_source_ids")
        if any(sid not in sources for sid in ids):
            raise DistributionError("unmapped_case_source")
        metadata_path = folder + "/sources.json"
        metadata = read_json(root, metadata_path)
        if not isinstance(metadata, list) or any(not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in metadata) or sorted(m["id"] for m in metadata) != sorted(ids):
            raise DistributionError("case_source_metadata_mismatch")
        expected_case_files.update([relative, metadata_path])
        for item in metadata:
            sid = item["id"]
            source = sources[sid]
            mirror = folder + f"/sources/{sid}.txt"
            mirror_bytes = safe_file(root, mirror).read_bytes()
            if (digest(mirror_bytes) != source["evidence_sha256"]
                    or item.get("text_sha256") != source["evidence_sha256"]
                    or item.get("url") != source["url"]):
                raise DistributionError("source_mirror_binding_mismatch")
            expected_case_files.add(mirror)
            source["public_mirrors"].append({"path": mirror, "sha256": source["evidence_sha256"]})
            source["affected_cases"].append(cid)
            related_paths = [
                (relative, "public_question"), (metadata_path, "public_source_metadata"),
                (f"private_evaluator/legacy/source_grounded/cases/{cid}.json", "evaluator_case"),
                (f"private_evaluator/legacy/source_grounded/gold/{cid}.json", "evaluator_gold"),
                (f"private_evaluator/legacy/source_grounded/controls/{cid}.json", "evaluator_controls"),
            ]
            for related, role in related_paths:
                source["related_case_files"].append({
                    "path": related, "role": role,
                    "sha256": digest(safe_file(root, related).read_bytes()),
                    "exported_by_source_packet": role.startswith("public_") and all(sources[x]["disposition"] == "eligible_exact_excerpt" for x in ids),
                })
        blocked = sorted(sid for sid in ids if sources[sid]["disposition"] != "eligible_exact_excerpt")
        eligible = not blocked
        selected_paths = [relative, metadata_path, *(folder + f"/sources/{sid}.txt" for sid in ids)]
        if eligible and any(packet_files.get(p) != digest(safe_file(root, p).read_bytes()) for p in selected_paths):
            raise DistributionError("case_packet_review_required")
        cases.append({
            "id": cid,
            "question_path": relative,
            "question_sha256": digest(safe_file(root, relative).read_bytes()),
            "source_ids": ids,
            "blocked_source_ids": blocked,
            "packet_eligible": eligible,
            "availability": "included_in_source_packet" if eligible else "omitted_pending_source_review",
        })
    if set(case_files) != expected_case_files:
        raise DistributionError("unmapped_public_source_file")
    eligible_paths = {
        p for c in cases if c["packet_eligible"]
        for p in [c["question_path"], c["question_path"].replace("case.json", "sources.json"),
                  *(f"{CASE_ROOT}/{c['id']}/sources/{sid}.txt" for sid in c["source_ids"])]
    }
    if set(packet_files) != eligible_paths:
        raise DistributionError("packet_allowlist_mismatch")
    rows = [sources[sid] for sid in sorted(sources)]
    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "policy_sha256": digest(safe_file(root, POLICY_PATH).read_bytes()),
        "scope": "manifested_source_evidence_mirrors_and_associated_case_artifact_paths",
        "publisher_policies": publishers,
        "counts": {
            "sources": len(rows),
            "canonical_evidence_files": len(rows),
            "public_mirrors": sum(len(s["public_mirrors"]) for s in rows),
            "source_cases": len(cases),
            "legacy_restriction_flags": sum(s["legacy_restriction_flag"] is True for s in rows),
            "eligible_exact_excerpts": sum(s["disposition"] == "eligible_exact_excerpt" for s in rows),
            "link_only_sources": sum(s["disposition"] == "link_only" for s in rows),
            "eligible_source_cases": sum(c["packet_eligible"] for c in cases),
            "omitted_source_cases": sum(not c["packet_eligible"] for c in cases),
        },
        "sources": rows,
        "cases": sorted(cases, key=lambda c: c["id"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", type=Path, help="Compare with a saved deterministic inventory")
    args = parser.parse_args(argv)
    try:
        result = build_inventory(args.repo)
        if args.check:
            if args.check.read_bytes() != json_bytes(result):
                raise DistributionError("inventory_snapshot_mismatch")
            print(json.dumps({"status": "ok", "counts": result["counts"]}, sort_keys=True))
        else:
            print(json_bytes(result).decode(), end="")
        return 0
    except (DistributionError, OSError) as exc:
        code = str(exc) if isinstance(exc, DistributionError) else "filesystem_error"
        print(json.dumps({"status": "blocked", "code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
