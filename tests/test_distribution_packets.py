"""Prove the export boundary excludes uncleared evidence and evaluator state."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from distribution.export import export_packet, main, packet_contents, verify_packet
from distribution.inventory import (
    CASE_ROOT, DEFAULT_ROOT, POLICY_PATH, SOURCE_ROOT, DistributionError,
    build_inventory, digest, json_bytes, safe_file,
)


class DistributionRepositoryTests(unittest.TestCase):
    def test_inventory_snapshot_and_all_source_mirrors(self):
        inventory = build_inventory(DEFAULT_ROOT)
        self.assertEqual(json_bytes(inventory), (DEFAULT_ROOT / "distribution/source_inventory.json").read_bytes())
        self.assertEqual(inventory["counts"], {
            "sources": 72, "canonical_evidence_files": 72, "public_mirrors": 72,
            "source_cases": 36, "legacy_restriction_flags": 58,
            "eligible_exact_excerpts": 20, "link_only_sources": 52,
            "eligible_source_cases": 7, "omitted_source_cases": 29,
        })
        for row in inventory["sources"]:
            self.assertTrue(row["affected_cases"])
            self.assertTrue(row["public_mirrors"])
            for mirror in row["public_mirrors"]:
                self.assertEqual(mirror["sha256"], row["evidence_sha256"])

    def test_links_contain_no_question_or_evidence_text(self):
        files = packet_contents(DEFAULT_ROOT)
        self.assertEqual(set(files), {"README.txt", "source_inventory.json", "case_availability.json", "packet_manifest.json"})
        output = b"\n".join(files.values())
        inventory = build_inventory(DEFAULT_ROOT)
        for source in inventory["sources"]:
            self.assertNotIn((DEFAULT_ROOT / source["evidence_path"]).read_bytes().strip(), output)
        for case in inventory["cases"]:
            prompt = json.loads((DEFAULT_ROOT / case["question_path"]).read_text())["prompt"]
            self.assertNotIn(prompt.encode(), output)

    def test_source_packet_only_copies_eligible_public_files(self):
        files = packet_contents(DEFAULT_ROOT, "source-packet")
        inventory = build_inventory(DEFAULT_ROOT)
        expected_cases = ["energy-001", "energy-002", "macro-001", "macro-002", "macro-003",
                          "portfolio-002", "systematic-001"]
        availability = json.loads(files["case_availability.json"])
        self.assertEqual(availability["included_case_ids"], expected_cases)
        self.assertEqual(sum(row["included"] for row in availability["cases"]), 7)
        self.assertEqual(len(files), 35)
        output = b"\n".join(files.values())
        for source in inventory["sources"]:
            if source["disposition"] == "link_only":
                self.assertNotIn((DEFAULT_ROOT / source["evidence_path"]).read_bytes().strip(), output)
            else:
                for mirror in source["public_mirrors"]:
                    case_id = Path(mirror["path"]).parts[-3]
                    if case_id in expected_cases:
                        self.assertEqual(digest(files[mirror["path"]]), source["evidence_sha256"])
                    else:
                        self.assertNotIn(mirror["path"], files)
        self.assertFalse(any(p.startswith("private_evaluator/") for p in files))
        self.assertFalse(any("answers" in p or "future" in p or "reserved" in p for p in files))

    def test_reviewed_government_source_does_not_release_incomplete_case(self):
        inventory = build_inventory(DEFAULT_ROOT)
        sources = {s["id"]: s for s in inventory["sources"]}
        cases = {c["id"]: c for c in inventory["cases"]}
        files = packet_contents(DEFAULT_ROOT, "source-packet")
        for cid, reviewed in [("energy-003", "exp-vistra-2023-index"),
                              ("systematic-002", "systematic-lyft-original-index"),
                              ("h2-011", "h2b-fdic-joint")]:
            self.assertEqual(sources[reviewed]["disposition"], "eligible_exact_excerpt")
            self.assertFalse(cases[cid]["packet_eligible"])
            self.assertTrue(cases[cid]["blocked_source_ids"])
            self.assertFalse(any(p.startswith(f"{CASE_ROOT}/{cid}/") for p in files))
        for sid in ["h2b-fdic-bridge", "h2b-fdic-initial", "h2b-fdic-ownership",
                    "exp-vistra-2023-issuer", "systematic-lyft-original", "systematic-lyft-correction"]:
            self.assertEqual(sources[sid]["disposition"], "link_only")
            self.assertIsNone(sources[sid]["publisher_policy_key"])

    def test_roundtrip_deterministic_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a", Path(temp) / "b"
            a = export_packet(DEFAULT_ROOT, first, "source-packet")
            b = export_packet(DEFAULT_ROOT, second, "source-packet")
            self.assertEqual(a, b)
            self.assertEqual(a, verify_packet(DEFAULT_ROOT, first))


class DistributionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        for relative in [SOURCE_ROOT, CASE_ROOT, *(f"private_evaluator/legacy/source_grounded/{name}" for name in ["cases", "gold", "controls"])]:
            shutil.copytree(DEFAULT_ROOT / relative, self.root / relative)
        (self.root / "distribution").mkdir()
        shutil.copy2(DEFAULT_ROOT / POLICY_PATH, self.root / POLICY_PATH)

    def mutate_json(self, relative, edit):
        path = self.root / relative
        value = json.loads(path.read_text())
        edit(value)
        path.write_bytes(json_bytes(value))

    def assert_blocked(self, code):
        output = self.base / "out"
        with self.assertRaisesRegex(DistributionError, f"^{code}$"):
            export_packet(self.root, output, "source-packet")
        self.assertFalse(output.exists())

    def test_unknown_source_blocks_export_without_repeating_content(self):
        source = self.root / SOURCE_ROOT / "new-source"
        source.mkdir()
        (source / "evidence.txt").write_text("UNREVIEWED_EVIDENCE_SENTINEL")
        (source / "manifest.json").write_bytes(json_bytes({"id": "new-source"}))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = main(["--repo", str(self.root), "create", "--out", str(self.base / "out")])
        self.assertEqual(status, 1)
        self.assertNotIn("UNREVIEWED_EVIDENCE_SENTINEL", stream.getvalue())
        self.assertNotIn(str(self.root), stream.getvalue())
        self.assertFalse((self.base / "out").exists())

    def test_changed_canonical_evidence_is_blocked(self):
        (self.root / SOURCE_ROOT / "bls-20240202/evidence.txt").write_text("UNREVIEWED_TEXT")
        self.assert_blocked("source_evidence_binding_mismatch")

    def test_new_hash_in_manifest_does_not_self_authorize(self):
        relative = SOURCE_ROOT + "/bls-20240202/manifest.json"
        evidence = self.root / SOURCE_ROOT / "bls-20240202/evidence.txt"
        evidence.write_text("UNREVIEWED_TEXT")
        self.mutate_json(relative, lambda m: m.update(raw_sha256=digest(evidence.read_bytes()), text_sha256=digest(evidence.read_bytes())))
        self.assert_blocked("source_policy_binding_mismatch")

    def test_changed_public_mirror_is_blocked(self):
        (self.root / CASE_ROOT / "macro-001/sources/bls-20240202.txt").write_text("UNREVIEWED_TEXT")
        self.assert_blocked("source_mirror_binding_mismatch")

    def test_newly_reviewed_statement_cannot_change_with_its_manifest(self):
        relative = SOURCE_ROOT + "/portfolio-fed-20230322/manifest.json"
        evidence = self.root / SOURCE_ROOT / "portfolio-fed-20230322/evidence.txt"
        evidence.write_text("UNREVIEWED_STATEMENT_SENTINEL")
        self.mutate_json(relative, lambda m: m.update(raw_sha256=digest(evidence.read_bytes()),
                                                    text_sha256=digest(evidence.read_bytes())))
        self.assert_blocked("source_policy_binding_mismatch")

    def test_unknown_public_file_is_not_silently_copied(self):
        (self.root / CASE_ROOT / "macro-001/answers.json").write_text("PRIVATE_ANSWER_SENTINEL")
        self.assert_blocked("unmapped_public_source_file")

    def test_unknown_private_source_file_is_not_silently_copied(self):
        (self.root / SOURCE_ROOT / "bls-20240202/original.pdf").write_bytes(b"UNREVIEWED_TEXT")
        self.assert_blocked("unmapped_source_file")

    def test_evaluator_keys_answers_and_future_states_do_not_enter_export(self):
        evaluator = self.root / "private_evaluator"
        for name in ["evaluator.key", "answer_key.json", "future_states.json"]:
            (evaluator / name).write_text("PRIVATE_STATE_SENTINEL")
        files = packet_contents(self.root, "source-packet")
        self.assertNotIn(b"PRIVATE_STATE_SENTINEL", b"\n".join(files.values()))
        self.assertEqual(files, packet_contents(DEFAULT_ROOT, "source-packet"))

    def test_changed_question_packet_needs_review(self):
        self.mutate_json(CASE_ROOT + "/macro-001/case.json", lambda m: m.update(prompt="PRIVATE_ANSWER_SENTINEL"))
        self.assert_blocked("case_packet_review_required")

    def test_url_or_rights_change_is_not_trusted(self):
        self.mutate_json(SOURCE_ROOT + "/h2a-boeing-terms/manifest.json", lambda m: m.update(redistribution_status="approved"))
        self.assert_blocked("source_policy_binding_mismatch")

    def test_manifest_cannot_redirect_evidence_to_private_file(self):
        self.mutate_json(SOURCE_ROOT + "/bls-20240202/manifest.json", lambda m: m.update(text_path="../../answer_key.json"))
        self.assert_blocked("source_evidence_binding_mismatch")

    def test_symlinked_evidence_is_rejected(self):
        path = self.root / SOURCE_ROOT / "bls-20240202/evidence.txt"
        path.unlink()
        path.symlink_to(DEFAULT_ROOT / SOURCE_ROOT / "bls-20240202/evidence.txt")
        self.assert_blocked("symlink_not_allowed")

    def test_symlinked_source_directory_is_rejected(self):
        path = self.root / SOURCE_ROOT / "bls-20240202"
        shutil.rmtree(path)
        path.symlink_to(DEFAULT_ROOT / SOURCE_ROOT / "bls-20240202", target_is_directory=True)
        self.assert_blocked("symlink_not_allowed")

    def test_existing_output_is_never_overwritten(self):
        output = self.base / "out"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(DistributionError, "output_must_be_new"):
            export_packet(self.root, output)
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_output_inside_repo_or_symlink_alias_is_rejected(self):
        with self.assertRaisesRegex(DistributionError, "output_must_be_outside_repository"):
            export_packet(self.root, self.root / "packet")
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(DistributionError, "output_must_be_outside_repository"):
            export_packet(self.root, alias / "packet")

    def test_verifier_rejects_extra_file(self):
        output = self.base / "out"
        export_packet(self.root, output)
        (output / "answer.json").write_text("PRIVATE_ANSWER_SENTINEL")
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            verify_packet(self.root, output)

    def test_verifier_rejects_extra_empty_directory(self):
        output = self.base / "out"
        export_packet(self.root, output)
        (output / "unreviewed").mkdir()
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            verify_packet(self.root, output)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable on this platform")
    def test_verifier_rejects_special_file_without_reading_it(self):
        output = self.base / "out"
        export_packet(self.root, output)
        os.mkfifo(output / "unreviewed.pipe")
        with self.assertRaisesRegex(DistributionError, "special_file_not_allowed"):
            verify_packet(self.root, output)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable on this platform")
    def test_source_inventory_rejects_special_file(self):
        os.mkfifo(self.root / SOURCE_ROOT / "bls-20240202/unreviewed.pipe")
        self.assert_blocked("special_file_not_allowed")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable on this platform")
    def test_verifier_does_not_read_a_fifo_replacing_its_manifest(self):
        output = self.base / "out"
        export_packet(self.root, output)
        manifest = output / "packet_manifest.json"
        manifest.unlink()
        os.mkfifo(manifest)
        with self.assertRaisesRegex(DistributionError, "required_file_unavailable"):
            verify_packet(self.root, output)

    def test_unknown_publisher_metadata_never_enters_packet(self):
        self.mutate_json(POLICY_PATH, lambda m: m["publisher_policies"]["bls"].update(
            private_answer="PRIVATE_METADATA_SENTINEL"))
        for mode in ("links", "source-packet"):
            with self.subTest(mode=mode), self.assertRaisesRegex(DistributionError, "invalid_publisher_policy"):
                packet_contents(self.root, mode)

    def test_publisher_metadata_requires_plain_strings_and_iso_date(self):
        policy = json.loads((self.root / POLICY_PATH).read_text())
        for field, value in [("observed_on", "2026-99-99"), ("observed_on", "20261112"),
                             ("reviewed_scope", {"private": "PRIVATE_METADATA_SENTINEL"}),
                             ("basis_summary", "")]:
            with self.subTest(field=field, value=value):
                (self.root / POLICY_PATH).write_bytes(json_bytes(policy))
                self.mutate_json(POLICY_PATH, lambda m: m["publisher_policies"]["bls"].update({field: value}))
                self.assert_blocked("invalid_publisher_policy")

    def test_verifier_rejects_tampered_source_even_with_updated_packet_hash(self):
        output = self.base / "out"
        export_packet(self.root, output, "source-packet")
        relative = CASE_ROOT + "/macro-001/sources/bls-20240202.txt"
        (output / relative).write_bytes(b"changed")
        manifest = json.loads((output / "packet_manifest.json").read_text())
        manifest["files"][relative] = {"sha256": digest(b"changed"), "bytes": 7}
        (output / "packet_manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(DistributionError, "packet_bytes_mismatch"):
            verify_packet(self.root, output)

    def test_policy_cannot_add_evaluator_file_to_packet(self):
        self.mutate_json(POLICY_PATH, lambda m: m["packet_files"].update({"private_evaluator/answer.json": "0" * 64}))
        self.assert_blocked("packet_allowlist_mismatch")

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaisesRegex(DistributionError, "unsupported_export_mode"):
            packet_contents(self.root, "include-private")

    def test_safe_path_rejects_traversal(self):
        for path in ["../outside", "/absolute", "a/../../outside", "a\\..\\outside"]:
            with self.subTest(path=path), self.assertRaises(DistributionError):
                safe_file(self.root, path)


if __name__ == "__main__":
    unittest.main()
