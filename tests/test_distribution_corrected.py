"""Version matching and information boundaries of the corrected source packet."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distribution import corrected
from distribution.export import packet_contents as frozen_contents, verify_packet as verify_frozen
from distribution.inventory import CASE_ROOT, DEFAULT_ROOT, DistributionError


class CorrectedPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = corrected.packet_contents()
        cls.frozen = frozen_contents(DEFAULT_ROOT, "source-packet")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_packet(self, files=None):
        packet = self.root / "packet"
        packet.mkdir()
        for relative, data in (self.files if files is None else files).items():
            destination = packet / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return packet

    def verify(self, packet):
        # Boundary tests use one independently built expected packet; derivation
        # and its dependency checks are exercised separately below.
        with patch.object(corrected, "packet_contents", return_value=self.files):
            return corrected.verify_packet(packet)

    def test_all_seven_contracts_match_canonical_export(self):
        from scoring_v3.grade import public_question
        manifest = json.loads(self.files[corrected.MANIFEST])
        self.assertEqual(len(manifest["included_case_ids"]), 7)
        self.assertEqual(len(self.files), 35)
        for cid in manifest["included_case_ids"]:
            path = f"{CASE_ROOT}/{cid}/case.json"
            contract = json.loads(self.files[path])
            self.assertEqual(contract, public_question(cid))
            self.assertEqual(contract["scorer_binding"], manifest["scorer_binding"])
            self.assertEqual(contract["scorer_version"], manifest["scorer_version"])

    def test_export_changes_only_public_contract_additions(self):
        for path, frozen in self.frozen.items():
            if not path.startswith(CASE_ROOT + "/"):
                continue
            if not path.endswith("/case.json"):
                self.assertEqual(self.files[path], frozen)
                continue
            contract = json.loads(self.files[path])
            corrections = contract.pop("scoring_corrections")
            self.assertEqual(set(corrections), {
                "integer_fields", "integer_rule", "source_citation_rule"})
            contract.pop("scorer_version")
            contract.pop("scorer_binding")
            for field in contract["output_fields"]:
                field.pop("integer_selectors", None)
            self.assertEqual(contract, json.loads(frozen))
        expected = {p for p in self.frozen if p.startswith(CASE_ROOT + "/")}
        expected.update({"source_inventory.json", "case_availability.json", "README.txt", corrected.MANIFEST})
        self.assertEqual(set(self.files), expected)
        self.assertEqual(self.files["source_inventory.json"], self.frozen["source_inventory.json"])

    def test_manifest_binds_policy_scorer_and_original_packet(self):
        from distribution.inventory import digest
        manifest = json.loads(self.files[corrected.MANIFEST])
        original = json.loads(self.frozen["packet_manifest.json"])
        self.assertEqual(manifest["packet_type"], corrected.FORMAT)
        self.assertEqual(manifest["source_policy_sha256"], original["policy_sha256"])
        self.assertEqual(manifest["frozen_source_packet_manifest_sha256"], digest(self.frozen["packet_manifest.json"]))
        self.assertEqual(set(manifest["files"]), set(self.files) - {corrected.MANIFEST})
        for path, metadata in manifest["files"].items():
            self.assertEqual(metadata, {"sha256": digest(self.files[path]), "bytes": len(self.files[path])})

    def test_reconstructs_and_rejects_manifest_tampering(self):
        packet = self.write_packet()
        self.assertEqual(self.verify(packet)["status"], "ok")
        manifest = json.loads((packet / corrected.MANIFEST).read_bytes())
        manifest["scorer_binding"] = "0" * 64
        (packet / corrected.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DistributionError, "packet_bytes_mismatch"):
            self.verify(packet)

    def test_rejects_missing_changed_and_extra_files(self):
        packet = self.write_packet()
        target = packet / "README.txt"
        original = target.read_bytes()
        target.unlink()
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            self.verify(packet)
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(DistributionError, "packet_bytes_mismatch"):
            self.verify(packet)
        target.write_bytes(original)
        (packet / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            self.verify(packet)

    def test_rejects_extra_directory_symlink_and_fifo(self):
        packet = self.write_packet()
        extra = packet / "extra"
        extra.mkdir()
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            self.verify(packet)
        extra.rmdir()
        extra.symlink_to(packet / "README.txt")
        with self.assertRaisesRegex(DistributionError, "symlink_not_allowed"):
            self.verify(packet)
        extra.unlink()
        os.mkfifo(extra)
        with self.assertRaisesRegex(DistributionError, "special_file_not_allowed"):
            self.verify(packet)

    def test_formats_cannot_be_misidentified(self):
        packet = self.write_packet()
        with self.assertRaises(DistributionError):
            verify_frozen(DEFAULT_ROOT, packet)
        (packet / corrected.MANIFEST).unlink()
        (packet / "packet_manifest.json").write_bytes(self.frozen["packet_manifest.json"])
        with self.assertRaisesRegex(DistributionError, "packet_membership_mismatch"):
            self.verify(packet)

    def test_create_new_external_directory_and_never_overwrite(self):
        destination = self.root / "new"
        with patch.object(corrected, "packet_contents", return_value=self.files):
            result = corrected.export_packet(destination)
            self.assertEqual(result["files"], 35)
            with self.assertRaisesRegex(DistributionError, "output_must_be_new"):
                corrected.export_packet(destination)
        alias = self.root / "alias"
        alias.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(DistributionError, "output_must_be_new"):
            corrected.export_packet(alias)
        with self.assertRaisesRegex(DistributionError, "packet_directory_unavailable"):
            self.verify(alias)

    def test_rejects_in_repository_and_missing_parent_before_build(self):
        with patch.object(corrected, "packet_contents") as build:
            with self.assertRaisesRegex(DistributionError, "output_must_be_outside_repository"):
                corrected.export_packet(DEFAULT_ROOT / "unused-corrected-packet-test")
            with self.assertRaisesRegex(DistributionError, "output_parent_unavailable"):
                corrected.export_packet(self.root / "missing" / "packet")
            build.assert_not_called()

    def test_rejects_dependency_or_source_drift(self):
        with patch("scoring_v3.common.verify", side_effect=ValueError("drift")):
            with self.assertRaisesRegex(DistributionError, "scorer_binding_invalid"):
                corrected._current_contents()
        with patch.object(corrected, "frozen_packet_contents", side_effect=DistributionError("source_policy_binding_mismatch")):
            with self.assertRaisesRegex(DistributionError, "source_policy_binding_mismatch"):
                corrected._current_contents()

    def test_rejects_mixed_repository_or_changing_scorer(self):
        with patch("scoring_v3.common.ROOT", self.root):
            with self.assertRaisesRegex(DistributionError, "scorer_repository_mismatch"):
                corrected._current_contents()
        with patch("scoring_v3.common.binding", side_effect=["a" * 64, "b" * 64]):
            with self.assertRaisesRegex(DistributionError, "scorer_changed_during_export"):
                corrected._current_contents()

    def test_library_export_ignores_stale_imported_scorer_rules(self):
        from scoring_v3 import common, constraints, source
        # Simulate a long-lived caller retaining rules from a different revision.
        # An in-process build would silently omit every integer selector here.
        with patch.object(constraints, "APPROVED", {}):
            with patch.object(constraints, "INTEGER_NOTE", "stale integer rule"):
                with patch.object(source, "DUPLICATE_RULE", "stale cached rule"):
                    with patch.object(common, "VERSION", "stale version"):
                        self.assertEqual(corrected.packet_contents(), self.files)

    def test_failure_before_build_does_not_create_destination(self):
        destination = self.root / "new"
        with patch.object(corrected, "packet_contents", side_effect=DistributionError("bad_binding")):
            with self.assertRaises(DistributionError):
                corrected.export_packet(destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
