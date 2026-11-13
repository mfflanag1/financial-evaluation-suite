"""A release cannot pass with a missing or stale runtime protocol manifest."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/runtime_v1"


def freezer():
    spec = importlib.util.spec_from_file_location("runtime_release_freezer", CONFIG / "freeze_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeProtocol(unittest.TestCase):
    def test_frozen_manifest_is_required_and_matches_complete_payload(self):
        manifest = CONFIG / "manifest.json"
        self.assertTrue(manifest.is_file(), "The reviewed runtime must be frozen before release")
        self.assertEqual(json.loads(manifest.read_text()), freezer().build_manifest())

    def test_existing_manifest_cannot_be_overwritten(self):
        module = freezer()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("preserved")
            with patch.object(module, "MANIFEST", manifest), \
                 patch.object(module, "build_manifest", return_value={}), \
                 patch("sys.argv", ["freeze_manifest.py", "--write"]):
                with self.assertRaises(FileExistsError):
                    module.main()
            self.assertEqual(manifest.read_text(), "preserved")

    def test_payload_tree_binds_resources_and_rejects_symlinks(self):
        module = freezer()
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            resource = tree / "schema.json"
            resource.write_text('{}')
            self.assertEqual(module.regular_payload_tree(tree), {resource})
            (tree / "linked-resource").symlink_to(resource)
            with self.assertRaisesRegex(ValueError, "regular files"):
                module.regular_payload_tree(tree)


if __name__ == "__main__":
    unittest.main()
