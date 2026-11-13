"""Real isolated subprocess regressions for installed resource validation."""

import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

from scripts.check_installed import ROOT, source_inventory, verify_inventory
from scripts import bootstrap


class InstalledWheelBoundary(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="installed-check-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.environment = self.root / "venv"
        venv.EnvBuilder(with_pip=False).create(self.environment)
        self.python = self.environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        self.site = Path(subprocess.check_output(
            [str(self.python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        self.package = self.site / "fixture_runtime"
        self.package.mkdir()
        payloads = {"__init__.py": b"VALUE = 1\n", "instructions.txt": b"Read the public task.\n"}
        for name, data in payloads.items():
            (self.package / name).write_bytes(data)
        self.metadata = self.site / "fixture_runtime-0.1.0.dist-info"
        self.metadata.mkdir()
        (self.metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: fixture-runtime\nVersion: 0.1.0\n")
        self.expected = {"checkout": str(self.checkout), "packages": [{"name": "fixture-runtime",
                         "module": "fixture_runtime", "version": "0.1.0", "plugins": [],
                         "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}]}

    def check(self):
        result = verify_inventory(self.expected, executable=self.python)
        self.assertNotIn(str(self.root), json.dumps(result))
        return result

    def test_clean_installed_resources_pass_in_isolated_child(self):
        with patch.dict(os.environ, {"PYTHONPATH": str(self.checkout), "PYTHONHOME": "/nonexistent/fixture-home"}):
            result = self.check()
        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(result["isolated"])
        self.assertEqual(result["packages"][0]["resources"], 2)

    def test_missing_nonpython_resource_is_detected(self):
        (self.package / "instructions.txt").unlink()
        result = self.check()
        self.assertEqual(result["code"], "installed_resource_missing")
        self.assertEqual(result["resource"], "instructions.txt")

    def test_mutated_resource_is_detected_before_import(self):
        (self.package / "__init__.py").write_text("raise RuntimeError('must not execute')\n")
        result = self.check()
        self.assertEqual(result["code"], "installed_resource_mismatch")
        self.assertEqual(result["resource"], "__init__.py")

    def test_pth_checkout_import_cannot_replace_installed_package(self):
        (self.site / "unintended-checkout.pth").write_text(str(self.checkout) + "\n")
        self.assertEqual(self.check()["code"], "checkout_on_import_path")

    def test_editable_distribution_is_rejected(self):
        (self.metadata / "direct_url.json").write_text(json.dumps({"url": self.checkout.as_uri(),
                                                                   "dir_info": {"editable": True}}))
        self.assertEqual(self.check()["code"], "editable_install")

    def test_extra_installed_source_is_detected(self):
        (self.package / "unexpected.py").write_text("UNEXPECTED = True\n")
        self.assertEqual(self.check()["code"], "installed_resource_extra")

    def test_installed_resource_symlink_is_rejected(self):
        target = self.root / "outside.txt"
        target.write_text("Read the public task.\n")
        resource = self.package / "instructions.txt"
        resource.unlink()
        try:
            resource.symlink_to(target)
        except OSError:
            self.skipTest("This platform cannot create the filesystem boundary fixture")
        self.assertEqual(self.check()["code"], "installed_symlink")


class InstalledSourceInventory(unittest.TestCase):
    def test_all_seven_packages_and_nonpython_resources_are_bound(self):
        inventory = source_inventory(ROOT)
        self.assertEqual(len(inventory["packages"]), 7)
        staged = next(row for row in inventory["packages"] if row["name"] == "financial-portfolio-staged")
        self.assertIn("instructions.txt", staged["files"])
        self.assertIn("THIRD_PARTY_LICENSES.txt", staged["files"])
        self.assertEqual(sum(len(row["plugins"]) for row in inventory["packages"]), 9)


class BootstrapDestination(unittest.TestCase):
    def test_in_checkout_environment_rejected_before_any_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = (Path(directory) / "checkout").resolve()
            checkout.mkdir()
            with patch.object(bootstrap, "ROOT", checkout), patch.object(sys, "version_info", (3, 12, 0)), \
                 patch.object(sys, "argv", ["bootstrap", "--venv", str(checkout / ".venv")]), \
                 patch.object(bootstrap.venv, "EnvBuilder") as builder, \
                 patch.object(bootstrap.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bootstrap.main()
                self.assertEqual(error.exception.code, 2)
                builder.assert_not_called()
                run.assert_not_called()
            self.assertFalse((checkout / ".venv").exists())

    def test_default_environment_is_an_external_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = (Path(directory) / "checkout").resolve()
            checkout.mkdir()
            with patch.object(bootstrap, "ROOT", checkout), patch.object(sys, "version_info", (3, 12, 0)), \
                 patch.object(sys, "argv", ["bootstrap"]), patch.object(bootstrap, "packages", return_value=[]), \
                 patch.object(bootstrap.venv, "EnvBuilder") as builder, \
                 patch.object(bootstrap.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
                bootstrap.main()
                target = builder.return_value.create.call_args.args[0]
                self.assertEqual(target, checkout.with_name("checkout-venv"))
                self.assertFalse(target.is_relative_to(checkout))


if __name__ == "__main__":
    unittest.main()
