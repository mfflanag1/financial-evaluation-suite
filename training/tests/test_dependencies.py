"""Dependency boundaries run without third-party packages or live services."""
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("training_bootstrap", ROOT / "training/scripts/bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class DependencyBoundaryTests(unittest.TestCase):
    def test_native_dependency_versions_and_financial_core_extras(self):
        project = tomllib.loads((ROOT / "training/pyproject.toml").read_text())["project"]
        sources = json.loads((BOOTSTRAP.DEPENDENCIES / "sources.json").read_text())
        expected = {f"{name}=={version}" for pin in sources["sources"].values()
                    for name, version in pin["distributions"].items()}
        expected.update({"financial-research==0.2.0", "financial-portfolio==0.3.0"})
        self.assertEqual(set(project["dependencies"]), expected)
        self.assertEqual(project["requires-python"], "~=3.12.0")
        self.assertNotIn("license", project)

    def test_registry_lock_has_no_local_paths_unpinned_or_excluded_dependencies(self):
        names = BOOTSTRAP.validate_registry_lock(BOOTSTRAP.DEPENDENCIES / "cpu-requirements.txt")
        self.assertEqual(len(names), 150)
        self.assertFalse(names & BOOTSTRAP.FORBIDDEN)
        for native in ("verifiers", "renderers", "prime-rl", "prime-rl-configs", "prime-pydantic-config"):
            self.assertNotIn(native, names)

    def test_invalid_registry_lock_fails_closed(self):
        good_hash = "    --hash=sha256:" + "0" * 64 + "\n"
        for content in ("example>=1\n", "https://example.invalid/archive.whl\n",
                        "example==1 \\\n", "torch==1 \\\n" + good_hash,
                        "example==1 \\\n" + good_hash + "example==2 \\\n" + good_hash,
                        "example==1 \\\n    --hash=sha256:broken\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lock.txt"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    BOOTSTRAP.validate_registry_lock(path)

    def test_source_commit_mismatch_is_rejected(self):
        with patch.object(BOOTSTRAP, "capture", return_value="b" * 40):
            with self.assertRaisesRegex(ValueError, "commit"):
                BOOTSTRAP.verify_source(Path("source"), {"commit": "a" * 40})

    def test_dirty_source_is_rejected(self):
        with patch.object(BOOTSTRAP, "capture", side_effect=["a" * 40, " M pyproject.toml"]):
            with self.assertRaisesRegex(ValueError, "clean"):
                BOOTSTRAP.verify_source(Path("source"), {"commit": "a" * 40})

    def test_ignored_source_injection_is_rejected(self):
        with patch.object(BOOTSTRAP, "capture", side_effect=["a" * 40, "", "verifiers/injected.py"]):
            with self.assertRaisesRegex(ValueError, "contaminate"):
                BOOTSTRAP.verify_source(Path("source"), {"commit": "a" * 40})

    def test_generated_vcs_version_and_caches_are_allowed(self):
        with patch.object(BOOTSTRAP, "capture", side_effect=["a" * 40, "", "renderers/_version.py\nverifiers/__pycache__/types.pyc"]):
            BOOTSTRAP.verify_source(Path("source"), {"commit": "a" * 40})

    def test_native_commits_are_full_length_and_match_binding(self):
        sources = json.loads((BOOTSTRAP.DEPENDENCIES / "sources.json").read_text())["sources"]
        bindings = json.loads((ROOT / "training/src/financial_portfolio_training/source_bindings.json").read_text())
        for pin in sources.values():
            self.assertRegex(pin["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(sources["verifiers"]["commit"], bindings["native_commit"])
        self.assertGreater(len(bindings["native_sources"]), 0)

    def test_wheel_metadata_requires_single_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("test-1.dist-info/METADATA", "Name: test\nVersion: 1\n")
            self.assertEqual(BOOTSTRAP.wheel_metadata(path), ("test", "1"))
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("other-1.dist-info/METADATA", "Name: other\nVersion: 1\n")
            with self.assertRaises(ValueError):
                BOOTSTRAP.wheel_metadata(path)

    def test_isolated_installed_check_is_syntactically_valid(self):
        compile(BOOTSTRAP.CHECK, "<installed-check>", "exec")


if __name__ == "__main__":
    unittest.main()
