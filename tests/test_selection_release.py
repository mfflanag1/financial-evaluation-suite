"""Release discovery and immutable image selection regressions."""
import ast
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import check
from tests import test_selection_package as bridge


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


class SelectionDiscovery(unittest.TestCase):
    def test_contained_test_methods_are_discovered_once(self):
        native = bridge.native_available()
        suite = bridge.package_suite(unittest.TestLoader(), include_native=native)
        identifiers = [test.id() for test in cases(suite)]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        expected = []
        for path in sorted((bridge.PACKAGE / "tests").glob("test_*.py")):
            if path.name == "test_native.py" and not native:
                continue
            tree = ast.parse(path.read_text())
            for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
                for method in cls.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name.startswith("test_"):
                        expected.append(f"selection_package_{path.stem}.{cls.name}.{method.name}")
        actual = [name for name in identifiers if name.startswith("selection_package_")]
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertTrue(any("test_verifier" in name for name in actual))
        self.assertTrue(any("test_replay" in name for name in actual))
        self.assertTrue((bridge.PACKAGE / "tests/test_native.py").is_file())
        if native:
            self.assertTrue(any("test_native" in name for name in actual))

    def test_core_discovery_does_not_import_optional_native_modules(self):
        code = '''
import importlib.abc, json, pathlib, sys, unittest
sys.path.insert(0, sys.argv[1])
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'verifiers', 'pydantic'}:
            raise AssertionError('Core discovery imported native dependency')
sys.meta_path.insert(0, DenyNative())
from tests import test_selection_package as bridge
suite = bridge.package_suite(unittest.TestLoader(), include_native=False)
def flat(suite):
    for item in suite:
        yield from flat(item) if isinstance(item, unittest.TestSuite) else [item]
items = list(flat(suite))
print(json.dumps({'tests': len(items), 'skipped_native': sum('NativeDependencies' in t.id() for t in items)}))
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(bridge.ROOT)],
                                capture_output=True, text=True, check=True, timeout=20)
        report = json.loads(result.stdout)
        self.assertGreater(report["tests"], 1)
        self.assertEqual(report["skipped_native"], 1)

    def test_core_sources_parse_as_python_311(self):
        paths = (bridge.PACKAGE / "src/financial_selection").glob("*.py")
        for path in paths:
            if path.name != "prime_adapter.py":
                with self.subTest(source=path.name):
                    ast.parse(path.read_text(), filename=path.name, feature_version=(3, 11))

    def test_native_dependency_skip_cannot_pass_the_native_release_gate(self):
        suite = bridge.package_suite(unittest.TestLoader(), include_native=False)
        sentinel = next(test for test in cases(suite) if "NativeDependencies" in test.id())
        result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite([sentinel]))
        self.assertEqual(len(result.skipped), 1)
        reason = result.skipped[0][1]
        self.assertFalse(any(flag in reason for flag in check.DOCKER_FLAGS))
        self.assertNotEqual(reason, "Opt-in Docker integration")


class SelectionImageBinding(unittest.TestCase):
    def test_docker_tier_enables_both_selection_suites(self):
        self.assertTrue({"FINANCIAL_SELECTION_DOCKER_TESTS", "FINANCIAL_SELECTION_NATIVE_DOCKER_TESTS"}
                        <= set(check.DOCKER_FLAGS))

    def test_local_tag_is_resolved_once_to_immutable_identity(self):
        identity = "sha256:" + "a" * 64
        with patch.dict(os.environ, {}, clear=True), patch.object(
                check.subprocess, "run", return_value=SimpleNamespace(stdout=identity + "\n")) as run:
            self.assertEqual(check.bind_selection_image(), identity)
            self.assertEqual(os.environ[check.SELECTION_IMAGE_ENV], identity)
            self.assertEqual(run.call_args.args[0],
                             ["docker", "image", "inspect", "python:3.12-slim", "--format", "{{.Id}}"])
            self.assertTrue(run.call_args.kwargs["check"])

    def test_mutable_override_and_invalid_daemon_identity_fail_closed(self):
        with patch.dict(os.environ, {check.SELECTION_IMAGE_ENV: "python:latest"}), \
             patch.object(check.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "must be immutable"):
                check.bind_selection_image()
            run.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), patch.object(
                check.subprocess, "run", return_value=SimpleNamespace(stdout="not-an-image-id\n")):
            with self.assertRaisesRegex(ValueError, "immutable local image"):
                check.bind_selection_image()
            self.assertNotIn(check.SELECTION_IMAGE_ENV, os.environ)

    def test_missing_local_image_is_not_pulled_or_replaced(self):
        identity = "sha256:" + "b" * 64
        with patch.dict(os.environ, {check.SELECTION_IMAGE_ENV: identity}), patch.object(
                check.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "inspect")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                check.bind_selection_image()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(os.environ[check.SELECTION_IMAGE_ENV], identity)


if __name__ == "__main__":
    unittest.main()
