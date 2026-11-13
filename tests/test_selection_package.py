"""Expose the contained selection controls to the root verification tiers.

The financial calculations are standard-library code and run on Python 3.11.
Only the native adapter suite needs Python 3.12 and the pinned runtime packages.
An unavailable native suite is an explicit skip, never a silent omission.
"""
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "environments/financial_selection"
for source in (ROOT / "environments/financial_research/src", PACKAGE / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def native_available():
    if sys.version_info[:2] != (3, 12):
        return False
    try:
        return importlib.metadata.version("verifiers") == "0.4.1"
    except importlib.metadata.PackageNotFoundError:
        return False


def package_suite(loader, *, include_native=None):
    if include_native is None:
        include_native = native_available()
    suite = unittest.TestSuite()
    for path in sorted((PACKAGE / "tests").glob("test_*.py")):
        if path.name == "test_native.py" and not include_native:
            class NativeDependencies(unittest.TestCase):
                @unittest.skip("Selection native suite requires Python 3.12 and pinned native dependencies")
                def test_native_suite_available(self):
                    pass
            suite.addTests(loader.loadTestsFromTestCase(NativeDependencies))
            continue
        name = "selection_package_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Errors in an available suite must fail discovery, not become skips.
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


def load_tests(loader, standard_tests, pattern):
    return package_suite(loader)
