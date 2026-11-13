"""Verify installed wheel contents and native discovery outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PLUGINS = {
    "financial-research": ["taskset"],
    "financial-portfolio": ["taskset"],
    "financial-prerequisites": ["taskset"],
    "financial-portfolio-staged": ["taskset", "harness"],
    "financial-portfolio-clarified": ["taskset"],
    "financial-portfolio-operations": ["harness"],
    "financial-portfolio-runtime": ["taskset", "harness"],
}


# Only this stdlib validation program executes in the child, passed with -c.
# It neither imports this script nor puts the checkout on its import path.
CHILD = r'''
import hashlib, importlib, importlib.metadata, importlib.util, json, pathlib, sys, sysconfig

def fail(code, package=None, resource=None):
    result = {"schema_version": 1, "status": "failed", "code": code}
    if package is not None:
        result["package"] = package
    if resource is not None:
        result["resource"] = resource
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(1)

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def validate(expected):
    checkout = pathlib.Path(expected["checkout"]).resolve()
    installed_roots = {pathlib.Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")}
    if not sys.flags.isolated:
        fail("interpreter_not_isolated")
    if pathlib.Path.cwd().resolve().is_relative_to(checkout):
        fail("child_cwd_inside_checkout")
    for entry in sys.path:
        if entry and pathlib.Path(entry).resolve().is_relative_to(checkout):
            fail("checkout_on_import_path")
    report = []
    for row in expected["packages"]:
        name, module_name = row["name"], row["module"]
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            fail("distribution_missing", name)
        if distribution.version != row["version"]:
            fail("distribution_version_mismatch", name)
        direct_url = distribution.read_text("direct_url.json")
        if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable") is True:
            fail("editable_install", name)
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin:
            fail("package_missing", name)
        package = pathlib.Path(spec.origin).resolve().parent
        if package.is_relative_to(checkout) or not any(package.is_relative_to(root) for root in installed_roots):
            fail("package_import_outside_installed_roots", name)
        if pathlib.Path(spec.origin).is_symlink() or pathlib.Path(spec.origin).parent.is_symlink():
            fail("installed_symlink", name)
        actual = {}
        for path in package.rglob("*"):
            relative = path.relative_to(package)
            if "__pycache__" in relative.parts:
                continue
            if path.is_symlink():
                fail("installed_symlink", name, relative.as_posix())
            if path.is_dir():
                continue
            if not path.is_file():
                fail("installed_special_file", name, relative.as_posix())
            actual[relative.as_posix()] = digest(path.read_bytes())
        if set(actual) != set(row["files"]):
            missing = sorted(set(row["files"]) - set(actual))
            extra = sorted(set(actual) - set(row["files"]))
            fail("installed_resource_missing" if missing else "installed_resource_extra", name, (missing or extra)[0])
        for relative, wanted in row["files"].items():
            if actual[relative] != wanted:
                fail("installed_resource_mismatch", name, relative)
        report.append({"name": name, "version": row["version"], "resources": len(actual),
                       "resources_sha256": digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode())})
    plugins = []
    if any(row["plugins"] for row in expected["packages"]):
        if importlib.metadata.version("verifiers") != "0.4.1":
            fail("unsupported_native_dependency")
        from verifiers.v1.utils.loaders import harness_class, taskset_class
        for row in expected["packages"]:
            for kind in row["plugins"]:
                cls = (taskset_class if kind == "taskset" else harness_class)(row["name"])
                # Native discovery must resolve to the verified installed package.
                if not cls.__module__.startswith(row["module"] + "."):
                    fail("plugin_owned_by_different_package", row["name"])
                plugins.append({"id": row["name"], "kind": kind, "class": cls.__name__})
    print(json.dumps({"schema_version": 1, "status": "ok", "isolated": True,
                      "packages": report, "plugins": plugins}, sort_keys=True))

try:
    validate(json.load(sys.stdin))
except Exception:
    # Import/decode exceptions can contain local paths or arbitrary package text.
    fail("installed_validation_error")
'''


def source_inventory(root: Path = ROOT) -> dict:
    """Hash every package source/resource; cache artifacts are excluded explicitly."""
    root = Path(root).resolve()
    packages = []
    for spec_path in sorted((root / "environments").glob("*/pyproject.toml")):
        spec = tomllib.loads(spec_path.read_text())["project"]
        name = spec["name"]
        if name not in PLUGINS:
            raise ValueError("unregistered_local_package")
        module = name.replace("-", "_")
        package = spec_path.parent / "src" / module
        if package.is_symlink() or not package.is_dir():
            raise ValueError("source_package_unavailable")
        payloads = {}
        for path in package.rglob("*"):
            relative = path.relative_to(package)
            if "__pycache__" in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError("source_symlink")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("source_special_file")
            payloads[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if "__init__.py" not in payloads:
            raise ValueError("source_package_init_missing")
        packages.append({"name": name, "module": module, "version": spec["version"],
                         "files": dict(sorted(payloads.items())), "plugins": PLUGINS[name]})
    if {row["name"] for row in packages} != set(PLUGINS) or len(packages) != len(PLUGINS):
        raise ValueError("local_package_membership_mismatch")
    return {"checkout": str(root), "packages": packages}


def verify_inventory(expected: dict, *, executable=sys.executable) -> dict:
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    environment["PRIME_DISABLE_VERSION_CHECK"] = "1"
    with tempfile.TemporaryDirectory(prefix="installed-wheel-check-") as directory:
        process = subprocess.run([str(executable), "-I", "-B", "-c", CHILD],
                                 input=json.dumps(expected), capture_output=True, text=True,
                                 cwd=directory, env=environment, timeout=120)
    try:
        result = json.loads(process.stdout)
    except (json.JSONDecodeError, TypeError):
        return {"schema_version": 1, "status": "failed", "code": "invalid_child_report"}
    if (not isinstance(result, dict) or result.get("status") not in ("ok", "failed")
            or (process.returncode == 0) != (result["status"] == "ok")):
        return {"schema_version": 1, "status": "failed", "code": "invalid_child_status"}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Installed environment's interpreter")
    args = parser.parse_args(argv)
    try:
        report = verify_inventory(source_inventory(), executable=args.python)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        report = {"schema_version": 1, "status": "failed", "code": "validation_setup_failed"}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
