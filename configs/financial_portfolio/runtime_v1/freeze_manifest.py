"""Write once or verify the runtime successor and its frozen predecessors."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
PACKAGE = ROOT / "environments/financial_portfolio_runtime"
MANIFEST = CONFIG / "manifest.json"
PREDECESSOR = ROOT / "configs/financial_portfolio/operational_v2/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_payload_tree(directory):
    """Include non-Python resources and reject special files or symlink trees."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Runtime payload tree must be a regular directory")
    paths = set()
    for path in directory.rglob("*"):
        if "__pycache__" in path.relative_to(directory).parts:
            continue
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Runtime payload must contain regular files")
        if path.is_file():
            paths.add(path)
    return paths


def build_manifest():
    spec = importlib.util.spec_from_file_location("runtime_predecessor", PREDECESSOR.with_name("freeze_manifest.py"))
    predecessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predecessor)
    frozen = json.loads(PREDECESSOR.read_text())
    if frozen != predecessor.build_manifest():
        raise ValueError("Frozen predecessor changed")
    hashes = dict(frozen["sha256"])
    hashes[PREDECESSOR.relative_to(ROOT).as_posix()] = digest(PREDECESSOR)
    payload = {PACKAGE / "pyproject.toml", PACKAGE / "README.md"}
    payload.update(regular_payload_tree(PACKAGE / "src"))
    payload.update(regular_payload_tree(CONFIG) - {MANIFEST})
    payload.update((ROOT / "tests").glob("test_runtime_release_*.py"))
    for path in payload:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Runtime payload must contain regular files")
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    return {"protocol_id": "portfolio-runtime-v1", "predecessor_manifest_sha256": digest(PREDECESSOR),
            "packages": {**frozen["packages"], "financial-portfolio-runtime": "0.1.0"},
            "financial_scorer_changed": False, "default_guidance": "", "model_validation_performed": False,
            "training_update_verified": False, "sha256": dict(sorted(hashes.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--write", action="store_true")
    commands.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.write:
        with MANIFEST.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print("Runtime manifest written.")
    elif json.loads(MANIFEST.read_text()) != manifest:
        raise SystemExit("Runtime manifest differs from its payload")
    else:
        print("Runtime manifest verified.")


if __name__ == "__main__":
    main()
