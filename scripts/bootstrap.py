"""Install all local environment packages in a new Python 3.12 virtual environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]


def packages() -> list[Path]:
    paths = sorted((ROOT / "environments").glob("*/pyproject.toml"))
    names = []
    for path in paths:
        spec = tomllib.loads(path.read_text())
        names.append(spec["project"]["name"])
    if len(names) != len(set(names)) or not names:
        raise ValueError("Local package names must be unique and nonempty")
    return [path.parent for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT.with_name(ROOT.name + "-venv"),
                        help="New environment outside the checkout (default: sibling directory)")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Run this installer with Python 3.12")
    if args.venv.exists() or args.venv.is_symlink():
        parser.error("Choose a new virtual-environment directory; existing paths are never overwritten")
    target = args.venv.resolve()
    if target.is_relative_to(ROOT):
        parser.error("Choose an environment outside the checkout for isolated installed-package validation")
    if args.offline and args.wheelhouse is None:
        parser.error("--offline requires --wheelhouse containing runtime and build dependencies")
    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        parser.error("Wheelhouse must be an existing directory")
    local = packages()
    venv.EnvBuilder(with_pip=True).create(target)
    executable = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    command = [str(executable), "-m", "pip", "install", "--disable-pip-version-check",
               "--constraint", str(ROOT / "requirements/runtime-constraints.txt")]
    if args.offline:
        command.append("--no-index")
    if wheelhouse is not None:
        command.extend(["--find-links", str(wheelhouse)])
    # Submit local projects together so pinned inter-package dependencies resolve locally.
    command.extend(str(path) for path in local)
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([str(executable), "-m", "pip", "check"], cwd=ROOT, check=True)
    subprocess.run([str(executable), "-B", str(ROOT / "scripts/check_installed.py")],
                   cwd=ROOT, check=True)
    print(f"Installed {len(local)} local packages. Run {executable} scripts/check.py --tier native")


if __name__ == "__main__":
    main()
