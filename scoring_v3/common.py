"""Pinned predecessor imports and a distinct scorer/session content binding."""
import hashlib
from pathlib import Path

from scoring_v2.common import (
    ROOT, FROZEN, catalog, digest, read, number, schema_match,
    verify as verify_frozen,
)
from . import VERSION

PINS = Path(__file__).with_name('dependencies.json')
DISCRETE_MAP = ROOT / 'quality/discrete_fields.json'


def verify_dependencies():
    pins = read(PINS)
    for relative, expected in pins['files'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('V3 dependency binding differs: ' + relative)
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'scoring_v2').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    expected = {p for p in pins['files'] if p.startswith('scoring_v2/')}
    if actual != expected:
        raise ValueError('V3 predecessor file set differs')
    return pins


def verify():
    verify_frozen()
    verify_dependencies()


def binding():
    """Changing an approved map, predecessor or v3 artifact invalidates sessions."""
    pins = verify_dependencies()
    paths = [FROZEN / 'MANIFEST.json', *(ROOT / p for p in pins['files'])]
    paths += [p for p in Path(__file__).parent.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def stamped(result):
    return dict(result, scorer_version=VERSION, scorer_binding=binding(),
                predecessor_version='financial-reasoning-v2.0.0',
                baseline_commit=read(ROOT / 'scoring_v2/baseline.json')['commit'])
