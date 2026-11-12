"""Frozen data loading, version binding and strict numeric primitives."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from . import VERSION

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / 'private_evaluator'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value


def verify():
    pins = read(Path(__file__).with_name('baseline.json'))
    for name, expected in pins['manifests'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen baseline differs: ' + name)
    spec = importlib.util.spec_from_file_location('_frozen_collection', FROZEN / 'grade.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()


verify()
# Frozen dependencies are imported only after their manifests are checked.
sys.path[:0] = [str(FROZEN), str(FROZEN / 'legacy')]


def binding():
    """Bind sessions to both the frozen bank and this corrected implementation."""
    paths = [FROZEN / 'MANIFEST.json']
    paths += sorted(p for p in Path(__file__).parent.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts)
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def integer(value):
    return number(value) and (type(value) is int or value.is_integer())


def normalize_units(value):
    if isinstance(value, list):
        return [normalize_units(v) for v in value]
    return int(value) if type(value) is float and integer(value) else value


def schema_match(value, reference):
    """Check structure and types without requiring the correct financial value."""
    if type(reference) is bool:
        return type(value) is bool
    if number(reference):
        return number(value)
    if isinstance(reference, list):
        return isinstance(value, list) and len(value) == len(reference) and all(
            schema_match(v, r) for v, r in zip(value, reference))
    return type(value) is type(reference)


@lru_cache(None)
def _catalog():
    from financial_evals import core
    result = {}
    for path in sorted((FROZEN / 'legacy/source_grounded/cases').glob('*.json')):
        case, _, _ = core.load_case(FROZEN / 'legacy/source_grounded', path.stem)
        result[case['id']] = ('source', case)
    for track in ('quant100', 'workflows', 'research58'):
        for case in read(FROZEN / 'legacy' / track / 'evaluator.json'):
            result[case['id']] = (track, case)
    for track in ('challenges', 'variants'):
        for case in read(FROZEN / track / 'cases.json'):
            result[case['id']] = (track, case)
    return result


def catalog():
    return deepcopy(_catalog())


def stamped(result):
    return dict(result, scorer_version=VERSION, scorer_binding=binding(),
                baseline_commit=read(Path(__file__).with_name('baseline.json'))['commit'])
