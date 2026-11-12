"""Frozen workflow-case contracts and executed-answer scoring.

Reference Python is private construction material; loading/scoring never executes it.
"""
from copy import deepcopy
import json
from pathlib import Path

from .core import read_json
from .quant import actor, digest, require, score, typed, validate_question

RUNTIME_FILES = {'workflows.py', 'python_broker.py', 'model_runner.py',
                 'quant.py', 'core.py', '__init__.py'}

PROGRAM_INSTRUCTIONS = (
    'Submit strict JSON with exactly one key, code, whose value is Python source defining '
    'solve(data). The controller supplies the inputs object as data. Return a JSON-native '
    'dictionary mapping every declared output name to its answer. Use Python stdlib and '
    'NumPy only; do not read files, access the network, spawn processes, or use native tools. '
    'Write a general calculation from data, not constants for this instance. Your program '
    'will also be evaluated on undisclosed changed inputs under the same conventions. '
    'The controller will execute the first submission and give you its public outputs or '
    'error once, then ask for a final code submission. It never reveals reference answers '
    'or hidden test results. Supporting numerical ledgers and diagnostics are the declared '
    'array outputs; follow their ordering exactly. No markdown fences around your JSON.'
)


def validate_case(q):
    require(isinstance(q, dict), 'Workflow case must be an object')
    require(q.get('data_origin') in ('synthetic', 'public_observations_with_stipulated_protocol'), 'Invalid workflow provenance')
    legacy = deepcopy(q)
    legacy['data_origin'] = 'synthetic'  # Only adapt the legacy schema check, never stored provenance.
    validate_question(legacy)
    require(q.get('workflow_area') in ('pnl', 'research', 'estimation'), 'Invalid workflow area')
    require(isinstance(q.get('provenance'), dict) and q['provenance'], 'Missing provenance')
    require(isinstance(q.get('reference_code'), str) and 'def solve(' in q['reference_code'], 'Missing reference program')
    critical = q.get('critical_outputs')
    require(isinstance(critical, list) and critical
            and all(isinstance(name, str) for name in critical)
            and len(set(critical)) == len(critical)
            and set(critical) <= set(q['answer']), 'Invalid critical outputs')
    variants = q.get('variants')
    require(isinstance(variants, list) and len(variants) == 2, 'Exactly two robustness variants required')
    names, scenarios = set(), [q['inputs']]
    for v in variants:
        require(isinstance(v, dict), 'Variant must be an object')
        require(isinstance(v.get('name'), str) and v['name'] and v['name'] != 'base', 'Invalid variant name')
        require(v['name'] not in names, 'Variant names must differ')
        names.add(v['name'])
        require(isinstance(v.get('inputs'), dict) and v['inputs']
                and isinstance(v.get('purpose'), str) and v['purpose'].strip(), 'Incomplete variant')
        require(all(v['inputs'] != prior for prior in scenarios), 'Variant inputs must differ from base and other variants')
        scenarios.append(v['inputs'])
        require(isinstance(v.get('answer'), dict) and set(v['answer']) == set(q['answer']), 'Variant answer keys differ')
        require(all(typed(v['answer'][f['name']], f['type']) for f in q['outputs']), 'Invalid variant answer type')
        require(all(len(v['answer'][f['name']]) == len(q['answer'][f['name']])
                    for f in q['outputs'] if f['type'] == 'number_array'), 'Variant output array shape differs')
        require(not score(q, {'question_id':q['id'], 'answers':v['answer']})['all_correct'], 'Variant changes no scored output')
    json.dumps(q, allow_nan=False)
    return q


def public_case(q):
    public = actor(q)
    public['instructions'] = PROGRAM_INSTRUCTIONS
    for name in ('workflow_area', 'data_origin', 'provenance'):
        public[name] = deepcopy(q[name])
    return public


def grade_answers(q, answers, variant=None):
    reference = q if variant is None else dict(q, answer=variant['answer'])
    result = score(reference, {'question_id':q['id'], 'answers':answers})
    critical = q.get('critical_outputs', [f['name'] for f in q['outputs']])
    result['critical_outputs'] = critical
    result['critical_all_correct'] = all(f['correct'] for f in result['fields'] if f['name'] in critical)
    return result


def load_bank(path):
    root = Path(path).resolve()
    manifest = read_json(root/'manifest.json')
    require(manifest.get('schema_version') == 1 and manifest.get('question_count') == 6, 'Invalid workflow manifest')
    require(not any(p.is_symlink() for p in root.rglob('*')), 'Symlinks forbidden in workflow bank')
    actual = {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p != root/'manifest.json'}
    require(actual == manifest.get('files'), 'Workflow bank file set or hash drift')
    pins = manifest.get('runtime_sha256')
    require(isinstance(pins, dict) and set(pins) == RUNTIME_FILES, 'Workflow runtime pin set differs')
    for filename, sha in pins.items():
        require(digest(Path(__file__).with_name(filename)) == sha, 'Use frozen workflow runtime: '+filename)
    cases = read_json(root/'evaluator.json')
    require(isinstance(cases, list) and len(cases) == 6, 'Need six workflow cases')
    for q in cases:
        validate_case(q)
        require(read_json(root/'questions'/f'{q["id"]}.json') == public_case(q), 'Public workflow differs from contract')
    require([q['id'] for q in cases] == [f'q{i}' for i in range(101,107)], 'Need q101 through q106')
    return cases, manifest
