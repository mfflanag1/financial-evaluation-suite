#!/usr/bin/env python3
"""Validate packaged questions and grade supplied outputs without executing submissions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_tree(root, package):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate manifest key')
            result[key] = value
        return result
    manifest_path = root/'MANIFEST.json'
    if manifest_path.is_symlink():
        raise ValueError('Manifest symlinks are forbidden')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'), object_pairs_hook=unique)
    json.dumps(manifest, allow_nan=False)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or manifest.get('package') != package:
        raise ValueError('Invalid package manifest')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Manifest requires a nonempty file hash map')
    for relative, expected in files.items():
        if (not isinstance(relative, str) or not relative or '\\' in relative
                or Path(relative).is_absolute() or any(part in ('', '.', '..') for part in relative.split('/'))
                or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)):
            raise ValueError('Invalid manifest path or SHA256')
        target = root
        for part in relative.split('/'):
            target = target/part
            if target.is_symlink():
                raise ValueError('Manifest file paths cannot contain symlinks')
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError('Manifest path escapes package or is not a file')
        if file_hash(target) != expected:
            raise ValueError('Manifest hash mismatch: '+relative)
    return manifest


def verify_package():
    manifest = manifest_tree(ROOT, 'private_evaluator')
    for folder in ('financial_evals', 'source_grounded', 'quant100', 'workflows', 'research58'):
        directory = ROOT/folder
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Missing or symlinked bank directory: '+folder)
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlinks are forbidden in bank directories')
            if path.is_file() and '__pycache__' not in path.relative_to(ROOT).parts:
                if path.relative_to(ROOT).as_posix() not in manifest['files']:
                    raise ValueError('Unlisted bank or runtime file: '+path.relative_to(ROOT).as_posix())
    expected = manifest.get('solver_manifest_sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Missing solver manifest hash binding')
    solver = ROOT.parent/'questions'
    if solver.exists() or solver.is_symlink():
        if solver.is_symlink() or not solver.is_dir() or (solver/'MANIFEST.json').is_symlink():
            raise ValueError('Invalid sibling questions package')
        if file_hash(solver/'MANIFEST.json') != expected:
            raise ValueError('Sibling questions manifest hash mismatch')
        manifest_tree(solver, 'questions')


def read_value(path):
    value = core.read_json(Path(path))
    json.dumps(value, allow_nan=False)  # Also rejects exponent overflow such as 1e999.
    return value


def load_catalog():
    verify_package()
    # Validate runtime bytes before importing their executable Python modules.
    global core, quant, workflows, research, check_controls
    from financial_evals import core, quant, workflows, research
    from financial_evals.cli import check_controls
    catalog = {}
    paths = sorted((ROOT/'source_grounded/cases').glob('*.json'))
    if len(paths) != 36:
        raise core.ContractError('Expected 36 source-grounded cases')
    for path in paths:
        question, _, _ = core.load_case(ROOT/'source_grounded', path.stem)
        catalog[question['id']] = ('source', question)
    for folder, count, first, validator in (
            ('quant100', 100, 1, quant.validate_question),
            ('workflows', 6, 101, workflows.validate_case)):
        rows = read_value(ROOT/folder/'evaluator.json')
        expected = {f'q{i:03}' for i in range(first, first+count)}
        if not isinstance(rows, list) or len(rows) != count:
            raise core.ContractError('Unexpected question count: '+folder)
        for question in rows:
            validator(question)
            if question['id'] in catalog:
                raise core.ContractError('Duplicate question identifier')
            catalog[question['id']] = (folder, question)
        if {q['id'] for q in rows} != expected:
            raise core.ContractError('Unexpected question identifiers: '+folder)
    additions=read_value(ROOT/'research58/evaluator.json')
    if not isinstance(additions,list) or len(additions)!=58:
        raise core.ContractError('Expected 58 research cases')
    if [q['id'] for q in additions]!=[f'q{i:03}' for i in range(107,165)]:
        raise core.ContractError('Research question identities/order differ')
    for question in additions:
        research.validate_case(question)
        expected_mode='single' if int(question['id'][1:])<=136 else 'multi'
        if question['mode']!=expected_mode or question['id'] in catalog:
            raise core.ContractError('Research mode or identity mismatch')
        catalog[question['id']]=('research58',question)
    return catalog


def grade_response(track, question, response_path, variant_name=None, turn=1):
    if track!='research58' and turn!=1:
        raise core.ContractError('Turn selection applies only to research cases')
    if track=='research58' and variant_name is not None:
        raise core.ContractError('Research cases use turns, not workflow variants')
    if variant_name == 'base':
        variant_name = None
    variant = None
    if variant_name is not None:
        if track != 'workflows':
            raise core.ContractError('Variants apply only to workflow cases')
        variants = {v['name']: v for v in question['variants']}
        if variant_name not in variants:
            raise core.ContractError('Unknown workflow variant')
        variant = variants[variant_name]
    error = None
    try:
        response = read_value(response_path)
    except (ValueError, UnicodeError, RecursionError) as exc:
        response, error = None, 'Invalid response JSON: '+type(exc).__name__
    if track=='research58':
        result=research.grade_turn(question,turn,response)
        if error:
            result['response_error']=error
        return result
    identity = 'case_id' if track == 'source' else 'question_id'
    valid = (isinstance(response, dict) and response.get(identity) == question['id']
             and isinstance(response.get('answers'), dict))
    if not valid:
        error = error or 'Invalid response identity or answer object'
    if track == 'source':
        # An empty, correctly identified answer object retains every field in the denominator.
        safe_response = response if valid else {'case_id': question['id'], 'answers': {}}
        result = core.grade(ROOT/'source_grounded', question['id'], safe_response)
        result['expected_fields'] = len(question['output_fields'])
        result['correct_fields'] = sum(f['value_correct'] for f in result['fields'])
    elif track == 'quant100':
        result = quant.score(question, response if valid else None)
    else:
        result = workflows.grade_answers(question, response['answers'] if valid else None, variant)
        result['variant'] = variant_name or 'base'
    if error:
        result['response_error'] = error
    return result


def controls(catalog):
    source_checks, numeric_checks, research_checks = [], [], []
    for cid, (track, question) in catalog.items():
        if track == 'source':
            source_checks.extend(check_controls(ROOT/'source_grounded', cid,
                                 read_value(ROOT/'source_grounded/controls'/f'{cid}.json')))
            continue
        if track=='research58':
            research_checks.extend(research.control_results(question))
            continue
        grade = lambda answers: quant.score(question, {'question_id': cid, 'answers': answers})
        numeric_checks.append(grade(question['answer'])['all_correct'])
        for wrong in question['wrong_answers']:
            numeric_checks.append(not grade(dict(question['answer'], **wrong['answers']))['all_correct'])
        for variant in question.get('variants', []):
            numeric_checks.append(workflows.grade_answers(question, variant['answer'], variant)['all_correct'])
            numeric_checks.append(not workflows.grade_answers(question, question['answer'], variant)['all_correct'])
    passed = all(r['matches'] for r in source_checks) and all(numeric_checks) and all(research_checks)
    return {'passed': passed, 'source_control_count': len(source_checks),
            'quantitative_and_workflow_control_count': len(numeric_checks),
            'research_control_count': len(research_checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate')
    commands.add_parser('controls', aliases=['control-check'])
    grading = commands.add_parser('grade')
    grading.add_argument('question_id')
    grading.add_argument('response', type=Path)
    grading.add_argument('--variant')
    grading.add_argument('--turn',type=int,default=1)
    args = parser.parse_args()
    try:
        catalog = load_catalog()  # Broken evaluator data is never scored as a solver failure.
        if args.command == 'validate':
            result = {'valid': True, 'questions': len(catalog), 'source_cases': 36,
                      'quant_questions': 100, 'workflow_cases': 6,
                      'research_single_turn_cases':30,'research_multi_turn_cases':28,
                      'research_scored_turns':sum(len(q['turns']) for track,q in catalog.values() if track=='research58'),
                      'structured_outputs': sum(sum(len(t['outputs']) for t in q['turns']) if track=='research58'
                                                else len(q.get('outputs', q.get('output_fields', []))) for track,q in catalog.values())}
        elif args.command in ('controls', 'control-check'):
            result = controls(catalog)
        else:
            if args.question_id not in catalog:
                raise core.ContractError('Unknown question identifier')
            track, question = catalog[args.question_id]
            result = grade_response(track, question, args.response, args.variant, args.turn)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get('passed') is False:
            return 1
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        parser.exit(2, f'Evaluator error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
