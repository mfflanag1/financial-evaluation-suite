"""Candidate-visible projections for fixed-case arithmetic audits.

Later turns share files with evaluator answers. These allowlisted projections
copy no answer values. Core output names and top-level types come from the
authored template, as in the public renderer. They describe frozen snapshots,
not the state reached by arbitrary candidate actions.
"""
from copy import deepcopy
import json
from pathlib import Path

from scoring_v3.research import public_turn

ROOT = Path(__file__).resolve().parents[1]


def _read_case(path, cid):
    cases = json.loads(path.read_text())
    matches = [case for case in cases if case['id'] == cid]
    if len(matches) != 1:
        raise ValueError('Unknown or duplicate case ID')
    return matches[0]


def _shape(value):
    return {bool: 'boolean', int: 'number', float: 'number', str: 'string',
            list: 'array', dict: 'object', type(None): 'null'}[type(value)]


def project_core(case):
    turns = []
    for number, stage in enumerate(case['stages'], 1):
        turns.append({
            'question_id': case['id'], 'title': case['title'], 'turn': number,
            'total_turns': len(case['stages']), 'prompt': stage['prompt'],
            'inputs': stage['inputs'],
            'sources': [doc for doc in case['documents']
                        if doc['release_stage'] <= number
                        and doc['id'] in case['oracle_document_ids']],
            'outputs': {name: {'type': _shape(value),
                               'description': stage.get('descriptions', {}).get(name)}
                        for name, value in stage['answers'].items()},
        })
    return deepcopy({'question_id': case['id'], 'turns': turns,
                     'audit_scope': 'Authored frozen stage inputs; simulated branches and off-reference trajectories require separate validation.',
                     'simulated': bool(case.get('simulator'))})


def project_research(case):
    return {'question_id': case['id'],
            'turns': [public_turn(case, number)
                      for number in range(1, len(case['turns']) + 1)]}


def load_core_packet(cid):
    return project_core(_read_case(ROOT / 'private_evaluator/challenges/cases.json', cid))


def load_variant_packet(cid):
    return project_core(_read_case(ROOT / 'private_evaluator/variants/cases.json', cid))


def load_research_packet(cid):
    return project_research(_read_case(ROOT / 'private_evaluator/legacy/research58/evaluator.json', cid))


def load_workflow_packet(cid):
    if cid not in {f'q{number:03d}' for number in range(101, 107)}:
        raise ValueError('Unknown workflow ID')
    return json.loads((ROOT / f'questions/legacy/workflows/questions/{cid}.json').read_text())
