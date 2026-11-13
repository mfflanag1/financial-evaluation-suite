"""Source evidence corrections with dependency-safe duplicate rejection."""
from copy import deepcopy

from scoring_v2 import source as predecessor
from financial_evals.core import normalized
from . import constraints

load_case = predecessor.load_case

DUPLICATE_RULE = (
    'Within each answer field, each (source_id, whitespace-normalized quote) pair '
    'must occur at most once. Distinct quotes from the same source are permitted. '
    'The same quote may support different answer fields. Duplicate direct citations '
    'invalidate that field evidence and cannot convey proof through support_refs. '
    'Existing source allowlists, quote verification and anchor requirements still apply.'
)


def grade(root, case_id, response):
    adjusted = deepcopy(response)
    duplicate_fields, integer_fields = [], []
    _, gold, _ = load_case(root, case_id)
    refs = {f['id']: f['expected'] for f in gold['fields']}
    answers = adjusted.get('answers') if isinstance(adjusted, dict) else None
    if isinstance(answers, dict):
        for field, expected in refs.items():
            answer = answers.get(field)
            if not isinstance(answer, dict):
                continue
            if 'value' in answer and not constraints.exact(case_id, 1, field, answer['value'], expected):
                answer['value'] = None
                integer_fields.append(field)
            evidence = answer.get('evidence')
            if not isinstance(evidence, list):
                continue
            seen = set()
            duplicate = False
            for cite in evidence:
                if not isinstance(cite, dict):
                    continue
                sid, quote = cite.get('source_id'), cite.get('quote')
                if not isinstance(sid, str) or not isinstance(quote, str):
                    continue
                key = (sid, normalized(quote))
                if key in seen:
                    duplicate = True
                seen.add(key)
            if duplicate:
                # The predecessor already propagates invalid evidence through
                # support_refs. An invalid sentinel prevents proof laundering.
                answer['evidence'] = evidence + [None]
                duplicate_fields.append(field)
    result = predecessor.grade(root, case_id, adjusted)
    result['duplicate_citation_fields'] = duplicate_fields
    result['integer_constraint_failures'] = integer_fields
    for field in result.get('fields', []):
        if field['id'] in duplicate_fields:
            field['failures'].append('duplicate_citation')
        if field['id'] in integer_fields:
            field['failures'].append('integer_value_or_type_mismatch')
    return result
