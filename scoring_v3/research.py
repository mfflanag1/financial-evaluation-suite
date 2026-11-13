"""Stage-aware integer corrections over the preserved research scorer."""
from copy import deepcopy

from scoring_v2 import research as predecessor
from . import VERSION, constraints
from .static import checked_answers

reference_response = predecessor.reference_response
validate_case = predecessor.validate_case


def public_turn(case, number=1):
    result = constraints.annotate(predecessor.public_turn(case, number), case['id'], number)
    result['scorer_version'] = VERSION
    return result


def grade_turn(case, number, response):
    predecessor.require(type(number) is int and 1 <= number <= len(case['turns']), 'Invalid turn')
    adjusted = deepcopy(response)
    rejected = []
    if isinstance(adjusted, dict):
        adjusted['answers'], rejected = checked_answers(
            case['id'], number, adjusted.get('answers'), case['turns'][number - 1]['answer'])
    result = predecessor.grade_turn(case, number, adjusted)
    result.update(scorer_version=VERSION, integer_constraint_failures=rejected)
    return result


def grade_episode(case, responses):
    predecessor.require(isinstance(responses, list), 'Responses must be an ordered list')
    predecessor.require(len(responses) <= len(case['turns']), 'Too many responses')
    grades = [grade_turn(case, i, responses[i - 1] if i <= len(responses) else None)
              for i in range(1, len(case['turns']) + 1)]
    total = sum(r['expected_fields'] for r in grades)
    return {'scorer_version': VERSION, 'question_id': case['id'],
            'expected_turns': len(grades), 'submitted_turns': len(responses),
            'all_turns_correct': all(r['all_correct'] for r in grades),
            'final_turn_correct': grades[-1]['all_correct'],
            'critical_path_pass': all(r['critical_pass'] for r in grades),
            'field_accuracy_micro': sum(r['correct_fields'] for r in grades) / total,
            'turn_accuracy_mean': sum(r['field_accuracy'] for r in grades) / len(grades),
            'expected_fields': total, 'turns': grades}
