"""Exact selected integer leaves layered onto unchanged static comparators."""
from copy import deepcopy

from scoring_v2.amendments import quant_case
from financial_evals import quant, workflows
from . import constraints


def checked_answers(cid, turn, answers, reference):
    if not isinstance(answers, dict):
        return answers, []
    adjusted = deepcopy(answers)
    rejected = []
    for field, expected in reference.items():
        if field in answers and not constraints.exact(cid, turn, field, answers[field], expected):
            adjusted[field] = None
            rejected.append(field)
    return adjusted, rejected


def quant_score(case, response):
    adjusted = deepcopy(response)
    rejected = []
    if isinstance(adjusted, dict):
        adjusted['answers'], rejected = checked_answers(case['id'], 1, adjusted.get('answers'), case['answer'])
    result = quant.score(quant_case(case), adjusted)
    result['integer_constraint_failures'] = rejected
    return result


def workflow_answers(case, answers, variant=None):
    reference = case['answer'] if variant is None else variant['answer']
    adjusted, rejected = checked_answers(case['id'], 1, answers, reference)
    result = workflows.grade_answers(case, adjusted, variant)
    result['integer_constraint_failures'] = rejected
    return result
