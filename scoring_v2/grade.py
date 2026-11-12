"""Opt-in corrected scoring and public question export.

Run from the repository root with python -m scoring_v2.grade.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .common import FROZEN, catalog, read, stamped, verify, VERSION
from .amendments import quant_case
from . import engine, research, source
from financial_evals import quant, workflows


def grade_response(cid, response, turn=1, variant=None):
    track, case = catalog()[cid]
    if track != 'research58' and turn != 1:
        raise ValueError('Use sessions for sequential challenges')
    if variant is not None and track != 'workflows':
        raise ValueError('Variants apply only to workflow executed-answer scoring')
    if track in ('challenges', 'variants'):
        if case['mode'] == 'multi':
            raise ValueError('Use scoring_v2.sessions for stateful challenges')
        if isinstance(response, dict) and 'tool' in response:
            raise ValueError('Static grading accepts final responses only')
        session = engine.new_session(case, 'supplied')
        engine.public_view(session)
        engine.step(session, response)
        result = engine.report(session)
    elif track == 'research58':
        result = research.grade_turn(case, turn, response)
    elif track == 'source':
        result = source.grade(FROZEN / 'legacy/source_grounded', cid, response)
    elif track == 'quant100':
        result = quant.score(quant_case(case), response)
    else:
        if isinstance(response, dict) and 'code' in response:
            raise ValueError('Code submissions require python -m scoring_v2.workflow; '
                             'this explicit executed-answer endpoint never executes code')
        variants = {v['name']: v for v in case['variants']}
        if variant not in (None, 'base') and variant not in variants:
            raise ValueError('Unknown workflow variant')
        answers = response.get('answers') if isinstance(response, dict) and response.get('question_id') == cid else None
        result = workflows.grade_answers(case, answers, variants.get(variant))
        result.update(mode='executed-answers', variant=variant or 'base')
    return stamped(result)


def public_question(cid, turn=1, allow_holdout=False):
    track, case = catalog()[cid]
    if track != 'research58' and turn != 1:
        raise ValueError('Only opening packets may be exported; use sessions for later states')
    if track in ('challenges', 'variants'):
        if case.get('split') == 'holdout' and not allow_holdout:
            raise ValueError('Reserved release requires --allow-holdout')
        result = engine.public_view(engine.new_session(case, 'supplied'))
    elif track == 'quant100':
        result = quant.actor(quant_case(case))
    elif track == 'workflows':
        result = workflows.public_case(case)
        result['controller'] = 'python -m scoring_v2.workflow'
    elif track == 'research58':
        if turn != 1:
            raise ValueError('Use sessions to release later research turns')
        result = research.public_turn(case, turn)
    else:
        result = read(FROZEN.parent / 'questions/legacy/source_grounded/cases' / cid / 'case.json')
    return dict(deepcopy(result), scorer_version=VERSION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['grade', 'export', 'validate'])
    parser.add_argument('case_id', nargs='?')
    parser.add_argument('response', type=Path, nargs='?')
    parser.add_argument('--turn', type=int, default=1)
    parser.add_argument('--variant')
    parser.add_argument('--allow-holdout', action='store_true')
    args = parser.parse_args()
    verify()
    if args.action == 'validate':
        result = stamped({'cases': len(catalog()), 'baseline_verified': True})
    elif args.action == 'export':
        result = public_question(args.case_id, args.turn, args.allow_holdout)
    else:
        if args.response is None:
            parser.error('grade requires a response file')
        try:
            response = read(args.response)
        except (ValueError, UnicodeError, RecursionError):
            response = None
        result = grade_response(args.case_id, response, args.turn, args.variant)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise SystemExit(str(exc))
