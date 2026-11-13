"""Two-submission workflow controller with isolated container execution."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .common import VERSION, ROOT, binding, catalog, read, stamped, number
from .sessions import outside, save
from financial_evals import workflows

from scoring_v2.workflow import (
    LIMIT, MAX_CODE, ExecutorUnavailable, native, failure, envelope, DockerExecutor,
)
from .static import workflow_answers
from . import constraints


def start(question):
    workflows.validate_case(question)
    return {'scorer_version': VERSION, 'binding': binding(), 'revision': 0,
            'question': deepcopy(question), 'history': [], 'finished': False}


def check(state):
    if state.get('scorer_version') != VERSION or state.get('binding') != binding():
        raise ValueError('Workflow scorer/bank binding differs')


def public_view(state):
    check(state)
    result = {'scorer_version': VERSION, 'scorer_binding': state['binding'],
              'session_revision': state['revision'],
              'finished': state['finished'], 'history': deepcopy(state['history'])}
    if not state['finished']:
        result.update(question=constraints.annotate(workflows.public_case(state['question']), state['question']['id']),
                      submission='first' if state['revision'] == 0 else 'final')
    return result


def submit(state, response, executor, revision):
    check(state)
    if type(revision) is not int or revision != state['revision'] or state['finished']:
        raise ValueError('Stale revision or completed workflow')
    try:
        valid = (type(response) is dict and set(response) == {'code'} and
                 type(response['code']) is str and 0 < len(response['code'].encode()) <= MAX_CODE)
    except UnicodeError:
        valid = False
    case = state['question']
    # Finish all executions before committing the attempt, so infrastructure errors
    # leave state retryable. Candidate execution errors still consume the attempt.
    if revision == 0:
        result = envelope(executor(response['code'], deepcopy(case['inputs']))) if valid else failure('Expected a code-only JSON object within the code limit')
        state['history'].extend([{'role': 'assistant', 'content': deepcopy(response)},
                                 {'role': 'tool', 'content': result},
                                 {'role': 'user', 'content': 'Submit your final code. No further execution feedback will be shown.'}])
    else:
        results = []
        for variant in [None] + case['variants']:
            inputs = case['inputs'] if variant is None else variant['inputs']
            result = envelope(executor(response['code'], deepcopy(inputs))) if valid else failure('Invalid code submission')
            grade = workflow_answers(case, result.get('outputs') if result['ok'] else None, variant)
            results.append({'variant': 'base' if variant is None else variant['name'],
                            'execution_ok': result['ok'], 'error': result.get('error'), 'grade': grade})
        state['results'] = results
        state['history'].append({'role': 'assistant', 'content': deepcopy(response)})
        state['finished'] = True
    state['revision'] += 1


def report(state):
    check(state)
    results = deepcopy(state.get('results', []))
    return stamped({'case_id': state['question']['id'], 'mode': 'isolated-code',
                    'completed': state['finished'], 'session_revision': state['revision'],
                    'execution_image': state.get('execution_image'), 'instances': results,
                    'all_correct': len(results) == 3 and all(r['grade']['all_correct'] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'submit', 'resume', 'report'])
    parser.add_argument('case_id', nargs='?')
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--image', default='financial-evals-workflow:v2')
    parser.add_argument('--response', type=Path)
    parser.add_argument('--revision', type=int)
    args = parser.parse_args()
    private = args.session.resolve()
    public = args.out.resolve() if args.out else None
    outside(private, public)
    if args.action != 'report' and public is None:
        parser.error('--out is required')
    private.mkdir(parents=True, exist_ok=True)
    lock = private / 'coordinator.lock'
    with lock.open('x') as handle:
        handle.write('Coordinator active\n')
    try:
        path = private / 'state.json'
        if args.action == 'start':
            if path.exists():
                raise ValueError('Session already exists')
            track, case = catalog()[args.case_id]
            if track != 'workflows':
                raise ValueError('Select q101 through q106')
            executor = DockerExecutor(args.image)
            state = start(case)
            state['execution_image'] = executor.image_id
        else:
            state = read(path)
            check(state)
        if args.action == 'report':
            print(json.dumps(report(state), indent=2, allow_nan=False))
            return
        if args.action == 'submit':
            if args.response is None:
                raise ValueError('--response is required')
            executor = DockerExecutor(state['execution_image'])
            try:
                if args.response.stat().st_size > 2 * MAX_CODE:
                    response = None
                else:
                    response = read(args.response)
            except (ValueError, UnicodeError, RecursionError):
                response = None
            submit(state, response, executor, args.revision)
        view = public_view(state)
        save(path, state)
        public.mkdir(parents=True, exist_ok=True)
        save(public / 'current.json', view)
        print(json.dumps({'scorer_version': VERSION, 'revision': state['revision'], 'finished': state['finished']}))
    finally:
        lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, RecursionError, ExecutorUnavailable) as exc:
        raise SystemExit(str(exc))
