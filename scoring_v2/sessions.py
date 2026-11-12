"""Version-bound coordinator for corrected core and research cases."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .common import ROOT, VERSION, binding, catalog, read, stamped
from . import engine, research


def save(path, value):
    if path.is_symlink():
        raise ValueError('Symlinked state/output file')
    temporary = path.with_suffix('.tmp')
    created = False
    try:
        with temporary.open('x') as handle:
            created = True
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def outside(private, public=None):
    protected = [ROOT / name for name in ('private_evaluator', 'questions', 'scoring_v2')]
    for folder in [private] + ([public] if public else []):
        if any(folder.is_relative_to(p) or p.is_relative_to(folder) for p in protected):
            raise ValueError('Session files must be outside package directories')
    if public and (private.is_relative_to(public) or public.is_relative_to(private)):
        raise ValueError('Private state and solver output must be separate')


def start(cid, condition, allow_holdout=False):
    track, case = catalog()[cid]
    if track not in ('research58', 'challenges', 'variants'):
        raise ValueError('Use the static grader or workflow controller for this track')
    if case.get('split') == 'holdout' and not allow_holdout:
        raise ValueError('Reserved release requires --allow-holdout')
    if track == 'research58' and condition != 'supplied':
        raise ValueError('Research cases use supplied packets')
    state = {'scorer_version': VERSION, 'binding': binding(), 'revision': 0, 'track': track}
    if track == 'research58':
        state.update(case=case, responses=[], history=[], finished=False)
    else:
        state['engine'] = engine.new_session(case, condition)
    return state


def public_view(state):
    if state['track'] != 'research58':
        result = engine.public_view(state['engine'])
    elif state['finished']:
        result = {'finished': True}
    else:
        n = len(state['responses']) + 1
        question = research.public_turn(state['case'], n)
        result = {'question': question, 'history': deepcopy(state['history']) +
                  [{'role': 'user', 'content': deepcopy(question)}]}
    return dict(result, session_revision=state['revision'], scorer_version=VERSION)


def submit(state, response, revision):
    if state.get('scorer_version') != VERSION or state.get('binding') != binding():
        raise ValueError('Session scorer/bank binding differs')
    if type(revision) is not int or revision != state['revision']:
        raise ValueError('Stale or absent revision')
    if state['track'] != 'research58':
        engine.step(state['engine'], response)
    else:
        if state['finished']:
            raise ValueError('Session already complete')
        state['history'] = public_view(state)['history']
        state['history'].append({'role': 'assistant', 'content': deepcopy(response)})
        state['responses'].append(deepcopy(response))
        state['finished'] = len(state['responses']) == len(state['case']['turns'])
    state['revision'] += 1


def report(state):
    if state.get('binding') != binding() or state.get('scorer_version') != VERSION:
        raise ValueError('Session scorer/bank binding differs')
    result = (research.grade_episode(state['case'], state['responses']) if state['track'] == 'research58'
              else engine.report(state['engine']))
    return stamped(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'submit', 'resume', 'report'])
    parser.add_argument('case_id', nargs='?')
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--revision', type=int)
    parser.add_argument('--condition', choices=['supplied', 'retrieval'], default='supplied')
    parser.add_argument('--allow-holdout', action='store_true')
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
            state = start(args.case_id, args.condition, args.allow_holdout)
        else:
            state = read(path)
            if state.get('binding') != binding() or state.get('scorer_version') != VERSION:
                raise ValueError('Session scorer/bank binding differs')
        if args.action == 'report':
            print(json.dumps(report(state), indent=2, allow_nan=False))
            return
        if args.action == 'submit':
            if args.response is None:
                raise ValueError('--response is required')
            try:
                response = read(args.response)
            except (ValueError, UnicodeError, RecursionError):
                response = None
            submit(state, response, args.revision)
        view = public_view(state)
        save(path, state)
        public.mkdir(parents=True, exist_ok=True)
        save(public / 'current.json', view)
        print(json.dumps({'scorer_version': VERSION, 'revision': state['revision'],
                          'finished': view.get('finished', False)}))
    finally:
        lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise SystemExit(str(exc))
