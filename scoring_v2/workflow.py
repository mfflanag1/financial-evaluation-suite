"""Two-submission workflow controller with isolated container execution."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from .common import VERSION, ROOT, binding, catalog, read, stamped, number
from .sessions import outside, save
from financial_evals import workflows

LIMIT = 1_048_576
MAX_CODE = 65_536


class ExecutorUnavailable(RuntimeError):
    """Infrastructure failure; do not consume a candidate attempt."""


def native(value, depth=0):
    if depth > 32:
        return False
    if value is None or type(value) in (str, bool):
        return True
    if type(value) in (int, float):
        return number(value)
    if type(value) is list:
        return all(native(v, depth + 1) for v in value)
    return type(value) is dict and all(type(k) is str and native(v, depth + 1) for k, v in value.items())


def failure(message):
    return {'ok': False, 'error': message}


def envelope(value):
    if type(value) is not dict or type(value.get('ok')) is not bool or not native(value):
        return failure('Invalid execution result')
    try:
        if len(json.dumps(value, allow_nan=False).encode()) > LIMIT:
            return failure('Execution result exceeds size limit')
    except (ValueError, RecursionError):
        return failure('Invalid execution result')
    if value['ok']:
        if type(value.get('outputs')) is not dict:
            return failure('Program must return a JSON object')
        result = {'ok': True, 'outputs': deepcopy(value['outputs'])}
    else:
        result = failure(str(value.get('error', 'Execution failed'))[:600])
    for key in ('stdout', 'stderr'):
        if type(value.get(key)) is str:
            result[key] = value[key][:4096]
    return result


class DockerExecutor:
    """No host execution fallback, host mounts or network access."""
    def __init__(self, image, timeout=20, max_output=LIMIT):
        if not number(timeout) or not 0 < timeout <= 120:
            raise ValueError('Timeout must be in (0,120] seconds')
        if type(max_output) is not int or not 1024 <= max_output <= LIMIT:
            raise ValueError('Invalid output limit')
        try:
            probe = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                   capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorUnavailable('Docker is unavailable') from exc
        if probe.returncode or not probe.stdout.strip().startswith('sha256:'):
            raise ExecutorUnavailable('Build the workflow image locally first; automatic pulls are disabled')
        self.image_id = probe.stdout.strip()
        self.timeout = timeout
        self.max_output = max_output

    def __call__(self, code, inputs):
        payload = json.dumps({'code': code, 'inputs': inputs}, allow_nan=False).encode()
        if len(payload) > 2 * LIMIT:
            return failure('Execution input exceeds size limit')
        name = 'financial-eval-' + uuid.uuid4().hex
        args = ['docker', 'run', '--rm', '-i', '--name', name, '--pull=never',
                '--network=none', '--read-only', '--user=65534:65534',
                '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--memory=512m', '--memory-swap=512m', '--cpus=1', '--pids-limit=64',
                '--ulimit=core=0:0', '--ulimit=nofile=64:64',
                '--log-driver=none', '--entrypoint=python', self.image_id,
                '-I', '-B', '/opt/workflow/worker.py']
        with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            stdin.write(payload)
            stdin.seek(0)
            try:
                process = subprocess.Popen(args, stdin=stdin, stdout=stdout, stderr=stderr)
            except OSError as exc:
                raise ExecutorUnavailable('Cannot start Docker') from exc
            error = None
            deadline = time.monotonic() + self.timeout
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        error = 'Execution timed out'
                        break
                    if any(os.fstat(f.fileno()).st_size > self.max_output for f in (stdout, stderr)):
                        error = 'Execution output exceeds size limit'
                        break
                    time.sleep(.02)
                if error:
                    return failure(error)
                if any(os.fstat(f.fileno()).st_size > self.max_output for f in (stdout, stderr)):
                    return failure('Execution output exceeds size limit')
                if process.returncode:
                    return failure('Execution exited with status ' + str(process.returncode))
                stdout.seek(0)
                try:
                    def unique(items):
                        result = {}
                        for key, value in items:
                            if key in result:
                                raise ValueError('Duplicate output key')
                            result[key] = value
                        return result
                    result = json.loads(stdout.read(self.max_output + 1), object_pairs_hook=unique)
                    return envelope(result)
                except (ValueError, UnicodeError, RecursionError):
                    return failure('Invalid JSON from execution')
            finally:
                self.cleanup(name, process)

    @staticmethod
    def cleanup(name, process):
        try:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            # Removing the container is mandatory even if the client cannot be reaped.
            try:
                removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True, timeout=10)
                if removed.returncode and 'No such container' not in removed.stderr:
                    raise ExecutorUnavailable('Could not confirm container cleanup: ' + name)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ExecutorUnavailable('Could not confirm container cleanup: ' + name) from exc


def start(question):
    workflows.validate_case(question)
    return {'scorer_version': VERSION, 'binding': binding(), 'revision': 0,
            'question': deepcopy(question), 'history': [], 'finished': False}


def check(state):
    if state.get('scorer_version') != VERSION or state.get('binding') != binding():
        raise ValueError('Workflow scorer/bank binding differs')


def public_view(state):
    check(state)
    result = {'scorer_version': VERSION, 'session_revision': state['revision'],
              'finished': state['finished'], 'history': deepcopy(state['history'])}
    if not state['finished']:
        result.update(question=workflows.public_case(state['question']),
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
            grade = workflows.grade_answers(case, result.get('outputs') if result['ok'] else None, variant)
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
