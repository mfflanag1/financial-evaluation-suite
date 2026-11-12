"""Protocol regressions; optional real-container checks never run code on the host."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scoring_v2.common import ROOT, catalog
from scoring_v2 import workflow as w


class WorkflowProtocol(unittest.TestCase):
    def setUp(self):
        self.case = catalog()['q101'][1]

    def executor(self, calls):
        def run(code, inputs):
            calls.append(deepcopy(inputs))
            rows = [self.case] + self.case['variants']
            answer = next(row['answer'] for row in rows if row['inputs'] == inputs)
            return {'ok': True, 'outputs': deepcopy(answer)}
        return run

    def test_two_submissions_base_feedback_then_hidden_grading(self):
        state = w.start(self.case)
        calls = []
        execute = self.executor(calls)
        code = {'code': 'def solve(data): return {}'}
        w.submit(state, code, execute, 0)
        self.assertEqual(calls, [self.case['inputs']])
        public = w.public_view(state)
        self.assertEqual(public['history'][1]['content']['outputs'], self.case['answer'])
        self.assertNotIn('results', public)
        self.assertNotIn('variants', public['question'])
        self.assertNotIn('reference_code', public['question'])
        self.assertNotIn('answer', public['question'])
        self.assertEqual(w.report(state)['instances'], [])
        w.submit(state, code, execute, 1)
        self.assertEqual(len(calls), 4)
        self.assertTrue(w.report(state)['all_correct'])
        final = w.public_view(state)
        self.assertTrue(final['finished'])
        self.assertEqual(final['history'][:-1], public['history'])
        self.assertNotIn('instances', final)
        with self.assertRaises(ValueError):
            w.submit(state, code, execute, 2)

    def test_base_only_constants_fail_hidden_instances(self):
        state = w.start(self.case)
        execute = lambda code, inputs: {'ok': True, 'outputs': self.case['answer']}
        for n in range(2):
            w.submit(state, {'code': 'def solve(data): return {}'}, execute, n)
        result = w.report(state)
        self.assertTrue(result['instances'][0]['grade']['all_correct'])
        self.assertFalse(result['all_correct'])
        self.assertTrue(all(not r['grade']['all_correct'] for r in result['instances'][1:]))

    def test_malformed_consumes_attempt_without_execution(self):
        for invalid in (None, {}, {'code': 1}, {'code': ''}, {'code': 'x' * (w.MAX_CODE + 1)},
                        {'code': 'x', 'answers': {}}, {'code': '\ud800'}):
            state = w.start(self.case)
            def no_execution(code, inputs):
                self.fail('Malformed submission executed')
            w.submit(state, invalid, no_execution, 0)
            self.assertEqual(state['revision'], 1)
            self.assertFalse(w.public_view(state)['history'][1]['content']['ok'])
            w.submit(state, invalid, no_execution, 1)
            self.assertEqual(len(w.report(state)['instances']), 3)
            self.assertFalse(w.report(state)['all_correct'])

    def test_bad_execution_results_fail(self):
        for value in (None, {}, {'ok': True, 'outputs': []}, {'ok': True, 'outputs': {'x': float('nan')}},
                      {'ok': True, 'outputs': {'x': 10**400}}, {'ok': False, 'error': 'candidate failure'}):
            state = w.start(self.case)
            execute = lambda code, inputs: value
            for n in range(2):
                w.submit(state, {'code': 'def solve(data): return {}'}, execute, n)
            self.assertFalse(w.report(state)['all_correct'])

    def test_revision_binding_and_infrastructure_failure(self):
        state = w.start(self.case)
        before = deepcopy(state)
        def unavailable(code, inputs):
            raise w.ExecutorUnavailable('missing image')
        with self.assertRaises(w.ExecutorUnavailable):
            w.submit(state, {'code': 'def solve(data): return {}'}, unavailable, 0)
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            w.submit(state, {'code': 'x'}, unavailable, 1)
        state['binding'] = 'frozen'
        with self.assertRaises(ValueError):
            w.public_view(state)

    def test_cleanup_survives_client_wait_timeout(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired('docker', 5)
        removed = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(w.subprocess, 'run', return_value=removed) as run:
            w.DockerExecutor.cleanup('test-container', process)
        process.kill.assert_called_once()
        self.assertEqual(run.call_args.args[0], ['docker', 'rm', '-f', 'test-container'])


@unittest.skipUnless(os.environ.get('RUN_WORKFLOW_DOCKER') == '1', 'Opt-in Docker integration')
class WorkflowContainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = w.DockerExecutor('financial-evals-workflow:v2')

    def test_all_six_reference_programs_and_hidden_variants(self):
        for cid in [f'q{i}' for i in range(101, 107)]:
            case = catalog()[cid][1]
            state = w.start(case)
            for n in range(2):
                w.submit(state, {'code': case['reference_code']}, self.executor, n)
            result = w.report(state)
            self.assertTrue(result['all_correct'], (cid, result))

    def test_exceptions_nonfinite_output_and_no_host_files(self):
        for code in ("def solve(data): raise ValueError('test failure')",
                     "def solve(data): return {'x':float('nan')}",
                     "def solve(data): return {'x': (1,2)}"):
            self.assertFalse(self.executor(code, {})['ok'])
        result = self.executor("import os\ndef solve(data): return {'host_absent':not os.path.exists('/private_evaluator'), 'uid':os.getuid()}", {})
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['outputs'], {'host_absent': True, 'uid': 65534})

    def test_timeout_output_limits_and_container_cleanup(self):
        executor = w.DockerExecutor(self.executor.image_id, timeout=2, max_output=65536)
        for code in ("def solve(data):\n while True: pass",
                     "import os\ndef solve(data):\n while True: os.write(2,b'x'*65536)"):
            result = executor(code, {})
            self.assertFalse(result['ok'])
            self.assertTrue('timed out' in result['error'] or 'size limit' in result['error'], result)
        running = subprocess.check_output(['docker', 'ps', '-a', '--filter', 'name=financial-eval-', '--format', '{{.Names}}'], text=True)
        self.assertEqual(running.strip(), '')

    def test_candidate_exit_code_consumes_attempt(self):
        state = w.start(catalog()['q101'][1])
        code = {'code': 'import os\ndef solve(data): os._exit(127)'}
        w.submit(state, code, self.executor, 0)
        self.assertEqual(state['revision'], 1)
        self.assertFalse(w.public_view(state)['history'][1]['content']['ok'])
        w.submit(state, code, self.executor, 1)
        self.assertTrue(state['finished'])
        self.assertFalse(w.report(state)['all_correct'])

    def test_cli_round_trip_and_stale_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [sys.executable, '-B', '-m', 'scoring_v2.workflow']
            paths = ['--session', str(root / 'private'), '--out', str(root / 'public')]
            def run(args):
                return subprocess.run(base + args, cwd=ROOT, capture_output=True, text=True)
            result = run(['start', 'q101'] + paths)
            self.assertEqual(result.returncode, 0, result.stderr)
            response = root / 'response.json'
            response.write_text(json.dumps({'code': catalog()['q101'][1]['reference_code']}))
            for revision in (0, 1):
                args = ['submit'] + paths + ['--response', str(response), '--revision', str(revision)]
                result = run(args)
                self.assertEqual(result.returncode, 0, result.stderr)
                stale = run(args)
                self.assertNotEqual(stale.returncode, 0)
            result = run(['report', '--session', str(root / 'private')])
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['all_correct'], report)
            self.assertTrue(report['execution_image'].startswith('sha256:'))


if __name__ == '__main__':
    unittest.main()
