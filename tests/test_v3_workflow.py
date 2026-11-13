"""V3 workflow scoring uses the unchanged isolated execution boundary."""
from copy import deepcopy
import unittest
import os

from scoring_v2 import workflow as old_workflow
from scoring_v3 import VERSION, constraints, workflow
from scoring_v3.common import binding, catalog


class VersionedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = next(case for cid, (track, case) in catalog().items()
                        if track == 'workflows' and constraints.public_contract(cid)['integer_fields'])
        cls.field = next(iter(constraints.public_contract(cls.case['id'])['integer_fields']))

    def executor(self, mutate=False):
        def run(code, inputs):
            row = next(v for v in [self.case] + self.case['variants'] if v['inputs'] == inputs)
            outputs = deepcopy(row['answer'])
            if mutate:
                value = outputs[self.field]
                outputs[self.field] = value + 5e-7 if type(value) in (int, float) else [value[0] + 5e-7, *value[1:]]
            return {'ok': True, 'outputs': outputs}
        return run

    def test_all_three_instances_apply_semantic_integer_rule(self):
        for mutate in [False, True]:
            state = workflow.start(self.case)
            for revision in range(2):
                workflow.submit(state, {'code': 'def solve(data): return {}'}, self.executor(mutate), revision)
            result = workflow.report(state)
            self.assertEqual(result['all_correct'], not mutate)
            self.assertEqual(result['scorer_version'], VERSION)
            self.assertEqual(len(result['instances']), 3)
            if mutate:
                self.assertTrue(all(self.field in r['grade']['integer_constraint_failures'] for r in result['instances']))

    def test_predecessor_and_successor_states_cannot_cross_controllers(self):
        old, current = old_workflow.start(self.case), workflow.start(self.case)
        for module, state in [(workflow, old), (old_workflow, current)]:
            for call in [lambda: module.public_view(state), lambda: module.report(state),
                         lambda: module.submit(state, {'code': 'x'}, self.executor(), 0)]:
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    call()

    def test_public_packet_has_integer_contract_and_no_hidden_results(self):
        state = workflow.start(self.case)
        opening = workflow.public_view(state)
        self.assertEqual(opening['scorer_binding'], binding())
        self.assertIn(self.field, opening['question']['scoring_corrections']['integer_fields'])
        for key in ['answer', 'variants', 'reference_code']:
            self.assertNotIn(key, opening['question'])
        workflow.submit(state, {'code': 'x'}, self.executor(), 0)
        feedback = workflow.public_view(state)
        self.assertNotIn('results', feedback)
        self.assertNotIn('instances', feedback)
        self.assertEqual(state['revision'], 1)

    def test_infrastructure_failure_preserves_attempt_and_candidate_failure_consumes_it(self):
        state = workflow.start(self.case)
        before = deepcopy(state)
        def unavailable(code, inputs):
            raise workflow.ExecutorUnavailable('Unavailable')
        with self.assertRaises(workflow.ExecutorUnavailable):
            workflow.submit(state, {'code': 'x'}, unavailable, 0)
        self.assertEqual(state, before)
        workflow.submit(state, {'code': 'x'}, lambda code, inputs: {'ok': False, 'error': 'Candidate failure'}, 0)
        self.assertEqual(state['revision'], 1)

    def test_execution_boundary_is_shared_without_monkeypatching(self):
        self.assertIs(workflow.DockerExecutor, old_workflow.DockerExecutor)
        self.assertIs(workflow.envelope, old_workflow.envelope)
        self.assertEqual(old_workflow.VERSION, 'financial-reasoning-v2.0.0')
        self.assertEqual(workflow.VERSION, VERSION)


@unittest.skipUnless(os.environ.get('RUN_WORKFLOW_DOCKER') == '1', 'Set RUN_WORKFLOW_DOCKER=1 for real Docker execution')
class VersionedWorkflowContainerTests(unittest.TestCase):
    def test_reference_and_fractional_count_program_through_real_executor(self):
        case = catalog()['q103'][1]
        executor = workflow.DockerExecutor('financial-evals-workflow:v2')
        for mutate in [False, True]:
            state = workflow.start(case)
            state['execution_image'] = executor.image_id
            code = case['reference_code']
            if mutate:
                code += '\n_original = solve\ndef solve(data):\n    result = _original(data)\n    result["chosen_lookback_months"] += 0.0000005\n    return result\n'
            for revision in range(2):
                workflow.submit(state, {'code': code}, executor, revision)
            report = workflow.report(state)
            self.assertEqual(report['all_correct'], not mutate)
            self.assertTrue(all(r['execution_ok'] for r in report['instances']))
            if mutate:
                self.assertTrue(all(r['grade']['integer_constraint_failures'] == ['chosen_lookback_months']
                                    for r in report['instances']))


if __name__ == '__main__':
    unittest.main()
