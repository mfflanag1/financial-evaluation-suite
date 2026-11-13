"""Program binding and real fresh-container controls, without model inference."""
import asyncio
from copy import deepcopy
from importlib.resources import files
import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from financial_selection.contract import PacketError
from financial_selection.generator import generate
from financial_selection import reference, replay, verifier

KEY = bytes(range(32))  # Public fixture key; not used for private evaluation.


class ReferenceCrosschecks(unittest.TestCase):
    def test_separate_formulas_match_on_keyed_conditions_and_splits(self):
        for split in ('train', 'dev', 'final'):
            for condition in ('mixed', 'positive', 'negative', 'unsupported', 'sparse'):
                packet = generate(KEY, split, 11, condition=condition).public_files
                candidate = reference.solve(packet)
                self.assertEqual(candidate, verifier.expected_policy(packet), (split, condition))
                self.assertEqual(verifier.grade(packet, json.dumps(candidate))['reward'], 1)


class ReplayBinding(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.public = generate(KEY, 'dev', 0, condition='positive').public_files
        self.fresh = [generate(KEY, 'dev', index, condition='positive').public_files for index in (1, 2)]
        self.policy = json.dumps(reference.solve(self.public)).encode()
        self.source = Path(reference.__file__).read_bytes()

    async def test_fixed_answer_program_cannot_earn_fresh_input_credit(self):
        # This program reproduces its initial answer, but is input-insensitive.
        with patch.object(replay, 'run_public_program', AsyncMock(return_value={
                'status': 'passed', 'policy_bytes': self.policy})) as run:
            result = await replay.verify_program(self.public, self.source, self.policy, self.fresh, runtime_config=None)
        self.assertEqual(run.await_count, 3)
        self.assertTrue(result['reproduction_verified'])
        self.assertEqual(result['packet_rewards'], [1, 0, 0])
        self.assertEqual(result['reward'], 1 / 3)

    async def test_initial_mismatch_zeros_aggregate_without_fresh_runs(self):
        other = json.dumps(verifier.expected_policy(self.public, [0, 0, 0, 0])).encode()
        self.assertNotEqual(other, self.policy)
        with patch.object(replay, 'run_public_program', AsyncMock(return_value={
                'status': 'passed', 'policy_bytes': other})) as run:
            result = await replay.verify_program(self.public, self.source, self.policy, self.fresh, runtime_config=None)
        self.assertEqual(run.await_count, 1)
        self.assertFalse(result['reproduction_verified'])
        self.assertEqual(result['reward'], 0)

    async def test_bool_integer_equality_does_not_bypass_reproduced_schema(self):
        policy = json.loads(self.policy)
        one = next((index for index, quantity in enumerate(policy['allocation']) if quantity == 1), None)
        if one is None:
            policy = verifier.expected_policy(self.public, [0, 0, 0, 0])
            submitted = json.dumps(policy).encode()
            policy['allocation'][0] = False
        else:
            submitted = self.policy
            policy['allocation'][one] = True
        with patch.object(replay, 'run_public_program', AsyncMock(return_value={
                'status': 'passed', 'policy_bytes': json.dumps(policy).encode()})):
            result = await replay.verify_program(self.public, self.source, submitted, self.fresh, runtime_config=None)
        self.assertFalse(result['reproduction_verified'])
        self.assertEqual(result['reward'], 0)

    async def test_infrastructure_failure_is_not_candidate_zero(self):
        with patch.object(replay, 'run_public_program', AsyncMock(side_effect=replay.ReplayInfrastructureError('service_failed'))):
            with self.assertRaises(replay.ReplayInfrastructureError):
                await replay.verify_program(self.public, self.source, self.policy, self.fresh, runtime_config=None)

    async def test_repeated_inputs_are_not_fresh_packet_evidence(self):
        with self.assertRaises(replay.ReplayInfrastructureError):
            await replay.verify_program(self.public, self.source, self.policy, [self.public], runtime_config=None)

    async def test_bad_public_packet_raises_before_candidate_scoring(self):
        bad = deepcopy(self.public)
        del bad['deployment.json']
        with self.assertRaises(PacketError):
            await replay.verify_program(bad, self.source, b'{}', self.fresh, runtime_config=None)

    async def test_bad_source_is_rejected_before_runtime_import(self):
        for source in (b'', b'\xff', b' ', b'x' * (replay.MAX_SOURCE_BYTES + 1)):
            result = await replay.run_public_program(self.public, source, runtime_config=None)
            self.assertEqual(result['status'], 'candidate_failure')


@unittest.skipUnless(os.environ.get('FINANCIAL_SELECTION_DOCKER_TESTS') == '1', 'set FINANCIAL_SELECTION_DOCKER_TESTS=1 for real Docker controls')
class RealContainerReplay(unittest.IsolatedAsyncioTestCase):
    async def test_reference_reproduces_and_solves_two_fresh_packets(self):
        from verifiers.v1.runtimes import DockerConfig
        public = generate(KEY, 'dev', 20).public_files
        fresh = [generate(KEY, 'dev', n).public_files for n in (21, 22)]
        source = Path(reference.__file__).read_bytes()
        result = await replay.verify_program(public, source, json.dumps(reference.solve(public)).encode(), fresh,
                                            runtime_config=DockerConfig(image=os.environ.get('FINANCIAL_SELECTION_DOCKER_IMAGE', 'python:3.12-slim'), allow=[]))
        self.assertEqual(result['packet_rewards'], [1, 1, 1])
        self.assertEqual(result['reward'], 1)
        self.assertTrue(result['reproduction_verified'])

    async def test_missing_output_and_timeout_are_candidate_failures(self):
        from verifiers.v1.runtimes import DockerConfig
        public = generate(KEY, 'dev', 23).public_files
        config = DockerConfig(image=os.environ.get('FINANCIAL_SELECTION_DOCKER_IMAGE', 'python:3.12-slim'), allow=[])
        for source, reason in [(b'pass\n', 'output_missing_or_unsafe'),
                               (b'import time; time.sleep(20)\n', 'analysis_timeout'),
                               (b'import os; os.symlink("/etc/passwd", "policy.json")\n', 'output_missing_or_unsafe')]:
            result = await replay.run_public_program(public, source, runtime_config=config, timeout_seconds=1)
            self.assertEqual((result['status'], result['reason']), ('candidate_failure', reason))

    async def test_solver_container_has_no_grader_or_generation_key(self):
        from verifiers.v1.runtimes import DockerConfig
        source = b'''import os, importlib.util, json
assert "FINANCIAL_SELECTION_KEY" not in os.environ
assert importlib.util.find_spec("financial_selection") is None
assert importlib.util.find_spec("financial_research") is None
assert set(os.listdir("/workspace")) == {"contract.json", "deployment.json", "feature_releases.csv", "observed_labels.csv", "universe.csv", "research.py"}
open("policy.json", "w").write("{}")
'''
        result = await replay.run_public_program(generate(KEY, 'dev', 24).public_files, source,
                                                runtime_config=DockerConfig(image=os.environ.get('FINANCIAL_SELECTION_DOCKER_IMAGE', 'python:3.12-slim'), allow=[]))
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(json.loads(result['policy_bytes']), {})


if __name__ == '__main__':
    unittest.main()
