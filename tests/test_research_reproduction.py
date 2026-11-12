"""Public-only source reproduction, semantic comparison, and failure attribution."""

import copy
from contextlib import asynccontextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "environments" / "financial_research" / "src" / "financial_research"
sys.path.insert(0, str(PACKAGE.parent))
from financial_research.generator import build_episode
from financial_research.reproduction import (ReproductionInfrastructureError, fresh_config,
    normalize_policy, parse_policy, policies_match, verify_reproduction)
from financial_research.verifier import reference_submission

HAS_PRIME = importlib.util.find_spec("verifiers") is not None


class ReproductionComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "public-reproduction-fixture-not-for-evaluation"}):
            cls.episode = build_episode(0)
        cls.policy = reference_submission(cls.episode)
        cls.contract = json.loads(cls.episode["public_files"]["contract.json"])

    def test_semantic_order_and_tolerances(self):
        other = copy.deepcopy(self.policy)
        other["audit"].reverse()
        other["trials"].reverse()
        other["trials"][0]["mean_net"] += 5e-9
        self.assertTrue(policies_match(self.policy, other, self.contract))
        other["trials"][0]["mean_net"] += 2e-8
        self.assertFalse(policies_match(self.policy, other, self.contract))

    def test_changed_decision_is_not_reproduced(self):
        other = copy.deepcopy(self.policy)
        other["decision"] = "abstain"
        other["selected"] = None
        self.assertFalse(policies_match(self.policy, other, self.contract))

    def test_schema_rejects_bools_duplicates_missing_and_huge_values(self):
        for value in (True, float("nan"), float("inf"), 10**1000):
            other = copy.deepcopy(self.policy)
            other["trials"][0]["mean_net"] = value
            with self.assertRaises(ValueError):
                normalize_policy(other, self.contract)
        for field in ("audit", "trials"):
            for duplicate in (False, True):
                other = copy.deepcopy(self.policy)
                if duplicate:
                    other[field][-1] = other[field][0]
                else:
                    other[field].pop()
                with self.assertRaises(ValueError):
                    normalize_policy(other, self.contract)

    def test_strict_json_and_size(self):
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff', b' '*(2*1024*1024+1)):
            with self.assertRaises((ValueError, UnicodeError)):
                parse_policy(payload)

    @unittest.skipUnless(HAS_PRIME, "Install pinned Prime optional dependency")
    def test_fresh_config_rejects_mounts_and_clears_inherited_configuration(self):
        from verifiers.v1.runtimes import DockerConfig, PrimeConfig, SubprocessConfig
        original = DockerConfig(image="python:3.12-slim", workdir="/other", allow=["*"], cpu=8, memory=8)
        config = fresh_config(original)
        self.assertEqual(config.workdir, "/workspace")
        self.assertEqual(config.allow, [])
        self.assertEqual((config.cpu, config.memory), (1, 1))
        for unsafe in (SubprocessConfig(),
                       DockerConfig(mounts={"/data": {"source": "/tmp"}}),
                       PrimeConfig(checkpoint="old-workspace")):
            with self.assertRaises(ReproductionInfrastructureError):
                fresh_config(unsafe)


@unittest.skipUnless(HAS_PRIME, "Install pinned Prime optional dependency")
class ReproductionPrivacy(unittest.IsolatedAsyncioTestCase):
    async def test_provider_cause_does_not_enter_native_error_trace(self):
        import verifiers.v1 as vf
        from verifiers.v1.runtimes import DockerConfig
        sentinel = "sensitive-provider-token-test-sentinel"

        @asynccontextmanager
        async def broken_provider(*args, **kwargs):
            raise RuntimeError(sentinel)
            yield  # Marks this as an async context manager without starting a runtime.

        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "public-reproduction-fixture-not-for-evaluation"}):
            episode = build_episode(0)
        with patch("verifiers.v1.runtimes.provision_runtime", broken_provider):
            try:
                await verify_reproduction(episode["public_files"], b"pass\n", reference_submission(episode),
                                          runtime_config=DockerConfig())
            except ReproductionInfrastructureError as error:
                trace = vf.Trace(task=vf.TraceTask(type="test", data=vf.TaskData(id="test", prompt="test")),
                                 agent=vf.AgentInfo(config=vf.AgentConfig(runtime=DockerConfig())))
                trace.record_error(error)
                self.assertNotIn(sentinel, trace.model_dump_json(exclude={"state"}))
                self.assertEqual(str(error), "reproduction_service_failed")
            else:
                self.fail("Provider failure was not propagated")


@unittest.skipUnless(HAS_PRIME and os.environ.get("RUN_RESEARCH_DOCKER_TESTS") == "1",
                     "Set RUN_RESEARCH_DOCKER_TESTS=1 for fresh-runtime checks")
class ReproductionDocker(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "public-reproduction-fixture-not-for-evaluation"}):
            cls.episode = build_episode(0)
        cls.policy = reference_submission(cls.episode)

    async def run_source(self, source, policy=None, timeout=30):
        from verifiers.v1.runtimes import DockerConfig
        return await verify_reproduction(self.episode["public_files"], source,
            self.policy if policy is None else policy,
            runtime_config=DockerConfig(image="python:3.12-slim"), timeout_seconds=timeout)

    async def test_independent_source_with_no_private_access(self):
        probe = b'''import os
from pathlib import Path
assert os.getuid()==65534
assert 'FINANCIAL_RESEARCH_KEY' not in os.environ
assert not Path('/workspace/policy.json').exists()
try:
    Path('/opt/financial-reproduction/controller.py').read_bytes()
except PermissionError:
    pass
else:
    raise AssertionError('controller visible')
'''
        result = await self.run_source(probe + (PACKAGE / "reference.py").read_bytes())
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["reproduction_verified"])

    async def test_missing_mismatched_and_failed_programs_are_candidate_failures(self):
        mismatch = copy.deepcopy(self.policy)
        mismatch["trials"][0]["mean_net"] += 0.01
        sources = [(b"pass\n", "output_missing_or_unsafe"),
                   (b"raise RuntimeError('candidate error')\n", "analysis_exit"),
                   (("from pathlib import Path\nPath('policy.json').write_text(" + repr(json.dumps(mismatch)) + ")\n").encode(), "policy_mismatch"),
                   (b"from pathlib import Path\nPath('policy.json').symlink_to('/etc/passwd')\n", "output_missing_or_unsafe"),
                   (b"import socket\ns=socket.socket(socket.AF_UNIX)\ns.bind('policy.json')\n", "output_not_regular"),
                   (b"from pathlib import Path\nPath('policy.json').write_text('x' * 2097153)\n", "output_too_large")]
        for source, reason in sources:
            with self.subTest(reason=reason):
                result = await self.run_source(source)
                self.assertEqual(result["status"], "candidate_failure")
                self.assertEqual(result["reason"], reason)

    async def test_timeout_and_detached_children_do_not_escape(self):
        result = await self.run_source(b"import time\ntime.sleep(10)\n", timeout=0.2)
        self.assertEqual(result["reason"], "analysis_timeout")
        source = b"import os,time\nif os.fork()==0:\n    os.setsid()\n    time.sleep(10)\n    raise SystemExit(0)\n"
        result = await self.run_source(source)
        self.assertEqual(result["reason"], "output_missing_or_unsafe")


if __name__ == "__main__":
    unittest.main()
