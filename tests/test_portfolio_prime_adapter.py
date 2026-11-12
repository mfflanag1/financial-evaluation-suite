"""Regression coverage for the portfolio task's narrow native adapter."""

import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/financial_portfolio/src"))
sys.path.insert(0, str(ROOT / "environments/financial_research/src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None

if vf is not None:
    from financial_research import prime_adapter as base
    from financial_portfolio import prime_adapter as adapter


def archive(path, payload):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        entry = tarfile.TarInfo(path.lstrip("/"))
        entry.size = len(payload)
        bundle.addfile(entry, io.BytesIO(payload))
    return buffer.getvalue()


class FrozenResearchPackage(unittest.TestCase):
    def test_v02_package_resources_are_byte_identical(self):
        root = ROOT / "environments/financial_research"
        digest, count = hashlib.sha256(), 0
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
                count += 1
        self.assertEqual(count, 12)
        self.assertEqual(digest.hexdigest(), "90331dfc4d114ecfa75e982723c7a98207018940c7b79e3d0cbb5d6444e2373d")


class RecordingRuntime:
    type = "docker"
    name = "portfolio-verifier"
    config = SimpleNamespace(mounts={})

    def __init__(self):
        self.files, self.commands = {}, []
        self.result = SimpleNamespace(exit_code=0, stdout='{"reward":0.8,"metrics":{"schema_valid":1.0}}')

    async def write(self, path, content):
        self.files[path] = content

    async def run(self, command, env):
        self.commands.append((command, env))
        return self.result


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class PortfolioAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = patch.dict(os.environ, {
            "FINANCIAL_RESEARCH_KEY": "portfolio-adapter-test-key-not-for-evaluation-0000"})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.reproduction = patch("financial_research.reproduction.verify_reproduction", new_callable=AsyncMock,
            return_value={"status": "passed", "reason": "policy_reproduced", "reproduction_verified": True})
        self.reproduce = self.reproduction.start()
        self.addCleanup(self.reproduction.stop)

    def task(self):
        return next(iter(adapter.FinancialPortfolioTaskset(adapter.PortfolioTasksetConfig(num_tasks=1))))

    def trace(self, task):
        from financial_portfolio.generator import resolve_episode
        from financial_research.verifier import reference_submission

        policy = json.dumps(reference_submission(resolve_episode(task.data.id))).encode()
        return vf.Trace(
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
            state=base.ResearchState(artifacts={
                "/workspace/policy.json": archive("/workspace/policy.json", policy),
                "/workspace/research.py": archive("/workspace/research.py", b"# Submitted analysis\n"),
            }),
        )

    def test_execution_and_accounting_methods_are_inherited_unchanged(self):
        for name in ("setup", "finalize", "grading_task", "_grade", "_grade_checked", "_reproduce",
                     "research_metrics", "research_reward"):
            self.assertIs(getattr(adapter.FinancialPortfolioTask, name), getattr(base.FinancialResearchTask, name))
        self.assertEqual(importlib.metadata.version("financial-research"), "0.2.0")
        self.assertEqual(importlib.metadata.version("verifiers"), "0.4.1")

    def test_native_loader_and_public_record_round_trip(self):
        from verifiers.v1.utils.loaders import taskset_class

        self.assertIs(taskset_class("financial-portfolio"), adapter.FinancialPortfolioTaskset)
        self.assertIs(adapter.FinancialPortfolioTask.data_type(), base.ResearchData)
        self.assertIs(adapter.FinancialPortfolioTask.config_type(), base.ResearchTaskConfig)
        task = self.task()
        restored = adapter.FinancialPortfolioTask(base.ResearchData.model_validate_json(task.data.model_dump_json()))
        self.assertEqual(restored.data, task.data)
        self.assertEqual(set(task.data.model_dump()) - set(vf.TaskData.model_fields), {"public_files"})
        self.assertEqual(len(json.loads(task.data.public_files["contract.json"])["candidates"]), 48)

    def test_resource_and_artifact_contract_matches_frozen_adapter(self):
        task = self.task()
        original = next(iter(base.FinancialResearchTaskset(base.ResearchTasksetConfig(num_tasks=1))))
        excluded = {"id", "prompt", "public_files"}
        self.assertEqual(task.data.model_dump(exclude=excluded), original.data.model_dump(exclude=excluded))

    def test_inherited_split_bounds_and_integer_validation(self):
        self.assertEqual(adapter.PortfolioTasksetConfig(split="dev").seed_start, 128)
        self.assertEqual(adapter.PortfolioTasksetConfig(split="test").seed_start, 160)
        self.assertEqual(adapter.PortfolioTasksetConfig(num_tasks="1").num_tasks, 1)
        for options in ({"num_tasks": True}, {"seed_start": 1.0}, {"split": "test", "num_tasks": 33}):
            with self.assertRaises(ValueError):
                adapter.PortfolioTasksetConfig(**options)
        for family in ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal"):
            self.assertEqual(adapter.PortfolioTasksetConfig(family=family).family, family)
        with self.assertRaises(ValueError):
            adapter.PortfolioTasksetConfig(family="signal")

    async def test_solver_receives_exactly_four_public_inputs(self):
        task, runtime = self.task(), RecordingRuntime()
        await task.setup(runtime)
        self.assertEqual(set(runtime.files), {f"/workspace/{name}" for name in base.PUBLIC_FILES})
        self.assertEqual(runtime.commands, [])

    async def test_private_staging_uses_frozen_scorer_without_generators(self):
        from financial_portfolio.generator import resolve_episode

        task, runtime = self.task(), RecordingRuntime()
        trace = self.trace(task)
        await task.stage_verifier(trace, runtime)
        prefix = trace.state.verifier_directory
        self.assertEqual(json.loads(runtime.files[f"{prefix}/episode.json"]), resolve_episode(task.data.id))
        self.assertEqual(runtime.files[f"{prefix}/score.py"], base._SCORE_PROGRAM.encode())
        for name in ("verifier.py", "analytics.py"):
            self.assertEqual(runtime.files[f"{prefix}/financial_research/{name}"],
                             Path(base.__file__).with_name(name).read_bytes())
        self.assertFalse(any("generator.py" in path or "/financial_portfolio/" in path for path in runtime.files))
        self.assertFalse(any(path.startswith("/workspace/") for path in runtime.files))

    async def test_modified_public_record_is_rejected(self):
        task, runtime = self.task(), RecordingRuntime()
        public = copy.deepcopy(task.data.public_files)
        public["prices.csv"] += "\nchanged"
        changed = task.with_data(public_files=public)
        with self.assertRaisesRegex(ValueError, "registered episode"):
            await changed.stage_verifier(self.trace(task), runtime)
        self.assertEqual(runtime.files, {})

    async def test_old_task_id_cannot_resolve_as_portfolio_episode(self):
        task, runtime = self.task(), RecordingRuntime()
        original = next(iter(base.FinancialResearchTaskset(base.ResearchTasksetConfig(num_tasks=1))))
        changed = task.with_data(id=original.data.id)
        with self.assertRaises(ValueError):
            await changed.stage_verifier(self.trace(task), runtime)
        self.assertEqual(runtime.files, {})

    async def test_reproduction_and_reward_use_inherited_pipeline(self):
        task, runtime = self.task(), RecordingRuntime()
        trace = self.trace(task)
        await task.stage_verifier(trace, runtime)
        await task.score(trace, runtime)
        self.assertEqual(trace.reward, 0.8)
        self.assertEqual(trace.metrics["reproduction_verified"], 1)
        self.assertEqual(self.reproduce.await_count, 1)
        self.assertEqual(self.reproduce.await_args.args[0], task.data.public_files)
        self.assertEqual(len(runtime.commands), 1)
        self.assertEqual(trace.info["financial_research_outcome"]["status"], "passed")

    async def test_candidate_failure_retains_zero_reward_accounting(self):
        task, runtime = self.task(), RecordingRuntime()
        trace = self.trace(task)
        await task.stage_verifier(trace, runtime)
        self.reproduce.return_value = {"status": "candidate_failure", "reason": "policy_mismatch",
                                       "reproduction_verified": False}
        await task.score(trace, runtime)
        self.assertEqual(trace.reward, 0)
        self.assertEqual(trace.metrics["candidate_failure"], 1)
        self.assertEqual(runtime.commands, [])

    async def test_private_upload_error_is_sanitized_in_native_trace(self):
        task, runtime = self.task(), RecordingRuntime()
        trace = self.trace(task)
        runtime.write = AsyncMock(side_effect=RuntimeError("portfolio-provider-secret-sentinel"))
        try:
            await task.stage_verifier(trace, runtime)
        except RuntimeError as error:
            trace.record_error(error)
        else:
            self.fail("Upload failure did not propagate")
        self.assertNotIn("portfolio-provider-secret-sentinel", trace.model_dump_json())
        self.assertEqual(trace.info["financial_research_outcome"]["status"], "infrastructure_error")


if __name__ == "__main__":
    unittest.main()
