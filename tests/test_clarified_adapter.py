"""Verify that the clarification changes only one public prompt suffix."""

import copy
import hashlib
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
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
                "financial_portfolio_clarified"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None
if vf is not None:
    from financial_research import prime_adapter as base
    from financial_portfolio import prime_adapter as frozen
    from financial_portfolio_staged import prime_adapter as staged
    from financial_portfolio_staged.validation import inventory_for_public_files
    from financial_portfolio_clarified import prime_adapter as adapter


def archive(path, payload):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        member = tarfile.TarInfo(path.lstrip("/"))
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


class RecordingRuntime:
    type = "docker"
    config = SimpleNamespace(mounts={})

    def __init__(self, name="clarified-fixture-runtime"):
        self.name, self.files = name, {}

    async def write(self, path, content):
        self.files[path] = content

    async def run(self, *args, **kwargs):
        raise AssertionError("Candidate commands must never execute in these tests")


@unittest.skipUnless(vf is not None, "Install pinned native dependencies")
class ClarifiedAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "FINANCIAL_RESEARCH_KEY": "clarified-adapter-test-key-not-for-evaluation-0000"})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = frozen.PortfolioTasksetConfig(seed_start=4, num_tasks=4)
        self.task = next(iter(adapter.FinancialPortfolioClarifiedTaskset(self.config)))
        self.trace = vf.Trace(
            task=vf.TraceTask(type=type(self.task).__name__, data=self.task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
            state=base.ResearchState())
        self.runtime = RecordingRuntime()
        self.addCleanup(staged._SOLVERS.clear)

    def test_native_export_has_exactly_one_taskset_and_no_harness(self):
        import financial_portfolio_clarified as package
        from verifiers.v1.utils.loaders import taskset_class

        self.assertEqual(package.__all__, ["FinancialPortfolioClarifiedTaskset"])
        self.assertIs(taskset_class("financial-portfolio-clarified"),
                      adapter.FinancialPortfolioClarifiedTaskset)
        with self.assertRaises(AttributeError):
            getattr(package, "FinancialPortfolioStagedHarness")

    def test_all_four_tasks_differ_only_by_exact_system_prompt_suffix(self):
        baseline = list(staged.FinancialPortfolioStagedTaskset(self.config))
        clarified = list(adapter.FinancialPortfolioClarifiedTaskset(self.config))
        expected = ("shares_per_original is the cumulative number of current shares corresponding "
                    "to one original share; close is price per current share. Define adjusted_close "
                    "= close * shares_per_original.")
        self.assertEqual(adapter.CLARIFICATION, expected)
        self.assertEqual(len(clarified), 4)
        for original, updated in zip(baseline, clarified, strict=True):
            with self.subTest(task=original.data.id):
                self.assertEqual(updated.data.system_prompt, original.data.system_prompt + "\n\n" + expected)
                self.assertEqual(updated.data.model_dump(exclude={"system_prompt"}),
                                 original.data.model_dump(exclude={"system_prompt"}))
                self.assertEqual(updated.config, original.config)

    def test_all_workflow_and_financial_hooks_remain_identical_without_duplicates(self):
        inherited = ("finalize", "checkpoint", "candidate_protocol_failure", "candidate_workflow_failure",
                     "workflow_complete", "_workflow_failure_grade", "_grade", "_reproduce", "validate",
                     "grading_task", "research_metrics", "research_reward")
        for method in inherited:
            with self.subTest(method=method):
                self.assertIs(getattr(adapter.FinancialPortfolioClarifiedTask, method),
                              getattr(staged.FinancialPortfolioStagedTask, method))
        self.assertEqual([hook.__name__ for hook in self.task.hooks("reward")], ["research_reward"])
        self.assertEqual([hook.__name__ for hook in self.task.hooks("metric")], ["research_metrics"])
        self.assertCountEqual([hook.__name__ for hook in self.task.hooks("stop")],
                              ["candidate_protocol_failure", "candidate_workflow_failure", "workflow_complete"])

    async def test_setup_keeps_public_files_registry_and_separate_condition_metadata(self):
        await self.task.setup(self.runtime, self.trace)
        self.assertEqual(self.runtime.files,
                         {f"/workspace/{name}": value.encode() for name, value in self.task.data.public_files.items()})
        self.assertIs(staged._SOLVERS[self.trace.id], self.runtime)
        self.assertEqual(self.trace.info[staged.WORKFLOW_KEY]["protocol_id"], staged.PROTOCOL_ID)
        self.assertEqual(self.trace.info[adapter.CONDITION_KEY], {
            "protocol_id": "portfolio-clarified-v1", "baseline_protocol_id": "portfolio-staged-v1",
            "environment": "financial-portfolio-clarified", "environment_version": "0.5.0",
            "clarification_sha256": hashlib.sha256(adapter.CLARIFICATION.encode()).hexdigest()})
        self.assertEqual(copy.deepcopy(self.task).data, self.task.data)
        self.assertNotIn(self.runtime.name, self.trace.model_dump_json())
        with patch.object(staged.vf, "collect", AsyncMock(return_value={path: None for path in staged.PATHS})):
            await self.task.finalize(self.trace, self.runtime)
        self.assertNotIn(self.trace.id, staged._SOLVERS)

    async def test_tampered_prompt_rejected_without_writes_and_releases_solver_reference(self):
        for prompt in (staged.workflow_prompt(self.task.data.public_files),
                       self.task.data.system_prompt + "\n", ""):
            with self.subTest(prompt_length=len(prompt)):
                original_data = self.task.data
                self.task.data = original_data.model_copy(update={"system_prompt": prompt})
                with self.assertRaisesRegex(ValueError, "registered protocol"):
                    await self.task.setup(self.runtime, self.trace)
                staged._SOLVERS[self.trace.id] = self.runtime
                with self.assertRaisesRegex(ValueError, "registered protocol"):
                    await self.task.stage_verifier(self.trace, self.runtime)
                self.assertNotIn(self.trace.id, staged._SOLVERS)
                self.assertEqual(self.runtime.files, {})
                self.assertEqual(self.task.data.system_prompt, prompt)
                self.task.data = original_data

    async def test_frozen_private_staging_is_identical_and_rechecks_original_episode(self):
        baseline = next(iter(staged.FinancialPortfolioStagedTaskset(self.config)))
        baseline_runtime = RecordingRuntime("baseline-verifier")
        baseline_trace = self.trace.model_copy(deep=True)
        await baseline.stage_verifier(baseline_trace, baseline_runtime)
        await self.task.setup(self.runtime, self.trace)
        self.runtime.files.clear()
        self.trace.state.grade = {"reward": 1.0}
        await self.task.stage_verifier(self.trace, self.runtime)
        self.assertNotIn(self.trace.id, staged._SOLVERS)
        self.assertIsNone(self.trace.state.grade)
        self.assertEqual({path.removeprefix(self.trace.state.verifier_directory): content
                          for path, content in self.runtime.files.items()},
                         {path.removeprefix(baseline_trace.state.verifier_directory): content
                          for path, content in baseline_runtime.files.items()})
        self.assertFalse(any("generator.py" in path for path in self.runtime.files))
        self.runtime.files.clear()
        original = self.task.data
        for change in ({"prompt": original.prompt + "changed"}, {"id": "unknown-episode"},
                       {"public_files": {**original.public_files, "prices.csv": "changed"}}):
            with self.subTest(field=next(iter(change))):
                altered = original.model_copy(update=change)
                self.task.data = altered.model_copy(update={
                    "system_prompt": adapter.clarified_prompt(altered.public_files)})
                with self.assertRaises((ValueError, KeyError)):
                    await self.task.stage_verifier(self.trace, self.runtime)
                self.assertEqual(self.runtime.files, {})
        self.task.data = original

    async def test_malformed_tool_call_remains_zero_candidate_failure(self):
        await self.task.setup(self.runtime, self.trace)
        response = vf.Response(id="fixture", created=0, model="fixture", finish_reason="tool_calls",
            message=vf.AssistantMessage(tool_calls=[vf.ToolCall(
                id="invalid", name="bash", arguments='{"command":"invalid\\ escape"}')]))
        self.assertTrue(await self.task.candidate_protocol_failure(response, self.trace))
        with patch.object(base.FinancialResearchTask, "_reproduce",
                          AsyncMock(spec=self.task._reproduce)) as reproduce:
            await self.task.score(self.trace, self.runtime)
        reproduce.assert_not_awaited()
        self.assertEqual(self.trace.reward, 0)
        self.assertEqual(self.trace.metrics["candidate_failure"], 1)
        self.assertEqual(self.trace.info["financial_research_outcome"]["reason"], "malformed_tool_arguments")

    async def test_metric_and_reward_call_frozen_grader_and_reproduction_once(self):
        from financial_portfolio.generator import resolve_episode
        from financial_research.verifier import reference_submission

        policy = reference_submission(resolve_episode(self.task.data.id))
        payloads = {
            staged.PATHS[0]: b"raise RuntimeError('must not execute in host tests')\n",
            staged.PATHS[1]: json.dumps(inventory_for_public_files(self.task.data.public_files)).encode(),
            staged.PATHS[2]: json.dumps(policy["trials"]).encode(),
            staged.PATHS[3]: json.dumps(policy).encode(),
        }
        await self.task.setup(self.runtime, self.trace)
        workflow = self.trace.info[staged.WORKFLOW_KEY]
        workflow["completed"] = list(staged.STAGES)
        workflow["submission_sha256"] = staged._hashes(payloads)
        self.trace.state.artifacts = {path: archive(path, content) for path, content in payloads.items()}
        await self.task.stage_verifier(self.trace, self.runtime)
        trusted_grade = {"reward": 0.75, "metrics": {"schema_valid": 1.0}}
        result = SimpleNamespace(exit_code=0, stdout=json.dumps(trusted_grade))
        with (patch.object(base.FinancialResearchTask, "_reproduce", AsyncMock(
                spec=self.task._reproduce, return_value={
                "status": "passed", "reason": "reproduced", "reproduction_verified": True})) as reproduce,
                patch.object(self.runtime, "run", AsyncMock(return_value=result)) as score):
            await self.task.score(self.trace, self.runtime)
        reproduce.assert_awaited_once()
        self.assertIs(reproduce.await_args.args[0], self.trace)
        self.assertIs(reproduce.await_args.args[1], self.runtime)
        score.assert_awaited_once_with(["python", "-I", f"{self.trace.state.verifier_directory}/score.py"], {})
        self.assertEqual(self.trace.reward, 0.75)
        self.assertEqual(self.trace.metrics["reproduction_verified"], 1)
        self.assertEqual(self.trace.info[staged.WORKFLOW_KEY]["protocol_id"], "portfolio-staged-v1")
        self.assertEqual(self.trace.info[adapter.CONDITION_KEY]["protocol_id"], "portfolio-clarified-v1")


if __name__ == "__main__":
    unittest.main()
