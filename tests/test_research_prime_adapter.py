"""Verifiers 0.4.1 contract tests; no model requests or candidate-code execution."""

import asyncio
import copy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "environments/financial_research/src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None

if vf is not None:
    from financial_research import prime_adapter as adapter


def artifact(path, payload, *, kind=tarfile.REGTYPE, duplicate=False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        entry = tarfile.TarInfo(path.lstrip("/"))
        entry.size, entry.type = len(payload), kind
        bundle.addfile(entry, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
        if duplicate:
            bundle.addfile(entry, io.BytesIO(payload))
    return buffer.getvalue()


class RecordingRuntime:
    """In-memory boundary recorder, not a replacement execution backend."""

    type = "docker"
    name = "verifier-box"
    config = SimpleNamespace(mounts={})

    def __init__(self):
        self.files = {}
        self.commands = []
        self.result = SimpleNamespace(
            exit_code=0, stdout='{"reward":0.75,"metrics":{"contract_valid":1.0}}', stderr="")

    async def write(self, path, data):
        self.files[path] = data

    async def run(self, argv, env):
        self.commands.append((argv, env))
        return self.result


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class PrimeAdapterContract(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key_patch = patch.dict(os.environ, {
            "FINANCIAL_RESEARCH_KEY": "adapter-test-only-key-not-for-evaluation-0000"})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.reproduction_patch = patch("financial_research.reproduction.verify_reproduction",
            new_callable=AsyncMock,
            return_value={"status": "passed", "reason": "policy_reproduced", "reproduction_verified": True})
        self.reproduce = self.reproduction_patch.start()
        self.addCleanup(self.reproduction_patch.stop)

    def task(self):
        from financial_research.generator import build_episode

        episode = build_episode(0, split="train")
        data = adapter.ResearchData(
            id=episode["id"], prompt=episode["prompt"], public_files=episode["public_files"])
        return adapter.FinancialResearchTask(data), episode

    def trace(self, task):
        from financial_research.generator import resolve_episode
        from financial_research.verifier import reference_submission

        policy = json.dumps(reference_submission(resolve_episode(task.data.id))).encode()
        return vf.Trace(
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
            state=adapter.ResearchState(artifacts={
                "/workspace/policy.json": artifact("/workspace/policy.json", policy),
                "/workspace/research.py": artifact("/workspace/research.py", b"# Analysis\n"),
            }),
        )

    def test_real_pinned_types_and_taskset_discovery(self):
        from verifiers.v1.utils.loaders import taskset_class

        self.assertEqual(importlib.metadata.version("verifiers"), "0.4.1")
        self.assertIs(adapter.FinancialResearchTask.data_type(), adapter.ResearchData)
        self.assertIs(adapter.FinancialResearchTask.config_type(), adapter.ResearchTaskConfig)
        self.assertIs(adapter.FinancialResearchTaskset.task_type(), adapter.FinancialResearchTask)
        self.assertIs(taskset_class("financial-research"), adapter.FinancialResearchTaskset)
        tasks = list(adapter.FinancialResearchTaskset(adapter.ResearchTasksetConfig(num_tasks=2)))
        self.assertEqual([t.data.idx for t in tasks], [0, 1])
        self.assertNotEqual(tasks[0].data.id, tasks[1].data.id)
        self.assertEqual(tasks[0].data.scoring.mode, "isolated")
        self.assertTrue(tasks[0].data.network.network_restricted)
        self.assertTrue(all(not a.required for a in tasks[0].data.artifacts))

    def test_wire_data_has_no_private_episode_fields(self):
        task, episode = self.task()
        wire = task.data.model_dump(mode="json")
        self.assertEqual(set(wire) - set(vf.TaskData.model_fields), {"public_files"})
        for key in ("private", "seed", "family", "answer", "continuations"):
            self.assertNotIn(key, wire)
        # Reconstruct without an in-process registry or Task instance attributes.
        reconstructed = adapter.FinancialResearchTask(
            adapter.ResearchData.model_validate_json(task.data.model_dump_json()))
        self.assertEqual(reconstructed.data.id, episode["id"])

    async def test_setup_stages_only_public_files(self):
        task, episode = self.task()
        runtime = RecordingRuntime()
        await task.setup(runtime)
        self.assertEqual(set(runtime.files), {f"/workspace/{name}" for name in adapter.PUBLIC_FILES})
        self.assertEqual(runtime.commands, [])
        for name, content in episode["public_files"].items():
            self.assertEqual(runtime.files[f"/workspace/{name}"], content.encode())

    async def test_setup_rejects_extra_paths(self):
        task, _ = self.task()
        task = task.with_data(public_files={**task.data.public_files, "../secret": "x"})
        with self.assertRaisesRegex(ValueError, "file set"):
            await task.setup(RecordingRuntime())

    async def test_runtime_and_placement_guards(self):
        task, _ = self.task()
        runtime = RecordingRuntime()
        with self.assertRaisesRegex(ValueError, "isolated"):
            task.grading_task(runtime, scoring=vf.ScoringConfig(mode="shared"))
        runtime.type = "subprocess"
        with self.assertRaisesRegex(ValueError, "container"):
            await task.setup(runtime)
        runtime.type = "docker"
        runtime.config = SimpleNamespace(mounts={"/workspace": "host"})
        with self.assertRaisesRegex(ValueError, "host data"):
            await task.setup(runtime)
        runtime.config = SimpleNamespace(mounts={})
        runtime.env = {"FINANCIAL_RESEARCH_KEY": "not-allowed-in-box"}
        with self.assertRaisesRegex(ValueError, "trusted worker"):
            await task.setup(runtime)

    async def test_finalize_does_not_claim_reproduction_or_execute_analysis(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.finalize(trace, runtime)
        self.assertFalse(trace.info["research_artifact_contract"]["analysis_execution_verified"])
        self.assertEqual(runtime.commands, [])

    async def test_private_staging_is_separate_and_reconstructible(self):
        task, episode = self.task()
        task = adapter.FinancialResearchTask(adapter.ResearchData.model_validate_json(
            task.data.model_dump_json()))
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        root = trace.state.verifier_directory
        self.assertTrue(root.startswith("/tmp/financial-verifier-"))
        self.assertEqual(json.loads(runtime.files[f"{root}/episode.json"]), episode)
        self.assertFalse(any(path.startswith("/workspace/") for path in runtime.files))
        self.assertNotIn('"private":', trace.model_dump_json())
        self.assertEqual(runtime.commands, [])

    async def test_changed_public_record_is_rejected_before_private_staging(self):
        task, _ = self.task()
        public = copy.deepcopy(task.data.public_files)
        public["prices.csv"] += "\nmodified"
        task = task.with_data(public_files=public)
        runtime = RecordingRuntime()
        with self.assertRaisesRegex(ValueError, "registered episode"):
            await task.stage_verifier(self.trace(task), runtime)
        self.assertEqual(runtime.files, {})

    async def test_native_score_records_metrics_and_reward_once(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        await task.score(trace, runtime)
        self.assertEqual(trace.reward, 0.75)
        self.assertEqual(trace.metrics["contract_valid"], 1.0)
        self.assertEqual(len(runtime.commands), 1)
        self.assertEqual(runtime.commands[0][0][:2], ["python", "-I"])
        self.assertNotIn("research.py", " ".join(runtime.commands[0][0]))
        self.assertEqual(self.reproduce.await_count, 1)
        self.assertEqual(trace.info["financial_research_outcome"]["status"], "passed")
        self.assertEqual(trace.metrics["reproduction_verified"], 1)
        args = self.reproduce.await_args
        self.assertEqual(args.args[0], task.data.public_files)
        self.assertEqual(args.args[1], b"# Analysis\n")
        self.assertNotIn("private", args.kwargs)

    async def test_infrastructure_failure_propagates_without_zero_reward(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        runtime.result = SimpleNamespace(exit_code=1, stdout="", stderr="grader unavailable")
        with self.assertRaisesRegex(RuntimeError, "trusted_grader_failed"):
            await task.research_reward(trace, runtime)
        self.assertIsNone(trace.state.grade)

    async def test_scoring_requires_staged_matching_runtime(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        with self.assertRaisesRegex(RuntimeError, "scoring_error"):
            await task.research_reward(trace, runtime)
        await task.stage_verifier(trace, runtime)
        runtime.name = "solver-box"
        with self.assertRaisesRegex(RuntimeError, "scoring_error"):
            await task.research_reward(trace, runtime)

    async def test_nonfinite_or_out_of_range_grader_output_is_infrastructure_error(self):
        task, _ = self.task()
        for payload in ('{"reward":NaN,"metrics":{}}', '{"reward":2,"metrics":{}}'):
            runtime, trace = RecordingRuntime(), self.trace(task)
            await task.stage_verifier(trace, runtime)
            runtime.result.stdout = payload
            with self.assertRaises(RuntimeError):
                await task.research_reward(trace, runtime)
            self.assertIsNone(trace.state.grade)

    async def test_candidate_reproduction_failure_is_scored_zero_without_private_execution(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        self.reproduce.return_value = {"status": "candidate_failure", "reason": "policy_mismatch",
                                       "reproduction_verified": False}
        await task.score(trace, runtime)
        self.assertEqual(trace.reward, 0)
        self.assertEqual(trace.metrics["candidate_failure"], 1)
        self.assertEqual(runtime.commands, [])
        outcome = trace.info["financial_research_outcome"]
        self.assertEqual(outcome["reason"], "policy_mismatch")
        self.assertEqual(trace.info["scoring_diagnostics"]["financial_research_outcome"], outcome)

    async def test_reproduction_infrastructure_failure_is_preserved_and_propagates(self):
        from financial_research.reproduction import ReproductionInfrastructureError

        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        self.reproduce.side_effect = ReproductionInfrastructureError("runtime_unavailable")
        with self.assertRaises(ReproductionInfrastructureError):
            await task.research_reward(trace, runtime)
        self.assertIsNone(trace.state.grade)
        self.assertEqual(runtime.commands, [])
        self.assertEqual(trace.info["scoring_diagnostics"]["financial_research_outcome"]["status"],
                         "infrastructure_error")

    async def test_provider_exception_text_is_absent_from_native_trace(self):
        task, _ = self.task()
        runtime, trace = RecordingRuntime(), self.trace(task)
        await task.stage_verifier(trace, runtime)
        runtime.run = AsyncMock(side_effect=RuntimeError("sentinel-provider-secret"))
        try:
            await task.research_reward(trace, runtime)
        except RuntimeError as error:
            trace.record_error(error)
        else:
            self.fail("Provider failure did not propagate")
        self.assertNotIn("sentinel-provider-secret", trace.model_dump_json())
        self.assertEqual(trace.info["financial_research_outcome"]["status"], "infrastructure_error")

    async def test_setup_and_private_upload_exceptions_are_sanitized(self):
        task, _ = self.task()
        for phase in ("setup", "staging"):
            runtime, trace = RecordingRuntime(), self.trace(task)
            runtime.write = AsyncMock(side_effect=RuntimeError("sentinel-upload-secret"))
            try:
                if phase == "setup":
                    await task.setup(runtime, trace)
                else:
                    await task.stage_verifier(trace, runtime)
            except RuntimeError as error:
                trace.record_error(error)
            else:
                self.fail("Upload failure did not propagate")
            self.assertNotIn("sentinel-upload-secret", trace.model_dump_json())
            self.assertEqual(trace.info["financial_research_outcome"]["phase"], phase)

    async def test_bad_submission_archives_are_candidate_failures(self):
        task, _ = self.task()
        bad_archives = (None, b"not a tar", artifact("/elsewhere", b"x"),
                        artifact("/workspace/research.py", b"x", kind=tarfile.SYMTYPE),
                        artifact("/workspace/research.py", b"x", duplicate=True),
                        artifact("/workspace/research.py", b"x" * (adapter.MAX_ANALYSIS_BYTES + 1)))
        for archive in bad_archives:
            runtime, trace = RecordingRuntime(), self.trace(task)
            await task.stage_verifier(trace, runtime)
            trace.state.artifacts["/workspace/research.py"] = archive
            grade = await task._grade(trace, runtime)
            self.assertEqual(grade["reward"], 0)
            self.assertEqual(trace.info["financial_research_outcome"]["phase"], "artifacts")
            self.assertEqual(runtime.commands, [])
        self.reproduce.assert_not_awaited()

    async def test_duplicate_or_non_object_policy_is_rejected_before_reproduction(self):
        task, _ = self.task()
        for payload in (b'{"x":1,"x":2}', b"NaN", b"[]", b"{"):
            runtime, trace = RecordingRuntime(), self.trace(task)
            await task.stage_verifier(trace, runtime)
            trace.state.artifacts["/workspace/policy.json"] = artifact("/workspace/policy.json", payload)
            self.assertEqual((await task._grade(trace, runtime))["reward"], 0)
        self.reproduce.assert_not_awaited()

    def test_split_ranges_are_disjoint_and_bounded(self):
        self.assertEqual(adapter.ResearchTasksetConfig(num_tasks="1", seed_start="0").num_tasks, 1)
        self.assertEqual(adapter.ResearchTasksetConfig(split="train").seed_start, 0)
        self.assertEqual(adapter.ResearchTasksetConfig(split="dev").seed_start, 128)
        self.assertEqual(adapter.ResearchTasksetConfig(split="test").seed_start, 160)
        for options in ({"split": "train", "seed_start": 128},
                        {"split": "dev", "num_tasks": 33},
                        {"split": "test", "seed_start": 159},
                        {"num_tasks": True}, {"num_tasks": 1.0},
                        {"num_tasks": "1.0"},
                        {"seed_start": False}, {"seed_start": 0.0},
                        {"family": "unknown"}):
            with self.assertRaises(ValueError):
                adapter.ResearchTasksetConfig(**options)


@unittest.skipUnless(vf is not None and os.environ.get("RUN_RESEARCH_DOCKER_TESTS") == "1",
                     "Set RUN_RESEARCH_DOCKER_TESTS=1 for isolated artifact probes")
class PrimeArtifactBoundary(unittest.IsolatedAsyncioTestCase):
    @patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "adapter-test-only-key-not-for-evaluation-0000"})
    async def test_actual_runtime_reproduces_analysis_only_in_separate_public_box(self):
        from financial_research.generator import resolve_episode
        from financial_research.verifier import reference_submission, score_submission
        from verifiers.v1.runtimes import make_backend

        task = next(iter(adapter.FinancialResearchTaskset(
            adapter.ResearchTasksetConfig(num_tasks=1))))
        episode = resolve_episode(task.data.id)
        reference = reference_submission(episode)
        expected = score_submission(episode, reference)["reward"]
        trace = vf.Trace(
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
            state=adapter.ResearchState(),
        )
        backend = make_backend(vf.DockerConfig(image="python:3.12-slim",
                                               workdir="/workspace", allow=[]))
        try:
            async with asyncio.timeout(180):
                await backend.start()
                runtime = backend.bind()
                await backend.prepare_setup()
                await task.setup(runtime)
                probe = await runtime.run(["python", "-I", "-c",
                    "import os; assert 'FINANCIAL_RESEARCH_KEY' not in os.environ"], {})
                self.assertEqual(probe.exit_code, 0)
                await task.stage_verifier(trace, runtime)
                await backend.prepare_execution([])
                await runtime.write("/workspace/policy.json", json.dumps(reference).encode())

                async def grade_submission():
                    trace.state.artifacts = await vf.collect(
                        runtime, task.data.artifacts, max_bytes=task.data.artifact_max_bytes)
                    trace.state.grade = trace.state.reproduction = None
                    trace.state.analysis_artifact_valid = False
                    return await task._grade(trace, runtime)

                for payload in (None, b"", b"  \n", b"\xff", b"x" * (1048576 + 1)):
                    if payload is None:
                        await runtime.run(["rm", "-f", "/workspace/research.py"], {})
                    else:
                        await runtime.write("/workspace/research.py", payload)
                    grade = await grade_submission()
                    self.assertEqual(grade["reward"], 0)
                    self.assertEqual(grade["metrics"]["analysis_artifact_valid"], 0)
                # Candidate code runs in the public replay box and fails normally.
                await runtime.write("/workspace/research.py", b"raise RuntimeError('candidate failure')\n")
                grade = await grade_submission()
                self.assertEqual(grade["reward"], 0)
                self.assertEqual(trace.info["financial_research_outcome"]["status"], "candidate_failure")
                # The standalone reference sees only the four public inputs.
                source = Path(adapter.__file__).with_name("reference.py").read_bytes()
                await runtime.write("/workspace/research.py", source)
                grade = await grade_submission()
                self.assertAlmostEqual(grade["reward"], expected)
                self.assertEqual(grade["metrics"]["analysis_artifact_valid"], 1)
                self.assertEqual(grade["metrics"]["reproduction_verified"], 1)
                for payload in (b'{"audit":[],"audit":[]}', b"NaN", b"[]",
                                b" " * (adapter.MAX_POLICY_BYTES + 1)):
                    await runtime.write("/workspace/policy.json", payload)
                    self.assertEqual((await grade_submission())["reward"], 0)
                await runtime.run(["rm", "-f", "/workspace/policy.json"], {})
                await runtime.run(["ln", "-s", f"{trace.state.verifier_directory}/episode.json",
                                   "/workspace/policy.json"], {})
                self.assertEqual((await grade_submission())["reward"], 0)
                self.assertNotIn(os.environ["FINANCIAL_RESEARCH_KEY"], trace.model_dump_json())
        finally:
            await backend.stop()


if __name__ == "__main__":
    unittest.main()
