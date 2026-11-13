"""Native wire, isolation and aggregate-admission controls without model calls."""
import asyncio
import base64
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
import verifiers.v1 as vf
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.runtime import DockerConfig
from verifiers.v1.trace import AgentInfo, TraceTask
from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

from financial_selection import registry, replay
from financial_selection.contract import PUBLIC_FILES, input_digest
from financial_selection.generator import generate
from financial_selection.prime_adapter import (
    INFO_KEY, PROMPT, FinancialSelectionTask, FinancialSelectionTaskset,
    SelectionData, SelectionState, SelectionTaskConfig, SelectionTasksetConfig,
    _CAPTURE, evaluation_admission,
)


KEY = "a9" * 32  # Synthetic test key only.
IMAGE = "sha256:" + "a" * 64


def task():
    return next(iter(FinancialSelectionTaskset(SelectionTasksetConfig(
        num_tasks=1, task=SelectionTaskConfig(image=IMAGE)))))


def runtime(name="verifier", **updates):
    config = DockerConfig(image=IMAGE, allow=[], workdir="/workspace")
    return SimpleNamespace(type="docker", config=config, env={}, name=name,
                           write=AsyncMock(), run=AsyncMock(return_value=SimpleNamespace(
                               exit_code=0, stdout='{"research.py":null,"policy.json":null}')), **updates)


def archive(path, payload=b"{}", kind=tarfile.REGTYPE):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo(path.lstrip("/"))
        info.type, info.size = kind, len(payload)
        if kind == tarfile.SYMTYPE:
            info.linkname = "/opt/private"
        tar.addfile(info, io.BytesIO(payload))
    return out.getvalue()


def trace(current):
    return vf.Trace[SelectionData, SelectionState](
        task=TraceTask(type=type(current).__name__, data=current.data),
        agent=AgentInfo(config=AgentConfig(runtime=DockerConfig(image=IMAGE, allow=[]))),
        state=SelectionState(solver_runtime="solver", verifier_runtime="verifier", artifacts={
            "/workspace/research.py": archive("/workspace/research.py", b"print('fixture')\n"),
            "/workspace/policy.json": archive("/workspace/policy.json"),
        }),
    )


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {registry.KEY_ENV: KEY})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_identity_matches_generator_and_is_bounded(self):
        for split in registry.SPLITS:
            for condition in registry.CONDITIONS:
                identity, public = registry.public_episode(split, 63, condition)
                expected = generate(bytes.fromhex(KEY), split, 63, condition=condition)
                self.assertEqual(identity, expected.identity)
                self.assertEqual(public, expected.public_files)
        for index in (-1, 64, True, 1.0):
            with self.assertRaises(registry.RegistryError):
                registry.public_episode("dev", index)

    def test_three_packets_are_distinct_and_fresh_ids_cannot_schedule(self):
        identity, current = registry.public_episode("dev", 3, "positive")
        resolved = registry.resolve(identity)
        self.assertEqual(resolved.public_files, current)
        self.assertEqual(len({input_digest(p) for p in [current, *resolved.fresh_packets]}), 3)
        for offset, fresh in enumerate(resolved.fresh_packets):
            expected = generate(bytes.fromhex(KEY), "dev", 134 + offset, condition="positive")
            self.assertEqual(fresh, expected.public_files)
            with self.assertRaises(registry.RegistryError):
                registry.resolve(expected.identity)

    def test_key_validation_and_unknown_identity(self):
        for key in ("", "aa" * 31, "g" * 64, "a" * 65):
            with patch.dict(os.environ, {registry.KEY_ENV: key}):
                with self.assertRaisesRegex(registry.RegistryError, "worker_key_unavailable"):
                    registry.worker_key()
        for value in (None, "0" * 64, "not-an-id", "A" * 64):
            with self.assertRaisesRegex(registry.RegistryError, "unknown_task_identity"):
                registry.resolve(value)

    def test_spawned_interpreter_resolves_without_registration(self):
        identity, public = registry.public_episode("final", 11, "sparse")
        source_root = str(Path(registry.__file__).resolve().parents[1])
        code = ("import sys; sys.path.insert(0,sys.argv[1]); "
                "from financial_selection.registry import resolve; "
                "from financial_selection.contract import input_digest; "
                "print(input_digest(resolve(sys.argv[2]).public_files))")
        result = subprocess.run([sys.executable, "-I", "-c", code, source_root, identity],
                                env={**os.environ, registry.KEY_ENV: KEY},
                                capture_output=True, text=True, timeout=20, check=True,
                                cwd=Path(sys.executable).parent)
        self.assertEqual(result.stdout.strip(), input_digest(public))
        self.assertNotIn(KEY, result.stdout + result.stderr)

    def test_native_loader_and_wire_roundtrip_only_public_payload(self):
        plugin = "financial-selection"
        config = taskset_config_type(plugin)(id=plugin, num_tasks=6, task={"image": IMAGE})
        tasks = list(load_taskset(config))
        self.assertEqual([t.data.id for t in tasks], sorted(t.data.id for t in tasks))
        self.assertEqual([t.data.idx for t in tasks], list(range(6)))
        for current in tasks:
            self.assertEqual(set(current.data.public_files), set(PUBLIC_FILES))
            wire = current.data.model_dump_json()
            restored = SelectionData.model_validate_json(wire)
            self.assertEqual(restored, current.data)
            self.assertFalse(set(restored.model_dump()) & {"key", "index", "condition", "split", "fresh_packets", "private_population"})
            self.assertNotIn(KEY, wire)
            self.assertEqual(current.runtime_env(), {})

    def test_strict_config_bounds_and_immutable_images(self):
        self.assertEqual(SelectionTasksetConfig(num_tasks="2", index_start="3").num_tasks, 2)
        for values in ({"num_tasks": True}, {"num_tasks": 1.0}, {"num_tasks": 0},
                       {"index_start": 63, "num_tasks": 2}, {"condition": "other"}):
            with self.assertRaises(ValidationError):
                SelectionTasksetConfig(**values)
        for image in ("python:3.12-slim", "latest", "sha256:bad"):
            with self.assertRaises(ValidationError):
                SelectionTaskConfig(image=image)
        with self.assertRaises(ValueError):
            list(FinancialSelectionTaskset(SelectionTasksetConfig()))


class NativeHookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {registry.KEY_ENV: KEY})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.task = task()
        self.trace = trace(self.task)
        self.runtime = runtime()

    async def test_setup_writes_exactly_five_public_files(self):
        await self.task.setup(self.runtime, self.trace)
        writes = self.runtime.write.await_args_list
        self.assertEqual({a.args[0] for a in writes}, {"/workspace/" + f for f in PUBLIC_FILES})
        self.assertEqual(len(writes), 5)
        for args in writes:
            self.assertNotIn(KEY.encode(), args.args[1])

    async def test_altered_public_packet_or_prompt_fails_before_write(self):
        for update in ({"prompt": PROMPT + " altered"}, {"id": "0" * 64}):
            current = self.task.with_data(**update)
            with self.assertRaisesRegex(RuntimeError, "public_setup_failed"):
                await current.setup(self.runtime, self.trace)
        self.runtime.write.assert_not_awaited()
        self.assertEqual(self.trace.info[INFO_KEY]["status"], "infrastructure_error")

    async def test_network_mount_env_provider_and_image_overrides_rejected(self):
        configs = [DockerConfig(image=IMAGE), DockerConfig(image="python:3.12-slim", allow=[]),
                   DockerConfig(image=IMAGE, allow=[], mounts={"/data": {"source": "/tmp"}})]
        for config in configs:
            unsafe = runtime()
            unsafe.config = config
            with self.assertRaisesRegex(RuntimeError, "unsafe_selection_runtime"):
                self.task._require_runtime(unsafe)
        for values in ({"type": "subprocess"}, {"env": {registry.KEY_ENV: KEY}}, {"env": {"OTHER": "value"}}):
            unsafe = runtime()
            for name, value in values.items():
                setattr(unsafe, name, value)
            with self.assertRaises(RuntimeError):
                self.task._require_runtime(unsafe)

    async def test_scoring_policy_requires_isolated_fixed_placement(self):
        valid = self.task.data.scoring
        self.assertIs(self.task.grading_task(self.runtime, scoring=valid), self.task)
        for update in ({"mode": "shared"}, {"runtime": self.runtime.config},
                       {"env": {registry.KEY_ENV: KEY}}, {"network": None},
                       {"network": {"allow": ["*"]}}, {"retries": {"max_retries": 1}},
                       {"skip_on_timeout": True}, {"skip_on_truncated": True}):
            bad = vf.ScoringConfig.model_validate({**valid.model_dump(), **update})
            with self.assertRaisesRegex(RuntimeError, "unsafe_selection_scoring_policy"):
                self.task.grading_task(self.runtime, scoring=bad)

    async def test_verifier_must_be_separate_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "verifier_binding_failed"):
            await self.task.stage_verifier(self.trace, runtime("solver"))
        await self.task.stage_verifier(self.trace, self.runtime)
        self.assertEqual(self.trace.state.verifier_runtime, "verifier")

    async def test_finalize_captures_only_regular_bounded_files_without_executing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = b"raise RuntimeError('must not execute during collection')\n"
            (directory / "research.py").write_bytes(source)
            (directory / "policy.json").write_bytes(b"{}")
            result = subprocess.run([sys.executable, "-I", "-c", _CAPTURE, temporary],
                                    capture_output=True, text=True, check=True, timeout=10)
            self.runtime.run.return_value = SimpleNamespace(exit_code=0, stdout=result.stdout)
            await self.task.finalize(self.trace, self.runtime)
            from financial_research.prime_adapter import _artifact_bytes
            self.assertEqual(_artifact_bytes(self.trace.state.artifacts, "/workspace/research.py", 1048576), source)
            self.assertEqual(set(self.trace.state.artifacts), {"/workspace/research.py", "/workspace/policy.json"})
            (directory / "research.py").unlink()
            (directory / "research.py").symlink_to(directory / "policy.json")
            (directory / "policy.json").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
            result = subprocess.run([sys.executable, "-I", "-c", _CAPTURE, temporary],
                                    capture_output=True, text=True, check=True, timeout=10)
            self.runtime.run.return_value = SimpleNamespace(exit_code=0, stdout=result.stdout)
            # Re-finalization must overwrite prior captured bytes, not trust them.
            await self.task.finalize(self.trace, self.runtime)
            self.assertEqual(self.trace.state.artifacts, {"/workspace/research.py": None, "/workspace/policy.json": None})

    async def test_missing_artifacts_survive_actual_native_restore_then_score_zero(self):
        from verifiers.v1.grading import stage_verifier
        solver = runtime("solver")
        await self.task.finalize(self.trace, solver)
        await stage_verifier(self.task, self.trace, self.runtime)
        with patch.object(replay, "verify_program", new_callable=AsyncMock) as verify:
            await self.task.score(self.trace, self.runtime)
        verify.assert_not_awaited()
        self.assertEqual(evaluation_admission(self.trace), {"performance_usable": True, "status": "candidate_failure"})

    async def test_capture_service_failure_remains_unscored(self):
        self.runtime.run.side_effect = RuntimeError("transport unavailable")
        with self.assertRaisesRegex(RuntimeError, "artifact_capture_failed"):
            await self.task.finalize(self.trace, self.runtime)
        self.assertFalse(evaluation_admission(self.trace)["performance_usable"])

    async def test_missing_and_unsafe_artifacts_are_candidate_zero_without_replay(self):
        for artifact in (None, archive("../../research.py"),
                         archive("workspace/research.py", kind=tarfile.SYMTYPE),
                         archive("workspace/research.py", b"\xff")):
            current_trace = trace(self.task)
            current_trace.state.artifacts["/workspace/research.py"] = artifact
            with patch.object(replay, "verify_program", new_callable=AsyncMock) as verify:
                grade = await self.task._grade(current_trace, self.runtime)
            verify.assert_not_awaited()
            self.assertEqual(grade["reward"], 0)
            self.assertEqual(current_trace.info[INFO_KEY]["status"], "candidate_failure")

    async def test_concurrent_hooks_run_once_and_expose_only_aggregates(self):
        checked = {"status": "scored", "reward": .75, "reproduction_verified": True,
                   "packet_rewards": [1, .25, 1], "source_sha256": "private-diagnostic"}
        with patch.object(replay, "verify_program", new_callable=AsyncMock, return_value=checked) as verify:
            metrics, reward = await asyncio.gather(self.task.selection_metrics(self.trace, self.runtime),
                                                    self.task.selection_reward(self.trace, self.runtime))
        self.assertEqual(verify.await_count, 1)
        self.assertEqual(reward, .75)
        self.assertEqual(metrics["exact_success_fraction"], 2 / 3)
        self.assertEqual(metrics["current_exact_success"], 1)
        self.assertEqual(metrics["fresh_exact_success_fraction"], .5)
        self.assertEqual(metrics["all_packets_correct"], 0)
        args = verify.await_args
        self.assertEqual(len(args.args[3]), 2)
        exposed = json.dumps({"info": self.trace.info, "grade": self.trace.state.grade})
        for forbidden in (KEY, "packet_rewards", "private-diagnostic", "fresh_packets"):
            self.assertNotIn(forbidden, exposed)
        await self.task.score(self.trace, self.runtime)
        self.assertTrue(evaluation_admission(self.trace)["performance_usable"])
        stored = copy.deepcopy(self.trace)
        stored.state.grade = None  # Persisted native traces omit runtime state.
        stored.rewards["selection_reward"].score = .1
        self.assertFalse(evaluation_admission(stored)["performance_usable"])
        with self.assertRaises(replay.ReplayInfrastructureError):
            await self.task._grade(self.trace, runtime("solver"))

    async def test_native_task_score_records_exact_success_separately_from_reward(self):
        checked = {"status": "scored", "reward": 1, "reproduction_verified": True, "packet_rewards": [1, 1, 1]}
        with patch.object(replay, "verify_program", new_callable=AsyncMock, return_value=checked) as verify:
            await self.task.score(self.trace, self.runtime)
        self.assertEqual(verify.await_count, 1)
        self.assertEqual(self.trace.reward, 1)
        self.assertEqual(self.trace.metrics["all_packets_correct"], 1)
        self.assertEqual(self.trace.metrics["packets_scored"], 3)
        self.assertEqual(evaluation_admission(self.trace), {"performance_usable": True, "status": "scored"})
        for mutate in (lambda t: setattr(t, "is_timeout", True),
                       lambda t: setattr(t, "dropped", "service_failure"),
                       lambda t: t.metrics.pop("all_packets_correct"),
                       lambda t: t.metrics.update({"all_packets_correct": 0}),
                       lambda t: t.rewards.update({"other": t.rewards["selection_reward"]}),
                       lambda t: t.info["scoring_diagnostics"].clear(),
                       lambda t: setattr(t.rewards["selection_reward"], "score", .1)):
            corrupted = copy.deepcopy(self.trace)
            mutate(corrupted)
            self.assertFalse(evaluation_admission(corrupted)["performance_usable"])

    async def test_reproduction_failure_is_zero_not_service_failure(self):
        checked = {"status": "candidate_failure", "reward": 0, "reproduction_verified": False, "packet_rewards": []}
        with patch.object(replay, "verify_program", new_callable=AsyncMock, return_value=checked):
            await self.task.score(self.trace, self.runtime)
        self.assertEqual(self.trace.reward, 0)
        self.assertEqual(self.trace.metrics["candidate_failure"], 1)
        self.assertEqual(self.trace.info[INFO_KEY]["status"], "candidate_failure")

    async def test_service_failure_raises_without_reward_or_private_exception(self):
        with patch.object(replay, "verify_program", new_callable=AsyncMock,
                          side_effect=RuntimeError("private " + KEY)):
            with self.assertRaisesRegex(replay.ReplayInfrastructureError, "^scoring_service_failed$"):
                await self.task._grade(self.trace, self.runtime)
        self.assertIsNone(self.trace.state.grade)
        self.assertEqual(self.trace.rewards, {})
        self.assertEqual(self.trace.info["scoring_diagnostics"][INFO_KEY]["status"], "infrastructure_error")
        self.assertNotIn(KEY, json.dumps(self.trace.info))

    async def test_malformed_trusted_replay_result_is_infrastructure_error(self):
        values = [None, float("nan"), True, 1.1]
        for value in values:
            checked = {"status": "scored", "reward": value, "reproduction_verified": True, "packet_rewards": [1, 1, 1]}
            with patch.object(replay, "verify_program", new_callable=AsyncMock, return_value=checked):
                with self.assertRaises(replay.ReplayInfrastructureError):
                    await self.task._grade(trace(self.task), self.runtime)

    async def test_cancellation_propagates_as_unscored_infrastructure_event(self):
        with patch.object(replay, "verify_program", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.task._grade(self.trace, self.runtime)
        self.assertIsNone(self.trace.state.grade)
        self.assertEqual(self.trace.info[INFO_KEY]["reason"], "scoring_cancelled")


@unittest.skipUnless(os.environ.get("FINANCIAL_SELECTION_NATIVE_DOCKER_TESTS") == "1", "set FINANCIAL_SELECTION_NATIVE_DOCKER_TESTS=1 for real Docker lifecycle")
class NativeDockerTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_candidate_files_and_unrelated_logs_are_bounded_candidate_zero(self):
        from verifiers.v1.runtimes import provision_runtime
        image = os.environ["FINANCIAL_SELECTION_DOCKER_IMAGE"]
        with patch.dict(os.environ, {registry.KEY_ENV: KEY}):
            current = next(iter(FinancialSelectionTaskset(SelectionTasksetConfig(
                num_tasks=1, task=SelectionTaskConfig(image=image)))))
            config = DockerConfig(image=image, allow=[], workdir="/workspace", cpu=1, memory=1)
            current_trace = trace(current)
            current_trace.agent.config.runtime = config
            current_trace.state = SelectionState()
            async with provision_runtime(config, env={}) as live:
                await live.backend.prepare_setup()
                await current.setup(live, current_trace)
                created = await live.run(["python", "-I", "-c",
                    "from pathlib import Path; "
                    "Path('/workspace/research.py').symlink_to('/workspace/contract.json'); "
                    "Path('/workspace/policy.json').write_bytes(b'x'*(2097152+1)); "
                    "Path('/logs/artifacts').mkdir(parents=True,exist_ok=True); "
                    "Path('/logs/artifacts/unrelated.bin').write_bytes(b'x'*(6*1024*1024))"], {})
                self.assertEqual(created.exit_code, 0)
                await live.backend.prepare_execution([])
                await current.finalize(current_trace, live)
                self.assertEqual(current_trace.state.artifacts, {"/workspace/research.py": None, "/workspace/policy.json": None})
                with patch.object(replay, "verify_program", new_callable=AsyncMock) as verify:
                    await vf.grade_submission(current, current_trace, live)
                verify.assert_not_awaited()
            self.assertEqual(current_trace.reward, 0)
            self.assertEqual(evaluation_admission(current_trace), {"performance_usable": True, "status": "candidate_failure"})

    async def test_reference_native_collection_isolated_grading_and_fresh_replays(self):
        from verifiers.v1.runtimes import provision_runtime
        image = os.environ["FINANCIAL_SELECTION_DOCKER_IMAGE"]
        with patch.dict(os.environ, {registry.KEY_ENV: KEY}):
            current = next(iter(FinancialSelectionTaskset(SelectionTasksetConfig(
                num_tasks=1, task=SelectionTaskConfig(image=image)))))
            config = DockerConfig(image=image, allow=[], workdir="/workspace", cpu=1, memory=1)
            current_trace = trace(current)
            current_trace.agent.config.runtime = config
            current_trace.state = SelectionState()
            async with provision_runtime(config, env={}) as live:
                await live.backend.prepare_setup()
                await current.setup(live, current_trace)
                await live.backend.prepare_execution([])
                self.assertTrue(await current.validate(current_trace, live))
            self.assertEqual(current_trace.reward, 1)
            self.assertEqual(current_trace.metrics["all_packets_correct"], 1)
            self.assertTrue(evaluation_admission(current_trace)["performance_usable"])
            self.assertNotEqual(current_trace.state.solver_runtime, current_trace.state.verifier_runtime)
            self.assertNotIn(KEY, current_trace.model_dump_json())


if __name__ == "__main__":
    unittest.main()
