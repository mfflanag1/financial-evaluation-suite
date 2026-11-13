"""Selection hooks preserve finance while observing actual inherited boundaries."""
import asyncio
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import DockerConfig
from financial_selection import replay
from financial_selection.prime_adapter import FinancialSelectionTask, SelectionData, SelectionState, SelectionTasksetConfig
from financial_selection_training.journal import WorkerJournal, INFO_KEY
from financial_selection_training.task import FinancialSelectionTrainingTaskset

KEY = "ab" * 32
IMAGE = "sha256:" + "a" * 64


def archive(path, content):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as target:
        member = tarfile.TarInfo(path.lstrip("/"))
        member.size = len(content)
        target.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def runtime(name):
    return SimpleNamespace(name=name, type="docker", config=DockerConfig(image=IMAGE, allow=[]), env={},
        write=AsyncMock(), run=AsyncMock(return_value=SimpleNamespace(exit_code=0,
            stdout='{"research.py":null,"policy.json":null}')))


class TaskObservationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.secret = patch.dict(os.environ, {"FINANCIAL_SELECTION_KEY": KEY})
        self.secret.start()
        self.addCleanup(self.secret.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.journal = WorkerJournal(Path(self.temporary.name))
        self.addCleanup(self.journal.close)
        self.task = next(iter(FinancialSelectionTrainingTaskset(SelectionTasksetConfig(num_tasks=1, task={"image": IMAGE})).load()))
        self.trace = vf.Trace[SelectionData, SelectionState](task=vf.TraceTask(type=type(self.task).__name__, data=self.task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig()), state=SelectionState())
        self.attempt = self.journal.register(self.trace, episode_id="fixture", episode_attempt=1,
            expected_messages=[{"role": "system", "content": "fixture"}, {"role": "user", "content": self.task.data.prompt}], model="fixture")

    async def staged(self):
        await self.task.setup(runtime("solver"), self.trace)
        await self.task.finalize(self.trace, runtime("solver"))
        await self.task.setup(runtime("verifier"), self.trace)
        await self.task.stage_verifier(self.trace, runtime("verifier"))
        self.trace.state.artifacts = {
            "/workspace/research.py": archive("/workspace/research.py", b"print('fixture')\n"),
            "/workspace/policy.json": archive("/workspace/policy.json", b"{}"),
        }

    async def test_inherited_isolated_boundaries_and_one_cached_scoring_observation(self):
        await self.staged()
        checked = {"status": "scored", "reward": .5, "reproduction_verified": True, "packet_rewards": [.5, .5, .5]}
        with patch.object(replay, "verify_program", AsyncMock(return_value=checked)) as verifier:
            first = await self.task._grade(self.trace, runtime("verifier"))
            second = await self.task._grade(self.trace, runtime("verifier"))
        self.assertEqual(first, second)
        self.assertEqual(first["reward"], .5)
        verifier.assert_awaited_once()
        counts = {name: sum(e["event"] == name for e in self.attempt.events) for name in
                  ("solver_setup_complete", "artifacts_captured", "verifier_setup_complete", "verifier_staged", "scoring_complete")}
        self.assertTrue(all(v == 1 for v in counts.values()), counts)
        self.assertNotIn(KEY, json.dumps(self.attempt.events))

    async def test_candidate_program_timeout_is_financial_zero_and_service_timeout_raises(self):
        await self.staged()
        with patch.object(replay, "verify_program", AsyncMock(return_value={"status": "candidate_failure", "reward": 0,
                "reproduction_verified": False, "reason": "analysis_timeout"})):
            result = await self.task._grade(self.trace, runtime("verifier"))
        self.assertEqual(result["reward"], 0)
        self.assertEqual(self.trace.info["financial_selection_outcome"]["status"], "candidate_failure")
        self.trace.state.grade = None
        with patch.object(replay, "verify_program", AsyncMock(side_effect=TimeoutError)):
            with self.assertRaises(replay.ReplayInfrastructureError):
                await self.task._grade(self.trace, runtime("verifier"))
        self.assertEqual(self.trace.info["financial_selection_outcome"]["status"], "infrastructure_error")
        self.assertTrue(any(e["event"] == "scoring_failed" for e in self.attempt.events))

    async def test_cancelled_scoring_propagates_and_keeps_original_rewards(self):
        await self.staged()
        self.trace.rewards = {"selection_reward": vf.Reward(score=.25)}
        with patch.object(replay, "verify_program", AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await self.task._grade(self.trace, runtime("verifier"))
        self.assertEqual(self.trace.reward, .25)
        self.assertEqual(self.trace.info["financial_selection_outcome"]["reason"], "scoring_cancelled")
        self.assertEqual(sum(e["event"] == "scoring_failed" for e in self.attempt.events), 1)

    async def test_tool_gate_and_result_are_observed_without_request_replacement(self):
        request = vf.Request(messages=[vf.AssistantMessage(content="", tool_calls=[
            vf.ToolCall(id="call", name="bash", arguments='{"command": "true"}')]),
            vf.ToolMessage(tool_call_id="call", content="")])
        self.assertIsNone(await self.task.observe_tool(request, self.trace))
        self.assertEqual(list(self.attempt.tools.values()), ["allowed"])
        request.messages[-1].content = "exit code 0"
        self.assertIsNone(await self.task.observe_tool(request, self.trace))
        self.assertEqual(list(self.attempt.tools.values()), ["result_observed"])

    async def test_model_free_validation_is_explicitly_marked_nontrainable(self):
        with patch.object(FinancialSelectionTask, "validate", AsyncMock(return_value=True)):
            self.assertTrue(await self.task.validate(self.trace, runtime("solver")))
        self.assertTrue(self.trace.info[INFO_KEY]["model_free_reference"])
