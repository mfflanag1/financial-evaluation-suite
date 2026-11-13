"""Exercise native Env orchestration with synthetic agents, without containers."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from uuid import uuid4
import unittest
from unittest.mock import AsyncMock, patch

import verifiers.v1 as vf
import msgpack
from prime_rl.env.encoding import unpack
from prime_rl.env.types import RunRequest
from prime_rl.env.worker import EnvWorker
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.interception import server

from financial_portfolio_training.journal import ObservedTrainClient, digest
from support import response, trace
from test_contract import configuration


class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.key = patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "synthetic-host-fixture-key-0000000000000000"})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.config = configuration(self.root).orchestrator.train.source[0].env
        self.config.agent.runtime.image = self.config.taskset.task.image = "sha256:" + "1" * 64
        self.env = vf.load_environment(self.config)
        self.task = next(iter(self.env.taskset.load()))
        self.ctx = ModelContext(model="fixture", client=TrainClientConfig(
            base_url="http://127.0.0.1:1", api_key_var="FIXTURE_TOKEN"), sampling=vf.Sampling(top_p=1.0))
        self.addAsyncCleanup(self.env.stop)

    def agents(self, *, reward=1.0, pending_tool=False, fail_run=False, corrupt_graph=False):
        def factory(ctx, completed, on_trace, on_discard):
            async def run(task):
                native = trace(uuid4().hex, reward)
                native.task = vf.TraceTask(type=type(task).__name__, data=task.data)
                system = "\n\n".join(p for p in (task.data.system_prompt, "External network access is disabled.") if p)
                messages = self.env._harnesses["agent"].outbound_messages(task.data.model_copy(update={"system_prompt": system}))
                native.nodes[0].message.content = messages[0]["content"]
                native.nodes[1].message.content = messages[1]["content"]
                on_trace(native)
                completed.append(native)
                if fail_run:
                    raise RuntimeError("synthetic agent interruption")
                journal = self.env.journal
                attempt = journal.attempts[native.id]
                journal.bind(native, messages)
                journal.emit(attempt, "harness_started")
                client = ObservedTrainClient(SimpleNamespace(get_response=AsyncMock(return_value=response())))
                await client.get_response(None, {"messages": messages, "model": "fixture"}, ctx.sampling, session_id=native.id)
                journal.emit(attempt, "harness_terminal", outcome="returned", exit_code=0)
                journal.emit(attempt, "runtime_health", alive=True)
                journal.emit(attempt, "scoring_complete", reward=reward,
                             outcome_sha256=digest(native.info["financial_research_outcome"]))
                if pending_tool:
                    attempt.tools["pending-tool"] = "allowed"
                if corrupt_graph:
                    native.nodes[1].parent = 1
                return native
            return SimpleNamespace(agent=SimpleNamespace(run=run))
        return factory

    async def test_sequential_same_task_episodes_and_worker_restart(self):
        original = server.resolve_client
        await self.env.start()
        first_worker = self.env.journal.worker_id
        with patch.object(self.env, "_episode_agents", self.agents()):
            first = await self.env.run_episode(self.task, self.ctx)
        with patch.object(self.env, "_episode_agents", self.agents(reward=0)):
            second = await self.env.run_episode(self.task, self.ctx)
        self.assertTrue(first.ok and second.ok)
        self.assertFalse(first.traces[0].is_dropped or second.traces[0].is_dropped)
        self.assertEqual(first.task.data.id, second.task.data.id)
        self.assertNotEqual(first.traces[0].id, second.traces[0].id)
        self.assertEqual(self.env.journal.attempts, {})
        await self.env.stop()
        self.assertIs(server.resolve_client, original)
        await self.env.start()
        self.assertNotEqual(self.env.journal.worker_id, first_worker)

    async def test_env_finalize_drops_unanswered_tool_without_changing_reward(self):
        await self.env.start()
        with patch.object(self.env, "_episode_agents", self.agents(pending_tool=True)):
            episode = await self.env.run_episode(self.task, self.ctx)
        self.assertTrue(episode.ok)
        self.assertTrue(episode.traces[0].is_dropped)
        self.assertEqual(episode.traces[0].reward, 1.0)

    async def test_broken_admission_gate_leaves_native_episode_not_ok(self):
        await self.env.start()
        with patch.object(self.env, "_episode_agents", self.agents()), \
             patch("financial_portfolio_training.environment.admission.apply", side_effect=RuntimeError("fixture gate failure")):
            episode = await self.env.run_episode(self.task, self.ctx)
        self.assertFalse(episode.ok)
        self.assertTrue(episode.traces[0].is_dropped)
        self.assertEqual(episode.traces[0].reward, 1.0)
        self.assertEqual(self.env.journal.attempts, {})

    async def test_interrupted_env_run_still_records_one_attempt_terminal(self):
        await self.env.start()
        directory = self.env.journal.directory
        with patch.object(self.env, "_episode_agents", self.agents(fail_run=True)):
            episode = await self.env.run_episode(self.task, self.ctx)
        self.assertFalse(episode.ok)
        self.assertTrue(episode.traces[0].is_dropped)
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["event"] == "attempt_terminal" for row in rows), 1)

    async def test_worker_stop_drains_cancelled_episode_before_closing_journal(self):
        original = server.resolve_client
        await self.env.start()
        journal = self.env.journal
        started = asyncio.Event()
        def agents(ctx, completed, on_trace, on_discard):
            async def run(task):
                native = trace(uuid4().hex)
                native.task = vf.TraceTask(type=type(task).__name__, data=task.data)
                on_trace(native)
                completed.append(native)
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(.01)
                    journal.emit(journal.attempts[native.id], "fixture_finalizer_complete")
            return SimpleNamespace(agent=SimpleNamespace(run=run))
        with patch.object(self.env, "_episode_agents", agents):
            active = asyncio.create_task(self.env.run_episode(self.task, self.ctx))
            await started.wait()
            active.cancel()
            await self.env.stop()
            with self.assertRaises(asyncio.CancelledError):
                await active
        self.assertTrue(journal.closed)
        self.assertIs(server.resolve_client, original)
        rows = [json.loads(line) for line in (journal.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["event"] == "fixture_finalizer_complete" for row in rows), 1)
        self.assertEqual(sum(row["event"] == "attempt_terminal" for row in rows), 1)
        self.assertEqual(self.env._active_episodes, set())

    async def test_invalid_graph_returns_native_worker_error_without_serialization_hang(self):
        await self.env.start()
        worker = object.__new__(EnvWorker)
        worker.env, worker.task_cls = self.env, type(self.task)
        worker.data_cls = worker.task_cls.data_type()
        worker._gate, worker._running, worker._kept, worker._delivered = None, {}, {}, {}
        worker.frontend = SimpleNamespace(send_multipart=AsyncMock())
        request = RunRequest(task_data=self.task.data.model_dump(mode="json"),
                             client=self.ctx.client, model=self.ctx.model, sampling=self.ctx.sampling)
        with patch.object(self.env, "_episode_agents", self.agents(corrupt_graph=True)):
            await asyncio.wait_for(worker._handle(b"client", b"request", b"run",
                msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)), timeout=5)
        frames = [call.args[0] for call in worker.frontend.send_multipart.await_args_list]
        [reply] = [unpack(frame[-1]) for frame in frames if frame[-2] == b"reply"]
        self.assertFalse(reply["success"])
        self.assertIn("invalid native token graph", reply["error"])
        rows = [json.loads(line) for line in (self.env.journal.directory / "events.jsonl").read_text().splitlines()]
        [terminal] = [row for row in rows if row["event"] == "attempt_terminal"]
        self.assertFalse(terminal["admitted"])
        self.assertEqual(terminal["reason"], "native_graph_unresolved")
        self.assertEqual(next(row["reward"] for row in rows if row["event"] == "scoring_complete"), 1)

    async def test_training_defaults_reject_nonzero_retries_and_inference_override(self):
        for setting in ("retries", "model", "mounts", "mutable_image", "different_image", "placeholder_image"):
            with self.subTest(setting=setting):
                config = deepcopy(self.config)
                if setting == "retries":
                    config.agent.retries.max_retries = 1
                elif setting == "model":
                    config.agent.model = "unbound-model"
                elif setting == "mounts":
                    config.agent.runtime.mounts = {str(self.root): "/unapproved-host-context"}
                elif setting == "mutable_image":
                    config.agent.runtime.image = config.taskset.task.image = "python:3.12-slim"
                elif setting == "different_image":
                    config.taskset.task.image = "sha256:" + "2" * 64
                else:
                    config.agent.runtime.image = config.taskset.task.image = "sha256:" + "0" * 64
                env = vf.load_environment(config)
                with self.assertRaises(ValueError):
                    await env.start()
                await env.stop()
