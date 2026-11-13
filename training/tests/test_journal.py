"""Per-worker sessions, cancellation accounting and transparent native responses."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig
from verifiers.v1.interception import server

from financial_portfolio_training.journal import WorkerJournal, ObservedTrainClient, install, uninstall, owner
from support import MESSAGES, response, trace


class JournalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.journal = WorkerJournal(Path(self.temporary.name))
        self.addCleanup(self.journal.close)
        self.native = trace()
        self.attempt = self.journal.register(self.native, episode_id="episode", episode_attempt=1, expected_messages=MESSAGES, model="fixture")
        self.journal.bind(self.native, MESSAGES)

    async def test_response_identity_and_every_training_array_preserved(self):
        expected = response()
        source = SimpleNamespace(get_response=AsyncMock(return_value=expected))
        client = ObservedTrainClient(source)
        turn, headers, sampling, dialect = object(), {"fixture": "value"}, object(), object()
        actual = await client.get_response(dialect, {"messages": deepcopy(MESSAGES), "model": "fixture"}, sampling,
                                           session_id=self.native.id, turn=turn, headers=headers)
        self.assertIs(actual, expected)
        self.assertIs(actual.tokens.sampler_support, expected.tokens.sampler_support)
        self.assertIs(actual.tokens.weight_versions, expected.tokens.weight_versions)
        self.assertIs(source.get_response.call_args.kwargs["turn"], turn)
        self.assertIs(source.get_response.call_args.kwargs["headers"], headers)
        self.assertEqual(self.attempt.requests[0]["outcome"], "success")

    async def test_unknown_and_changed_prompt_never_contact_provider(self):
        source = SimpleNamespace(get_response=AsyncMock(return_value=response()))
        client = ObservedTrainClient(source)
        with self.assertRaises(RuntimeError):
            await client.get_response(None, {"messages": MESSAGES, "model": "fixture"}, None, session_id="foreign")
        changed = deepcopy(MESSAGES)
        changed[0]["content"] += " changed"
        with self.assertRaises(ValueError):
            await client.get_response(None, {"messages": changed, "model": "fixture"}, None, session_id=self.native.id)
        source.get_response.assert_not_awaited()

    async def test_registered_task_prompt_cannot_be_replaced_before_binding(self):
        other = trace("another-trace")
        self.journal.register(other, episode_id="other", episode_attempt=1, expected_messages=MESSAGES, model="fixture")
        changed = deepcopy(MESSAGES)
        changed[1]["content"] = "unregistered task"
        with self.assertRaises(ValueError):
            self.journal.bind(other, changed)

    async def test_cancellation_gets_one_terminal_and_propagates(self):
        client = ObservedTrainClient(SimpleNamespace(get_response=AsyncMock(side_effect=asyncio.CancelledError)))
        with self.assertRaises(asyncio.CancelledError):
            await client.get_response(None, {"messages": MESSAGES, "model": "fixture"}, None, session_id=self.native.id)
        self.assertEqual(self.attempt.requests[0]["outcome"], "cancelled")
        self.assertEqual(sum(e["event"] == "request_terminal" for e in self.attempt.events), 1)

    async def test_session_ids_cannot_be_reused_and_completed_sessions_are_closed(self):
        with self.assertRaises(ValueError):
            self.journal.register(self.native, episode_id="other", episode_attempt=1, expected_messages=MESSAGES, model="fixture")
        self.journal.finish(self.native, admitted=False, reason="fixture")
        with self.assertRaises(RuntimeError):
            owner(self.native.id)

    async def test_same_task_siblings_are_independent_and_worker_stop_accounts_abandoned(self):
        sibling = trace("sibling")
        self.journal.register(sibling, episode_id="sibling-episode", episode_attempt=1, expected_messages=MESSAGES, model="fixture")
        self.assertEqual(self.native.task.data.id, sibling.task.data.id)
        self.journal.close()
        records = [json.loads(row) for row in (self.journal.directory / "events.jsonl").read_text().splitlines()]
        endings = [r for r in records if r["event"] == "attempt_terminal"]
        self.assertEqual(len(endings), 2)
        self.assertEqual({r["trace_id"] for r in endings}, {self.native.id, sibling.id})
        self.assertNotIn("Public instructions", json.dumps(records))

    async def test_resolver_lifecycle_accepts_train_only_and_restores_on_stop(self):
        previous = server.resolve_client
        with patch.object(server, "resolve_client", return_value=object()) as factory:
            install(self.journal)
            try:
                config = TrainClientConfig(base_url="http://127.0.0.1:1", api_key_var="FIXTURE_TOKEN")
                self.assertIsInstance(server.resolve_client(config), ObservedTrainClient)
                with self.assertRaises(ValueError):
                    server.resolve_client(EvalClientConfig(api_key_var="FIXTURE_TOKEN"))
            finally:
                uninstall(self.journal)
            self.assertIs(server.resolve_client, factory)
        self.assertIs(server.resolve_client, previous)

    async def test_invalid_token_alignment_is_provider_failure_without_fabricated_tokens(self):
        invalid = response()
        invalid.tokens.completion_logprobs = np.array([], dtype=np.float32)
        client = ObservedTrainClient(SimpleNamespace(get_response=AsyncMock(return_value=invalid)))
        with self.assertRaises(ValueError):
            await client.get_response(None, {"messages": MESSAGES, "model": "fixture"}, None, session_id=self.native.id)
        self.assertEqual(self.attempt.requests[0]["outcome"], "error")
        self.assertEqual(len(invalid.tokens.completion_logprobs), 0)
