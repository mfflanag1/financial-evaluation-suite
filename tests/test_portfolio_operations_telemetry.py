"""Offline journal, cancellation, and pinned native interception regressions."""

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
                "financial_portfolio_operations"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))

from financial_portfolio_operations.telemetry import (
    RequestJournal, TelemetryClient, active_journal, capture_requests,
    read_journal, summarize_journal,
)

try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None


def registered(directory):
    journal = RequestJournal(directory)
    journal.register(trace_id="trace-a", task_id="task-a", model="fixture-model")
    return journal


class JournalTests(unittest.TestCase):
    def test_fresh_private_directory_and_pending_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal"
            journal = registered(path)
            try:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(journal.path.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    RequestJournal(path)
                with self.assertRaises(RuntimeError):
                    journal.start("unregistered", transport="response")
                journal.start("trace-a", transport="response")
                summary = summarize_journal(path)
                self.assertTrue(summary["integrity_ok"])
                self.assertEqual(summary["pending_count"], 1)
                self.assertIsNone(summary["known_usage"]["cost"])
                self.assertFalse(summary["cost_complete"])
            finally:
                journal.close()

    def test_real_process_exit_leaves_fsynced_pending_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal"
            code = ("import os, sys\nfrom financial_portfolio_operations.telemetry import RequestJournal\n"
                    "j=RequestJournal(sys.argv[1])\n"
                    "j.register(trace_id='crashed',task_id='task-a',model='fixture-model')\n"
                    "j.start('crashed',transport='response')\nos._exit(17)\n")
            env = {**os.environ, "PYTHONPATH": str(ROOT / "environments/financial_portfolio_operations/src")}
            result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], env=env,
                                    capture_output=True, text=True, check=False, timeout=10)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual(summarize_journal(path)["pending_count"], 1)

    def test_corruption_duplicate_orphan_identity_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = registered(Path(tmp) / "journal")
            request = journal.start("trace-a", transport="response")
            request.finish("success", usage={"prompt_tokens": 2, "completion_tokens": 3,
                                             "cost": 0.2, "secret": "do-not-write"})
            journal.close()
            records = read_journal(journal.path)["records"]
            self.assertNotIn("do-not-write", journal.path.read_text())
            summary = summarize_journal(journal.path)
            self.assertEqual(summary["known_usage"]["cost"], 0.2)
            self.assertTrue(summary["cost_complete"])
            self.assertFalse(summary["usage_complete"])
            with journal.path.open("a") as stream:
                stream.write(json.dumps(records[1]) + "\n")
                stream.write(json.dumps({**records[1], "request_id": "orphan"}) + "\n")
                stream.write('{"partial":')
            summary = summarize_journal(journal.path)
            kinds = {item["kind"] for item in summary["issues"]}
            self.assertTrue({"duplicate_terminal", "orphan_terminal", "invalid_record"} <= kinds)
            self.assertFalse(summary["integrity_ok"])
            self.assertFalse(summary["cost_complete"])

    def test_strict_integer_and_finite_usage_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = registered(Path(tmp) / "journal")
            journal.start("trace-a", transport="response").finish("success", usage={
                "prompt_tokens": True, "completion_tokens": -1, "cost": 10 ** 1000})
            journal.close()
            records = read_journal(journal.path)["records"]
            self.assertTrue(all(value is None for value in records[1]["usage"].values()))
            mutations = [{**records[0], "schema_version": True},
                         {**records[0], "request_index": True},
                         {**records[1], "duration_seconds": 10 ** 1000},
                         {**records[1], "usage_is_final": 1},
                         {**records[1], "usage": {**records[1]["usage"], "cost": float("nan")}}]
            journal.path.write_text("".join(json.dumps(record) + "\n" for record in mutations))
            loaded = read_journal(journal.path)
            self.assertEqual(loaded["records"], [])
            self.assertEqual(len(loaded["errors"]), len(mutations))


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_records_before_await_and_never_serializes_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = registered(Path(tmp) / "journal")
            body = {"messages": [{"content": "prompt-secret"}]}
            headers = {"Authorization": "credential-secret"}
            response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3,
                                       cached_input_tokens=4, reasoning_tokens=1, cost=0.25,
                                       secret="usage-secret"), raw={"text": "response-secret"})

            async def call(*args, **kwargs):
                self.assertEqual(summarize_journal(journal.path)["pending_count"], 1)
                self.assertIs(args[1], body)
                self.assertIs(kwargs["headers"], headers)
                return response

            delegate = SimpleNamespace(get_response=AsyncMock(side_effect=call))
            result = await TelemetryClient(delegate, journal).get_response(None, body, None,
                                                                         session_id="trace-a", headers=headers)
            self.assertIs(result, response)
            self.assertEqual(delegate.get_response.await_count, 1)
            journal.close()
            text = journal.path.read_text()
            for secret in ("prompt-secret", "credential-secret", "response-secret", "usage-secret"):
                self.assertNotIn(secret, text)
            summary = summarize_journal(journal.path)
            self.assertEqual(summary["outcomes"], {"success": 1, "error": 0, "cancelled": 0})
            self.assertTrue(summary["usage_complete"])

    async def test_error_is_not_retried_and_exception_message_is_omitted(self):
        class ProviderFailure(Exception):
            status_code = 429

        with tempfile.TemporaryDirectory() as tmp:
            journal = registered(Path(tmp) / "journal")
            failure = ProviderFailure("provider-secret https://credential-secret@host")
            delegate = SimpleNamespace(get_response=AsyncMock(side_effect=failure))
            with self.assertRaises(ProviderFailure) as caught:
                await TelemetryClient(delegate, journal).get_response(None, {}, None, session_id="trace-a")
            self.assertIs(caught.exception, failure)
            self.assertEqual(delegate.get_response.await_count, 1)
            journal.close()
            terminal = read_journal(journal.path)["records"][1]
            self.assertEqual(terminal["http_status"], 429)
            self.assertEqual(terminal["exception_type"], "ProviderFailure")
            self.assertNotIn("secret", journal.path.read_text())

    async def test_cancelled_request_persists_terminal_before_propagating(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = registered(Path(tmp) / "journal")
            began = asyncio.Event()

            async def blocked(*args, **kwargs):
                began.set()
                await asyncio.Event().wait()

            delegate = SimpleNamespace(get_response=AsyncMock(side_effect=blocked))
            task = asyncio.create_task(TelemetryClient(delegate, journal).get_response(None, {}, None,
                                                                                    session_id="trace-a"))
            await began.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            summary = summarize_journal(journal.path)
            self.assertEqual(summary["outcomes"]["cancelled"], 1)
            self.assertEqual(summary["pending_count"], 0)
            self.assertEqual(summary["missing_usage_by_field"]["cost"], 1)
            self.assertEqual(delegate.get_response.await_count, 1)
            journal.close()


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class NativeTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_midstream_cancel_and_truncation_keep_observed_usage_without_claiming_complete_cost(self):
        from verifiers.v1.clients.client import RelayReply
        from verifiers.v1.dialects import ChatDialect
        from verifiers.v1.errors import ProviderError
        from verifiers.v1.interception.server import _collect_stream

        for mode in ("cancel", "truncated"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                journal = registered(Path(tmp) / "journal")
                yielded = asyncio.Event()
                chunk = {"id": "fixture-response", "object": "chat.completion.chunk", "created": 0,
                         "model": "fixture-model", "choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": "secret"}, "finish_reason": None}],
                         "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                                   "prompt_tokens_details": {"cached_tokens": 2}, "cost": 0.125}}

                async def chunks():
                    yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
                    yielded.set()
                    if mode == "cancel":
                        await asyncio.Event().wait()

                close = AsyncMock()
                delegate = SimpleNamespace(relay=AsyncMock(return_value=RelayReply(
                    content_type="text/event-stream", chunks=chunks(), close=close)))
                dialect = ChatDialect()
                reply = await TelemetryClient(delegate, journal).relay(dialect, {}, session_id="trace-a")
                collect = asyncio.create_task(_collect_stream(dialect, reply))
                await yielded.wait()
                if mode == "cancel":
                    collect.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await collect
                else:
                    with self.assertRaises(ProviderError):
                        await collect
                summary = summarize_journal(journal.path)
                self.assertEqual(summary["request_starts"], 1)
                self.assertEqual(summary["matched_terminals"], 1)
                self.assertEqual(summary["outcomes"]["cancelled" if mode == "cancel" else "error"], 1)
                self.assertEqual(summary["known_usage"]["prompt_tokens"], 5)
                self.assertEqual(summary["known_usage"]["cost"], 0.125)
                self.assertEqual(summary["nonfinal_usage_requests"], 1)
                self.assertFalse(summary["cost_complete"])
                close.assert_awaited_once()
                self.assertEqual(delegate.relay.await_count, 1)
                self.assertNotIn("secret", journal.path.read_text())
                journal.close()

    async def test_context_restoration_and_harness_preserves_inherited_launch(self):
        from financial_portfolio_operations.harness import (
            FinancialPortfolioOperationsHarness, FinancialPortfolioOperationsHarnessConfig,
        )
        from financial_portfolio_staged.harness import FinancialPortfolioStagedHarness
        from verifiers.v1.configs.harness import HarnessConfig
        from verifiers.v1.interception import server
        from verifiers.v1.utils.generic import concrete_type

        self.assertIs(concrete_type(FinancialPortfolioOperationsHarness, HarnessConfig),
                      FinancialPortfolioOperationsHarnessConfig)
        harness = FinancialPortfolioOperationsHarness(FinancialPortfolioOperationsHarnessConfig())
        self.assertEqual(harness.config.sdk_max_retries, 0)
        self.assertEqual(harness.config.resolved_env, {})
        with self.assertRaises(RuntimeError):
            await harness.launch(None, None, None, None, None, {}, None)
        original = server.resolve_client
        with tempfile.TemporaryDirectory() as tmp:
            with capture_requests(Path(tmp) / "journal") as journal:
                with self.assertRaises(RuntimeError):
                    with capture_requests(Path(tmp) / "second"):
                        pass
                with patch.object(FinancialPortfolioStagedHarness, "launch", new_callable=AsyncMock) as launch:
                    trace = SimpleNamespace(id="trace-a", task=SimpleNamespace(data=SimpleNamespace(id="task-a")))
                    ctx = SimpleNamespace(model="fixture-model")
                    await harness.launch(ctx, trace, None, "endpoint", "secret", {}, None,
                                         tool_interception_url="gate")
                    launch.assert_awaited_once_with(ctx, trace, None, "endpoint", "secret", {}, None,
                                                   tool_interception_url="gate")
                    journal.start("trace-a", transport="response").finish("success")
                self.assertIs(active_journal(), journal)
            self.assertIs(server.resolve_client, original)
            with self.assertRaises(RuntimeError):
                active_journal()

    async def test_native_interception_concurrent_streams_and_cancelled_trace(self):
        from aiohttp import web
        import httpx
        from verifiers.v1.clients import ModelContext
        from verifiers.v1.configs.agent import AgentConfig
        from verifiers.v1.configs.client import EvalClientConfig
        from verifiers.v1.interception.server import InterceptionServer
        from verifiers.v1.session import RolloutSession
        from verifiers.v1.task import TaskData
        from verifiers.v1.trace import AgentInfo, Trace, TraceTask

        seen, waiting = [], asyncio.Event()

        async def provider(request):
            seen.append(request.headers.get("X-Session-ID"))
            body = await request.json()
            if request.headers.get("X-Session-ID") == "trace-cancel":
                waiting.set()
                await asyncio.Event().wait()
            chunk = {"id": "fixture-response", "object": "chat.completion.chunk", "created": 0,
                     "model": body["model"], "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "response-secret"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
                               "prompt_tokens_details": {"cached_tokens": 2},
                               "completion_tokens_details": {"reasoning_tokens": 1}, "cost": 0.125}}
            return web.Response(body=b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/chat/completions", provider)
        upstream = web.AppRunner(app, shutdown_timeout=0.01)
        await upstream.setup()
        site = web.TCPSite(upstream, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with capture_requests(Path(tmp) / "journal") as journal:
                    ctx = ModelContext(model="fixture-model", client=EvalClientConfig(
                        base_url=f"http://127.0.0.1:{port}/v1", api_key_var="UNSET_FIXTURE_API_KEY"))
                    sessions = [RolloutSession(ctx=ctx, trace=Trace(
                        id=f"trace-{name}", task=TraceTask(type="fixture", data=TaskData(id=f"task-{name}")),
                        agent=AgentInfo(config=AgentConfig()))) for name in ("a", "b", "cancel")]
                    for session in sessions:
                        journal.register(trace_id=session.trace.id, task_id=session.trace.task.data.id,
                                         model=ctx.model)
                    async with InterceptionServer() as interception, AsyncExitStack() as stack:
                        slots = [await stack.enter_async_context(interception.acquire(session)) for session in sessions]
                        async with httpx.AsyncClient(timeout=5) as client:
                            async def post(slot):
                                return await client.post(slot[0] + "/v1/chat/completions",
                                    headers={"Authorization": "Bearer " + slot[1]},
                                    json={"model": "ignored", "messages": [{"role": "user", "content": "prompt-secret"}],
                                          "stream": True})
                            responses = await asyncio.gather(*(post(slot) for slot in slots[:2]))
                            self.assertEqual([response.status_code for response in responses], [200, 200])
                            blocked = asyncio.create_task(post(slots[2]))
                            await asyncio.wait_for(waiting.wait(), 3)
                            sessions[2].release()
                            # Native cancellation must reach the wrapped provider await.
                            for _ in range(100):
                                if summarize_journal(journal.path)["outcomes"]["cancelled"]:
                                    break
                                await asyncio.sleep(0.01)
                            blocked.cancel()
                            await asyncio.gather(blocked, return_exceptions=True)
                    summary = summarize_journal(journal.path)
                    self.assertEqual(summary["outcomes"], {"success": 2, "error": 0, "cancelled": 1})
                    self.assertTrue(summary["integrity_ok"])
                    self.assertEqual(summary["known_usage"]["cost"], 0.25)
                    self.assertEqual(summary["missing_usage_by_field"]["cost"], 1)
                    self.assertCountEqual(seen, ["trace-a", "trace-b", "trace-cancel"])
                    self.assertEqual(len(sessions[2].trace.calls), 0)
                    self.assertNotIn("secret", journal.path.read_text())
        finally:
            await upstream.cleanup()


if __name__ == "__main__":
    unittest.main()
