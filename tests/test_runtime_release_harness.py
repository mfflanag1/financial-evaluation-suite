"""Runtime v1 regressions against the pinned native prompt and tool boundaries."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
for package in ROOT.joinpath("environments").iterdir():
    if (package / "src").is_dir():
        sys.path.insert(0, str(package / "src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None

if vf is not None:
    from financial_portfolio_runtime.harness import (
        FinancialPortfolioRuntimeHarness, FinancialPortfolioRuntimeHarnessConfig, GUIDANCE_MARKER,
    )
    from financial_portfolio_runtime.telemetry import capture_runtime, INFO_KEY
    from financial_portfolio_runtime.prime_adapter import FinancialPortfolioRuntimeTask
    from verifiers.v1.runtimes import ProgramResult
    from verifiers.v1.task import TaskData


@unittest.skipUnless(vf is not None, "Install pinned runtime dependencies")
class RuntimePromptBoundary(unittest.IsolatedAsyncioTestCase):
    def trace(self):
        return SimpleNamespace(id="fixture-trace", task=SimpleNamespace(data=SimpleNamespace(id="fixture-task")), info={})

    async def test_native_resolve_copy_launch_adds_guidance_once_and_binds_network_notice(self):
        harness = FinancialPortfolioRuntimeHarness(FinancialPortfolioRuntimeHarnessConfig(guidance="Read the contract."))
        data = TaskData(id="fixture", system_prompt="Public instructions.\n\nExternal network access is disabled.",
                        prompt="Solve the supplied task.")
        system, prompt = harness.resolve_prompt(data)
        copied = data.model_copy(update={"system_prompt": system, "prompt": prompt})
        trace = self.trace()
        runtime = SimpleNamespace(alive=AsyncMock(return_value=True))
        with tempfile.TemporaryDirectory() as directory, capture_runtime(Path(directory) / "journal") as journal:
            with patch("financial_portfolio_runtime.harness.launch_chat_program", AsyncMock(return_value=ProgramResult(0, "", ""))) as launch:
                await harness.launch(SimpleNamespace(model="fixture-model"), trace, runtime, "http://fixture", "secret", {}, copied)
            outbound = harness.outbound_messages(copied)
            self.assertEqual(outbound[0]["content"].count(GUIDANCE_MARKER), 1)
            self.assertIn("External network access is disabled.", outbound[0]["content"])
            self.assertEqual(launch.call_args.args[8], outbound[0]["content"])
            journal.verify(trace.id, {"messages": outbound})
            later = [*outbound, {"role": "assistant", "content": "Checking data."}]
            journal.verify(trace.id, {"messages": later})
            altered = deepcopy(later)
            altered[0]["content"] += " changed"
            with self.assertRaisesRegex(Exception, "Outbound prompt differs"):
                journal.verify(trace.id, {"messages": altered})
        self.assertEqual(data.system_prompt.count(GUIDANCE_MARKER), 0)

    async def test_default_has_no_guidance_and_missing_notice_fails_exact_binding(self):
        harness = FinancialPortfolioRuntimeHarness(FinancialPortfolioRuntimeHarnessConfig())
        data = TaskData(id="fixture", prompt="Task", system_prompt="Rules\n\nExternal network access is disabled.")
        expected = harness.outbound_messages(data)
        self.assertNotIn(GUIDANCE_MARKER, expected[0]["content"])
        with tempfile.TemporaryDirectory() as directory, capture_runtime(Path(directory) / "journal") as journal:
            journal.bind(self.trace(), expected)
            wrong = deepcopy(expected)
            wrong[0]["content"] = wrong[0]["content"].replace("\n\nExternal network access is disabled.", "")
            with self.assertRaisesRegex(Exception, "Outbound prompt differs"):
                journal.verify("fixture-trace", {"messages": wrong})

    async def test_cancelled_launch_persists_terminal_and_does_not_swallow_cancel(self):
        harness = FinancialPortfolioRuntimeHarness(FinancialPortfolioRuntimeHarnessConfig())
        trace = self.trace()
        with tempfile.TemporaryDirectory() as directory, capture_runtime(Path(directory) / "journal"):
            with patch("financial_portfolio_runtime.harness.launch_chat_program", AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await harness.launch(SimpleNamespace(model="fixture-model"), trace,
                                         SimpleNamespace(alive=AsyncMock(return_value=False)), "endpoint", "secret", {},
                                         TaskData(id="fixture", prompt="Task"))
            events = [json.loads(line) for line in (Path(directory) / "journal/lifecycle.jsonl").read_text().splitlines()]
        terminal = next(event for event in events if event["event"] == "harness_terminal")
        self.assertEqual(terminal["outcome"], "cancelled")
        self.assertEqual(events[-1]["alive"], False)
        self.assertNotIn("secret", json.dumps(events))

    async def test_tool_gate_and_empty_result_are_distinct_host_events(self):
        from financial_portfolio_clarified.prime_adapter import FinancialPortfolioClarifiedTask
        assistant = vf.AssistantMessage(tool_calls=[vf.ToolCall(id="call", name="bash", arguments='{"command":"true"}')])
        request = vf.Request(messages=[vf.UserMessage(content="Task"), assistant,
                                       vf.ToolMessage(tool_call_id="call", name="bash", content="")])
        trace = self.trace()
        with tempfile.TemporaryDirectory() as directory, capture_runtime(Path(directory) / "journal"):
            with patch.object(FinancialPortfolioClarifiedTask, "checkpoint", AsyncMock(return_value=None)):
                task = object.__new__(FinancialPortfolioRuntimeTask)
                await task.checkpoint(request, trace)
                state = trace.info[INFO_KEY]["tools"]
                self.assertEqual(next(iter(state.values()))["status"], "allowed")
                await task.checkpoint(request, trace)
                self.assertEqual(next(iter(state.values()))["status"], "result_observed")

    async def test_cancellation_during_health_probe_propagates(self):
        harness = FinancialPortfolioRuntimeHarness(FinancialPortfolioRuntimeHarnessConfig())
        trace = self.trace()
        with tempfile.TemporaryDirectory() as directory, capture_runtime(Path(directory) / "journal"):
            with patch("financial_portfolio_runtime.harness.launch_chat_program", AsyncMock(return_value=ProgramResult(0, "", ""))):
                with self.assertRaises(asyncio.CancelledError):
                    await harness.launch(SimpleNamespace(model="fixture-model"), trace,
                                         SimpleNamespace(alive=AsyncMock(side_effect=asyncio.CancelledError)),
                                         "endpoint", "secret", {}, TaskData(id="fixture", prompt="Task"))
        health = trace.info[INFO_KEY]["events"][-1]
        self.assertEqual(health["event"], "runtime_health")
        self.assertIsNone(health["alive"])
        self.assertTrue(health["cancelled"])

    async def test_conflicting_guidance_and_resumed_transcripts_rejected(self):
        harness = FinancialPortfolioRuntimeHarness(FinancialPortfolioRuntimeHarnessConfig(guidance="One"))
        for data in (TaskData(id="fixture", prompt="Task", system_prompt=GUIDANCE_MARKER + "Other"),
                     TaskData(id="fixture", prompt=[vf.UserMessage(content="Task")])):
            with self.assertRaises(ValueError):
                harness.resolve_prompt(data)
        self.assertFalse(harness.SUPPORTS_RESUME)


if __name__ == "__main__":
    unittest.main()
