"""Offline regressions against the actual bundled chat program and a fake endpoint."""

import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None

if vf is not None:
    from financial_portfolio_staged import _chat_core
    from financial_portfolio_staged.harness import (
        BUNDLED_DEPENDENCIES,
        FinancialPortfolioStagedHarness,
        FinancialPortfolioStagedHarnessConfig,
        chat_program_source,
    )


class FakeEndpoint:
    """Only test-authored commands are ever returned; no live provider is contacted."""

    def __init__(self, mode="tool", gate="allow", arguments=None, tool_name="bash"):
        self.mode, self.gate, self.tool_name = mode, gate, tool_name
        self.arguments = arguments or json.dumps({"command": "printf fixture > fixture.txt"})
        self.calls, self.gates = [], []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tool":
                    fixture.gates.append(body)
                    decision = {"action": fixture.gate}
                    if fixture.gate == "deny":
                        decision["message"] = {
                            "role": "tool", "tool_call_id": body["tool_call_id"],
                            "name": body["name"], "content": "checkpoint accepted",
                        }
                    elif fixture.gate == "stop":
                        decision["reason"] = "malformed_tool_arguments"
                    self.respond(json.dumps(decision).encode(), "application/json")
                    return
                fixture.calls.append({"path": self.path, "body": body,
                                      "retry_count": self.headers.get("x-stainless-retry-count")})
                if fixture.mode in (429, 500):
                    payload = {"error": {"message": "fixture failure", "type": "fixture_error"}}
                    self.respond(json.dumps(payload).encode(), "application/json", fixture.mode)
                    return
                if fixture.mode == "truncated":
                    chunk = self.chunk({"role": "assistant", "content": "unfinished"}, None)
                    self.respond(self.event(chunk), "text/event-stream")
                    return
                if fixture.mode == "tool" and len(fixture.calls) == 1:
                    delta = {"role": "assistant", "tool_calls": [{
                        "index": 0, "id": "fixture_call", "type": "function", "function": {
                            "name": fixture.tool_name, "arguments": fixture.arguments,
                        },
                    }]}
                    finish = "tool_calls"
                else:
                    delta, finish = {"role": "assistant", "content": "done"}, "stop"
                chunk = self.chunk(delta, finish)
                self.respond(self.event(chunk) + b"data: [DONE]\n\n", "text/event-stream")

            @staticmethod
            def chunk(delta, finish):
                return {"id": "fixture_response", "object": "chat.completion.chunk",
                        "created": 0, "model": "fixture-model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            @staticmethod
            def event(chunk):
                return b"data: " + json.dumps(chunk).encode() + b"\n\n"

            def respond(self, content, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class StagedHarnessConfig(unittest.TestCase):
    def test_restricted_config_rejects_unsupported_settings(self):
        config = FinancialPortfolioStagedHarnessConfig()
        self.assertEqual(config.sdk_max_retries, 0)
        self.assertEqual(config.resolved_env, {})
        for options in (
            {"sdk_max_retries": 1}, {"sdk_max_retries": False}, {"sdk_max_retries": 0.0},
            {"dialect": "responses"}, {"search": True}, {"compaction": {}},
            {"forward_env": ["MODEL_SECRET"]}, {"env": {"MODEL_SECRET": "x"}},
            {"mcp_header_env": {"server": {"Authorization": "MODEL_SECRET"}}},
            {"skills": ["/tmp/fixture"]}, {"disabled_tools": ["bash"]},
            {"unsupported_setting": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FinancialPortfolioStagedHarnessConfig(**options)

    def test_vendored_core_has_exactly_one_change_and_bundle_keeps_license(self):
        from verifiers.v1.harnesses.utils import core

        original, modified = inspect.getsource(core), inspect.getsource(_chat_core)
        insertion = "        max_retries=0,\n"
        self.assertEqual(modified.count(insertion), 1)
        self.assertEqual(modified.replace(insertion, ""), original)
        source = chat_program_source()
        self.assertIn("Copyright (c) 2026 Prime Intellect", source)
        self.assertIn("Permission is hereby granted", source)
        self.assertIn("max_retries=0", source)
        metadata = source.split("# ///\n", 1)[0]
        for name, release in BUNDLED_DEPENDENCIES.items():
            self.assertIn(f'"{name}=={release}"', metadata)
        compile(source, "bundled_staged_harness.py", "exec")

    def test_native_safety_and_lifecycle_capabilities(self):
        from verifiers.v1.harness import Harness

        self.assertTrue(FinancialPortfolioStagedHarness.NEEDS_CONTAINER)
        self.assertTrue(FinancialPortfolioStagedHarness.SUPPORTS_TOOL_INTERCEPTION)
        self.assertTrue(FinancialPortfolioStagedHarness.SUPPORTS_RESUME)
        self.assertFalse(FinancialPortfolioStagedHarness.SUPPORTS_MCP)
        self.assertIs(FinancialPortfolioStagedHarness.session, Harness.session)
        self.assertIs(FinancialPortfolioStagedHarness.resume, Harness.resume)
        self.assertIs(FinancialPortfolioStagedHarness._check_result, Harness._check_result)


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class StagedHarnessBundle(unittest.TestCase):
    def run_program(self, endpoint, root, *, edit=False):
        script = root / "chat.py"
        script.write_text(chat_program_source())
        args = [sys.executable, str(script), f"--base-url={endpoint.url}/v1",
                "--api-key=fixture-token", "--model=fixture-model", "--bash",
                "--prompt=Run only the fixture command.",
                f"--tool-interception-url={endpoint.url}/tool"]
        if edit:
            args.append("--edit")
        return subprocess.run(args, cwd=root, env={"PATH": os.defpath},
                              capture_output=True, text=True, timeout=30, check=False)

    def test_http_and_stream_failures_make_one_request(self):
        for mode in (429, 500, "truncated"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, FakeEndpoint(mode) as endpoint:
                result = self.run_program(endpoint, Path(directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(endpoint.calls), 1, result.stderr)
                self.assertEqual(endpoint.gates, [])
                self.assertEqual(endpoint.calls[0]["path"], "/v1/chat/completions")
                self.assertIn(endpoint.calls[0]["retry_count"], (None, "0"))
                if mode == "truncated":
                    self.assertEqual(result.returncode, 97)
                    self.assertIn("Model stream ended before a completion finished", result.stderr)

    def test_allowed_fixture_tool_runs_and_result_returns_to_model(self):
        with tempfile.TemporaryDirectory() as directory, FakeEndpoint() as endpoint:
            root = Path(directory)
            result = self.run_program(endpoint, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "fixture.txt").read_text(), "fixture")
            self.assertEqual(len(endpoint.calls), 2)
            self.assertEqual(endpoint.gates, [{"tool_call_id": "fixture_call", "name": "bash",
                                              "arguments": {"command": "printf fixture > fixture.txt"}}])
            messages = endpoint.calls[1]["body"]["messages"]
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "fixture_call")
            self.assertEqual(messages[-2]["tool_calls"][0]["function"]["arguments"], endpoint.arguments)

    def test_denied_gate_result_replaces_execution_in_next_request(self):
        with tempfile.TemporaryDirectory() as directory, FakeEndpoint(gate="deny") as endpoint:
            root = Path(directory)
            result = self.run_program(endpoint, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "fixture.txt").exists())
            self.assertEqual(len(endpoint.calls), 2)
            self.assertEqual(endpoint.calls[1]["body"]["messages"][-1]["content"], "checkpoint accepted")

    def test_stop_gate_preserves_malformed_arguments_without_another_model_request(self):
        malformed = '{"command":"printf bad\\ x"}'
        with tempfile.TemporaryDirectory() as directory, FakeEndpoint(gate="stop", arguments=malformed) as endpoint:
            result = self.run_program(endpoint, Path(directory))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(endpoint.calls), 1)
            self.assertEqual(endpoint.gates[0]["arguments"], malformed)
            self.assertIn("malformed_tool_arguments", result.stderr)

    def test_edit_fixture_uses_same_gate_and_executes(self):
        arguments = json.dumps({"path": "fixture.txt", "old_str": "before", "new_str": "after"})
        with tempfile.TemporaryDirectory() as directory, FakeEndpoint(arguments=arguments, tool_name="edit") as endpoint:
            root = Path(directory)
            (root / "fixture.txt").write_text("before")
            result = self.run_program(endpoint, root, edit=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "fixture.txt").read_text(), "after")
            self.assertEqual(endpoint.gates[0]["name"], "edit")


class RecordingRuntime:
    def __init__(self):
        self.prepared, self.commands, self.writes = [], [], {}

    async def prepare_uv_script(self, source, env, **options):
        self.prepared.append((source, env, options))
        return ["fixture-python", "fixture-program.py"]

    async def run(self, command, env):
        self.commands.append((command, env))
        return SimpleNamespace(exit_code=0)

    async def write(self, path, data):
        self.writes[path] = data


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class StagedHarnessLaunch(unittest.IsolatedAsyncioTestCase):
    async def test_launch_preserves_native_endpoint_gate_and_prompt_wiring(self):
        runtime = RecordingRuntime()
        harness = FinancialPortfolioStagedHarness(FinancialPortfolioStagedHarnessConfig())
        await harness.setup(runtime)
        await harness.launch(SimpleNamespace(model="fixture-model"), SimpleNamespace(id="trace-fixture"),
                             runtime, "http://127.0.0.1:1/v1", "fixture-token", {},
                             vf.TaskData(prompt="task question", system_prompt="workflow instruction"),
                             tool_interception_url="http://127.0.0.1:1/tool")
        self.assertEqual(len(runtime.prepared), 2)
        self.assertEqual(runtime.prepared[0][0], runtime.prepared[1][0])
        self.assertEqual(runtime.prepared[1][2], {"activate": False})
        command, env = runtime.commands[0]
        self.assertEqual(env, {})
        self.assertIn("--base-url=http://127.0.0.1:1/v1", command)
        self.assertIn("--tool-interception-url=http://127.0.0.1:1/tool", command)
        self.assertIn("--prompt=task question", command)
        self.assertTrue(any(arg.endswith("\n\nworkflow instruction") for arg in command))
        for disabled in ("--search", "--compaction", "--mcp-config="):
            self.assertFalse(any(arg.startswith(disabled) for arg in command))

    async def test_mcp_override_rejected_before_program_preparation(self):
        runtime = RecordingRuntime()
        harness = FinancialPortfolioStagedHarness(FinancialPortfolioStagedHarnessConfig())
        with self.assertRaises(ValueError):
            await harness.launch(None, None, runtime, "endpoint", "token",
                                 {"unexpected": "http://127.0.0.1:1"}, vf.TaskData())
        self.assertEqual(runtime.prepared, [])


@unittest.skipUnless(vf is not None and os.environ.get("RUN_STAGED_DOCKER_TESTS") == "1",
                     "Set RUN_STAGED_DOCKER_TESTS=1 for native bundle installation")
class StagedHarnessDocker(unittest.IsolatedAsyncioTestCase):
    async def test_pinned_runtime_bundle_and_offline_fixtures(self):
        from verifiers.v1.runtimes import make_backend

        fixture = (
            "import json, os, subprocess, sys, tempfile, threading, unittest\n"
            "from pathlib import Path\n"
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "vf = True\n"
            "def chat_program_source():\n"
            "    return Path('/workspace/bundle.py').read_text()\n\n"
            + inspect.getsource(FakeEndpoint) + "\n"
            + inspect.getsource(StagedHarnessBundle)
            + '\nif __name__ == "__main__": unittest.main(verbosity=2)\n'
        )
        backend = make_backend(vf.DockerConfig(image="python:3.12-slim", workdir="/workspace", allow=[]))
        try:
            async with asyncio.timeout(180):
                await backend.start()
                runtime = backend.bind()
                await backend.prepare_setup()
                source = chat_program_source()
                argv = await runtime.prepare_uv_script(source, {}, activate=False)
                help_result = await runtime.run([*argv, "--help"], {})
                self.assertEqual(help_result.exit_code, 0, help_result.stderr)
                packages = list(BUNDLED_DEPENDENCIES)
                version_result = await runtime.run([argv[0], "-c",
                    "import importlib.metadata as m, json; "
                    f"print(json.dumps({{name:m.version(name) for name in {packages!r}}}))"], {})
                self.assertEqual(version_result.exit_code, 0, version_result.stderr)
                self.assertEqual(json.loads(version_result.stdout), BUNDLED_DEPENDENCIES)
                await runtime.write("/workspace/bundle.py", source.encode())
                await runtime.write("/workspace/fixture_test.py", fixture.encode())
                result = await runtime.run([argv[0], "/workspace/fixture_test.py"], {})
                self.assertEqual(result.exit_code, 0, result.stderr)
                self.assertIn("Ran 5 tests", result.stderr)
        finally:
            await backend.stop()


if __name__ == "__main__":
    unittest.main()
