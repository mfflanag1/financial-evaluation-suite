"""Real native Docker rollouts against a local, deterministic scripted provider."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from test_runtime_release_harness import ROOT, vf


class ScriptedProvider:
    """Only fixture-authored messages are served; no external inference occurs."""
    def __init__(self, responses):
        self.responses, self.calls = responses, []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.calls.append(request)
                index = len(fixture.calls) - 1
                response = fixture.responses[index] if index < len(fixture.responses) else {"content": "done"}
                delta = {"role": "assistant", **response}
                if "tool_calls" in delta:
                    finish = "tool_calls"
                else:
                    finish = "stop"
                chunk = {"id": f"fixture-response-{index}", "object": "chat.completion.chunk", "created": 0,
                         "model": "fixture-model", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                payload = b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


def tool(command=None, *, arguments=None, identity="fixture-tool"):
    return {"tool_calls": [{"index": 0, "id": identity, "type": "function", "function": {
        "name": "bash", "arguments": json.dumps({"command": command}) if arguments is None else arguments}}]}


@unittest.skipUnless(vf is not None and os.environ.get("RUN_RUNTIME_DOCKER_TESTS") == "1",
                     "Enable RUN_RUNTIME_DOCKER_TESTS=1 with Docker and pinned packages installed")
class RuntimeNativeRollouts(unittest.TestCase):
    def fixture(self, responses, *, rollout_timeout=90, guidance=""):
        from financial_portfolio_runtime.accounting import report_path
        from financial_portfolio_operations.telemetry import summarize_journal

        with tempfile.TemporaryDirectory() as directory, ScriptedProvider(responses) as endpoint:
            root = Path(directory)
            image_id = subprocess.check_output(["docker", "image", "inspect", "python:3.12-slim", "--format", "{{.Id}}"], text=True).strip()
            config = {"model": "fixture-model", "num_rollouts": 1, "max_concurrent": 1, "dry_run": False,
                      "output_dir": str(root), "run": {"name": "runtime-fixture", "dir": "run"},
                      "client": {"type": "eval", "base_url": endpoint.url, "api_key_var": "RUNTIME_FIXTURE_TOKEN"},
                      "sampling": {"max_tokens": 32768, "temperature": 0.2},
                      "select": {"limit": 1, "shuffle": False},
                      "env": {"taskset": {"id": "financial-portfolio-runtime", "num_tasks": 1, "seed_start": 4,
                                          "task": {"image": image_id}},
                              "retries": {"max_retries": 0},
                              "agent": {"max_turns": 8, "retries": {"max_retries": 0},
                                        "timeout": {"setup": 180, "rollout": rollout_timeout, "finalize": 30, "scoring": 180},
                                        "harness": {"id": "financial-portfolio-runtime", "guidance": guidance},
                                        "runtime": {"type": "docker", "image": image_id, "allow": []},
                                        "scoring": {"mode": "isolated", "retries": {"max_retries": 0}}}}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            launcher = root / "launch.py"
            launcher.write_text("import sys,json,hashlib\nfrom pathlib import Path\nfrom copy import deepcopy\n"
                                "from financial_portfolio_runtime.telemetry import capture_runtime\n"
                                "from financial_portfolio_runtime.binding import build_run_binding\n"
                                "from financial_portfolio_runtime.prime_adapter import FinancialPortfolioRuntimeTaskset\n"
                                "from verifiers.v1.configs.cli.run import RunnerConfig\n"
                                "from verifiers.v1.cli.run import main\n"
                                "config=json.loads(Path(sys.argv[1]).read_text())\n"
                                "tasks=list(FinancialPortfolioRuntimeTaskset(RunnerConfig.model_validate(deepcopy(config)).env.taskset).load())\n"
                                "with capture_runtime(sys.argv[2]):\n"
                                "    bound=Path(sys.argv[2])/'bound.json'\n"
                                "    bound.write_text(json.dumps(config))\n"
                                "    binding=build_run_binding(config,tasks,config_sha256=hashlib.sha256(bound.read_bytes()).hexdigest())\n"
                                "    (bound.parent/'run_binding.json').write_text(json.dumps(binding))\n"
                                "    main(['@',str(bound)])\n")
            env = {**os.environ, "RUNTIME_FIXTURE_TOKEN": "fixture-token",
                   "FINANCIAL_RESEARCH_KEY": "runtime-native-fixture-key-not-for-evaluation-0000",
                   "PRIME_DISABLE_VERSION_CHECK": "1"}
            # Explicitly bind every package to this checkout, including child
            # native scoring processes; a historical editable install is not evidence.
            env["PYTHONPATH"] = os.pathsep.join(str(p / "src") for p in (ROOT / "environments").iterdir() if (p / "src").is_dir())
            process = subprocess.run([sys.executable, "-B", str(launcher), str(config_path), str(root / "journal")],
                                     capture_output=True, text=True, env=env, timeout=420)
            self.assertEqual(process.returncode, 0, process.stderr[-10000:])
            records = [json.loads(line) for line in (root / "run/traces.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            trace = records[0]["traces"][0]
            report = report_path(root / "run", expected=1, journal_dir=root / "journal")
            lifecycle = [json.loads(line) for line in (root / "journal/lifecycle.jsonl").read_text().splitlines()]
            self.assertTrue(all(e["valid"] for e in lifecycle if e["event"] == "outbound_prompt_checked"))
            request_report = summarize_journal(root / "journal/provider")
            # Optional host-local evidence destination; never publish raw task data.
            evidence_root = os.environ.get("RUNTIME_FIXTURE_EVIDENCE_DIR")
            if evidence_root:
                destination = Path(evidence_root) / self._testMethodName
                destination.mkdir(parents=True, exist_ok=False)
                # Preserve exact journal/config bytes so report_path can be
                # independently rerun after the temporary native run is removed.
                shutil.copytree(root / "journal", destination / "journal")
                shutil.copyfile(root / "run/traces.jsonl", destination / "traces.jsonl")
                for name, value in (("report.json", report), ("episode.json", records[0]),
                                    ("lifecycle.json", lifecycle), ("request_report.json", request_report),
                                    ("run_binding.json", json.loads((root / "journal/run_binding.json").read_text())),
                                    ("bound.json", json.loads((root / "journal/bound.json").read_text()))):
                    (destination / name).write_text(json.dumps(value, indent=2))
            return trace, report, endpoint.calls

    def reference_responses(self, *, corrupt_policy=False):
        from financial_portfolio.generator import build_episode
        from financial_portfolio_operations.references import _evaluator_key
        from financial_portfolio_staged.validation import inventory_for_public_files
        with _evaluator_key("runtime-native-fixture-key-not-for-evaluation-0000"):
            public = build_episode(4, split="train")["public_files"]
        source = base64.b64encode(files("financial_portfolio").joinpath("reference.py").read_bytes()).decode()
        inventory = inventory_for_public_files(public)
        program = ("import base64,json,runpy\n"
                   f"open('/workspace/research.py','wb').write(base64.b64decode({source!r}))\n"
                   f"json.dump({inventory!r},open('/workspace/ingestion.json','w'))\n"
                   "runpy.run_path('/workspace/research.py',run_name='__main__')\n"
                   "policy=json.load(open('/workspace/policy.json'))\n"
                   "json.dump(policy['trials'],open('/workspace/trials.json','w'))\n")
        if corrupt_policy:
            program += ("policy['audit'][0]['value'] += 0.1\n"
                        "json.dump(policy,open('/workspace/policy.json','w'))\n")
        responses = [tool("python - <<'PY'\n" + program + "PY", identity="artifacts")]
        responses.extend(tool("financial-checkpoint " + stage, identity=stage) for stage in ("ingestion", "trials", "submission"))
        return responses

    def test_all_checkpoints_reproduce_and_grade_in_native_isolation(self):
        trace, report, calls = self.fixture(self.reference_responses(), guidance="Read the public contract carefully.")
        self.assertEqual(report["counts"]["passed"], 1, report)
        self.assertTrue(report["performance_usable"], report)
        self.assertTrue(report["entries"][0]["reproduction_verified"])
        self.assertEqual(trace["info"]["portfolio_workflow"]["completed"], ["ingestion", "trials", "submission"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0]["messages"][0]["content"].count("<portfolio-runtime-guidance>"), 1)
        self.assertIn("External network access is disabled.", calls[0]["messages"][0]["content"])

    def test_completed_nonreproducing_submission_is_candidate_failure(self):
        trace, report, calls = self.fixture(self.reference_responses(corrupt_policy=True))
        self.assertEqual(trace["info"]["portfolio_workflow"]["completed"], ["ingestion", "trials", "submission"])
        self.assertEqual(report["counts"]["candidate_failure"], 1, report)
        self.assertFalse(report["entries"][0]["reproduction_verified"])
        self.assertEqual(report["mean_reward"], 0)

    def test_malformed_tool_remains_candidate_failure(self):
        trace, report, calls = self.fixture([tool(arguments='{"command":"invalid\\ escape"}')])
        self.assertEqual(report["counts"]["candidate_failure"], 1, report)
        self.assertEqual(report["entries"][0]["reason"], "malformed_tool_arguments")
        self.assertEqual(len(calls), 1)

    def test_intentional_empty_completion_remains_candidate_failure(self):
        trace, report, calls = self.fixture([{"content": ""}])
        self.assertEqual(report["counts"]["candidate_failure"], 1, report)
        self.assertEqual(len(calls), 1)

    def test_timeout_during_tool_execution_is_not_a_financial_zero(self):
        trace, report, calls = self.fixture([tool("sleep 30")], rollout_timeout=3)
        self.assertEqual(report["counts"]["unknown"], 1, report)
        self.assertIsNone(report["mean_reward"])
        self.assertEqual(report["entries"][0]["unanswered_tool_calls"], 1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
