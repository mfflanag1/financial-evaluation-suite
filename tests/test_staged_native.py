"""Opt-in full native Docker rollouts against a local scripted provider."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_staged_harness import FakeEndpoint, vf


@unittest.skipUnless(vf is not None and os.environ.get("RUN_STAGED_DOCKER_TESTS") == "1",
                     "Enable RUN_STAGED_DOCKER_TESTS=1 with Docker and pinned packages installed")
class StagedNativeRollouts(unittest.TestCase):
    def run_fixture(self, arguments, root):
        from financial_research.accounting import report_path

        with FakeEndpoint(arguments=arguments) as endpoint:
            config = {
                "model": "fixture-model", "num_rollouts": 1, "max_concurrent": 1,
                "output_dir": str(root), "run": {"name": "native-fixture", "dir": "run"},
                "client": {"type": "eval", "base_url": endpoint.url + "/v1",
                           "api_key_var": "STAGED_FIXTURE_TOKEN"},
                "sampling": {"max_tokens": 512, "temperature": 0.2},
                "select": {"limit": 1, "shuffle": False},
                "env": {
                    "taskset": {"id": "financial-portfolio-staged", "num_tasks": 1},
                    "retries": {"max_retries": 0},
                    "agent": {
                        "max_turns": 4, "retries": {"max_retries": 0},
                        "harness": {"id": "financial-portfolio-staged", "sdk_max_retries": 0},
                        "runtime": {"type": "docker", "image": "python:3.12-slim", "allow": []},
                        "scoring": {"mode": "isolated", "retries": {"max_retries": 0}},
                    },
                },
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            env = {**os.environ, "STAGED_FIXTURE_TOKEN": "fixture-token",
                   "FINANCIAL_RESEARCH_KEY": "staged-native-fixture-key-not-for-evaluation-0000",
                   "PRIME_DISABLE_VERSION_CHECK": "1"}
            result = subprocess.run([str(Path(sys.executable).with_name("vf-run")), "@", str(path)],
                                    env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            episodes = [json.loads(line) for line in (root / "run/traces.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            trace = episode["traces"][0]
            self.assertEqual(episode["errors"], [])
            self.assertEqual(trace["errors"], [])
            report = report_path(root / "run", expected=1)
            self.assertEqual(report["counts"]["candidate_failure"], 1)
            self.assertTrue(report["performance_usable"])
            self.assertEqual(report["mean_reward"], 0)
            return trace, endpoint.calls

    def test_malformed_tool_is_one_call_candidate_failure_in_native_pipeline(self):
        arguments = '{"command":"invalid\\ escape"}'
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(arguments, Path(directory))
            self.assertEqual(len(calls), 1)
            self.assertEqual(trace["stop_condition"], "candidate_protocol_failure")
            emitted = [node["message"] for node in trace["nodes"] if node["message"]["role"] == "assistant"]
            self.assertEqual(emitted[-1]["tool_calls"][0]["arguments"], arguments)
            self.assertEqual(trace["info"]["financial_research_outcome"]["reason"], "malformed_tool_arguments")

    def test_oversized_file_is_candidate_failure_in_native_pipeline(self):
        # Fixed fixture code executes only inside the isolated solver container.
        arguments = json.dumps({"command": "python -c \"open('/workspace/policy.json','wb').write(b' ' * 3000000)\""})
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(arguments, Path(directory))
            self.assertEqual(len(calls), 2)
            self.assertEqual(trace["info"]["financial_research_outcome"]["reason"], "artifact_transfer_limit")
            violations = trace["info"]["portfolio_workflow"]["artifact_collection_violations"]
            self.assertEqual([item["path"] for item in violations], ["/workspace/policy.json"])
            self.assertFalse(violations[0]["payload_saved"])

    def test_virtual_checkpoint_reply_survives_native_tool_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(json.dumps({"command": "financial-checkpoint ingestion"}), Path(directory))
            self.assertEqual(len(calls), 2)
            replies = [message for message in calls[-1]["body"]["messages"] if message["role"] == "tool"]
            feedback = json.loads(replies[-1]["content"])
            self.assertEqual(feedback["checkpoint"], "ingestion")
            self.assertFalse(feedback["valid"])
            self.assertEqual(feedback["next_stage"], "ingestion")
            self.assertEqual(trace["info"]["financial_research_outcome"]["reason"], "workflow_incomplete")


if __name__ == "__main__":
    unittest.main()
