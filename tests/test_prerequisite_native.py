"""Opt-in native pipeline checks against a scripted local model endpoint."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_staged_harness import FakeEndpoint, vf


@unittest.skipUnless(vf is not None and os.environ.get("RUN_PREREQUISITE_DOCKER_TESTS") == "1",
                     "Enable RUN_PREREQUISITE_DOCKER_TESTS=1 with Docker and pinned packages")
class PrerequisiteNative(unittest.TestCase):
    def run_fixture(self, arguments, root):
        from financial_research.accounting import report_path

        with FakeEndpoint(arguments=arguments) as endpoint:
            config = {
                "model": "fixture-model", "num_rollouts": 1, "max_concurrent": 1,
                "output_dir": str(root), "run": {"name": "prerequisite-fixture", "dir": "run"},
                "client": {"type": "eval", "base_url": endpoint.url + "/v1",
                           "api_key_var": "PREREQUISITE_FIXTURE_TOKEN"},
                "sampling": {"max_tokens": 512, "temperature": 0.2},
                "select": {"limit": 1, "shuffle": False},
                "env": {"taskset": {"id": "financial-prerequisites"},
                        "retries": {"max_retries": 0},
                        "agent": {"max_turns": 4, "retries": {"max_retries": 0},
                                  "harness": {"id": "financial-portfolio-staged", "sdk_max_retries": 0},
                                  "runtime": {"type": "docker", "image": "python:3.12-slim", "allow": []},
                                  "scoring": {"mode": "isolated", "retries": {"max_retries": 0}}}},
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            env = {**os.environ, "PREREQUISITE_FIXTURE_TOKEN": "fixture-token",
                   "PRIME_DISABLE_VERSION_CHECK": "1"}
            env.pop("FINANCIAL_RESEARCH_KEY", None)
            result = subprocess.run([str(Path(sys.executable).with_name("vf-run")), "@", str(path)],
                                    env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            records = [json.loads(line) for line in (root / "run/traces.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["errors"], [])
            trace = record["traces"][0]
            self.assertEqual(trace["errors"], [])
            report = report_path(root / "run", expected=1)
            self.assertTrue(report["performance_usable"], report)
            self.assertEqual(report["counts"]["candidate_failure"], 1)
            self.assertEqual(report["mean_reward"], 0)
            return trace, endpoint.calls

    def test_malformed_tool_stops_after_one_request_and_remains_candidate_zero(self):
        arguments = '{"command":"invalid\\ escape"}'
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(arguments, Path(directory))
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace["stop_condition"], "candidate_protocol_failure")
        self.assertEqual(trace["info"]["financial_research_outcome"]["reason"], "malformed_tool_arguments")
        self.assertEqual([node["message"] for node in trace["nodes"]
                          if node["message"]["role"] == "assistant"][-1]["tool_calls"][0]["arguments"], arguments)

    def test_oversized_output_remains_candidate_zero(self):
        arguments = json.dumps({"command": "python -c \"open('policy.json','wb').write(b' ' * 3000000)\""})
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(arguments, Path(directory))
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace["info"]["financial_research_outcome"]["reason"], "artifact_transfer_limit")
        violations = trace["info"]["prerequisite_check"]["artifact_collection_violations"]
        self.assertEqual([item["path"] for item in violations], ["/workspace/policy.json"])
        self.assertFalse(violations[0]["payload_saved"])

    def test_hardcoded_output_fails_alternate_after_full_native_rollout(self):
        from financial_prerequisites.cases import load_cases

        expected = load_cases()[0]["expected_json"]
        source = "import json\nwith open('policy.json','w') as f:\n    json.dump(" + repr(expected) + ",f)\n"
        encoded = base64.b64encode(source.encode()).decode()
        # This fixed fixture source executes only inside isolated native runtimes.
        command = ("python - <<'PY'\nimport base64\n"
                   f"open('research.py','wb').write(base64.b64decode('{encoded}'))\n"
                   "PY\npython research.py")
        with tempfile.TemporaryDirectory() as directory:
            trace, calls = self.run_fixture(json.dumps({"command": command}), Path(directory))
        self.assertEqual(len(calls), 2)
        info = trace["info"]
        self.assertEqual(info["prerequisite_check"]["original_reproduction"], "passed")
        self.assertEqual(info["prerequisite_check"]["alternate_reproduction"], "output_mismatch")
        self.assertEqual(info["financial_research_outcome"]["phase"], "alternate_reproduction")
        self.assertFalse(info["financial_research_outcome"]["reproduction_verified"])


if __name__ == "__main__":
    unittest.main()
