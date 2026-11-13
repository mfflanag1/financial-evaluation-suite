"""Unanswered execution must not become an invented zero financial score."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio_operations", "financial_portfolio_runtime"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))
from financial_portfolio_runtime.accounting import classify, report_path


def trace():
    return {"version": 1, "id": "trace", "task": {}, "errors": [], "is_completed": True,
            "ok": True, "is_timeout": False, "dropped": None, "stop_condition": "candidate_workflow_failure",
            "rewards": {"financial": {"score": 0, "weight": 1}},
            "info": {"financial_research_outcome": {"status": "candidate_failure", "reason": "workflow_incomplete",
                     "phase": "workflow", "reproduction_verified": False},
                     "portfolio_runtime": {"events": [{"event": "harness_terminal", "outcome": "returned", "exit_code": 0},
                                                       {"event": "runtime_health", "alive": True}], "tools": {}}}}


class RuntimeAccounting(unittest.TestCase):
    def test_intentional_empty_completion_remains_candidate_failure(self):
        result = classify(trace())
        self.assertEqual((result["status"], result["reward"]), ("candidate_failure", 0))

    def test_unanswered_tool_is_unknown_even_with_native_candidate_zero(self):
        data = trace()
        data["info"]["portfolio_runtime"]["tools"]["call"] = {"status": "allowed"}
        result = classify(data)
        self.assertEqual((result["status"], result["reason"], result["reward"]),
                         ("unknown", "tool_execution_unresolved", None))

    def test_cancel_dead_and_timeout_preserve_nonfinancial_status(self):
        data = trace()
        data["info"]["portfolio_runtime"]["events"][0]["outcome"] = "cancelled"
        self.assertEqual(classify(data)["reason"], "harness_execution_interrupted")
        data["info"]["portfolio_runtime"]["events"][-1]["alive"] = False
        self.assertEqual(classify(data)["status"], "infrastructure_error")
        data = trace()
        data["is_timeout"] = True
        self.assertEqual(classify(data)["reward"], None)

    def test_malformed_tool_stop_is_explicit_candidate_failure(self):
        data = trace()
        data["stop_condition"] = "candidate_protocol_failure"
        data["info"]["financial_research_outcome"]["reason"] = "malformed_tool_arguments"
        data["info"]["portfolio_runtime"]["events"][0]["exit_code"] = 1
        self.assertEqual(classify(data)["status"], "candidate_failure")

    def test_missing_terminal_is_unknown_and_blocks_mean(self):
        data = trace()
        data["info"]["portfolio_runtime"]["events"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text(json.dumps(data) + "\n")
            report = report_path(path, expected=1)
            self.assertFalse(report["performance_usable"])
            self.assertIsNone(report["mean_reward"])

    def test_missing_or_inconclusive_runtime_health_blocks_mean(self):
        for health in ([], [{"event": "runtime_health", "alive": None}],
                       [{"event": "runtime_health", "alive": None, "cancelled": True}]):
            data = trace()
            data["info"]["portfolio_runtime"]["events"] = data["info"]["portfolio_runtime"]["events"][:1] + health
            result = classify(data)
            self.assertEqual((result["status"], result["reason"], result["reward"]),
                             ("unknown", "runtime_health_unresolved", None))

    def test_duplicate_or_missing_attempts_block_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text((json.dumps(trace()) + "\n") * 2)
            self.assertFalse(report_path(path, expected=2)["performance_usable"])
            path.write_text(json.dumps(trace()) + "\n")
            self.assertEqual(report_path(path, expected=2)["missing_attempts"], 1)

    def test_reused_trace_under_distinct_episode_ids_blocks_mean(self):
        episodes = [{"id": identity, "traces": [trace()], "errors": [], "ok": True}
                    for identity in ("episode-one", "episode-two")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in episodes))
            report = report_path(path, expected=2)
            self.assertTrue(report["duplicate_or_missing_identity"])
            self.assertIsNone(report["mean_reward"])


if __name__ == "__main__":
    unittest.main()
