"""Saved v1 Episode/Trace accounting without a Verifiers dependency."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "financial_research" / "src"))

from financial_research.accounting import OUTCOME_KEY, report_path


def trace(index=0, status="passed", reward=1.0):
    # Shape verified against Trace.model_dump_json(exclude_none=True) from
    # verifiers==0.4.1. Reward is NOT a serialized top-level scalar property.
    return {
        "version": 1,
        "id": f"trace-{index}",
        "verifiers": {"version": "0.4.1"},
        "task": {"type": "ResearchTask", "data": {"id": f"task-{index}", "prompt": "Fixture"}},
        "agent": {"config": {}, "name": "agent", "trainable": True},
        "nodes": [], "calls": [], "metrics": {}, "errors": [],
        "rewards": {"research_reward": {"score": reward, "weight": 1.0}},
        "info": {OUTCOME_KEY: {"status": status, "reason": "fixture_outcome", "phase": "scoring",
                                "reproduction_verified": status == "passed"}},
        "is_completed": True, "ok": True, "is_timeout": False,
    }


def episode(index=0, status="passed", reward=1.0):
    solver = trace(index, status, reward)
    return {"id": f"episode-{index}", "env": {"id": "financial-research"},
            "task": copy.deepcopy(solver["task"]), "ok": True, "is_timeout": False,
            "errors": [], "traces": [solver]}


class ResearchAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="research-accounting-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_jsonl(self, records, name="traces.jsonl"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
        return path

    def manifest(self, count):
        (self.root / "attempt_manifest.json").write_text(json.dumps({"expected_attempts": count}))

    def test_native_episode_rewards_include_candidate_failures_in_denominator(self):
        self.write_jsonl([episode(0), episode(1, reward=0.5), episode(2, "candidate_failure", 0)])
        self.manifest(3)
        report = report_path(self.root)
        self.assertEqual(report["denominator"], 3)
        self.assertEqual(report["counts"]["passed"], 2)
        self.assertEqual(report["counts"]["candidate_failure"], 1)
        self.assertEqual(report["scored_attempts"], 3)
        self.assertEqual(report["full_credit_attempts"], 1)
        self.assertEqual(report["reproduction_verified_attempts"], 2)
        self.assertTrue(report["performance_usable"])
        self.assertTrue(report["comparison_ready"])
        self.assertEqual(report["mean_reward"], 0.5)
        self.assertEqual(report["confirmed_full_credit_lower_bound"], 1 / 3)

    def test_direct_trace_json_and_weighted_reward_shape(self):
        data = trace()
        data["rewards"] = {"research_reward": {"score": 0.8, "weight": 0.5}}
        path = self.root / "trace.json"
        path.write_text(json.dumps(data))
        report = report_path(path, expected=1)
        self.assertTrue(report["performance_usable"])
        self.assertEqual(report["mean_reward"], 0.4)
        self.assertEqual(report["full_credit_attempts"], 0)

    def test_provider_error_is_infrastructure_and_not_candidate_zero(self):
        failed = episode(1)
        failed["ok"] = False
        failed["traces"][0].update(ok=False, errors=[{"type": "ProviderError", "message": "Unavailable"}],
                                   rewards={"research_reward": None}, info={})
        self.write_jsonl([episode(0), failed])
        report = report_path(self.root, expected=2)
        self.assertEqual(report["counts"]["infrastructure_error"], 1)
        self.assertEqual(report["counts"]["candidate_failure"], 0)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["confirmed_full_credit_lower_bound"], 0.5)
        self.assertIsNone(report["mean_reward"])
        self.assertEqual(report["mean_reward_scored_only"], 1)
        self.assertFalse(report["performance_usable"])

    def test_diagnostics_fallback_survives_native_scoring_exception(self):
        failed = episode(0, "infrastructure_error", 0)
        solver = failed["traces"][0]
        outcome = solver["info"].pop(OUTCOME_KEY)
        solver["info"]["scoring_diagnostics"] = {OUTCOME_KEY: outcome}
        solver.update(ok=False, errors=[{"type": "TaskError", "message": "Hook failed"}], rewards={})
        failed["ok"] = False
        self.write_jsonl([failed])
        report = report_path(self.root, expected=1)
        self.assertEqual(report["counts"]["infrastructure_error"], 1)
        self.assertEqual(report["attempts"][0]["reason"], "fixture_outcome")
        self.assertIsNone(report["attempts"][0]["reward"])

    def test_unclassified_errors_timeouts_and_dropped_entries_are_retained(self):
        task_error = episode(0)
        task_error["traces"][0]["errors"] = [{"type": "TaskError", "message": "Unexpected"}]
        timeout = episode(1)
        timeout["traces"][0]["is_timeout"] = True
        dropped = episode(2)
        dropped["traces"][0]["dropped"] = "consumer_policy"
        self.write_jsonl([task_error, timeout, dropped])
        report = report_path(self.root, expected=3)
        self.assertEqual(report["counts"]["unknown"], 3)
        self.assertEqual(report["dropped_trace_entries"], 1)
        self.assertEqual(report["denominator"], 3)
        self.assertEqual(report["scored_attempts"], 0)
        self.assertIsNone(report["mean_reward"])

    def test_missing_planned_attempts_and_conflicting_count_do_not_shrink_denominator(self):
        self.write_jsonl([episode(0), episode(1)])
        self.manifest(5)
        report = report_path(self.root)
        self.assertEqual(report["counts"]["missing"], 3)
        self.assertEqual(report["denominator"], 5)
        self.assertEqual(report["recorded_coverage"], 0.4)
        self.assertEqual(report["resolved_coverage"], 0.4)
        self.assertIsNone(report["mean_reward"])
        conflict = report_path(self.root, expected=2)
        self.assertEqual(conflict["denominator"], 5)
        self.assertIn("conflicting_expected_counts", [issue["code"] for issue in conflict["issues"]])

    def test_unknown_planned_count_prevents_overall_performance_claim(self):
        self.write_jsonl([episode()])
        report = report_path(self.root)
        self.assertIsNone(report["expected_attempts"])
        self.assertFalse(report["performance_usable"])
        self.assertIsNone(report["mean_reward"])
        self.assertEqual(report["mean_reward_scored_only"], 1)
        self.assertIn("expected_attempts_unknown", [issue["code"] for issue in report["issues"]])

    def test_bad_json_lines_duplicate_keys_and_bad_utf8_remain_counted(self):
        path = self.write_jsonl([episode(0)])
        with path.open("ab") as handle:
            handle.write(b'{"traces":\n')
            handle.write(b'{"id":"x","id":"y"}\n')
            handle.write(b'\xff\n')
            handle.write(b'NaN\n')
            handle.write(b'[]\n\n')
        report = report_path(path, expected=6)
        self.assertEqual(report["recorded_entries"], 6)
        self.assertEqual(report["counts"]["malformed"], 5)
        self.assertEqual(report["counts"]["passed"], 1)
        self.assertEqual(report["denominator"], 6)
        self.assertFalse(report["performance_usable"])

    def test_candidate_failure_must_have_explicit_recorded_zero(self):
        missing = episode(0, "candidate_failure", 0)
        missing["traces"][0]["rewards"] = {}
        positive = episode(1, "candidate_failure", 0.5)
        boolean = episode(2, "candidate_failure", False)
        self.write_jsonl([missing, positive, boolean])
        report = report_path(self.root, expected=3)
        self.assertEqual(report["counts"]["unknown"], 3)
        self.assertEqual(report["counts"]["candidate_failure"], 0)
        self.assertEqual(report["scored_attempts"], 0)

    def test_passed_requires_reproduction_and_unambiguous_metadata(self):
        unreproduced = episode(0)
        unreproduced["traces"][0]["info"][OUTCOME_KEY]["reproduction_verified"] = False
        conflict = episode(1)
        other = copy.deepcopy(conflict["traces"][0]["info"][OUTCOME_KEY])
        other["status"] = "candidate_failure"
        conflict["traces"][0]["info"]["scoring_diagnostics"] = {OUTCOME_KEY: other}
        missing = episode(2)
        missing["traces"][0]["info"] = {}
        incomplete = episode(3)
        incomplete["traces"][0]["is_completed"] = False
        self.write_jsonl([unreproduced, conflict, missing, incomplete])
        report = report_path(self.root, expected=4)
        self.assertEqual(report["counts"]["unknown"], 4)
        self.assertIsNone(report["mean_reward_scored_only"])

    def test_episode_errors_and_multiple_traces_do_not_drop_the_episode(self):
        failed = episode(0)
        failed.update(traces=[], ok=False, errors=[{"type": "EnvError", "message": "No runtime"}])
        multiple = episode(1)
        multiple["traces"].append(trace(2))
        self.write_jsonl([failed, multiple])
        report = report_path(self.root, expected=2)
        self.assertEqual(report["counts"]["infrastructure_error"], 1)
        self.assertEqual(report["counts"]["unknown"], 1)
        self.assertEqual(report["recorded_trace_count"], 2)
        self.assertEqual(report["denominator"], 2)

    def test_actual_native_validation_shape_is_never_model_performance(self):
        # Mirrors persisted vf-validate records: valid:true means an environment
        # check passed, including a noop that correctly earned reward zero.
        record = {"elapsed": 1.0, "error": None, "error_type": None, "index": 0,
                  "mode": "all", "name": None, "reason": "valid", "task_key": "fixture",
                  "task_position": 0, "valid": True,
                  "gold": {"mode": "gold", "reason": "valid", "valid": True},
                  "noop": {"mode": "noop", "reason": "valid", "valid": True,
                           "reward": 0.0, "rewards": {"research_reward": 0.0}}}
        self.write_jsonl([record], "results.jsonl")
        report = report_path(self.root, expected=1)
        self.assertEqual(report["counts"]["validation_only"], 1)
        self.assertEqual(report["validation_counts"], {"valid": 1})
        self.assertEqual(report["scored_attempts"], 0)
        self.assertIsNone(report["mean_reward"])
        self.assertFalse(report["performance_usable"])

    def test_directory_scanning_excludes_candidate_artifacts_and_configs(self):
        self.write_jsonl([episode(0)], "run/traces.jsonl")
        self.write_jsonl([episode(1)], "run/configs/misleading.jsonl")
        self.write_jsonl([episode(2)], "run/artifacts/trace/traces.jsonl")
        (self.root / "summary.json").write_text('{"total":999}')
        report = report_path(self.root, expected=1)
        self.assertTrue(report["performance_usable"])
        self.assertEqual(report["recorded_entries"], 1)

    def test_candidate_artifact_cannot_supply_a_trusted_attempt_manifest(self):
        self.write_jsonl([episode(0)], "run/traces.jsonl")
        for directory in ("artifacts/trace", "configs", "logs"):
            target = self.root / "run" / directory / "attempt_manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"expected_attempts":1}')
        report = report_path(self.root)
        self.assertIsNone(report["expected_attempts"])
        self.assertFalse(report["comparison_ready"])
        self.assertIsNone(report["mean_reward"])

    def test_duplicate_episode_ids_are_visible_and_never_double_credited(self):
        self.write_jsonl([episode(0)], "first/traces.jsonl")
        self.write_jsonl([episode(0)], "copy/traces.jsonl")
        report = report_path(self.root, expected=2)
        self.assertEqual(report["recorded_entries"], 2)
        self.assertEqual(report["counts"]["unknown"], 2)
        self.assertEqual(report["full_credit_attempts"], 0)
        self.assertEqual(report["denominator"], 2)
        self.assertFalse(report["performance_usable"])

    def test_episode_and_standalone_trace_copy_are_not_independent_attempts(self):
        self.write_jsonl([episode(0)])
        (self.root / "trace.json").write_text(json.dumps(trace(0)))
        report = report_path(self.root, expected=2)
        self.assertEqual(report["recorded_entries"], 2)
        self.assertEqual(report["counts"]["unknown"], 2)
        self.assertEqual(report["reproduction_verified_attempts"], 0)
        self.assertFalse(report["comparison_ready"])

    def test_empty_run_manifest_and_invalid_manifest_fail_closed(self):
        self.write_jsonl([])
        self.manifest(3)
        report = report_path(self.root)
        self.assertEqual(report["counts"]["missing"], 3)
        self.assertEqual(report["denominator"], 3)
        self.assertIsNone(report["mean_reward"])
        (self.root / "attempt_manifest.json").write_text('{"expected_attempts":true}')
        self.write_jsonl([episode()])
        report = report_path(self.root, expected=1)
        self.assertIn("invalid_attempt_manifest", [issue["code"] for issue in report["issues"]])
        self.assertFalse(report["performance_usable"])

    def test_more_results_than_expected_and_invalid_expected_are_rejected(self):
        self.write_jsonl([episode(0), episode(1)])
        report = report_path(self.root, expected=1)
        self.assertEqual(report["denominator"], 2)
        self.assertFalse(report["performance_usable"])
        for value in (-1, True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                report_path(self.root, expected=value)

    def test_reward_overflow_nonfinite_missing_and_wrong_types_fail_closed(self):
        records = []
        for index, value in enumerate((10 ** 1000, None, "1", True, -0.1, 1.1)):
            records.append(episode(index, reward=value))
        path = self.write_jsonl(records)
        with path.open("a") as handle:
            handle.write(json.dumps(episode(99)).replace('"score": 1.0', '"score": 1e999') + "\n")
        report = report_path(path, expected=7)
        self.assertEqual(report["counts"]["unknown"], 7)
        self.assertEqual(report["scored_attempts"], 0)
        self.assertIsNone(report["mean_reward"])

    def test_unreadable_file_is_an_unresolved_entry_and_missing_path_errors(self):
        path = self.write_jsonl([episode()])
        with patch.object(Path, "open", side_effect=PermissionError("fixture")):
            report = report_path(path, expected=1)
        self.assertEqual(report["counts"]["unknown"], 1)
        self.assertEqual(report["attempts"][0]["reason"], "result_file_unreadable")
        with self.assertRaises(FileNotFoundError):
            report_path(self.root / "absent.jsonl")

    def test_json_array_preserves_malformed_members(self):
        path = self.root / "traces.json"
        path.write_text(json.dumps([episode(), None, {"unknown": True}]))
        report = report_path(path, expected=3)
        self.assertEqual(report["counts"]["passed"], 1)
        self.assertEqual(report["counts"]["malformed"], 2)
        self.assertEqual(report["denominator"], 3)

    def run_cli(self, expected):
        return subprocess.run(
            [sys.executable, "-m", "financial_research.cli", "report", str(self.root),
             "--expected", str(expected)],
            env={"PATH": os.defpath,
                 "PYTHONPATH": str(ROOT / "environments" / "financial_research" / "src")},
            capture_output=True, text=True, timeout=10,
        )

    def test_report_cli_returns_zero_for_resolved_pass_and_candidate_failure(self):
        self.write_jsonl([episode(0), episode(1, "candidate_failure", 0)])
        result = self.run_cli(2)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["counts"]["candidate_failure"], 1)
        self.assertEqual(report["mean_reward"], 0.5)
        self.assertTrue(report["comparison_ready"])

    def test_report_cli_returns_two_with_null_mean_for_missing_attempt(self):
        self.write_jsonl([episode(0)])
        result = self.run_cli(2)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertIsNone(report["mean_reward"])
        self.assertFalse(report["comparison_ready"])


if __name__ == "__main__":
    unittest.main()
