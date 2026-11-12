"""Integrity and fixed historical pairing for the split clarification condition."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/clarified_v1"
for name in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
             "financial_portfolio_clarified"):
    sys.path.insert(0, str(ROOT / "environments" / name / "src"))


def load_freezer():
    spec = importlib.util.spec_from_file_location("clarified_freezer", CONFIG / "freeze_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClarifiedProtocol(unittest.TestCase):
    @unittest.skipUnless((CONFIG / "manifest.json").exists(), "Freeze after implementation review")
    def test_manifest_binds_complete_payload_and_frozen_predecessors(self):
        manifest = json.loads((CONFIG / "manifest.json").read_text())
        self.assertEqual(manifest, load_freezer().build_manifest())
        staged = ROOT / "configs/financial_portfolio/staged_v1/manifest.json"
        prerequisite = ROOT / "configs/financial_prerequisites/prerequisite_v1/manifest.json"
        self.assertEqual(hashlib.sha256(staged.read_bytes()).hexdigest(),
                         manifest["predecessor_manifest_sha256"])
        self.assertEqual(hashlib.sha256(prerequisite.read_bytes()).hexdigest(),
                         manifest["prerequisite_manifest_sha256"])

    def test_frozen_predecessors_still_match(self):
        load_freezer().verify_predecessors()

    def test_preregistered_metrics_and_complete_new_file_inventory(self):
        manifest = load_freezer().build_manifest()
        self.assertEqual(manifest["protocol_id"], "portfolio-clarified-v1")
        self.assertEqual(manifest["comparison"], {
            "design": "descriptive_historical_pair",
            "baseline_protocol_id": "portfolio-staged-v1",
            "baseline_source_commit": "339eb6677a40fb989fe1aea80e8b0678bdf02f21",
            "model": "moonshotai/kimi-k2-0905",
            "split": "train", "seed_start": 4, "num_tasks": 4,
            "attempts_per_task": 1, "expected_attempts": 4,
            "eligible_attempts": "all_scheduled_attempts",
            "changed_solver_input": "appended_split_convention_system_paragraph",
            "metric_order": ["attempt_accounting", "checkpoint_completion",
                             "source_reproduction", "full_credit", "financial_reward",
                             "audit_accuracy", "trial_accuracy", "decision_quality",
                             "split_sensitive_return_diagnostic",
                             "split_direction_evidence"],
            "split_diagnostic": {
                "role": "primary_objective_diagnostic_of_targeted_clarification",
                "requests_per_episode": [
                    {"entity_id": "E02", "day": 36, "field": "forward_return"},
                    {"entity_id": "E05", "day": 74, "field": "forward_return"},
                ],
                "expected_observations_per_condition": 8,
                "expected_source": "saved_public_prices_and_securities_csv",
                "ticker_mapping": "inclusive_validity_intervals_at_each_day",
                "adjusted_close": "close * shares_per_original",
                "forward_return": "adjusted_close(d+1) / adjusted_close(d) - 1",
                "absolute_tolerance": 1e-8,
                "historical_correct_counts_by_seed": [0, 0, 2, 0],
                "historical_total_correct": 2,
                "candidate_source_execution": False,
                "source_reading": "secondary_evidence_only",
            },
            "repair_or_rerun": False,
            "large_calibration_included": False,
        })
        expected = set()
        for directory in (CONFIG, ROOT / "environments/financial_portfolio_clarified"):
            expected.update(path.relative_to(ROOT).as_posix()
                            for path in directory.rglob("*") if path.is_file()
                            and "__pycache__" not in path.parts and path.suffix != ".pyc"
                            and path != CONFIG / "manifest.json")
        expected.update(path.relative_to(ROOT).as_posix()
                        for path in ROOT.glob("tests/test_clarified*.py"))
        self.assertEqual(set(manifest["sha256"]), expected)

    def test_freezer_refuses_to_overwrite_an_existing_manifest(self):
        freezer = load_freezer()
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "manifest.json"
            existing.write_text("preserved baseline\n")
            with patch.object(freezer, "MANIFEST", existing), \
                    patch.object(freezer, "build_manifest", return_value={"test": True}), \
                    patch.object(sys, "argv", ["freeze_manifest.py", "--write"]):
                with self.assertRaises(FileExistsError):
                    freezer.main()
            self.assertEqual(existing.read_text(), "preserved baseline\n")

    def test_only_taskset_and_run_labels_change_from_staged_config(self):
        baseline = tomllib.loads((ROOT / "configs/financial_portfolio/staged_v1/smoke.toml").read_text())
        config = tomllib.loads((CONFIG / "smoke.toml").read_text())
        expected = copy.deepcopy(baseline)
        expected["env"]["taskset"]["id"] = "financial-portfolio-clarified"
        expected["output_dir"] = "outputs/financial_portfolio/clarified_v1"
        expected["run"] = {"name": "clarified-smoke", "dir": "clarified-smoke"}
        self.assertEqual(config, expected)
        self.assertIs(config["dry_run"], True)
        self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
        self.assertEqual(config["client"]["base_url"], "https://endpoint.invalid/v1")
        self.assertEqual(config["num_rollouts"], 1)
        self.assertEqual(config["max_concurrent"], 2)
        self.assertEqual(config["select"], {"shuffle": False, "limit": 4})
        self.assertEqual(config["sampling"], {"temperature": 0.2, "max_tokens": 8192})
        self.assertEqual(config["env"]["taskset"], {"id": "financial-portfolio-clarified",
                         "split": "train", "seed_start": 4, "num_tasks": 4})

    @unittest.skipUnless(importlib.util.find_spec("verifiers"), "Install pinned native dependency")
    def test_native_config_preserves_all_limits_and_retry_layers(self):
        from verifiers.v1.configs.cli.run import RunnerConfig

        config = RunnerConfig.model_validate(tomllib.loads((CONFIG / "smoke.toml").read_text()))
        self.assertEqual(config.env.taskset.id, "financial-portfolio-clarified")
        self.assertEqual(config.env.retries.max_retries, 0)
        self.assertEqual(config.env.agent.retries.max_retries, 0)
        self.assertEqual(config.env.agent.scoring.retries.max_retries, 0)
        self.assertEqual(config.env.agent.harness.id, "financial-portfolio-staged")
        self.assertEqual(config.env.agent.harness.sdk_max_retries, 0)
        self.assertEqual(config.env.agent.max_turns, 32)
        self.assertEqual(config.env.agent.max_output_tokens, 65536)
        self.assertEqual(config.env.agent.max_total_tokens, 262144)
        self.assertEqual(config.env.agent.scoring.mode, "isolated")
        self.assertTrue(config.dry_run)


if __name__ == "__main__":
    unittest.main()
