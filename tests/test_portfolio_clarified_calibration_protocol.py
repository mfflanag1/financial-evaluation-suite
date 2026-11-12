"""Fixed schedule and integrity for the separate clarified development batch."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/clarified_calibration_v1"
for name in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
             "financial_portfolio_clarified"):
    sys.path.insert(0, str(ROOT / "environments" / name / "src"))


def load_freezer():
    spec = importlib.util.spec_from_file_location("clarified_calibration_freezer",
                                                CONFIG / "freeze_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClarifiedCalibrationProtocol(unittest.TestCase):
    @unittest.skipUnless((CONFIG / "manifest.json").exists(), "Freeze after implementation review")
    def test_manifest_binds_complete_payload_and_frozen_source_chain(self):
        self.assertEqual(json.loads((CONFIG / "manifest.json").read_text()),
                         load_freezer().build_manifest())

    def test_only_schedule_and_run_labels_change_from_clarified_pilot(self):
        baseline = tomllib.loads((ROOT / "configs/financial_portfolio/clarified_v1/smoke.toml").read_text())
        config = tomllib.loads((CONFIG / "development.toml").read_text())
        expected = copy.deepcopy(baseline)
        expected["num_rollouts"] = 2
        expected["output_dir"] = "outputs/financial_portfolio/clarified_calibration_v1"
        expected["run"] = {"name": "clarified-development", "dir": "clarified-development"}
        expected["select"]["limit"] = 32
        expected["env"]["taskset"].update(split="dev", seed_start=128, num_tasks=32)
        self.assertEqual(config, expected)
        self.assertIs(config["dry_run"], True)
        self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
        self.assertEqual(config["client"]["base_url"], "https://endpoint.invalid/v1")

    def test_exact_schedule_and_episode_level_analysis_plan(self):
        from financial_portfolio.generator import FAMILIES
        from financial_research.generator import SPLITS

        manifest = load_freezer().build_manifest()
        self.assertEqual(manifest["protocol_id"], "portfolio-clarified-calibration-v1")
        self.assertEqual(manifest["schedule"], {
            "split": "dev", "seed_start": 128, "num_tasks": 32,
            "attempts_per_task": 2, "expected_attempts": 64,
            "episodes_per_family": 8, "attempts_per_family": 16,
            "family_rotation": list(FAMILIES),
            "reserved_test_seeds": [160, 191], "reserved_test_selected": False,
        })
        seeds = range(128, 160)
        self.assertEqual(SPLITS["dev"], (128, 160))
        self.assertFalse(set(seeds) & set(range(0, 8)))
        self.assertFalse(set(seeds) & set(range(*SPLITS["test"])))
        self.assertEqual([sum(seed % 4 == index for seed in seeds) for index in range(4)], [8] * 4)
        self.assertEqual(manifest["analysis"], {
            "eligible_attempts": "all_scheduled_attempts",
            "aggregation_unit": "episode_mean_of_both_attempts",
            "episode_mean_distribution": ["mean", "median", "minimum", "maximum", "quartiles"],
            "repeat_stability": ["absolute_reward_gap", "both_one_neither_full_credit",
                                 "both_one_neither_reproduction",
                                 "decision_and_selection_agreement_on_two_valid_scored_policies"],
            "unresolved_attempts_block_aggregate": True,
            "candidate_failure_reward": 0,
            "continuous_reward_pass_at_k": False,
            "repair_or_replacement_rerun": False,
            "pilot_before_after_comparison": False,
        })

    def test_predecessor_chain_and_all_new_protocol_files_are_bound(self):
        freezer = load_freezer()
        inherited = freezer.verify_predecessors()
        manifest = freezer.build_manifest()
        for relative, expected in inherited.items():
            self.assertEqual(manifest["sha256"][relative], expected)
        payload = {path.relative_to(ROOT).as_posix() for path in CONFIG.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                   and path != CONFIG / "manifest.json"}
        payload.add(Path(__file__).relative_to(ROOT).as_posix())
        self.assertEqual(set(manifest["sha256"]) - set(inherited), payload)
        self.assertEqual(manifest["predecessor_manifest_sha256"], freezer.digest(freezer.CLARIFIED))
        self.assertEqual(manifest["schedule_predecessor_manifest_sha256"], freezer.digest(freezer.CALIBRATION))

    def test_freezer_refuses_to_replace_an_existing_manifest(self):
        freezer = load_freezer()
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "manifest.json"
            existing.write_text("preserved\n")
            with patch.object(freezer, "MANIFEST", existing), \
                    patch.object(freezer, "build_manifest", return_value={"fixture": True}), \
                    patch.object(sys, "argv", ["freeze_manifest.py", "--write"]):
                with self.assertRaises(FileExistsError):
                    freezer.main()
            self.assertEqual(existing.read_text(), "preserved\n")

    @unittest.skipUnless(importlib.util.find_spec("verifiers"), "Install pinned native dependency")
    def test_native_config_preserves_limits_isolation_and_explicit_zero_retries(self):
        from verifiers.v1.configs.cli.run import RunnerConfig

        config = RunnerConfig.model_validate(tomllib.loads((CONFIG / "development.toml").read_text()))
        self.assertEqual(config.env.taskset.id, "financial-portfolio-clarified")
        self.assertEqual(config.env.taskset.split, "dev")
        self.assertEqual(config.env.taskset.seed_start, 128)
        self.assertEqual(config.env.taskset.num_tasks, 32)
        self.assertEqual(config.num_rollouts, 2)
        self.assertEqual(config.max_concurrent, 2)
        self.assertEqual(config.env.retries.max_retries, 0)
        self.assertEqual(config.env.agent.retries.max_retries, 0)
        self.assertEqual(config.env.agent.scoring.retries.max_retries, 0)
        self.assertEqual(config.env.agent.harness.sdk_max_retries, 0)
        self.assertEqual(config.env.agent.harness.id, "financial-portfolio-staged")
        self.assertEqual(config.env.agent.scoring.mode, "isolated")
        self.assertEqual(config.env.agent.max_turns, 32)
        self.assertEqual(config.env.agent.max_output_tokens, 65536)
        self.assertEqual(config.env.agent.max_total_tokens, 262144)
        self.assertTrue(config.dry_run)


if __name__ == "__main__":
    unittest.main()
