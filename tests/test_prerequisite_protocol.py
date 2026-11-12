"""Integrity, paired schedule and native configuration of prerequisite protocol."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_prerequisites/prerequisite_v1"
PACKAGE = ROOT / "environments/financial_prerequisites"
for name in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
             "financial_prerequisites"):
    sys.path.insert(0, str(ROOT / "environments" / name / "src"))


def load_freezer():
    spec = importlib.util.spec_from_file_location("prerequisite_freezer", CONFIG / "freeze_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrerequisiteProtocol(unittest.TestCase):
    def test_manifest_binds_complete_payload_and_frozen_predecessors(self):
        manifest = json.loads((CONFIG / "manifest.json").read_text())
        self.assertEqual(manifest, load_freezer().build_manifest())
        predecessor = ROOT / "configs/financial_portfolio/staged_v1/manifest.json"
        self.assertEqual(hashlib.sha256(predecessor.read_bytes()).hexdigest(),
                         manifest["predecessor_manifest_sha256"])

    def test_exact_task_order_and_strict_advancement(self):
        from financial_prerequisites.cases import load_cases

        manifest = json.loads((CONFIG / "manifest.json").read_text())
        cases = load_cases()
        self.assertEqual([case["id"] for case in cases], load_freezer().CASE_IDS)
        self.assertEqual(manifest["comparison"], {
            "case_ids": [case["id"] for case in cases], "model_count": 2,
            "attempts_per_task_per_model": 1, "expected_attempts_per_model": 4,
            "expected_attempts": 8,
        })
        self.assertEqual(manifest["advancement"], {
            "required_passes": 4, "required_attempts": 4, "saved_policy_correct": True,
            "source_replays": ["original", "alternate"],
            "allowed_missing_unknown_or_infrastructure": 0,
            "portfolio_split": "train", "portfolio_seed_start": 4, "portfolio_num_tasks": 4,
        })

    def test_identical_limits_safe_placeholders_and_zero_retries(self):
        config = tomllib.loads((CONFIG / "comparison.toml").read_text())
        self.assertIs(config["dry_run"], True)
        self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
        self.assertEqual(config["client"], {"type": "eval",
                         "base_url": "https://endpoint.invalid/v1",
                         "api_key_var": "PORTFOLIO_MODEL_API_KEY"})
        self.assertEqual(config["env"]["taskset"], {"id": "financial-prerequisites"})
        self.assertEqual(config["num_rollouts"], 1)
        self.assertEqual(config["max_concurrent"], 2)
        self.assertEqual(config["select"], {"shuffle": False, "limit": 4})
        self.assertEqual(config["sampling"], {"temperature": 0.2, "max_tokens": 8192})
        self.assertEqual(config["env"]["retries"], {"max_retries": 0})
        agent = config["env"]["agent"]
        self.assertEqual(agent["retries"], {"max_retries": 0})
        self.assertEqual(agent["scoring"], {"mode": "isolated", "retries": {"max_retries": 0}})
        self.assertEqual(agent["harness"], {
            "id": "financial-portfolio-staged", "dialect": "chat_completions",
            "edit": True, "search": False, "sdk_max_retries": 0, "forward_env": [],
        })
        self.assertEqual(agent["runtime"], {"type": "docker", "image": "python:3.12-slim", "allow": []})
        self.assertEqual([agent[key] for key in ("max_turns", "max_output_tokens", "max_total_tokens")],
                         [16, 32768, 131072])

    @unittest.skipUnless(importlib.util.find_spec("verifiers"), "Install pinned native dependency")
    def test_native_config_preserves_all_limits_and_retry_layers(self):
        from verifiers.v1.configs.cli.run import RunnerConfig

        config = RunnerConfig.model_validate(tomllib.loads((CONFIG / "comparison.toml").read_text()))
        self.assertEqual(config.env.taskset.id, "financial-prerequisites")
        self.assertEqual(config.env.retries.max_retries, 0)
        self.assertEqual(config.env.agent.retries.max_retries, 0)
        self.assertEqual(config.env.agent.scoring.retries.max_retries, 0)
        self.assertEqual(config.env.agent.harness.sdk_max_retries, 0)
        self.assertEqual(config.env.agent.max_turns, 16)
        self.assertEqual(config.env.agent.max_output_tokens, 32768)
        self.assertEqual(config.env.agent.max_total_tokens, 131072)
        self.assertEqual(config.env.agent.scoring.mode, "isolated")
        self.assertTrue(config.dry_run)


if __name__ == "__main__":
    unittest.main()
