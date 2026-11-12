"""Offline integrity and native-schema checks for the frozen calibration plan."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/calibration_v1"
for name in ("financial_research", "financial_portfolio"):
    sys.path.insert(0, str(ROOT / "environments" / name / "src"))


class FrozenCalibration(unittest.TestCase):
    def test_frozen_files_and_package_inventory_match_manifest(self):
        manifest = json.loads((CONFIG / "manifest.json").read_text())
        self.assertEqual(manifest["protocol_id"], "portfolio-calibration-v1")
        required_configs = {f"configs/financial_portfolio/calibration_v1/{name}"
                            for name in ("README.md", "smoke.toml", "development.toml")}
        self.assertTrue(required_configs <= set(manifest["sha256"]))
        for relative, expected in manifest["sha256"].items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        for name in ("financial_research", "financial_portfolio"):
            prefix = f"environments/{name}/"
            actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / prefix).rglob("*")
                      if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
            self.assertEqual(actual, {name for name in manifest["sha256"] if name.startswith(prefix)})

    def test_schedules_are_balanced_disjoint_and_exclude_reserved_seeds(self):
        manifest = json.loads((CONFIG / "manifest.json").read_text())
        covered = []
        for stage, split, start, count, repeats in (("smoke", "train", 0, 4, 1),
                                                   ("development", "dev", 128, 32, 2)):
            config = tomllib.loads((CONFIG / f"{stage}.toml").read_text())
            taskset = config["env"]["taskset"]
            self.assertEqual(taskset, {"id": "financial-portfolio", "split": split,
                                       "seed_start": start, "num_tasks": count})
            self.assertEqual(config["select"], {"shuffle": False, "limit": count})
            self.assertEqual(config["num_rollouts"], repeats)
            self.assertEqual(manifest["runs"][stage], {"split": split, "seed_start": start,
                                                       "num_tasks": count, "attempts_per_task": repeats,
                                                       "expected_attempts": count * repeats})
            seeds = set(range(start, start + count))
            self.assertTrue(seeds.isdisjoint(range(160, 192)))
            self.assertEqual([sum(seed % 4 == family for seed in seeds) for family in range(4)], [count // 4] * 4)
            covered.append(seeds)
        self.assertTrue(covered[0].isdisjoint(covered[1]))

    def test_preparation_configs_cannot_target_a_real_endpoint_by_default(self):
        for stage in ("smoke", "development"):
            config = tomllib.loads((CONFIG / f"{stage}.toml").read_text())
            self.assertIs(config["dry_run"], True)
            self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
            self.assertEqual(config["client"], {"type": "eval", "base_url": "https://endpoint.invalid/v1",
                                                "api_key_var": "PORTFOLIO_MODEL_API_KEY"})
            agent = config["env"]["agent"]
            self.assertEqual(agent["harness"]["forward_env"], [])
            self.assertFalse(agent["harness"]["search"])
            self.assertEqual(agent["runtime"], {"type": "docker", "image": "python:3.12-slim", "allow": []})
            self.assertEqual(agent["scoring"], {"mode": "isolated", "retries": {"max_retries": 0}})

    @unittest.skipUnless(importlib.util.find_spec("verifiers"), "Install the pinned prime optional dependency")
    def test_native_parser_preserves_schedule_budgets_and_isolation_without_running(self):
        from verifiers.v1.configs.cli.run import RunnerConfig

        self.assertEqual(importlib.metadata.version("verifiers"), "0.4.1")
        for stage, attempts in (("smoke", 4), ("development", 64)):
            config = RunnerConfig.model_validate(tomllib.loads((CONFIG / f"{stage}.toml").read_text()))
            self.assertTrue(config.dry_run)
            self.assertEqual(config.env.taskset.num_tasks * config.num_rollouts, attempts)
            self.assertEqual(config.env.agent.max_turns, 32)
            self.assertEqual(config.env.agent.max_output_tokens, 65536)
            self.assertEqual(config.env.agent.max_total_tokens, 262144)
            self.assertEqual(config.env.agent.scoring.mode, "isolated")
            self.assertEqual(config.env.agent.scoring.retries.max_retries, 0)


if __name__ == "__main__":
    unittest.main()
