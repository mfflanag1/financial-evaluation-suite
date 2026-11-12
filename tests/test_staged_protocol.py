"""Integrity and schedule checks for the separate staged protocol."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/staged_v1"
PACKAGE = ROOT / "environments/financial_portfolio_staged"
for name in ("financial_research", "financial_portfolio", "financial_portfolio_staged"):
    sys.path.insert(0, str(ROOT / "environments" / name / "src"))


class StagedProtocol(unittest.TestCase):
    def test_new_manifest_binds_new_package_and_frozen_predecessor(self):
        manifest = json.loads((CONFIG / "manifest.json").read_text())
        self.assertEqual(manifest["protocol_id"], "portfolio-staged-v1")
        predecessor = ROOT / "configs/financial_portfolio/calibration_v1/manifest.json"
        self.assertEqual(hashlib.sha256(predecessor.read_bytes()).hexdigest(),
                         manifest["predecessor_manifest_sha256"])
        for relative, digest in manifest["sha256"].items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        package_files = {p.relative_to(ROOT).as_posix() for p in PACKAGE.rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        self.assertEqual(package_files, {p for p in manifest["sha256"]
                                        if p.startswith("environments/financial_portfolio_staged/")})

    def test_safe_config_fresh_training_schedule_and_explicit_retries(self):
        config = tomllib.loads((CONFIG / "smoke.toml").read_text())
        self.assertTrue(config["dry_run"])
        self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
        self.assertEqual(config["client"]["base_url"], "https://endpoint.invalid/v1")
        self.assertEqual(set(config["client"]), {"type", "base_url", "api_key_var"})
        self.assertEqual(config["env"]["taskset"], {"id": "financial-portfolio-staged",
                          "split": "train", "seed_start": 4, "num_tasks": 4})
        self.assertEqual(config["num_rollouts"], 1)
        self.assertEqual(config["select"], {"shuffle": False, "limit": 4})
        self.assertEqual(config["env"]["retries"], {"max_retries": 0})
        agent = config["env"]["agent"]
        self.assertEqual(agent["retries"], {"max_retries": 0})
        self.assertEqual(agent["scoring"], {"mode": "isolated", "retries": {"max_retries": 0}})
        self.assertEqual(agent["harness"]["id"], "financial-portfolio-staged")
        self.assertEqual(agent["harness"]["sdk_max_retries"], 0)
        self.assertEqual(agent["harness"]["forward_env"], [])
        self.assertEqual(agent["runtime"], {"type": "docker", "image": "python:3.12-slim", "allow": []})
        self.assertEqual([agent[key] for key in ("max_turns", "max_output_tokens", "max_total_tokens")],
                         [32, 65536, 262144])

    @unittest.skipUnless(importlib.util.find_spec("verifiers"), "Install pinned native dependency")
    def test_native_config_preserves_all_four_retry_layers(self):
        from verifiers.v1.configs.cli.run import RunnerConfig

        config = RunnerConfig.model_validate(tomllib.loads((CONFIG / "smoke.toml").read_text()))
        self.assertEqual(config.env.retries.max_retries, 0)
        self.assertEqual(config.env.agent.retries.max_retries, 0)
        self.assertEqual(config.env.agent.scoring.retries.max_retries, 0)
        self.assertEqual(config.env.agent.harness.sdk_max_retries, 0)
        self.assertTrue(config.dry_run)


if __name__ == "__main__":
    unittest.main()
