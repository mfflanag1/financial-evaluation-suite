"""Operational changes cannot silently alter the frozen financial condition."""

from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/financial_portfolio/operational_v2"
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
                "financial_portfolio_clarified", "financial_portfolio_operations"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))

from financial_portfolio_operations.cli import (
    validate_run_config, validate_runtime_sources, validate_reference_schedule,
)
from financial_research.accounting import report_path


def freezer():
    spec = importlib.util.spec_from_file_location("operations_freezer", CONFIG / "freeze_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperationalProtocol(unittest.TestCase):
    def setUp(self):
        self.template = tomllib.loads((CONFIG / "development.toml").read_text())

    def test_only_host_harness_and_labels_change(self):
        old = tomllib.loads((ROOT / "configs/financial_portfolio/clarified_calibration_v1/development.toml").read_text())
        old["env"]["agent"]["harness"]["id"] = "financial-portfolio-operations"
        old["output_dir"] = self.template["output_dir"]
        old["run"] = self.template["run"]
        self.assertEqual(self.template, old)

    def test_run_binding_rejects_reserved_seeds_retries_and_context_overrides(self):
        config = deepcopy(self.template)
        config["output_dir"] = "/tmp/operations-fixture"
        validate_run_config(config, self.template, ROOT)
        changes = [
            ("taskset", "seed_start", 160), ("taskset", "num_tasks", 31),
            ("retries", "max_retries", 1),
        ]
        for section, key, value in changes:
            mutated = deepcopy(config)
            mutated["env"][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_run_config(mutated, self.template, ROOT)
        for field, value in (("forward_env", ["MODEL_API_KEY"]), ("sdk_max_retries", False),
                             ("id", "financial-portfolio-staged")):
            mutated = deepcopy(config)
            mutated["env"]["agent"]["harness"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_run_config(mutated, self.template, ROOT)

    def test_run_binding_rejects_credential_urls_and_repo_outputs(self):
        config = deepcopy(self.template)
        config["output_dir"] = "/tmp/operations-fixture"
        for url in ("https://secret@example.invalid/v1", "https://example.invalid/v1?key=secret",
                    "https://example.invalid/v1#secret"):
            mutated = deepcopy(config)
            mutated["client"]["base_url"] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_run_config(mutated, self.template, ROOT)
        for directory in ("relative-output", str(ROOT / "outputs")):
            mutated = deepcopy(config)
            mutated["output_dir"] = directory
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                validate_run_config(mutated, self.template, ROOT)

    def test_deadline_or_provider_error_cannot_become_candidate_zero(self):
        # Reproduce the historical ambiguity: a hook stored zero before the native
        # runner subsequently marked the attempt timed out or failed upstream.
        trace = {"id": "fixture", "version": 1, "task": {"data": {"id": "task"}},
                 "is_completed": True, "ok": True, "is_timeout": False, "errors": [],
                 "rewards": {"financial": {"score": 0, "weight": 1}},
                 "info": {"financial_research_outcome": {
                     "status": "candidate_failure", "reason": "workflow_incomplete",
                     "phase": "scoring", "reproduction_verified": False}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            for error, timeout, status in (([], True, "unknown"),
                    ([{"type": "ProviderError"}], False, "infrastructure_error")):
                current = deepcopy(trace)
                current.update(errors=error, is_timeout=timeout)
                path.write_text(json.dumps(current) + "\n")
                report = report_path(path, expected=1)
                self.assertEqual(report["counts"][status], 1)
                self.assertIsNone(report["mean_reward"])
                self.assertIsNone(report["attempts"][0]["reward"])

    def test_new_freeze_refuses_overwrite(self):
        module = freezer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("existing")
            with patch.object(module, "MANIFEST", path), \
                    patch.object(module, "build_manifest", return_value={}), \
                    patch.object(sys, "argv", ["freeze_manifest.py", "--write"]):
                with self.assertRaises(FileExistsError):
                    module.main()
            self.assertEqual(path.read_text(), "existing")

    def test_runtime_refuses_stale_import_and_dependency_version(self):
        relative = "environments/financial_portfolio/src/financial_portfolio/generator.py"
        manifest = {"packages": {"verifiers": "0.4.1"}, "sha256": {relative: "unused"}}
        with patch("financial_portfolio_operations.cli.version", return_value="0.4.1"), \
                patch("financial_portfolio_operations.cli.importlib.import_module",
                      return_value=SimpleNamespace(__file__="/tmp/stale/generator.py")):
            with self.assertRaisesRegex(ValueError, "Imported source differs"):
                validate_runtime_sources(ROOT, manifest)
        with patch("financial_portfolio_operations.cli.version", return_value="0.0.0"):
            with self.assertRaisesRegex(ValueError, "dependency version differs"):
                validate_runtime_sources(ROOT, manifest)

    def test_live_schedule_cannot_use_foreign_reference_ids_or_inputs(self):
        sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
        families = ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal")
        tasks, records = [], []
        for seed in range(128, 160):
            data = SimpleNamespace(id=f"task-{seed}", public_files={"fixture": str(seed)},
                                   prompt="user", system_prompt="system")
            tasks.append(SimpleNamespace(data=data))
            records.append({"seed": seed, "task_id": data.id, "family": families[seed % 4],
                            "public_sha256": {"fixture": sha(str(seed))},
                            "prompt_sha256": sha("user"), "system_prompt_sha256": sha("system")})
        self.assertEqual(len(validate_reference_schedule(tasks, {"records": records})), 32)
        for field in ("task_id", "public_sha256", "prompt_sha256", "system_prompt_sha256", "family"):
            mutated = deepcopy(records)
            mutated[0][field] = "foreign"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reference_schedule(tasks, {"records": mutated})

    @unittest.skipUnless((CONFIG / "manifest.json").exists(), "Freeze after implementation review")
    def test_frozen_manifest_matches_source_and_predecessor(self):
        self.assertEqual(json.loads((CONFIG / "manifest.json").read_text()), freezer().build_manifest())


if __name__ == "__main__":
    unittest.main()
