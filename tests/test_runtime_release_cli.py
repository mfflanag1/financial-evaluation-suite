"""Prevent effective native settings from escaping the bootstrap binding."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

from test_runtime_release_harness import ROOT, vf


@unittest.skipUnless(vf is not None, "Install pinned runtime dependencies")
class RuntimeCliBindings(unittest.TestCase):
    def setUp(self):
        from financial_research.prime_adapter import ResearchData
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = tomllib.loads((ROOT / "configs/financial_portfolio/runtime_v1/smoke.toml").read_text())
        self.config["output_dir"] = str(self.root / "output")
        self.config["select"]["limit"] = 1
        self.config["env"]["taskset"]["num_tasks"] = 1
        self.image = "sha256:" + "a" * 64
        data = ResearchData(id="fixture-task", prompt="Task", system_prompt="Rules", public_files={"prices.csv": "data"})
        self.task = SimpleNamespace(data=data)
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        self.reference = {"binding": {"seeds": [4], "docker_image": {"image_id": self.image}}, "records": [{
            "task_id": data.id, "public_sha256": {name: sha(value) for name, value in data.public_files.items()},
            "prompt_sha256": sha(data.prompt), "system_prompt_sha256": sha(data.system_prompt)}]}
        self.bootstrap = self.root / "references.json"
        self.bootstrap.write_text("{}")
        (self.root / "evaluator.key").write_text("fixture-only-evaluator-key-not-for-evaluation-0000")
        self.protocol = self.root / "manifest.json"
        self.protocol.write_text("{}")

    def invoke(self, config):
        from financial_portfolio_runtime.cli import run
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return run(path, bootstrap_path=self.bootstrap, repo_root=ROOT, protocol_manifest=self.protocol,
                   journal_dir=self.root / "journal")

    def test_scoring_and_inference_overrides_are_rejected_before_execution(self):
        for section, field, value in (
            ("scoring", "runtime", {"type": "docker", "image": "unverified:other", "allow": []}),
            ("scoring", "env", {"UNBOUND_VALUE": "x"}),
            ("scoring", "network", {"allow": ["example.com"]}),
            (None, "model", "unverified-model"),
            (None, "client", {"type": "eval", "base_url": "https://other.invalid/v1", "api_key_var": "OTHER_KEY"}),
            (None, "sampling", {"temperature": 0.9}),
        ):
            config = deepcopy(self.config)
            target = config["env"]["agent"]
            if section:
                target = target[section]
            target[field] = value
            with self.subTest(section=section, field=field):
                with patch("financial_portfolio_runtime.bootstrap.validate_bootstrap", return_value=self.reference), \
                     patch("financial_portfolio_runtime.bootstrap.inspect_image") as inspect, \
                     patch("verifiers.v1.cli.run.main") as native:
                    with self.assertRaisesRegex(ValueError, "forbids agent inference and scoring"):
                        self.invoke(config)
                inspect.assert_not_called()
                native.assert_not_called()

    def test_solver_and_task_tags_are_pinned_before_native_launch(self):
        with patch("financial_portfolio_runtime.bootstrap.validate_bootstrap", return_value=self.reference), \
             patch("financial_portfolio_runtime.bootstrap.inspect_image", return_value={"image_id": self.image}), \
             patch("financial_portfolio_runtime.prime_adapter.FinancialPortfolioRuntimeTaskset.load", return_value=[self.task]), \
             patch("verifiers.v1.cli.run.main") as native:
            self.invoke(self.config)
        self.assertEqual(native.call_count, 1)
        bound = json.loads((self.root / "journal/bound.json").read_text())
        self.assertEqual(bound["env"]["agent"]["runtime"]["image"], self.image)
        self.assertEqual(bound["env"]["taskset"]["task"]["image"], self.image)
        binding = json.loads((self.root / "journal/run_binding.json").read_text())
        self.assertEqual(binding["docker_image_id"], self.image)
        self.assertEqual(binding["schedule"][0]["task_id"], "fixture-task")


if __name__ == "__main__":
    unittest.main()
