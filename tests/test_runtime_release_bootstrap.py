"""Fresh-key bootstrap binding tests; container execution is explicitly mocked."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_runtime_release_harness import ROOT, vf


@unittest.skipUnless(vf is not None, "Install pinned runtime dependencies")
class RuntimeBootstrap(unittest.TestCase):
    def test_new_key_and_complete_artifact_binding_without_historical_preflight(self):
        from financial_portfolio_runtime import bootstrap as module
        from financial_portfolio.generator import build_episode
        from financial_research.verifier import reference_submission

        image = {"image_id": "sha256:" + "a" * 64, "repo_digests": []}
        def fake_container(workspace, image_id):
            # This tests bootstrap bookkeeping, not public execution or financial independence.
            policy = reference_submission(build_episode(4, split="train"))
            (workspace / "policy.json").write_text(json.dumps(policy))
            return {"kind": "mock_for_unit_test", "image_id": image_id}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "manifest.json"
            target = ROOT / "environments/financial_portfolio/src/financial_portfolio/reference.py"
            protocol.write_text(json.dumps({"protocol_id": "portfolio-runtime-v1", "packages": {},
                                           "sha256": {target.relative_to(ROOT).as_posix(): module._sha(target.read_bytes())}}))
            output = root / "bootstrap"
            with patch.object(module, "inspect_image", return_value=image), patch.object(module, "_container_reference", fake_container):
                report = module.bootstrap(output, repo_root=ROOT, protocol_manifest=protocol, num_tasks=1)
                self.assertEqual(report["reference_count"], 1)
                self.assertEqual(report["model_calls"], 0)
                self.assertFalse(report["native_agent_harness_validation"])
                self.assertEqual((output / "evaluator.key").stat().st_mode & 0o777, 0o600)
                self.assertNotIn("baseline", json.dumps(report))
                self.assertFalse(any(output.rglob("episode.json")))
                module.validate_bootstrap(output / "references.json", repo_root=ROOT, protocol_manifest=protocol)
                with self.assertRaises(FileExistsError):
                    module.bootstrap(output, repo_root=ROOT, protocol_manifest=protocol, num_tasks=1)
            policy = output / "train-4/policy.json"
            policy.write_text(policy.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                module.validate_bootstrap(output / "references.json", repo_root=ROOT, protocol_manifest=protocol)

    def test_runtime_config_resolves_concrete_native_plugins(self):
        import tomllib
        from verifiers.v1.configs.cli.run import RunnerConfig
        from financial_portfolio_runtime.harness import FinancialPortfolioRuntimeHarnessConfig
        resolved = RunnerConfig.model_validate(tomllib.loads((ROOT / "configs/financial_portfolio/runtime_v1/smoke.toml").read_text()))
        self.assertIsInstance(resolved.env.agent.harness, FinancialPortfolioRuntimeHarnessConfig)
        self.assertEqual(resolved.env.taskset.id, "financial-portfolio-runtime")
        self.assertEqual(resolved.max_concurrent, 1)
        self.assertTrue(resolved.dry_run)


if __name__ == "__main__":
    unittest.main()
