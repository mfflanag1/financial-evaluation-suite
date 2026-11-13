"""Prelaunch file binding controls; fixture manifests contain no model weights."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import tempfile
import unittest

from financial_selection_supervised.contract import make_config, validate_config
from financial_selection_supervised.gpu_observer import digest
from financial_selection_supervised.launcher import optimizer_inputs


class OptimizerStagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.raw = make_config(str(self.root / "model"), "sha256:" + "1" * 64, self.root / "run")
        self.config = validate_config(self.raw, self.root / "run", require_resources=False)
        self.initial = self.root / "initial"
        self.initial.mkdir()
        sources = {**json.loads(files("financial_selection_training").joinpath("source_bindings.json").read_text()),
                   "supervised": json.loads(files("financial_selection_supervised").joinpath("source_bindings.json").read_text())}
        self.manifest = {"model_id": "fixture-only", "files": {"model.safetensors": "1" * 64}}
        for name, payload in (("config.json", self.config.model_dump(mode="json")),
                              ("sources.json", sources), ("initial_model_manifest.json", self.manifest)):
            (self.initial / name).write_text(json.dumps(payload))
        (self.initial / "collector.py").write_bytes(files("financial_selection_supervised").joinpath("gpu_observer.py").read_bytes())
        self.plan = {"model_path": self.config.trainer.model.name, "initial_weights_sha256": digest(self.manifest)}
        for name, field in (("config.json", "config_sha256"), ("sources.json", "sources_sha256"), ("collector.py", "collector_sha256")):
            self.plan[field] = hashlib.sha256((self.initial / name).read_bytes()).hexdigest()
        self.write_plan()

    def tearDown(self):
        self.temporary.cleanup()

    def write_plan(self):
        (self.initial / "plan.json").write_text(json.dumps(self.plan))

    def test_only_four_bound_regular_artifacts_are_selected(self):
        (self.initial / "unrelated-private.txt").write_text("must not be staged")
        plan, payloads = optimizer_inputs(self.initial / "plan.json", self.config)
        self.assertEqual(set(payloads), {"config.json", "sources.json", "collector.py", "initial_model_manifest.json"})
        self.assertEqual(plan, self.plan)

    def test_changed_artifact_denied_before_native_start(self):
        (self.initial / "collector.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            optimizer_inputs(self.initial / "plan.json", self.config)

    def test_rebinding_unknown_collector_does_not_authorize_it(self):
        (self.initial / "collector.py").write_text("changed")
        self.plan["collector_sha256"] = hashlib.sha256(b"changed").hexdigest()
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "installed sources"):
            optimizer_inputs(self.initial / "plan.json", self.config)

    def test_wrong_actual_launch_configuration_denied(self):
        raw = make_config(str(self.root / "another-model"), "sha256:" + "1" * 64, self.root / "run")
        config = validate_config(raw, self.root / "run", require_resources=False)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            optimizer_inputs(self.initial / "plan.json", config)

    def test_symlink_and_changed_model_manifest_denied(self):
        target = self.initial / "initial_model_manifest.json"
        data = target.read_bytes()
        target.unlink()
        outside = self.root / "outside.json"
        outside.write_bytes(data)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "invalid prelaunch"):
            optimizer_inputs(self.initial / "plan.json", self.config)
        target.unlink()
        target.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            optimizer_inputs(self.initial / "plan.json", self.config)


if __name__ == "__main__":
    unittest.main()
