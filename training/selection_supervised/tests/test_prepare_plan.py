"""Separate-process prepare-plan to launcher validation with synthetic local assets."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest

from financial_selection_supervised.contract import KEY_ENV, make_config


class PreparePlanTests(unittest.TestCase):
    def test_preparation_and_launch_resolution_match_across_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model = root / "synthetic-model"
            model.mkdir()
            (model / "config.json").write_text('{"fixture":true}')
            (model / "tokenizer.json").write_text('{"fixture":true}')
            (model / "model.safetensors").write_bytes(b"synthetic fixture bytes; never load these as a model")
            config = root / "launch.json"
            config.write_text(json.dumps(make_config(str(model), "sha256:" + "1" * 64, root / "run")))
            key = secrets.token_hex(32)
            environment = {**os.environ, KEY_ENV: key, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            prepared = subprocess.run([sys.executable, "-B", "-m", "financial_selection_supervised.gpu_observer",
                                       "prepare-plan", "--config", str(config), "--model-id", "fixture-only-policy",
                                       "--model-revision", "fixture-only-revision", "--evidence-dir", str(root / "initial"),
                                       "--run-id", "fixture-only-run"], env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            script = ("import json,sys;from pathlib import Path;"
                      "from financial_selection_supervised.contract import validate_config;"
                      "from financial_selection_supervised.launcher import optimizer_inputs;"
                      "c=validate_config(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]));"
                      "p,files=optimizer_inputs(Path(sys.argv[3]),c);"
                      "assert len(files)==4;print('bound')")
            checked = subprocess.run([sys.executable, "-B", "-c", script, str(config), str(root / "run"), str(root / "initial/plan.json")],
                                     env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout.strip(), "bound")
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(key.encode(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
