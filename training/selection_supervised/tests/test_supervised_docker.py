"""Opt-in wrapped native pool + Docker, with scripted tokens and no learned policy."""
import asyncio
import base64
from importlib.resources import files
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from financial_selection_supervised.contract import KEY_ENV, make_config, validate_config
from financial_selection_supervised.hooks import COMPONENT_SHA_ENV, install_orchestrator
from financial_selection_supervised.ledger import LEDGER_ENV, Ledger, Limits, BudgetExhausted
from financial_selection_supervised.launcher import watchdog_reason
from financial_selection_supervised.processes import collect_tree, stop_tree, cleanup_containers
from synthetic_provider import ScriptedTrainProvider, make_tokenizer, tool


def schedule(corrupt=False):
    encoded = base64.b64encode(files("financial_selection").joinpath("reference.py").read_bytes()).decode()
    code = ("import base64,json,runpy\n"
            f"open('/workspace/research.py','wb').write(base64.b64decode({encoded!r}))\n"
            "runpy.run_path('/workspace/research.py',run_name='__main__')\n")
    if corrupt:
        code += "p=json.load(open('/workspace/policy.json'));p['input_sha256']='0'*64;json.dump(p,open('/workspace/policy.json','w'))\n"
    return [tool("python - <<'PY'\n" + code + "PY"), "Done.<|im_end|>"]


@unittest.skipUnless(os.environ.get("RUN_SELECTION_SUPERVISED_DOCKER_TESTS") == "1",
                     "Enable serial supervised native Docker fixture explicitly")
class SupervisedDockerTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrapped_pool_preserves_positive_and_candidate_zero_and_global_budget(self):
        import hashlib
        import verifiers.v1 as vf
        from verifiers.v1.configs.client import TrainClientConfig
        from prime_rl.orchestrator.envs import Env
        from financial_selection_training.journal import INFO_KEY
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            KEY_ENV: secrets.token_hex(32), "FIXTURE_TOKEN": "synthetic-local-token",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}):
            root = Path(temporary).resolve()
            image = os.environ["SELECTION_TRAINING_DOCKER_IMAGE"]
            self.assertEqual(subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip(), image)
            ledger = Ledger.create(root / "run", Limits(2, 180, 600, 30), "a" * 64, "b" * 64)
            raw = make_config(str(root / "model"), image, root / "run")
            config = validate_config(raw, root / "run", require_resources=False)
            source = config.orchestrator.train.source[0]
            address = root / "address"
            pool_config = {"env": source.env.model_dump(mode="json"),
                           "serve": source.serve.model_dump(mode="json"),
                           "address_file": str(address), "log": {"level": "error"}}
            path = root / "pool.json"
            path.write_text(json.dumps(pool_config))
            scope = ledger.snapshot()["run"]["scope"]
            environment = {**os.environ, LEDGER_ENV: str(ledger.path), "VF_RUN_ID": scope,
                           COMPONENT_SHA_ENV: hashlib.sha256(path.read_bytes()).hexdigest()}
            tracked, summary = {}, []
            tokenizer = make_tokenizer(root / "tokenizer")
            with (root / "pool.log").open("w") as output:
                process = subprocess.Popen([sys.executable, "-B", "-m", "financial_selection_supervised",
                                            "component", "pool", str(path)],
                                           stdout=output, stderr=output, env=environment, start_new_session=True)
            restore = install_orchestrator(ledger)
            env = None
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    collect_tree(process.pid, tracked)
                    snapshot = ledger.snapshot()
                    if address.exists() and any(w["ready"] is not None for w in snapshot["workers"]):
                        break
                    if process.poll() is not None:
                        self.fail((root / "pool.log").read_text())
                    await asyncio.sleep(.05)
                self.assertEqual(ledger.report()["worker_ready"], 1, (root / "pool.log").read_text())
                env = Env(source, address.read_text().strip(), [])
                await env.start()
                task = next(env.tasks)
                for corrupt in (False, True):
                    with ScriptedTrainProvider(tokenizer, schedule(corrupt)) as provider:
                        episode = await asyncio.wait_for(env.run(
                            TrainClientConfig(base_url=provider.url, api_key_var="FIXTURE_TOKEN",
                                              renderer={"name": "qwen3"}, renderer_model_name=str(root / "tokenizer")),
                            "fixture-policy", vf.Sampling(temperature=1, top_p=1, max_tokens=65536),
                            None, task.data.model_dump(mode="json")), timeout=360)
                        trace = episode.traces[0]
                        summary.append({"reward": trace.reward, "dropped": trace.is_dropped,
                                        "admission": trace.info.get(INFO_KEY, {}).get("admission"),
                                        "provider_calls": len(provider.calls), "task_id": episode.task.data.id})
                self.assertGreater(summary[0]["reward"], 0)
                self.assertEqual(summary[1]["reward"], 0)
                self.assertEqual(summary[0]["task_id"], summary[1]["task_id"])
                self.assertTrue(all(row["admission"]["admitted"] and not row["dropped"] for row in summary))
                self.assertEqual(ledger.report()["outcomes"]["returned"], 2)
                self.assertEqual(len({a["request_id"] for a in ledger.snapshot()["attempts"]}), 2)
                self.assertTrue(all(a["request_id"] is not None for a in ledger.snapshot()["attempts"]))
                self.assertEqual(watchdog_reason(ledger.snapshot(), time.monotonic()), "attempt_budget")
                with self.assertRaises(BudgetExhausted):
                    ledger.reserve()
                (root / "summary.json").write_text(json.dumps({"kind": "fixture", "episodes": summary,
                                                               "supervisor": ledger.report()}, indent=2))
            finally:
                restore()
                if env is not None:
                    for client in env.env_clients:
                        await client.close()
                ledger.close("attempt_budget")
                self.assertTrue(await asyncio.to_thread(stop_tree, process, tracked, 5))
                cleanup_containers(scope)
                secret = os.environ[KEY_ENV].encode()
                for artifact in root.rglob("*"):
                    if artifact.is_file():
                        self.assertNotIn(secret, artifact.read_bytes(), "Key persisted in private fixture output")
                evidence = os.environ.get("SELECTION_SUPERVISED_FIXTURE_EVIDENCE_DIR")
                if evidence:
                    shutil.copytree(root, Path(evidence) / self._testMethodName)


if __name__ == "__main__":
    unittest.main()
