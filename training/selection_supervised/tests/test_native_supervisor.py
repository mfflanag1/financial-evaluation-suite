"""Real spawned native pool failures: no model calls, Docker or policy updates."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from financial_selection_supervised.contract import KEY_ENV, make_config
from financial_selection_supervised.hooks import COMPONENT_SHA_ENV
from financial_selection_supervised.ledger import Ledger, LEDGER_ENV, Limits
from financial_selection_supervised.launcher import watchdog_reason, process_group_class
from financial_selection_supervised.processes import collect_tree, stop_tree, alive


class NativePoolTests(unittest.TestCase):
    def test_healthy_native_broker_with_failed_spawned_worker_is_not_ready(self):
        from prime_rl.env.client import EnvClient
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger = Ledger.create(root / "run", Limits(8, 15, 30, 1), "a" * 64, "b" * 64)
            raw = make_config(str(root / "model"), "sha256:" + "0" * 64, root / "run")
            address = root / "address"
            config = {"env": raw["orchestrator"]["train"]["source"][0]["env"],
                      "serve": {"pool": {"type": "static", "num_workers": 1}, "max_concurrent": 1},
                      "address_file": str(address), "log": {"level": "error"}}
            path = root / "pool.json"
            path.write_text(json.dumps(config))
            environment = {**os.environ, LEDGER_ENV: str(ledger.path), KEY_ENV: "c" * 64,
                           COMPONENT_SHA_ENV: hashlib.sha256(path.read_bytes()).hexdigest()}
            tracked = {}
            with (root / "pool.log").open("w") as output:
                process = subprocess.Popen([sys.executable, "-B", "-m", "financial_selection_supervised",
                                            "component", "pool", str(path)],
                                           stdout=output, stderr=output, env=environment, start_new_session=True)
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    collect_tree(process.pid, tracked)
                    snapshot = ledger.snapshot()
                    if address.exists() and snapshot["workers"] and not alive(snapshot["workers"][0]["pid"], snapshot["workers"][0]["created"]):
                        break
                    if process.poll() is not None:
                        self.fail("Broker unexpectedly exited: " + (root / "pool.log").read_text())
                    time.sleep(.05)
                snapshot = ledger.snapshot()
                self.assertEqual(len(snapshot["workers"]), 1, (root / "pool.log").read_text())
                self.assertIsNone(snapshot["workers"][0]["ready"])
                self.assertEqual(watchdog_reason(snapshot, time.monotonic()), "worker_died")
                async def health():
                    client = EnvClient(address.read_text().strip())
                    try:
                        return await client.health()
                    finally:
                        await client.close()
                self.assertTrue(asyncio.run(health()))
                ledger.close("worker_died")
            finally:
                self.assertTrue(stop_tree(process, tracked, 1))
            self.assertEqual(ledger.report()["worker_ready"], 0)
            self.assertNotIn("c" * 64, json.dumps(ledger.report()))

    def test_real_native_process_group_routes_only_expected_components_and_strips_secrets(self):
        from prime_rl.utils.process import ProcessGroup
        import psutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger = Ledger.create(root / "run", Limits(), "a" * 64, "b" * 64)
            path = root / "component.json"
            path.write_text("{}")
            captured = []
            # Patch the native spawning boundary only. Command rewrite, environment
            # sanitation and component ledger recording are the production wrapper.
            def start(instance, name, cmd, env, log_path):
                captured.append((name, cmd, env))
                instance.processes.append(type("Process", (), {"pid": os.getpid()})())
            key = "d" * 64
            with patch.object(ProcessGroup, "start", start), patch.dict(os.environ, {LEDGER_ENV: str(ledger.path)}):
                group = process_group_class(key)()
                environment = {KEY_ENV: key, "ALIAS": key.upper(), "SAFE": "1", LEDGER_ENV: str(ledger.path)}
                group.start("inference", ["inference", "@", str(path)], environment, root / "inference.log")
                self.assertNotIn(KEY_ENV, captured[-1][2])
                self.assertNotIn("ALIAS", captured[-1][2])
                group.start("orchestrator", ["orchestrator", "@", str(path)], environment, root / "orchestrator.log")
                self.assertEqual(captured[-1][1][:5], [sys.executable, "-I", "-B", "-m", "financial_selection_supervised"])
                self.assertEqual(captured[-1][2][KEY_ENV], key)
                self.assertNotIn("ALIAS", captured[-1][2])
                group.start("trainer", ["torchrun", "--nproc-per-node=1", "-m", "prime_rl.trainer.rl.train", "@", str(path)], environment, root / "trainer.log")
                self.assertNotIn(KEY_ENV, captured[-1][2])
                with self.assertRaises(ValueError):
                    group.start("redis", ["redis-server"], environment, root / "unknown.log")
            self.assertEqual({r["name"] for r in ledger.snapshot()["components"]}, {"inference", "orchestrator", "trainer"})


if __name__ == "__main__":
    unittest.main()
