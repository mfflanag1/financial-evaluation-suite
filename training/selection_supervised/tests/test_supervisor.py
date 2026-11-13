"""Behavioral controls; fixtures below are never represented as policy rollouts."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from financial_selection_supervised.contract import (
    KEY_ENV, make_config, sanitized_environment, validate_config, verify_sources,
)
from financial_selection_supervised.ledger import BudgetExhausted, LEDGER_ENV, Ledger, Limits
from financial_selection_supervised.launcher import process_group_class, watchdog_reason
from financial_selection_supervised.processes import alive, cleanup_containers, collect_tree, stop_tree


def reserve_in_process(path):
    ledger = Ledger(path)
    try:
        return ledger.reserve()
    except BudgetExhausted:
        return None


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "run"
        self.ledger = Ledger.create(self.root, Limits(8, 20, 60, 1), "a" * 64, "b" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_concurrent_processes_share_eight_attempt_cap(self):
        with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(reserve_in_process, [self.ledger.path] * 24))
        self.assertEqual(sum(r is not None for r in results), 8)
        self.assertEqual(len(set(r for r in results if r)), 8)
        self.assertEqual(self.ledger.report()["outcomes"]["reserved"], 8)

    def test_key_and_contents_never_in_ledger(self):
        with patch.dict(os.environ, {KEY_ENV: "d" * 64}):
            attempt = self.ledger.reserve()
            self.ledger.bind_request(attempt, "e" * 32)
            self.ledger.finish(attempt, "returned")
        self.assertNotIn(b"d" * 64, self.ledger.path.read_bytes())
        self.assertEqual(self.ledger.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)

    def test_duplicate_request_and_finalization_fail(self):
        first, second = self.ledger.reserve(), self.ledger.reserve()
        self.ledger.bind_request(first, "a" * 32)
        with self.assertRaises(RuntimeError):
            self.ledger.bind_request(first, "b" * 32)
        with self.assertRaises(Exception):
            self.ledger.bind_request(second, "a" * 32)
        self.ledger.finish(first, "failed")
        with self.assertRaises(RuntimeError):
            self.ledger.finish(first, "returned")

    def test_worker_replacement_is_not_a_fresh_budget(self):
        self.ledger.worker_boot(100, 2)
        with self.assertRaises(RuntimeError):
            self.ledger.worker_boot(101, 3)
        self.ledger.worker_ready(100)
        with self.assertRaises(RuntimeError):
            self.ledger.worker_ready(100)

    def test_stop_keeps_unresolved_reservations_visible(self):
        self.ledger.reserve()
        self.ledger.close("time_budget")
        with self.assertRaises(BudgetExhausted):
            self.ledger.reserve()
        self.assertFalse(self.ledger.report()["accounting_complete"])
        self.assertEqual(self.ledger.report()["outcomes"]["reserved"], 1)

    def test_invalid_limits(self):
        for limit in (True, 0, 1, 65, 2.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                Limits(limit)
        for limit in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                Limits(total_seconds=limit)

    def test_time_budget_checked_in_admission(self):
        with patch("financial_selection_supervised.ledger.time.monotonic", return_value=time.monotonic() + 61):
            with self.assertRaises(BudgetExhausted):
                self.ledger.reserve()

    def test_broker_health_never_substitutes_for_worker_ready(self):
        snapshot = self.ledger.snapshot()
        now = snapshot["run"]["started"] + 21
        self.assertEqual(watchdog_reason(snapshot, now, lambda *_: True), "startup_timeout")
        self.ledger.worker_boot(100, 2)
        self.assertEqual(watchdog_reason(self.ledger.snapshot(), now, lambda *_: True), "startup_timeout")
        self.ledger.worker_ready(100)
        self.assertEqual(watchdog_reason(self.ledger.snapshot(), now, lambda *_: True), "startup_timeout")
        self.ledger.reserve()
        self.assertIsNone(watchdog_reason(self.ledger.snapshot(), now, lambda *_: True))
        self.assertEqual(watchdog_reason(self.ledger.snapshot(), now, lambda *_: False), "worker_died")

    def test_successful_native_wait_allows_only_bounded_worker_shutdown(self):
        self.ledger.worker_boot(100, 2)
        self.ledger.worker_ready(100)
        self.ledger.finish(self.ledger.reserve(), "returned")
        self.ledger.native_finished()
        with self.assertRaises(BudgetExhausted):
            self.ledger.reserve()
        snapshot = self.ledger.snapshot()
        finished = snapshot["run"]["native_finished"]
        self.assertIsNone(watchdog_reason(snapshot, finished + .5, lambda *_: False))
        self.assertEqual(watchdog_reason(snapshot, finished + 1.1, lambda *_: False), "native_failed")

    def test_equal_reward_groups_cannot_run_forever(self):
        self.ledger.worker_boot(100, 2)
        self.ledger.worker_ready(100)
        for _ in range(8):
            self.ledger.finish(self.ledger.reserve(), "returned")
        self.assertEqual(watchdog_reason(self.ledger.snapshot(), time.monotonic(), lambda *_: True), "attempt_budget")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "run"
        self.model = Path(self.temp.name).resolve() / "model"
        self.model.mkdir()
        for path in ("config.json", "tokenizer.json", "model.safetensors"):
            (self.model / path).write_text("fixture; not real model weights")
        self.raw = make_config(str(self.model), "sha256:" + "1" * 64, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_contract_resolves_supervised_env_and_unchanged_taskset(self):
        config = validate_config(self.raw, self.root)
        self.assertEqual(config.orchestrator.train.source[0].env.id, "financial-selection-supervised")
        self.assertEqual(config.orchestrator.train.source[0].env.taskset.id, "financial-selection-training")
        self.assertEqual(config.trainer.max_steps, 1)
        self.assertEqual(config.deployment.num_train_gpus, 1)
        self.assertEqual(config.trainer.model.lora.rank, 8)
        self.assertEqual(config.trainer.model.attn, "flash_attention_2")
        self.assertEqual(config.inference.vllm.max_lora_rank, 8)
        self.assertEqual(config.trainer.weight_broadcast.type, "filesystem")

    def test_every_unreviewed_surface_denied(self):
        mutations = [
            lambda c: c.update(slurm={}), lambda c: c.update(resume={}),
            lambda c: c.update(max_steps=True), lambda c: c.update(max_steps=2),
            lambda c: c.update(inference=None), lambda c: c.update(env_vars={KEY_ENV: "secret"}),
            lambda c: c.update(dashboard=True), lambda c: c.update(dry_run=True),
            lambda c: c["orchestrator"]["train"]["source"][0]["serve"].update(address="tcp://external:1234"),
            lambda c: c["orchestrator"]["train"]["source"][0]["serve"]["pool"].update(num_workers=2),
            lambda c: c["orchestrator"]["train"]["source"][0]["env"]["retries"].update(max_retries=1),
            lambda c: c["orchestrator"]["train"]["source"][0]["env"]["agent"].update(client={"api_key_var": KEY_ENV}),
            lambda c: c["orchestrator"]["train"]["source"][0]["env"]["agent"]["harness"].update(forward_env=[KEY_ENV]),
        ]
        for mutate in mutations:
            candidate = deepcopy(self.raw)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_config(candidate, self.root)

    def test_git_checkout_private_output_is_denied(self):
        (Path(self.temp.name) / ".git").mkdir()
        with self.assertRaisesRegex(ValueError, "outside Git"):
            validate_config(self.raw, self.root)

    def test_remote_or_absent_model_and_placeholder_image_denied(self):
        for name in ("remote/model", str(self.model / "absent")):
            raw = make_config(name, "sha256:" + "1" * 64, self.root)
            with self.assertRaises(ValueError):
                validate_config(raw, self.root)
        raw = make_config(str(self.model), "sha256:" + "0" * 64, self.root)
        with self.assertRaises(ValueError):
            validate_config(raw, self.root)

    def test_pinned_sources_verify_and_tampering_fails(self):
        self.assertEqual(len(verify_sources()), 64)
        from financial_selection_supervised import contract
        real = contract.files
        fake_manifest = json.loads(real("financial_selection_supervised").joinpath("source_bindings.json").read_text())
        fake_manifest["supervisor_sources"]["ledger.py"] = "0" * 64
        class Resource:
            def joinpath(self, name):
                if name == "source_bindings.json":
                    return SimpleNamespace(read_bytes=lambda: json.dumps(fake_manifest).encode())
                return real("financial_selection_supervised").joinpath(name)
        with patch.object(contract, "files", side_effect=lambda package: Resource() if package == "financial_selection_supervised" else real(package)):
            with self.assertRaisesRegex(RuntimeError, "Supervisor source drift"):
                verify_sources()

    def test_secret_aliases_stripped_without_mutating_environment(self):
        key = "a" * 64
        environment = {KEY_ENV: key, "ALIAS": key.upper(), "HEADER": "prefix" + key + "suffix", "SAFE": "1"}
        self.assertEqual(sanitized_environment(environment), {"SAFE": "1"})
        self.assertIn(KEY_ENV, environment)


class HooksTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "run"
        self.ledger = Ledger.create(self.root, Limits(), "a" * 64, "b" * 64)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_exception_before_task_validation_counts(self):
        from prime_rl.orchestrator.envs import Env
        from financial_selection_supervised.hooks import install_orchestrator
        async def fail(*_args, **_kwargs):
            self.assertEqual(self.ledger.report()["attempts"], 1)
            raise ValueError("malformed task before any trace")
        with patch.object(Env, "run", fail):
            restore = install_orchestrator(self.ledger)
            try:
                with self.assertRaises(ValueError):
                    await Env.run(object())
            finally:
                restore()
        self.assertEqual(self.ledger.report()["outcomes"]["failed"], 1)
        self.assertIsNone(self.ledger.snapshot()["attempts"][0]["request_id"])

    async def test_native_request_binds_before_send_and_preserves_result(self):
        from prime_rl.orchestrator.envs import Env
        from prime_rl.env.client import EnvClient
        from financial_selection_supervised.hooks import install_orchestrator
        sentinel = object()
        async def request(_self, req, result_type, timeout=None, on_delta=None, request_id=None):
            self.assertEqual(self.ledger.snapshot()["attempts"][0]["request_id"], request_id)
            return sentinel
        async def run(_self):
            return await EnvClient._request(object(), SimpleNamespace(method="run"), object, request_id="c" * 32)
        with patch.object(Env, "run", run), patch.object(EnvClient, "_request", request):
            restore = install_orchestrator(self.ledger)
            try:
                self.assertIs(await Env.run(object()), sentinel)
                with self.assertRaises(RuntimeError):
                    await EnvClient._request(object(), SimpleNamespace(method="run"), object)
            finally:
                restore()
        self.assertEqual(self.ledger.report()["outcomes"]["returned"], 1)

    async def test_cancellation_is_terminal_without_reward_rewrite(self):
        from prime_rl.orchestrator.envs import Env
        from financial_selection_supervised.hooks import install_orchestrator
        entered = asyncio.Event()
        async def run(_self):
            entered.set()
            await asyncio.Future()
        with patch.object(Env, "run", run):
            restore = install_orchestrator(self.ledger)
            try:
                task = asyncio.create_task(Env.run(object()))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                restore()
        self.assertEqual(self.ledger.report()["outcomes"]["cancelled"], 1)


class ProcessTests(unittest.TestCase):
    def test_cleanup_reaches_new_session_grandchild_and_kill_fallback(self):
        import psutil
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready"
            child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
            script = ("import subprocess,sys,time,signal,pathlib;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                      f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True);"
                      f"pathlib.Path({str(ready)!r}).write_text(str(p.pid));time.sleep(60)")
            process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
            tracked = {}
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                child = int(ready.read_text())
                created = psutil.Process(child).create_time()
                collect_tree(process.pid, tracked)
                self.assertIn(child, tracked)
                self.assertTrue(stop_tree(process, tracked, .1))
                self.assertFalse(alive(child, created))
            finally:
                stop_tree(process, tracked, .1)

    def test_cleanup_uses_only_exact_run_labels(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(stdout="a" * 64 + "\n" if len(calls) == 1 else "")
        self.assertEqual(cleanup_containers("b" * 32, runner=runner), 1)
        self.assertEqual(calls[0][-1], "label=verifiers.run=" + "b" * 32)
        self.assertEqual(calls[1], ["docker", "rm", "-f", "a" * 64])
        with self.assertRaises(ValueError):
            cleanup_containers("broad*", runner=runner)


class CleanupFailureTests(unittest.TestCase):
    def exercise(self, failure):
        from financial_selection_supervised import launcher
        import sqlite3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "run"
            raw = make_config(str(root.parent / "model"), "sha256:" + "1" * 64, root)
            actual_popen = subprocess.Popen
            actual_snapshot = Ledger.snapshot
            children = []
            snapshots = 0
            def child(*_args, **kwargs):
                process = actual_popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                       start_new_session=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                children.append(process)
                return process
            def snapshot(instance):
                nonlocal snapshots
                snapshots += 1
                if failure == "snapshot" and snapshots > 1:
                    raise sqlite3.OperationalError("fixture unavailable ledger")
                return actual_snapshot(instance)
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {KEY_ENV: "a" * 64}))
                stack.enter_context(patch.object(launcher, "verify_sources", return_value="b" * 64))
                stack.enter_context(patch.object(launcher, "validate_config"))
                stack.enter_context(patch.object(launcher.subprocess, "run"))
                stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=child))
                stack.enter_context(patch.object(launcher, "cleanup_containers"))
                stack.enter_context(patch.object(launcher, "watchdog_reason", return_value="time_budget"))
                stack.enter_context(patch.object(Ledger, "snapshot", snapshot))
                if failure == "close":
                    stack.enter_context(patch.object(Ledger, "close", side_effect=sqlite3.OperationalError("fixture locked ledger")))
                if failure == "snapshot":
                    with self.assertRaises(sqlite3.OperationalError):
                        launcher.supervise(raw, root, Limits(8, 1, 2, .1))
                else:
                    launcher.supervise(raw, root, Limits(8, 1, 2, .1))
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            report = json.loads((root / "supervisor-report.json").read_text())
            self.assertTrue(report["process_cleanup_complete"])
            self.assertFalse(report["control_ledger_complete"])
            self.assertFalse(report["success"])
            self.assertFalse(report["optimizer_validated"])

    def test_locked_close_cannot_bypass_real_process_teardown(self):
        self.exercise("close")

    def test_broken_snapshot_cannot_bypass_real_process_teardown(self):
        self.exercise("snapshot")


if __name__ == "__main__":
    unittest.main()
