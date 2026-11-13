"""Pinned plugin discovery, unchanged selection data and installed source gates."""
from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import verifiers.v1 as vf
from prime_rl.configs.rl import RLConfig
from prime_rl.env.worker import EnvWorker
from financial_selection.prime_adapter import FinancialSelectionTaskset
from financial_selection.reference import solve
from financial_selection.verifier import grade
from financial_selection_training.environment import verify_native_sources
from financial_selection_training.task import FinancialSelectionTrainingTaskset

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "training/selection/configs/training_v1/contract.json"


def configuration(journal_root):
    config = json.loads(CONFIG.read_text())
    config["orchestrator"]["train"]["source"][0]["env"]["journal_root"] = str(journal_root)
    return RLConfig.model_validate(deepcopy(config))


class NativeContractTests(unittest.TestCase):
    def test_native_plugins_and_bounded_group_two_serial_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(Path(directory))
            source = config.orchestrator.train.source[0]
            self.assertEqual(config.orchestrator.batch_size, 2)
            self.assertEqual(source.group_size, 2)
            self.assertEqual(source.serve.pool.num_workers, 1)
            self.assertEqual(source.serve.max_concurrent, 1)
            self.assertEqual(config.orchestrator.concurrency.max_inflight, 2)
            self.assertTrue(config.orchestrator.constant_trainer_batch_size)
            self.assertEqual(source.sampling.top_p, 1.0)
            self.assertIsNone(config.monitors.prime)
            self.assertIsNone(config.monitors.wandb)
            env = vf.load_environment(source.env)
            self.assertEqual(type(env).__name__, "FinancialSelectionTrainingEnv")
            self.assertEqual(type(env.taskset).__name__, "FinancialSelectionTrainingTaskset")
            self.assertEqual(type(env._harnesses["agent"]).__name__, "FinancialSelectionTrainingHarness")
            self.assertEqual(env.config.max_worker_episodes, 8)
            self.assertEqual(env.config.max_worker_seconds, 900)

    def test_native_worker_wire_and_public_financial_parity(self):
        secret = "12" * 32
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FINANCIAL_SELECTION_KEY": secret}):
            source = configuration(Path(directory)).orchestrator.train.source[0]
            source.env.taskset.task.image = "sha256:" + "1" * 64
            source.env.taskset.num_tasks = 4
            tasks = list(FinancialSelectionTrainingTaskset(source.env.taskset).load())
            baseline = list(FinancialSelectionTaskset(source.env.taskset).load())
            worker = object.__new__(EnvWorker)
            worker.env = vf.load_environment(source.env)
            worker.task_cls = type(tasks[0])
            worker.data_cls = worker.task_cls.data_type()
            self.assertEqual(len(tasks), 4)
            self.assertEqual([t.data.id for t in tasks], sorted(t.data.id for t in tasks))
            for task, original in zip(tasks, baseline):
                wire = task.data.model_dump(mode="json")
                self.assertNotIn(secret, json.dumps(wire))
                self.assertEqual(task.data, original.data)
                self.assertEqual(worker._build_task(wire).data, task.data)
                self.assertEqual(worker._build_task(wire).hash, task.hash)
                result = grade(task.data.public_files, json.dumps(solve(task.data.public_files)))
                self.assertEqual(result["reward"], 1.0)

    def test_sources_and_native_chat_port_are_bound(self):
        verify_native_sources()
        package = files("financial_selection_training")
        pins = json.loads(package.joinpath("source_bindings.json").read_text())
        for relative, expected in pins["financial_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        for relative, expected in pins["ported_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        native = files("verifiers").joinpath("v1/harnesses/utils/core.py").read_text()
        self.assertEqual(package.joinpath("_chat_core.py").read_text(),
                         native.replace("    client = AsyncOpenAI(\n", "    client = AsyncOpenAI(\n        max_retries=0,\n"))

    def test_installed_finance_replay_and_native_worker_mutation_fail_closed(self):
        from financial_selection_training import environment
        actual_files = environment.files
        for package, filename in (("financial_selection", "replay.py"), ("financial_research", "reproduction.py"),
                                  ("prime_rl", "env/worker.py"), ("verifiers", "v1/graph.py")):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / package
                shutil.copytree(Path(actual_files(package).joinpath(filename)).parents[len(Path(filename).parts)-1], copy)
                (copy / filename).write_text("# changed installed dependency\n")
                with patch.object(environment, "files", side_effect=lambda name: copy if name == package else actual_files(name)):
                    with self.assertRaises(RuntimeError):
                        verify_native_sources()
