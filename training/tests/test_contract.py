"""Installed native loaders, financial parity, worker wire and fixed smoke shape."""

from copy import deepcopy
import hashlib
import importlib.util
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
from financial_portfolio.generator import build_episode
from financial_research.verifier import reference_submission, score_submission

from financial_portfolio_training.environment import verify_native_sources
from financial_portfolio_training.task import FinancialPortfolioTrainingTaskset, system_prompt
from financial_portfolio_training import validation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "training/configs/training_v1/contract.json"


def configuration(journal_root):
    config = json.loads(CONFIG.read_text())
    config["orchestrator"]["train"]["source"][0]["env"]["journal_root"] = str(journal_root)
    return RLConfig.model_validate(deepcopy(config))


class NativeContractTests(unittest.TestCase):
    def test_native_plugins_and_group_two_serial_contract(self):
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
            self.assertIsNone(config.orchestrator.monitors.prime)
            self.assertIsNone(config.trainer.monitors.prime)
            env = vf.load_environment(source.env)
            self.assertEqual(type(env).__name__, "FinancialPortfolioTrainingEnv")
            self.assertEqual(type(env.taskset).__name__, "FinancialPortfolioTrainingTaskset")
            self.assertEqual(type(env._harnesses["agent"]).__name__, "FinancialPortfolioTrainingHarness")

    def test_four_native_worker_task_wire_roundtrips_and_financial_parity(self):
        secret = "synthetic-test-host-key-not-a-live-credential-00000000"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": secret}):
            source = configuration(Path(directory)).orchestrator.train.source[0]
            tasks = list(FinancialPortfolioTrainingTaskset(source.env.taskset).load())
            worker = object.__new__(EnvWorker)
            worker.env = vf.load_environment(source.env)
            worker.task_cls = type(tasks[0])
            worker.data_cls = worker.task_cls.data_type()
            self.assertFalse(any(field.exclude for field in worker.data_cls.model_fields.values()))
            self.assertEqual(len(tasks), 4)
            for seed, task in zip(range(4, 8), tasks):
                wire = task.data.model_dump(mode="json")
                self.assertNotIn(secret, json.dumps(wire))
                rebuilt = worker._build_task(wire)
                self.assertEqual(rebuilt.data, task.data)
                self.assertEqual(rebuilt.hash, task.hash)
                self.assertEqual(rebuilt.key, task.key)
                baseline = build_episode(seed, split="train")
                self.assertEqual(task.data.public_files, baseline["public_files"])
                self.assertEqual(task.data.prompt, baseline["prompt"])
                self.assertEqual(task.data.id, baseline["id"])
                self.assertEqual(task.data.system_prompt, system_prompt(baseline["public_files"]))
                score = score_submission(baseline, reference_submission(baseline))
                self.assertGreater(score["reward"], 0)

    def test_frozen_financial_sources_and_minimal_port_are_exactly_bound(self):
        verify_native_sources()
        package = files("financial_portfolio_training")
        pins = json.loads(package.joinpath("source_bindings.json").read_text())
        for relative, expected in pins["financial_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        original = ROOT / "environments/financial_portfolio_staged/src/financial_portfolio_staged"
        for name in ("instructions.txt", "validation.py"):
            self.assertEqual(package.joinpath(name).read_bytes(), (original / name).read_bytes())
        expected = (original / "prime_adapter.py").read_text().replace(
            "FinancialPortfolioStagedTaskset", "WorkflowTaskset").replace(
            "FinancialPortfolioStagedTask", "WorkflowTask").replace(
            "financial_portfolio_staged", "financial_portfolio_training").replace(
            "portfolio-staged-v1", "portfolio-training-workflow-v1")
        self.assertEqual(package.joinpath("workflow.py").read_text(), expected)
        native = files("verifiers").joinpath("v1/harnesses/utils/core.py").read_text()
        self.assertEqual(package.joinpath("_chat_core.py").read_text(),
                         native.replace("    client = AsyncOpenAI(\n", "    client = AsyncOpenAI(\n        max_retries=0,\n"))

    def test_original_and_ported_checkpoint_validation_agree(self):
        original = ROOT / "environments/financial_portfolio_staged/src/financial_portfolio_staged/validation.py"
        spec = importlib.util.spec_from_file_location("frozen_public_validation", original)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "synthetic-fixture-host-key-0000000000000000"}):
            for seed in range(4, 8):
                episode = build_episode(seed, split="train")
                public = episode["public_files"]
                policy = reference_submission(episode)
                payloads = {"/workspace/research.py": files("financial_portfolio").joinpath("reference.py").read_bytes(),
                            "/workspace/ingestion.json": json.dumps(validation.inventory_for_public_files(public)).encode(),
                            "/workspace/trials.json": json.dumps(policy["trials"]).encode(),
                            "/workspace/policy.json": json.dumps(policy).encode()}
                self.assertEqual(validation.stage_contract(public), frozen.stage_contract(public))
                for stage in validation.STAGES:
                    self.assertEqual(validation.validate_stage(stage, public, payloads), frozen.validate_stage(stage, public, payloads))
                payloads["/workspace/trials.json"] = b'{"invalid": true}'
                self.assertEqual(validation.validate_stage("submission", public, payloads), frozen.validate_stage("submission", public, payloads))

    def test_installed_financial_dependency_mutation_fails_source_gate(self):
        from financial_portfolio_training import environment
        actual_files = environment.files
        for package, filename in (("financial_portfolio", "analytics.py"),
                                  ("financial_research", "reproduction.py")):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / package
                shutil.copytree(actual_files(package), copy)
                (copy / filename).write_text("# changed installed dependency\n")
                with patch.object(environment, "files", side_effect=lambda name: copy if name == package else actual_files(name)):
                    with self.assertRaisesRegex(RuntimeError, "Financial dependency differs"):
                        verify_native_sources()
