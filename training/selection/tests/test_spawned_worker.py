"""Opt-in, serial native spawned-worker/Docker protocol tests; no learned model."""

import asyncio
import base64
from contextlib import suppress
from importlib.resources import files
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.configs.client import TrainClientConfig
from prime_rl.env.client import EnvClient
from prime_rl.env.encoding import pack
from prime_rl.env.worker import EnvWorker, set_episode_hook
from prime_rl.transports.payload import PayloadHook, StoreWriter, _decoder
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY

from financial_selection_training.journal import INFO_KEY
from financial_selection_training.environment import FinancialSelectionTrainingEnvConfig
from financial_selection_training.task import FinancialSelectionTrainingTaskset
from scripted_train_provider import ScriptedTrainProvider, make_tokenizer, tool
from test_contract import configuration


def _spawn_worker(raw_config, directory, ready, stop):
    """Runs in an actual spawned interpreter, using installed packages only."""
    root = Path(directory)
    with (root / "worker.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        async def main():
            worker = EnvWorker(FinancialSelectionTrainingEnvConfig.model_validate(raw_config),
                               "tcp://127.0.0.1:0", max_concurrent=1)
            store = root / "payloads"
            (store / "live").mkdir(parents=True)
            set_episode_hook(PayloadHook(store=StoreWriter(store, "http://127.0.0.1:1")))
            run = asyncio.create_task(worker.run())
            ready.put({"address": worker.address, "pid": os.getpid()})
            try:
                while not stop.is_set() and not run.done():
                    await asyncio.sleep(0.05)
                if run.done():
                    await run
            finally:
                run.cancel()
                with suppress(asyncio.CancelledError):
                    await run
                set_episode_hook(None)
        asyncio.run(main())


def _trainable(episode):
    source = files("prime_rl").joinpath("orchestrator/algo/base.py")
    spec = importlib.util.spec_from_file_location("fixture_native_admission", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.iter_trainable_traces([episode]))


def reference_schedule(public_files, *, corrupt=False):
    source = base64.b64encode(files("financial_selection").joinpath("reference.py").read_bytes()).decode()
    program = ("import base64,json,runpy\n"
               f"open('/workspace/research.py','wb').write(base64.b64decode({source!r}))\n"
               "runpy.run_path('/workspace/research.py',run_name='__main__')\n")
    if corrupt:
        program += "policy=json.load(open('/workspace/policy.json'))\npolicy['input_sha256']='0'*64\njson.dump(policy,open('/workspace/policy.json','w'))\n"
    return [tool("python - <<'PY'\n" + program + "PY"), "Done.<|im_end|>"]


@unittest.skipUnless(os.environ.get("RUN_SELECTION_TRAINING_DOCKER_TESTS") == "1",
                     "Enable RUN_SELECTION_TRAINING_DOCKER_TESTS=1 for serial native Docker fixtures")
class SpawnedWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self, mode):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "FINANCIAL_SELECTION_KEY": secrets.token_hex(32), "FIXTURE_TOKEN": "synthetic-local-token",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}):
            root = Path(directory)
            image = subprocess.check_output(["docker", "image", "inspect", os.environ.get("SELECTION_TRAINING_DOCKER_IMAGE", "python:3.12-slim"), "--format", "{{.Id}}"], text=True).strip()
            config = configuration(root / "journal").orchestrator.train.source[0].env
            config.agent.runtime.image = config.taskset.task.image = image
            config.agent.timeout.setup = 180
            config.agent.timeout.rollout = 3 if mode == "timeout" else 120
            config.agent.timeout.scoring = 180
            config.agent.timeout.finalize = 30
            task = next(iter(FinancialSelectionTrainingTaskset(config.taskset).load()))
            schedules = [[tool("sleep 30")]] if mode == "timeout" else [
                reference_schedule(task.data.public_files), reference_schedule(task.data.public_files, corrupt=True)]
            tokenizer = make_tokenizer(root / "tokenizer")
            context = multiprocessing.get_context("spawn")
            ready, stop = context.Queue(), context.Event()
            worker = context.Process(target=_spawn_worker, args=(config.model_dump(mode="json"), str(root), ready, stop))
            worker.start()
            client = None
            episodes, summaries = [], []
            try:
                startup = await asyncio.to_thread(ready.get, True, 120)
                self.assertNotEqual(startup["pid"], os.getpid())
                client = EnvClient(startup["address"])
                await client.wait_for_pool_startup(timeout=45)
                for schedule in schedules:
                    with ScriptedTrainProvider(tokenizer, schedule) as endpoint:
                        episode = await asyncio.wait_for(client.run(
                            TrainClientConfig(base_url=endpoint.url, api_key_var="FIXTURE_TOKEN",
                                renderer={"name": "qwen3"}, renderer_model_name=str(root / "tokenizer")),
                            "fixture-policy", vf.Sampling(temperature=1, top_p=1, max_tokens=65536),
                            task.data.model_dump(mode="json")), timeout=360)
                        episodes.append(episode)
                        trace = episode.traces[0]
                        (root / f"episode-{len(episodes)}.msgpack").write_bytes(pack(episode.model_dump(mode="python")))
                        sampled = [int(token) for node in trace.nodes for token, flag in zip(node.token_ids, node.mask) if flag]
                        expected = [token for completion in endpoint.completions for token in completion]
                        self.assertEqual(sampled, expected)
                        self.assertTrue(all(float(value) == -0.25 for node in trace.nodes for value in node.logprobs))
                        self.assertTrue(any(node.weight_versions for node in trace.nodes))
                        references = trace.info.get(PAYLOAD_INFO_KEY, {}).get("branches", {})
                        self.assertTrue(references)
                        for reference in references.values():
                            stored = _decoder.decode((root / "payloads/live" / reference["id"]).read_bytes())
                            np.testing.assert_array_equal(np.frombuffer(stored.sampler_support.ids, dtype=np.int32), sampled)
                            self.assertEqual(int(np.frombuffer(stored.sampler_support.counts, dtype=np.int32).sum()), len(sampled))
                        summaries.append({"episode_ok": episode.ok, "reward": trace.reward,
                            "dropped": trace.is_dropped, "admission": trace.info.get(INFO_KEY, {}).get("admission"),
                            "trainable_count": len(_trainable(episode)), "provider_calls": len(endpoint.calls),
                            "sampled_tokens": len(sampled),
                            "outcome": trace.info.get("financial_selection_outcome")})
                (root / "summary.json").write_text(json.dumps(summaries, indent=2))
            finally:
                if client:
                    await client.close()
                stop.set()
                await asyncio.to_thread(worker.join, 20)
                if worker.is_alive():
                    worker.terminate()
                    await asyncio.to_thread(worker.join, 10)
                evidence = os.environ.get("SELECTION_TRAINING_FIXTURE_EVIDENCE_DIR")
                if evidence:
                    destination = Path(evidence) / self._testMethodName
                    shutil.copytree(root, destination)
                log = root / "worker.log"
                self.assertEqual(worker.exitcode, 0, log.read_text()[-10000:] if log.exists() else "Worker did not finish importing")
            rows = [json.loads(line) for path in (root / "journal").glob("*/events.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "attempt_terminal" for row in rows), len(episodes))
            secret = os.environ["FINANCIAL_SELECTION_KEY"].encode()
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret, path.read_bytes(), "Worker key found in persisted output")
            return episodes, summaries, rows

    async def test_same_task_positive_and_candidate_zero_survive_native_transport(self):
        episodes, summary, rows = await self.fixture("pair")
        self.assertEqual([row["provider_calls"] for row in summary], [2, 2])
        self.assertEqual([row["trainable_count"] for row in summary], [1, 1], summary)
        self.assertEqual([row["dropped"] for row in summary], [False, False], summary)
        self.assertGreater(summary[0]["reward"], 0)
        self.assertEqual(summary[1]["reward"], 0)
        self.assertEqual(episodes[0].task.data.id, episodes[1].task.data.id)
        self.assertNotEqual(episodes[0].traces[0].id, episodes[1].traces[0].id)
        self.assertEqual(len({row["worker_id"] for row in rows}), 1)
        self.assertTrue(all(row.get("valid", True) for row in rows if row["event"] == "prompt_checked"))

    async def test_tool_timeout_is_dropped_before_native_training_admission(self):
        episodes, summary, rows = await self.fixture("timeout")
        self.assertEqual(summary[0]["provider_calls"], 1)
        self.assertEqual(summary[0]["trainable_count"], 0, summary)
        self.assertTrue(summary[0]["dropped"], summary)
        self.assertFalse(summary[0]["admission"]["admitted"], summary)
        self.assertTrue(any(row["event"] == "tool_gate_terminal" and row["status"] == "allowed" for row in rows))
