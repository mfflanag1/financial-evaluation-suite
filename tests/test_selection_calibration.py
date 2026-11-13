"""Five planned attempts, exact native bindings, and honest missingness controls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import selection_calibration as cal


IMAGE = "sha256:" + "a" * 64
MODEL = "fixture-instruct"
ENDPOINT = "https://endpoint.invalid/v1"
KEY = "b7" * 32  # Public fixture only; never a private evaluation key.
NATIVE = importlib.util.find_spec("verifiers") is not None and sys.version_info >= (3, 12)


def scored_trace(task, agent, index, *, failure=False, value=0.8):
    outcome = dict(status="candidate_failure" if failure else "scored", reason="fixture",
                   reproduction_verified=not failure)
    return dict(id=f"trace-{index}", task=task, agent=dict(config=agent,
                runtime=dict(type="docker", image=IMAGE, allow=[], workdir="/workspace",
                             borrowed=False, id=f"runtime-{index}")),
                run=dict(type="eval", id=f"run-{index}", name=f"run-{index}"),
                verifiers=dict(version="0.4.1"), ok=True, is_completed=True,
                errors=[], is_timeout=False, info={cal.INFO_KEY: outcome,
                "scoring_diagnostics": {cal.INFO_KEY: copy.deepcopy(outcome)}},
                nodes=[dict(parent=None, sampled=False, message=dict(role="system", content="fixture-system")),
                       dict(parent=0, sampled=False, message=dict(role="user", content="fixture-user")),
                       dict(parent=1, sampled=True, message=dict(role="assistant", content="done"))],
                calls=[dict(model=MODEL, endpoint="/chat/completions", node=2,
                            sampling=dict(temperature=0.2, max_tokens=8192))],
                rewards={"selection_reward": dict(score=0 if failure else value, weight=1.0)},
                metrics=dict(candidate_failure=int(failure), reproduction_verified=int(not failure),
                             packets_scored=0 if failure else 3, exact_success_fraction=0,
                             all_packets_correct=0, current_exact_success=0,
                             fresh_exact_success_fraction=0))


def bundle(directory):
    attempts = []
    for index, condition in enumerate(cal.CONDITIONS):
        task = dict(type="FinancialSelectionTask", data=dict(id=f"task-{index}"))
        agent = dict(model=MODEL, client=dict(base_url=ENDPOINT), runtime=dict(image=IMAGE))
        trace = scored_trace(task, agent, index, failure=index == 0)
        record = dict(id=f"episode-{index}", ok=True, is_timeout=False, errors=[],
                      task=task, run=trace["run"], traces=[trace])
        config_text = f'condition = "{condition}"\n'
        resolved = dict(fixture=condition)
        (directory / (condition + ".toml")).write_text(config_text)
        result = directory / "results" / condition
        (result / "configs/resolved").mkdir(parents=True)
        (result / "configs/resolved/run.json").write_bytes(cal.canonical(resolved))
        (result / "traces.jsonl").write_bytes(cal.canonical(record) + b"\n")
        attempts.append(dict(attempt_id=condition + "-1", condition=condition, split="train", index=0,
                             task_id=task["data"]["id"], task_sha256=cal.object_digest(task),
                             config_sha256=cal.digest(config_text.encode()),
                             resolved_config_sha256=cal.object_digest(resolved),
                             agent_sha256=cal.object_digest(agent), run_name=trace["run"]["name"],
                             result_dir=condition))
    binding = dict(schema=1, protocol_id="selection-calibration-v1", manifest_sha256="fixture-manifest",
                   model_class="instruct", model=MODEL, endpoint=ENDPOINT, image=IMAGE,
                   expected_attempts=5, split="train", final_selected=False, retries=0,
                   task_prompt_sha256=cal.digest(b"fixture-user"),
                   harness_system_prompt_sha256=cal.digest(b"fixture-system"),
                   automatic_expansion=False, attempts=attempts)
    (directory / "binding.json").write_bytes(cal.canonical(binding))
    return cal.digest(cal.read_bytes(directory / "binding.json"))


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.hash = bundle(self.directory)
        self.freeze = patch.object(cal, "verify_manifest", return_value="fixture-manifest")
        self.freeze.start()
        self.addCleanup(self.freeze.stop)

    def report(self):
        return cal.report(self.directory, binding_sha256=self.hash)

    def mutate(self, function, condition="positive"):
        path = self.directory / "results" / condition / "traces.jsonl"
        record = json.loads(path.read_text())
        function(record)
        path.write_bytes(cal.canonical(record) + b"\n")

    def test_candidate_zero_is_retained_in_five_attempt_denominator(self):
        result = self.report()
        self.assertTrue(result["accounting_complete"])
        self.assertEqual((result["planned_attempts"], result["resolved_attempts"]), (5, 5))
        self.assertAlmostEqual(result["mean_reward"], .64)
        self.assertEqual(result["candidate_failure_count"], 1)
        self.assertEqual(result["attempts"][0]["reward"], 0)
        self.assertIsNone(result["components"])

    def test_valid_scored_zero_is_also_retained(self):
        self.mutate(lambda r: r["traces"][0]["rewards"]["selection_reward"].update(score=0))
        result = self.report()
        self.assertTrue(result["accounting_complete"])
        self.assertAlmostEqual(result["mean_reward"], .48)

    def test_missing_attempt_blocks_mean_without_imputing_zero(self):
        (self.directory / "results/positive/traces.jsonl").unlink()
        result = self.report()
        self.assertFalse(result["accounting_complete"])
        self.assertEqual(result["unresolved_attempts"], 1)
        self.assertIsNone(result["mean_reward"])
        self.assertIsNone(result["attempts"][1]["reward"])

    def test_duplicate_attempt_blocks_aggregate(self):
        path = self.directory / "results/positive/traces.jsonl"
        path.write_bytes(path.read_bytes() * 2)
        self.assertFalse(self.report()["accounting_complete"])

    def test_extra_result_directory_or_trace_file_blocks_aggregate(self):
        extra = self.directory / "results/repair"
        extra.mkdir()
        self.assertIn("unplanned_result_directory", self.report()["issues"])
        extra.rmdir()
        (self.directory / "results/positive/traces-retry.jsonl").write_text("{}\n")
        self.assertIn("unplanned_trace_file", self.report()["issues"])

    def test_unresolved_native_outcomes_are_not_financial_zeros(self):
        mutations = [lambda r: r.update(is_timeout=True), lambda r: r.update(errors=[{}]),
                     lambda r: r["traces"][0].update(is_completed=False),
                     lambda r: r["traces"][0].update(dropped="infrastructure"),
                     lambda r: r["traces"][0]["info"][cal.INFO_KEY].update(status="infrastructure_error")]
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                path.write_bytes(original)
                self.mutate(mutate)
                result = self.report()
                self.assertIsNone(result["mean_reward"])
                self.assertIsNone(result["attempts"][1]["reward"])

    def test_task_agent_model_prompt_config_and_identity_mutations_fail(self):
        mutations = [lambda r: r["task"]["data"].update(id="other"),
                     lambda r: r["traces"][0]["task"]["data"].update(prompt="injected"),
                     lambda r: r["traces"][0]["agent"]["config"].update(model="other"),
                     lambda r: r["traces"][0]["calls"][0].update(model="other"),
                     lambda r: r["traces"][0]["calls"][0].update(endpoint="/responses"),
                     lambda r: r["traces"][0]["calls"][0]["sampling"].update(temperature=1),
                     lambda r: r["traces"][0]["agent"]["runtime"].update(image="sha256:"+"b"*64),
                     lambda r: r["traces"][0]["agent"]["runtime"].update(borrowed=True),
                     lambda r: r["traces"][0]["agent"]["runtime"].update(allow=["*"]),
                     lambda r: r["traces"][0]["agent"]["runtime"].update(id="runtime-0"),
                     lambda r: r["traces"][0].update(run=dict(type="eval", id="wrong", name="run-1")),
                     lambda r: r.update(id="episode-0")]
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                path.write_bytes(original)
                self.mutate(mutate)
                self.assertFalse(self.report()["accounting_complete"])
        path.write_bytes(original)
        (self.directory / "results/positive/configs/resolved/run.json").write_text('{}')
        self.assertFalse(self.report()["accounting_complete"])

    def test_impossible_success_and_boolean_metrics_fail(self):
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        for updates in [dict(exact_success_fraction=1), dict(reproduction_verified=True),
                        dict(packets_scored=2), dict(current_exact_success=0.5)]:
            path.write_bytes(original)
            self.mutate(lambda r: r["traces"][0]["metrics"].update(updates))
            self.assertFalse(self.report()["accounting_complete"])
        path.write_bytes(original)
        self.mutate(lambda r: r["traces"][0]["rewards"].update(extra=dict(score=0, weight=1)))
        self.assertFalse(self.report()["accounting_complete"])

    def test_extreme_numeric_fields_remain_unresolved_instead_of_crashing(self):
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        for mutate in (lambda r: r["traces"][0]["rewards"]["selection_reward"].update(score=10**1000),
                       lambda r: r["traces"][0]["metrics"].update(exact_success_fraction=10**1000)):
            path.write_bytes(original)
            self.mutate(mutate)
            result = self.report()
            self.assertEqual(result["planned_attempts"], 5)
            self.assertEqual(result["unresolved_attempts"], 1)
            self.assertIsNone(result["mean_reward"])

    def test_strict_json_and_symlink_rejection(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}', b'\xff'):
            with self.assertRaises(cal.CalibrationError):
                cal.loads(payload)
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        path.unlink()
        target = self.directory / "foreign"
        target.write_bytes(original)
        path.symlink_to(target)
        self.assertFalse(self.report()["accounting_complete"])

    def test_prelaunch_digest_anchor_rejects_changed_binding(self):
        with self.assertRaises(cal.CalibrationError):
            cal.report(self.directory, binding_sha256="0" * 64)
        path = self.directory / "binding.json"
        binding = json.loads(path.read_text())
        binding["attempts"][0]["split"] = "final"
        path.write_bytes(cal.canonical(binding))
        with self.assertRaises(cal.CalibrationError):
            cal.report(self.directory, binding_sha256=cal.digest(path.read_bytes()))

    def test_changed_prepared_toml_blocks_result(self):
        path = self.directory / "positive.toml"
        path.write_text(path.read_text() + "# change\n")
        self.assertFalse(self.report()["accounting_complete"])

    def test_outbound_prompt_and_actual_call_ancestry_are_bound(self):
        mutations = [lambda r: r["traces"][0]["nodes"][0]["message"].update(content="wrong"),
                     lambda r: r["traces"][0]["nodes"][1]["message"].update(content="wrong"),
                     lambda r: r["traces"][0]["nodes"][2].update(parent=0),
                     lambda r: r["traces"][0]["nodes"][1].update(parent=2),
                     lambda r: r["traces"][0]["calls"][0].update(node=1),
                     lambda r: r["traces"][0]["nodes"].append(dict(parent=2, sampled=True,
                               message=dict(role="assistant", content="unaccounted")))]
        path = self.directory / "results/positive/traces.jsonl"
        original = path.read_bytes()
        for mutate in mutations:
            path.write_bytes(original)
            self.mutate(mutate)
            self.assertFalse(self.report()["accounting_complete"])


class FrozenProtocolTests(unittest.TestCase):
    def test_frozen_payload_binds_config_schedule_sources_and_tests(self):
        self.assertEqual(cal.read_json(cal.MANIFEST), cal.build_manifest())
        files = cal.build_manifest()["sha256"]
        self.assertIn("scripts/selection_calibration.py", files)
        self.assertIn("tests/test_selection_calibration.py", files)
        self.assertIn("environments/financial_selection/src/financial_selection/prime_adapter.py", files)
        self.assertIn("environments/financial_portfolio_staged/src/financial_portfolio_staged/_chat_core.py", files)

    def test_five_exact_train_configs_instruct_offline_no_retries(self):
        schedule = cal.read_json(cal.CONFIG / "schedule.json")
        self.assertEqual(schedule["conditions"], list(cal.CONDITIONS))
        self.assertEqual(schedule["expected_attempts"], 5)
        for condition in cal.CONDITIONS:
            config = tomllib.loads((cal.CONFIG / (condition + ".toml")).read_text())
            self.assertTrue(config["dry_run"])
            self.assertEqual(config["model"], "INSTRUCT_MODEL_ID")
            self.assertEqual(config["client"]["base_url"], ENDPOINT)
            self.assertEqual(config["num_rollouts"], 1)
            self.assertEqual(config["max_concurrent"], 1)
            self.assertEqual(config["env"]["taskset"]["split"], "train")
            self.assertEqual(config["env"]["taskset"]["condition"], condition)
            self.assertEqual(config["env"]["taskset"]["index_start"], 0)
            self.assertEqual(config["env"]["taskset"]["num_tasks"], 1)
            self.assertFalse(config["env"]["agent"]["network_notice"])
            self.assertEqual(config["env"]["retries"]["max_retries"], 0)
            self.assertEqual(config["env"]["agent"]["retries"]["max_retries"], 0)
            self.assertEqual(config["env"]["agent"]["scoring"]["retries"]["max_retries"], 0)
            self.assertEqual(config["env"]["agent"]["harness"]["sdk_max_retries"], 0)
            self.assertEqual(config["env"]["agent"]["runtime"]["allow"], [])

    def test_freeze_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            cal.main(["freeze", "--write"])


@unittest.skipUnless(NATIVE, "Install pinned native dependency")
class NativePreparationTests(unittest.TestCase):
    def clean_process(self):
        # Other repository tests import editable source paths. Validate this
        # installed boundary in an isolated interpreter, not their module cache.
        if os.environ.get("SELECTION_CALIBRATION_FIXTURE_CHILD") == "1":
            return False
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                                     "NativePreparationTests." + self._testMethodName],
                                    env={**os.environ, "SELECTION_CALIBRATION_FIXTURE_CHILD": "1"},
                                    cwd=temporary, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def test_installed_sources_and_actual_cli_dry_runs_make_no_model_call(self):
        if self.clean_process():
            return
        from verifiers.v1.cli.run import main
        from verifiers.v1.configs.cli.run import RunnerConfig
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"FINANCIAL_SELECTION_KEY": KEY}):
            destination = Path(temporary).resolve() / "prepared"
            prepared = cal.prepare(destination, model=MODEL, endpoint=ENDPOINT, image=IMAGE)
            self.assertEqual(prepared["model_calls"], 0)
            binding = cal.read_json(destination / "binding.json")
            self.assertNotIn(KEY, json.dumps(binding))
            self.assertEqual(len({row["task_id"] for row in binding["attempts"]}), 5)
            for condition in cal.CONDITIONS:
                with patch("verifiers.v1.cli.run.run_episodes", side_effect=AssertionError("inference forbidden")):
                    main(["@", str(destination / (condition + ".toml"))])
                resolved = cal.read_json(destination / "results" / condition / "configs/resolved/run.json")
                row = next(row for row in binding["attempts"] if row["condition"] == condition)
                self.assertEqual(cal.object_digest(resolved), row["resolved_config_sha256"])
                self.assertFalse((destination / "results" / condition / "traces.jsonl").exists())
                offline = RunnerConfig.model_validate(tomllib.loads((destination / (condition + ".toml")).read_text()))
                live = offline.model_copy(update={"dry_run": False})
                self.assertTrue(offline.dry_run)
                self.assertFalse(live.dry_run)
                self.assertEqual(offline.model_dump(mode="json"), live.model_dump(mode="json"))
            result = cal.report(destination, binding_sha256=prepared["binding_sha256"])
            self.assertEqual(result["unresolved_attempts"], 5)
            self.assertIsNone(result["mean_reward"])
            with self.assertRaises(cal.CalibrationError):
                cal.prepare(destination, model=MODEL, endpoint=ENDPOINT, image=IMAGE)

    def test_actual_native_wire_record_joins_prepared_plan(self):
        if self.clean_process():
            return
        from verifiers.v1.configs.cli.run import RunnerConfig
        from verifiers.v1.configs.agent import resolve_agent
        from verifiers.v1.utils.loaders import load_taskset
        from verifiers.v1.trace import AgentInfo, EvalRunInfo, TraceTask
        from verifiers.v1.episode import Episode
        import verifiers.v1 as vf
        from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"FINANCIAL_SELECTION_KEY": KEY}):
            directory = Path(temporary).resolve() / "prepared"
            prepared = cal.prepare(directory, model=MODEL, endpoint=ENDPOINT, image=IMAGE)
            binding = cal.read_json(directory / "binding.json")
            for index, row in enumerate(binding["attempts"]):
                config = RunnerConfig.model_validate(tomllib.loads((directory / (row["condition"] + ".toml")).read_text()))
                task = next(iter(load_taskset(config.env.taskset)))
                agent = resolve_agent(config.env.agent, model=config.model, client=config.client, sampling=config.sampling)
                task_wire = TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash)
                fixture = scored_trace(task_wire.model_dump(mode="json", exclude_none=True),
                                       agent.model_dump(mode="json", exclude_none=True), index, failure=index == 0)
                fixture["run"]["name"] = row["run_name"]
                fixture["nodes"][0]["message"]["content"] = BASH_SYSTEM_PROMPT + " " + EDIT_SYSTEM_PROMPT
                fixture["nodes"][1]["message"]["content"] = task.data.prompt
                trace = vf.Trace(task=task_wire, agent=AgentInfo(config=agent, runtime=fixture["agent"]["runtime"]), **{key: value for key, value in fixture.items()
                              if key not in {"task", "agent"}})
                episode = Episode(task=task_wire, ok=True, traces=[trace])
                episode.record_run(EvalRunInfo(id=f"fixture-{index}", name=row["run_name"]))
                result = directory / "results" / row["condition"]
                (result / "configs/resolved").mkdir(parents=True)
                (result / "configs/resolved/run.json").write_text(config.model_dump_json())
                (result / "traces.jsonl").write_bytes(episode.to_json() + b"\n")
            result = cal.report(directory, binding_sha256=prepared["binding_sha256"])
            self.assertTrue(result["accounting_complete"], result)
            self.assertAlmostEqual(result["mean_reward"], .64)


if __name__ == "__main__":
    unittest.main()
