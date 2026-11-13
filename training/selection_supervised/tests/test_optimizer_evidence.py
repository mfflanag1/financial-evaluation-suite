"""Synthetic consistency fixtures only: these tests do not execute an optimizer."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/financial_selection_supervised/optimizer_evidence.py"
SPEC = importlib.util.spec_from_file_location("optimizer_evidence_under_test", SOURCE)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class Bundle:
    """Construct explicitly labelled fixture records, never an observed GPU run."""

    def __init__(self, root):
        self.root = root
        self.artifacts = {}
        algo = {"type": "grpo", "length_penalty": None, "length_weighted_baseline": False,
                "loss_aggregation": "token"}
        self.config = {"orchestrator": {"train": {"algo": algo, "source": [{"algo": algo}]}},
                       "trainer": {"data": {"fake": None}, "resume": None, "model": {
                           "init_from": None, "name": "fixture-policy", "lora": None}, "max_steps": 1,
                           "optim": {"max_norm": 1}, "ckpt": {"skip_optimizer": False}}}
        self.put("config.json", self.config)
        self.put("sources.json", {"native_sources": evidence.PINNED_SOURCES})
        self.put("collector.py", b"# synthetic fixture collector, never executed\n")
        self.plan = {"schema": evidence.SCHEMA, "run_id": "fixture-run-unique", "model_id": "fixture-policy",
                     "model_path": "fixture-policy",
                     "model_revision": "fixture-revision", "update_scope": "full",
                     "initial_weights_sha256": evidence.digest("initial"),
                     "prime_rl_commit": evidence.PRIME_RL_COMMIT, "world_size": 1, "step": 1,
                     "task_sha256": evidence.digest({"id": "same-task"}), **{
                         key + "_sha256": self.artifacts[name] for key, name in (
                             ("config", "config.json"), ("sources", "sources.json"), ("collector", "collector.py"))}}
        self.rows = {}
        for number, stage in enumerate(("runtime", "cohort", "before", "after", "checkpoint", "reload",
                                        "receiver", "inference"), 1):
            self.rows[stage] = {"stage": stage, "run_id": self.plan["run_id"], "kind": "fixture",
                                "binding_sha256": evidence.digest(self.plan), "monotonic_ns": number * 10}
        self.rows["runtime"].update(backend="cuda", world_size=1, device_uuid="fixture-device",
                                   cuda_version="fixture", torch_version="fixture", trainer_process_id="trainer-1",
                                   **{key: self.plan[key] for key in ("model_id", "model_path", "model_revision", "initial_weights_sha256")})
        members = []
        for i, reward in enumerate((0.0, 1.0)):
            trace, journal = f"raw/trace-{i}.json", f"raw/journal-{i}.json"
            self.put(trace, {"id": f"trace-{i}", "ok": True, "is_completed": True, "is_timeout": False,
                             "dropped": None, "task": {"data": {"id": "same-task"}},
                             "info": {"selection_training": {"admission": {"admitted": True}}},
                             "calls": [{"node": 0, "error": None}], "nodes": [{
                                 "sampled": True, "token_ids": [10, 11], "mask": [True, True],
                                 "logprobs": [evidence._f32(-.1), evidence._f32(-.2)], "advantages": [reward - .5] * 2,
                                 "weight_versions": [{"start": 0, "end": 2, "version": "0"}]}],
                             "rewards": {"financial": {"value": reward}}})
            self.put(journal, [{"trace_id": f"trace-{i}", "event": "attempt_started",
                                "task_sha256": self.plan["task_sha256"]},
                               {"trace_id": f"trace-{i}", "event": "scoring_complete", "reward": reward},
                               {"trace_id": f"trace-{i}", "event": "attempt_terminal", "admitted": True}])
            member = self.sample(i, 0)
            response_path = f"raw/response-{i}.json"
            self.put(response_path, {"trace_id": f"trace-{i}", "provider_request_id": f"request-{i}",
                                     "prompt_sha256": member["prompt_sha256"], "model": "fixture-policy",
                                     "monotonic_ns": 11 + i, **{key: member[key] for key in
                                         ("token_ids", "logprobs", "weight_versions")}})
            member.update(trace_id=f"trace-{i}", task_sha256=self.plan["task_sha256"],
                          admission="admitted", outcome="resolved", reward=reward, advantage=reward - .5,
                          token_advantages=[reward - .5] * 2,
                          native_trace=trace, native_trace_sha256=self.artifacts[trace],
                          admission_journal=journal, admission_journal_sha256=self.artifacts[journal],
                          provider_responses=[response_path])
            members.append(member)
        self.rows["cohort"].update(cohort_id="cohort-1", task_sha256=self.plan["task_sha256"], members=members)
        batch = [{field: member[field] for field in
                  ("trace_id", "token_ids", "logprobs", "token_advantages")} for member in members]
        self.put("raw/trainer-batch.json", batch)
        for stage, changed in (("before", False), ("after", True)):
            progress = {"step": 1, "total_tokens": 4 if changed else 0, "total_samples": 2 if changed else 0}
            state = {"model": {"weight": evidence.digest("post" if changed else "pre")},
                     "optimizers": {"weight/step": evidence.digest(1 if changed else 0)},
                     "scheduler": {"last_epoch": evidence.digest(1 if changed else 0)},
                     "progress": {key: evidence.digest(value) for key, value in progress.items()}}
            self.rows[stage].update(state=state, optimizer_steps={"weight": 1 if changed else 0}, progress=progress,
                                    cohort_id="cohort-1", step=1, trace_ids=["trace-0", "trace-1"],
                                    batch_sampled_tokens=4, batch_micro_batches=2, batch_sha256=evidence.digest(batch))
        self.rows["after"].update(gradient={"l2_norm": .3, "total_elements": 1, "finite_elements": 1},
                                  parameter_delta_l2=.01, loss=.2, optimizer_step_monotonic_ns=35)
        checkpoint_files = {}
        for path in ("trainer/.metadata", "trainer/__0_0.distcp", "orchestrator/progress.pt"):
            name = "checkpoints/step_1/" + path
            self.put(name, b"synthetic non-checkpoint bytes: checker never unpickles these")
            checkpoint_files[name] = self.artifacts[name]
        self.rows["checkpoint"].update(step=1, state=copy.deepcopy(self.rows["after"]["state"]),
                                        files=checkpoint_files, orchestrator_state_sha256=evidence.digest("orch"))
        self.rows["reload"].update(process_id="reload-2", method="CheckpointManager.load_from_path",
                                   checkpoint_files=checkpoint_files, optimizer_updates=0,
                                   optimizer_initialization_calls=0, optimizer_initialization=[],
                                   orchestrator_state_sha256=evidence.digest("orch"), **{
                                       key: copy.deepcopy(self.rows["after"][key]) for key in
                                       ("state", "optimizer_steps", "progress")})
        self.rows["receiver"].update(hook="WeightWatcher.on_update", applied_version=1,
                                     model_state_sha256=evidence.digest(self.rows["after"]["state"]["model"]),
                                     receiver_id="receiver-1")
        self.rows["inference"].update(**self.sample(2, 1), receiver_id="receiver-1", model_id="fixture-policy")
        inference = self.rows["inference"]
        self.put("raw/post-update-response.json", {
            "trace_id": "post-update-probe", "provider_request_id": "request-2",
            "model": "fixture-policy", "monotonic_ns": 75,
            **{field: inference[field] for field in ("prompt_sha256", "token_ids", "logprobs", "weight_versions")}})
        inference["provider_responses"] = ["raw/post-update-response.json"]

    @staticmethod
    def sample(index, version):
        return {"token_ids": [10, 11], "logprobs": [evidence._f32(-.1), evidence._f32(-.2)], "weight_versions": [version, version],
                "provider_request_ids": [f"request-{index}"], "prompt_sha256": evidence.digest("prompt"),
                "origin": "policy"}

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else json.dumps(value, allow_nan=True).encode()
        path.write_bytes(data)
        self.artifacts[name] = hashlib.sha256(data).hexdigest()

    def write(self):
        for stage, row in self.rows.items():
            self.put(f"records/{stage}.json", row)
        bundle = {"schema": evidence.SCHEMA, "kind": "fixture", "binding": self.plan,
                  "config": "config.json", "sources": "sources.json", "collector": "collector.py",
                  "artifacts": self.artifacts,
                  "records": {stage: f"records/{stage}.json" for stage in self.rows}}
        (self.root / "bundle.json").write_text(json.dumps(bundle))
        (self.root / "expected.json").write_text(json.dumps(self.plan))
        return self.root / "bundle.json", self.root / "expected.json"

    def inspect(self):
        return evidence.inspect_bundle(*self.write())


class OptimizerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Bundle(Path(self.temp.name))

    def reject(self):
        result = self.bundle.inspect()
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(result["errors"])
        self.assertFalse(result["gpu_execution_independently_verified"])

    def test_complete_fixture_cannot_claim_gpu_and_cli_is_non_success(self):
        result = self.bundle.inspect()
        self.assertEqual(result["status"], "fixture_consistent", result)
        self.assertFalse(result["gpu_execution_independently_verified"])
        self.assertFalse(result["training_readiness_claim"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evidence.main([str(self.bundle.root / "bundle.json"), "--expected-plan",
                                            str(self.bundle.root / "expected.json")]), 3)

    def test_unresolved_or_model_free_members_fail(self):
        for field, value in (("admission", "dropped"), ("outcome", "unresolved"), ("origin", "reference"),
                             ("task_sha256", evidence.digest("other"))):
            with self.subTest(field=field):
                original = self.bundle.rows["cohort"]["members"][0][field]
                self.bundle.rows["cohort"]["members"][0][field] = value
                self.reject()
                self.bundle.rows["cohort"]["members"][0][field] = original

    def test_equal_rewards_and_incorrect_std_normalization_fail(self):
        members = self.bundle.rows["cohort"]["members"]
        members[0]["advantage"], members[1]["advantage"] = -1, 1
        for member in members:
            member["token_advantages"] = [member["advantage"]] * 2
            path = member["native_trace"]
            trace = json.loads((self.bundle.root / path).read_text())
            trace["nodes"][0]["advantages"] = member["token_advantages"]
            self.bundle.put(path, trace)
            member["native_trace_sha256"] = self.bundle.artifacts[path]
        self.reject()
        members[0]["reward"] = 1
        self.reject()

    def test_duplicate_trace_request_and_stale_policy_fail(self):
        for field, value in (("trace_id", "trace-0"), ("provider_request_ids", ["request-0"]),
                             ("weight_versions", [0, 1])):
            with self.subTest(field=field):
                member = self.bundle.rows["cohort"]["members"][1]
                previous = member[field]
                member[field] = value
                self.reject()
                member[field] = previous

    def test_token_advantage_transport_and_batch_join_fail(self):
        self.bundle.rows["cohort"]["members"][0]["token_advantages"] = [-.5, 0]
        self.reject()
        self.bundle.rows["cohort"]["members"][0]["token_advantages"] = [-.5, -.5]
        self.bundle.rows["after"]["trace_ids"] = ["trace-1", "trace-0"]
        self.reject()

    def test_nonfinite_or_missing_gradient_loss_and_delta_fail(self):
        after = self.bundle.rows["after"]
        for value in (0, float("nan"), float("inf"), True, -1):
            with self.subTest(value=value):
                after["gradient"]["l2_norm"] = value
                self.reject()
        after["gradient"]["l2_norm"] = .3
        after["loss"] = float("inf")
        self.reject()
        after["loss"] = .2
        after["parameter_delta_l2"] = 0
        self.reject()
        del after["gradient"]
        self.reject()

    def test_step_and_parameter_deltas_required(self):
        self.bundle.rows["after"]["optimizer_steps"]["weight"] = 0
        self.reject()
        self.bundle.rows["after"]["optimizer_steps"]["weight"] = 1
        self.bundle.rows["after"]["state"]["model"] = self.bundle.rows["before"]["state"]["model"]
        self.reject()

    def test_missing_native_checkpoint_part_or_unlisted_file_fails(self):
        key = "checkpoints/step_1/orchestrator/progress.pt"
        del self.bundle.rows["checkpoint"]["files"][key]
        self.reject()
        self.bundle.rows["checkpoint"]["files"][key] = self.bundle.artifacts[key]
        (self.bundle.root / "checkpoints/step_1/unlisted").write_text("not inventoried")
        self.reject()

    def test_guarded_native_load_initialization_is_distinct_from_update(self):
        witness = {"method": "torch.distributed.checkpoint.state_dict._init_optim_state",
                   "learning_rates": [0.0], "optimizer_state_empty_before": True,
                   "gradient_total_elements": 1, "gradient_zero_finite_elements": 1,
                   "model_parameters_before": {"weight": evidence.digest("fresh-initial")},
                   "model_parameters_after": {"weight": evidence.digest("fresh-initial")},
                   "started_monotonic_ns": 51, "finished_monotonic_ns": 52}
        row = self.bundle.rows["reload"]
        row.update(optimizer_initialization_calls=1, optimizer_initialization=[witness])
        self.assertEqual(self.bundle.inspect()["status"], "fixture_consistent")
        for key, invalid in (("learning_rates", [1e-6]), ("optimizer_state_empty_before", False),
                             ("gradient_zero_finite_elements", 0),
                             ("model_parameters_after", {"weight": evidence.digest("changed")}),
                             ("finished_monotonic_ns", 70)):
            previous = witness[key]
            witness[key] = invalid
            self.reject()
            witness[key] = previous
        row["optimizer_initialization_calls"] = 2
        self.reject()
        row["optimizer_initialization_calls"] = True
        self.reject()

    def test_model_only_same_process_updated_or_wrong_reload_fails(self):
        row = self.bundle.rows["reload"]
        for key, value in (("method", "load_weights"), ("process_id", "trainer-1"), ("optimizer_updates", 1),
                           ("optimizer_updates", False), ("orchestrator_state_sha256", evidence.digest("wrong"))):
            with self.subTest(key=key, value=value):
                previous = row[key]
                row[key] = value
                self.reject()
                row[key] = previous
        row["state"]["scheduler"] = {"last_epoch": evidence.digest(99)}
        self.reject()

    def test_sender_finished_is_not_receiver_apply_or_inference(self):
        self.bundle.rows["receiver"]["hook"] = "WeightSender.finished"
        self.reject()
        self.bundle.rows["receiver"]["hook"] = "WeightWatcher.on_update"
        self.bundle.rows["inference"]["weight_versions"] = [0, 0]
        self.reject()

    def test_stale_stage_and_fixture_kind_mismatch_fail(self):
        self.bundle.rows["after"]["run_id"] = "old-run"
        self.reject()
        self.bundle.rows["after"]["run_id"] = self.bundle.plan["run_id"]
        self.bundle.rows["after"]["kind"] = "observed"
        self.reject()

    def test_artifact_tampering_duplicate_json_and_symlink_fail(self):
        bundle, plan = self.bundle.write()
        (self.bundle.root / "collector.py").write_text("tampered")
        self.assertEqual(evidence.inspect_bundle(bundle, plan)["status"], "rejected")
        self.bundle.put("collector.py", b"original")
        bundle, plan = self.bundle.write()
        plan.write_text('{"schema": "x", "schema": "y"}')
        self.assertEqual(evidence.inspect_bundle(bundle, plan)["status"], "rejected")
        bundle, plan = self.bundle.write()
        path = self.bundle.root / "records/after.json"
        other = self.bundle.root / "outside.json"
        path.rename(other)
        path.symlink_to(other)
        self.assertEqual(evidence.inspect_bundle(bundle, plan)["status"], "rejected")

    def test_expected_plan_is_independent(self):
        bundle, plan = self.bundle.write()
        expected = json.loads(plan.read_text())
        expected["run_id"] = "separate-expected-run"
        plan.write_text(json.dumps(expected))
        self.assertEqual(evidence.inspect_bundle(bundle, plan)["status"], "rejected")

    def test_actual_native_orchestrator_config_uses_train_source_nesting(self):
        try:
            from prime_rl.configs.orchestrator import OrchestratorConfig
        except ImportError:
            self.skipTest("Pinned native configuration package is optional")
        config = OrchestratorConfig(train={"source": [{"env": {
            "journal_root": "/tmp/fixture-only-journal", "taskset": {"id": "financial-selection-training"}}}]})
        self.bundle.config["orchestrator"] = config.model_dump(mode="json")
        self.bundle.put("config.json", self.bundle.config)
        self.bundle.plan["config_sha256"] = self.bundle.artifacts["config.json"]
        for row in self.bundle.rows.values():
            row["binding_sha256"] = evidence.digest(self.bundle.plan)
        self.assertEqual(self.bundle.inspect()["status"], "fixture_consistent")

    def test_obsolete_flat_orchestrator_layout_rejected(self):
        self.bundle.config["orchestrator"] = self.bundle.config["orchestrator"]["train"]
        self.bundle.put("config.json", self.bundle.config)
        self.bundle.plan["config_sha256"] = self.bundle.artifacts["config.json"]
        for row in self.bundle.rows.values():
            row["binding_sha256"] = evidence.digest(self.bundle.plan)
        self.reject()

    def test_wrong_algorithm_or_weighted_baseline_rejected(self):
        self.bundle.config["orchestrator"]["train"]["algo"]["length_weighted_baseline"] = True
        self.bundle.put("config.json", self.bundle.config)
        self.bundle.plan["config_sha256"] = self.bundle.artifacts["config.json"]
        for row in self.bundle.rows.values():
            row["binding_sha256"] = evidence.digest(self.bundle.plan)
        self.reject()

    def test_bool_tokens_and_oversized_numbers_fail(self):
        member = self.bundle.rows["cohort"]["members"][0]
        member["token_ids"] = [True, 11]
        self.reject()
        member["token_ids"] = [10, 11]
        member["reward"] = 10 ** 400
        self.reject()

    def test_missing_record_and_backward_time_fail(self):
        self.bundle.rows["reload"]["monotonic_ns"] = 1
        self.reject()
        del self.bundle.rows["reload"]
        self.reject()

    def test_raw_native_admission_and_reward_must_reconcile(self):
        member = self.bundle.rows["cohort"]["members"][0]
        trace = json.loads((self.bundle.root / member["native_trace"]).read_text())
        trace["rewards"]["financial"]["value"] = .7
        self.bundle.put(member["native_trace"], trace)
        member["native_trace_sha256"] = self.bundle.artifacts[member["native_trace"]]
        self.reject()

    def test_missing_artifact_cannot_use_null_hash(self):
        member = self.bundle.rows["cohort"]["members"][0]
        member["native_trace"] = "nonexistent.json"
        member["native_trace_sha256"] = None
        self.reject()

    def test_lora_scope_and_applied_artifact_join(self):
        self.bundle.plan["update_scope"] = "lora"
        self.bundle.config["trainer"]["model"]["lora"] = {"rank": 8}
        self.bundle.put("config.json", self.bundle.config)
        self.bundle.plan["config_sha256"] = self.bundle.artifacts["config.json"]
        for row in self.bundle.rows.values():
            row["binding_sha256"] = evidence.digest(self.bundle.plan)
        self.bundle.rows["after"]["trainable_state"] = self.bundle.rows["after"]["state"]["model"]
        name = "published/step_1/adapter.safetensors"
        self.bundle.put(name, b"synthetic adapter bytes")
        applied = "applied/step_1/adapter.safetensors"
        self.bundle.put(applied, b"synthetic adapter bytes")
        receiver = self.bundle.rows["receiver"]
        receiver.update(transport="filesystem_lora", published_files={name: self.bundle.artifacts[name]},
                        applied_files={name: self.bundle.artifacts[name]},
                        applied_copy_files={applied: self.bundle.artifacts[applied]},
                        published_state=copy.deepcopy(self.bundle.rows["after"]["trainable_state"]))
        result = self.bundle.inspect()
        self.assertEqual(result["status"], "fixture_consistent", result)
        self.assertEqual(result["update_scope"], "lora")
        copies = receiver.pop("applied_copy_files")
        self.reject()
        receiver["applied_copy_files"] = copies
        self.bundle.put(applied, b"different applied adapter bytes")
        copies[applied] = self.bundle.artifacts[applied]
        self.reject()
        self.bundle.put(applied, b"synthetic adapter bytes")
        copies[applied] = self.bundle.artifacts[applied]
        receiver["applied_files"][name] = evidence.digest("stale")
        self.reject()

    def test_micro_batch_progress_cannot_be_replaced_by_cohort_size(self):
        self.bundle.rows["before"]["batch_micro_batches"] = 1
        self.bundle.rows["after"]["batch_micro_batches"] = 1
        self.reject()
        self.bundle.rows["after"]["progress"]["total_samples"] = 1
        self.bundle.rows["after"]["state"]["progress"]["total_samples"] = evidence.digest(1)
        self.bundle.rows["checkpoint"]["state"] = copy.deepcopy(self.bundle.rows["after"]["state"])
        self.bundle.rows["reload"]["state"] = copy.deepcopy(self.bundle.rows["after"]["state"])
        self.bundle.rows["reload"]["progress"] = copy.deepcopy(self.bundle.rows["after"]["progress"])
        self.assertEqual(self.bundle.inspect()["status"], "fixture_consistent")

    def test_raw_provider_response_arrays_and_ids_cannot_be_relabelled(self):
        member = self.bundle.rows["cohort"]["members"][0]
        path = member["provider_responses"][0]
        response = json.loads((self.bundle.root / path).read_text())
        response["token_ids"] = [10, 999]
        self.bundle.put(path, response)
        self.reject()

    def test_provider_observation_cannot_substitute_unparsed_wire_logprobs(self):
        member = self.bundle.rows["cohort"]["members"][0]
        path = member["provider_responses"][0]
        response = json.loads((self.bundle.root / path).read_text())
        response["logprobs"] = [-.1, -.2]
        self.bundle.put(path, response)
        self.reject()

    def test_actual_native_response_logprobs_match_cohort_precision(self):
        try:
            from verifiers.v1.clients.train import response_from_generate
        except ImportError:
            self.skipTest("Pinned native training installation required")
        response = response_from_generate({"request_id": "request-0", "content": "fixture",
            "finish_reason": "stop", "prompt_ids": [1], "completion_ids": [10, 11],
            "completion_logprobs": [-.1, -.2],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "weight_versions": [{"start": 0, "end": 2, "version": "0"}]}, "fixture-policy")
        member = self.bundle.rows["cohort"]["members"][0]
        self.assertEqual(response.tokens.completion_logprobs.tolist(), member["logprobs"])
        self.assertNotEqual(response.tokens.completion_logprobs.tolist(), [-.1, -.2])
        path = member["provider_responses"][0]
        observation = json.loads((self.bundle.root / path).read_text())
        observation["logprobs"] = response.tokens.completion_logprobs.tolist()
        self.bundle.put(path, observation)
        self.assertEqual(self.bundle.inspect()["status"], "fixture_consistent")

    def test_raw_trainer_batch_cannot_be_missing_or_contradictory(self):
        path = "raw/trainer-batch.json"
        rows = json.loads((self.bundle.root / path).read_text())
        rows[0]["token_ids"] = [999, 11]
        self.bundle.put(path, rows)
        self.reject()
        # Updating both batch summaries cannot make foreign training data valid.
        for stage in ("before", "after"):
            self.bundle.rows[stage]["batch_sha256"] = evidence.digest(rows)
        self.reject()
        del self.bundle.artifacts[path]
        self.reject()

    def test_post_update_raw_response_must_exist_and_match_summary(self):
        inference = self.bundle.rows["inference"]
        inference["provider_responses"] = ["missing.json"]
        self.reject()
        path = "raw/post-update-response.json"
        inference["provider_responses"] = [path]
        original = json.loads((self.bundle.root / path).read_text())
        for field, value in (("weight_versions", [0, 0]), ("token_ids", [999, 11]),
                             ("provider_request_id", "request-0"), ("monotonic_ns", 69),
                             ("model", "different-policy")):
            with self.subTest(field=field):
                row = copy.deepcopy(original)
                row[field] = value
                self.bundle.put(path, row)
                self.reject()


if __name__ == "__main__":
    unittest.main()
