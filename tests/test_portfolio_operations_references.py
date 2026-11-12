"""Reference-control regressions without Docker, provider calls or private data."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/financial_portfolio_operations/src"))
from financial_portfolio_operations import references as ref


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def perfect_score(quality=0.5):
    return {"reward": 0.8 + 0.2 * quality,
            "metrics": {"schema_valid": 1.0, "audit_accuracy": 1.0, "trial_accuracy": 1.0,
                        "decision_correct": 1.0, "economic_quality": quality}}


class ReferenceScoreValidation(unittest.TestCase):
    def test_subunit_reference_ceiling_is_not_normalized(self):
        score = perfect_score(0.5)
        self.assertEqual(ref._checked_score(score)["reward"], 0.9)
        self.assertEqual(ref._checked_score(perfect_score(0))["reward"], 0.8)

    def test_invalid_nonfinite_boolean_and_inconsistent_scores_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), True, -1, 1.1, 10 ** 1000):
            with self.subTest(bad_type=type(bad).__name__):
                score = perfect_score()
                score["metrics"]["economic_quality"] = bad
                with self.assertRaises(ValueError):
                    ref._checked_score(score)
        for change in (lambda s: s["metrics"].update(audit_accuracy=0.99),
                       lambda s: s.update(reward=1.0),
                       lambda s: s["metrics"].pop("schema_valid")):
            score = perfect_score()
            change(score)
            with self.assertRaises(ValueError):
                ref._checked_score(score)

    def test_only_complete_dev_schedule_accepted_and_integer_types_are_strict(self):
        self.assertEqual(ref._validate_seeds(range(128, 160)), tuple(range(128, 160)))
        for seeds in ([128], range(160, 192), range(127, 159),
                      [128] * 32, list(range(128, 159)) + [159.0],
                      list(range(128, 159)) + [True]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                ref._validate_seeds(seeds)

    def test_key_environment_restored_even_after_failure(self):
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "previous"}):
            with self.assertRaises(RuntimeError):
                with ref._evaluator_key("temporary"):
                    self.assertEqual(os.environ["FINANCIAL_RESEARCH_KEY"], "temporary")
                    raise RuntimeError("fixture")
            self.assertEqual(os.environ["FINANCIAL_RESEARCH_KEY"], "previous")


class ContainerReferenceBoundaries(unittest.TestCase):
    def test_container_is_public_only_pinned_offline_and_has_no_key_environment(self):
        completed = subprocess.CompletedProcess([], 0, b"", b"")
        removed = subprocess.CompletedProcess([], 1, b"", b"Error: No such container: fixture")
        pin = "sha256:" + "a" * 64
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "secret-fixture-value"}), \
                patch.object(ref.subprocess, "run", side_effect=[completed, removed]) as run:
            result = ref._container_reference(Path("/tmp/public-fixture"), pin)
        self.assertTrue(result["container_cleanup_confirmed"])
        self.assertFalse(result["native_agent_harness_gold_validation"])
        command, kwargs = run.call_args_list[0].args[0], run.call_args_list[0].kwargs
        self.assertIn(pin, command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--mount") + 1],
                         "type=bind,src=/tmp/public-fixture,dst=/workspace")
        self.assertNotIn("FINANCIAL_RESEARCH_KEY", kwargs["env"])
        self.assertNotIn("secret-fixture-value", str(command))
        self.assertEqual(kwargs["timeout"], 30)
        self.assertIn("--read-only", command)
        self.assertEqual(run.call_args_list[1].args[0][:3], ["docker", "rm", "--force"])

    def test_timed_out_client_still_removes_container(self):
        with patch.object(ref.subprocess, "run", side_effect=[
                subprocess.TimeoutExpired("docker", 30),
                subprocess.CompletedProcess([], 0, b"", b"")]) as run:
            with self.assertRaisesRegex(RuntimeError, "exceeded 30 seconds"):
                ref._container_reference(Path("/tmp/public-fixture"), "sha256:" + "a" * 64)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[0][:3], ["docker", "rm", "--force"])

    def test_failure_to_confirm_cleanup_fails_control(self):
        with patch.object(ref.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 0, b"", b""),
                subprocess.CompletedProcess([], 1, b"", b"daemon unavailable")]):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                ref._container_reference(Path("/tmp/public-fixture"), "sha256:" + "a" * 64)


class FullReferenceRun(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.source = self.repo / "environments/financial_portfolio/src/financial_portfolio/reference.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("# trusted standalone fixture\n")
        self.hashes = {self.source.relative_to(self.repo).as_posix(): digest(self.source.read_bytes())}
        self.manifest = self.repo / "manifest.json"
        write_json(self.manifest, {"protocol_id": "fixture", "sha256": self.hashes})
        self.key = self.root / "key"
        self.key.write_text("test-evaluator-key-do-not-use-00000")
        self.public = {name: "fixture\n" for name in ref.PUBLIC_FILES}
        self.policy = {"audit": [], "trials": [], "decision": "abstain", "selected": None}
        self.families = ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal")
        self.schedule = [{"seed": seed, "task_id": f"fp_{seed:024x}",
                          "family": self.families[seed % 4],
                          "public_sha256": {name: digest(data.encode()) for name, data in self.public.items()},
                          "prompt_sha256": digest(b"prompt"), "system_prompt_sha256": digest(b"system")}
                         for seed in ref.DEVELOPMENT_SEEDS]
        self.preflight = self.root / "preflight.json"
        self.baseline = {"source_files_sha256": self.hashes,
                         "evaluator_key_sha256": digest(self.key.read_bytes()),
                         "episode_schedule": self.schedule, "repository_commit": "a" * 40,
                         "docker_image": {"image_id": "sha256:" + "b" * 64}}
        write_json(self.preflight, self.baseline)
        self.generator = SimpleNamespace(FAMILIES=self.families, build_episode=Mock(side_effect=self.episode))
        self.verifier = SimpleNamespace(score_submission=Mock(side_effect=lambda e, p: perfect_score()),
                                        reference_submission=Mock(return_value=deepcopy(self.policy)))
        self.validation = SimpleNamespace(
            STAGES=("ingestion", "trials", "submission"),
            inventory_for_public_files=Mock(return_value={"inventory": "public only"}),
            validate_stage=Mock(return_value={"valid": True}))
        self.clarified = SimpleNamespace(clarified_prompt=Mock(return_value="system"))
        self.dependencies = patch.object(ref, "_dependencies", return_value=(
            self.generator, self.verifier, self.validation, self.clarified))
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        self.container = patch.object(ref, "_container_reference", side_effect=self.execute)
        self.container_mock = self.container.start()
        self.addCleanup(self.container.stop)
        self.output = self.root / "local-references"

    def episode(self, seed, split):
        self.assertEqual(split, "dev")
        self.assertIn(seed, ref.DEVELOPMENT_SEEDS)
        return {"id": f"fp_{seed:024x}", "prompt": "prompt", "public_files": deepcopy(self.public),
                "private": {"never_save_this_hidden_payload": "PRIVATE_FIXTURE"}}

    def execute(self, workspace, image_id):
        self.assertEqual({p.name for p in workspace.iterdir()}, ref.PUBLIC_FILES | {"research.py"})
        write_json(workspace / "policy.json", self.policy)
        return {"kind": "standalone_public_reference_container", "image_id": image_id,
                "network": "none", "wall_seconds": 0.1, "wall_timeout_seconds": 30,
                "address_space_limit_bytes": 268435456, "returncode": 0,
                "container_cleanup_confirmed": True, "native_agent_harness_gold_validation": False}

    def generate(self, **kwargs):
        return ref.generate_reference_scores(self.output, repo_root=self.repo,
            protocol_manifest=self.manifest, baseline_preflight=self.preflight, key_file=self.key, **kwargs)

    def test_all_32_independent_public_scores_exact_ceiling_and_no_private_payload(self):
        result = self.generate()
        self.assertEqual(result["reference_count"], 32)
        self.assertEqual([row["seed"] for row in result["records"]], list(range(128, 160)))
        self.assertEqual(self.container_mock.call_count, 32)
        self.assertEqual(self.verifier.score_submission.call_count, 64)
        self.assertEqual(self.verifier.reference_submission.call_count, 32)
        self.assertTrue(all(row["attainable_financial_score_ceiling"] == 0.9 for row in result["records"]))
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"PRIVATE_FIXTURE", path.read_bytes())
                self.assertNotIn(self.key.read_bytes(), path.read_bytes())
        verified = ref.validate_reference_report(self.output / "references.json", repo_root=self.repo,
                                                 protocol_manifest=self.manifest)
        self.assertEqual(verified, result)

    def test_changed_artifact_invalidates_completed_report(self):
        self.generate()
        (self.output / "dev-128/policy.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ref.validate_reference_report(self.output / "references.json", repo_root=self.repo,
                                          protocol_manifest=self.manifest)

    def test_report_integer_and_protocol_bindings_are_strict(self):
        result = self.generate()
        path = self.output / "references.json"
        result["model_calls"] = False
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, "complete development control"):
            ref.validate_reference_report(path, repo_root=self.repo, protocol_manifest=self.manifest)
        result["model_calls"] = 0
        write_json(path, result)
        write_json(self.manifest, {"protocol_id": "other-condition", "sha256": self.hashes})
        with self.assertRaisesRegex(ValueError, "different protocol"):
            ref.validate_reference_report(path, repo_root=self.repo, protocol_manifest=self.manifest)

    def test_partial_or_duplicate_tasks_do_not_create_completed_report(self):
        self.generator.build_episode.side_effect = lambda seed, split: self.episode(128, split)
        with self.assertRaisesRegex(ValueError, "Duplicate generated task"):
            self.generate()
        self.assertEqual(self.container_mock.call_count, 1)
        self.assertFalse((self.output / "references.json").exists())

    def test_changed_inputs_fail_before_container_runs(self):
        self.public["prices.csv"] = "changed\n"
        with self.assertRaisesRegex(ValueError, "public inputs"):
            self.generate()
        self.container_mock.assert_not_called()

    def test_unmatched_key_or_changed_source_rejects_without_output(self):
        self.key.write_text("different-key-000000000000000000000")
        with self.assertRaisesRegex(ValueError, "key fingerprint"):
            self.generate()
        self.assertFalse(self.output.exists())
        self.source.write_text("changed source\n")
        with self.assertRaisesRegex(ValueError, "Frozen source differs"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_imperfect_reference_fails_and_gold_disagreement_fails(self):
        self.verifier.score_submission.return_value = None
        score = perfect_score()
        score["metrics"]["trial_accuracy"] = 0.99
        self.verifier.score_submission.side_effect = lambda e, p: score
        with self.assertRaisesRegex(ValueError, "not publicly perfect"):
            self.generate()
        self.assertFalse((self.output / "references.json").exists())
        self.output = self.root / "different-local-run"
        self.verifier.score_submission.side_effect = [perfect_score(0.5), perfect_score(0.6)]
        with self.assertRaisesRegex(ValueError, "differs from trusted gold"):
            self.generate()

    def test_existing_output_and_repository_destination_rejected(self):
        self.output.mkdir()
        marker = self.output / "preserve.txt"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(marker.read_text(), "preserve")
        self.output = self.repo / "references"
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            self.generate()
        self.generator.build_episode.assert_not_called()

    def test_source_mutation_by_execution_rejected(self):
        original = self.execute
        def mutate(workspace, image_id):
            result = original(workspace, image_id)
            (workspace / "research.py").write_text("changed")
            return result
        self.container_mock.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed its source"):
            self.generate()
        self.assertFalse((self.output / "references.json").exists())

    def test_duplicate_baseline_schedule_rejected_before_generation(self):
        self.baseline["episode_schedule"][1]["task_id"] = self.baseline["episode_schedule"][0]["task_id"]
        write_json(self.preflight, self.baseline)
        with self.assertRaisesRegex(ValueError, "Duplicate task IDs"):
            self.generate()
        self.generator.build_episode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
