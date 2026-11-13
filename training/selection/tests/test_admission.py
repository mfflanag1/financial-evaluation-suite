"""Use the pinned native admission predicate before any sample conversion."""

from copy import deepcopy
import importlib.util
from importlib.resources import files
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import prime_rl
import verifiers.v1 as vf
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.env.encoding import pack, unpack

from financial_selection_training.admission import apply
from financial_selection_training.journal import WorkerJournal, INFO_KEY
from support import completed, episode


# Import the exact installed predicate without importing all optional algorithms.
SPEC = importlib.util.spec_from_file_location("installed_native_admission", files("prime_rl").joinpath("orchestrator/algo/base.py"))
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.journal = WorkerJournal(Path(self.temporary.name))
        self.addCleanup(self.journal.close)

    async def test_positive_and_candidate_zero_remain_native_trainable(self):
        for index, reward in enumerate((1.0, 0.0)):
            native, attempt = await completed(self.journal, f"trace-{index}", reward)
            before = deepcopy(native.rewards)
            self.assertTrue(apply(native, self.journal))
            self.assertEqual(before, native.rewards)
            wire = vf.Episode.model_validate(unpack(pack(episode(native).model_dump(mode="python"))))
            self.assertEqual(len(list(NATIVE.iter_trainable_traces([wire]))), 1)
            [sample] = trace_to_samples(wire.traces[0], env_name="financial-selection-training")
            self.assertEqual(sample.token_ids.tolist(), [1, 2, 3, 4])
            self.assertEqual(sample.mask.tolist(), [False, False, True, True])
            self.assertIsNone(sample.advantages)
            self.assertEqual(len(sample.logprobs), len(sample.token_ids))
            self.assertEqual(sum(row["event"] == "attempt_terminal" for row in attempt.events), 1)

    async def test_unresolved_cases_drop_after_wire_without_reward_rewrite(self):
        def event(attempt, kind):
            return next(row for row in attempt.events if row["event"] == kind)
        mutations = {
            "unknown_tool": lambda t, a: a.tools.update(tool="allowed"),
            "missing_terminal": lambda t, a: a.events.remove(event(a, "harness_terminal")),
            "duplicate_terminal": lambda t, a: a.events.append(dict(event(a, "harness_terminal"))),
            "dead_runtime": lambda t, a: event(a, "runtime_health").update(alive=False),
            "unknown_runtime": lambda t, a: event(a, "runtime_health").update(alive=None),
            "provider_error": lambda t, a: a.requests[0].update(outcome="error"),
            "invalid_binding": lambda t, a: t.info[INFO_KEY].update(task_sha256="changed"),
            "scorer_exception": lambda t, a: a.events.append({"event": "scoring_failed"}),
            "validation": lambda t, a: t.info[INFO_KEY].update(model_free_reference=True),
            "timeout": lambda t, a: setattr(t, "is_timeout", True),
            "native_failure": lambda t, a: setattr(t, "ok", False),
            "missing_logprob": lambda t, a: setattr(t.nodes[-1], "logprobs", []),
            "reward_mismatch": lambda t, a: event(a, "scoring_complete").update(reward=.5),
            "completion_ids_changed": lambda t, a: t.nodes[-1].token_ids.__setitem__(0, 99),
            "logprobs_changed": lambda t, a: t.nodes[-1].logprobs.__setitem__(0, -.9),
            "support_removed": lambda t, a: setattr(t.nodes[-1], "sampler_support", None),
            "support_changed": lambda t, a: t.nodes[-1].sampler_support.ids.__setitem__(0, 99),
            "policy_version_missing": lambda t, a: setattr(t.nodes[-1], "weight_versions", None),
            "policy_version_changed": lambda t, a: setattr(t.nodes[-1].weight_versions[0], "version", "changed"),
            "call_model_changed": lambda t, a: setattr(t.calls[0], "model", "changed"),
            "call_node_changed": lambda t, a: setattr(t.calls[0], "node", 0),
            "prompt_token_changed": lambda t, a: t.nodes[0].token_ids.__setitem__(0, 99),
            "call_parent_changed": lambda t, a: setattr(t.nodes[-1], "parent", 0),
        }
        for identity, mutate in mutations.items():
            with self.subTest(identity=identity):
                native, attempt = await completed(self.journal, identity, reward=0)
                before = deepcopy(native.rewards)
                mutate(native, attempt)
                self.assertFalse(apply(native, self.journal))
                self.assertEqual(before, native.rewards)
                wire = vf.Episode.model_validate(unpack(pack(episode(native).model_dump(mode="python"))))
                self.assertTrue(wire.traces[0].is_dropped)
                self.assertEqual(list(NATIVE.iter_trainable_traces([wire])), [])
                self.assertEqual(sum(row["event"] == "attempt_terminal" for row in attempt.events), 1)

    async def test_cyclic_or_interleaved_prompt_graph_fails_admission(self):
        for kind in ("cycle", "interleaved"):
            native, _ = await completed(self.journal, kind)
            if kind == "cycle":
                native.nodes[1].parent = 1
            else:
                # Keep extracted prompt [1,2] and completion [3,4] unchanged,
                # but corrupt their causal order in the physical token stream.
                native.nodes[-1].parent = 0
                native.nodes[-1].token_ids = np.array([3, 2, 4], dtype=np.int32)
                native.nodes[-1].mask = np.array([True, False, True])
            self.assertFalse(apply(native, self.journal))
            self.assertTrue(native.is_dropped)
            # Intentionally cyclic graphs cannot be passed to upstream graph
            # serialization, whose path traversal assumes an acyclic graph.
            self.assertEqual(native.reward, 1)

    async def test_nonzero_harness_exit_is_unresolved_even_with_financial_reward(self):
        native, attempt = await completed(self.journal, "nonzero")
        next(row for row in attempt.events if row["event"] == "harness_terminal")["exit_code"] = 1
        self.assertFalse(apply(native, self.journal))
        self.assertEqual(native.reward, 1)

    async def test_info_only_flag_does_not_replace_native_drop(self):
        native, _ = await completed(self.journal)
        native.info["training_eligible"] = False
        self.assertEqual(len(list(NATIVE.iter_trainable_traces([episode(native)]))), 1)
        self.journal.attempts[native.id].tools["pending"] = "gating"
        apply(native, self.journal)
        self.assertEqual(list(NATIVE.iter_trainable_traces([episode(native)])), [])

    async def test_duplicate_admission_and_unregistered_trace_fail_closed(self):
        native, _ = await completed(self.journal)
        apply(native, self.journal)
        with self.assertRaises(RuntimeError):
            apply(native, self.journal)
        native.id = "unregistered"
        with self.assertRaises(RuntimeError):
            apply(native, self.journal)

    async def test_financial_service_or_component_corruption_cannot_become_trainable_zero(self):
        for kind in ("service", "components", "diagnostic", "extra_reward", "setup", "isolation"):
            native, attempt = await completed(self.journal, kind, reward=0.0)
            before = deepcopy(native.rewards)
            if kind == "service":
                native.info["financial_selection_outcome"]["status"] = "infrastructure_error"
            elif kind == "components":
                native.metrics["candidate_failure"] = 0.0
            elif kind == "diagnostic":
                native.info["scoring_diagnostics"] = {}
            elif kind == "extra_reward":
                native.rewards["unexpected"] = vf.Reward(score=1)
                before = deepcopy(native.rewards)
            elif kind == "setup":
                attempt.events[:] = [r for r in attempt.events if r["event"] != "verifier_setup_complete"]
            else:
                solver = next(r["runtime_name_sha256"] for r in attempt.events if r["event"] == "solver_setup_complete")
                for row in attempt.events:
                    if row["event"] in ("verifier_setup_complete", "verifier_staged"):
                        row["runtime_name_sha256"] = solver
            self.assertFalse(apply(native, self.journal), kind)
            self.assertEqual(native.rewards, before)

    async def test_reproduced_three_packet_zero_is_admitted_without_reward_rewrite(self):
        from financial_selection_training.journal import digest
        native, attempt = await completed(self.journal, "scored-zero", reward=0.0)
        outcome = {"status": "scored", "reason": "three_packet_program_checked", "reproduction_verified": True}
        native.info["financial_selection_outcome"] = outcome
        native.info["scoring_diagnostics"]["financial_selection_outcome"] = dict(outcome)
        native.metrics.update(candidate_failure=0.0, reproduction_verified=1.0, packets_scored=3.0)
        score = next(row for row in attempt.events if row["event"] == "scoring_complete")
        score.update(outcome_sha256=digest(outcome), grade_sha256=digest({"reward": 0.0, "metrics": dict(native.metrics)}))
        self.assertTrue(apply(native, self.journal))
        self.assertEqual(native.reward, 0)
        self.assertEqual(len(list(NATIVE.iter_trainable_traces([episode(native)]))), 1)
