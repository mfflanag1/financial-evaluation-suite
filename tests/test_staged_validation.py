"""Public structural checkpoints reject malformed and incomplete submissions."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/financial_portfolio_staged/src"))

from financial_portfolio_staged.validation import (
    ARTIFACT_LIMITS, MAX_JSON_DEPTH, TRIAL_FIELDS, WINDOW_FIELDS,
    inventory_for_public_files, stage_contract, validate_stage,
)


def encoded(value):
    return json.dumps(value).encode("utf-8")


class StageValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "training_start": 0, "training_end": 72, "validation_start": 72,
            "active_end": 120, "validation_end": 122, "winddown_sessions": 2,
            "candidates": [{"id": f"candidate-{index:02d}"} for index in range(48)],
            "audit_requests": [
                {"entity_id": "E0", "day": 0, "field": "feature"},
                {"entity_id": "E1", "day": 72, "field": "forward_return"},
                {"entity_id": "candidate-00", "day": 120, "field": "turnover"},
            ],
        }
        self.public = {
            "prices.csv": "entity_id,close\r\nE0,100\r\nE1,101\r\n",
            "features.csv": 'entity_id,value,note\nE0,1,"two\nlines"\nE1,2,plain\n',
            "securities.csv": "entity_id,ticker\nE0,OLD\nE0,NEW\nE1,SECOND\n",
            "contract.json": json.dumps(self.contract, indent=2),
        }
        self.inventory = {
            "files": {name: {"sha256": hashlib.sha256(self.public[name].encode()).hexdigest(), "rows": count}
                      for name, count in (("prices.csv", 2), ("features.csv", 2), ("securities.csv", 3))},
            "contract_sha256": hashlib.sha256(self.public["contract.json"].encode()).hexdigest(),
            "entity_ids": ["E0", "E1"],
            "candidate_ids": [row["id"] for row in self.contract["candidates"]],
            "windows": {key: self.contract[key] for key in WINDOW_FIELDS},
        }
        self.trials = [{"id": row["id"], **dict.fromkeys(TRIAL_FIELDS, 0.0)}
                       for row in self.contract["candidates"]]
        self.policy = {
            "audit": [{**row, "value": None} for row in self.contract["audit_requests"]],
            "trials": copy.deepcopy(self.trials), "decision": "abstain", "selected": None,
        }
        self.artifacts = {
            "/workspace/research.py": b"raise RuntimeError('This source must never execute in a checkpoint')\n",
            "/workspace/ingestion.json": encoded(self.inventory),
            "/workspace/trials.json": encoded(self.trials),
            "/workspace/policy.json": encoded(self.policy),
        }

    def check(self, stage="submission", artifacts=None):
        return validate_stage(stage, self.public, self.artifacts if artifacts is None else artifacts)

    def put(self, filename, value):
        self.artifacts[f"/workspace/{filename}"] = encoded(value)

    def assert_invalid(self, reason=None, stage="submission"):
        result = self.check(stage)
        self.assertFalse(result["valid"], result)
        if reason:
            self.assertEqual(result["reason"], reason)
        self.assertLess(len(json.dumps(result)), 160)

    def test_complete_structure_passes_without_financial_reward_or_execution(self):
        for stage in ("ingestion", "trials", "submission"):
            result = self.check(stage)
            self.assertTrue(result["valid"], result)
            self.assertFalse(result["summary"]["financial_correctness_checked"])
            self.assertNotIn("reward", result)
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "should-not-exist"
            self.artifacts["/workspace/research.py"] = f"open({str(marker)!r}, 'w').write('bad')\n".encode()
            self.assertTrue(self.check()["valid"])
            self.assertFalse(marker.exists())

    def test_inventory_uses_original_bytes_csv_records_and_distinct_entities(self):
        self.assertEqual(inventory_for_public_files(self.public), self.inventory)
        self.assertEqual(self.inventory["files"]["features.csv"]["rows"], 2)
        self.assertEqual(len(self.public["features.csv"].splitlines()), 4)

    def test_contract_exposes_schemas_and_counts_without_inventory_answers(self):
        contract = stage_contract(self.public)
        self.assertEqual(contract["public_counts"], {"entities": 2, "candidates": 48, "audit_requests": 3})
        self.assertEqual(set(contract["ingestion_schema"]), set(self.inventory))
        self.assertEqual(set(contract["trials_schema"]["row"]), {"id", *TRIAL_FIELDS})
        self.assertEqual(contract["source"]["maximum_bytes"], ARTIFACT_LIMITS["/workspace/research.py"])
        self.assertEqual(contract["json"]["maximum_bytes_per_file"],
                         {path: limit for path, limit in ARTIFACT_LIMITS.items() if path.endswith(".json")})
        serialized = json.dumps(contract)
        self.assertNotIn(self.inventory["contract_sha256"], serialized)
        self.assertNotIn("candidate-00", serialized)
        self.assertNotIn('"E0"', serialized)

    def test_each_stage_requires_its_artifacts_and_revalidates_prior_inventory(self):
        required = {
            "ingestion": ("research.py", "ingestion.json"),
            "trials": ("research.py", "ingestion.json", "trials.json"),
            "submission": ("research.py", "ingestion.json", "trials.json", "policy.json"),
        }
        for stage, filenames in required.items():
            for filename in filenames:
                with self.subTest(stage=stage, filename=filename):
                    artifacts = dict(self.artifacts)
                    del artifacts[f"/workspace/{filename}"]
                    self.assertFalse(self.check(stage, artifacts)["valid"])
        self.assertTrue(self.check("ingestion")["valid"])
        self.inventory["files"]["prices.csv"]["rows"] += 1
        self.put("ingestion.json", self.inventory)
        self.assert_invalid("ingestion_inventory_mismatch", "trials")
        self.assert_invalid("ingestion_inventory_mismatch")

    def test_source_is_bounded_utf8_nonempty_and_parsed_only(self):
        cases = (
            (b"", "research_size_limit"), (b" " * 12, "research_empty"),
            (b"# comment only\n", "research_empty"), (b"\xff", "research_invalid_utf8"),
            (b"if :\n", "research_invalid_syntax"), (b"a = '\x00'", "research_invalid_syntax"),
            (b"x" * (ARTIFACT_LIMITS["/workspace/research.py"] + 1), "research_size_limit"),
            ("pass", "research_invalid_bytes"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.artifacts["/workspace/research.py"] = value
                self.assert_invalid(reason, "ingestion")

    def test_invalid_json_duplicates_nonfinite_overflow_depth_and_size(self):
        cases = (
            (b'{"files":{},"files":{}}', "ingestion_invalid_json"),
            (b'{"nested":{"a":1,"a":2}}', "ingestion_invalid_json"),
            (b"NaN", "ingestion_invalid_json"), (b"Infinity", "ingestion_invalid_json"),
            (b"1e999", "ingestion_nonfinite_number"),
            (b"9" * 400, "ingestion_nonfinite_number"),
            (b"[" * (MAX_JSON_DEPTH + 1) + b"0" + b"]" * (MAX_JSON_DEPTH + 1), "ingestion_depth_limit"),
            (b" " * (ARTIFACT_LIMITS["/workspace/ingestion.json"] + 1), "ingestion_size_limit"),
            (b"\xff", "ingestion_invalid_utf8"), (b"{}{}", "ingestion_invalid_json"),
        )
        for payload, reason in cases:
            with self.subTest(reason=reason, payload=payload[:32]):
                self.artifacts["/workspace/ingestion.json"] = payload
                self.assert_invalid(reason, "ingestion")

    def test_per_artifact_byte_limits_accept_boundary_and_reject_one_extra_byte(self):
        original = dict(self.artifacts)
        for path, maximum_bytes in ARTIFACT_LIMITS.items():
            with self.subTest(path=path):
                self.artifacts = dict(original)
                payload = self.artifacts[path]
                self.artifacts[path] = payload + b" " * (maximum_bytes - len(payload))
                self.assertTrue(self.check()["valid"])
                self.artifacts[path] += b" "
                reason = path.rsplit("/", 1)[1].split(".")[0] + "_size_limit"
                self.assert_invalid(reason)

    def test_integer_fields_reject_booleans_and_integral_floats(self):
        for value in (True, False, 0.0, 72.0):
            for target in ("rows", "window", "day"):
                with self.subTest(value=value, target=target):
                    self.setUp()
                    if target == "rows":
                        self.inventory["files"]["prices.csv"]["rows"] = value
                        self.put("ingestion.json", self.inventory)
                    elif target == "window":
                        self.inventory["windows"]["training_start"] = value
                        self.put("ingestion.json", self.inventory)
                    else:
                        self.policy["audit"][0]["day"] = value
                        self.put("policy.json", self.policy)
                    self.assert_invalid()

    def test_inventory_rejects_extra_fields_missing_or_duplicate_identities_and_stale_digest(self):
        changes = (
            lambda value: value.update(extra=1),
            lambda value: value["files"].update(extra={"sha256": "0" * 64, "rows": 0}),
            lambda value: value["entity_ids"].append("E0"),
            lambda value: value["candidate_ids"].pop(),
            lambda value: value["candidate_ids"].__setitem__(0, ["unhashable"]),
            lambda value: value.update(contract_sha256="0" * 64),
            lambda value: value["windows"].update(validation_end=999),
        )
        for change in changes:
            with self.subTest(change=change):
                inventory = copy.deepcopy(self.inventory)
                change(inventory)
                self.put("ingestion.json", inventory)
                self.assert_invalid(stage="ingestion")

    def test_trial_coverage_and_exact_field_types(self):
        changes = (
            lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(id="unknown"), lambda rows: rows[0].update(extra=0),
            lambda rows: rows[0].pop("mean_net"), lambda rows: rows[0].update(sharpe=True),
            lambda rows: rows[0].update(sharpe="0"), lambda rows: rows[0].update(sharpe=None),
            lambda rows: rows[0].update(sharpe=float("inf")),
            lambda rows: rows[0].update(sharpe=float("nan")),
        )
        for change in changes:
            with self.subTest(change=change):
                trials = copy.deepcopy(self.trials)
                change(trials)
                self.put("trials.json", trials)
                self.assert_invalid(stage="trials")

    def test_empty_or_partial_policy_regression(self):
        for policy in ({}, [], {"audit": [], "trials": [], "decision": "abstain", "selected": None},
                       {key: value for key, value in self.policy.items() if key != "selected"}):
            with self.subTest(policy=policy):
                self.put("policy.json", policy)
                self.assert_invalid()

    def test_policy_requires_full_audit_coverage_and_valid_types(self):
        changes = (
            lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(entity_id="unknown"), lambda rows: rows[0].update(field="unknown"),
            lambda rows: rows[0].update(day=1), lambda rows: rows[0].update(extra=1),
            lambda rows: rows[0].update(value=True), lambda rows: rows[0].update(value="0"),
            lambda rows: rows[0].update(value=float("nan")),
        )
        for change in changes:
            with self.subTest(change=change):
                policy = copy.deepcopy(self.policy)
                change(policy["audit"])
                self.put("policy.json", policy)
                self.assert_invalid()

    def test_trials_must_match_current_checkpoint_order_independently(self):
        self.inventory["entity_ids"].reverse()
        self.inventory["candidate_ids"].reverse()
        self.put("ingestion.json", self.inventory)
        self.policy["trials"].reverse()
        self.policy["audit"].reverse()
        self.put("policy.json", self.policy)
        self.assertTrue(self.check()["valid"])
        self.trials[0]["sharpe"] = 1
        self.put("trials.json", self.trials)
        self.assertTrue(self.check("trials")["valid"])
        self.assert_invalid("policy_trials_mismatch")

    def test_decision_is_structurally_legal_but_not_economically_verified(self):
        for decision, selected, valid in (
            ("abstain", None, True), ("deploy", "candidate-00", True),
            ("abstain", "candidate-00", False), ("deploy", None, False),
            ("deploy", "unknown", False), ("deploy", ["candidate-00"], False),
            ("unknown", None, False), (True, None, False),
        ):
            with self.subTest(decision=decision, selected=selected):
                self.policy.update(decision=decision, selected=selected)
                self.put("policy.json", self.policy)
                self.assertEqual(self.check()["valid"], valid)

    def test_unknown_stage_or_untrusted_input_shapes_fail_closed(self):
        self.assert_invalid("unknown_stage", "unknown")
        self.assertEqual(validate_stage("ingestion", self.public, []),
                         {"valid": False, "reason": "invalid_artifacts"})
        for public in ({}, [], {**self.public, "private.json": "{}"}):
            self.assertEqual(validate_stage("ingestion", public, self.artifacts),
                             {"valid": False, "reason": "invalid_public_data"})


if __name__ == "__main__":
    unittest.main()
