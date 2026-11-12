"""Fixture and trusted-reference checks without executing candidate code."""

import ast
import builtins
import io
import json
from pathlib import Path, PurePosixPath
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/financial_prerequisites/src"))

from financial_prerequisites.cases import load_cases


def trusted_reference_output(source, files):
    """Run only repository-owned reference solve(), with in-memory public files."""
    namespace = {"__name__": "trusted_reference_fixture_test"}
    exec(compile(source, "<trusted reference>", "exec"), namespace)

    def public_open(name, mode="r", **kwargs):
        if mode != "r" or name not in files:
            raise AssertionError(f"Unexpected reference file access: {name!r} {mode!r}")
        return io.StringIO(files[name])

    with patch.object(builtins, "open", public_open):
        return namespace["solve"]()


class PrerequisiteCases(unittest.TestCase):
    def assert_exact_contract(self, expected, actual):
        if isinstance(expected, dict):
            self.assertIs(type(actual), dict)
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_exact_contract(expected[key], actual[key])
        elif isinstance(expected, list):
            self.assertIs(type(actual), list)
            self.assertEqual(len(expected), len(actual))
            for wanted, got in zip(expected, actual):
                self.assert_exact_contract(wanted, got)
        elif type(expected) is int:
            self.assertIs(type(actual), int)
            self.assertEqual(expected, actual)
        elif type(expected) is float:
            self.assertIn(type(actual), (int, float))
            self.assertAlmostEqual(expected, actual, delta=1e-9)
        else:
            self.assertIs(type(actual), type(expected))
            self.assertEqual(expected, actual)

    def test_four_distinct_bounded_cases_and_public_inventory(self):
        cases = load_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({case["id"] for case in cases}), 4)
        self.assertEqual({case["kind"] for case in cases}, {
            "schema_inspection", "saved_calculation", "coding_error_recovery", "structured_coverage"})
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(set(case), {"id", "kind", "prompt", "public_files",
                    "expected_json", "reference_source", "alternate_public_files", "alternate_expected_json"})
                self.assertLess(len(case["prompt"]) + sum(map(len, case["public_files"].values())), 6000)
                self.assertEqual(set(case["public_files"]), set(case["alternate_public_files"]))
                for field in ("public_files", "alternate_public_files"):
                    for name, content in case[field].items():
                        self.assertEqual(PurePosixPath(name).name, name)
                        self.assertNotIn(name, {"policy.json", "research.py"})
                        self.assertIsInstance(content, str)
                        if name.endswith(".json"):
                            json.loads(content)
                self.assertIn("same\nschema", case["prompt"])
                self.assertNotEqual(case["expected_json"], case["alternate_expected_json"])
                ast.parse(case["reference_source"])

    def test_reference_solves_both_input_variants(self):
        for case in load_cases():
            for public_key, expected_key in (("public_files", "expected_json"),
                                             ("alternate_public_files", "alternate_expected_json")):
                with self.subTest(case=case["id"], variant=public_key):
                    actual = trusted_reference_output(case["reference_source"], case[public_key])
                    self.assert_exact_contract(case[expected_key], actual)
                    json.dumps(actual, allow_nan=False)

    def test_loader_returns_independent_nested_values(self):
        first = load_cases()
        first[0]["public_files"]["contract.json"] = "changed"
        first[0]["expected_json"]["schema"]["entity_ids"].append("changed")
        self.assertNotEqual(first, load_cases())

    def test_schema_fixture_requires_flat_window_and_preserves_string_ids(self):
        case = load_cases()[0]
        self.assertEqual(set(json.loads(case["public_files"]["contract.json"])),
                         {"source_window_start", "source_window_end"})
        self.assertEqual(case["expected_json"]["schema"]["entity_ids"], ["001", "010", "B"])

    def test_repair_starter_is_broken_and_same_for_both_variants(self):
        case = load_cases()[2]
        starter = case["public_files"]["starter.py"]
        self.assertEqual(starter, case["alternate_public_files"]["starter.py"])
        self.assertIn('csv.reader(handle)', starter)
        self.assertIn('row["account"]', starter)
        self.assertNotEqual(starter, case["reference_source"])

    def test_coverage_requires_all_request_ids_and_empty_selection_semantics(self):
        case = load_cases()[3]
        for prefix in ("", "alternate_"):
            requests = json.loads(case[prefix + "public_files"]["contract.json"])["requests"]
            result = case[prefix + "expected_json"]
            self.assertEqual(result["request_count"], len(requests))
            self.assertEqual([row["request_id"] for row in requests],
                             [row["request_id"] for row in result["results"]])
            for request, row in zip(requests, result["results"]):
                self.assertIs(type(row["observation_count"]), int)
                if request["operation"] == "count":
                    self.assertIs(type(row["value"]), int)
                if request["operation"] == "mean" and row["observation_count"] == 0:
                    self.assertIsNone(row["value"])


if __name__ == "__main__":
    unittest.main()
