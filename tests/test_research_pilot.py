"""Independent public-data solver, data pitfalls, and deterministic reward tests."""

import copy
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "environments" / "financial_research"
sys.path.insert(0, str(PACKAGE / "src"))

from financial_research.analytics import CANDIDATES, portfolio_metrics, select_candidate
from financial_research.generator import FAMILIES, SPLITS, _registry, build_episode, resolve_episode
from financial_research.verifier import reference_submission, score_submission

REFERENCE = PACKAGE / "src" / "financial_research" / "reference.py"
spec = importlib.util.spec_from_file_location("public_research_example", REFERENCE)
public_solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_solver)


def csv_change(text, mutate):
    rows = list(csv.DictReader(io.StringIO(text)))
    mutate(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def run_public_solution(files):
    # Isolated Python cannot accidentally import the evaluator via PYTHONPATH.
    with tempfile.TemporaryDirectory(prefix="research-public-test-") as directory:
        workspace = Path(directory)
        for filename, contents in files.items():
            (workspace / filename).write_text(contents, encoding="utf-8")
        (workspace / "research.py").write_bytes(REFERENCE.read_bytes())
        subprocess.run([sys.executable, "-I", "research.py"], cwd=workspace,
                       env={"PATH": os.defpath}, check=True, capture_output=True,
                       text=True, timeout=30)
        return json.loads((workspace / "policy.json").read_text())


class PublicResearchPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key_patch = patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "public-test-key-only-do-not-use-for-held-out-data"})
        cls.key_patch.start()
        cls.addClassCleanup(cls.key_patch.stop)
        cls.episodes = {family: build_episode(index) for index, family in enumerate(FAMILIES)}
        cls.solutions = {family: run_public_solution(episode["public_files"])
                         for family, episode in cls.episodes.items()}

    def test_independent_public_solver_matches_all_four_families(self):
        for family, episode in self.episodes.items():
            with self.subTest(family=family):
                result = score_submission(episode, self.solutions[family])
                self.assertEqual(result["reward"], 1.0)
                self.assertTrue(all(value == 1 for value in result["metrics"].values()))
                expected = reference_submission(episode)
                self.assertEqual(self.solutions[family]["selected"], expected["selected"])
                self.assertEqual(len(self.solutions[family]["trials"]), 16)
        self.assertEqual(self.solutions["signal"]["decision"], "deploy")
        for family in ("null", "cost_trap", "revision_trap"):
            self.assertEqual(self.solutions[family]["decision"], "abstain")

    def test_source_duplicates_and_row_order_do_not_change_solution(self):
        episode = self.episodes["signal"]
        files = dict(episode["public_files"])
        def duplicate_and_reverse(rows):
            rows.extend(copy.deepcopy(rows[:20]))
            rows.reverse()
        for name in ("prices.csv", "features.csv", "securities.csv"):
            if name != "securities.csv":
                files[name] = csv_change(files[name], duplicate_and_reverse)
        self.assertEqual(run_public_solution(files), self.solutions["signal"])

    def test_future_publications_and_final_revisions_create_detectable_lookahead(self):
        episode = self.episodes["revision_trap"]
        files = dict(episode["public_files"])
        def release_early(rows):
            for row in rows:
                row["available_day"] = row["observation_day"]
        files["features.csv"] = csv_change(files["features.csv"], release_early)
        leaked = run_public_solution(files)
        clean_best = max(row["sharpe"] for row in self.solutions["revision_trap"]["trials"])
        leaked_best = max(row["sharpe"] for row in leaked["trials"])
        self.assertGreater(leaked_best, clean_best + 3)
        result = score_submission(episode, leaked)
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertLess(result["metrics"]["trial_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_unadjusted_split_prices_fail_audit(self):
        episode = self.episodes["signal"]
        files = dict(episode["public_files"])
        def remove_adjustment(rows):
            for row in rows:
                row["shares_per_original"] = "1"
        files["prices.csv"] = csv_change(files["prices.csv"], remove_adjustment)
        result = score_submission(episode, run_public_solution(files))
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_ignored_costs_fail_trial_verification(self):
        episode = self.episodes["cost_trap"]
        files = dict(episode["public_files"])
        contract = json.loads(files["contract.json"])
        contract["cost_bps"] = 0
        contract["borrow_bps_daily"] = 0
        files["contract.json"] = json.dumps(contract)
        result = score_submission(episode, run_public_solution(files))
        self.assertEqual(result["metrics"]["audit_accuracy"], 1)
        self.assertLess(result["metrics"]["trial_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_release_priority_and_future_observation_filter(self):
        securities = [{"entity_id": "E", "ticker": "T", "valid_from": "0", "valid_to": "3"}]
        prices = [{"ticker": "T", "day": str(day), "close": "100", "shares_per_original": "1"}
                  for day in range(4)]
        features = []
        for observation, available, revision, value in ((0, 0, 0, 1), (0, 1, 1, 2),
                                                       (1, 1, 0, 3), (2, 0, 0, 4)):
            features.append({"entity_id": "E", "metric": "m", "observation_day": str(observation),
                             "available_day": str(available), "revision": str(revision), "value": str(value)})
        _, panel = public_solver.assemble_panel(prices, features, securities, ["m"])
        self.assertEqual([panel["E", day]["m"] for day in range(3)], [1, 3, 4])

    def test_expiring_ticker_mapping_does_not_merge_entities(self):
        securities = [
            {"entity_id": "A", "ticker": "T", "valid_from": "0", "valid_to": "0"},
            {"entity_id": "A", "ticker": "N", "valid_from": "1", "valid_to": "2"},
            {"entity_id": "B", "ticker": "U", "valid_from": "0", "valid_to": "0"},
            {"entity_id": "B", "ticker": "T", "valid_from": "1", "valid_to": "2"},
        ]
        prices = [{"ticker": ticker, "day": str(day), "close": str(price), "shares_per_original": "1"}
                  for ticker, day, price in (("T", 0, 100), ("N", 1, 110), ("N", 2, 121),
                                            ("U", 0, 200), ("T", 1, 180), ("T", 2, 162))]
        _, panel = public_solver.assemble_panel(prices, [], securities, ["m"])
        self.assertAlmostEqual(panel["A", 0]["forward_return"], 0.1)
        self.assertAlmostEqual(panel["B", 0]["forward_return"], -0.1)
        self.assertIsNone(panel["A", 0]["m"])

    def test_strict_schema_rejects_malformed_structures(self):
        base = self.solutions["signal"]
        malformed = [None, [], {}, {**base, "unexpected": 0}, {k: v for k, v in base.items() if k != "audit"}]
        def changed(mutate):
            submission = copy.deepcopy(base)
            mutate(submission)
            malformed.append(submission)
        changed(lambda s: s.update(audit={}))
        changed(lambda s: s["audit"].pop())
        changed(lambda s: s["trials"].pop())
        changed(lambda s: s["audit"].__setitem__(1, copy.deepcopy(s["audit"][0])))
        changed(lambda s: s["trials"].__setitem__(1, copy.deepcopy(s["trials"][0])))
        changed(lambda s: s["audit"][0].update(entity_id="UNKNOWN"))
        changed(lambda s: s["audit"][0].update(field="UNKNOWN"))
        changed(lambda s: s["trials"][0].update(id="UNKNOWN"))
        changed(lambda s: s["audit"][0].update(extra=1))
        changed(lambda s: s["trials"][0].update(extra=1))
        changed(lambda s: s["audit"][0].update(day=True))
        changed(lambda s: s["audit"][0].update(day=0.0))
        changed(lambda s: s["audit"][0].update(value=True))
        changed(lambda s: s["trials"][0].update(mean_net=True))
        changed(lambda s: s["trials"][0].update(mean_net=None))
        changed(lambda s: s["trials"][0].update(mean_net="0"))
        changed(lambda s: s["audit"][0].update(value=10 ** 1000))
        changed(lambda s: s["trials"][0].update(mean_net=10 ** 1000))
        changed(lambda s: s.update(selected=[]))
        changed(lambda s: s.update(decision=True))
        changed(lambda s: s.update(decision="abstain"))
        changed(lambda s: s.update(selected=None))
        for value in (math.nan, math.inf, -math.inf):
            changed(lambda s, value=value: s["audit"][0].update(value=value))
            changed(lambda s, value=value: s["trials"][0].update(sharpe=value))
        for index, submission in enumerate(malformed):
            with self.subTest(index=index):
                result = score_submission(self.episodes["signal"], submission)
                self.assertEqual(result["reward"], 0)
                self.assertEqual(result["metrics"]["schema_valid"], 0)

    def test_wrong_audit_or_trial_cannot_collect_economic_credit(self):
        for section in ("audit", "trials"):
            submission = copy.deepcopy(self.solutions["signal"])
            if section == "audit":
                submission[section][0]["value"] += 0.01
            else:
                submission[section][0]["mean_net"] += 0.01
            result = score_submission(self.episodes["signal"], submission)
            self.assertEqual(result["metrics"]["schema_valid"], 1)
            self.assertEqual(result["metrics"]["decision_correct"], 1)
            self.assertEqual(result["metrics"]["economic_quality"], 0)
            self.assertLess(result["reward"], 0.8)

    def test_always_abstain_does_not_solve_signal_episode(self):
        submission = copy.deepcopy(self.solutions["signal"])
        submission.update(decision="abstain", selected=None)
        result = score_submission(self.episodes["signal"], submission)
        self.assertEqual(result["metrics"]["decision_correct"], 0)
        self.assertEqual(result["metrics"]["economic_quality"], 0)
        self.assertAlmostEqual(result["reward"], 0.6)

    def test_validation_selection_and_withheld_quality_are_distinct(self):
        episode = copy.deepcopy(self.episodes["signal"])
        chosen = self.solutions["signal"]["selected"]
        episode["private"]["hidden_utilities"] = {candidate["id"]: 1.0 for candidate in CANDIDATES}
        episode["private"]["hidden_utilities"][chosen] = 0.4
        result = score_submission(episode, self.solutions["signal"])
        self.assertEqual(result["metrics"]["decision_correct"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0.4)
        self.assertAlmostEqual(result["reward"], 0.88)

    def test_trial_order_does_not_change_selection_or_grading(self):
        submission = copy.deepcopy(self.solutions["signal"])
        submission["audit"].reverse()
        submission["trials"].reverse()
        self.assertEqual(score_submission(self.episodes["signal"], submission)["reward"], 1)
        self.assertEqual(select_candidate([{"id": "z", "screening_score": 1},
                                           {"id": "a", "screening_score": 1}]), "a")
        self.assertIsNone(select_candidate([{"id": "a", "screening_score": 0}]))

    def test_entry_exit_costs_sample_variance_and_zero_variance(self):
        panel = [{"entity_id": f"E{asset}", "day": day, "metric_a": asset, "forward_return": 0}
                 for day in range(2) for asset in range(4)]
        contract = {"cost_bps": 20, "borrow_bps_daily": 0}
        candidate = {"feature": "metric_a", "direction": 1, "gross": 1}
        metrics = portfolio_metrics(panel, candidate, contract, 0, 2)
        self.assertEqual(metrics, {"mean_net": -0.002, "sharpe": 0,
                                   "mean_turnover": 1, "screening_score": -0.002})
        for row in panel:
            row["metric_a"] = None
        self.assertTrue(all(value == 0 for value in portfolio_metrics(panel, candidate, contract, 0, 2).values()))

        daily = (0.01, 0.02, -0.01)
        panel = [{"entity_id": f"E{asset}", "day": day, "metric_a": asset,
                  "forward_return": 4 * daily[day] if asset == 3 else 0}
                 for day in range(3) for asset in range(4)]
        metrics = portfolio_metrics(panel, candidate, {"cost_bps": 0, "borrow_bps_daily": 1}, 0, 3)
        mean = sum(daily) / 3 - 0.5 / 10000
        deviation = math.sqrt(sum((value - sum(daily) / 3) ** 2 for value in daily) / 2)
        self.assertAlmostEqual(metrics["mean_net"], mean)
        self.assertAlmostEqual(metrics["sharpe"], math.sqrt(252) * mean / deviation)
        self.assertAlmostEqual(metrics["screening_score"], mean - 2 * deviation / math.sqrt(3))

    def test_missing_or_short_evaluator_key_is_rejected(self):
        for key in (None, "short-key"):
            with self.subTest(key=key), patch.dict(os.environ):
                if key is None:
                    os.environ.pop("FINANCIAL_RESEARCH_KEY", None)
                else:
                    os.environ["FINANCIAL_RESEARCH_KEY"] = key
                with self.assertRaises(ValueError):
                    build_episode(0)

    def test_key_rotation_changes_paths_and_restoring_key_reproduces_episode(self):
        expected = self.episodes["signal"]
        with patch.dict(os.environ, {"FINANCIAL_RESEARCH_KEY": "different-public-test-key-only-123456789"}):
            changed = build_episode(0)
            self.assertNotEqual(changed["public_files"], expected["public_files"])
            self.assertNotEqual(changed["private"]["hidden_utilities"], expected["private"]["hidden_utilities"])
        self.assertEqual(build_episode(0), expected)

    def test_split_ids_are_disjoint_and_resolution_is_deterministic(self):
        registry = _registry(os.environ["FINANCIAL_RESEARCH_KEY"])
        by_split = {split: {key for key, (_, actual, _) in registry.items() if actual == split} for split in SPLITS}
        for split, (low, high) in SPLITS.items():
            self.assertEqual(len(by_split[split]), (high - low) * len(FAMILIES))
        self.assertFalse(by_split["train"] & by_split["dev"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["dev"] & by_split["test"])
        for episode in self.episodes.values():
            self.assertEqual(resolve_episode(episode["id"]), episode)
            self.assertFalse(any(family in episode["id"] for family in FAMILIES))
        for seed, split in ((True, "train"), (-1, "train"), (128, "train"), (0, "dev"),
                            (160, "dev"), (192, "test"), (0, "unknown")):
            with self.subTest(seed=seed, split=split), self.assertRaises(ValueError):
                build_episode(seed, split)
        with self.assertRaises(ValueError):
            resolve_episode("fr_unknown")


if __name__ == "__main__":
    unittest.main()
