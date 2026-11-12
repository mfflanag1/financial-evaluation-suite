"""Independent public reconstruction and analytical portfolio controls."""

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
PACKAGE = ROOT / "environments" / "financial_portfolio"
sys.path.insert(0, str(ROOT / "environments" / "financial_research" / "src"))
sys.path.insert(0, str(PACKAGE / "src"))

from financial_portfolio.analytics import CANDIDATES, estimate_risk, execute, simulate, target_weights
from financial_portfolio.generator import FAMILIES, build_episode
from financial_research.analytics import clean_public
from financial_research.verifier import reference_submission, score_submission

REFERENCE = PACKAGE / "src" / "financial_portfolio" / "reference.py"
spec = importlib.util.spec_from_file_location("public_portfolio_reference", REFERENCE)
public_solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_solver)


def run_public_solution(files):
    with tempfile.TemporaryDirectory(prefix="portfolio-public-test-") as directory:
        workspace = Path(directory)
        for filename, contents in files.items():
            (workspace / filename).write_text(contents, encoding="utf-8")
        (workspace / "research.py").write_bytes(REFERENCE.read_bytes())
        # No evaluator key, package import path, credentials or process state.
        subprocess.run([sys.executable, "-I", "research.py"], cwd=workspace,
                       env={"PATH": os.defpath}, check=True, capture_output=True,
                       text=True, timeout=20)
        return json.loads((workspace / "policy.json").read_text())


def changed_csv(payload, mutate):
    rows = list(csv.DictReader(io.StringIO(payload)))
    mutate(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class PublicPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key_patch = patch.dict(os.environ, {
            "FINANCIAL_RESEARCH_KEY": "public-portfolio-test-key-only-not-for-real-data",
        })
        cls.key_patch.start()
        cls.addClassCleanup(cls.key_patch.stop)
        cls.episodes = {family: build_episode(index) for index, family in enumerate(FAMILIES)}
        cls.solutions = {family: run_public_solution(episode["public_files"])
                         for family, episode in cls.episodes.items()}

    def test_public_reference_matches_four_family_grading_and_canonical_selection(self):
        self.assertEqual(set(FAMILIES), {"impact_sizing", "liquidity_tight", "risk_concentration", "no_signal"})
        for family, episode in self.episodes.items():
            with self.subTest(family=family):
                solution = self.solutions[family]
                actual = score_submission(episode, solution)
                expected = score_submission(episode, reference_submission(episode))
                self.assertEqual(len(solution["trials"]), 48)
                self.assertEqual(actual, expected)
                self.assertEqual(actual["metrics"]["audit_accuracy"], 1)
                self.assertEqual(actual["metrics"]["trial_accuracy"], 1)
                self.assertEqual(actual["metrics"]["decision_correct"], 1)
                for trial in solution["trials"]:
                    self.assertEqual(trial["screening_score"], round(trial["screening_score"], 12))
                winner = min(solution["trials"], key=lambda trial: (-trial["screening_score"], trial["id"]))
                self.assertEqual(solution["selected"], winner["id"] if winner["screening_score"] > 0 else None)

    def test_source_duplicates_and_order_do_not_change_analysis(self):
        files = dict(self.episodes["impact_sizing"]["public_files"])
        def duplicate_and_shuffle(rows):
            rows.extend(copy.deepcopy(rows[:17]))
            rows.reverse()
        for filename in ("prices.csv", "features.csv"):
            files[filename] = changed_csv(files[filename], duplicate_and_shuffle)
        self.assertEqual(run_public_solution(files), self.solutions["impact_sizing"])

    def test_removing_split_adjustments_fails_source_audit_and_economic_gate(self):
        episode = self.episodes["impact_sizing"]
        files = dict(episode["public_files"])
        def remove_splits(rows):
            for row in rows:
                row["shares_per_original"] = "1"
        files["prices.csv"] = changed_csv(files["prices.csv"], remove_splits)
        result = score_submission(episode, run_public_solution(files))
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_omitting_spread_impact_and_borrow_fails_independent_accounting(self):
        episode = self.episodes["impact_sizing"]
        files = dict(episode["public_files"])
        contract = json.loads(files["contract.json"])
        contract.update(spread_bps=0, impact_bps=0, borrow_bps_daily=0)
        files["contract.json"] = json.dumps(contract)
        result = score_submission(episode, run_public_solution(files))
        self.assertLess(result["metrics"]["trial_accuracy"], 1)
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_ignoring_participation_limit_fails_liquidity_checkpoints(self):
        episode = self.episodes["liquidity_tight"]
        files = dict(episode["public_files"])
        contract = json.loads(files["contract.json"])
        contract["participation_limit"] = 1000
        files["contract.json"] = json.dumps(contract)
        result = score_submission(episode, run_public_solution(files))
        self.assertLess(result["metrics"]["trial_accuracy"], 1)
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_estimating_covariance_with_validation_returns_is_detected(self):
        episode = self.episodes["risk_concentration"]
        files = dict(episode["public_files"])
        contract = json.loads(files["contract.json"])
        contract["training_end"] = contract["active_end"]
        files["contract.json"] = json.dumps(contract)
        result = score_submission(episode, run_public_solution(files))
        self.assertLess(result["metrics"]["trial_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)

    def test_execution_constraints_and_complete_winddown_hold_for_every_candidate(self):
        for family, episode in self.episodes.items():
            contract = json.loads(episode["public_files"]["contract.json"])
            panel = clean_public(episode["public_files"])
            risk = estimate_risk(panel, contract)
            expected_tail = math.ceil(max(contract["per_asset_cap"] / (
                contract["participation_limit"] * adv / contract["reference_nav"])
                for adv in contract["dollar_adv"].values()))
            self.assertEqual(contract["winddown_sessions"], expected_tail)
            self.assertEqual(contract["validation_end"], contract["active_end"] + expected_tail)
            for candidate in CANDIDATES:
                with self.subTest(family=family, candidate=candidate["id"]):
                    trial, history = simulate(panel, candidate, contract, risk)
                    self.assertEqual(len(history), 48 + expected_tail)
                    self.assertTrue(all(abs(value) <= 1e-12 for value in history[-1]["weights"].values()))
                    for day in history:
                        self.assertLessEqual(day["actual_gross"], contract["gross_cap"] + 1e-12)
                        self.assertLessEqual(max(abs(value) for value in day["weights"].values()),
                                             contract["per_asset_cap"] + 1e-12)
                        self.assertAlmostEqual(sum(day["weights"].values()), 0, places=12)
                        self.assertLessEqual(day["forecast_vol"], contract["annual_vol_cap"] + 1e-12)
                        self.assertLessEqual(day["max_participation"], contract["participation_limit"] + 1e-12)
                    self.assertAlmostEqual(trial["mean_net"], sum(day["net_return"] for day in history) / len(history))
                    self.assertTrue(any(day["turnover"] > 0 for day in history if day["day"] >= contract["active_end"]))

    def test_forged_portfolio_checkpoint_cannot_receive_economic_credit(self):
        episode = self.episodes["impact_sizing"]
        submission = copy.deepcopy(self.solutions["impact_sizing"])
        checkpoint = next(row for row in submission["audit"] if row["field"] == "actual_gross")
        checkpoint["value"] += 0.01
        result = score_submission(episode, submission)
        self.assertLess(result["metrics"]["audit_accuracy"], 1)
        self.assertEqual(result["metrics"]["economic_quality"], 0)


class PortfolioMathematicsTests(unittest.TestCase):
    def setUp(self):
        self.entities = ["A", "B", "C", "D"]
        self.contract = {
            "training_start": 0, "training_end": 3, "covariance_diagonal_shrinkage": 0.5,
            "gross_cap": 1, "per_asset_cap": 0.35, "annual_vol_cap": 1000,
            "dollar_adv": {"A": 200000, "B": 400000, "C": 400000, "D": 400000},
            "reference_nav": 1000000, "participation_limit": 0.1,
            "spread_bps": 10, "impact_bps": 100, "borrow_bps_daily": 1,
        }
        self.covariance = [[0.0001 * (i + 1) ** 2 if i == j else 0 for j in range(4)] for i in range(4)]
        self.risk = {"entities": self.entities, "covariance": self.covariance}

    def test_candidate_universe_is_exactly_the_declared_cross_product(self):
        self.assertEqual(len(CANDIDATES), 48)
        self.assertEqual(len({candidate["id"] for candidate in CANDIDATES}), 48)
        self.assertEqual({candidate["gross"] for candidate in CANDIDATES}, {0.25, 0.5, 1.0})
        self.assertEqual({candidate["allocation"] for candidate in CANDIDATES}, {"equal", "inverse_vol"})

    def test_sample_covariance_shrinkage_and_training_boundary(self):
        entities = ["A", "B"]
        values = {"A": (-0.01, 0, 0.01, 1000), "B": (0.02, 0, -0.02, -1000)}
        panel = [{"entity_id": entity, "day": day, "forward_return": values[entity][day]}
                 for day in range(4) for entity in entities]
        contract = {**self.contract, "dollar_adv": {entity: 1000 for entity in entities}}
        covariance = estimate_risk(panel, contract)["covariance"]
        independent = public_solver.training_covariance(
            {(row["entity_id"], row["day"]): row for row in panel}, contract, entities)
        expected = [[0.0001, -0.0001], [-0.0001, 0.0004]]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(covariance[i][j], expected[i][j])
                self.assertAlmostEqual(independent[i][j], expected[i][j])

    def test_inverse_vol_normalizes_each_leg_then_scales_the_whole_portfolio(self):
        observations = [{"entity_id": entity, "metric_a": index} for index, entity in enumerate(self.entities)]
        candidate = {"feature": "metric_a", "direction": 1, "gross": 1, "allocation": "inverse_vol"}
        target = target_weights(observations, candidate, self.contract, self.risk)
        expected = [-1 / 3, -1 / 6, 2 / 7, 3 / 14]
        for entity, weight in zip(self.entities, expected):
            self.assertAlmostEqual(target[entity], weight)
        capped_contract = {**self.contract, "annual_vol_cap": 0.02}
        capped = target_weights(observations, candidate, capped_contract, self.risk)
        uncapped_volatility = math.sqrt(252 * sum(value * value * self.covariance[i][i]
                                                 for i, value in enumerate(expected)))
        factor = 0.02 / uncapped_volatility
        for entity, weight in zip(self.entities, expected):
            self.assertAlmostEqual(capped[entity], weight * factor)
        per_asset_contract = {**self.contract, "per_asset_cap": 0.1}
        per_asset = target_weights(observations, candidate, per_asset_contract, self.risk)
        self.assertAlmostEqual(max(abs(value) for value in per_asset.values()), 0.1)
        self.assertAlmostEqual(sum(per_asset.values()), 0)

    def test_common_fill_fraction_preserves_neutrality_and_all_cost_components(self):
        old = dict.fromkeys(self.entities, 0.0)
        target = dict(zip(self.entities, (0.25, -0.25, 0.25, -0.25)))
        actual, metrics = execute(old, target, self.contract, self.risk)
        self.assertAlmostEqual(metrics["execution_fraction"], 0.08)
        for entity, expected in zip(self.entities, (0.02, -0.02, 0.02, -0.02)):
            self.assertAlmostEqual(actual[entity], expected)
        self.assertAlmostEqual(sum(actual.values()), 0)
        self.assertAlmostEqual(metrics["turnover"], 0.08)
        self.assertAlmostEqual(metrics["max_participation"], 0.1)
        impact = 100 / 10000 * (0.02 * math.sqrt(0.1) + 0.06 * math.sqrt(0.05))
        self.assertAlmostEqual(metrics["impact_cost"], impact)
        self.assertAlmostEqual(metrics["total_cost"], 10 / 10000 * 0.08 + impact + 1 / 10000 * 0.04)
        independent, independent_metrics = public_solver.execute_target(
            [0] * 4, list(target.values()), self.contract, self.entities, self.covariance)
        self.assertEqual(independent, list(actual.values()))
        for key, value in metrics.items():
            self.assertAlmostEqual(value, independent_metrics[key])

    def test_missing_features_trigger_a_capped_unwind_instead_of_instant_cash(self):
        observations = [{"entity_id": entity, "metric_a": index if index < 3 else None}
                        for index, entity in enumerate(self.entities)]
        candidate = {"feature": "metric_a", "direction": 1, "gross": 1, "allocation": "equal"}
        target = target_weights(observations, candidate, self.contract, self.risk)
        self.assertTrue(all(value == 0 for value in target.values()))
        previous = dict(zip(self.entities, (0.25, -0.25, 0.25, -0.25)))
        actual, metrics = execute(previous, target, self.contract, self.risk)
        self.assertAlmostEqual(metrics["execution_fraction"], 0.08)
        self.assertAlmostEqual(actual["A"], 0.23)
        self.assertAlmostEqual(sum(actual.values()), 0)

    def test_winddown_sessions_all_enter_return_and_turnover_denominators(self):
        winddown = math.ceil(self.contract["per_asset_cap"] / 0.02)
        contract = {**self.contract, "validation_start": 0, "active_end": 1,
                    "validation_end": 1 + winddown}
        panel = [{"entity_id": entity, "day": day, "metric_a": index,
                  "forward_return": 0.01 * index if day == 0 else 0}
                 for day in range(1 + winddown) for index, entity in enumerate(self.entities)]
        candidate = {"id": "test", "feature": "metric_a", "direction": 1,
                     "gross": 1, "allocation": "equal"}
        trial, history = simulate(panel, candidate, contract, self.risk)
        self.assertEqual(len(history), 19)
        self.assertGreater(history[1]["turnover"], 0)
        self.assertLess(history[1]["net_return"], 0)
        self.assertTrue(all(row["net_return"] == 0 for row in history[2:]))
        self.assertTrue(all(value == 0 for value in history[-1]["weights"].values()))
        self.assertAlmostEqual(trial["mean_net"], sum(row["net_return"] for row in history) / 19)
        self.assertAlmostEqual(trial["mean_turnover"], sum(row["turnover"] for row in history) / 19)

    def test_risk_equivalent_policies_have_canonical_scores_before_selection(self):
        contract = {**self.contract, "validation_start": 3, "active_end": 6,
                    "validation_end": 7, "annual_vol_cap": 0.001}
        panel = [{"entity_id": entity, "day": day, "metric_a": index,
                  "forward_return": 0.002 * (day + 1) * (index - 1.5) if day < 6 else 0}
                 for day in range(7) for index, entity in enumerate(self.entities)]
        risk = estimate_risk(panel, contract)
        indexed = {(row["entity_id"], row["day"]): row for row in panel}
        candidates = [{"id": identity, "feature": "metric_a", "direction": 1,
                       "gross": gross, "allocation": "equal"}
                      for identity, gross in (("z", 0.5), ("a", 1))]
        public_trials, private_trials = [], []
        for candidate in candidates:
            independent, _ = public_solver.evaluate_candidate(
                candidate, indexed, contract, self.entities, risk["covariance"])
            trusted, _ = simulate(panel, candidate, contract, risk)
            public_trials.append(independent)
            private_trials.append(trusted)
        self.assertEqual(public_trials[0]["screening_score"], public_trials[1]["screening_score"])
        self.assertEqual([trial["screening_score"] for trial in public_trials],
                         [trial["screening_score"] for trial in private_trials])
        self.assertEqual(min(public_trials, key=lambda trial: (-trial["screening_score"], trial["id"]))["id"], "a")


if __name__ == "__main__":
    unittest.main()
