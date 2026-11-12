"""Hand-calculated regressions for the experimental execution contract."""

import copy
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))

from financial_portfolio.analytics import estimate_risk, execute, forecast_vol, simulate, target_weights


class PortfolioMathTests(unittest.TestCase):
    def setUp(self):
        self.entities = ["A", "B", "C", "D"]
        self.risk = {"entities": self.entities, "covariance": [
            [0.0001 * (i + 1) ** 2 if i == j else 0 for j in range(4)] for i in range(4)]}
        self.contract = {"training_start": 0, "training_end": 3,
                         "covariance_diagonal_shrinkage": 0.5,
                         "reference_nav": 100.0, "dollar_adv": dict.fromkeys(self.entities, 100.0),
                         "participation_limit": 1.0, "gross_cap": 1.0, "per_asset_cap": 0.35,
                         "annual_vol_cap": 10.0, "spread_bps": 2.0, "impact_bps": 10.0,
                         "borrow_bps_daily": 1.0, "validation_start": 3, "active_end": 5,
                         "validation_end": 6}
        self.candidate = {"id": "test", "feature": "metric_a", "direction": 1,
                          "gross": 1.0, "allocation": "equal"}
        self.observations = [{"entity_id": entity, "metric_a": i, "forward_return": 0.0}
                             for i, entity in enumerate(self.entities)]

    def test_sample_covariance_and_diagonal_shrinkage(self):
        panel = [{"entity_id": entity, "day": day, "forward_return": ret}
                 for entity in self.entities for day, ret in enumerate((-.01, 0, .01, 10))]
        covariance = estimate_risk(panel, self.contract)["covariance"]
        self.assertAlmostEqual(covariance[0][0], .0001)
        self.assertAlmostEqual(covariance[0][1], .00005)
        panel[-1]["forward_return"] = -1000
        self.assertEqual(estimate_risk(panel, self.contract)["covariance"], covariance)

    def test_inverse_vol_normalizes_each_leg(self):
        candidate = {**self.candidate, "allocation": "inverse_vol", "gross": .5}
        weights = target_weights(self.observations, candidate, self.contract, self.risk)
        self.assertAlmostEqual(weights["A"], -1 / 6)
        self.assertAlmostEqual(weights["B"], -1 / 12)
        self.assertAlmostEqual(weights["C"], 1 / 7)
        self.assertAlmostEqual(weights["D"], 3 / 28)
        self.assertAlmostEqual(sum(weights.values()), 0)
        self.assertAlmostEqual(sum(abs(w) for w in weights.values()), .5)

    def test_global_scaling_preserves_neutrality_and_caps(self):
        contract = {**self.contract, "annual_vol_cap": .02, "per_asset_cap": .04}
        weights = target_weights(self.observations, self.candidate, contract, self.risk)
        self.assertAlmostEqual(sum(weights.values()), 0)
        self.assertLessEqual(max(abs(w) for w in weights.values()), .04 + 1e-15)
        self.assertLessEqual(forecast_vol(weights, self.risk), .02 + 1e-15)

    def test_limited_fill_costs_use_actual_trades_and_exposures(self):
        contract = {**self.contract, "participation_limit": .1,
                    "dollar_adv": {"A": 20.0, "B": 100.0, "C": 100.0, "D": 100.0}}
        target = {"A": -.2, "B": 0.0, "C": .2, "D": 0.0}
        weights, result = execute(dict.fromkeys(self.entities, 0.0), target, contract, self.risk)
        self.assertAlmostEqual(result["execution_fraction"], .1)
        self.assertAlmostEqual(weights["A"], -.02)
        self.assertAlmostEqual(result["turnover"], .04)
        self.assertAlmostEqual(result["max_participation"], .1)
        impact = .001 * (.02 * math.sqrt(.1) + .02 * math.sqrt(.02))
        self.assertAlmostEqual(result["impact_cost"], impact)
        self.assertAlmostEqual(result["total_cost"], .0002 * .04 + impact + .0001 * .02)

    def test_impact_is_convex_in_unconstrained_trade_size(self):
        target = {"A": -.1, "B": -.1, "C": .1, "D": .1}
        flat = dict.fromkeys(self.entities, 0.0)
        _, small = execute(flat, target, self.contract, self.risk)
        _, large = execute(flat, {e: 2 * w for e, w in target.items()}, self.contract, self.risk)
        self.assertAlmostEqual(large["impact_cost"] / small["impact_cost"], 2 ** 1.5)

    def test_round_trip_charges_entry_and_exit_with_common_denominator(self):
        panel = [{**row, "day": day} for day in range(3, 6) for row in self.observations]
        metrics, history = simulate(panel, self.candidate, self.contract, self.risk)
        self.assertEqual([r["turnover"] for r in history], [1.0, 0.0, 1.0])
        self.assertEqual(history[-1]["actual_gross"], 0)
        total_cost = 2 * (.0002 + .001 * .5) + 2 * .0001 * .5
        self.assertAlmostEqual(metrics["mean_net"], -total_cost / 3)
        self.assertAlmostEqual(metrics["mean_turnover"], 2 / 3)

    def test_missing_features_mean_cash_target_with_capped_unwind(self):
        target = target_weights(self.observations[:3], self.candidate, self.contract, self.risk)
        self.assertEqual(target, dict.fromkeys(self.entities, 0))
        previous = {"A": -.25, "B": -.25, "C": .25, "D": .25}
        weights, result = execute(previous, target, {**self.contract, "participation_limit": .05}, self.risk)
        self.assertAlmostEqual(result["execution_fraction"], .2)
        self.assertAlmostEqual(sum(abs(w) for w in weights.values()), .8)

    def test_fixed_winddown_bound_liquidates_without_breaching_caps(self):
        rng = random.Random(1701)
        for _ in range(25):
            contract = copy.deepcopy(self.contract)
            contract["participation_limit"] = .1
            contract["dollar_adv"] = {e: rng.uniform(20, 150) for e in self.entities}
            previous = {"A": -.35, "B": -.15, "C": .35, "D": .15}
            k = math.ceil(max(.35 / (.1 * adv / 100) for adv in contract["dollar_adv"].values()))
            for _ in range(k):
                previous, metrics = execute(previous, dict.fromkeys(self.entities, 0.0), contract, self.risk)
                self.assertLessEqual(metrics["max_participation"], .1 + 1e-14)
                self.assertAlmostEqual(sum(previous.values()), 0)
            self.assertEqual(previous, dict.fromkeys(self.entities, 0.0))

    def test_prior_weights_do_not_drift_with_realized_returns(self):
        observations = [{**row, "forward_return": .02 if row["entity_id"] == "D" else 0}
                        for row in self.observations]
        panel = [{**row, "day": day, "forward_return": row["forward_return"] if day < 5 else 0}
                 for day in range(3, 6) for row in observations]
        _, history = simulate(panel, self.candidate, self.contract, self.risk)
        self.assertEqual(history[0]["weights"], history[1]["weights"])
        self.assertEqual(history[1]["turnover"], 0)

    def test_winddown_length_cannot_be_silently_shortened(self):
        panel = [{**row, "day": day} for day in range(3, 6) for row in self.observations]
        contract = {**self.contract, "participation_limit": .05}
        with self.assertRaisesRegex(ValueError, "winddown failed"):
            simulate(panel, self.candidate, contract, self.risk)


if __name__ == "__main__":
    unittest.main()
