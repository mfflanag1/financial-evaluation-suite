"""Primal/dual certificates and financing invariants for allocation cases."""
import json
import math
from pathlib import Path
import unittest

from quality import allocation_numerical as a
from quality.numerical import solve_linear

ROOT=Path(__file__).resolve().parents[1]


def inputs(cid):return json.loads((ROOT/f"questions/legacy/quant100/questions/{cid}.json").read_text())["inputs"]


class AllocationCertificates(unittest.TestCase):
    def test_restricted_short_cash_cannot_reduce_long_borrowing(self):
        data=inputs("q001");base=a.q001(data)[0]
        data["short_usd_m"]+=50;changed=a.q001(data)[0]
        self.assertEqual(base["borrowing"],changed["borrowing"])
        self.assertAlmostEqual(changed["pnl"]-base["pnl"],50*(data["short_collateral_rebate"]-data["short_security_total_return"]))
        self.assertAlmostEqual(changed["gross_leverage"]-base["gross_leverage"],50/data["equity_usd_m"])

    def test_hedge_factor_residuals_and_homogeneity(self):
        data=inputs("q003");base,certificate=a.q003(data)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["remaining_factor_exposures"]))
        data["existing_factor_exposure"]=[2*x for x in data["existing_factor_exposure"]]
        changed,_=a.q003(data)
        for first,second in zip(base["hedge"],changed["hedge"]):self.assertAlmostEqual(second,2*first)
        self.assertGreater(base["gross_notional"],abs(base["net_notional"]))

    def test_two_asset_risk_parity_correlation_invariance_and_cap_release(self):
        data=inputs("q004");base,_=a.q004(data)
        data["correlation"]=.6;changed,_=a.q004(data)
        self.assertAlmostEqual(changed["unconstrained_a"],base["unconstrained_a"])
        self.assertNotAlmostEqual(changed["a_risk_share"],base["a_risk_share"])
        data["asset_a_weight_cap"]=1;relaxed,certificate=a.q004(data)
        self.assertAlmostEqual(relaxed["a_risk_share"],.5)
        self.assertAlmostEqual(certificate["contribution_sum_equals_variance"],0)

    def test_active_hedge_removes_market_variance_and_mean(self):
        data=inputs("q005");base,_=a.q005(data)
        data["market_variance"]*=100;data["market_expected_return"]+=.3
        changed,certificate=a.q005(data)
        self.assertEqual(changed,base)
        self.assertEqual(certificate["active_beta_after_overlay"],0)

    def test_box_uncertainty_uses_high_mean_for_short_and_zero_radius_restores_nominal(self):
        data=inputs("q007");_,certificate=a.q007(data)
        corner=certificate["menu_results"]["D"]["worst_corner"]
        self.assertEqual(corner[0],data["nominal_mean"][0]-data["mean_box_radius"][0])
        self.assertEqual(corner[1],data["nominal_mean"][1]+data["mean_box_radius"][1])
        data["mean_box_radius"]=[0,0];answer,_=a.q007(data)
        self.assertEqual(answer["robust_choice"],answer["nominal_choice"])

    def test_turnover_counts_both_legs_and_has_correct_shadow_units(self):
        data=inputs("q008");answer,_=a.q008(data)
        self.assertAlmostEqual(answer["turnover"],2*(answer["weight_a"]-data["initial_weight_a"]))
        data["total_absolute_weight_change_cap"]*=.5
        reduced,_=a.q008(data)
        self.assertAlmostEqual(reduced["weight_a"]-data["initial_weight_a"],.5*(answer["weight_a"]-data["initial_weight_a"]))
        self.assertAlmostEqual(answer["turnover_shadow_price"],.01,places=9)

    def test_proportional_cost_no_trade_band_includes_both_boundaries(self):
        data=inputs("q009");base=a.q009(data)[0]
        for alpha in base["no_trade_alpha_interval"]:
            data["base_alpha"]=alpha
            self.assertAlmostEqual(a.q009(data)[0]["base_weight"],data["risky_weight_before"])
        data["base_alpha"]=base["no_trade_alpha_interval"][0]-.01
        self.assertLess(a.q009(data)[0]["base_weight"],data["risky_weight_before"])

    def test_tracking_error_caps_the_relaxed_active_position(self):
        data=inputs("q010");answer,certificate=a.q010(data)
        self.assertEqual(answer["relaxed_binding"],"tracking_error")
        self.assertAlmostEqual(answer["relaxed_a"],certificate["tracking_error_feasible_a_interval"][1])
        self.assertLess(answer["base_tracking_error"],data["tracking_error_limit"])
        data["tracking_error_limit"]=1
        self.assertEqual(a.q010(data)[0]["relaxed_binding"],"concentration")

    def test_active_set_cap_multiplier_matches_envelope_and_releases(self):
        data=inputs("q011");answer,certificate=a.q011(data)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["free_coordinate_stationarity"]))
        data["asset_a_upper"]+=1e-6;changed=a.q011(data)[0]
        self.assertAlmostEqual((answer["variance"]-changed["variance"])/1e-6,answer["cap_multiplier"],delta=1e-7)
        data["asset_a_upper"]=.8;relaxed=a.q011(data)[0]
        self.assertEqual(relaxed["cap_multiplier"],0)
        self.assertAlmostEqual(relaxed["weights"][1],data["asset_b_lower"])

    def test_neutral_signal_portfolio_is_minimal_along_feasible_null_direction(self):
        data=inputs("q012");answer,certificate=a.q012(data)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["constraint_residuals"]+certificate["stationarity_residuals"]))
        rows=[[1]*4,data["factor_loadings"],data["signal_loadings"]]
        direction=solve_linear([row[:3] for row in rows],[-row[3] for row in rows])+[1]
        for sign in (-1,1):
            changed=[w+sign*.05*d for w,d in zip(answer["weights"],direction)]
            value=sum(w*w*v for w,v in zip(changed,data["diagonal_covariance"]))
            self.assertGreater(value,answer["variance"])
        data["required_signal_exposure"]*=2;scaled=a.q012(data)[0]
        self.assertAlmostEqual(scaled["variance"],4*answer["variance"])

    def test_ticket_cost_threshold_tie_retains_original_position(self):
        data=inputs("q014");answer,_=a.q014(data)
        data["fixed_cost"]=answer["indifference_fixed_cost"]
        tied,certificate=a.q014(data)
        self.assertEqual(tied["base_weight"],data["initial_weight"])
        self.assertEqual(tied["incremental_utility"],0)
        self.assertTrue(certificate["tie_prefers_no_trade"])

    def test_ellipsoid_certificate_is_feasible_and_cash_threshold_removes_risk(self):
        data=inputs("q015");answer,certificate=a.q015(data)
        self.assertLessEqual(sum(answer["weights"]),1)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["unconstrained_stationarity_residuals"]))
        self.assertLess(certificate["volatility"],data["volatility_cap"])
        data["ellipsoid_radius"]=answer["cash_threshold_radius"]
        cash,_=a.q015(data)
        self.assertEqual(cash["weights"],[0,0]);self.assertEqual(cash["robust_utility"],0)

    def test_partial_correlation_completions_require_different_allocations(self):
        data=inputs("q016");answer,_=a.q016(data)
        self.assertEqual(answer["identification"],"not_identified")
        self.assertFalse(answer["same_decision"])
        self.assertGreater(abs(answer["completion_a_weights"][1]-answer["completion_a_weights"][0]),.1)
        self.assertTrue(all(data["unknown_correlation_interval"][0]<=r<=data["unknown_correlation_interval"][1] for r in data["completion_correlations"]))

    def test_tier_entry_boundary_respects_one_sided_funding_cost(self):
        data=inputs("q022");answer,certificate=a.q022(data)
        self.assertAlmostEqual(certificate["optimum_at_entry_threshold"],1+data["cheap_borrowing_capacity"])
        data["updated_expected_excess_return"]=answer["high_tier_entry_alpha"]+1e-4
        self.assertGreater(a.q022(data)[0]["updated_leverage"],1+data["cheap_borrowing_capacity"])

    def test_risk_budget_composition_does_not_change_with_margin_or_covariance_scale(self):
        data=inputs("q024");answer,certificate=a.q024(data)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["risk_budget_residuals"]))
        self.assertAlmostEqual(sum(answer["weights"]),1)
        data["margin_cap"]*=10;relaxed,_=a.q024(data)
        self.assertEqual(relaxed["weights"],answer["weights"])
        self.assertTrue(relaxed["target_reachable"])
        self.assertEqual(relaxed["attainable_volatility"],data["target_volatility"])
        data=inputs("q024");data["covariance"]=[[4*v for v in row] for row in data["covariance"]]
        scaled,_=a.q024(data)
        for left,right in zip(answer["weights"],scaled["weights"]):self.assertAlmostEqual(left,right)
        self.assertAlmostEqual(answer["max_leverage"],scaled["max_leverage"])

    def test_entropy_dual_certifies_both_active_constraints_and_infeasible_target(self):
        data=inputs("q025");answer,certificate=a.q025(data)
        self.assertGreater(answer["alpha_multiplier"],0);self.assertGreater(answer["cap_multiplier"],0)
        self.assertTrue(all(abs(v)<1e-12 for v in certificate["stationarity_differences"]))
        self.assertAlmostEqual(certificate["alpha_complementarity"],0);self.assertAlmostEqual(certificate["cap_complementarity"],0)
        data["alpha_target"]=.02;loose=a.q025(data)[0]
        self.assertEqual(loose["alpha_multiplier"],0)
        for left,right in zip(loose["weights"],data["benchmark"]):self.assertAlmostEqual(left,right)
        data["alpha_target"]=.06
        with self.assertRaisesRegex(ValueError,"feasible target"):a.q025(data)


if __name__=="__main__":unittest.main()
