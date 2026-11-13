"""Independent model identities and counterfactual checks, without answer keys."""
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import NormalDist
import unittest

from quality import numerical as n

ROOT=Path(__file__).resolve().parents[1]


def inputs(cid):
    return json.loads((ROOT/f"questions/legacy/quant100/questions/{cid}.json").read_text())["inputs"]


class ModelCertificates(unittest.TestCase):
    def test_psd_bounds_are_attainable_and_outside_is_infeasible(self):
        data=inputs("q045");data.update(corr_ab=.2,corr_ac=-.7)
        answer,certificate=n.q045(data)
        low,high=answer["correlation_bounds"];a,b=data["corr_ab"],data["corr_ac"]
        determinant=lambda c:1+2*a*b*c-a*a-b*b-c*c
        for endpoint in (low,high):
            self.assertAlmostEqual(determinant(endpoint),0)
            self.assertGreaterEqual(1-endpoint**2,0)
        self.assertLess(determinant(low-1e-4),0)
        self.assertLess(determinant(high+1e-4),0)
        data["volatility_limit"]=answer["volatility_bounds"][1]
        self.assertEqual(n.q045(data)[0]["limit_decision"],"certainly_met")

    def test_frozen_smile_reduces_to_standard_delta_and_hedge_is_second_order(self):
        data=inputs("q055");data["eta"]=0
        answer,_=n.q055(data)
        d1=(math.log(data["S0"]/data["K"])+(data["r"]-data["q"]+data["sigma0"]**2/2)*data["T"])/(data["sigma0"]*math.sqrt(data["T"]))
        self.assertAlmostEqual(answer["frozen_delta"],math.exp(-data["q"]*data["T"])*NormalDist().cdf(d1),places=9)
        self.assertAlmostEqual(answer["smile_delta"],answer["frozen_delta"],places=10)
        data=inputs("q055");data["new_spot"]=data["S0"]+.02
        first=n.q055(data)[0]["hedged_pnl"]
        data["new_spot"]=data["S0"]+.01
        second=n.q055(data)[0]["hedged_pnl"]
        self.assertAlmostEqual(first/second,4,delta=.002)

    def test_payoff_quadrature_agrees_with_closed_form_and_tail_probability(self):
        data=inputs("q065");answer,_=n.q065(data)
        normal=NormalDist();sd=data["normal_volatility"]*math.sqrt(data["T"])
        z=(data["forward"]-data["strike"])/sd;scale=data["notional"]*data["accrual"]*data["discount_factor"]
        closed=scale*((data["forward"]-data["strike"])*normal.cdf(z)+sd*normal.pdf(z))
        self.assertAlmostEqual(answer["normal_price"],closed,delta=1e-5)
        self.assertGreater(answer["normal_forward_pv01"],0)
        self.assertLess(answer["normal_forward_pv01"],scale*1e-4)
        self.assertFalse(answer["unshifted_black_admissible"])

    def test_constant_annuity_removes_rate_weighting_adjustment(self):
        data=inputs("q066");data["time1_annuities"]=[3.2,3.2]
        answer,certificate=n.q066(data)
        self.assertEqual(answer["fair_strike"],answer["arithmetic_rate"])
        self.assertEqual(answer["receiver_pv_at_arithmetic_rate"],0)
        self.assertEqual(certificate["fair_strike_pv"],0)

    def test_futures_discounted_wealth_is_zero_and_constant_rates_remove_basis(self):
        data=inputs("q067");answer,certificate=n.q067(data)
        self.assertGreater(answer["expected_terminal_wealth"],0)
        self.assertEqual(certificate["discounted_futures_wealth"],0)
        data["state_cash_growth_1_to_2"]=[1.2,1.2]
        answer,_=n.q067(data)
        self.assertEqual(answer["basis"],0)
        self.assertEqual(answer["expected_terminal_wealth"],0)

    def test_quanto_correlation_effect_disappears_without_fx_risk(self):
        data=inputs("q069");data["fx_vol"]=0
        answer,_=n.q069(data)
        self.assertEqual(answer["quanto_price"],answer["zero_correlation_price"])
        self.assertEqual(answer["correlation_derivative"],0)
        data["fixed_usd_per_eur"]*=2
        self.assertAlmostEqual(n.q069(data)[0]["quanto_price"],2*answer["quanto_price"])

    def test_currency_cashflow_netting_and_basis_sign(self):
        data=inputs("q070");answer,certificate=n.q070(data)
        self.assertLess(answer["usd_basis"],0)
        self.assertEqual(certificate["spread_adjusted_pv"],0)
        data.update(eur_coupon=0,usd_coupon=0,usd_per_eur_forwards=[data["spot"]]*2)
        answer,_=n.q070(data)
        self.assertEqual(answer["pv"],0)
        self.assertEqual(answer["usd_basis"],0)
        self.assertAlmostEqual(answer["eur_discount_1y"],data["usd_discount_factors"][-1])

    def test_credit_solvent_price_has_zero_default_and_spread(self):
        data=inputs("q072");data["bond_price"]=data["solvent_cashflow"]*math.exp(-data["continuous_rate"]*data["T"])
        answer,certificate=n.q072(data)
        for key in answer:self.assertAlmostEqual(answer[key],0,places=8)
        self.assertAlmostEqual(certificate["bond_repricing_error"],0)

    def test_gaussian_signal_sign_symmetry_and_information_limit(self):
        data=inputs("q081");positive=n.q081(data)[0]
        data["raw_forecast_pct"]*=-1;negative=n.q081(data)[0]
        self.assertAlmostEqual(positive["prob_positive"]+negative["prob_positive"],1)
        self.assertEqual(positive["posterior_sd_pct"],negative["posterior_sd_pct"])
        data["forecast_noise_sd_pct"]=1e-4
        self.assertAlmostEqual(n.q081(data)[0]["posterior_mean_pct"],data["raw_forecast_pct"],places=6)

    def test_bayesian_quotes_collapse_without_information_and_separate_with_perfect_information(self):
        data=inputs("q086");data["prob_informed"]=0
        answer,_=n.q086(data)
        self.assertEqual(answer["ask_initial"],answer["bid_initial"])
        self.assertEqual(answer["ask_after_two_buys"],answer["ask_initial"])
        data["prob_informed"]=1;answer,_=n.q086(data)
        self.assertEqual(answer["ask_initial"],data["value_high"])
        self.assertEqual(answer["bid_initial"],data["value_low"])
        self.assertEqual(answer["ask_after_two_buys"],data["value_high"])

    def test_joint_gaussian_terminal_marginal_matches_recursive_filter_and_translation(self):
        data=inputs("q089");mean,var=data["prior_mean"],data["prior_var"]
        for observation in data["observations"]:
            var+=data["state_noise_var"]
            gain=var/(var+data["obs_noise_var"])
            mean+=gain*(observation-mean);var*=1-gain
        answer,certificate=n.q089(data)
        self.assertAlmostEqual(answer["filtered_mean"],mean,places=12)
        self.assertAlmostEqual(answer["filtered_variance"],var,places=12)
        self.assertLess(certificate["joint_precision_normal_equation_residual"],1e-10)
        data["observations"]=[y+.03 for y in data["observations"]];data["prior_mean"]+=.03
        shifted=n.q089(data)[0]
        self.assertAlmostEqual(shifted["filtered_mean"],answer["filtered_mean"]+.03,places=12)
        self.assertEqual(shifted["filtered_variance"],answer["filtered_variance"])

    def test_forecast_intercept_absorbs_bias_without_refitting_on_evaluation_covariance(self):
        data=inputs("q094");answer,_=n.q094(data)
        data["forecast_error_bias_bps"]=[4,-3]
        changed,_=n.q094(data)
        self.assertEqual(changed["weights"],answer["weights"])
        self.assertEqual(changed["fitting_mse"],answer["fitting_mse"])
        self.assertNotEqual(changed["intercept"],answer["intercept"])
        data["evaluation_error_covariance_bps_squared"]=[[1,0],[0,100]]
        altered,_=n.q094(data)
        self.assertEqual(altered["weights"],answer["weights"])
        self.assertEqual(altered["fitting_mse"],answer["fitting_mse"])
        self.assertNotEqual(altered["evaluation_mse"],answer["evaluation_mse"])

    def test_measurement_noise_absence_aligns_structural_and_prediction_slopes(self):
        data=inputs("q095");answer,certificate=n.q095(data)
        self.assertAlmostEqual(answer["mse_penalty_using_structural_slope"],certificate["penalty_square_identity"])
        data["known_measurement_noise_variance"]=0
        answer,_=n.q095(data)
        self.assertEqual(answer["structural_beta"],answer["predictive_slope"])
        self.assertEqual(answer["mse_penalty_using_structural_slope"],0)

    def test_spread_funding_is_charged_to_long_leg_and_trend_shift_cancels(self):
        data=inputs("q096");answer,certificate=n.q096(data)
        data["annual_long_funding_rate"]+=.01
        altered,_=n.q096(data)
        expected_cost=.01*data["short_X_shares"]*answer["cointegrating_hedge"]*data["current_Y"]/data["days_per_year"]
        self.assertAlmostEqual(answer["net_expected_pnl"]-altered["net_expected_pnl"],expected_cost)
        data["current_W"]+=8;data["current_X"]+=8;data["current_Y"]+=4
        shifted,_=n.q096(data)
        self.assertEqual(shifted["next_expected_spread"],answer["next_expected_spread"])
        self.assertEqual(certificate["cointegrating_trend_loading"],0)

    def test_fill_selection_bounds_preserve_both_marginals_and_fee_units(self):
        data=inputs("q099");answer,certificate=n.q099(data)
        for table in certificate["attaining_joint_tables"]:
            self.assertAlmostEqual(sum(table),1)
            self.assertAlmostEqual(table[0]+table[1],data["fill_probability"])
            self.assertAlmostEqual(table[0]+table[2],data["favorable_markout_probability"])
        data["fee_per_filled_notional_bp"]+=1
        changed,_=n.q099(data)
        for before,after in zip(answer["filled_alpha_bounds"],changed["filled_alpha_bounds"]):self.assertAlmostEqual(before-after,1)
        for before,after in zip(answer["submitted_alpha_bounds"],changed["submitted_alpha_bounds"]):self.assertAlmostEqual(before-after,data["fill_probability"])


if __name__=="__main__":unittest.main()
