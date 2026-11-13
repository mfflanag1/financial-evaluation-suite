"""Financial identities and counterfactuals for the remaining quantitative cases."""
import json
import math
from pathlib import Path
import unittest

from quality import remaining_numerical as r
from quality.numerical import fraction
ROOT=Path(__file__).resolve().parents[1]
def inputs(cid):return json.loads((ROOT/f'questions/legacy/quant100/questions/{cid}.json').read_text())['inputs']

class RemainingCertificates(unittest.TestCase):
    def test_desk_capital_tightening_is_a_nested_feasible_set(self):
        data=inputs('q013');answer,certificate=r.q013(data)
        self.assertLessEqual(certificate['base_capital'],data['capital_base_usd_m'])
        self.assertLessEqual(certificate['tight_capital'],data['capital_tight_usd_m'])
        self.assertGreaterEqual(answer['objective_loss'],0)
        data['capital_tight_usd_m']=data['capital_base_usd_m']
        changed,_=r.q013(data);self.assertEqual(changed['tight_units'],changed['base_units']);self.assertEqual(changed['objective_loss'],0)

    def test_lot_tax_order_uses_one_ticket_and_prefers_high_basis(self):
        data=inputs('q017');answer,certificate=r.q017(data)
        self.assertEqual(answer['sales'][0],0)
        self.assertEqual(certificate['execution'],data['fixed_sale_ticket_usd']+sum(answer['sales'])*data['per_share_execution_cost_usd'])
        self.assertGreaterEqual(answer['ending_cash'],data['required_cash_usd'])
        data['basis_per_share_usd'].reverse()
        self.assertEqual(r.q017(data)[0]['sales'],list(reversed(answer['sales'])))

    def test_post_fee_targets_and_contingent_policy_probability_conservation(self):
        cash,risky,fee=r.rebalance(fraction(.8),fraction(.2),fraction(.75),fraction(.03))
        self.assertEqual(risky/(cash+risky),fraction(.75));self.assertEqual(cash+risky+fee,1)
        cash,risky,fee=r.rebalance(cash,risky,fraction(.1),fraction(.03))
        self.assertEqual(risky/(cash+risky),fraction(.1))
        data=inputs('q018');answer,certificate=r.q018(data)
        self.assertAlmostEqual(sum(certificate['terminal_probabilities']),1)
        self.assertGreaterEqual(min(certificate['terminal_wealth']),data['terminal_wealth_floor'])
        self.assertEqual(answer['worst_terminal_wealth'],min(certificate['terminal_wealth']))
        data['one_way_trading_cost']=0
        self.assertGreaterEqual(r.q018(data)[0]['expected_log_wealth'],answer['expected_log_wealth'])

    def test_sector_menu_preserves_sector_neutrality_and_stress_feasibility(self):
        data=inputs('q019');answer,certificate=r.q019(data)
        for pair,risk in zip((answer['base_pair_weights'],answer['stressed_pair_weights']),certificate['selected_tracking_variances']):
            active=[pair[0],-pair[0],pair[1],-pair[1]]
            self.assertEqual(active[0]+active[1],0);self.assertEqual(active[2]+active[3],0)
            self.assertTrue(all(b+w>=0 for b,w in zip(data['benchmark'],active)))
            self.assertLessEqual(risk,data['tracking_error_cap']**2+1e-12)
        data['stressed_pair_covariance']=data['base_pair_covariance']
        same=r.q019(data)[0];self.assertEqual(same['base_pair_weights'],same['stressed_pair_weights'])

    def test_bond_ladder_cannot_spend_later_receipts_early(self):
        data=inputs('q020');answer,certificate=r.q020(data)
        self.assertTrue(all(value>=0 for value in certificate['end_of_date_cash_balances']))
        self.assertAlmostEqual(answer['purchase_cost'],sum(u*p for u,p in zip(answer['bond_units'],data['bond_prices'])))
        data['liability_cashflows']=[3,2,1]
        earlier=r.q020(data)[0]
        self.assertGreater(earlier['bond_units'][0],answer['bond_units'][0])

    def test_unknown_overlap_changes_margin_but_not_objective(self):
        data=inputs('q023');answer,certificate=r.q023(data)
        self.assertFalse(answer['optimal_scale_identified'])
        self.assertEqual(answer['robust_scale'],min(answer['completion_scales']))
        self.assertGreater(answer['completion_scales'][0],answer['completion_scales'][1])
        data['capital_usd_m']*=10
        loose=r.q023(data)[0];self.assertTrue(loose['optimal_scale_identified'])
        self.assertEqual(loose['completion_scales'],[certificate['unconstrained_objective_optimum']]*2)

    def test_currency_replication_spot_scaling_and_equal_rate_forward(self):
        data=inputs('q052');answer,certificate=r.q052(data)
        self.assertAlmostEqual(answer['present_value'],certificate['spot_converted_discounted_eur_receivable'])
        data['usd_per_eur_spot']*=2
        scaled=r.q052(data)[0]
        for key in answer:self.assertAlmostEqual(scaled[key],2*answer[key])
        data['eur_rate']=data['usd_rate']
        self.assertAlmostEqual(r.q052(data)[0]['forward'],data['usd_per_eur_spot'])

    def test_integer_superhedge_covers_both_terminal_states_and_scales_liability(self):
        data=inputs('q056');answer,certificate=r.q056(data)
        self.assertEqual(certificate['integer_positions_searched'],101)
        self.assertTrue(all(value>=0 for value in certificate['terminal_superhedge_surplus']))
        self.assertLess(answer['cash_account'],0)
        data['calls_owed']=0
        zero=r.q056(data)[0];self.assertEqual(zero['shares'],0);self.assertEqual(zero['minimum_outlay'],0)

    def test_coupon_curve_repricing_and_last_bond_monotonicity(self):
        data=inputs('q061');answer,certificate=r.q061(data)
        self.assertTrue(all(abs(value)<1e-10 for value in certificate['cashflow_repricing_residuals']))
        data['bond_B_price']+=.1
        higher=r.q061(data)[0]
        self.assertAlmostEqual(higher['discounts'][0],answer['discounts'][0])
        self.assertGreater(higher['discounts'][1],answer['discounts'][1]);self.assertLess(higher['forward_rate'],answer['forward_rate'])

    def test_swap_projection_and_discount_sensitivities_are_separate(self):
        data=inputs('q062');answer,certificate=r.q062(data)
        self.assertGreater(answer['projection_pv01'],0);self.assertLess(answer['discount_pv01'],0)
        self.assertAlmostEqual(answer['discount_pv01'],certificate['discount_derivative_check'],places=7)
        self.assertAlmostEqual(certificate['par_coupon_repricing'],0,places=8)
        data['fixed_rate']=answer['par_rate']
        self.assertAlmostEqual(r.q062(data)[0]['pv'],0,places=8)

    def test_key_rate_hedge_cash_is_not_an_economic_charge(self):
        data=inputs('q063');answer,certificate=r.q063(data)
        self.assertLessEqual(answer['collateral'],data['cash_limit'])
        self.assertEqual(answer['worst_loss'],max(certificate['scenario_losses_including_zero']))
        self.assertFalse(certificate['collateral_deducted_from_pnl'])
        data['cash_limit']=0
        zero=r.q063(data)[0];self.assertEqual(zero['hedge_units'],[0,0,0]);self.assertGreaterEqual(zero['worst_loss'],answer['worst_loss'])

    def test_bond_convexity_improves_small_rate_move_approximation(self):
        data=inputs('q064');answer,certificate=r.q064(data)
        self.assertGreater(certificate['estimated_price_convexity'],0)
        self.assertLess(abs(answer['convexity_pnl']-answer['exact_pnl']),abs(answer['duration_pnl']-answer['exact_pnl']))
        data['new_yield']=data['initial_yield']
        self.assertEqual(r.q064(data)[0],{'exact_pnl':0,'duration_pnl':0,'convexity_pnl':0})

    def test_carry_finances_storage_and_respects_whole_barrel_budget(self):
        data=inputs('q068');answer,certificate=r.q068(data)
        self.assertLessEqual(certificate['initial_borrowing'],data['initial_borrow_limit'])
        cost=data['spot_per_barrel']+data['prepaid_storage_per_barrel']
        self.assertGreater((answer['quantity']+1)*cost,data['initial_borrow_limit'])
        data['forward_bid']=answer['carry_bound']-.1
        no_trade=r.q068(data)[0];self.assertEqual(no_trade['quantity'],0);self.assertEqual(no_trade['terminal_profit'],0)

    def test_wrong_way_cva_disappears_with_equal_conditional_default_rates(self):
        data=inputs('q073');answer,_=r.q073(data)
        self.assertGreater(answer['understatement'],0)
        data['conditional_default_probabilities']=[.02,.02]
        independent=r.q073(data)[0];self.assertEqual(independent['joint_cva'],independent['independent_cva']);self.assertEqual(independent['understatement'],0)

    def test_turnover_break_even_cost_zeroes_return_and_sharpe(self):
        data=inputs('q077');answer,certificate=r.q077(data)
        self.assertEqual(certificate['total_traded_notional_per_nav'],8)
        data['cost_per_dollar_traded_bps']=answer['breakeven_cost_bps']
        zero=r.q077(data)[0];self.assertEqual(zero['net_return_pct'],0);self.assertEqual(zero['net_sharpe'],0)

    def test_ar_moments_and_half_life_are_stationary_identities(self):
        data=inputs('q082');answer,certificate=r.q082(data)
        self.assertAlmostEqual(certificate['stationary_mean_residual'],0)
        self.assertAlmostEqual(certificate['stationary_variance_residual'],0)
        self.assertAlmostEqual(certificate['half_life_decay'],.5)
        data['current_spread']=answer['long_run_mean']
        centered=r.q082(data)[0];self.assertEqual(centered['z_score'],0);self.assertAlmostEqual(centered['expected_spread_5d'],answer['long_run_mean'])

    def test_ou_series_matches_separate_stated_first_passage_integrals(self):
        data=inputs('q090');answer,_=r.q090(data)
        upper=answer['best_threshold']*math.sqrt(data['theta_per_day'])/data['sigma_per_sqrt_day']
        count=4000;step=upper/count
        def integrate(kind):
            values=[math.exp((i*step)**2)*kind(i*step) for i in range(count+1)]
            return step/3*(values[0]+values[-1]+4*sum(values[1:-1:2])+2*sum(values[2:-1:2]))
        cycle=math.sqrt(math.pi)/data['theta_per_day']*(integrate(math.erf)+integrate(math.erfc))
        self.assertAlmostEqual(cycle,answer['cycle_days_at_best'],places=9)
        self.assertAlmostEqual(answer['best_profit_rate_per_day']*cycle,answer['best_threshold']-data['round_trip_cost_points'])

    def test_execution_risk_uses_only_post_slice_intervals(self):
        data=inputs('q091');answer,certificate=r.q091(data)
        self.assertEqual(sum(answer['schedule']),data['initial_inventory_lots'])
        self.assertEqual(certificate['price_intervals'],3)
        self.assertEqual(certificate['post_slice_inventory_bearing_risk'][0],data['initial_inventory_lots']-answer['schedule'][0])
        self.assertAlmostEqual(answer['cost_variance'],sum(certificate['interval_variances']))
        data['variance_penalty_per_usd']=0
        no_penalty=r.q091(data)[0];self.assertAlmostEqual(no_penalty['objective'],no_penalty['expected_cost'])
        self.assertLessEqual(no_penalty['expected_cost'],answer['expected_cost'])

    def test_quote_fill_states_include_mark_and_inventory_penalty(self):
        data=inputs('q092');answer,certificate=r.q092(data)
        self.assertAlmostEqual(sum(row['probability'] for row in certificate['selected_quote_fill_states']),1)
        average=sum(row['probability']*row['net_value'] for row in certificate['selected_quote_fill_states'])
        self.assertAlmostEqual(average,answer['inventory_objective'])
        self.assertLess(answer['expected_final_inventory'],data['initial_inventory_blocks'])
        data['initial_inventory_blocks']=0
        flat=r.q092(data)[0];self.assertEqual(flat['inventory_quote_offsets'],flat['flat_quote_offsets'])
        self.assertAlmostEqual(flat['expected_final_inventory'],0)

    def test_capacity_roots_bracket_operating_profit_and_objectives_differ(self):
        data=inputs('q093');answer,certificate=r.q093(data)
        self.assertTrue(all(abs(v)<1e-10 for v in certificate['break_even_profit_residuals']))
        self.assertAlmostEqual(certificate['dollar_profit_derivative'],0)
        self.assertAlmostEqual(certificate['return_objective_derivative'],0)
        self.assertLess(answer['return_maximizing_aum'],answer['profit_maximizing_aum'])
        low,high=answer['positive_break_even_aum'];self.assertLess(low,answer['profit_maximizing_aum']);self.assertGreater(high,answer['profit_maximizing_aum'])

if __name__=='__main__':unittest.main()
