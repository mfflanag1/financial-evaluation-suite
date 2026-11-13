"""Independent risk invariants for proposed public-case arithmetic coverage."""
import json
import math
from pathlib import Path
import unittest

from quality import risk_numerical as r
from quality.numerical import fraction, solve_linear

ROOT=Path(__file__).resolve().parents[1]

def inputs(cid):return json.loads((ROOT/f'questions/legacy/quant100/questions/{cid}.json').read_text())['inputs']

class RiskCertificates(unittest.TestCase):
    def test_full_trade_tail_reordering_and_translation(self):
        data=inputs('q028');base,_=r.q028(data)
        self.assertNotEqual(base['incremental_es'],base['standalone_trade_es'])
        data['trade_losses']=[v+1 for v in data['trade_losses']]
        shifted,_=r.q028(data)
        for key in base:self.assertAlmostEqual(shifted[key]-base[key],1)

    def test_reverse_stress_global_bound_and_null_direction(self):
        data=inputs('q030');answer,certificate=r.q030(data)
        self.assertAlmostEqual(answer['severity'],certificate['global_cauchy_schwarz_lower_bound'])
        self.assertTrue(all(abs(v)<1e-12 for v in certificate['stationarity_residuals']))
        direction=[data['loss_loadings'][1],-data['loss_loadings'][0]]
        for sign in (-1,1):
            shock=[x+sign*.1*d for x,d in zip(answer['shock'],direction)]
            precision=solve_linear(data['shock_covariance'],shock)
            self.assertGreater(math.sqrt(sum(x*y for x,y in zip(shock,precision))),answer['severity'])
        data['loss_barrier']*=2
        self.assertAlmostEqual(r.q030(data)[0]['severity'],2*answer['severity'])

    def test_nonlinear_grid_boundary_is_inclusive_and_discrete(self):
        data=inputs('q031');answer,_=r.q031(data)
        data['loss_barrier']=answer['loss'];self.assertEqual(r.q031(data)[0]['shock'],answer['shock'])
        data['loss_barrier']=answer['loss']+.01
        changed,_=r.q031(data)
        self.assertGreater(changed['severity'],answer['severity'])
        self.assertIn(changed['shock'][0],data['x_grid']);self.assertIn(changed['shock'][1],data['y_grid'])

    def test_late_receipt_cannot_cover_early_margin_failure(self):
        data=inputs('q032');answer,_=r.q032(data)
        data['event_cashflows'][-1]['flow']+=100
        self.assertEqual(r.q032(data)[0]['extra_buffer'],answer['extra_buffer'])
        data['event_cashflows'][-1]['time']='11:00'
        self.assertEqual(r.q032(data)[0]['extra_buffer'],0)

    def test_collateral_eligibility_limit_precedes_cost_optimization(self):
        data=inputs('q033');answer,_=r.q033(data)
        corporate=answer['bond_blocks'][1]*(1-data['haircuts'][1])
        self.assertLessEqual(corporate,data['corporate_recognized_cap'])
        self.assertAlmostEqual(answer['recognized_value'],data['required_recognized'])
        data['corporate_recognized_cap']=5
        relaxed,_=r.q033(data)
        self.assertGreater(relaxed['bond_blocks'][1],answer['bond_blocks'][1])
        self.assertEqual(relaxed['cash'],0);self.assertLess(relaxed['annual_cost'],answer['annual_cost'])

    def test_excess_collateral_in_one_legal_set_does_not_offset_another(self):
        data=inputs('q034');answer,_=r.q034(data)
        data['netting_sets'][1]['collateral']+=100
        self.assertEqual(r.q034(data)[0],answer)
        data=inputs('q034')
        data['netting_sets']=[{'id':'A','counterparty':'A','marks':[12,-8,7,-3],'collateral':8},data['netting_sets'][2]]
        self.assertLess(r.q034(data)[0]['total_ead'],answer['total_ead'])

    def test_conditioning_on_default_changes_state_mix_and_equality_meets_buffer(self):
        data=inputs('q035');answer,_=r.q035(data)
        self.assertGreater(answer['state_probabilities_given_default'][-1],data['state_probabilities'][-1])
        joint=[fraction(p)*fraction(d) for p,d in zip(data['state_probabilities'],data['default_probability_given_state'])]
        probabilities=[p/sum(joint) for p in joint]
        losses=[fraction(x)*fraction(data['unsecured_loss_given_default']) for x in data['positive_receivables']]
        data['workout_buffer']=str(r.es(losses,probabilities,data['conditional_tail_alpha']))
        self.assertFalse(r.q035(data)[0]['buffer_inadequate'])
        data['default_probability_given_state']=[.02]*3
        independent,_=r.q035(data)
        self.assertEqual(independent['state_probabilities_given_default'],data['state_probabilities'])

    def test_closeout_uses_realization_not_second_recognition_haircut(self):
        data=inputs('q036');answer,_=r.q036(data)
        data['initial_recognition_haircut']=.8
        self.assertEqual(r.q036(data)[0],answer)
        data['liquidation_cost_rate']+=.01
        increased,_=r.q036(data)
        self.assertGreater(increased['conditional_expected_loss'],answer['conditional_expected_loss'])

    def test_desk_euler_contributions_keep_negative_sign_and_scale(self):
        data=inputs('q037');answer,certificate=r.q037(data)
        self.assertLess(answer['desk_contributions'][2],0)
        self.assertGreater(answer['remove_c_vol_change'],0)
        self.assertLess(abs(certificate['euler_contribution_sum_residual']),1e-7)
        data['desk_factor_loadings']=[[2*v for v in row] for row in data['desk_factor_loadings']]
        scaled,_=r.q037(data)
        self.assertAlmostEqual(scaled['volatility'],2*answer['volatility'])
        self.assertAlmostEqual(scaled['factor1_variance_share'],answer['factor1_variance_share'])
        for left,right in zip(answer['desk_contributions'],scaled['desk_contributions']):self.assertAlmostEqual(right,2*left,places=7)

    def test_dollar_neutral_currency_book_retains_basis_risk(self):
        data=inputs('q038');answer,_=r.q038(data)
        self.assertEqual(answer['net_fx_notional'],0);self.assertGreater(answer['residual_volatility'],0)
        data['fx_return_volatilities']=[.1,.1];data['fx_correlation']=1
        self.assertAlmostEqual(r.q038(data)[0]['residual_volatility'],0)
        data['stress_fx_returns']=[.05,.05]
        self.assertEqual(r.q038(data)[0]['stress_loss'],0)

    def test_regime_mixture_preserves_exact_var_atom_and_conditional_tail(self):
        data=inputs('q039');answer,certificate=r.q039(data)
        self.assertEqual(certificate['strict_exceedance_probability'],.05)
        self.assertEqual(answer['crisis_given_var_exceedance'],1)
        self.assertGreater(answer['unconditional_es95'],answer['calm_es95'])
        data['regime_probabilities']=[.96,.04]
        changed,_=r.q039(data)
        self.assertEqual(changed['calm_es95'],answer['calm_es95'])
        self.assertLess(changed['unconditional_es95'],answer['unconditional_es95'])

    def test_drawdown_depends_on_order_while_terminal_loss_does_not(self):
        data=inputs('q041');answer,_=r.q041(data)
        data['paths'][0]=[90,110,105]
        changed,_=r.q041(data)
        self.assertEqual(changed['maximum_drawdowns'][0],10)
        self.assertEqual(changed['terminal_loss_es75'],answer['terminal_loss_es75'])
        self.assertLess(changed['trigger_probability'],answer['trigger_probability'])

    def test_guarantee_failure_and_strict_breach_boundary(self):
        data=inputs('q042');answer,certificate=r.q042(data)
        self.assertEqual(certificate['guarantee_reimbursements'][-1],0)
        data['capital_limit']=answer['net_losses'][3]
        self.assertEqual(r.q042(data)[0]['first_breach_level'],4)
        data['capital_limit']-=1
        self.assertEqual(r.q042(data)[0]['first_breach_level'],3)

    def test_uniform_auction_impacts_unsold_assets_and_minimum_grid_cure(self):
        data=inputs('q044');answer,certificate=r.q044(data)
        self.assertTrue(certificate['all_smaller_menu_sales_infeasible'])
        self.assertEqual(certificate['equity_uniform_mark_identity_error'],0)
        self.assertLess(answer['final_equity'],data['shares']*data['price_intercept_usd']-data['debt_usd'])
        data['price_impact_usd_per_share']=0
        no_impact,_=r.q044(data)
        self.assertEqual(no_impact['shares_sold'],170)
        self.assertEqual(no_impact['final_equity'],20000)

    def test_principal_settlement_timing_changes_exposure_and_funding(self):
        data=inputs('q046');answer,_=r.q046(data)
        data['schedule'][2]['time']='09:30'
        early,_=r.q046(data)
        self.assertLess(early['peak_principal_exposure'],answer['peak_principal_exposure'])
        self.assertEqual(early['extra_line'],0)
        self.assertEqual(early['a_default_loss'],0)

    def test_annual_insurance_limit_is_after_occurrence_terms(self):
        data=inputs('q048');answer,_=r.q048(data)
        data['annual_reimbursement_cap']=100
        relaxed,_=r.q048(data)
        self.assertEqual(relaxed['retained_losses'][1],answer['retained_losses'][1])
        self.assertEqual(relaxed['retained_losses'][-1],12)
        self.assertLess(relaxed['expected_retained_loss'],answer['expected_retained_loss'])

    def test_entity_transfer_blockage_increases_funding_with_same_pooled_deficit(self):
        data=inputs('q049');answer,certificate=r.q049(data)
        self.assertGreater(answer['external_funding'],answer['naive_pooled_deficit'])
        self.assertTrue(all(value>=0 for snapshot in certificate['selected_cash_timeline'] for value in snapshot.values()))
        data['transfer_caps']['A_to_P']=0
        blocked,_=r.q049(data)
        self.assertGreater(blocked['external_funding'],answer['external_funding'])
        self.assertEqual(blocked['naive_pooled_deficit'],answer['naive_pooled_deficit'])

    def test_entity_transfer_certificate_rejects_an_unreviewed_arrival_schedule(self):
        data=inputs('q049');data['transfer_arrivals']['A_to_P']='11:30'
        with self.assertRaisesRegex(ValueError,'reviewed transfer'):r.q049(data)

if __name__=='__main__':unittest.main()
