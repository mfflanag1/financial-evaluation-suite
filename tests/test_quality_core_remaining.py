"""Public-input counterfactuals for the remaining fixed decision contracts."""
from copy import deepcopy
from fractions import Fraction as F
import itertools
import unittest

from quality import core_remaining as audit
from quality.public_contracts import load_core_packet


def question(cid, turn=1):
    return load_core_packet(cid)['turns'][turn-1]


class RemainingCoreArithmetic(unittest.TestCase):
    def test_all_authored_turn_outputs_without_transition_certification(self):
        turns=fields=0
        for cid,(calculate,_) in audit.CHECKS.items():
            for q in load_core_packet(cid)['turns']:
                answers,certificate=calculate(q)
                self.assertEqual(set(answers),set(q['outputs']))
                self.assertIn('snapshot',certificate['scope'])
                turns+=1;fields+=len(answers)
        self.assertEqual((len(audit.CHECKS),turns,fields),(16,52,330))

    def test_transcribed_rules_fail_closed_when_source_changes(self):
        for cid,(calculate,_) in audit.CHECKS.items():
            q=question(cid);q['sources'][0]['text']+=' Changed governing rule.'
            with self.assertRaisesRegex(ValueError,'differs from reviewed'):
                calculate(q)

    def test_regret_is_not_minimum_worst_expected_loss(self):
        answer,certificate=audit.c203(question('c203'))
        losses={action:list(map(F,values)) for action,values in certificate['endpoint_expected_losses'].items()}
        self.assertEqual(min(losses,key=lambda action:max(losses[action])),'pass')
        self.assertEqual(answer['action'],'insured_purchase')
        self.assertEqual(F(certificate['worst_regrets']['insured_purchase']),F(24,5))
        # Check every rational mixture on a 101-point grid against the endpoint bound.
        for mix in (F(k,100) for k in range(101)):
            expected={a:mix*v[0]+(1-mix)*v[1] for a,v in losses.items()}
            regret=expected[answer['action']]-min(expected.values())
            self.assertLessEqual(regret,F(24,5))

    def test_short_facility_is_limited_by_earlier_repayment(self):
        short,term,fee,final,total,_=audit.treasury_facilities(F(40),F(60),F(15),F(88))
        self.assertEqual((short,term,total),(15,5,20))
        self.assertEqual(fee,F(7,20))
        self.assertEqual(final,F(40-60+15+88)-fee)
        improved=audit.treasury_facilities(F(40),F(60),F(20),F(88))
        self.assertEqual(improved[:3],(20,0,F(1,5)))

    def test_facility_caps_can_make_early_deadline_infeasible(self):
        with self.assertRaisesRegex(ValueError,'No feasible'):
            audit.treasury_facilities(F(0),F(50),F(100),F(100))

    def test_signed_tail_mean_splits_atoms_and_preserves_gains(self):
        self.assertEqual(audit.tail_mean([F(-58),F(42)],[F(85,100),F(15,100)],F(1,5)),17)
        self.assertEqual(audit.tail_mean([F(-10),F(-30)],[F(1,10),F(9,10)],F(1,5)),-20)

    def test_dependence_envelope_covers_all_grid_laws(self):
        # Alternative state-loss vectors exercise every tail ordering and ties.
        for losses in itertools.permutations(map(F,(-54,-114,46,16))):
            worst,points=audit.dependence_es(losses)
            self.assertIn(F(0),{p for v,p in points});self.assertIn(F(1,4),{p for v,p in points})
            for p in (F(k,400) for k in range(101)):
                actual=audit.tail_mean(losses,[p,F(1,4)-p,F(1,2)-p,F(1,4)+p],F(1,5))
                self.assertLessEqual(actual,worst)

    def test_robust_bond_portfolio_profit_and_budget_identity(self):
        answer,_=audit.c215(question('c215'));a,b,h=answer['action']
        self.assertEqual(answer['purchase_cost'],80*a+50*b+3*h)
        # Each bond's marginal expected payoff is fixed; protection contributes 30*h*p.
        baseline=85*a+60*b-answer['purchase_cost']
        self.assertEqual(answer['expected_profit_interval'],[baseline,baseline+30*h/4])
        self.assertLessEqual(answer['worst_expected_shortfall'],55)

    def test_liquidation_feedback_and_cash_equalities(self):
        best=audit.cascade_plan((4,0,0,6))
        self.assertTrue(best['feasible']);self.assertEqual(best['early'],0);self.assertEqual(best['final'],9)
        self.assertFalse(audit.cascade_plan((4,0,0,5))['feasible'])
        self.assertEqual(audit.cascade_plan((2,0,0,0))['haircut'],0)
        self.assertEqual(audit.cascade_plan((3,0,0,0))['haircut'],8)
        self.assertLess(audit.cascade_plan((4,1,0,0))['cap'],audit.cascade_plan((4,0,0,0))['cap'])

    def test_repair_does_not_spend_unsettled_b_proceeds(self):
        q=question('c207');before,_=audit.c207(q)
        q['inputs']['terms']['sell_prices'][1]=100000
        after,_=audit.c207(q)
        self.assertEqual(before,after)
        q['inputs']['current_book']['cash']=-1000
        infeasible,_=audit.c207(q)
        self.assertEqual(infeasible['decision'],'NO_FEASIBLE_PLAN')
        self.assertEqual(infeasible['post_cash'],-1000)

    def test_research_timestamp_equality_is_inadmissible_and_trial_family_matters(self):
        q=question('c208');data=q['inputs']['observations'][1]['data']
        data['groups'][1]['availability']='09:00'
        result,_=audit.c208(q)
        self.assertEqual((result['admissible_count'],result['p_value']),(40,0.009))
        later,_=audit.c208(question('c208',3))
        self.assertEqual(later['status'],'reject');self.assertEqual(later['significance_cutoff'],0.05/8)
        last,_=audit.c208(question('c208',4))
        self.assertEqual(last['net_mean'],-0.3);self.assertFalse(last['deploy'])

    def test_physical_belief_does_not_reprice_tradable_replication(self):
        q=question('c209',2);before,_=audit.c209(q)
        q['inputs']['observations'][1]['data']['physical_probabilities']={'E':0.9,'not_E':0.1}
        after,_=audit.c209(q)
        self.assertEqual(before['fair_value'],after['fair_value'])
        self.assertNotEqual(before['physical_expected_payoff'],after['physical_expected_payoff'])
        recovered,_=audit.c209(question('c209',4))
        self.assertAlmostEqual(recovered['fair_value'],40+60*0.42)
        self.assertLess(recovered['var_80'],0);self.assertGreater(recovered['es_80'],0)

    def test_legal_entity_funding_rejects_late_transfer_and_segregated_cash(self):
        late,_=audit.c210(question('c210',3))
        self.assertEqual(late['projects'],[]);self.assertEqual(late['repo_capacity'],20)
        final,_=audit.c210(question('c210',4))
        self.assertEqual(final['projects'],['PY']);self.assertEqual(final['transfer_X_to_Y'],30-(10-8))
        self.assertEqual(final['cash_X'],20+15-28)

    def test_duplicate_information_and_sunk_fees(self):
        result,_=audit.c211(question('c211',2))
        self.assertEqual(result['D_incremental_value'],-1)
        self.assertEqual(result['T_incremental_value'],-2)
        self.assertAlmostEqual(result['posterior_G'],12/19)
        q=question('c211',2);q['inputs']['observations'][2]['data']['observed_purchases'][0]['fee_paid']=999
        self.assertEqual(audit.c211(q)[0],result)
        changed,_=audit.c211(question('c211',3))
        self.assertEqual(changed['query'],'T');self.assertEqual(changed['incremental_value'],13)
        final,_=audit.c211(question('c211',4))
        self.assertEqual(final['action'],'sell');self.assertAlmostEqual(final['posterior_G'],4/11)
        self.assertFalse(final['state_identified'])

    def test_board_lot_cash_boundary_and_live_cancellation_commitment(self):
        opening,_=audit.c212(question('c212'))
        self.assertEqual(opening['action'],'CANCEL');self.assertEqual(opening['cash_after_commitments'],7000-60*110)
        q=question('c212',2);q['inputs']['current_book']['cash']=1500
        exact,_=audit.c212(q);self.assertEqual(exact['quantity'],5)
        q['inputs']['current_book']['cash']=1499
        short,_=audit.c212(q);self.assertEqual((short['action'],short['quantity']),('WAIT',0))

    def test_robust_liquidation_keeps_unpaid_obligations(self):
        q=question('c219');q['inputs']['current_state'].update(cash=0,positions=[0,0],hedges=0,arrears=5)
        result,_=audit.c219(q)
        self.assertEqual(result['action'],[0,0,0])
        self.assertEqual(result['diagnostics'],[0,0,23])
        self.assertEqual(result['constraint_violation'],27)

    def test_strategy_admission_familywise_boundary_and_leakage(self):
        q=question('c220');terms=q['inputs']['current_terms']
        terms['p'][0]=0.00625
        exact,_=audit.c220(q);self.assertEqual(exact['diagnostics'][2][0],1)
        terms['p'][0]=0.0062501
        outside,_=audit.c220(q);self.assertEqual(outside['diagnostics'][2][0],0);self.assertEqual(outside['action'][0],0)
        terms['p'][0]=0;terms['leakage'][0]=True
        leaked,_=audit.c220(q);self.assertEqual(leaked['action'][0],0)

    def test_robust_hedge_reports_an_unattainable_cap(self):
        answer,_=audit.c221(question('c221',2))
        self.assertEqual(answer['diagnostics'][:2],[-52,52])
        self.assertEqual(answer['constraint_violation'],14)
        self.assertEqual(answer['objective'],52+1000*14)

    def test_collateral_no_withdrawal_below_floor_and_late_cash(self):
        q=question('c222');state=q['inputs']['current_state'];terms=q['inputs']['current_terms']
        state.update(a=7,b=9,central=0,debt=8,arrears=3)
        result,_=audit.c222(q)
        self.assertEqual(result['action'],[0,0,0]);self.assertEqual(result['diagnostics'][1],13)
        self.assertGreater(result['constraint_violation'],13)
        # The optimal opening B sale helps the forecast but does not pay today's call.
        opening,_=audit.c222(question('c222'))
        a,b,borrow=opening['action']
        self.assertAlmostEqual(opening['diagnostics'][0],3+1.6*a+0.75*borrow-10)
        self.assertAlmostEqual(opening['diagnostics'][2],1.8*b)

    def test_information_tree_last_round_and_unobserved_branches(self):
        q=question('c223');terms=q['inputs']['current_terms']
        actions,mean=audit.information_tree(terms['worlds'],terms['costs'],F(9),1,{})
        self.assertEqual({row[0] for row in actions},{'INVEST','ABSTAIN'})
        self.assertEqual(mean,12)
        # Opposite first-signal branch is derived from the public joint law too.
        actions,mean=audit.information_tree(terms['worlds'],terms['costs'],F(6),3,{'A':1})
        self.assertEqual(mean,-12)
        self.assertEqual(max(row[1] for row in actions),8)
        committed,_=audit.information_tree(terms['worlds'],terms['costs'],F(2),2,{'A':0,'B':1},True)
        self.assertEqual(committed,[('WAIT',0,0)])

    def test_pending_sale_cash_cannot_fund_buy_and_terminal_cancel_stays_live(self):
        q=question('c224',4);state=q['inputs']['current_state'];terms=q['inputs']['current_terms']
        state.update(cash=150,pending=100000)
        choices=audit.execution_choices(state,terms)
        self.assertFalse(any(row[1][0]=='BUY' for row in choices))
        state.update(cash=1000,inventory=12,live=8,price=40,cancel=False,target=15)
        choices=audit.execution_choices(state,terms)
        cancel=next(row for row in choices if row[1][0]=='CANCEL')
        self.assertEqual(cancel[4],[680,3,100000])
        self.assertEqual(cancel[0],F(154,5))
        state['cancel']=True
        self.assertNotIn('CANCEL',{row[1][0] for row in audit.execution_choices(state,terms)})


if __name__=='__main__':unittest.main()
