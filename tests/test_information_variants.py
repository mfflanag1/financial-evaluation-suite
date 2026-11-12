"""Independent certificates and adversarial policy checks for diagnostic variants."""
import copy
import unittest
from fractions import Fraction as F
from information_variants import (build_cases, parameters, enumerate_reference,
    backward_reference, policy_economics, economic_outcome, BASE_WEIGHTS, scalar_routes)


class InformationVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=build_cases();cls.byid={c['id']:c for c in cls.cases}

    def test_exact_conditional_certificate(self):
        for c in self.cases:
            p=c['information_parameters']
            self.assertEqual(enumerate_reference(p)[0][:4],backward_reference(p))
            self.assertTrue(all(type(x) is int and x>0 for row in p['weights'] for x in row))
            self.assertEqual(sum(map(sum,p['weights'])),p['denominator'])
            self.assertEqual(len(c['documents']),4)
            self.assertEqual(len(c['stages'][0]['answers']),8)
            self.assertEqual(c['variant_of'],'c217')

    def test_positive_continuation_margin_is_larger(self):
        p=self.byid['c225']['information_parameters'];base=parameters()
        margin=backward_reference(p,1)[0]-backward_reference(p)[0]
        self.assertEqual(margin,F(67,50))
        self.assertGreater(margin,backward_reference(base,1)[0]-backward_reference(base)[0])
        self.assertEqual(backward_reference(p)[1:3],('A',('B','B')))

    def test_early_stop_is_cost_driven_not_budget_forced(self):
        p=self.byid['c226']['information_parameters'];best=backward_reference(p)
        self.assertEqual(best[1:3],('A',('stop','stop')))
        self.assertEqual(F(p['costs']['A'])+F(p['costs']['B']),F(p['budget']))
        feasible_both=policy_economics(p,'A',['B','B'],['pass','hedge','hedge','invest'])
        self.assertEqual(feasible_both['violations'],[])
        self.assertGreater(feasible_both['objective'],best[0])
        cheaper=copy.deepcopy(p);cheaper['costs']['B']='1/10'
        self.assertLess(backward_reference(cheaper)[0],backward_reference(cheaper,1)[0])

    def test_no_information_can_be_optimal_even_when_queries_affordable(self):
        p=self.byid['c227']['information_parameters']
        self.assertEqual(backward_reference(p)[1],'none')
        self.assertEqual(backward_reference(p)[0],F(-1))
        feasible=policy_economics(p,'A',['B','B'],['pass','hedge','hedge','invest'])
        self.assertEqual(feasible['violations'],[])
        self.assertGreater(feasible['objective'],-1)

    def test_first_query_reversal(self):
        self.assertEqual(backward_reference(parameters())[1],'A')
        p=self.byid['c228']['information_parameters']
        self.assertEqual(backward_reference(p)[1:3],('B',('stop','A')))

    def test_path_budget_is_economically_binding(self):
        p=self.byid['c229']['information_parameters'];tight=backward_reference(p)
        relaxed=copy.deepcopy(p);relaxed['budget']='4';loose=backward_reference(relaxed)
        self.assertEqual(tight[1:3],('A',('stop','stop')))
        self.assertEqual(tight[0]-loose[0],F(3,25))
        rejected=policy_economics(p,loose[1],list(loose[2]),list(loose[3]))
        self.assertIn('Branch query-cost budget exceeded.',rejected['violations'])
        self.assertTrue(policy_economics(p,'C',['stop','stop'],['pass']*4)['violations'])
        # Expected fees cannot replace a pathwise constraint: a rare second
        # query can exceed the path limit even when ex ante fees fit a budget.
        p2=parameters(budget='3/4')
        row=policy_economics(parameters(),loose[1],list(loose[2]),list(loose[3]))
        self.assertLessEqual(row['cost'],F(3,4))
        self.assertTrue(policy_economics(p2,loose[1],list(loose[2]),list(loose[3]))['violations'])

    def test_semantic_isomorphism_preserves_all_policy_economics(self):
        old=parameters();new=self.byid['c230']['information_parameters']
        mapping={'A':'C','B':'A','C':'B','stop':'stop','none':'none'}
        old_rows=enumerate_reference(old)
        for row in old_rows:
            _,f,qs,acts,e=row
            transformed=policy_economics(new,mapping[f],[mapping[q] for q in qs],list(acts))
            self.assertEqual(e,transformed)
        ref=old_rows[0];newref=enumerate_reference(new)[0]
        self.assertEqual(newref[1:4],(mapping[ref[1]],tuple(mapping[q] for q in ref[2]),ref[3]))
        self.assertEqual(newref[0],ref[0])
        self.assertNotEqual(new['weights'],BASE_WEIGHTS)

    def test_outcome_uses_actions_not_claimed_payoff(self):
        for c in self.cases:
            a=copy.deepcopy(c['stages'][0]['answers'])
            expected=economic_outcome(c,a)
            self.assertTrue(expected['action_feasible'])
            self.assertEqual(expected['decision_regret'],0)
            a['optimal_net_value']=1000000;a['expected_query_cost']=-999
            self.assertEqual(economic_outcome(c,a),expected)
            a.update(first_query='none',second_queries=['stop','stop'],terminal_actions=['pass']*4)
            out=economic_outcome(c,a)
            self.assertTrue(out['action_feasible']);self.assertEqual(out['objective'],0)
            self.assertGreater(out['decision_regret'],0)

    def test_invalid_policy_types_and_structure(self):
        p=parameters()
        bad=[(True,['stop','stop'],['pass']*4),('A',( 'stop','stop'),['pass']*4),
             ('A',['B','stop'],['pass',True,'hedge','hedge']),('A',['A','stop'],['pass']*4),
             ('none',['B','stop'],['pass']*4),('none',['stop','stop'],['pass','hedge','pass','pass']),
             ('A',['stop','stop'],['pass','hedge','pass','pass']),('A',[None,'stop'],['pass']*4),
             ('unknown',['stop','stop'],['pass']*4)]
        for args in bad:self.assertTrue(policy_economics(p,*args)['violations'])
        self.assertFalse(economic_outcome(self.cases[0],None)['action_feasible'])
        a={'first_query':'A','second_queries':['A','stop'],'terminal_actions':['pass']*4}
        out=economic_outcome(self.cases[0],a)
        self.assertIsNone(out['objective']);self.assertIsNone(out['decision_regret'])

    def test_gross_route_requires_equivalent_optimal_gross(self):
        for c in self.cases:
            p=c['information_parameters'];ranked=enumerate_reference(p)
            winners=[r for r in ranked if r[0]==ranked[0][0]]
            self.assertEqual(len({r[4]['gross'] for r in winners}),1)
            self.assertIn(['in_joint','in_actions','in_query'],scalar_routes(p)['expected_gross_payoff'])
        tied=parameters(costs=('1/2','3/4','3/2'))
        ranked=enumerate_reference(tied)
        winners=[r for r in ranked if r[0]==ranked[0][0]]
        self.assertGreater(len({r[4]['gross'] for r in winners}),1)
        self.assertEqual(scalar_routes(tied)['expected_gross_payoff'],[['in_joint','in_actions','in_format']])

    def test_minimum_source_routes(self):
        from challenge_engine import score
        for c in self.cases:
            t=c['stages'][0]
            response={'question_id':c['id'],'turn':1,'answers':t['answers'],
                      'evidence':{k:v[0] for k,v in t['evidence_routes'].items()}}
            scored=score(t,response,c['id'],1,set(c['oracle_document_ids']))
            self.assertTrue(scored['all_supported_correct'])
            self.assertEqual(t['evidence_routes']['no_information_value'],[['in_joint','in_actions']])
            self.assertNotIn('in_actions',t['evidence_routes']['expected_query_cost'][0])
            self.assertNotIn('in_query',t['evidence_routes']['expected_gross_payoff'][0])
            response['evidence']['no_information_value']=['in_joint']
            self.assertFalse(score(t,response,c['id'],1,set(c['oracle_document_ids']))['all_supported_correct'])


if __name__=='__main__':unittest.main()
