"""Exhaustive policy controls for the six related information variants."""
from fractions import Fraction as F
import itertools
import unittest

from quality import variant_numerical as audit
from quality.public_contracts import load_variant_packet


def brute_policies(cells,payoffs,costs,budget):
    """Enumerate all policies and integrate whole histories, not Bellman values."""
    policies=[('none',('stop','stop'),(action,)*4) for action in payoffs]
    for first in costs:
        if costs[first]>budget:continue
        branches=[('stop',(action,action)) for action in payoffs]
        branches += [(second,pair) for second in costs if second!=first and costs[first]+costs[second]<=budget
                     for pair in itertools.product(payoffs,repeat=2)]
        policies += [(first,(left[0],right[0]),left[1]+right[1]) for left,right in itertools.product(branches,repeat=2)]
    outcomes=[]
    for first,seconds,terminal in policies:
        gross=fees=F(0)
        for state,bits,p in cells:
            first_bit=0 if first=='none' else bits['ABC'.index(first)]
            second=seconds[first_bit];second_bit=0 if second=='stop' else bits['ABC'.index(second)]
            gross+=p*payoffs[terminal[2*first_bit+second_bit]][state]
            fees+=p*((F(0) if first=='none' else costs[first])+(F(0) if second=='stop' else costs[second]))
        outcomes.append((gross-fees,(first,seconds,terminal),gross,fees))
    return min(outcomes,key=lambda row:(-row[0],row[1])),len(policies)


class VariantArithmetic(unittest.TestCase):
    def test_backward_induction_matches_independent_full_policy_enumeration(self):
        fields=0
        for cid,(calculate,_) in audit.CHECKS.items():
            q=load_variant_packet(cid)['turns'][0];inputs=audit.parse(q)
            (net,policy,gross,fees),count=brute_policies(*inputs)
            answer,certificate=calculate(q)
            self.assertEqual((answer['first_query'],tuple(answer['second_queries']),tuple(answer['terminal_actions'])),policy)
            self.assertEqual(F(certificate['optimal_net_fraction']),net)
            self.assertEqual(F(certificate['optimal_gross_fraction']),gross)
            self.assertEqual(F(certificate['optimal_fee_fraction']),fees)
            self.assertEqual(certificate['legal_full_policies'],count)
            self.assertEqual(set(answer),set(q['outputs']));fields+=len(answer)
        self.assertEqual(fields,48)

    def test_large_costs_rationally_stop_before_any_query(self):
        answer,_=audit.calculate(load_variant_packet('c227')['turns'][0])
        self.assertEqual(answer['first_query'],'none')
        self.assertEqual(answer['terminal_actions'],['hedge']*4)
        self.assertEqual(answer['expected_query_cost'],0)
        self.assertEqual(answer['incremental_information_value'],0)

    def test_path_budget_excludes_second_query_even_if_average_fee_fits(self):
        q=load_variant_packet('c229')['turns'][0]
        cells,payoffs,costs,budget=audit.parse(q)
        best,_,_=audit.backward_induction(cells,payoffs,costs,budget)
        self.assertEqual(best['policy'][1],('stop','stop'))
        self.assertEqual(best['fees'],F(1,2))
        relaxed,_,_=audit.backward_induction(cells,payoffs,costs,F(4))
        self.assertGreater(relaxed['net'],best['net'])

    def test_continuation_fee_is_weighted_by_visited_branch(self):
        answer,certificate=audit.calculate(load_variant_packet('c228')['turns'][0])
        self.assertEqual(answer['second_queries'],['stop','A'])
        # P(B=1)=0.48 in the public joint table; only that branch pays for A.
        self.assertEqual(F(certificate['optimal_fee_fraction']),F(1,10)+F(48,100)*2)
        self.assertLess(answer['expected_query_cost'],2.1)

    def test_zero_cost_tie_obeys_full_lexicographic_policy_order(self):
        cells=[('good',bits,F(1,8)) for bits in itertools.product((0,1),repeat=3)]
        payoffs={'pass':{'good':0,'bad':0},'invest':{'good':0,'bad':0},'hedge':{'good':0,'bad':0}}
        costs=dict.fromkeys('ABC',F(0))
        best,_,_=audit.backward_induction(cells,payoffs,costs,F(0))
        exhaustive,_=brute_policies(cells,payoffs,costs,F(0))
        self.assertEqual(best['policy'],exhaustive[1])
        self.assertEqual(best['policy'],('A',('B','B'),('hedge',)*4))

    def test_signal_renaming_moves_policy_without_changing_economics(self):
        # Rename old C,A,B as new A,B,C, carrying both joint cells and fees.
        q=load_variant_packet('c230')['turns'][0];cells,payoffs,costs,budget=audit.parse(q)
        remapped=[(state,(bits[2],bits[0],bits[1]),p) for state,bits,p in cells]
        original_costs={'A':costs['C'],'B':costs['A'],'C':costs['B']}
        original,_,_=audit.backward_induction(remapped,payoffs,original_costs,budget)
        changed,_,_=audit.backward_induction(cells,payoffs,costs,budget)
        self.assertEqual(original['net'],changed['net'])
        self.assertEqual(original['policy'][0],'A');self.assertEqual(changed['policy'][0],'C')

    def test_source_change_requires_explicit_reaudit(self):
        q=load_variant_packet('c225')['turns'][0];q['sources'][2]['text']=q['sources'][2]['text'].replace('DISTINCT','REPEATED')
        with self.assertRaisesRegex(ValueError,'differs from reviewed'):
            audit.calculate(q)


if __name__=='__main__':unittest.main()
