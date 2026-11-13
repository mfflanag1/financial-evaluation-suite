"""Hand-authored exact controls, independent of reference and generator files."""
from copy import deepcopy
import csv
from fractions import Fraction as F
import io
import itertools
import json
import unittest

from financial_selection.contract import HEADERS, PacketError, base_contract
from financial_selection.verifier import (ceil_sqrt_fraction, expected_policy,
                                           grade, reconstruct)


def csv_text(name, rows):
    output=io.StringIO(newline='');writer=csv.writer(output,lineterminator='\n')
    writer.writerow(HEADERS[name]);writer.writerows(rows)
    return output.getvalue()


def fixture(mask=(1,1,0,1,0,1), gate=0):
    """Six-unit population; labels contain only the realised selected sample."""
    ids=['a0','a1','a2','b0','b1','b2'];ys=[80,10,-30,30,-20,-40]
    probabilities=[(1,1),(1,2),(1,4)]*2
    contract=base_contract(20,regimes=('r',),audit_ids=ids)
    contract.update(minimum_observations=gate,minimum_ess=[gate,1])
    cells=contract['cells'];baskets=[]
    for name,weights in [('AA',([1,1],[0,1])),('TG',([1,4],[3,4]))]:
        baskets.append(dict(basket_id=name,weights=dict(zip(cells,weights)),principal_cents=4000,capacity=2,
                            entry_linear_cents=2,entry_quadratic_cents=1,exit_linear_cents=1))
    deployment=dict(initial_cash_cents=10000,credit_limit_cents=4000,lending_rate=[1,10000],borrowing_rate=[1,2500],baskets=baskets)
    universe=[[event,10,15,int(observed),*p] for event,observed,p in zip(ids,mask,probabilities)]
    features=[[event,'f'+event,9,0,-1 if index<3 else 1,'r'] for index,event in enumerate(ids)]
    labels=[[event,'y'+event,16,0,outcome] for event,observed,outcome in zip(ids,mask,ys) if observed]
    return {'contract.json':json.dumps(contract),'deployment.json':json.dumps(deployment),
            'universe.csv':csv_text('universe.csv',universe),'feature_releases.csv':csv_text('feature_releases.csv',features),
            'observed_labels.csv':csv_text('observed_labels.csv',labels)}


def change_json(public,name,update):
    value=json.loads(public[name]);update(value);public[name]=json.dumps(value)


def change_csv(public,name,update):
    rows=list(csv.reader(io.StringIO(public[name])))[1:];update(rows);public[name]=csv_text(name,rows)


def number(pair):
    return F(*pair)


class SelectionVerifierControls(unittest.TestCase):
    def test_six_unit_exact_ht_variance_and_effective_samples(self):
        result=expected_policy(fixture())
        a,b=result['cells']
        self.assertEqual(number(a['mean_bp']),F(100,3))
        self.assertEqual(number(b['mean_bp']),F(-130,3))
        self.assertEqual(number(a['variance_bp2']),F(200,9))
        self.assertEqual(number(b['variance_bp2']),F(6400,3))
        self.assertEqual(number(a['effective_sample_size']),F(9,5))
        self.assertEqual(number(b['effective_sample_size']),F(25,17))
        self.assertEqual(number(result['baskets'][1]['mean_bp']),F(-145,6))
        self.assertNotEqual(number(a['mean_bp']),F(90,2))  # complete-case mean
        self.assertNotEqual(number(b['mean_bp']),F(-130,5))  # self-normalized estimate
        self.assertEqual(result['allocation'],[2,0])

    def test_all_sampling_masks_prove_ht_unbiased_and_variance_identity(self):
        probabilities=[F(1),F(1,2),F(1,4)]*2
        weighted_mean=weighted_variance=expected_design_variance=mass=F(0)
        target=F(-5,2)
        for mask in itertools.product((0,1),repeat=6):
            probability=F(1)
            for included,p in zip(mask,probabilities):probability*=p if included else 1-p
            if not probability:continue
            result=expected_policy(fixture(mask));mean=number(result['baskets'][1]['mean_bp'])
            variance=F(1,16)*number(result['cells'][0]['variance_bp2'])+F(9,16)*number(result['cells'][1]['variance_bp2'])
            mass+=probability;weighted_mean+=probability*mean
            weighted_variance+=probability*(mean-target)**2;expected_design_variance+=probability*variance
        self.assertEqual(mass,1);self.assertEqual(weighted_mean,target)
        self.assertEqual(weighted_variance,F(3100,9));self.assertEqual(expected_design_variance,weighted_variance)

    def test_shared_cell_covariance_and_one_period_cash(self):
        result=expected_policy(fixture(),[2,1]);matrix=result['basket_covariance_cents2'];ledger=result['ledger']
        self.assertEqual(number(matrix[0][0]),F(32,9));self.assertEqual(number(matrix[0][1]),F(8,9))
        self.assertEqual(number(matrix[1][1]),F(1730,9))
        self.assertEqual(number(ledger['sampling_variance_cents2']),210)
        self.assertNotEqual(number(ledger['sampling_variance_cents2']),4*number(matrix[0][0])+number(matrix[1][1]))
        self.assertEqual(ledger['pretrade_cash_cents'],-2011)
        self.assertEqual(number(ledger['interest_cents']),F(-2011,2500))
        self.assertEqual(number(ledger['estimated_terminal_cents']),F(25005489,2500))
        self.assertEqual(number(ledger['estimated_excess_cents']),F(2989,2500))

    def test_credit_boundary_is_inclusive_and_next_cent_fails(self):
        public=fixture();change_json(public,'deployment.json',lambda d:d.update(credit_limit_cents=2011))
        self.assertEqual(expected_policy(public,[2,1])['ledger']['pretrade_cash_cents'],-2011)
        change_json(public,'deployment.json',lambda d:d.update(credit_limit_cents=2010))
        with self.assertRaisesRegex(ValueError,'credit limit'):expected_policy(public,[2,1])

    def test_exact_sqrt_bisection_at_and_next_to_large_squares(self):
        for root in (0,1,7,10**30):
            self.assertEqual(ceil_sqrt_fraction(F(root*root)),root)
            self.assertEqual(ceil_sqrt_fraction(F(root*root)+F(1,10**40)),root+1)
            if root:self.assertEqual(ceil_sqrt_fraction(F(root*root)-F(1,10**40)),root)

    def test_vintage_cutoffs_revision_ties_and_duplicates(self):
        public=fixture();base=expected_policy(public)
        def additions(rows):
            rows.append(['a0','fnew',10,1,-1,'r'])
            rows.append(['a0','flate',11,99,1,'r'])
            rows.append(deepcopy(rows[0]))
        change_csv(public,'feature_releases.csv',additions)
        change_csv(public,'observed_labels.csv',lambda rows:rows.extend([['a0','ynew',20,1,80],['a0','ylate',21,99,-99999]]))
        result=expected_policy(public)
        self.assertEqual(result['audit'][0]['feature_release_id'],'fnew')
        self.assertEqual(result['audit'][0]['label_release_id'],'ynew')
        self.assertEqual(result['cells'],base['cells'])
        # Later revision at the same eligible publication takes precedence.
        change_csv(public,'observed_labels.csv',lambda rows:rows.append(['a0','yrevised',20,2,83]))
        changed=expected_policy(public)
        self.assertEqual(number(changed['cells'][0]['mean_bp'])-number(base['cells'][0]['mean_bp']),1)

    def test_unmatured_events_have_null_audit_and_do_not_change_census(self):
        public=fixture();original=expected_policy(public)
        change_csv(public,'universe.csv',lambda rows:rows.append(['future',10,21,0,1,2]))
        change_json(public,'contract.json',lambda d:d['audit_event_ids'].append('future'))
        result=expected_policy(public)
        self.assertEqual(result['cells'],original['cells'])
        self.assertEqual(result['audit'][-1],{'event_id':'future','population_member':False,'cell_id':None,'feature_release_id':None,'label_release_id':None})

    def test_support_checks_unobserved_members_and_unknown_design(self):
        for probability in ((0,1),('','')):
            public=fixture()
            change_csv(public,'universe.csv',lambda rows:rows[2].__setitem__(slice(4,6),probability))
            result=expected_policy(public)
            self.assertEqual(result['cells'][0]['status'],'unsupported')
            self.assertIsNone(result['cells'][0]['mean_bp'])
            self.assertIsNone(result['baskets'][0]['mean_bp'])
            self.assertIsNone(result['basket_covariance_cents2'][0][1])
            self.assertEqual(result['allocation'],[0,0])

    def test_empty_supported_archive_retains_zero_estimates_but_is_gated(self):
        public=fixture(gate=4)
        change_csv(public,'universe.csv',lambda rows:[row.__setitem__(slice(3,6),[0,1,2]) for row in rows])
        public['observed_labels.csv']=csv_text('observed_labels.csv',[])
        result=expected_policy(public)
        for row in result['cells']:
            self.assertEqual(row['population_count'],3);self.assertEqual(row['observed_count'],0)
            self.assertEqual(row['status'],'insufficient_sample')
            self.assertEqual(row['mean_bp'],[0,1]);self.assertEqual(row['variance_bp2'],[0,1]);self.assertEqual(row['effective_sample_size'],[0,1])
        self.assertEqual(result['allocation'],[0,0])

    def test_census_probability_one_still_obeys_sample_gate(self):
        public=fixture(mask=(1,)*6,gate=4)
        change_csv(public,'universe.csv',lambda rows:[row.__setitem__(slice(4,6),[1,1]) for row in rows])
        result=expected_policy(public)
        self.assertTrue(all(row['status']=='insufficient_sample' and row['variance_bp2']==[0,1] for row in result['cells']))

    def test_malformed_public_inputs_raise_instead_of_candidate_zero(self):
        cases=[]
        missing=fixture();change_csv(missing,'feature_releases.csv',lambda rows:rows.__setitem__(slice(None),[row for row in rows if row[0]!='a2']));cases.append(missing)
        absent_label=fixture();change_csv(absent_label,'observed_labels.csv',lambda rows:rows.pop(0));cases.append(absent_label)
        impossible=fixture();change_csv(impossible,'universe.csv',lambda rows:rows[2].__setitem__(slice(4,6),[1,1]));cases.append(impossible)
        conflict=fixture();change_csv(conflict,'feature_releases.csv',lambda rows:rows.append(['a0','different',9,0,1,'r']));cases.append(conflict)
        excessive=fixture();excessive['contract.json']=' '*2097153;cases.append(excessive)
        for public in cases:
            with self.assertRaises(PacketError):grade(public,'{}')

    def test_optimum_ties_use_principal_then_lexicographic_vector(self):
        public=fixture()
        change_json(public,'contract.json',lambda d:d.update(**{'lambda':[0,1]}))
        def equal(d):
            d.update(initial_cash_cents=4000,credit_limit_cents=0,lending_rate=[0,1],borrowing_rate=[0,1])
            for b in d['baskets']:
                b.update(capacity=1,weights={'r:negative':[1,1],'r:nonnegative':[0,1]},entry_linear_cents=0,entry_quadratic_cents=0,exit_linear_cents=0)
        change_json(public,'deployment.json',equal)
        self.assertEqual(expected_policy(public)['allocation'],[0,1])
        change_csv(public,'observed_labels.csv',lambda rows:[row.__setitem__(4,0) for row in rows])
        self.assertEqual(expected_policy(public)['allocation'],[0,0])

    def test_correct_policy_and_nonoptimal_feasible_policy_have_separate_credit(self):
        public=fixture();solution=expected_policy(public)
        grade_ok=grade(public,json.dumps(solution))
        self.assertEqual(grade_ok['reward'],1);self.assertEqual(grade_ok['status'],'scored');self.assertEqual(grade_ok['outcome'],'correct')
        zero=expected_policy(public,[0,0]);partial=grade(public,json.dumps(zero))
        self.assertEqual(partial['reward'],0.7);self.assertEqual(partial['components']['ledger'],1)
        self.assertEqual(partial['components']['optimal_allocation'],0)

    def test_wrong_wellformed_statistics_lose_only_statistical_credit(self):
        public=fixture();solution=expected_policy(public)
        solution['cells'][0]['mean_bp']=[0,1]
        result=grade(public,json.dumps(solution))
        self.assertGreater(result['reward'],0.65);self.assertLess(result['reward'],1)
        self.assertEqual(result['components']['ledger'],1);self.assertEqual(result['components']['optimal_allocation'],1)
        self.assertEqual(result['components']['audit'],1)
        solution=expected_policy(public);solution['basket_covariance_cents2'][1][0]=[0,1]
        self.assertLess(grade(public,json.dumps(solution))['components']['statistics'],1)

    def test_schema_identity_and_infeasibility_zero_every_component(self):
        public=fixture();solution=expected_policy(public);mutations=[]
        def make(update):
            value=deepcopy(solution);update(value);mutations.append(json.dumps(value))
        make(lambda d:d.update(input_sha256='0'*64))
        make(lambda d:d.update(allocation=[2,2]))
        make(lambda d:d.update(allocation=[True,0]))
        make(lambda d:d.update(allocation=[2.0,0]))
        make(lambda d:d['cells'][0].update(population_count=True))
        make(lambda d:d['cells'][0].update(mean_bp=[2,2]))
        make(lambda d:d['cells'][0].update(mean_bp=[0,0]))
        make(lambda d:d['cells'].reverse())
        make(lambda d:d['ledger'].update(unknown=0))
        raw=json.dumps(solution)
        mutations += [raw[:-1]+',"schema":"duplicate"}',raw.replace('"population_count": 3','"population_count": 1e999',1),
                      raw.replace('"population_count": 3','"population_count": NaN',1),
                      raw.replace('"population_count": 3','"population_count": '+('1'*4097),1),' '*2097153]
        for text in mutations:
            result=grade(public,text)
            self.assertEqual(result['status'],'invalid_submission',text[:80])
            self.assertEqual(result['reward'],0);self.assertTrue(all(value==0 for value in result['components'].values()))

    def test_large_but_bounded_reduced_rational_is_wellformed_wrong_value(self):
        public=fixture();solution=expected_policy(public)
        solution['cells'][0]['mean_bp']=[10**4095+1,10**4095]
        result=grade(public,json.dumps(solution))
        self.assertEqual(result['status'],'scored');self.assertEqual(result['outcome'],'partial');self.assertGreater(result['reward'],0)

    def test_candidate_bytes_utf8_and_exact_size_boundary(self):
        public=fixture();raw=json.dumps(expected_policy(public)).encode('utf-8')
        at_limit=raw+b' '*(2097152-len(raw))
        self.assertEqual(grade(public,at_limit)['reward'],1)
        self.assertEqual(grade(public,at_limit+b' ')['status'],'invalid_submission')
        malformed=grade(public,b'\xff')
        self.assertEqual(malformed['reward'],0);self.assertEqual(malformed['reason'],'invalid UTF-8 submission')

    def test_exact_packet_identity_includes_line_endings_and_input_order(self):
        public=fixture();original=expected_policy(public)
        public['universe.csv']=public['universe.csv'].replace('\n','\r\n')
        changed=expected_policy(public)
        self.assertNotEqual(changed['input_sha256'],original['input_sha256'])
        self.assertEqual(changed['cells'],original['cells'])
        self.assertEqual(grade(public,json.dumps(original))['reward'],0)
        change_csv(public,'feature_releases.csv',lambda rows:rows.reverse())
        reordered=expected_policy(public)
        self.assertEqual(reordered['cells'],original['cells']);self.assertEqual(reordered['audit'],original['audit'])


if __name__=='__main__':unittest.main()
