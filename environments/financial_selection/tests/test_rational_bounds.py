"""Large valid exact-rational case under Python's default serialization limit."""
from fractions import Fraction as F
import json
import math
import unittest

from financial_selection.contract import MAX_FILE_BYTES, base_contract, pair, parse_public, strict_json
from financial_selection.generator import _csv
from financial_selection.reference import solve


class RationalBounds(unittest.TestCase):
    def test_full_census_with_distinct_probability_numerators_serializes(self):
        universe, features, labels = [], [], []
        for index, numerator in enumerate(range(488, 1000)):
            probability = F(numerator, 1000)
            event = f'e{index:03d}'
            universe.append(dict(event_id=event, decision_tick=1, label_maturity_tick=2,
                                 archived=1, pi_num=probability.numerator, pi_den=probability.denominator))
            features.append(dict(event_id=event, release_id=f'f{index}', published_tick=1,
                                 revision=0, signal=1, regime='r'))
            labels.append(dict(event_id=event, release_id=f'y{index}', published_tick=2,
                               revision=0, markout_bp=10**12 - index))
        contract = base_contract(2, ['r'])
        contract['lambda'] = [10, 1]
        deployment = dict(initial_cash_cents=10**12, credit_limit_cents=10**12,
                          lending_rate=[1, 999999999989], borrowing_rate=[1, 999999999959],
                          baskets=[dict(basket_id='b', weights={'r:negative': [0, 1], 'r:nonnegative': [1, 1]},
                                        principal_cents=10**12, capacity=1, entry_linear_cents=0,
                                        entry_quadratic_cents=0, exit_linear_cents=0)])
        public = {'contract.json': json.dumps(contract), 'deployment.json': json.dumps(deployment),
                  'universe.csv': _csv('universe.csv', universe),
                  'feature_releases.csv': _csv('feature_releases.csv', features),
                  'observed_labels.csv': _csv('observed_labels.csv', labels)}
        parse_public(public)
        answer = solve(public)
        encoded = json.dumps(answer)
        self.assertLess(len(encoded.encode()), MAX_FILE_BYTES)
        self.assertEqual(strict_json(encoded), answer)
        variance = next(row['variance_bp2'] for row in answer['cells'] if row['status'] == 'eligible')
        self.assertGreater(len(str(variance[1])), 400)
        denominator = math.lcm(*range(1, 1001))
        self.assertLess(len(str(denominator)), 434)


if __name__ == '__main__':
    unittest.main()
