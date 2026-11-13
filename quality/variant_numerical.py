"""Exact public-contract audit of the six related information variants.

Backward induction uses unnormalised probability mass at each information set;
it neither imports a stored policy nor assumes conditional signal independence.
The variants diagnose fee, budget, payoff and label changes of one mechanism.
"""
from fractions import Fraction as F
import hashlib
import itertools
import json
import re


SOURCE_HASHES = {
    "c225": "53eabbf659edf55dee2e75013adfb1beb42025327e3d0a4df18093aa89603a15",
    "c226": "b94865725d401e12d7586a3ca8c7a2de096cb42d2c4b787b06a0e6139f60e72f",
    "c227": "301b1d7b4e1f8a12cd96edaf0c1b320d60daa1f54ff5bb6f3b9c4e49d13e0660",
    "c228": "7036f300ed06a08bd82d3d9c6bfbed3081d54ec344c070de0cdeb01d4b2cdd77",
    "c229": "162c0c220cd14038b878707e603775dc4e45f61ba1a6f32aa21c152bec569601",
    "c230": "548d74c0ace21dcb8e87328fb634fb10620945481479a2abe45b3e3dc0c6278a"
}


def parse(question):
    documents = {row['id']: row['text'] for row in question['sources']}
    joint = re.findall(r'for (good|bad) are \[([^]]+)\]/(\d+)', documents['in_joint'])
    if len(joint) != 2:
        raise ValueError('Unrecognised public joint-law contract')
    cells = []
    for state, amounts, denominator in joint:
        probabilities = [F(int(value.strip()), int(denominator)) for value in amounts.split(',')]
        if len(probabilities) != 8 or any(p <= 0 for p in probabilities):
            raise ValueError('Expected eight positive joint cells per state')
        cells.extend((state, bits, probability)
                     for bits, probability in zip(itertools.product((0, 1), repeat=3), probabilities))
    if sum(row[2] for row in cells) != 1:
        raise ValueError('Joint law does not sum to one')
    payoffs = {action: {'good': F(good), 'bad': F(bad)}
               for action, good, bad in re.findall(r'(invest|hedge|pass) pays (-?\d+) in good and (-?\d+) in bad', documents['in_actions'])}
    if set(payoffs) != {'invest', 'hedge', 'pass'}:
        raise ValueError('Unrecognised terminal actions')
    costs = {name: F(value) for value, name in re.findall(r'(\d+(?:/\d+)?) for ([ABC])', documents['in_query'])}
    budget = F(re.search(r'path must be at most (\d+(?:/\d+)?)', documents['in_query'])[1])
    if set(costs) != set('ABC'):
        raise ValueError('Unrecognised query fees')
    return cells, payoffs, costs, budget


def backward_induction(cells, payoffs, costs, budget):
    """Optimize each reachable information set, including exact policy ties."""
    def terminal(subset):
        return min(((sum(p * payoffs[action][state] for state, bits, p in subset), action)
                    for action in payoffs), key=lambda row: (-row[0], row[1]))

    no_gross, no_action = terminal(cells)
    choices = [{'policy': ('none', ('stop', 'stop'), (no_action,) * 4),
                'gross': no_gross, 'fees': F(0), 'net': no_gross}]
    policy_count = len(payoffs)
    for first in sorted(costs):
        if costs[first] > budget:
            continue
        branches = []
        continuation_count = len(payoffs)
        permitted = [second for second in sorted(costs)
                     if second != first and costs[first] + costs[second] <= budget]
        continuation_count += len(permitted) * len(payoffs) ** 2
        policy_count += continuation_count ** 2
        for bit in (0, 1):
            subset = [row for row in cells if row[1]['ABC'.index(first)] == bit]
            gross, action = terminal(subset)
            options = [(gross, 'stop', (action, action), gross, F(0))]
            for second in permitted:
                descendants = [terminal([row for row in subset if row[1]['ABC'.index(second)] == value])
                               for value in (0, 1)]
                gross = sum(row[0] for row in descendants)
                fees = sum(row[2] for row in subset) * costs[second]
                options.append((gross - fees, second, tuple(row[1] for row in descendants), gross, fees))
            branches.append(min(options, key=lambda row: (-row[0], row[1], row[2])))
        gross = sum(row[3] for row in branches)
        fees = costs[first] + sum(row[4] for row in branches)
        choices.append({'policy': (first, tuple(row[1] for row in branches), branches[0][2] + branches[1][2]),
                        'gross': gross, 'fees': fees, 'net': gross - fees})
    best = min(choices, key=lambda row: (-row['net'], row['policy']))
    return best, no_gross, {'legal_full_policies': policy_count,
                          'root_optimal_values': {row['policy'][0]: str(row['net']) for row in choices},
                          'optimal_gross_fraction': str(best['gross']), 'optimal_fee_fraction': str(best['fees']),
                          'optimal_net_fraction': str(best['net'])}


def calculate(question):
    digest = hashlib.sha256(json.dumps(question['sources'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != SOURCE_HASHES[question['question_id']]:
        raise ValueError('Public source contract differs from reviewed text')
    best, baseline, certificate = backward_induction(*parse(question))
    first, second, terminal = best['policy']
    certificate.update(public_source_sha256=hashlib.sha256(json.dumps(question['sources'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                       method='Exact backward induction over joint probability cells; branch costs weighted by visitation mass.',
                       scope='All eight outputs of this fixed single-turn diagnostic variant; not an independent new mechanism.')
    return {'first_query': first, 'second_queries': list(second), 'terminal_actions': list(terminal),
            'expected_query_cost': float(best['fees']), 'expected_gross_payoff': float(best['gross']),
            'optimal_net_value': float(best['net']), 'no_information_value': float(baseline),
            'incremental_information_value': float(best['net'] - baseline)}, certificate


CHECKS = {f'c{number}': (calculate, 'Exact backward induction with branch-specific continuation choices and lexicographic policy ties.')
          for number in range(225, 231)}
