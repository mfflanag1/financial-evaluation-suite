"""Exact declarative verifier for the experimental selection contract.

Financial calculations are separate from the public reference and generator.
This module reads data only: it never imports or executes candidate source.
Malformed public inputs raise PacketError; malformed submissions score zero.
"""
from __future__ import annotations

from fractions import Fraction as F
import itertools
import math

from .contract import (MAX_FILE_BYTES, MAX_OUTPUT_DIGITS, OUTPUT_SCHEMA,
                       PacketError, identifier, input_digest, pair,
                       parse_public, strict_json)


AUDIT_KEYS = ('event_id', 'population_member', 'cell_id', 'feature_release_id', 'label_release_id')
CELL_KEYS = ('cell_id', 'population_count', 'observed_count', 'status', 'mean_bp',
             'variance_bp2', 'effective_sample_size')
BASKET_KEYS = ('basket_id', 'eligible', 'mean_bp')
LEDGER_INTEGERS = ('invested_principal_cents', 'entry_cost_cents', 'exit_cost_cents',
                   'pretrade_cash_cents', 'penalty_microcents', 'objective_microcents')
LEDGER_RATIONALS = ('interest_cents', 'estimated_gross_cents', 'estimated_terminal_cents',
                    'cash_benchmark_cents', 'estimated_excess_cents', 'sampling_variance_cents2')
POLICY_KEYS = ('schema', 'input_sha256', 'audit', 'cells', 'baskets',
               'basket_covariance_cents2', 'allocation', 'ledger')


def _rational(value):
    return F(value[0], value[1])


def _encode(value):
    return None if value is None else pair(value)


def reconstruct(public_files):
    """Select public vintages and reconstruct cell and basket statistics."""
    parsed = parse_public(public_files)
    contract, deployment = parsed['contract'], parsed['deployment']
    cutoff = contract['research_cutoff']
    events = {row['event_id']: row for row in parsed['universe']}
    matured = {event for event, row in events.items() if row['label_maturity_tick'] <= cutoff}
    selected = {}
    for kind, releases in (('feature', parsed['features']), ('label', parsed['labels'])):
        by_event = {}
        # Sort oldest first and overwrite only with an eligible newer revision.
        for row in sorted(releases, key=lambda r: (r['published_tick'], r['revision'])):
            event = row['event_id']
            deadline = events[event]['decision_tick'] if kind == 'feature' else cutoff
            if event in matured and row['published_tick'] <= deadline:
                by_event[event] = row
        selected[kind] = by_event

    members = {cell: [] for cell in contract['cells']}
    event_cells = {}
    for event in sorted(matured):
        feature = selected['feature'][event]
        cell = feature['regime'] + (':negative' if feature['signal'] < 0 else ':nonnegative')
        event_cells[event] = cell
        members[cell].append(event)

    audits = []
    for event in contract['audit_event_ids']:
        included = event in matured
        audits.append({'event_id': event, 'population_member': included,
                       'cell_id': event_cells[event] if included else None,
                       'feature_release_id': selected['feature'][event]['release_id'] if included else None,
                       'label_release_id': selected['label'][event]['release_id']
                       if included and events[event]['archived'] else None})

    cells, statistics = [], {}
    for cell, population in members.items():
        observed = [event for event in population if events[event]['archived']]
        size, sample_size = len(population), len(observed)
        supported = size > 0 and all(events[event]['probability'] is not None and events[event]['probability'] > 0 for event in population)
        if not supported:
            estimate = variance = ess = None
            status = 'unsupported'
        else:
            probabilities = [events[event]['probability'] for event in observed]
            outcomes = [F(selected['label'][event]['markout_bp']) for event in observed]
            inverse = [1 / probability for probability in probabilities]
            weighted = [outcome * weight for outcome, weight in zip(outcomes, inverse)]
            estimate = sum(weighted, F(0)) / size
            # E-design variance estimate written as expanded second moments.
            variance = sum((weighted_value ** 2 - outcome ** 2 * weight
                            for weighted_value, outcome, weight in zip(weighted, outcomes, inverse)), F(0)) / size ** 2
            ess = sum(inverse, F(0)) ** 2 / sum((weight ** 2 for weight in inverse), F(0)) if sample_size else F(0)
            status = 'eligible' if sample_size >= contract['minimum_observations'] and ess >= _rational(contract['minimum_ess']) else 'insufficient_sample'
        statistics[cell] = (estimate, variance, ess, status)
        cells.append({'cell_id': cell, 'population_count': size, 'observed_count': sample_size,
                      'status': status, 'mean_bp': _encode(estimate), 'variance_bp2': _encode(variance),
                      'effective_sample_size': _encode(ess)})

    definitions = sorted(deployment['baskets'], key=lambda basket: basket['basket_id'])
    means, exposures, baskets = [], [], []
    for definition in definitions:
        weights = {cell: _rational(weight) for cell, weight in definition['weights'].items()}
        required = [cell for cell, weight in weights.items() if weight > 0]
        known = all(statistics[cell][0] is not None for cell in required)
        mean = sum((weights[cell] * statistics[cell][0] for cell in required), F(0)) if known else None
        eligible = all(statistics[cell][3] == 'eligible' for cell in required)
        exposure = {cell: F(definition['principal_cents'], 10000) * weights[cell] for cell in contract['cells']}
        means.append(mean); exposures.append(exposure)
        baskets.append({'basket_id': definition['basket_id'], 'eligible': eligible, 'mean_bp': _encode(mean)})
    covariance = []
    for j in range(len(definitions)):
        row = []
        for k in range(len(definitions)):
            if means[j] is None or means[k] is None:
                row.append(None)
            else:
                row.append(sum((exposures[j][cell] * exposures[k][cell] * statistics[cell][1]
                                for cell in contract['cells'] if exposures[j][cell] and exposures[k][cell]), F(0)))
        covariance.append(row)
    return {'input_sha256': input_digest(public_files), 'contract': contract, 'deployment': deployment,
            'definitions': definitions, 'audit': audits, 'cells': cells, 'baskets': baskets,
            'means': means, 'covariance': covariance, 'statistics': statistics}


def ceil_sqrt_fraction(value):
    """Least integer r with r*r >= value, by exact integer bisection."""
    if value < 0:
        raise ValueError('negative squared uncertainty')
    if value == 0:
        return 0
    numerator, denominator = value.numerator, value.denominator
    low, high = 0, 1
    while high * high * denominator < numerator:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if mid * mid * denominator >= numerator:
            high = mid
        else:
            low = mid
    return high


def _allocation(context, allocation):
    definitions = context['definitions']
    if not isinstance(allocation, list) or len(allocation) != len(definitions):
        raise ValueError('allocation length')
    for quantity, definition, basket in zip(allocation, definitions, context['baskets']):
        if type(quantity) is not int or not 0 <= quantity <= definition['capacity'] or (quantity and not basket['eligible']):
            raise ValueError('infeasible quantity or eligibility')
    invested = sum(q * basket['principal_cents'] for q, basket in zip(allocation, definitions))
    entry = sum(basket['entry_linear_cents'] * q + basket['entry_quadratic_cents'] * q ** 2 for q, basket in zip(allocation, definitions))
    exit_cost = sum(basket['exit_linear_cents'] * q for q, basket in zip(allocation, definitions))
    deployment = context['deployment']
    cash = deployment['initial_cash_cents'] - invested - entry
    if cash < -deployment['credit_limit_cents']:
        raise ValueError('credit limit')
    # Derive using basket means and the complete q^T C q, including covariance.
    gross = sum((F(q * basket['principal_cents'], 10000) * mean
                 for q, basket, mean in zip(allocation, definitions, context['means']) if q), F(0))
    variance = sum((allocation[j] * allocation[k] * context['covariance'][j][k]
                    for j in range(len(allocation)) for k in range(len(allocation))
                    if allocation[j] and allocation[k]), F(0))
    interest = cash * _rational(deployment['lending_rate'] if cash >= 0 else deployment['borrowing_rate'])
    # Principal cancels from wealth, avoiding a duplicated principal charge.
    terminal = deployment['initial_cash_cents'] + gross - entry + interest - exit_cost
    benchmark = deployment['initial_cash_cents'] * (1 + _rational(deployment['lending_rate']))
    excess = terminal - benchmark
    penalty = ceil_sqrt_fraction(variance * (1000000 * _rational(context['contract']['lambda'])) ** 2)
    objective = (1000000 * excess).numerator // (1000000 * excess).denominator - penalty
    return {'invested_principal_cents': invested, 'entry_cost_cents': entry, 'exit_cost_cents': exit_cost,
            'pretrade_cash_cents': cash, 'interest_cents': pair(interest), 'estimated_gross_cents': pair(gross),
            'estimated_terminal_cents': pair(terminal), 'cash_benchmark_cents': pair(benchmark),
            'estimated_excess_cents': pair(excess), 'sampling_variance_cents2': pair(variance),
            'penalty_microcents': penalty, 'objective_microcents': objective}


def _best(context):
    best = None
    ranges = [range(basket['capacity'] + 1) if row['eligible'] else range(1)
              for basket, row in zip(context['definitions'], context['baskets'])]
    for quantities in itertools.product(*ranges):
        try:
            ledger = _allocation(context, list(quantities))
        except ValueError:
            continue
        key = (-ledger['objective_microcents'], ledger['invested_principal_cents'], quantities)
        if best is None or key < best[0]:
            best = (key, list(quantities), ledger)
    if best is None:
        raise PacketError('no feasible zero allocation')
    return best[1], best[2]


def _policy(context, allocation=None):
    if allocation is None:
        allocation, ledger = _best(context)
    else:
        ledger = _allocation(context, allocation)
    return {'schema': OUTPUT_SCHEMA, 'input_sha256': context['input_sha256'], 'audit': context['audit'],
            'cells': context['cells'], 'baskets': context['baskets'],
            'basket_covariance_cents2': [[_encode(value) for value in row] for row in context['covariance']],
            'allocation': allocation, 'ledger': ledger}


def expected_policy(public_files, allocation=None):
    """Public-data solution or exact ledger for a supplied feasible allocation."""
    return _policy(reconstruct(public_files), allocation)


class SubmissionError(ValueError):
    pass


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise SubmissionError('unexpected keys')


def _integer(value, *, nonnegative=False):
    if type(value) is not int or len(str(abs(value))) > MAX_OUTPUT_DIGITS or (nonnegative and value < 0):
        raise SubmissionError('invalid integer field')


def _pair(value, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise SubmissionError('invalid rational pair')
    numerator, denominator = value
    _integer(numerator); _integer(denominator)
    if denominator <= 0 or math.gcd(numerator, denominator) != 1:
        raise SubmissionError('rational pair must be reduced with positive denominator')


def _shape(candidate, context):
    _keys(candidate, POLICY_KEYS)
    if candidate['schema'] != OUTPUT_SCHEMA or candidate['input_sha256'] != context['input_sha256']:
        raise SubmissionError('wrong schema or input identity')
    audits = candidate['audit']
    if not isinstance(audits, list) or len(audits) != len(context['audit']):
        raise SubmissionError('audit length')
    for row, expected in zip(audits, context['audit']):
        _keys(row, AUDIT_KEYS)
        if row['event_id'] != expected['event_id'] or type(row['population_member']) is not bool:
            raise SubmissionError('audit identity or type')
        if row['cell_id'] is not None and row['cell_id'] not in context['contract']['cells']:
            raise SubmissionError('audit cell identifier')
        for name in ('feature_release_id', 'label_release_id'):
            if row[name] is not None:
                try:
                    identifier(row[name])
                except PacketError as error:
                    raise SubmissionError('release identifier') from error
    cells = candidate['cells']
    if not isinstance(cells, list) or len(cells) != len(context['cells']):
        raise SubmissionError('cell length')
    for row, expected in zip(cells, context['cells']):
        _keys(row, CELL_KEYS)
        if row['cell_id'] != expected['cell_id'] or row['status'] not in ('unsupported', 'insufficient_sample', 'eligible'):
            raise SubmissionError('cell identity or status')
        _integer(row['population_count'], nonnegative=True); _integer(row['observed_count'], nonnegative=True)
        for name in ('mean_bp', 'variance_bp2', 'effective_sample_size'):
            _pair(row[name], nullable=True)
    baskets = candidate['baskets']
    if not isinstance(baskets, list) or len(baskets) != len(context['baskets']):
        raise SubmissionError('basket length')
    for row, expected in zip(baskets, context['baskets']):
        _keys(row, BASKET_KEYS)
        if row['basket_id'] != expected['basket_id'] or type(row['eligible']) is not bool:
            raise SubmissionError('basket identity or eligibility type')
        _pair(row['mean_bp'], nullable=True)
    matrix = candidate['basket_covariance_cents2']; size = len(baskets)
    if not isinstance(matrix, list) or len(matrix) != size:
        raise SubmissionError('covariance dimensions')
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise SubmissionError('covariance dimensions')
        for value in row:
            _pair(value, nullable=True)
    allocation = candidate['allocation']
    if not isinstance(allocation, list) or len(allocation) != size:
        raise SubmissionError('allocation length')
    for value in allocation:
        _integer(value, nonnegative=True)
    _keys(candidate['ledger'], LEDGER_INTEGERS + LEDGER_RATIONALS)
    for name in LEDGER_INTEGERS:
        _integer(candidate['ledger'][name])
    for name in LEDGER_RATIONALS:
        _pair(candidate['ledger'][name])


def _fraction_correct(pairs):
    pairs = list(pairs)
    return F(sum(left == right for left, right in pairs), len(pairs)) if pairs else F(1)


def grade(public_files, candidate_text):
    """Grade a declarative JSON policy; errors in the public task propagate."""
    context = reconstruct(public_files)
    zero = {'reward': 0.0, 'components': {'audit': 0.0, 'statistics': 0.0, 'ledger': 0.0, 'optimal_allocation': 0.0},
            'status': 'invalid_submission'}
    try:
        if isinstance(candidate_text, bytes):
            if len(candidate_text) > MAX_FILE_BYTES:
                raise SubmissionError('submission exceeds text limit')
            candidate_text = candidate_text.decode('utf-8')
        elif not isinstance(candidate_text, str):
            raise SubmissionError('submission must be UTF-8 text or bytes')
        if len(candidate_text.encode('utf-8')) > MAX_FILE_BYTES:
            raise SubmissionError('submission exceeds text limit')
        candidate = strict_json(candidate_text)
        _shape(candidate, context)
        selected = _allocation(context, candidate['allocation'])
    except UnicodeError:
        return dict(zero, reason='invalid UTF-8 submission')
    except (PacketError, SubmissionError, ValueError, TypeError, RecursionError):
        return dict(zero, reason='invalid schema, input identity, or feasible allocation')
    expected = _policy(context)
    audit = _fraction_correct((row[name], correct[name]) for row, correct in zip(candidate['audit'], expected['audit']) for name in AUDIT_KEYS[1:])
    stats_pairs = [(row[name], correct[name]) for row, correct in zip(candidate['cells'], expected['cells']) for name in CELL_KEYS[1:]]
    stats_pairs += [(row[name], correct[name]) for row, correct in zip(candidate['baskets'], expected['baskets']) for name in BASKET_KEYS[1:]]
    # Each covariance entry is checked; changing only one symmetric half loses credit.
    stats_pairs += [(candidate['basket_covariance_cents2'][j][k], expected['basket_covariance_cents2'][j][k])
                    for j in range(len(context['baskets'])) for k in range(len(context['baskets']))]
    statistics = _fraction_correct(stats_pairs)
    ledger = _fraction_correct((candidate['ledger'][name], selected[name]) for name in LEDGER_INTEGERS + LEDGER_RATIONALS)
    optimum = F(candidate['allocation'] == expected['allocation'])
    reward = F(1,5)*audit + F(7,20)*statistics + F(3,20)*ledger + F(3,10)*optimum
    return {'reward': float(reward), 'components': {'audit': float(audit), 'statistics': float(statistics),
            'ledger': float(ledger), 'optimal_allocation': float(optimum)},
            'status': 'scored', 'outcome': 'correct' if reward == 1 else 'partial', 'reason': None}
