"""Keyed synthetic populations; hidden outcomes never enter public packets."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import io
import json
import random

from .contract import HEADERS, base_contract, pair, parse_public


@dataclass(frozen=True)
class Episode:
    public_files: dict[str, str]
    private_population: tuple[dict, ...]
    identity: str


def _csv(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=HEADERS[name], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _stream(key, split, index, purpose):
    message = json.dumps(['selection-generator-v1', split, index, purpose],
                         separators=(',', ':')).encode()
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return random.Random(int.from_bytes(digest, 'big'))


def generate(key: bytes, split: str, index: int, *, condition='mixed') -> Episode:
    """Generate from independent keyed population, archive and presentation streams.

    Split domains prevent accidental equality; they do not establish empirical
    out-of-distribution validity. Keep keys, generated packets and complete
    populations private. No guaranteed profitable action is planted after
    observing the archive realization.
    """
    if type(key) is not bytes or len(key) < 32:
        raise ValueError('A private key of at least 32 bytes is required')
    if split not in ('train', 'dev', 'final') or type(index) is not int or not 0 <= index < 2**63:
        raise ValueError('Invalid generation identity')
    if condition not in ('mixed', 'positive', 'negative', 'unsupported', 'sparse'):
        raise ValueError('Unknown synthetic condition')
    population_rng = _stream(key, split, index, 'population')
    sampling_rng = _stream(key, split, index, 'archive')
    presentation_rng = _stream(key, split, index, 'presentation')
    deployment_rng = _stream(key, split, index, 'deployment')
    regimes = ('active', 'calm')
    cells = sorted(f'{regime}:{sign}' for regime in regimes for sign in ('negative', 'nonnegative'))
    cutoff = 1000
    population = []
    for cell_index, cell in enumerate(cells):
        regime, sign = cell.split(':')
        location = (90 if condition == 'positive' else -90 if condition == 'negative'
                    else population_rng.randint(-50, 80))
        for member in range(32):
            # Sampling probability and outcomes share a predecision covariate.
            # Archive draws occur only after the whole fixed population exists.
            covariate = population_rng.choice((-2, -1, 1, 2))
            probability = Fraction(1, 8) if condition == 'sparse' else Fraction(3, 4) if covariate > 0 else Fraction(1, 2)
            if condition == 'unsupported' and cell_index == 0 and member == 0:
                probability = Fraction(0)
            event = f'e{cell_index:02d}_{member:03d}'
            decision = 50 + 2 * member
            population.append({'event_id': event, 'decision_tick': decision,
                               'label_maturity_tick': decision + 20,
                               'cell_id': cell, 'signal': (-1 if sign == 'negative' else 1) * abs(covariate),
                               'probability': pair(probability),
                               'markout_bp': location + 15 * covariate + population_rng.randint(-12, 12)})
    # Future observations must not enlarge the matured census.
    for member in range(4):
        population.append({'event_id': f'future_{member}', 'decision_tick': cutoff + member,
                           'label_maturity_tick': cutoff + member + 10,
                           'cell_id': cells[member], 'signal': -1 if 'negative' == cells[member].split(':')[1] else 1,
                           'probability': [1, 1], 'markout_bp': population_rng.randint(-200, 200)})

    universe, features, labels, private = [], [], [], []
    for member in population:
        numerator, denominator = member['probability']
        archived = sampling_rng.randrange(denominator) < numerator
        event, decision, maturity = member['event_id'], member['decision_tick'], member['label_maturity_tick']
        regime = member['cell_id'].split(':')[0]
        public_probability = ['', ''] if condition == 'unsupported' and event == 'e01_000' else member['probability']
        universe.append({'event_id': event, 'decision_tick': decision, 'label_maturity_tick': maturity,
                         'archived': int(archived), 'pi_num': public_probability[0], 'pi_den': public_probability[1]})
        current = {'event_id': event, 'release_id': f'f_{event}_current',
                   'published_tick': decision, 'revision': 2,
                   'signal': member['signal'], 'regime': regime}
        features.extend([
            dict(current, release_id=f'f_{event}_old', revision=1, signal=-member['signal']),
            current,
            dict(current, release_id=f'f_{event}_late', published_tick=decision + 1,
                 revision=99, signal=-member['signal'], regime=regimes[1 - regimes.index(regime)]),
        ])
        if archived:
            label = {'event_id': event, 'release_id': f'y_{event}_current',
                     'published_tick': maturity, 'revision': 2, 'markout_bp': member['markout_bp']}
            labels.extend([
                dict(label, release_id=f'y_{event}_old', revision=1, markout_bp=member['markout_bp'] + 500),
                label,
                dict(label, release_id=f'y_{event}_late', published_tick=max(cutoff + 1, maturity + 1),
                     revision=99, markout_bp=member['markout_bp'] - 1000),
            ])
        private.append(dict(member, archived=archived, matured=maturity <= cutoff))
    features.extend(dict(row) for row in features[:5])
    labels.extend(dict(row) for row in labels[:5])
    for rows in (universe, features, labels):
        presentation_rng.shuffle(rows)

    baskets = []
    for index_basket in range(4):
        weights = {cell: Fraction(0) for cell in cells}
        weights[cells[index_basket]] = Fraction(3, 4)
        weights[cells[(index_basket + 1) % len(cells)]] = Fraction(1, 4)
        baskets.append({'basket_id': f'b{index_basket}',
                        'weights': {cell: pair(weights[cell]) for cell in cells},
                        'principal_cents': deployment_rng.choice((10000, 15000, 20000)),
                        'capacity': deployment_rng.choice((1, 2, 3)),
                        'entry_linear_cents': deployment_rng.randint(1, 8),
                        'entry_quadratic_cents': deployment_rng.randint(1, 4),
                        'exit_linear_cents': deployment_rng.randint(1, 6)})
    deployment = {'initial_cash_cents': 40000, 'credit_limit_cents': 15000,
                  'lending_rate': [1, 10000], 'borrowing_rate': [3, 1000], 'baskets': baskets}
    contract = base_contract(cutoff, regimes, [row['event_id'] for row in population[::17]])
    public = {'contract.json': json.dumps(contract, sort_keys=True, separators=(',', ':')) + '\n',
              'deployment.json': json.dumps(deployment, sort_keys=True, separators=(',', ':')) + '\n',
              'universe.csv': _csv('universe.csv', universe),
              'feature_releases.csv': _csv('feature_releases.csv', features),
              'observed_labels.csv': _csv('observed_labels.csv', labels)}
    parse_public(public)
    identity = hmac.new(key, json.dumps(['selection-episode-v1', split, index, condition], separators=(',', ':')).encode(),
                        hashlib.sha256).hexdigest()
    return Episode(public, tuple(private), identity)
