"""Evaluator-only finite-population economics; never part of public reward."""
from fractions import Fraction as F
import itertools

from .contract import pair
from .verifier import reconstruct, _allocation


def population_diagnostics(episode, allocation):
    """Compare public estimates with complete synthetic population means.

    The oracle uses the SAME public eligibility, capacities, funding and fees.
    It maximizes finite-population expected excess, with no estimator penalty.
    This is a stipulated synthetic diagnostic, not future realized performance.
    Returned hidden means and choices must stay outside candidate traces.
    """
    context = reconstruct(episode.public_files)
    selected = _allocation(context, allocation)
    members = {cell: [] for cell in context['contract']['cells']}
    for row in episode.private_population:
        if row['matured']:
            members[row['cell_id']].append(row['markout_bp'])
    true_means = {cell: F(sum(values), len(values)) if values else None for cell, values in members.items()}
    baskets = context['definitions']

    def economics(quantities, ledger):
        gross = F(0)
        for quantity, basket in zip(quantities, baskets):
            if quantity:
                mean = sum(F(*weight) * true_means[cell] for cell, weight in basket['weights'].items() if weight[0])
                gross += F(quantity * basket['principal_cents'], 10000) * mean
        excess = (F(*ledger['estimated_excess_cents']) - F(*ledger['estimated_gross_cents']) + gross)
        return gross, excess

    selected_gross, selected_excess = economics(allocation, selected)
    feasible = []
    for quantities in itertools.product(*(range(row['capacity'] + 1) if status['eligible'] else range(1)
                                           for row, status in zip(baskets, context['baskets']))):
        try:
            ledger = _allocation(context, list(quantities))
        except ValueError:
            continue
        gross, excess = economics(quantities, ledger)
        feasible.append((excess, ledger['invested_principal_cents'], quantities))
    best = min(feasible, key=lambda row: (-row[0], row[1], row[2]))
    return {'scope': 'complete synthetic finite population; same public feasibility; separate from reward',
            'true_cell_mean_bp': {cell: None if value is None else pair(value) for cell, value in true_means.items()},
            'true_gross_cents': pair(selected_gross), 'true_excess_cents': pair(selected_excess),
            'estimated_minus_true_gross_cents': pair(F(*selected['estimated_gross_cents']) - selected_gross),
            'best_feasible_true_excess_cents': pair(best[0]),
            'finite_population_regret_cents': pair(best[0] - selected_excess),
            'best_feasible_allocation': list(best[2])}
