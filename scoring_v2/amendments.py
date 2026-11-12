"""Explicit adjudications; no changes are made to frozen bank files."""
from copy import deepcopy
from .common import FROZEN, read

COUNT_FIELDS = {
    'q076': {'min_months'},
    'q083': {'n_train', 'n_purged_before_test', 'n_removed_after_test'},
    'q088': {'n_bh_discoveries', 'n_bonferroni', 'n_holm'},
}

RESEARCH_SUPPORT = {
    ('q138', 3, 'delivery_units'): 'q138_t1_1',
    ('q139', 3, 'firm_new_q4_revenue'): 'q139_t3_1',
    ('q140', 3, 'optimal_eps'): 'q140_t1_2',
    ('q141', 4, 'historical_nrr_pct'): 'q141_t4_1',
    ('q144', 3, 'minimum_upfront_funding'): 'q144_t1_1',
    ('q146', 3, 'operating_enterprise_value'): 'q146_t1_1',
    ('q160', 3, 'standard_error'): 'q160_cost_correction',
}


def research_routes(cid, turn, stage, field):
    routes = [stage['evidence'][field]]
    allowed = set(stage.get('allowed_evidence', {}).get(field, routes[0]))
    extra = RESEARCH_SUPPORT.get((cid, turn, field))
    if extra:
        allowed.add(extra)
        if cid == 'q139':
            routes = routes + [[extra]]
    return routes, allowed


def stage_routes(cid, turn, stage):
    amended = deepcopy(stage)
    for filename in ('source_route_amendments.json', 'variant_source_amendments.json'):
        routes = read(FROZEN / 'amendments' / filename)['routes'].get(cid, {})
        for field, alternatives in routes.items():
            extra = [[sid.format(turn=turn) for sid in route] for route in alternatives]
            existing = amended['evidence_routes'][field]
            amended['evidence_routes'][field] = existing + [r for r in extra if r not in existing]
            allowed = amended.setdefault('allowed_evidence', {}).setdefault(
                field, list(dict.fromkeys(s for r in existing for s in r)))
            for sid in (s for route in extra for s in route):
                if sid not in allowed:
                    allowed.append(sid)
    return amended


def quant_case(case):
    case = deepcopy(case)
    for field in case['outputs']:
        if field['name'] in COUNT_FIELDS.get(case['id'], set()):
            field.update(atol=0, rtol=0, integer_required=True)
    if case['id'] == 'q056':
        case['inputs']['permitted_shares'] = list(range(101))
    return case
