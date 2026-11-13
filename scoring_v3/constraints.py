"""Explicit semantic integer selectors; no inference from reference values."""
from copy import deepcopy
import math

from .common import DISCRETE_MAP, read, verify_dependencies

verify_dependencies()
_rows = read(DISCRETE_MAP)['fields']
APPROVED = {}
for _row in _rows:
    if _row['status'] == 'approved_integer':
        _key = (_row['case_id'], _row['turn'], _row['field'])
        if _key in APPROVED:
            raise ValueError('Duplicate approved integer field')
        _selectors = _row['integer_selectors']
        if not isinstance(_selectors, list) or not _selectors or any(
            not isinstance(s, list) or any(p != '*' and (type(p) is not int or p < 0) for p in s)
            for s in _selectors
        ):
            raise ValueError('Invalid approved integer selector')
        APPROVED[_key] = deepcopy(_selectors)

INTEGER_NOTE = (
    'Each integer_selectors entry identifies integer leaves: [] selects a scalar, '
    '"*" visits every array element, and a numeric index selects that mixed-array position. '
    'Selected values must be finite exact integers; JSON booleans are not numbers. '
    'Integral JSON numbers such as 2.0 are accepted without rounding. '
    'Selected integer values are compared exactly; numeric tolerances apply only to other leaves. '
    'An integer requirement does not relax stated bounds, allowed menus or multiples.'
)


def integral(value):
    # Python integers need no float conversion and have no artificial magnitude cap.
    return type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer())


def selectors(cid, turn, field):
    return deepcopy(APPROVED.get((cid, turn, field), []))


def _selected(value, path):
    if not path:
        return [value]
    if not isinstance(value, list):
        raise ValueError('Integer selector needs an array')
    head, *tail = path
    if head == '*':
        if not value:
            raise ValueError('Integer selector needs nonempty array')
        return [leaf for child in value for leaf in _selected(child, tail)]
    if head >= len(value):
        raise ValueError('Integer selector position absent')
    return _selected(value[head], tail)


def valid(cid, turn, field, value):
    try:
        return all(integral(leaf) for path in selectors(cid, turn, field) for leaf in _selected(value, path))
    except (ValueError, RecursionError):
        return False


def exact(cid, turn, field, value, expected):
    """Check only selected leaves; caller retains the rest of the field schema."""
    try:
        for path in selectors(cid, turn, field):
            actual, reference = _selected(value, path), _selected(expected, path)
            if len(actual) != len(reference) or any(not integral(a) or not integral(b) or a != b
                                                  for a, b in zip(actual, reference)):
                return False
        return True
    except (ValueError, RecursionError):
        return False


def _normalize(value, path):
    if not path:
        return int(value) if type(value) is float else value
    head, *tail = path
    if head == '*':
        return [_normalize(child, tail) for child in value]
    value[head] = _normalize(value[head], tail)
    return value


def normalize(cid, turn, field, value):
    result = deepcopy(value)
    if valid(cid, turn, field, result):
        for path in selectors(cid, turn, field):
            result = _normalize(result, path)
    return result


def public_contract(cid, turn=1):
    return {
        'integer_fields': {field: deepcopy(paths) for (case, stage, field), paths in APPROVED.items()
                           if case == cid and stage == turn},
        'integer_rule': INTEGER_NOTE,
    }


def annotate(question, cid, turn=1):
    """Add schema corrections to a copied public packet, without expected values."""
    result = deepcopy(question)
    contract = public_contract(cid, turn)
    result['scoring_corrections'] = contract
    outputs = result.get('outputs', result.get('output_fields'))
    if isinstance(outputs, list):
        for output in outputs:
            field = output.get('name', output.get('id'))
            if field in contract['integer_fields']:
                output['integer_selectors'] = deepcopy(contract['integer_fields'][field])
    elif isinstance(outputs, dict):
        for field, output in outputs.items():
            if field in contract['integer_fields']:
                output['integer_selectors'] = deepcopy(contract['integer_fields'][field])
    return result
