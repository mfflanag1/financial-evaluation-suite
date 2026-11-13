"""Derive selected results using public inputs and separate numerical methods.

These functions implement the reviewed synthetic contracts. The runner verifies
the complete public packet hashes before calling them. They are not general
pricing, statistical inference, or portfolio optimization APIs.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cmp_to_key
import itertools
import math


def fraction(value):
    return Fraction(str(value))


def q033(question):
    """Exhaust the cash grid as well as both bond grids, without cash completion."""
    inputs = question["inputs"]
    available, haircuts, rates, steps = (
        list(map(fraction, inputs[key]))
        for key in ("availability", "haircuts", "annual_cost_rates", "face_block_size")
    )
    menus = [range(math.floor(limit / step) + 1) for limit, step in zip(available, steps)]
    required = fraction(inputs["required_recognized"])
    corporate_cap = fraction(inputs["corporate_recognized_cap"])
    best = None
    feasible = 0
    for treasury, corporate, cash_blocks in itertools.product(*menus):
        faces = [treasury * steps[0], corporate * steps[1], cash_blocks * steps[2]]
        recognized = [face * (1 - haircut) for face, haircut in zip(faces, haircuts)]
        if recognized[1] > corporate_cap or sum(recognized) < required:
            continue
        feasible += 1
        cost = sum(face * rate for face, rate in zip(faces, rates))
        candidate = (cost, faces[2], treasury, corporate, sum(recognized))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No feasible collateral allocation in the supplied finite menu")
    cost, cash, treasury, corporate, recognized = best
    return {
        "bond_blocks": [treasury, corporate], "cash": float(cash),
        "annual_cost": float(cost), "recognized_value": float(recognized),
    }, {
        "complete_grid_points": math.prod(map(len, menus)),
        "feasible_grid_points": feasible,
        "recognized_surplus_exact": str(recognized - required),
        "cash_grid_blocks": int(cash / steps[2]),
        "objective_exact": str(cost),
        "tie_order": ["cost", "cash", "treasury_blocks", "corporate_blocks"],
    }


def q065(question):
    """Closed-form Bachelier and shifted Black prices, without quadrature."""
    inputs = question["inputs"]
    normal_cdf = lambda value: math.erfc(-value / math.sqrt(2)) / 2
    normal_pdf = lambda value: math.exp(-value * value / 2) / math.sqrt(2 * math.pi)
    forward, strike = inputs["forward"], inputs["strike"]
    sd = inputs["normal_volatility"] * math.sqrt(inputs["T"])
    scale = inputs["notional"] * inputs["accrual"] * inputs["discount_factor"]
    z = (forward - strike) / sd
    normal = scale * ((forward - strike) * normal_cdf(z) + sd * normal_pdf(z))
    shifted_forward, shifted_strike = forward + inputs["shift"], strike + inputs["shift"]
    log_sd = inputs["shifted_black_volatility"] * math.sqrt(inputs["T"])
    d1 = math.log(shifted_forward / shifted_strike) / log_sd + log_sd / 2
    shifted = scale * (
        shifted_forward * normal_cdf(d1) - shifted_strike * normal_cdf(d1 - log_sd)
    )
    probability = normal_cdf(z)
    return {
        "normal_price": normal, "shifted_black_price": shifted,
        "normal_forward_pv01": scale * probability / 10000,
        "unshifted_black_admissible": forward > 0 and strike > 0,
    }, {
        "normal_exercise_probability": probability,
        "shifted_forward_positive": shifted_forward > 0,
        "shifted_strike_positive": shifted_strike > 0,
        "method": "Analytic positive-part normal and lognormal expectations",
    }


def q098(question):
    """Count every complete fair-null binary path, without state aggregation."""
    inputs = question["inputs"]
    null, alternative, alpha = map(fraction, (
        inputs["null_success_probability"], inputs["alternative_success_probability"], inputs["alpha"],
    ))
    horizon = inputs["horizon_days"]
    if null != Fraction(1, 2) or horizon != 18 or len(inputs["observations"]) != horizon:
        raise ValueError("This exhaustive check covers the reviewed fair-null 18-day contract")
    boundary = 1 / alpha
    ratios = {
        (day, successes): (alternative / null) ** successes
        * ((1 - alternative) / (1 - null)) ** (day - successes)
        for day in range(1, horizon + 1) for successes in range(day + 1)
    }
    crossings = {state for state, ratio in ratios.items() if ratio >= boundary}
    path_count = 2 ** horizon
    first_crossing_count = 0
    for path in range(path_count):
        successes = 0
        for day in range(1, horizon + 1):
            successes += (path >> (day - 1)) & 1
            if (day, successes) in crossings:
                first_crossing_count += 1
                break
    terminal_count = sum(math.comb(horizon, k) for k in range(horizon + 1)
                         if (horizon, k) in crossings)
    successes = stop = naive_stop = 0
    ratio_at_stop = None
    for day, bit in enumerate(inputs["observations"], 1):
        successes += bit
        if not stop and (day, successes) in crossings:
            stop, ratio_at_stop = day, ratios[day, successes]
        upper_count = sum(math.comb(day, k) for k in range(successes, day + 1))
        if not naive_stop and Fraction(upper_count, 2 ** day) <= alpha:
            naive_stop = day
    if ratio_at_stop is None:
        raise ValueError("The reviewed path must reach the stated promotion boundary")
    probability = Fraction(first_crossing_count, path_count)
    return {
        "likelihood_stop_day": stop, "likelihood_at_stop": float(ratio_at_stop),
        "naive_fixed_test_stop_day": naive_stop,
        "finite_horizon_false_promotion": float(probability),
        "anytime_false_promotion_bound": float(alpha),
    }, {
        "complete_paths": path_count, "first_crossing_paths": first_crossing_count,
        "terminal_exceedance_paths": terminal_count,
        "first_crossing_probability_exact": str(probability),
        "likelihood_at_stop_exact": str(ratio_at_stop),
        "counting_measure": "All complete paths have probability 2^-18 under the supplied fair null",
    }


def c201(question):
    """Enumerate signed trades with separate buy, early-sale and late-sale ledgers."""
    inputs = question["inputs"]
    opening, prices = inputs["opening_holdings"], inputs["transaction_prices"]
    menus = [range(-holding, 7 - holding) for holding in opening]
    feasible = []
    for trades in itertools.product(*menus):
        holdings = tuple(holding + trade for holding, trade in zip(opening, trades))
        factor = 2 * holdings[0] + holdings[1] - holdings[2] - holdings[3]
        if sum(holdings) < 16 or abs(factor) > 2 or holdings[0] + holdings[2] < 6:
            continue
        fee = sum(Fraction(1) + Fraction(trade * trade, 4) for trade in trades if trade)
        buys = sum(max(trade, 0) * price for trade, price in zip(trades, prices))
        early_receipts = sum(max(-trades[j], 0) * prices[j] for j in (0, 2))
        late_receipts = sum(max(-trades[j], 0) * prices[j] for j in (1, 3))
        early = fraction(inputs["opening_cash"]) + early_receipts - buys - fee - inputs["redemption_amount"]
        final = early + late_receipts
        if early >= 0 and final >= 5:
            feasible.append((fee, holdings, trades, early, final, factor))
    if not feasible:
        raise ValueError("No executable repair in the supplied finite trade menu")
    cost, holdings, trades, early, final, factor = min(feasible)
    optima = [row for row in feasible if row[0] == cost]
    return {
        "cost": float(cost), "early_cash": float(early), "factor_exposure": factor,
        "final_cash": float(final), "final_holdings": list(holdings),
        "optimal_plan_count": len(optima), "trade_quantities": list(trades),
    }, {
        "complete_trade_vectors": math.prod(map(len, menus)),
        "feasible_trade_vectors": len(feasible),
        "all_optimal_holdings": [list(row[1]) for row in optima],
        "selected_cost_exact": str(cost),
        "source_contract_ids": ["p_book", "p_limits", "p_execution", "p_cash"],
        "scope": "Constants transcribed from the hash-bound current mandate; no secondary preference required",
    }


def _score(values):
    total = sum(values)
    squares = sum(value * value for value in values)
    if squares == 0:
        raise ValueError("The stated score is undefined for an all-zero checkpoint")
    return total, squares


def _decimal_score(score):
    total, squares = score
    with localcontext() as context:
        context.prec = 80
        numerator = Decimal(total.numerator) / Decimal(total.denominator)
        denominator = Decimal(squares.numerator) / Decimal(squares.denominator)
        return numerator / denominator.sqrt()


def _compare_scores(left, right):
    """Order signed square-root scores exactly, including arbitrarily close ties."""
    a, aa = left
    b, bb = right
    sign_a, sign_b = (a > 0) - (a < 0), (b > 0) - (b < 0)
    if sign_a != sign_b:
        return (sign_a > sign_b) - (sign_a < sign_b)
    if sign_a == 0:
        return 0
    cross = a * a * bb - b * b * aa
    return sign_a * ((cross > 0) - (cross < 0))


def c214(question):
    """Reconstruct residuals and all shared-sign outcomes using direct scores."""
    documents = {source["id"]: source["text"] for source in question["sources"]}
    blocks = []
    for line in documents["qt_tape"].splitlines():
        if not line.startswith("b"):
            continue
        parts = [part.strip() for part in line.split("|")]
        decision, publication, training, market = map(int, parts[1:5])
        if publication > decision or training + 2 > decision:
            continue
        gross = list(map(fraction, parts[5].split(",")))
        # Current public cost/beta contract, bound by the complete packet hash.
        residual = [gross[0] - 1 - market, gross[1] - Fraction(5, 2), gross[2] - 1 + market]
        blocks.append((parts[0], residual))
    blocks.sort(key=lambda block: int(block[0][1:]))
    if len(blocks) != 6:
        raise ValueError("The reviewed protocol requires six eligible blocks")
    candidates = [(_score([row[1][column] for row in blocks[:length]]), column, length)
                  for column in range(3) for length in (3, 6)]

    def order(left, right):
        difference = _compare_scores(left[0], right[0])
        if difference:
            return difference
        # Larger comparison result wins: alphabetically first, then longer.
        return ((-left[1], left[2]) > (-right[1], right[2])) - ((-left[1], left[2]) < (-right[1], right[2]))

    observed, selected_column, checkpoint = max(candidates, key=cmp_to_key(order))
    observed_decimal = _decimal_score(observed)
    raw_hits = family_hits = 0
    decimal_agreement = True
    for signs in itertools.product((-1, 1), repeat=6):
        signed = [[sign * value for value in row[1]] for sign, row in zip(signs, blocks)]
        scores = { (column, length): _score([row[column] for row in signed[:length]])
                   for column in range(3) for length in (3, 6) }
        exact = {key: _compare_scores(score, observed) for key, score in scores.items()}
        raw_hits += exact[selected_column, checkpoint] >= 0
        family_hits += any(comparison >= 0 for comparison in exact.values())
        for key, score in scores.items():
            displayed = _decimal_score(score)
            comparison = (displayed > observed_decimal) - (displayed < observed_decimal)
            decimal_agreement &= comparison == exact[key]
    early = sum(row[1][selected_column] for row in blocks[:3]) / 3
    late = sum(row[1][selected_column] for row in blocks[3:]) / 3
    adjusted = Fraction(family_hits, 64)
    deploy = adjusted <= Fraction(1, 20) and checkpoint == 6 and early > 0 and late > 0 and late >= early / 2
    return {
        "deploy": deploy, "eligible_blocks": [row[0] for row in blocks],
        "late_regime_mean": float(late), "selected_checkpoint": checkpoint,
        "selected_policy": "ABC"[selected_column], "selected_statistic": float(observed_decimal),
        "selection_adjusted_p": float(adjusted), "unadjusted_p": float(Fraction(raw_hits, 64)),
    }, {
        "joint_sign_vectors": 64, "raw_exceedances": raw_hits,
        "selection_exceedances": family_hits, "direct_score_decimal_digits": 80,
        "decimal_ordering_agrees_with_exact_cross_products": decimal_agreement,
        "early_regime_mean_exact": str(early), "late_regime_mean_exact": str(late),
        "source_contract_ids": ["qt_tape", "qt_cost", "qt_registry", "qt_gate"],
        "decision_comparison": "Exact signed cross products; decimals are display and agreement checks only",
    }


CHECKS = {"q033": q033, "q065": q065, "q098": q098, "c201": c201, "c214": c214}
