"""Public-contract derivations for synthetic research questions.

No scorer or stored answers are imported. Each function states the financial
operation independently and supplies a calculation certificate where useful.
Coverage is limited to the authored public stages, not arbitrary unseen paths.
"""
import itertools
import math
import statistics

from .workflow_numerical import number as F, plain, dot, spd_solve


def values(items):
    return list(map(F, items))


def growth(new, old):
    return (new / old - 1) * 100


def q107(i):
    prior = [F(a) + F(b) for a, b in zip(i["prior_reported"], i["prior_reclassification"])]
    organic = [F(a) - F(b) - F(c) for a, b, c in zip(i["current_reported"], i["acquired_current_revenue"], i["current_fx_translation_effect"])]
    rates = [growth(a, b) for a, b in zip(organic, prior)]
    aggregate = growth(sum(organic), sum(prior))
    reported = growth(sum(values(i["current_reported"])), sum(values(i["prior_reported"])))
    return {"comparable_prior_revenue": prior, "organic_constant_currency_revenue": organic,
            "segment_growth_pct": rates, "aggregate_organic_growth_pct": aggregate,
            "reported_growth_pct": reported, "growth_bridge_pp": reported - aggregate,
            "both_segments_clear_hurdle": all(r >= F(i["required_organic_growth_pct"]) for r in rates)}, {}


def q108(i):
    rows = [dict(zip(i["columns"], row)) for row in i["rows"]]
    snapshots = []
    for cutoff in i["vintages"]:
        latest = {}
        for row in rows:
            if row["published"] <= cutoff and row["available"] <= cutoff and row["fiscal_year"] == i["target_year"]:
                broker = row["broker_group"]
                if broker not in latest or (row["published"], row["record"]) > (latest[broker]["published"], latest[broker]["record"]):
                    latest[broker] = row
        snapshots.append({broker: row for broker, row in latest.items() if row["status"] == "active"})

    def eps(row, fx):
        return F(row["eps"]) / (i["new_shares_per_old"] if row["share_basis"] == "old" else 1) * (F(fx) if row["currency"] == "EUR" else 1)

    a, b = snapshots
    matched = sorted(a.keys() & b.keys())
    earlier = statistics.mean(eps(row, i["fixed_usd_per_eur"]) for row in a.values())
    later = statistics.mean(eps(row, i["fixed_usd_per_eur"]) for row in b.values())
    first = statistics.mean(eps(a[g], i["fixed_usd_per_eur"]) for g in matched)
    second = statistics.mean(eps(b[g], i["fixed_usd_per_eur"]) for g in matched)
    spot = statistics.mean(eps(row, i["latest_spot_usd_per_eur"]) for row in b.values())
    return {"latest_record_ids": [b[g]["record"] for g in sorted(b)], "matched_broker_groups": matched,
            "constant_currency_vintage_means": [earlier, later], "matched_cohort_revision_pct": growth(second, first),
            "headline_revision_pct": growth(spot, earlier), "mean_change_bridge": [first - earlier, second - first, later - second, spot - later],
            "constant_currency_surprise_pct": growth(F(i["actual_eps_usd_new"]), later),
            "every_matched_broker_raised_eps": all(eps(b[g], i["fixed_usd_per_eur"]) > eps(a[g], i["fixed_usd_per_eur"]) for g in matched)}, {"bridge_total": spot - earlier}


def q109(i):
    eligible = sorted((r for r in i["listings"] if r["primary"] and r["start"] <= i["as_of"] < r["end"]), key=lambda r: r["group"])
    margins, leverage, passing, multiples = [], [], [], []
    for row in eligible:
        n = i["groups"].index(row["group"])
        net = F(i["debt"][n]) - F(i["cash"][n]) + F(i["restricted_cash"][n])
        margin = (F(i["cfo"][n]) - F(i["capex"][n])) / F(i["revenue"][n]) * 100
        ratio = net / F(i["ebitda"][n])
        margins.append(margin)
        leverage.append(ratio)
        if margin >= F(i["minimum_fcf_margin_pct"]) and ratio <= F(i["maximum_net_debt_ebitda"]) and i["free_float_cap"][n] >= i["minimum_free_float_cap"]:
            passing.append(row["id"])
            multiples.append((F(i["market_cap"][n]) + net) / F(i["ebitda"][n]))
    selected = min(range(len(passing)), key=lambda n: (multiples[n], passing[n]))
    return {"eligible_primary_ids": [row["id"] for row in eligible], "fcf_margin_pct": margins,
            "net_debt_ebitda": leverage, "passing_ids": passing, "passing_ev_ebitda": multiples, "selected_id": passing[selected]}, {}


def q110(i):
    reported = F(i["filed_ytd_cfo"]) - F(i["filed_ppe_capex"]) - F(i["filed_software_capex"])
    normal = reported - F(i["factoring_cash_current"]) - F(i["customer_advance_current"])
    prior = F(i["prior_ytd_cfo"]) - F(i["prior_ppe_capex"]) - F(i["prior_software_capex"])
    cash = F(i["cash_balance"]) - F(i["restricted_cash"])
    return {"authoritative_basis": "filed_schedule", "reported_fcf": reported, "normalized_fcf": normal,
            "normalized_growth_pct": growth(normal, prior), "omitted_software_capex": i["filed_software_capex"],
            "usable_cash": cash, "immediate_funding_gap": max(0, F(i["immediate_obligation"]) - cash)}, {}


def q111(i):
    design = [[1] + row for row in i["pre_event_factor_rows"]]
    columns = list(zip(*design))
    gram = [[dot(a, b) for b in columns] for a in columns]
    coefficients = [spd_solve(gram, [dot(column, i[key]) for column in columns]) for key in ("issuer_pre_returns", "control_pre_returns")]
    abnormal = [i[key] - dot([1] + i["event_factors"], beta) for key, beta in zip(("issuer_event_return", "control_event_return"), coefficients)]
    return {"issuer_coefficients": coefficients[0], "control_coefficients": coefficients[1],
            "event_abnormal_returns": abnormal, "matched_difference": abnormal[0] - abnormal[1],
            "issuer_abnormal_value": abnormal[0] * i["issuer_pre_event_equity_value"],
            "earnings_contribution_bounds": [0, abnormal[0]], "positive_earnings_effect_identified": False}, {"zero_earnings_attribution_feasible": abnormal[0] >= 0}


def q112(i):
    a = [F(x) - F(y) + F(z) for x, y, z in zip(i["a_prior_fy"], i["a_prior_nine_months"], i["a_current_nine_months"])]
    b = [F(x) - F(y) + F(z) for x, y, z in zip(i["b_fy_june"], i["b_prior_september"], i["b_current_september"])]
    normalized = [row[1] + row[2] for row in (a, b)]
    multiples = [F(i["enterprise_values_usd"][0]) / normalized[0], F(i["enterprise_values_usd"][1]) / (normalized[1] * F(i["usd_per_eur"]))]
    return {"ttm_revenue_local": [a[0], b[0]], "ttm_reported_ebit_local": [a[1], b[1]],
            "ttm_normalized_ebit_local": normalized, "normalized_margin_pct": [n / row[0] * 100 for n, row in zip(normalized, (a, b))],
            "ev_normalized_ebit": multiples, "cheaper_peer": "A" if multiples[0] <= multiples[1] else "B"}, {"b_quarter_revenue_residual": b[0] - sum(values(i["b_dec_mar_jun_revenue"])) - F(i["b_current_september"][0])}


def q113(i):
    capacity = [F(i["a_wafers"]) * F(i["a_yield"]) * F(i["a_modules_per_good_wafer"]), F(i["b_nominal_capacity"]) * (1 - F(i["b_unavailable_fraction"]))]
    def optimize(a, b):
        choices = []
        for x in range(i["demand"][0] + 1):
            for y in range(i["minimum_y"], i["demand"][1] + 1):
                if dot((x, y), i["module_a_per_product"]) <= a and dot((x, y), i["module_b_per_product"]) <= b:
                    choices.append((dot((x, y), i["gross_profit_per_unit"]), x, y))
        return min(choices, key=lambda row: (-row[0], row[1], row[2]))
    base = optimize(*capacity)
    extra_b = optimize(capacity[0], capacity[1] + F(i["expedite_b_modules"]))
    extra_a = optimize(capacity[0] + F(i["alternative_a_modules"]), capacity[1])
    net_b = extra_b[0] - base[0] - F(i["expedite_b_fee"])
    return {"base_units": list(base[1:]), "base_gross_profit": base[0],
            "base_module_usage": [dot(base[1:], i[k]) for k in ("module_a_per_product", "module_b_per_product")],
            "unserved_demand": [d - q for d, q in zip(i["demand"], base[1:])],
            "expedited_b_units": list(extra_b[1:]), "net_b_increment": net_b,
            "prefer_b_over_free_a": net_b > extra_a[0] - base[0]}, {"free_a_optimum": extra_a, "base_capacities": capacity}


def q114(i):
    base_channel = F(i["channel_begin"]) + F(i["channel_shipped"]) - F(i["channel_end_reported"]) - F(i["channel_returns"])
    states = [(base_channel + k) * F(i["channel_price"]) + (F(i["direct_accepted_reported"]) - t) * F(i["direct_price"]) - F(i["refund_allowance"])
              for k in range(i["returned_units_in_closing_range"][0], i["returned_units_in_closing_range"][1] + 1)
              for t in range(i["disputed_acceptances_range"][0], i["disputed_acceptances_range"][1] + 1)]
    bounds = [min(states), max(states)]
    historical = F(i["channel_receipts"]) + F(i["direct_receipts"])
    feasible = []
    prices = values([i["channel_price"], i["direct_price"]])
    for c, d in itertools.product(range(i["incremental_channel_cap"] + 1), range(i["incremental_direct_cap"] + 1)):
        outlay = dot((c, d), values(i["incremental_unit_costs"]))
        receipts = sum(q * p * f for q, p, f in zip((c, d), prices, values(i["incremental_upfront_fractions"])))
        revenue = bounds[0] + dot((c, d), prices)
        if revenue >= F(i["revenue_floor"]) and outlay <= F(i["opening_unrestricted_cash"]) + receipts:
            feasible.append((outlay, c + d, c, d, receipts, revenue))
    best = min(feasible)
    return {"recognized_revenue_bounds": bounds, "cash_received": historical,
            "net_contract_asset_bounds": [v - historical for v in bounds],
            "floor_guaranteed_without_new_orders": bounds[0] >= F(i["revenue_floor"]),
            "minimum_cost_order_units": list(best[2:4]), "incremental_production_cash": best[0],
            "incremental_receipts": best[4], "guaranteed_post_order_revenue": best[5],
            "funded_qualifying_plan_count": len(feasible)}, {"admissible_recognition_states": len(states)}


def q115(i):
    tonnes = [sum(F(row["quantity"]) / (1000 if row["unit"] == "kg" else 1) for row in i["production_records"] if row["mine"] == mine and row["scope"] == "production") for mine in i["mine_order"]]
    attributed = [[q * o for q, o in zip(tonnes, values(row))] for row in i["ownership"]]
    gross = [dot(row, [F(i["price_per_tonne"]) - F(c) for c in i["cost_per_tonne"]]) for row in attributed]
    emissions = [dot(row, values(i["emissions_per_tonne"])) for row in attributed]
    after = [profit - e * F(i["carbon_price"]) for profit, e in zip(gross, emissions)]
    yields = [p / F(v) * 100 for p, v in zip(after, i["equity_values"])]
    best = min(range(len(yields)), key=lambda n: (-yields[n], i["ownership_order"][n]))
    return {"attributed_production_tonnes": [sum(row) for row in attributed], "gross_operating_profit": gross,
            "attributed_emissions_tonnes": emissions, "after_carbon_profit": after, "after_carbon_yield_pct": yields,
            "unique_physical_production": sum(tonnes), "higher_yield_company": i["ownership_order"][best]}, {}


def q116(i):
    revenue, cost, future = values([i["base_revenue"], i["base_cost"], i["next_revenue"]])
    fixed = values(i["fixed_cost_bounds"])
    rates = [(cost - f) / revenue for f in fixed]
    ebit = [future - f - v * future for f, v in zip(fixed, rates)]
    cash = [e * (1 - F(i["tax_rate"])) - F(i["maintenance_cash"]) for e in ebit]
    threshold = ((F(i["cash_hurdle"]) + F(i["maintenance_cash"])) / (1 - F(i["tax_rate"])) - future * (1 - cost / revenue)) / (future / revenue - 1)
    label = "certainly_met" if min(cash) >= F(i["cash_hurdle"]) else "certainly_missed" if max(cash) < F(i["cash_hurdle"]) else "not_identified"
    return {"variable_cost_rate_bounds": sorted(rates), "next_ebit_bounds": sorted(ebit), "next_cash_bounds": sorted(cash),
            "minimum_fixed_cost_for_hurdle": max(min(fixed), threshold),
            "analyst_forecast_feasible": min(ebit) <= F(i["analyst_ebit_forecast"]) <= max(ebit),
            "hurdle_identification": label, "endpoint_base_costs": [f + v * revenue for f, v in zip(fixed, rates)]}, {"cash_affine_slope": (future / revenue - 1) * (1 - F(i["tax_rate"]))}


def q117(i):
    row = max((r for r in i["reports"] if r["entity"] == "Borrower" and r["period"] == "Y1" and r["published"] <= i["cutoff"]), key=lambda r: r["published"])
    ebitda = F(row["ebitda"]) + min(F(row["integration"]), F(i["addback_cap"]) * F(row["ebitda"]))
    cash = sum(F(r["cash"]) for r in i["cash"] if r["entity"] == "Borrower" and not r["restricted"])
    nettable = min(cash, F(i["cash_cap"]))
    debt = sum(F(r["amount"]) for r in i["debts"] if r["included"])
    capacity = max(0, min(F(i["facility_undrawn"]), F(i["leverage_limit"]) * ebitda - debt + nettable))
    post = cash + F(i["requested_draw"]) - F(i["draw_fee"])
    permitted = F(i["requested_draw"]) <= capacity and post >= F(i["minimum_cash"])
    return {"covenant_ebitda": ebitda, "nettable_cash": nettable, "opening_net_leverage": (debt - nettable) / ebitda,
            "maximum_draw": capacity, "post_draw_cash": post, "maximum_dividend": max(0, post - F(i["minimum_cash"])) if permitted else 0,
            "draw_permitted": permitted}, {}


def q118(i):
    valuations = [max((r for r in i["valuations"] if r["estate"] == estate and r["published"] <= i["cutoff"]), key=lambda r: r["published"])["value"] for estate in ("A", "B")]
    net = [F(v) * (1 - F(h)) - F(c) for v, h, c in zip(valuations, i["haircuts"], i["costs"])]
    secured = [min(F(debt), max(0, cash)) for debt, cash in zip(i["secured"], net)]
    junior = min(F(i["junior_A"]), max(0, net[0] - secured[0]))
    upstream = max(0, net[0] - secured[0] - junior) + max(0, net[1] - secured[1])
    holding = min(F(i["holding_debt"]), F(i["holding_cash"]) + upstream)
    loss = sum(values(i["secured"])) + F(i["junior_A"]) + F(i["holding_debt"]) - sum(secured) - junior - holding
    return {"estate_net_proceeds": net, "secured_recoveries": secured, "junior_a_recovery": junior,
            "holding_recovery": holding, "creditor_loss": loss, "junior_a_recovery_pct": junior / F(i["junior_A"]) * 100,
            "cross_estate_netting_permitted": False}, {"equity_upstream": upstream}


def q119(i):
    rates = [F(max((r for r in i["quotes"] if r["currency"] == "USD" and r["year"] == year and r["published"] <= i["cutoff"]), key=lambda r: r["published"])["par_rate"]) for year in range(1, len(i["liability"]) + 1)]
    def bootstrap(rates):
        discounts = []
        for rate in rates:
            discounts.append((1 - rate * sum(discounts)) / (1 + rate))
        return discounts
    discounts = bootstrap(rates)
    pv = dot(discounts, values(i["liability"]))
    shifted = dot(bootstrap([rate + F(i["shock"]) for rate in rates]), values(i["liability"]))
    gap = max(0, pv - F(i["cash"]))
    return {"eligible_par_rates": rates, "discount_factors": discounts, "liability_pv": pv,
            "year_two_to_three_forward": discounts[1] / discounts[2] - 1, "parallel_quote_shock_pv_change": shifted - pv,
            "funding_gap": gap, "prefunding_permitted": gap <= F(i["funding_limit"])}, {"par_pricing_residuals": [rate * sum(discounts[:year]) + discounts[year - 1] - 1 for year, rate in enumerate(rates, 1)]}


def q120(i):
    free = F(i["eur_cash"]) - F(i["trapped_eur"])
    invoice = F(i["invoice_eur"]) * F(i["fixing"])
    hedge = F(i["ndf_eur"]) * (F(i["fixing"]) - F(i["strike"]))
    loss = F(i["invoice_eur"]) * (F(i["fixing"]) - F(i["recognition_fx"]))
    gap = max(0, (F(i["invoice_eur"]) - free) * F(i["fixing"]) - F(i["usd_cash"]))
    return {"usable_eur_cash": free, "invoice_usd": invoice, "ndf_usd_receipt": hedge,
            "net_invoice_hedge_fx_pnl": hedge - loss, "peak_funding_gap": gap,
            "gap_after_ndf_payment": max(0, gap - hedge), "facility_covers_payment": gap <= F(i["facility"])}, {"hedge_settles_after_invoice": i["ndf_cash_day"] > i["invoice_day"]}


def q121(i):
    trigger = F(i["conversion_price"]) * F(i["trigger_multiple"])
    count = sum(F(price) >= trigger for price in i["eligible_closes"])
    put = F(i["principal"]) * F(i["put_fraction"]) * (1 + F(i["coupon"]) * F(i["accrual_years"]))
    extra = max(0, put - F(i["gross_cash"]) + F(i["restricted_cash"]) - F(i["facility"]) + F(i["facility_fee"]))
    return {"qualifying_close_count": count, "call_trigger_price": trigger,
            "forced_conversion_available": count >= i["required_count"], "put_cash_required": put,
            "remaining_conversion_shares": F(i["principal"]) * (1 - F(i["put_fraction"])) / F(i["conversion_price"]),
            "additional_cash_needed": extra, "funding_action": "additional_cash" if extra > 0 else "existing_facility"}, {}


def q122(i):
    cash = F(i["headline_cash"]) - F(i["special_dividend"])
    success = cash + F(i["ratio"]) * F(i["acquirer_price"]) + F(i["special_dividend"])
    carry = F(i["target_price"]) * F(i["annual_funding"]) * F(i["days"]) / 360 + F(i["fee"])
    cost = F(i["target_price"]) + carry
    failure = F(i["break_price"])
    met = all(i["other_conditions"]) and (i["days"] <= i["financing_expiry"] or i["extension_written"])
    return {"closing_cash_component": cash, "total_success_proceeds": success, "carry_and_fee": carry,
            "break_even_success_probability": (cost - failure) / (success - failure),
            "conditional_expected_pnl_range": [F(p) * success + (1 - F(p)) * failure - cost for p in i["probability_bounds"]],
            "all_closing_conditions_met": met, "research_conclusion": "conditions_satisfied" if met else "conditional_only"}, {}


def q123(i):
    def leaf(q0, q1, index):
        beginning = F(i["opening_inventory"]) + q0
        stored = beginning - F(i["initial_demand"]) + q1
        cash = q0 * F(i["initial_price"]) + q1 * F(i["signal_prices"][i["leaf_signal"][index]])
        available = max(0, stored - F(i["terminal_reserve"]))
        delivered = min(F(i["leaf_demand"][index]), F(i["terminal_withdrawal_cap"]), available)
        shortfall = F(i["leaf_demand"][index]) - delivered
        if beginning > i["storage_cap"] or beginning < i["initial_demand"] or stored > i["storage_cap"] or stored < i["terminal_reserve"] or cash > i["purchase_cash_budget"] or shortfall > i["maximum_leaf_shortfall"]:
            return None
        return cash + shortfall * F(i["shortfall_penalty"]), shortfall, stored - delivered, cash
    candidates = []
    for policy in itertools.product(i["grid"], repeat=3):
        records = [leaf(policy[0], policy[1 + i["leaf_signal"][n]], n) for n in range(len(i["leaf_demand"]))]
        if all(record is not None for record in records):
            cost = dot([r[0] for r in records], values(i["leaf_probability"]))
            candidates.append((cost, policy, records))
    cost, policy, records = min(candidates, key=lambda row: (row[0], row[1]))
    early = []
    for q0 in i["grid"]:
        by_leaf = [[r for q1 in i["grid"] if (r := leaf(q0, q1, n)) is not None] for n in range(len(i["leaf_demand"]))]
        if all(by_leaf):
            early.append(dot([min(r[0] for r in choices) for choices in by_leaf], values(i["leaf_probability"])))
    return {"nonanticipative_purchase_policy": policy, "expected_total_cost": cost,
            "leaf_shortfalls": [r[1] for r in records], "leaf_ending_inventory": [r[2] for r in records],
            "leaf_purchase_cash": [r[3] for r in records], "feasible_policy_count": len(candidates),
            "early_demand_information_cost": min(early), "value_of_early_demand_information": cost - min(early)}, {"enumerated_nonanticipative_policies": len(i["grid"]) ** 3}


def q124(i):
    variable = F(i["heat_rate"]) * F(i["gas_price"]) + F(i["emission_rate"]) * F(i["carbon_price"]) + F(i["variable_om"])
    candidates = []
    for dispatch in itertools.product(i["grid"], repeat=len(i["power_prices"])):
        boundary = (0,) + dispatch + (0,)
        if any(abs(a - b) > i["ramp"] for a, b in zip(boundary, boundary[1:])):
            continue
        lengths = [len(list(group)) for active, group in itertools.groupby(dispatch, key=lambda value: value > 0) if active]
        if any(length < i["minimum_run"] for length in lengths) or sum(dispatch) * F(i["heat_rate"]) > i["fuel_cap"]:
            continue
        starts = len(lengths)
        margin = dot(dispatch, [F(price) - variable for price in i["power_prices"]]) - starts * F(i["startup_cost"])
        candidates.append((margin, sum(dispatch), dispatch, starts))
    margin, total, dispatch, starts = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
    return {"dispatch": dispatch, "variable_cost_per_mwh": variable, "fuel_used": total * F(i["heat_rate"]),
            "emissions": total * F(i["emission_rate"]), "start_count": starts, "net_cash_margin": margin,
            "dispatch_authorized": margin >= F(i["minimum_margin"])}, {"feasible_dispatch_count": len(candidates)}


def q125(i):
    opening, closing, duration, convexity, rate, spread, fx0, fx1, coupon, funding = values([i[k] for k in ("opening_value", "closing_value", "duration", "convexity", "rate_change", "spread_change", "opening_fx", "closing_fx", "coupon", "funding_usd")])
    first_rate = -opening * duration * rate
    first_spread = -opening * duration * spread
    second = opening * convexity * (rate + spread) ** 2 / 2
    model = first_rate + first_spread + second
    bridge = [first_rate * fx1, first_spread * fx1, second * fx1, (closing - opening - model) * fx1, opening * (fx1 - fx0), coupon * fx1 - funding]
    pnl = sum(bridge)
    return {"aligned_security": i["security"], "model_local_price_change": model, "base_currency_attribution": bridge,
            "net_base_pnl": pnl, "net_base_return": pnl / (opening * fx0), "news_causality_identified": False,
            "loss_limit_breached": -pnl > F(i["loss_limit"])}, {"cash_wealth_reconciliation": pnl - ((closing + coupon) * fx1 - opening * fx0 - funding)}


def q126(i):
    payoffs = [0 if max(path) >= i["barrier"] else max(0, path[-1] - i["strike"]) for path in i["witness_paths"]]
    pnl = [F(payoff) - F(i["premium"]) - F(i["delta"]) * (F(path[-1]) - F(path[0])) for payoff, path in zip(payoffs, i["witness_paths"])]
    worst = max(0, -min(pnl))
    return {"witness_payoffs": payoffs, "delta_hedged_pnls": pnl,
            "both_paths_match_observed_endpoints": all(path[0] == i["initial_spot"] and path[-1] == i["terminal_spot"] for path in i["witness_paths"]),
            "unique_payoff_identified": len(set(payoffs)) == 1, "worst_witness_loss": worst,
            "loss_budget_guaranteed_by_endpoints": worst <= F(i["loss_budget"])}, {}


def q127(i):
    feasible = []
    for long, short in itertools.product(*i["lot_grids"]):
        lots = (long, short)
        costs = dot(lots, values(i["spread_per_lot"])) + dot([q*q for q in lots], values(i["impact_coefficients"])) + short * F(i["short_borrow_per_lot"]) + sum(q > 0 for q in lots) * F(i["ticket_fee"])
        cash = long * F(i["long_lot_shares"]) * F(i["mid_prices"][0]) + short * F(i["short_initial_margin_per_lot"]) + costs
        delta = dot(lots, values(i["delta_per_lot"]))
        if cash <= F(i["cash"]) and short * i["short_lot_shares"] <= i["short_locate_shares"] and abs(delta) <= F(i["absolute_delta_limit"]):
            feasible.append((dot(lots, values(i["alpha_per_lot"])) - costs, cash, long, short, delta))
    pnl, cash, long, short, delta = min(feasible, key=lambda r: (-r[0], r[1], r[2], r[3]))
    return {"lots": [long, short], "expected_net_pnl": pnl, "initial_cash_used": cash,
            "residual_delta": delta, "feasible_plan_count": len(feasible), "remaining_free_cash": F(i["cash"]) - cash}, {}


def q128(i):
    names = sorted(set(i["issuer_aliases"].values()))
    loadings = []
    for manager in ("manager_A", "manager_B"):
        exposure = {name: F(0) for name in names}
        for security, holding in i[manager].items():
            if security in i["issuer_aliases"]:
                exposure[i["issuer_aliases"][security]] += F(holding)
            else:
                for underlying, weight in i[security].items():
                    exposure[i["issuer_aliases"][underlying]] += F(holding) * F(weight)
        loadings.append(exposure)
    capital = values(i["capital_millions"])
    exposure = [sum(allocation * loading[name] for allocation, loading in zip(capital, loadings)) for name in names]
    excess = max(0, exposure[names.index("X")] - F(i["issuer_X_limit_millions"]))
    reductions = [F(0), F(0)]
    order = sorted(range(2), key=lambda n: F(i["alpha_rates"][n]) / loadings[n]["X"])
    for n in order:
        reductions[n] = min(capital[n] - F(i["minimum_manager_allocations"][n]), excess / loadings[n]["X"])
        excess -= reductions[n] * loadings[n]["X"]
    if excess > 0:
        raise ValueError("Issuer repair infeasible under minimum allocations")
    post = [c - r for c, r in zip(capital, reductions)]
    return {"issuer_exposures": exposure, "gross_underlying_exposure": sum(map(abs, exposure)),
            "stress_pnl": dot(exposure, [F(i["stress_returns"][name]) for name in names]),
            "post_reallocation": post + [sum(reductions)], "annual_alpha_foregone": dot(reductions, values(i["alpha_rates"])),
            "post_x_exposure": sum(c * loading["X"] for c, loading in zip(post, loadings))}, {"alpha_cost_per_unit_x_reduced": [F(i["alpha_rates"][n]) / loadings[n]["X"] for n in range(2)]}


def q129(i):
    units, requests = values(i["series_units"]), values(i["requested_redemption_units"])
    management = [F(nav) * F(i["management_rate"]) * F(year) for nav, year in zip(i["entry_nav"], i["year_fractions"])]
    feasible = []
    for a, b in itertools.product(range(i["sale_caps"][0] + 1), range(i["sale_caps"][1] + 1)):
        sale = (a, b)
        proceeds = dot(sale, values(i["asset_sale_prices"]))
        loss = dot(sale, [F(mark) - F(price) for mark, price in zip(i["asset_marks"], i["asset_sale_prices"])])
        gross = F(i["gross_nav_before_sales"]) - loss / sum(units)
        net = [gross - management[n] - F(i["incentive_rate"]) * max(0, gross - management[n] - F(i["old_high_water_marks"][n])) for n in range(len(units))]
        fees = sum(q * (gross - nav) for q, nav in zip(units, net))
        for gate in values(i["gate_grid"]):
            redemption = gate * dot(requests, net)
            cash = F(i["opening_cash"]) + proceeds - fees - redemption
            if cash >= F(i["minimum_cash"]):
                feasible.append((gate, loss, a + b, a, b, gross, net, fees, redemption, cash))
    gate, loss, _, a, b, gross, net, fees, redemption, cash = min(feasible, key=lambda r: (-r[0], r[1], r[2], r[3], r[4]))
    remaining = sum((q - gate * request) * nav for q, request, nav in zip(units, requests, net))
    return {"maximum_common_gate": gate, "asset_sale_units": [a, b], "realized_sale_loss": loss,
            "post_sale_gross_nav_per_unit": gross, "post_fee_series_nav": net, "fees_paid_cash": fees,
            "redemption_cash_paid": redemption, "ending_cash": cash, "remaining_investor_nav": remaining,
            "new_high_water_marks": [max(F(old), nav) for old, nav in zip(i["old_high_water_marks"], net)],
            "full_requested_redemptions_fundable": any(r[0] == 1 for r in feasible)}, {"nav_identity_residual": F(i["gross_nav_before_sales"]) * sum(units) - loss - fees - redemption - remaining, "feasible_sale_gate_pairs": len(feasible)}


def q130(i):
    returns, components = [], []
    for row in i["periods"]:
        pw, bw, pr, br = [values(row[k]) for k in ("portfolio_weights", "benchmark_weights", "portfolio_sector_returns", "benchmark_sector_returns")]
        benchmark = dot(bw, br)
        portfolio = dot(pw, pr) - F(row["fund_fee"])
        difference = [p - b for p, b in zip(pw, bw)]
        selection = [p - b for p, b in zip(pr, br)]
        components.append([dot(difference, [r - benchmark for r in br]), dot(bw, selection), dot(difference, selection), -F(row["fund_fee"])])
        returns.append((portfolio, benchmark))
    linked = [a * (1 + returns[1][0]) + b * (1 + returns[0][1]) for a, b in zip(*components)]
    pc = math.prod(1 + r[0] for r in returns) - 1
    bc = math.prod(1 + r[1] for r in returns) - 1
    return {"period_net_and_benchmark_returns": [value for row in returns for value in row], "linked_contributions": linked,
            "portfolio_cumulative_return": pc, "benchmark_cumulative_return": bc, "cumulative_active_return": pc - bc,
            "unlinked_active_sum": sum(p - b for p, b in returns)}, {"linked_residual": sum(linked) - pc + bc}


def q131(i):
    feasible = []
    for deployment in itertools.product(i["deployment_grid"], repeat=3):
        flows = [dot(row, deployment) for row in i["asset_flow_loadings"]]
        risk = dot(deployment, i["risk_units"])
        if sum(deployment) > i["capital_limit"] or risk > i["risk_unit_limit"] or any(x > cap for x, cap in zip(flows, i["asset_flow_caps"])):
            continue
        impact = [F(coefficient) * flow * flow for coefficient, flow in zip(i["impact_coefficients"], flows)]
        alpha = dot(deployment, values(i["gross_alpha_per_unit"])) - sum(impact) - sum(x > 0 for x in deployment) * F(i["active_strategy_fee"])
        feasible.append((alpha, sum(deployment), deployment, flows, impact, risk))
    alpha, capital, deployment, flows, impact, risk = min(feasible, key=lambda r: (-r[0], r[1], r[2]))
    return {"deployment": deployment, "net_expected_alpha": alpha, "asset_flows": flows,
            "impact_costs": impact, "risk_units_used": risk, "unused_capital": i["capital_limit"] - capital}, {"feasible_deployments": len(feasible)}


def q132(i):
    rows = [dict(zip(i["columns"], row)) for row in i["trials"]]
    registered = [r for r in rows if r["registered_before_holdout"]]
    families = sorted({r["family"] for r in registered})
    representatives = [min((r for r in registered if r["family"] == family), key=lambda r: (r["validation_loss"], r["id"])) for family in families]
    ordered = sorted(representatives, key=lambda r: (r["confirmatory_p"], r["id"]))
    count = max([0] + [n for n, r in enumerate(ordered, 1) if F(r["confirmatory_p"]) <= F(i["FDR_level"]) * n / len(ordered)])
    confirmed = ordered[:count]
    subsets = []
    for flags in itertools.product((False, True), repeat=count):
        chosen = [r for r, keep in zip(confirmed, flags) if keep]
        capacity = sum(r["launch_capacity"] for r in chosen)
        if capacity <= i["launch_capacity_limit"]:
            subsets.append((sum(F(r["forecast_net_alpha"]) for r in chosen), capacity, sorted(r["id"] for r in chosen)))
    alpha, _, selected = min(subsets, key=lambda r: (-r[0], r[1], r[2]))
    return {"family_representatives": [r["id"] for r in representatives], "confirmed_ids": sorted(r["id"] for r in confirmed),
            "funded_ids": selected, "bh_cutoff": ordered[count - 1]["confirmatory_p"] if count else 0,
            "funded_forecast_net_alpha": alpha, "excluded_ids": sorted(r["id"] for r in rows if not r["registered_before_holdout"])}, {"confirmatory_family_count": len(representatives)}


def q133(i):
    observed = values(i["observed_monthly_returns"])
    means = [(sum(observed) + sum(values(row))) / i["total_months"] - F(i["candidate_monthly_fee"]) for row in i["permitted_joint_completions"]]
    comparator = F(i["comparator_monthly_net_return"])
    choices = [(min(w * m + (1 - w) * comparator for m in means), w) for w in values(i["candidate_weight_grid"])]
    worst, weight = min(choices, key=lambda r: (-r[0], r[1]))
    label = "identified_better" if min(means) > comparator else "identified_worse" if max(means) < comparator else "not_identified"
    return {"complete_case_gross_mean": statistics.mean(observed), "net_mean_by_completion": means,
            "candidate_net_mean_bounds": [min(means), max(means)], "dominance_status": label,
            "robust_candidate_weight": weight, "robust_worst_net_mean": worst}, {}


def q134(i):
    initial = math.ceil(F(i["account1_requirement"]) / F(i["A_value_T0"]))
    stressed = math.ceil(F(i["account1_requirement"]) / F(i["A_value_T1"]))
    feasible = []
    for a, b, ra, rb in itertools.product(range(i["inventory_units"]["A"] + 1), range(i["inventory_units"]["B"] + 1), range(i["inventory_units"]["A"] + 1), range(i["inventory_units"]["B"] + 1)):
        if stressed + a + ra > i["inventory_units"]["A"] or b + rb > i["inventory_units"]["B"]:
            continue
        if any(a * F(i[key]) + b * F(i["B_value_both"]) < F(i["account2_requirement"]) for key in ("A_value_T0", "A_value_T1")):
            continue
        principal = ra * F(i["repo_cash_per_unit"]["A"]) + rb * F(i["repo_cash_per_unit"]["B"])
        repayment = ra * F(i["repo_repayment_per_unit"]["A"]) + rb * F(i["repo_repayment_per_unit"]["B"])
        cash0 = F(i["free_cash"]) + principal
        cash1 = cash0 - F(i["cash_payment_T1"])
        cash2 = cash1 + F(i["cash_receipt_T2"]) - repayment
        if min(cash0, cash1, cash2) >= 0:
            feasible.append((repayment - principal, ra + rb, a + b, (a, b, ra, rb), [cash0, cash1, cash2]))
    cost, _, _, plan, cash = min(feasible)
    a, b, ra, rb = plan
    return {"t0_collateral_units": [initial, a, b], "t1_collateral_units": [stressed, a, b], "repo_units": [ra, rb],
            "cash_balances": cash, "financing_cost": cost,
            "remaining_unencumbered_units": [i["inventory_units"]["A"] - stressed - a - ra, i["inventory_units"]["B"] - b - rb]}, {"feasible_plans": len(feasible)}


def q135(i):
    mse, bias, pooled = [], [], []
    for n in range(len(i["model_order"])):
        groups = [[F(row["realized"]) - F(row["forecasts"][n]) for row in i["validation_rows"] if row["regime"] == regime] for regime in i["regime_order"]]
        mse.append(dot([statistics.mean(e*e for e in group) for group in groups], values(i["target_regime_weights"])))
        bias.append(dot([statistics.mean(group) for group in groups], values(i["target_regime_weights"])))
        pooled.append(statistics.mean(e*e for group in groups for e in group))
    objective = [m + F(p) for m, p in zip(mse, i["deployment_penalty"])]
    eligible = [n for n, b in enumerate(bias) if abs(b) <= F(i["absolute_bias_limit"])]
    best = min(eligible, key=lambda n: (objective[n], i["model_order"][n]))
    incumbent = i["model_order"].index(i["incumbent"])
    selected = best if objective[incumbent] - objective[best] >= F(i["minimum_objective_improvement"]) else incumbent
    return {"target_weighted_mse": mse, "pooled_sample_mse": pooled, "target_weighted_bias": bias,
            "deployment_objectives": objective, "selected_model": i["model_order"][selected], "test_recalibration_permitted": False}, {}


def q136(i):
    exposures = {}
    for row in i["positions"]:
        if row["issuer"] not in i["excluded_issuers"]:
            exposures[row["issuer"]] = exposures.get(row["issuer"], 0) + F(row["market_value"])
    selected, rates = [], []
    for issuer in sorted(exposures):
        facts = [max((r for r in i["facts"] if r["issuer"] == issuer and r["period"] == period and r["accepted_at"] <= i["cutoff"]), key=lambda r: r["accepted_at"]) for period in (i["current_period"], i["prior_period"])]
        selected.extend(r["id"] for r in facts)
        rates.append(F(facts[0]["revenue"]) / F(facts[1]["revenue"]) - 1)
    exposure = [exposures[issuer] for issuer in sorted(exposures)]
    return {"eligible_fact_ids": sorted(selected), "issuer_exposures": exposure, "issuer_growth": rates,
            "gross_weighted_growth": dot(list(map(abs, exposure)), rates) / sum(map(abs, exposure)),
            "final_join_row_count": len(exposures), "query_plan": "ASOF_VERSION_THEN_ISSUER_JOIN"}, {}


def q137(i, turn):
    fcf = F(i["reported_cfo"]) - F(i["capex"]) - F(i.get("factoring_cfo_nonrecourse", i["memo_factoring_in_cfo"]))
    cash = F(i["closing_cash"]) - F(i["restricted_cash"]) - F(i.get("additional_cash_restriction", 0))
    uses = F(i["debt_due"]) - F(i.get("maturity_rolled_beyond_horizon", 0)) + F(i["minimum_cash"])
    if turn < 4:
        return {"normalized_operating_fcf": fcf, "usable_cash": cash, "funding_gap": max(0, uses - cash), "obligation_and_reserve_funded": cash >= uses}, {}
    sale = F(i["asset_sale_cash_price"]) - max(0, F(i["asset_sale_cash_price"]) - F(i["asset_tax_basis"])) * F(i["gain_tax_rate"])
    return {"normalized_operating_fcf": fcf, "net_asset_sale_cash": sale, "post_payment_cash_surplus": cash + sale - uses,
            "obligation_and_reserve_funded": cash + sale >= uses}, {}


def q138(i, turn):
    eligible = i["eligible_units"].copy()
    eligible[1] -= i.get("cancelled_b_units", 0)
    eligible[2] += i.get("c_acceptance_accelerated_units", 0)
    revenue, cost = values(i["unit_revenue"]), values(i["unit_cost"])
    revenue[2] -= F(i.get("c_pass_through_revenue_and_cost", 0))
    cost[2] += F(i.get("c_warranty_cost_per_unit", 0)) - F(i.get("c_pass_through_revenue_and_cost", 0))
    feasible = []
    for a in range(i["minimum_a"], eligible[0] + 1):
        for b in range(eligible[1] + 1):
            for c in range(min(eligible[2], i["capacity"] - a - b) + 1):
                plan = (a, b, c)
                gross = dot(plan, revenue)
                profit = gross - dot(plan, cost) - (F(i.get("fixed_expedite_fee", i["expedite_fee"])) if c else 0)
                feasible.append((profit, gross, plan))
    profit, gross, plan = min(feasible, key=lambda r: (-r[0], -r[1], r[2]))
    return {"delivery_units": plan, "recognized_revenue": gross, "net_gross_profit": profit,
            "revenue_floor_met": gross >= F(i["quarter_revenue_floor"])}, {"feasible_delivery_plans": len(feasible)}


def q139(i, turn):
    carry = F(i.get("of_that_accepted_in_q4", 0))
    prior = F(i["h1_reported_revenue"]) + F(i["q3_reported_revenue"]) - F(i.get("q3_unaccepted_bill_hold_revenue", 0))
    required = [F(target) - prior - carry for target in i["annual_revenue_guidance"]]
    capacity = F(i["q4_new_acceptance_capacity"])
    if turn < 3:
        return {"required_new_q4_revenue": required, "recognized_carry_into_q4": carry, "q4_new_capacity": capacity,
                "capacity_support": "full_range" if capacity >= max(required) else "floor_only" if capacity >= min(required) else "neither"}, {}
    firm = F(i["firm_new_q4_acceptances"]) + sum(F(row["amount"]) for row in i.get("new_contracts", []) if row["accepted_at"] <= i["decision_cutoff"])
    coverage = firm / required[0] * 100
    return {"firm_new_q4_revenue": firm, "floor_firm_coverage_pct": coverage, "uncovered_floor": max(0, required[0] - firm),
            "floor_contractually_supported": coverage >= F(i["required_firm_coverage_pct"])}, {"corrected_floor_new_acceptance_requirement": required[0]}


def q140(i, turn):
    outstanding = F(i["reported_share_count"]) - F(i.get("treasury_shares_in_reported_count", 0))
    price, tax = F(i["repurchase_price"]), 1 - F(i["tax_rate"])
    cash_fraction = F(i.get("cash_fraction", 1))
    issued = F(i.get("newly_exercised_shares", 0))
    proceeds = issued * F(i.get("exercise_price", 0))
    def plan(quantity):
        cash = quantity * price * cash_fraction
        borrowing = max(0, quantity * price * (1 - cash_fraction) - proceeds)
        earnings = F(i["earnings_before_financing"]) - tax * (cash * F(i["cash_yield"]) + borrowing * F(i.get("debt_rate", 0)))
        shares = outstanding - quantity + issued
        net_debt = F(i.get("existing_net_debt", 0)) + quantity * price - proceeds
        return earnings, shares, earnings / shares, net_debt
    if turn < 3:
        earnings, shares, eps, _ = plan(i["authorized_buyback_shares"])
        return {"post_financing_earnings": earnings, "post_buyback_shares": shares, "post_buyback_eps": eps, "eps_hurdle_met": eps >= F(i["minimum_eps"])}, {}
    candidates = [(q, plan(q)) for q in range(i["authorized_buyback_shares"] + 1) if plan(q)[3] <= F(i["maximum_net_debt_ebitda"]) * F(i["ebitda"])]
    q, record = min(candidates, key=lambda r: (-r[1][2], r[0]))
    full_leverage = plan(i["authorized_buyback_shares"])[3] / F(i["ebitda"])
    if turn == 3:
        return {"full_authorization_feasible": full_leverage <= F(i["maximum_net_debt_ebitda"]),
                "optimal_repurchase_shares": q, "optimal_eps": record[2], "planned_full_leverage": full_leverage}, {"feasible_quantities": len(candidates)}
    return {"optimal_repurchase_shares": q, "post_buyback_eps": record[2], "ending_net_debt": record[3],
            "leverage": record[3] / F(i["ebitda"]), "eps_hurdle_met": record[2] >= F(i["minimum_eps"])}, {"feasible_quantities": len(candidates)}


def q141(i, turn):
    opening = {k: F(v) for k, v in i["opening_arr"].items()}
    closing = {k: F(v) for k, v in i["closing_subscription_values"].items()}
    if "account_migration" in i:
        old, new = i["account_migration"]["old_id"], i["account_migration"]["new_id"]
        closing[old] += closing.pop(new)
    closing["D"] -= F(i.get("d_oneoff_services_in_closing_value", 0))
    if "genuine_new_customer_f_arr" in i:
        closing["F"] = F(i["genuine_new_customer_f_arr"])
    denominator = sum(opening.values())
    retained = sum(closing.get(name, 0) for name in opening)
    nrr = retained / denominator * 100
    if turn < 4:
        return {"ending_arr": sum(closing.values()), "existing_customer_nrr_pct": nrr,
                "gross_retention_pct": sum(min(value, closing.get(name, 0)) for name, value in opening.items()) / denominator * 100,
                "new_customer_arr": sum(value for name, value in closing.items() if name not in opening)}, {}
    prospective = sum(closing.values()) - closing["B"]
    return {"historical_nrr_pct": nrr, "next_renewal_arr": prospective,
            "next_renewal_original_cohort_pct": (retained - closing["B"]) / denominator * 100,
            "new_arr_needed_for_target": max(0, F(i["next_period_arr_target"]) - prospective)}, {}


def q142(i, turn):
    annuity = sum((1 + F(i["discount_rate"])) ** -year for year in range(1, i["years"] + 1))
    choices = []
    for flags in itertools.product((0, 1), repeat=3):
        a, b, c = flags
        cash = dot(flags, values(i["cash_cost"]))
        if cash > F(i.get("replacement_cash_budget", i["budget"])) or dot(flags, i["team_units"]) > i["team_capacity"] or (b and i["b_requires_c"] and not c):
            continue
        savings = dot(flags, values(i["annual_savings"])) - c * F(i["c_recurring_platform_cost"]) - b * c * F(i.get("bc_overlap_correction", i["bc_overlap"])) - a * F(i.get("mandatory_a_customer_rebate", i["a_required_rebate"]))
        selected = [name for name, flag in zip(i["initiative_ids"], flags) if flag]
        choices.append((savings * annuity - cash, cash, selected, savings))
    npv, cash, selected, savings = min(choices, key=lambda r: (-r[0], r[1], r[2]))
    return {"selected_initiatives": selected or ["none"], "annual_net_savings": savings, "upfront_cash": cash, "three_year_npv": npv}, {"feasible_subsets": len(choices)}


def q143(i, turn):
    amortization = F(i.get("replacement_rd_amortization", i["rd_amortization"]))
    opening_rd = F(i["opening_rd_asset"])
    closing_rd = opening_rd + F(i["current_rd_expense"]) - amortization
    ebit = F(i["reported_ebit"]) + F(i["current_rd_expense"]) - amortization + F(i.get("interest_component_of_reported_rent", 0))
    nopat = ebit * (1 - F(i["tax_rate"]))
    capital = F(i["reported_average_invested_capital"]) + (opening_rd + closing_rd) / 2 + (F(i.get("opening_lease_asset", 0)) + F(i.get("closing_lease_asset", 0))) / 2 - F(i.get("nonoperating_cash_in_reported_average_capital", 0))
    wacc = F(i.get("replacement_wacc", i["wacc"]))
    profit = nopat - wacc * capital
    answer = {"normalized_nopat": nopat, "average_invested_capital": capital, "economic_profit": profit}
    answer.update({"economic_profit_hurdle_met": profit >= F(i["economic_profit_hurdle"])} if turn == 4 else {"roic_spread_pp": (nopat / capital - wacc) * 100})
    return answer, {"closing_research_asset": closing_rd}


def q144(i, turn):
    def path(action):
        outflows, receipts = values(i["cash_outflows"]), values(i["cash_receipts"])
        receipts[1] -= F(i.get("delayed_amount", 0))
        if action == "layoffs":
            outflows[0] += F(i["layoff_severance_m1"])
            outflows[1:] = [x - F(i["layoff_savings_m2_m3_each"]) for x in outflows[1:]]
        if action == "asset_sale":
            sale = F(i["asset_sale_gross_m2"]) - max(0, F(i["asset_sale_gross_m2"]) - F(i["asset_tax_basis"])) * F(i["tax_rate"])
            receipts[2 if turn == 4 else 1] += sale
        cash = F(i["closing_cash_at_start"]) - F(i["restricted_cash"])
        before, after = [], []
        for outflow, receipt in zip(outflows, receipts):
            cash -= outflow
            before.append(cash)
            cash += receipt
            after.append(cash)
        funding = max(0, F(i["minimum_cash"]) - min(after + before if turn == 4 else after))
        return funding, after, before
    if turn < 3:
        funding, after, _ = path("none")
        breach = next((name for name, cash in zip(i["month_order"], after) if cash < F(i["minimum_cash"])), "none")
        return {"unfunded_month_end_cash": after, "minimum_upfront_funding": funding,
                "funded_month_end_cash": [cash + funding for cash in after], "first_reserve_breach": breach}, {}
    choices = {action: path(action) for action in ("none", "layoffs", "asset_sale")}
    action = min(choices, key=lambda name: (choices[name][0], name))
    funding, after, before = choices[action]
    answer = {"chosen_action": action, "minimum_upfront_funding": funding, "funded_month_end_cash": [cash + funding for cash in after]}
    answer.update({"funding_by_action": [choices[name][0] for name in ("none", "layoffs", "asset_sale")]} if turn == 3 else {"minimum_pre_receipt_cash": min(before) + funding})
    return answer, {"funding_by_action": {name: result[0] for name, result in choices.items()}}


def q145(i, turn):
    likelihood = values(i["positive_test_likelihoods"])
    likelihood[2] = F(i.get("corrected_cancel_positive_likelihood", likelihood[2]))
    joint = [p * l for p, l in zip(values(i["prior_probabilities"]), likelihood)]
    evidence = sum(joint)
    posterior = [p / evidence for p in joint]
    if "new_audit_likelihood_by_state" in i:
        joint = [p * l for p, l in zip(posterior, values(i["new_audit_likelihood_by_state"]))]
        posterior = [p / sum(joint) for p in joint]
    if turn < 3:
        expected = dot(posterior, values(i["payoffs"]))
        return {"state_probabilities": posterior, "evidence_probability": evidence, "expected_payoff": expected,
                "research_hurdle_met": expected >= F(i["initial_expected_payoff_hurdle"])}, {}
    approval = dot(posterior, values(i["approval_probability_by_state"]))
    state_payoffs = [F(i["approval_probability_by_state"][n]) * F(i["payoffs"][n]) + (1 - F(i["approval_probability_by_state"][n])) * F(i["unapproved_payoff_on_time_or_late"]) for n in range(2)] + [F(i["payoffs"][2])]
    expected = dot(posterior, state_payoffs)
    loss = 1 - approval
    result = {"expected_payoff": expected, "loss_probability": loss,
              "joint_policy_met": expected >= F(i["minimum_expected_payoff"]) and loss <= F(i["maximum_loss_probability"])}
    result.update({"approval_probability": approval} if turn == 3 else {"state_probabilities": posterior})
    return result, {"posterior_sum": sum(posterior), "duplicate_test_counted": False}


def q146(i, turn):
    cashflows = values(i["fcff_years_1_3"])
    cashflows[1] -= F(i.get("nonoperating_disposal_receipt_in_year2_fcff", 0))
    cashflows = [a - F(b) for a, b in zip(cashflows, i.get("segment_fcff_years_1_3", [0, 0, 0]))]
    cash = F(i["reported_cash"]) - F(i.get("restricted_cash_in_reported_balance", 0)) + F(i.get("net_sale_cash_at_valuation", 0))
    debt, shares = F(i["debt"]), F(i["shares"])
    wacc = F(i.get("replacement_wacc", i["wacc"]))
    explicit = sum(flow / (1 + wacc) ** t for t, flow in enumerate(cashflows, 1))
    def ev(growth):
        terminal = cashflows[-1] * (1 + growth) / (wacc - growth) / (1 + wacc) ** len(cashflows)
        return explicit + terminal, terminal
    def implied(target_ev):
        terminal = (target_ev - explicit) * (1 + wacc) ** len(cashflows)
        return (terminal * wacc - cashflows[-1]) / (terminal + cashflows[-1])
    value, terminal = ev(F(i["terminal_growth"]))
    if turn < 3:
        return {"operating_enterprise_value": value, "equity_value_per_share": (value - debt + cash) / shares,
                "terminal_share_of_ev_pct": terminal / value * 100,
                "market_implied_terminal_growth_pct": implied(F(i["market_enterprise_value"])) * 100}, {}
    if turn == 3:
        return {"remaining_fcff": cashflows, "operating_enterprise_value": value,
                "equity_value_per_share_before_distribution": (value - debt + cash) / shares,
                "maximum_immediate_distribution": max(0, cash - F(i["required_cash_reserve"]) - F(i["near_term_debt_payment_reserve"]))}, {}
    bounds = [(ev(F(g))[0] - debt + cash) / shares for g in i["admissible_terminal_growth"]]
    claim = F(i["share_value_claim"])
    return {"equity_value_bounds": bounds, "claim_implied_terminal_growth_pct": implied(claim * shares + debt - cash) * 100,
            "claim_feasible_in_growth_range": min(bounds) <= claim <= max(bounds),
            "claim_classification": "below_range" if claim < min(bounds) else "above_range" if claim > max(bounds) else "inside_range"}, {"terminal_value_monotone_in_growth": cashflows[-1] * (1 + wacc) > 0}


def q147(i, turn):
    ebitda = F(i["ebitda"]) + min(F(i["integration"]), F(i["cap"]) * F(i["ebitda"]))
    feasible = []
    for draw in range(0, i["facility"] + 1, 5):
        gross = F(i["cash"]) + draw * (1 - F(i["fee_rate"]))
        purchases = [i["required_buyback"]] if i["objective"] == "minimum_draw" else range(0, math.floor(gross) + 1, 5)
        for buy in purchases:
            cash = gross - buy
            leverage = (F(i["debt"]) + draw - min(cash, F(i["netting_cap"]))) / ebitda
            if buy % 5 == 0 and cash >= F(i["reserve"]) and leverage <= F(i["limit"]):
                feasible.append((buy, draw, cash, leverage))
    if not feasible:
        raise ValueError("No permissible authored repurchase plan")
    key = (lambda r: (r[1], -r[2])) if i["objective"] == "minimum_draw" else (lambda r: (-r[0], r[1]))
    buy, draw, cash, leverage = min(feasible, key=key)
    return {"covenant_ebitda": ebitda, "repurchase_cash": buy, "new_draw": draw, "remaining_cash": cash,
            "net_leverage": leverage, "plan_permitted": True}, {"feasible_plans": len(feasible)}


def q148(i, turn):
    feasible = []
    for cash, bond, foreign in itertools.product(range(0, i["cash"] - i["cash_reserve"] + 1, 5), range(0, i["bond_cap"] + 1, 5), range(0, i["foreign_cap"] + 1, 5)):
        gross = [F(cash), F(bond), foreign * F(i["fx"])]
        eligible = dot(gross, [1 - F(h) for h in i["haircuts"]])
        cost = dot(gross, values(i["fund_rates"]))
        if eligible >= F(i["requirement"]):
            feasible.append((cost, foreign, cash, bond, eligible))
    key = (lambda r: (r[1], r[0], r[2], r[3])) if i["objective"] == "minimum_foreign" else (lambda r: r[:4])
    cost, foreign, cash, bond, eligible = min(feasible, key=key)
    return {"posted_units": [cash, bond, foreign], "eligible_collateral_value": eligible, "funding_cost": cost,
            "unposted_cash": i["cash"] - cash, "foreign_haircut": i["haircuts"][2],
            "posting_authorized": cost <= F(i["cost_budget"])}, {"collateral_feasible_plans": len(feasible)}


def q149(i, turn):
    feasible = [(F(0), F(0), "none", 0)] if i["required_lots"] == 0 else []
    eligible = [row for row in i["bonds"] if row["deliverable"]]
    for row in eligible:
        dirty = (F(row["clean"]) + F(row["accrued"])) / 100
        capital = dirty * F(row["haircut"]) + F(i["margin"])
        carry = (F(i["future"]) * F(row["factor"]) + F(row["delivery_accrued"]) + F(row["coupon"])) / 100 - dirty * (1 + F(row["repo"]) * F(i["days"]) / 360)
        for lots in range(max(1, i["required_lots"]), 11):
            if lots * capital <= F(i["capital_budget"]):
                feasible.append((lots * carry, lots * capital, row["id"], lots))
    carry, capital, name, lots = min(feasible, key=lambda r: (-r[0], r[1], r[2], r[3]))
    return {"selected_issue": name, "delivery_lots": lots, "upfront_capital": capital, "net_carry": carry,
            "eligible_issue_count": len(eligible), "delivery_commitment_met": lots >= i["required_lots"]}, {"funded_choices": len(feasible)}


def q150(i, turn):
    eligible = [r for r in i["obligations"] if r["senior"] and r["currency"] == "USD" and r["published"] <= i["cutoff"]]
    price = min(F(r["auction_price"]) for r in eligible)
    protection = F(i["hedge_face"]) * (1 - price / 100)
    bond = F(i["bond_face"]) * price / 100
    cash = F(i["opening_cash"]) + (bond if i["bond_pay_day"] <= i["expense_day"] else 0) + (protection if i["protection_pay_day"] <= i["expense_day"] else 0)
    return {"eligible_obligation_ids": sorted(r["id"] for r in eligible), "auction_recovery_price": price,
            "protection_receipt": protection, "combined_investment_pnl": bond + protection - F(i["bond_face"]) * F(i["basis"]) / 100 - F(i["premium"]),
            "cash_at_expense_deadline": cash, "cash_shortfall": max(0, F(i["expense"]) - cash)}, {}


def q151(i, turn):
    feasible = []
    for nominations in itertools.product(*(range(0, cap + 1, 5) for cap in i["caps"])):
        delivered = [q * (1 - F(loss)) for q, loss in zip(nominations, i["loss"])]
        cash = dot(nominations, [F(p) + F(t) for p, t in zip(i["supply_prices"], i["transport"])])
        short = max(0, F(i["demand"]) - sum(delivered))
        if (i["objective"] == "meet_demand" and short == 0) or (i["objective"] == "min_shortfall" and cash <= F(i["cash_budget"])):
            feasible.append((short, cash, nominations, delivered))
    short, cash, nominations, delivered = min(feasible, key=lambda r: (r[0], r[1], r[2]))
    return {"nominations": nominations, "delivered_by_route": delivered, "cash_cost": cash,
            "unserved_volume": short, "cash_budget_satisfied": cash <= F(i["cash_budget"]), "demand_fully_met": short == 0}, {"admissible_plans": len(feasible)}


def q152(i, turn):
    revenue = [F(r) - F(i["sales_tax"]) for r in i["gross_revenues"]]
    earnout = [min(F(i["earnout_cap"]), max(0, r - F(i["threshold"])) * F(i["participation"])) for r in revenue]
    cash = F(i["base_cash"]) + max(0, min(i["days"], i["tick_stop"]) - i["tick_start"]) * F(i["tick"])
    worst = cash + min(earnout) - F(i["entry"]) * (1 + F(i["fund_rate"]) * F(i["days"]) / 365) - F(i["borrow_fee"])
    return {"eligible_revenues": revenue, "scenario_earnouts": earnout, "closing_cash_without_earnout": cash,
            "worst_net_profit": worst, "worst_simple_annualized_return": worst / F(i["entry"]) * 365 / i["days"],
            "entry_hurdle_met": worst >= F(i["hurdle"])}, {}


def q153(i, turn):
    index = (F(i["lag_indices"][0]) * (1 - F(i["weight"])) + F(i["lag_indices"][1]) * F(i["weight"])) / F(i["base_index"])
    invoice = (F(i["clean_real"]) + F(i["real_accrued"])) * index
    maturity = [100 * (max(1, F(ratio)) + F(i["coupon"]) * F(ratio)) for ratio in i["terminal_ratios"]]
    factor = 1 + F(i["funding_rate"]) * F(i["days"]) / 360
    profit = min(maturity) - invoice * factor
    maximum = min(F(i["facility"]), min(maturity) / factor) / index - F(i["real_accrued"])
    return {"settlement_index_ratio": index, "nominal_dirty_invoice": invoice, "scenario_maturity_cash": maturity,
            "worst_financed_profit": profit, "maximum_clean_real_entry": maximum,
            "entry_fundable_and_nonloss": invoice <= F(i["facility"]) and profit >= 0}, {}


def q154(i, turn):
    strikes = [90, 100, 110]
    def outlay(cash, calls):
        return cash + sum(F(quantity) * F(i["asks"][n] if quantity > 0 else i["bids"][n]) for n, quantity in enumerate(calls))
    def payoff(cash, calls, spot):
        return cash + dot(calls, [max(spot - strike, 0) for strike in strikes])
    feasible = []
    for calls in itertools.product(range(-2, 3), repeat=3):
        if sum(calls) < 0:
            continue
        for cash in range(6):
            if cash < -sum(min(0, q) for q in calls) * F(i["reserve_per_short"]):
                continue
            if min(payoff(cash, calls, spot) for spot in [0] + strikes) < 0:
                continue
            feasible.append((outlay(cash, calls), sum(map(abs, calls)), cash, calls))
    cost, _, cash, calls = min(feasible)
    return {"selected_cash_and_calls": [cash] + list(calls), "minimum_initial_outlay": cost,
            "expiry_payoffs": [payoff(cash, calls, spot) for spot in (0, 90, 100, 110, 120)],
            "far_right_payoff_slope": sum(calls), "test_butterfly_outlay": outlay(2 * F(i["reserve_per_short"]), [1, -2, 1]),
            "negative_cost_nonnegative_payoff_exists": cost < 0}, {"global_payoff_certificate": "Piecewise linear: nonnegative at zero and every strike, with nonnegative final slope.", "feasible_positions": len(feasible)}


def q155(i, turn):
    if i["official_valid"]:
        fixing, ids = F(i["official_rate"]), ["official"]
    else:
        eligible = [row for row in i["dealer_quotes"] if not row["affiliated"] and row["published"] <= i["fixing_cutoff"]]
        fixing, ids = statistics.mean(F(row["rate"]) for row in eligible), sorted(row["id"] for row in eligible)
    derivative = F(i["usd_notional"]) * (1 - F(i["strike"]) / fixing)
    available = F(i["usd_cash"]) - F(i["segregated"]) + (F(i["customer_receipt"]) if i["customer_time"] <= i["deadline"] else 0) + (derivative if i["ndf_time"] <= i["deadline"] else 0)
    bridge = max(0, F(i["payable"]) - available)
    return {"valid_fixing": fixing, "used_fixing_ids": ids, "ndf_usd_receipt": derivative,
            "available_deadline_cash": available, "bridge_required": bridge,
            "bridge_facility_sufficient": bridge <= F(i["bridge_limit"])}, {}


def q156(i, turn):
    base = F(i["performing_and_default_par"]) + F(i["principal_cash"]) - F(i["default_par"]) * (1 - F(i["default_credit"])) - max(0, F(i["ccc_nondefault"]) - F(i["ccc_limit"])) * F(i["ccc_haircut"])
    residual = F(i["interest_cash"]) - F(i["expenses"]) - F(i["senior_interest"]) - F(i["mezz_interest"])
    actions = [("retain_cash", base)]
    if i["purchase_allowed"] and F(i["purchase_cost"]) <= F(i["principal_cash"]):
        actions.append(("buy_loan", base - F(i["purchase_cost"]) + F(i["purchase_face"]) * F(i["purchase_credit"])))
    plans = []
    for action, collateral in actions:
        senior = collateral / F(i["senior"])
        combined = collateral / (F(i["senior"]) + F(i["mezz"]))
        permitted = senior >= F(i["senior_test"]) and combined >= F(i["combined_test"])
        plans.append((residual if permitted else F(0), collateral, action, senior, combined, permitted))
    equity, collateral, action, senior, combined, permitted = min(plans, key=lambda r: (-r[0], -r[1], r[2]))
    return {"manager_action": action, "adjusted_collateral": collateral, "senior_coverage": senior, "combined_coverage": combined,
            "cash_distributions": [i["senior_interest"], i["mezz_interest"], residual - equity, equity],
            "equity_release_permitted": permitted}, {"residual_interest_cash": residual}


def q157(i, turn):
    assets = F(i["cash_millions"]) + F(i["liquid_assets_millions"]) + F(i.get("corrected_locked_assets_millions", i["locked_assets_millions"]))
    nav = assets / F(i["initial_units"])
    requests = values(i["requested_units"])
    gate = min(sum(requests), F(i["initial_units"]) * F(i["gate_fraction"]))
    subscription = F(i.get("subscription_millions", 0))
    free = F(i["cash_millions"]) + subscription - F(i["cash_reserve_millions"])
    if turn < 3:
        paid = gate * nav
        draw = F(0)
        sale = max(0, paid - free)
    else:
        paid = min(gate * nav, free + F(i["bridge_limit_millions"]))
        draw = max(0, paid - free)
        sale = draw * (1 + F(i["bridge_interest_fraction"]))
        if sale > F(i["liquid_assets_millions"]):
            raise ValueError("Authored bridge repayment exceeds liquid assets")
    payments = [paid * request / sum(requests) for request in requests]
    answer = {"nav_per_unit": nav, "redemption_payments": payments,
              "pending_units": [request - payment / nav for request, payment in zip(requests, payments)],
              "asset_sale": sale, "bridge_draw": draw}
    if turn == 4:
        answer["ending_nav"] = assets + subscription - paid - draw * F(i["bridge_interest_fraction"])
    return answer, {"original_gate_units": gate, "payment_cash": paid}


def q158(i, turn):
    positions = i["positions"]
    accounts = i["accounts"].copy()
    accounts[1] = i.get("corrected_account_B", accounts[1])
    def margin(remaining, permit_offset=False):
        ab = abs(remaining[0] - remaining[1]) * F(i["initial_margin_per_contract"][0]) if accounts[0] == accounts[1] or permit_offset else dot(remaining[:2], values(i["initial_margin_per_contract"][:2]))
        return ab + remaining[2] * F(i["initial_margin_per_contract"][2])
    delta = dot(positions, values(i["signed_delta_per_contract"]))
    if turn < 3:
        required = margin(positions)
        return {"required_margin": required, "funding_shortfall": max(0, required - F(i["cash"])),
                "free_cash": max(0, F(i["cash"]) - required), "portfolio_delta": delta}, {}
    feasible = []
    for closes in itertools.product(*(range(p + 1) for p in positions)):
        remaining = [p - c for p, c in zip(positions, closes)]
        required = margin(remaining)
        delta = dot(remaining, values(i["signed_delta_per_contract"]))
        fees = sum(closes) * F(i["execution_fee_per_closed_contract"])
        economic = dot(closes, values(i["foregone_alpha_per_closed_contract"])) + fees
        if required + fees <= F(i["cash"]) and abs(delta) <= F(i["absolute_delta_limit"]):
            feasible.append((economic, required, closes, remaining, delta, fees))
    economic, required, closes, remaining, delta, fees = min(feasible)
    if turn == 3:
        return {"contracts_closed": closes, "remaining_margin": required, "execution_fees": fees,
                "remaining_delta": delta, "economic_cost": economic}, {"feasible_closeouts": len(feasible)}
    return {"contracts_closed": closes, "current_margin": required, "margin_after_approval": margin(remaining, True),
            "remaining_delta": delta, "free_cash_now": F(i["cash"]) - required - fees}, {"economic_cost": economic, "feasible_closeouts": len(feasible)}


def q159(i, turn):
    old = F(i["old_contract_quantity"]) * F(i["old_multiplier"]) * (F(i["old_exit_bid"]) - F(i["old_previous_settlement"]))
    multiplier = F(i.get("corrected_new_multiplier", i["reported_new_multiplier"]))
    new = F(i["new_contract_quantity"]) * multiplier * (F(i["new_settlement"]) - F(i["new_entry_ask"]))
    if turn == 1:
        continuous = F(i["old_contract_quantity"]) * F(i["old_multiplier"]) * (F(i["new_settlement"]) - F(i["old_previous_settlement"]))
        return {"old_contract_pnl": old, "new_contract_pnl": new, "total_pnl": old + new, "continuous_series_overstatement": continuous - old - new}, {}
    if turn == 2:
        point = F(i["new_contract_quantity"]) * multiplier
        return {"old_contract_pnl": old, "new_contract_pnl": new, "total_pnl": old + new,
                "new_dollars_per_point": point, "new_marked_notional": point * F(i["new_settlement"])}, {}
    remaining = min(i["new_contract_quantity"], math.floor(F(i["target_dollars_per_point"]) / multiplier), math.floor(F(i["maximum_margin"]) / F(i["margin_per_new_contract"])))
    closed = i["new_contract_quantity"] - remaining
    realized = closed * multiplier * (F(i["new_exit_bid"]) - F(i["new_entry_ask"]))
    marked = remaining * multiplier * (F(i.get("new_final_settlement", i["new_settlement"])) - F(i["new_entry_ask"]))
    cumulative = old + realized + marked
    if turn == 3:
        return {"new_contracts_remaining": remaining, "new_contracts_closed": closed,
                "required_margin": remaining * F(i["margin_per_new_contract"]), "cumulative_pnl": cumulative,
                "dollars_per_point": remaining * multiplier}, {}
    fees = (i["old_contract_quantity"] + i["new_contract_quantity"] + closed) * F(i["fee_per_filled_contract"])
    cash = F(i["initial_cash"]) + cumulative - fees - F(i["external_withdrawal"])
    margin = remaining * F(i["margin_per_remaining_contract"])
    return {"net_cumulative_pnl": cumulative - fees, "total_execution_fees": fees, "ending_cash": cash,
            "required_margin": margin, "free_cash": cash - margin}, {}


def q160(i, turn):
    counts = i["counts"].copy()
    means = values(i["mean_cost_bps"])
    means[2] = F(i.get("corrected_new_fast_mean_cost_bps", means[2]))
    if "replication_counts" in i:
        means = [(n * mean + extra * F(new_mean)) / (n + extra) for n, mean, extra, new_mean in zip(counts, means, i["replication_counts"], i["replication_means_bps"])]
        counts = [n + extra for n, extra in zip(counts, i["replication_counts"])]
    weights = values(i.get("new_deployment_stratum_weights", i["deployment_stratum_weights"]))
    improvement = dot(weights, [means[1] - means[0], means[3] - means[2]]) - F(i["implementation_charge_bps"])
    variance = sum(weight ** 2 * (F(i["known_individual_variances_bps_squared"][2*n]) / counts[2*n] + F(i["known_individual_variances_bps_squared"][2*n+1]) / counts[2*n+1]) for n, weight in enumerate(weights))
    se = math.sqrt(variance)
    lower = float(improvement) - 1.96 * se
    answer = {"net_cost_improvement": improvement, "standard_error": se, "lower_confidence_bound": lower, "promote_route": lower > 0}
    if turn == 1:
        answer["unstratified_improvement"] = sum(means[n] * counts[n] for n in (1, 3)) / sum(counts[n] for n in (1, 3)) - sum(means[n] * counts[n] for n in (0, 2)) / sum(counts[n] for n in (0, 2))
    return answer, {"cell_counts": counts, "pooled_cell_means": means, "target_variance": variance}


def q161(i, turn):
    loss_per_share = F(i["basis_per_P_share"]) - F(i["current_P_price"])
    matching = i.get("affiliated_executed_P_shares", i.get("affiliated_replacement_P_shares", 0))
    def economics(sold):
        disallowed = min(matching, sold) * loss_per_share
        allowed = sold * loss_per_share - disallowed
        credit = allowed * F(i["tax_rate"])
        trading = sold * (F(i["P_sale_cost_per_share"]) + F(i["Q_per_P_exposure"]) * F(i["Q_purchase_cost_per_share"]))
        benefit = credit - trading - sold * F(i["basis_risk_charge_per_P_sold"])
        return allowed, disallowed, credit, trading, benefit
    sold = i["original_P_shares"] if turn < 3 else min((q for q in i["sale_grid"] if q <= i["original_P_shares"] - i["minimum_original_p_retained"]), key=lambda q: (-economics(q)[4], q))
    allowed, disallowed, credit, trading, benefit = economics(sold)
    if turn == 1:
        answer = {"shares_sold": sold, "allowed_loss": allowed, "tax_credit": credit, "total_trading_cost": trading, "net_economic_benefit": benefit}
    elif turn == 2:
        answer = {"allowed_loss": allowed, "disallowed_loss": disallowed, "tax_credit": credit,
                  "replacement_basis_per_share": F(i["replacement_initial_basis_per_share"]) + disallowed / matching, "net_economic_benefit": benefit}
    elif turn == 3:
        answer = {"shares_sold": sold, "tax_credit": credit, "disallowed_loss": disallowed,
                  "net_economic_benefit": benefit, "original_p_retained": i["original_P_shares"] - sold}
    else:
        answer = {"shares_sold": sold, "q_shares_bought": sold * i["Q_per_P_exposure"], "allowed_loss": allowed,
                  "net_economic_benefit": benefit, "original_p_retained": i["original_P_shares"] - sold}
    return answer, {"loss_identity_residual": allowed + disallowed - sold * loss_per_share}


def q162(i, turn):
    bid = F(i.get("revised_B_bid", i.get("B_firm_bid", F(i["B_stale_price"]) * (1 - F(i["B_markdown_fraction"])))))
    nav = F(i["cash"]) + F(i["A_shares"]) * F(i["A_mid_price"]) + F(i["B_shares"]) * bid
    pnl = nav - F(i["opening_NAV"]) - F(i["external_deposit"])
    if turn < 3:
        answer = {"current_nav": nav, "valuation_pnl": pnl,
                  "modified_dietz_return": pnl / (F(i["opening_NAV"]) + F(i["deposit_remaining_period_fraction"]) * F(i["external_deposit"])), "available_cash": i["cash"]}
        answer.update({"valuation_reserve": F(i["B_shares"]) * (F(i["B_stale_price"]) - bid)} if turn == 1 else {"immediate_b_sale_proceeds": bid * min(i["B_shares"], i["B_executable_bid_depth"])})
        return answer, {}
    depth = i.get("revised_B_bid_depth", i["B_executable_bid_depth"])
    feasible = []
    for a, b in itertools.product(range(min(i["A_shares"], i["A_sale_limit"]) + 1), range(min(i["B_shares"], depth) + 1)):
        cash_without_borrowing = F(i["cash"]) + a * F(i["A_bid"]) + b * bid
        payable = min(F(i["redemption_requested"]), cash_without_borrowing + F(i["credit_limit"]) - F(i["cash_reserve"]))
        if turn == 3 and payable < F(i["redemption_requested"]):
            continue
        draw = max(0, payable + F(i["cash_reserve"]) - cash_without_borrowing)
        cost = a * (F(i["A_mid_price"]) - F(i["A_bid"])) + draw * F(i["credit_interest_fraction"])
        feasible.append((payable, cost, draw, a, b, cash_without_borrowing + draw - payable))
    payable, cost, draw, a, b, cash = min(feasible, key=lambda r: (-r[0], r[1], r[2], r[3], r[4]))
    ending = nav - payable - cost
    if turn == 3:
        return {"shares_sold": [a, b], "credit_draw": draw, "total_economic_cost": cost, "ending_nav": ending, "ending_cash": cash}, {"feasible_plans": len(feasible)}
    return {"maximum_redemption": payable, "unpaid_shortfall": F(i["redemption_requested"]) - payable,
            "shares_sold": [a, b], "credit_draw": draw, "ending_nav": ending}, {"economic_cost": cost, "pretrade_nav": nav}


def q163(i, turn):
    prior = F(i["prior_good_probability"])
    true_positive = F(i.get("replacement_study_positive_given_good", i["positive_given_good"]))
    false_positive = F(i.get("replacement_study_positive_given_bad", i.get("corrected_positive_given_bad", i["positive_given_bad"])))
    probabilities = [prior * true_positive + (1 - prior) * false_positive]
    probabilities.append(1 - probabilities[0])
    posterior = [prior * true_positive / probabilities[0], prior * (1 - true_positive) / probabilities[1]]
    actions = {name: values(row) for name, row in i["payoffs_by_state"].items()}
    if "maximum_investment_loss" in i:
        actions.pop("A")
        actions[i["replacement_A_action"]] = values(i["half_A_payoffs"])
        actions = {name: row for name, row in actions.items() if min(row) >= -F(i["maximum_investment_loss"])}
    def expected(name, probability):
        good, bad = actions[name]
        return probability * good + (1 - probability) * bad
    policy = [min(actions, key=lambda name: (-expected(name, p), name)) for p in posterior]
    net = sum(probability * expected(name, p) for probability, name, p in zip(probabilities, policy, posterior)) - F(i["study_cost"])
    unconditional = max(expected(name, prior) for name in actions)
    return {"signal_probabilities": probabilities, "posterior_good_probabilities": posterior, "conditional_policy": policy,
            "study_net_expected_payoff": net, "purchase_study": net > unconditional}, {"best_unconditional_payoff": unconditional, "study_count": 1}


def q164(i, turn):
    fills, cancelled, pending = {}, set(), set()
    for event in i["event_history"]:
        kind = event["event"]
        if kind in ("fill", "fill_duplicate"):
            key = event["execution_id"]
            record = (event["order"], event["shares"])
            if key in fills and fills[key] != record:
                raise ValueError("Execution identifier has inconsistent fill records")
            fills[key] = record
        elif kind == "cancel_requested":
            pending.add(event["order"])
        elif kind == "cancel_ack":
            cancelled.add(event["order"])
            pending.discard(event["order"])
        elif kind == "bust":
            if event["execution_id"] not in fills:
                raise ValueError("Bust references unknown execution")
            order, quantity = fills[event["execution_id"]]
            fills[event["execution_id"]] = order, quantity - event["shares"]
    filled = {order["id"]: sum(q for name, q in fills.values() if name == order["id"]) for order in i["orders"]}
    confirmed = i["starting_inventory"] + sum(filled.values())
    live = sum(order["quantity"] - filled[order["id"]] for order in i["orders"] if order["side"] == "buy" and order["id"] not in cancelled)
    target = i.get("new_target_shares", i["target_shares"])
    if turn == 4:
        if live or pending:
            raise ValueError("Replacement requires resolved original orders")
        lot = i["new_order_board_lot"]
        quantity = max(0, min((target - confirmed) // lot, math.floor(F(i["remaining_spendable_cash"]) / (lot * F(i["new_executable_ask"])))) * lot)
        return {"confirmed_inventory": confirmed, "new_order_quantity": quantity, "cash_required": quantity * F(i["new_executable_ask"]),
                "inventory_if_filled": confirmed + quantity, "remaining_target_shortfall": max(0, target - confirmed - quantity), "next_action": "SUBMIT"}, {"live_buy_quantity": live, "cancelled_orders": sorted(cancelled)}
    action = "WAIT" if pending else "CANCEL_O2_WAIT" if confirmed + live > target and "O2" not in cancelled else "WAIT"
    answer = {"confirmed_inventory": confirmed, "live_buy_quantity": live, "potential_overfill": max(0, confirmed + live - target), "next_action": action}
    answer.update({"worst_case_inventory": confirmed + live} if turn == 1 else {"replacement_quantity_after_clean_cancel": max(0, target - confirmed)})
    return answer, {"valid_execution_quantities": {key: quantity for key, (_, quantity) in fills.items()}, "pending_cancellations": sorted(pending)}


SOLVERS = {f"q{n}": globals()[f"q{n}"] for n in range(107, 165)}


def audit_case(packet):
    cid = packet["question_id"]
    turns = packet.get("turns", [packet])
    if cid not in SOLVERS:
        return {"answers_by_turn": {}, "certificate": {}, "unsupported": ["Authored staged derivation not implemented"], "ambiguities": []}
    results, certificates, cumulative = {}, {}, {}
    for stage in turns:
        if cid == "q164":
            cumulative.setdefault("event_history", []).extend(stage["inputs"].get("events", stage["inputs"].get("new_events", [])))
        cumulative.update(stage["inputs"])
        cumulative.update(stage["inputs"].get("amendment", {}))
        answers, certificate = SOLVERS[cid](cumulative, stage["turn"]) if int(cid[1:]) >= 137 else SOLVERS[cid](cumulative)
        if set(answers) != {field["name"] for field in stage["outputs"]}:
            raise ValueError(f"Independent calculation does not cover the output schema for {cid} turn {stage['turn']}")
        results[str(stage["turn"])] = answers
        certificates[str(stage["turn"])] = certificate
    return plain({"answers_by_turn": results, "certificate": certificates, "unsupported": [], "ambiguities": []})
