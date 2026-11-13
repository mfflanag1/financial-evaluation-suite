"""Exact, fixed-version audits of retained source questions.

Operands are independently transcribed from the public question and retained
excerpts, never from stored solutions. Canonical packet hashes bind that review
to its exact scope. These are arithmetic and conditional interpretation checks,
not arbitrary-document parsers, legal opinions, or new redistribution grants.
Reusable helpers expose economically meaningful boundaries for controls.
"""
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction as F
from functools import wraps
import hashlib
import json


REVIEWED_PACKETS = {
    "credit-001": "319315fe2e5428fd8a2d815eb0bff635f801353ff27b8263371de49bd74cec02",
    "credit-002": "dfb3799289b0bdfc2e016eccd8ae007d89bd48238dd37e66974424751bd70089",
    "credit-003": "35e879045e55c2f2fa5ae119f896cc07997126224a784e5883aa3a6cacc6c763",
    "energy-003": "8ace400f3156a21d03299cbb21f34042ed1e007876efca8f5075d1ba9c3dca84",
    "equity-001": "2b7b9b57a7aa4ceef21eee48b60964ee848d6d97dd5223839916bf730ed95253",
    "equity-002": "31518a114b61c18cf2219df9c61d6dc5e5c889b0a6930f77801917e3d73e9d78",
    "equity-003": "3ebf40b546cd381d5457c72e72a547d6223d75ca7a942086e4eafb8264e29de0",
    "event-001": "b78aff99721708589be56d292e5019b8d793a78b9436fbec83428cd210a35f5c",
    "event-002": "1bfd239924b56e9f9b4a07ae066c46eb1f5ccb64a55cbb5c3a6d97ab6b581f4f",
    "event-003": "6c47fc50eced4e7562af8bee11ddf3443db629c2d2983d5ecddd62c18faa8c96",
    "h2-003": "1d07f39111d8c6eddb2e66d15ee8d668ff82d5826b8fac200eb10149597aedd3",
    "h2-004": "b512146f8c704efc11dc83a57e00dfb28dcf642b071523a8c6b8c9f3f3a1f609",
    "h2-006": "d133fec6a95aa7f8395be442d05a2900a5bae03b86d7621a7edf7e7612e6562b",
    "h2-008": "81c56ff2c061a49882ca646e747a668673ab4362a745f82e6ca682faf6d8460a",
    "h2-009": "84199b4bdf12c8486ce190c73ffa8da2d0df0b4cb50390f966d12aa3f0cdfd35",
    "h2-010": "93486aaa711fd8febfbd8b2882a79b8d6d4aab79ca97216d5034b56c833bc1ce",
    "h2-011": "887a074297dd53d8fb5300573be6e701bc7d1976173d99e008a7a6d62754b8ca",
    "h2-012": "278372d93ea2894654dcd1e29f9041f60a2c0a8bc23009ba52ea561c486aa1d9",
    "portfolio-001": "96fa0052350b62314402ad8ed85baa8b3b1561cf724674c2858b8c47df5acebb",
    "portfolio-003": "f746e37c4480a76115486d7a5df26680ab01747c2e20bd80937018c0b5c4433c",
    "systematic-002": "d316f602292a43d1e56f6a07bbbf13c4a63c267b3b93a36e10ee690932f3ed7e",
    "systematic-003": "9de8280ae89abffcb0c60ede9d639410b923bbae879c3888a70814f9c91d5157",
    "vol-001": "11ca903c8485d12f7e02a1b8bf36ce86c316dd219bb193a1f87287968f10c663",
    "vol-002": "e666de0a42b84a7fc45ec065da82f26a0d76678e58061854f89f9ac03d0e1d4d",
    "vol-003": "df463434edb8d60343bcc0fc2152c22c512bf5884c19d8db703588fff4a55eec"
}
CHECKS = {}


def packet_hash(question, sources):
    payload = json.dumps({"question": question, "sources": sources}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_exact(item) for item in value]
    return value


def reviewed(case_id):
    def decorate(calculate):
        @wraps(calculate)
        def bound(question, sources):
            digest = packet_hash(question, sources)
            if digest != REVIEWED_PACKETS[case_id]:
                raise ValueError("Public source packet differs from reviewed input")
            answers, certificate = calculate()
            if set(answers) != {field["id"] for field in question["output_fields"]}:
                raise ValueError("Independent derivation does not cover every declared field")
            certificate.update({"canonical_public_packet_sha256": digest,
                "exact_results": _exact(answers),
                "scope": "All declared fields under the retained excerpts and synthetic assumptions; assistant arithmetic review, not external revalidation, legal advice or practitioner certification."})
            return {key: (value.numerator if value.denominator == 1 else float(value))
                    if isinstance(value, F) else value for key, value in answers.items()}, _exact(certificate)
        CHECKS[case_id] = bound
        return bound
    return decorate


def adjusted_call(shares, fixed_cash, spot, strike, premium, contracts, multiplier=100):
    """One-underlying adjusted call at expiry, with fixed cash in its basket."""
    basket = shares * spot + fixed_cash
    intrinsic = max(F(0), basket - multiplier * strike)
    return {"basket": basket, "intrinsic": contracts * intrinsic,
            "premium": contracts * multiplier * premium,
            "pnl": contracts * (intrinsic - multiplier * premium),
            "breakeven": (multiplier * (strike + premium) - fixed_cash) / shares,
            "itm_slope": contracts * shares if intrinsic > 0 else 0}


@reviewed("vol-001")
def natwest():
    result = adjusted_call(F(92), F("42.30"), F("5.20"), F(5), F(".17"), 20)
    return {
        "initial_total_cash_determined": False,
        "net_special_dividend_unrounded": 100 * F(".381441"),
        "final_cash_per_contract": F("42.30"), "book_share_delivery": 20 * 92,
        "book_strike_payment": 20 * 5 * 100,
        "basket_value_per_contract": result["basket"], "book_expiry_intrinsic": result["intrinsic"],
        "book_premium": result["premium"], "book_expiry_pnl": result["pnl"],
        "spot_breakeven": result["breakeven"], "book_local_expiry_sensitivity": result["itm_slope"],
    }, {"method": "Final fixed-cash basket and positive expiry payoff, with premium multiplier unchanged.",
        "cash_rounding": "Use the final memo's 42.30 cash, not unrounded special dividend plus separately rounded fractional proceeds.",
        "initial_insufficiency": "Initial memo leaves fractional-share cash undetermined."}


@reviewed("vol-002")
def ge():
    result = adjusted_call(F(12), F("52.22"), F(120), F(14), F(".25"), 50)
    quote = result["basket"] / 100
    return {"initial_final_cash_known": False, "provisional_equivalent_quote": F(120, 8),
        "fixed_cash_per_contract": F("52.22"), "book_share_delivery": 50 * 12,
        "book_fixed_cash_delivery": 50 * F("52.22"), "final_equivalent_quote": quote,
        "quote_revision_at_same_spot": quote - F(120, 8), "book_strike_payment": 50 * 14 * 100,
        "book_expiry_pnl": result["pnl"], "spot_breakeven": result["breakeven"],
        "book_local_expiry_sensitivity": result["itm_slope"],
    }, {"method": "Replace provisional fractional shares with final fixed cash; retain 100 strike/premium multiplier.",
        "fixed_cash_derivative": 0, "initial_insufficiency": "The first two memos do not fix the fractional-share cash."}


@reviewed("vol-003")
def mmm():
    basket = 100 * F(92) + 25 * F(80)
    intrinsic = 40 * max(0, basket - 110 * 100)
    return {"book_mmm_delivery": 40 * 100, "book_solv_delivery": 40 * 25,
        "basket_value_per_contract": basket, "equivalent_underlying_quote": basket / 100,
        "book_strike_payment": 40 * 110 * 100, "book_expiry_intrinsic": intrinsic,
        "book_expiry_pnl": intrinsic - 40 * F("1.2") * 100,
        "stress_pnl_change": 40 * (100 * 1 + 25 * -3),
        "solv_breakeven": (110 * 100 + F("1.2") * 100 - 100 * 92) / 25,
        "book_mmm_local_sensitivity": 4000, "book_solv_local_sensitivity": 1000,
        "settlement_allocation_proves_tax_basis": False,
    }, {"method": "Two-asset physical basket at expiry; unchanged strike multiplier and sunk premium treated separately.",
        "stress_itm_check": basket + 100 - 75 > 11000,
        "allocation_scope": "The 80/20 allocation is solely a clearing-interface rule, not a customer tax-basis rule."}


def rounded_ratio(value, decimals=3):
    """Exact nearest rounding, half up, for the nonnegative collar ratio."""
    scaled = value * 10 ** decimals
    return F((2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator), 10 ** decimals)


def att_ratio(average):
    if average <= 0:
        raise ValueError("Averaging-window price must be positive")
    if average < F("37.411"):
        return F("1.437")
    if average > F("41.349"):
        return F("1.300")
    return rounded_ratio(F("53.75") / average)


@reviewed("event-001")
def att():
    ratios = [att_ratio(F(a)) for a in (36, 40, 44)]
    values = [F("53.75") + ratio * mark for ratio, mark in zip(ratios, (34, 42, 40))]
    cost = F(108) + F(".20")
    return {"a_exchange_ratio": ratios[0], "b_exchange_ratio": ratios[1], "c_exchange_ratio": ratios[2],
        "a_value_per_target": values[0], "b_value_per_target": values[1], "c_value_per_target": values[2],
        "b_net_profit_per_target": values[1] - cost,
        "b_break_even_settlement_price": (cost - F("53.75")) / ratios[1],
        "b_positive_net_profit": values[1] > cost, "b_short_hedge_shares": 1000 * ratios[1],
        "u_unique_value": False, "u_value_lower_bound": F("53.75") + F("1.3") * 40,
        "u_value_upper_bound": F("53.75") + F("1.437") * 40,
    }, {"method": "Use averaging-window price to fix rounded ratio, then separately mark delivered shares.",
        "bound_witness_averages": [36, 44], "unknown_average_ratio_range": [F("1.3"), F("1.437")],
        "rounding_scope": "No half-way ratio occurs at the supplied scenario prices."}


def election_fractions(total, cash, stock, none):
    """Cash fraction per election group; a zero-sized group has fraction zero."""
    if min(total, cash, stock, none) < 0 or total != cash + stock + none:
        raise ValueError("Election counts must be a nonnegative partition")
    quota = F(total, 2)
    if cash >= quota:
        return {"cash": quota / cash if cash else F(0), "stock": F(0), "none": F(0)}
    shortfall = quota - cash
    if shortfall <= none:
        return {"cash": F(1), "stock": F(0), "none": shortfall / none if none else F(0)}
    return {"cash": F(1), "stock": (shortfall - none) / stock, "none": F(1)}


@reviewed("event-002")
def vmware():
    s = election_fractions(10000, 1000, 8000, 1000)["stock"]
    c = election_fractions(10000, 6000, 2000, 2000)["cash"]
    cash, stock = 800 * s * F("142.5"), 800 * (1 - s) * F(".252")
    value, cost = cash + stock * 600, 800 * F("146.5")
    return {"s_stock_election_cash_fraction": s, "s_holder_cash": cash,
        "s_holder_broadcom_shares": stock, "s_holder_settlement_value": value,
        "s_holder_net_profit": value - cost, "s_meets_profit_policy": value - cost >= 1000,
        "s_short_hedge_shares": stock, "s_break_even_broadcom_price": (cost - cash) / stock,
        "c_cash_fulfillment_fraction": c, "c_holder_cash": 1500 * c * F("142.5"),
        "c_holder_broadcom_shares": 1500 * (1 - c) * F(".252"),
        "c_non_election_consideration": "stock", "u_exact_stock_fraction_known": False,
        "u_marked_entitlement_lower_bound": 1000 * (F("142.5") + F(".252") * 600) / 2,
        "u_marked_entitlement_upper_bound": 1000 * F(".252") * 600,
    }, {"method": "Allocate the 50% cash quota across cash, non-election, then stock-election groups.",
        "unknown_stock_fraction_range": [F(1, 2), F(1)],
        "bound_witness_cash_stock_none": [[0, 10000, 0], [5000, 5000, 0]],
        "bound_argument": "Stock-electors receive at least half stock: in the oversubscribed branch, stock fraction equals total/(2*stock elections). All-stock and half-cash elections attain the endpoints."}


def ticking_fee(close):
    tiers = [(date(2023, 2, 22), date(2023, 3, 22), F(".15333333"), F(".0025")),
             (date(2023, 3, 22), date(2023, 4, 22), F(".22333333"), F(".00333333")),
             (date(2023, 4, 22), date(2023, 5, 22), F(".3266667"), F(".00416667"))]
    for start, end, base, daily in tiers:
        if start <= close < end:
            return base + daily * (close - start).days
    raise ValueError("No retained ticking tier covers that close date")


@reviewed("event-003")
def tegna():
    entry = date(2023, 2, 22)
    closes = [date(2023, month, 21) for month in (3, 4, 5)]
    fees = [ticking_fee(day) for day in closes]
    finance = [23 * F(".08") * F((day - entry).days, 365) for day in closes]
    net = [24 + tick - 23 - F(".05") - funding for tick, funding in zip(fees, finance)]
    return {"outside_date_extension_calendar_days": (date(2023, 5, 22) - entry).days,
        "new_outside_date_utc": "2023-05-22T21:00:00Z",
        "march21_ticking_fee": fees[0], "april21_ticking_fee": fees[1], "may21_ticking_fee": fees[2],
        "march21_net_profit": net[0], "april21_to_may21_incremental_net_profit": net[2] - net[1],
        "may21_break_even_success_probability": (23 + F(".05") + finance[2] - 17) / (24 + fees[2] - 17),
        "may22_fee_identifiable_from_supplied_schedule": False,
        "termination_fee_is_direct_per_share_consideration": False,
    }, {"method": "Actual calendar-day tier increments and ACT/365 financing; solve zero expected profit with common success/failure cost.",
        "holding_days": [(day - entry).days for day in closes], "tier_increment_days": [27, 30, 29],
        "failure_net_profit": 17 - 23 - F(".05") - finance[2],
        "scope_limits": "May 22 is excluded from the last supplied tier; the termination fee is payable to the company."}


@reviewed("h2-008")
def jnj():
    accepted = 190955436 - 6529526
    entitlement = 99 * F("8.0324")
    whole = entitlement.numerator // entitlement.denominator
    return {"nonodd_shares_accepted": accepted,
        "nonodd_acceptance_fraction": F(accepted, 793920159),
        "cap_minus_final_ratio": F("8.0549") - F("8.0324"),
        "oddlot_whole_kvue_shares": whole, "oddlot_fractional_interest": entitlement - whole,
        "oddlot_marked_economic_profit": entitlement * 23 - 99 * (166 + 1),
        "actual_fractional_cash_known": False, "launch_only_exact_ratio_known": False,
    }, {"method": "Remove exempt odd lots from both aggregate counts; use full acceptance for the qualifying 99-share holder.",
        "tender_partition_check": 793920159 + 6529526 == 800449685,
        "fraction_scope": "A synthetic mark is not the unknown cash from the agent's later aggregated fractional-share sale."}


def leverage_metrics(debt, cash, cap, ebitda, ceiling):
    permitted = min(cash, cap)
    net = debt - permitted
    return {"cash": permitted, "net": net, "leverage": net / ebitda,
            "capacity": ceiling * ebitda - net, "minimum_ebitda": net / ceiling,
            "decline": ebitda - net / ceiling, "minimum_cash": max(F(0), debt - ceiling * ebitda)}


@reviewed("credit-001")
def arcosa():
    ebitda = F(180 + 60 + 45 + 95 + 25 + 20 + 15 + 25 - 5 - 10 - 30)
    r = leverage_metrics(F(2200), F(200), F(150), ebitda, F(5))
    return {"covenant_ebitda": ebitda, "permitted_cash_netting": r["cash"],
        "net_debt_numerator": r["net"], "maximum_ratio_june_2025": 5,
        "maximum_ratio_september_2025": F("4.5"), "maximum_ratio_march_2026": 4,
        "june_2025_leverage": r["leverage"], "june_2025_extra_debt_capacity": r["capacity"],
        "september_2025_external_paydown_needed": max(0, r["net"] - F("4.5") * ebitda),
        "june_2025_maximum_ebitda_decline": r["decline"],
        "june_2025_minimum_qualifying_cash": r["minimum_cash"],
        "cash_unknown_branch_compliance_identifiable": False,
        "closing_amendments_effective_on_signing_alone": False,
    }, {"method": "Disjoint EBITDA additions/deductions, capped eligible cash, and full-quarter count from stipulated October close.",
        "full_quarter_indices": {"June2025": 2, "September2025": 3, "March2026": 5},
        "cash_unknown_leverage_endpoints": [F(2200, 1) / ebitda, r["leverage"]],
        "effectiveness": "Acquisition amendments require the additional closing conditions; signing alone is insufficient."}


@reviewed("credit-002")
def vestis():
    debt, cash = F(1800 + 100 + 20 + 10), F(130 + 40 + 30)
    r = leverage_metrics(debt, cash, cash, F(400), F("4.5"))
    return {"included_indebtedness": debt, "eligible_cash": cash, "excluded_cash": 25 + 90,
        "net_debt": r["net"], "applicable_maximum_ratio": F("4.5"),
        "base_leverage": r["leverage"], "extra_spent_debt_capacity": r["capacity"],
        "minimum_ebitda_at_boundary": r["minimum_ebitda"], "maximum_ebitda_decline": r["decline"],
        "stressed_leverage": r["net"] / 380, "stress_meets_leverage_condition": r["net"] / 380 <= F("4.5"),
        "interest_coverage_compliance_identifiable": False,
    }, {"method": "Restricted-group debt/cash perimeter; include unpaid hedge termination but exclude ordinary MTM, undrawn amounts and nonrecourse unrestricted-group debt.",
        "excluded_debt_inputs": [40, 300, 50, 400],
        "interest_insufficiency": "No interest expense is supplied; principal alone does not identify the separate 2x interest test."}


def springing_compliance(exposure, commitments, net_debt, ebitda):
    active = exposure > F(".35") * commitments
    return active, not active or net_debt <= F("6.75") * ebitda


@reviewed("credit-003")
def dnb():
    threshold = F(".35") * 850
    return {"activation_exposure_threshold": threshold,
        "base_trigger_active": springing_compliance(F("297.5"), 850, 2000, 300)[0],
        "draw25_trigger_active": springing_compliance(F("322.5"), 850, 2025, 300)[0],
        "draw25_first_lien_net_leverage": F(2025, 300),
        "maximum_additional_spent_draw": F("6.75") * 300 - 2000,
        "stress_draw20_leverage": F(2020, 280),
        "stress_minimum_revolver_repayment_to_deactivate": F(290) + F("27.5") - threshold,
        "stress_term_debt_repayment_to_meet_active_ratio": F(2020) - F("6.75") * 280,
        "unknown_lc_trigger_identifiable": False,
    }, {"method": "Strict exposure activation and inclusive leverage ceiling, with no double count of already-included revolver debt.",
        "unknown_lc_exposure_endpoints": [290, 310],
        "draw_bound": "At baseline exposure equality, every positive draw activates the test; its debt bound is 6.75*300-2000."}


def generator_margin(mw, hours, availability, heat, node, hub_gas, basis, om, hedge_mw, fixed, hub_power):
    generation = mw * hours * availability
    fuel = generation * heat
    swap_volume = hedge_mw * hours
    hedge = swap_volume * (fixed - hub_power)
    physical = generation * (node - om) - fuel * (hub_gas + basis)
    return {"generation": generation, "fuel": fuel, "physical": physical, "hedge": hedge,
            "total": physical + hedge, "hub_sensitivity": generation - swap_volume,
            "basis_sensitivity": -fuel,
            "break_even_basis": (generation * (node - om) + hedge) / fuel - hub_gas}


@reviewed("energy-003")
def odessa():
    r = generator_margin(F(1054), 120, F(".8"), F("7.2"), 65, 3, F(".5"), 3, 600, 60, 68)
    return {"scenario_generation": r["generation"], "scenario_fuel_burn": r["fuel"],
        "scenario_physical_margin": r["physical"], "scenario_hedge_cash_settlement": r["hedge"],
        "scenario_total_margin": r["total"], "hub_price_sensitivity": r["hub_sensitivity"],
        "gas_basis_sensitivity": r["basis_sensitivity"], "break_even_delivered_gas_basis": r["break_even_basis"],
        "total_margin_identifiable_without_gas_basis": False, "actual_odessa_hedge_terms_identifiable": False,
    }, {"method": "Physical net output times availability; a separate all-hours financial swap; solve the affine delivered-gas margin.",
        "scope_limits": "Fleet hedge percentages do not specify plant hedge terms; 7.2 is a sensitivity convention, not a measured plant heat rate."}


@reviewed("equity-001")
def altice():
    ebitda, operations, capex = 3608890, 1826398, 1704811
    cash_fcf, operating_fcf = operations - capex, ebitda - capex
    return {"adjusted_ebitda": ebitda, "operating_cash_flow": operations, "cash_capex": capex,
        "operating_free_cash_flow": operating_fcf, "free_cash_flow": cash_fcf,
        "metric_gap": operating_fcf - cash_fcf, "cash_flow_starting_metric": "free_cash_flow",
        "meets_cash_hurdle": cash_fcf >= 150000, "minimum_capex_reduction": max(0, 150000 - cash_fcf),
        "replacement_capex_basis": "accrued_capital_expenditures", "replacement_computable": False,
    }, {"method": "Use FY2023 columns shared by release and annual report; separately subtract cash capex from earnings and from operating cash.",
        "cross_document_operands": [ebitda, operations, capex],
        "cash_to_accrual_limit": "The three permitted inputs do not include accrued capex or its payable/timing bridge."}


@reviewed("equity-002")
def intel():
    old, recast, current = F(15521), F(12635), F(12817)
    return {"dcai_2023_basis_difference": old - recast,
        "dcai_comparable_growth_2024_pct": 100 * (current / recast - 1),
        "dcai_old_new_growth_comparable": False,
        "recast_2023_consolidated_revenue": 47667 + 18910 + 5608 - 17957,
        "revenue_overstatement_without_eliminations_2024": 48949 + 17543 + 3824 - 53101,
        "foundry_comparable_growth_2024_pct": 100 * (F(17543, 18910) - 1),
        "disclosed_external_revenue_share": F(100 * 385, 17543),
        "dcai_recast_operating_margin_2023_pct": F(100 * 1620, 12635),
        "segment_change_revises_prior_consolidated_statements": False,
        "q1_2023_recast_dcai_revenue_identifiable": False,
    }, {"method": "Comparable segment vintages and parent-level consolidation eliminate internal sales without counting child segments twice.",
        "product_sum_2023": 29258 + 12635 + 5774, "product_sum_2024": 30290 + 12817 + 5842,
        "identification_limits": "External revenue is explicitly a subset; the annual sum cannot uniquely allocate quarters; the recast leaves prior consolidated statements unchanged."}


@reviewed("equity-003")
def microsoft():
    old, current, impact = F(211915), F(245122), F(5729)
    operating_impact = F(-1362)
    return {"reported_revenue_increase": current - old, "reported_growth_pct": 100 * (current / old - 1),
        "growth_after_removing_disclosed_net_acquisition_impact_pct": 100 * ((current - impact) / old - 1),
        "pro_forma_combined_growth_pct": 100 * (F(247442, 219790) - 1),
        "pro_forma_minus_reported_revenue_2024": 247442 - current,
        "pro_forma_minus_reported_revenue_2023": 219790 - old,
        "disclosed_net_impact_operating_margin_pct": 100 * operating_impact / impact,
        "hypothetical_cost_synergy_required": F(".05") * impact - (operating_impact - 400),
        "residual_growth_is_identified_organic_growth": False,
        "pro_forma_results_are_actual_combined_results": False,
    }, {"method": "Keep reported, net acquisition impact and common-date pro forma denominators distinct; solve an operating-income target without annualizing.",
        "interpretation": "The net impact includes third-party to first-party reclassification; subtraction does not isolate organic growth or standalone target results."}


@reviewed("h2-003")
def danaher():
    sales_growth = 100 * (F(23890, 26643) - 1)
    current_fcf, comparable_fcf = F(6490 - 1383 + 12), F(7613 - 1118 + 9)
    transfers = 2600 - 427
    return {"comparable_sales_growth_2023_pct": sales_growth,
        "stale_sales_denominator_growth_error_pp": 100 * (F(23890, 31471) - 1) - sales_growth,
        "fy2022_operating_cash_removed_from_continuing": 8519 - 7613,
        "continuing_free_cash_flow_2023": current_fcf,
        "continuing_free_cash_flow_growth_pct": 100 * (current_fcf / comparable_fcf - 1),
        "net_cash_received_in_separation": transfers,
        "financing_cash_flow_without_separation_transfers": -273 - transfers,
        "veralto_debt_remains_danaher_obligation_after_separation": False,
        "no_separation_full_year_cash_flow_identified": False,
    }, {"method": "Continuing-perimeter recast and cash-flow bridge; remove the two financing transfers from consolidated financing cash, not from recurring operating cash.",
        "financing_reconciliation": {"continuing": 154, "cash_to_separated_entity": -427, "total": -273},
        "limits": "The guarantee terminated at distribution. Missing fourth-quarter and counterfactual operating effects prevent a no-separation full-year cash estimate."}


def coverage_repayment(ebitda, interest, liquidity, floor, target, interest_saving):
    required = max(F(0), (interest - ebitda / target) / interest_saving)
    maximum = max(F(0), liquidity - floor)
    remaining_interest = interest - maximum * interest_saving
    return {"required": required, "maximum": maximum, "remaining_interest": remaining_interest,
            "coverage": ebitda / remaining_interest,
            "ebitda_gap": max(F(0), target * remaining_interest - ebitda), "feasible": required <= maximum}


@reviewed("h2-004")
def carnival():
    r = coverage_repayment(F(3000), F(1200), F(3000), F(1500), F(3), F(".1"))
    return {"export_credit_test_precedes_new_revolver_months": 8 - 5,
        "november2025_maximum_net_interest": F(3000) / F("2.5"),
        "february2026_maximum_net_interest": F(3000, 3),
        "repayment_required_for_feb2026_coverage": r["required"],
        "maximum_repayment_from_liquidity": r["maximum"], "coverage_after_maximum_liquidity_repayment": r["coverage"],
        "minimum_additional_ebitda_after_maximum_repayment": r["ebitda_gap"],
        "repayment_alone_feasible_under_both_tests": r["feasible"],
        "all_contractual_permissions_and_covenants_established": False,
    }, {"method": "Intersect the coverage-implied minimum repayment with the cash-liquidity maximum; retain the later 3x schedule.",
        "schedule": {"export_start": "2024-05-31", "revolver_start": "2024-08-31", "november2025": F("2.5"), "february2026": 3},
        "unestablished_other_constraints": "Contract-defined capital/reserves and secured-asset restrictions lack required inputs."}


@reviewed("h2-006")
def occidental():
    unit_face_millions = F(100000, 1000000)
    redeemed = (100000 - 84897) * unit_face_millions
    remaining = 84897 * unit_face_millions
    synthetic_face = 1000 * unit_face_millions
    return {"preferred_face_redeemed_2023": redeemed, "preferred_face_remaining_end2023": remaining,
        "ordinary_annual_dividend_run_rate": remaining * F(".08"),
        "cash_dividend_reduction_2024_vs2023": 762 - 679,
        "common_income_charge_reduction_2024_vs2023": 923 - 679,
        "redemption_related_charge_beyond_premium_2023": 923 - 736 - 151,
        "synthetic_redemption_cash_outlay": synthetic_face * F("1.1"),
        "premium_only_recovery_years": synthetic_face * F(".1") / (synthetic_face * F(".08")),
        "voluntary_2025_redemption_permitted": False,
        "year_end_common_share_count_identifies_trigger": False,
    }, {"method": "Exact share-face rollforward distinguishes cash dividends, accrued dividends and common-income charges; recover only the redemption premium from ordinary savings.",
        "face_rollforward": [100000 * unit_face_millions, redeemed, remaining],
        "cash_vs_accrued_2023": [762, 736],
        "limits": "Voluntary redemption is barred before August 2029; annual totals and an endpoint share count do not determine rolling per-share distributions."}


def basis_hedge(volume, size, physical, entry, reference_a, reference_b, fee):
    contracts = volume / size
    if contracts.denominator != 1:
        raise ValueError("Quantity is not a whole number of contracts")
    settlement = reference_a - reference_b
    gain = volume * (entry - settlement)
    return contracts, settlement, gain, volume * physical + gain - contracts * fee


@reviewed("h2-009")
def waha():
    contracts, settlement, gain, cash = basis_hedge(F(62500), 2500, F("-.1"), F("-.6"), F("-.3"), F("2.2"), 4)
    return {"hedge_contracts": contracts, "final_basis_settlement": settlement,
        "basis_hedge_gain": gain, "net_physical_and_hedge_cash": cash,
        "residual_henry_hub_sensitivity": 62500, "residual_monthly_basis_sensitivity": 62500 - 62500,
        "combined_cash_known_without_physical_price": False, "negative_day_share_determines_annual_revenue": False,
    }, {"method": "Short settlement is entry minus (A-B); physical daily-to-monthly shape is not hedged by monthly basis.",
        "physical_receipts": F(62500) * F("-.1"), "hedge_fees": contracts * 4,
        "remaining_exposures": "Outright Henry Hub and physical-versus-monthly shape remain; negative-day frequency lacks price magnitudes and sales volumes."}


@reviewed("h2-010")
def pjm():
    days = (date(2026, 5, 31) - date(2025, 6, 1)).days + 1
    ratings, future = (79, 14, 59), (78, 13, 57)
    prices = (F("444.26"), F("466.35"), F("269.92"))
    revenue = days * sum(mw * price for mw, price in zip(ratings, prices))
    later = days * sum(mw * price for mw, price in zip(future, prices))
    return {"delivery_days": days, "portfolio_proxy_ucap": sum(ratings),
        "annual_gross_capacity_revenue": revenue,
        "locational_revenue_uplift": revenue - days * sum(ratings) * prices[2],
        "future_rating_vintage_revenue_change": later - revenue,
        "storage_break_even_capacity_price": F(6000000, 59 * days),
        "actual_resource_revenue_identified": False, "next_year_auction_prices_identified": False,
    }, {"method": "Inclusive delivery days times class-proxy accredited MW times the matching location price; change only the rating vintage in the counterfactual.",
        "current_proxy_mw": ratings, "future_proxy_mw": future,
        "limits": "Class ratings do not identify specific awards; updated ratings contain no next-year auction prices."}


def treasury_bridge(deposits, insured_limit, outside, sunday, monday, loan, rate):
    insured = min(deposits, insured_limit)
    bridge = max(F(0), sunday - outside)
    residual = outside + bridge - sunday
    due = monday + loan + bridge * (1 + rate)
    return insured, bridge, max(F(0), due - residual - insured), residual + deposits - due


@reviewed("h2-011")
def svb():
    deposits = F(600000 + 400000)
    insured, bridge, shortfall, surplus = treasury_bridge(deposits, F(250000), F(200000), F(250000), F(750000), F(100000), F(".01"))
    return {"insured_corporate_deposits": insured, "initial_uninsured_balance": deposits - insured,
        "initial_guaranteed_monday_access": insured, "minimum_sunday_bridge": bridge,
        "initial_monday_guaranteed_funding_shortfall": shortfall, "later_monday_cash_surplus": surplus,
        "initial_uninsured_recovery_amount_known": False, "shareholders_protected_with_depositors": False,
    }, {"method": "Aggregate unincorporated desks into one insured corporation; fund Sunday first, then repay principal plus fee alongside Monday obligations.",
        "monday_obligation": F(750000 + 100000) + bridge * F("1.01"),
        "guaranteed_bound": "An unspecified later uninsured advance is not guaranteed Monday cash; full later access does not forgive the ordinary loan."}


@reviewed("h2-012")
def credit_suisse():
    loss = F(1000000) * F(".8") + F(1500000) * F(".7")
    delivery = F(22480) / F("22.48")
    equity = delivery * 18 - 22480 * 2
    short_gain = 1000 * (20 - 18)
    tier2 = 1000000 * (F(".92") - F(".95"))
    return {"shared_at1_mark_loss": loss, "contractual_ubs_share_delivery": delivery,
        "credit_suisse_equity_mark_pnl": equity, "eventual_net_ubs_share_exposure": delivery - 1000,
        "aggregate_portfolio_mark_pnl": -loss + equity + short_gain + tier2,
        "immediate_cash_financing_need": max(F(0), loss * F(".5") - 120000 - short_gain),
        "tier2_zero_required_by_write_down": False, "aggregate_pnl_known_without_tier2_mark": False,
        "ultimate_at1_recovery_identified": False,
    }, {"method": "Look through the wrapper once; mark AT1, equity entitlement, independent Tier2 and existing short against their stated baselines; segregate later share delivery from current cash.",
        "immediate_cash_resources": 120000 + short_gain, "margin_call": loss / 2,
        "limits": "Tier2 is outside the write-down instruction but still requires its own market mark. These event documents do not settle subsequent legal recoveries."}


PLAN_ORDER = ("NONE", "A", "B", "C", "AB", "AC", "BC", "ABC")


def funding_plans(base, early, hedges, fees, margins):
    """Enumerate all eight subsets with terminal P&L and gross staged cash."""
    result = {}
    for code in PLAN_ORDER:
        letters = () if code == "NONE" else tuple(code)
        fee = sum((fees[key] for key in letters), F(0))
        locked = fee + sum((margins[key] for key in letters), F(0))
        terminal, outflow = [], []
        for scenario, value in enumerate(base):
            terminal.append(value + sum((hedges[key][scenario] for key in letters), F(0)) - fee)
            outflow.append(early[scenario] + sum((max(F(0), -hedges[key][scenario]) for key in letters), F(0)))
        required = locked + max([F(0)] + outflow)
        result[code] = {"fees": fee, "t0_locked": locked, "t1_outflow": outflow,
                        "terminal_pnl": terminal, "worst": min(terminal), "required_cash": required}
    return result


def select_plan(rows, budget):
    feasible = [code for code in PLAN_ORDER if rows[code]["required_cash"] <= budget]
    if not feasible:
        return {"feasible": [], "optima": []}
    best = max(rows[code]["worst"] for code in feasible)
    optima = [code for code in feasible if rows[code]["worst"] == best]
    return {"feasible": feasible, "optima": optima, "worst": best}


def funding_result(rows, budget):
    selected = select_plan(rows, budget)
    if len(selected["optima"]) != 1:
        raise ValueError("Reviewed scenario requires a unique feasible optimizer")
    chosen = selected["optima"][0]
    # Feasibility can change only at one of these finite thresholds; the
    # objective is independent of starting cash between thresholds.
    thresholds = sorted({row["required_cash"] for row in rows.values() if row["required_cash"] > budget})
    changes = [bound for bound in thresholds if select_plan(rows, bound)["optima"] != selected["optima"]]
    if not changes:
        raise ValueError("No optimizer change above this budget")
    return {"feasible_plans": "|".join(selected["feasible"]), "optimal_plan": chosen,
            "optimal_worst_pnl": rows[chosen]["worst"],
            "optimal_minimum_cash": budget - rows[chosen]["required_cash"],
            "first_optimizer_change_budget": changes[0]}


def _number_string(value):
    value = F(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("Scenario string requires a terminating decimal")
    places = max(twos, fives)
    scaled = value.numerator * (10 ** places // value.denominator)
    if not places:
        return str(scaled)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).zfill(places + 1)
    return (sign + digits[:-places] + "." + digits[-places:]).rstrip("0").rstrip(".")


def _scenario_string(names, values):
    return ";".join(f"{name}={_number_string(value)}" for name, value in zip(names, values))


def separation_book(c_margin=F(1000)):
    participation = F(500000000) * F(".0002")
    dk, dklg, credit_changes = (-4, 3, -1), (-6, 4, 2), (F("-.04"), F(".01"), F("-.01"))
    baseline = max(0, 100 * 50 + 25 * 15 - 100 * 40)
    option, base = [], []
    for k, klg, loan in zip(dk, dklg, credit_changes):
        changed = max(0, 100 * (50 + k) + 25 * (15 + klg) - 100 * 40)
        option.append(-30 * (changed - baseline))
        base.append(4000 * k + 1000 * klg + option[-1] + participation * loan)
    hedges = {"A": [-500 * delta for delta in dk], "B": [-250 * delta for delta in dklg],
              "C": [-F(1, 2) * participation * delta for delta in credit_changes]}
    rows = funding_plans(base, [max(0, -value) for value in option], hedges,
                         {"A": 100, "B": 100, "C": 150}, {"A": 2500, "B": 2000, "C": c_margin})
    return participation, base, rows


@reviewed("portfolio-001")
def separation_portfolio():
    participation, base, rows = separation_book()
    result = funding_result(rows, F(18000))
    witnesses = {str(margin): select_plan(separation_book(F(margin))[2], 18000)["optima"] for margin in (0, 5000)}
    result.update({"financing_ledger_difference": 664 - 663,
        "credit_participation_principal": participation,
        "net_share_equivalents": _scenario_string(("K", "KLG"), (4000 - 30 * 100, 1000 - 30 * 25)),
        "base_scenario_pnl": _scenario_string(("S1", "S2", "S3"), base),
        "missing_margin_optimizer_status": "insufficient_data" if len({tuple(v) for v in witnesses.values()}) > 1 else "determined"})
    return result, {"method": "Positive-part adjusted call payoff plus share and debt pods; exhaustive eight-plan minimax with separate fee, margin and gross T1 requirements.",
        "all_plan_certificates": rows, "missing_margin_witness_optima": witnesses,
        "output_units": {"financing_ledger_difference": "USD_million", "credit_participation_principal": "USD"},
        "audit_correction": "After reference comparison, financing_ledger_difference was converted from the initially derived USD amount to USD_million as explicitly required by the public field schema. The economic difference is unchanged.",
        "ledger_scope": "September borrowings less distribution is a transaction-ledger difference, not actual cash or post-October revolver debt."}


def freeze_book(freeze_dispatch=F(40)):
    shock = F("23.86") - F("3.86")
    hub = (shock, F(-2), F(5))
    basis = (10, 0, 25)
    power = (250, 80, 180)
    production = (500, 1000, 750)
    dispatch = (freeze_dispatch, F(100), F(60))
    base_generator = 10000 - 100 * 7 * F("3.86")
    base, early = [], []
    for dh, db, p, output, generated in zip(hub, basis, power, production, dispatch):
        local = F("3.86") + dh + db
        producer = output * local - 1000 * F("3.86")
        replacement = (100 - generated) * p
        generator = 10000 - generated * 7 * local - replacement - base_generator
        existing = -300 * dh
        base.append(producer + generator + existing)
        early.append(replacement + max(F(0), -existing))
    hedges = {"A": [300 * change for change in hub],
              "B": [40 * max(0, price - 100) for price in power], "C": [200 * change for change in basis]}
    rows = funding_plans(base, early, hedges, {"A": 100, "B": 500, "C": 200}, {"A": 1000, "B": 0, "C": 1500})
    return shock, base, early, rows


@reviewed("portfolio-003")
def freeze_portfolio():
    shock, base, early, rows = freeze_book()
    result = funding_result(rows, F(23200))
    witnesses = {str(dispatch): select_plan(freeze_book(F(dispatch))[3], 23200)["optima"] for dispatch in (40, 100)}
    result.update({"henry_hub_shock": shock, "freeze_generator_fuel": 40 * 7,
        "base_scenario_pnl": _scenario_string(("F", "R", "B"), base), "freeze_base_early_outflow": early[0],
        "missing_dispatch_optimizer_status": "insufficient_data" if len({tuple(v) for v in witnesses.values()}) > 1 else "determined"})
    return result, {"method": "Reconstruct producer, generator replacement-power and short-gas pods; separate gross early obligations from terminal receipts; enumerate every hedge subset.",
        "all_plan_certificates": rows, "missing_dispatch_witness_optima": witnesses,
        "unit_scope": "The source price is per MMBtu; all synthetic volumes are period energy, not capacity or cubic feet."}


def vintage_choice(snapshot, ready, original_margin, corrected_margin, peer_margin):
    margin = corrected_margin if snapshot >= ready else original_margin
    return "lyft" if margin > peer_margin else "peer"


def fixed_round_replay(choices, returns, notional=F(10000), cash=F(12000), cost=F(".001")):
    ledger = []
    initial = cash
    for choice, rates in zip(choices, returns):
        entry_cost = notional * cost
        if cash < notional + entry_cost:
            raise ValueError("Insufficient cash for fixed round entry plus its cost")
        proceeds = notional * (1 + rates[choice])
        exit_cost = proceeds * cost
        cash = cash - notional - entry_cost + proceeds - exit_cost
        ledger.append({"asset": choice, "entry_notional": notional, "exit_notional": proceeds,
                       "entry_cost": entry_cost, "exit_cost": exit_cost, "ending_cash": cash})
    return cash - initial, ledger


@reviewed("systematic-002")
def lyft():
    original, corrected = F("1.6") + F(500, 100), F("1.6") + F(50, 100)
    ready = datetime(2024, 2, 14, 21, 45, 30, tzinfo=timezone.utc)
    select = lambda snapshot: vintage_choice(snapshot, ready, original, corrected, F(3))
    snapshots = [datetime(2024, 2, day, 22, tzinfo=timezone.utc) for day in (13, 14)]
    opens = [datetime(2024, 2, day, 14, 30, tzinfo=timezone.utc) for day in (14, 15, 16)]
    executions = [(next(i for i, op in enumerate(opens) if op > snap)) for snap in snapshots]
    returns = [{"lyft": F("-.08"), "peer": F(".01")}, {"lyft": F(".02"), "peer": F(".01")}]
    choices = [select(snapshot) for snapshot in snapshots]
    valid, ledger = fixed_round_replay(choices, [returns[i] for i in executions])
    backfill, bad_ledger = fixed_round_replay([select(ready)] * 2, returns)
    return {"original_projected_margin": original, "corrected_projected_margin": corrected,
        "original_modeled_ebitda": 16000 * original / 100, "corrected_modeled_ebitda": 16000 * corrected / 100,
        "before_correction_ready_choice": select(ready - timedelta(seconds=1)),
        "at_correction_ready_choice": select(ready), "valid_toy_net_profit": valid,
        "backfill_net_profit_inflation": backfill - valid,
        "actual_correction_213000_eligibility": "not_established", "empirical_performance_status": "not_established",
    }, {"method": "Add basis-point guidance to the exact synthetic base; preserve decision vintages, inclusive readiness and next-open fixed-notional execution with entry/exit costs.",
        "valid_ledger": ledger, "backfilled_ledger": bad_ledger,
        "execution_intervals": [[opens[i].isoformat(), opens[i + 1].isoformat()] for i in executions],
        "availability_scope": "Synthetic vendor times and original accession metadata do not establish the correction's actual public-access time or actual performance."}


@reviewed("systematic-003")
def meta():
    proceeds, dividend, exit_proceeds = F(60 * 102), F(60), F(80 * 20)
    cost = proceeds * F(".001")
    terminal = proceeds + dividend + exit_proceeds - cost
    return {"rename_required_turnover": 0, "meta_total_return": F(102 + 1 - 100, 100),
        "exiting_security_gross_profit": 80 * (20 - 50), "terminal_liquidation_cost": cost,
        "valid_terminal_equity": terminal, "valid_net_profit": terminal - 10000,
        "naive_ticker_join_meta_return": F(102 + 1 - 10, 10),
        "exit_proceeds_break_even": (10000 - proceeds - dividend + cost) / 80,
        "ticker_alone_unique_identity": False, "empirical_total_return_status": "not_established",
    }, {"method": "Carry permanent-security holdings through the rename, retain dividend cash and the exiting member, and charge only the stipulated terminal sale.",
        "cash_ledger": {"sale_proceeds": proceeds, "dividend_cash": dividend, "exit_cash": exit_proceeds, "sale_fee": cost},
        "identity_scope": "Unchanged Class A security/CUSIP requires no turnover. A ticker collision cannot merge distinct issuer/venue identities.",
        "empirical_limit": "Unadjusted rows and a rename notice do not replace missing distributions, exit recovery or the historical universe."}
