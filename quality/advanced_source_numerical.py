"""Independent reconciliations of four retained multi-document source cases.

Only public questions and retained excerpts enter these calculations. Parsed
table figures are treated as exact for the stipulated exercise. This module
neither imports stored answers nor validates external documents or tax advice.
"""
from datetime import datetime
from fractions import Fraction as F
import re


def _one(pattern, text):
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches or len(set(matches)) != 1:
        raise ValueError("Missing or conflicting reviewed source operand")
    return matches[0]


def _number(raw):
    if raw == "—":
        return F(0)
    negative = raw.startswith("(")
    value = F(raw.strip("()").replace(",", ""))
    return -value if negative else value


def _row(text, label):
    tail = _one(r"^\s*" + re.escape(label) + r"[ \t.$]*(?=[\d(—+-])([^\n]+)$", text)
    return [_number(value) for value in re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|—", tail)]


def _flat(text):
    return " ".join(text.split())


def _require(text, *patterns):
    if any(re.search(pattern, _flat(text), re.I) is None for pattern in patterns):
        raise ValueError("Reviewed interpretation or column contract differs")


def _finish(answers, operands, method, **certificates):
    def encode(value):
        if isinstance(value, F):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        return value
    return ({k: float(v) if isinstance(v, F) else v for k, v in answers.items()}, {
        "method": method, "operands": encode(operands), **encode(certificates),
        "scope": "All declared fields under the retained excerpts and stipulated branches; no external source revalidation, citation-grader, tax-advice or practitioner certification.",
    })


def paypal(question, sources):
    old, new = sources["h2a-paypal-2023"], sources["h2a-paypal-2024"]
    _require(old, r"do not include stock-based compensation")
    _require(new, r"now include the impact of stock-based compensation", r"recast")
    _require(question["prompt"], r"first-nine-month", r"inflow positive|cash-flow contribution")
    annual_old = old.split("FY'23 Financial Results", 1)[1].split("\f", 1)[0].split("Non-GAAP", 1)[1]
    annual_new = new.split("FY'24 Financial Results", 1)[1].split("\f", 1)[0].split("Non-GAAP", 1)[1]
    original_eps = _row(annual_old, "Earnings per diluted share")[0]
    current_eps, comparable_eps = _row(annual_new, "Earnings per diluted share")[:2]
    original_income = _row(annual_old, "Net income")[0]
    comparable_income = _row(annual_new, "Net income")[1]
    cash = new.split("Reconciliation of Operating Cash Flow", 1)[1]
    _require(cash, r"2024\s+2023\s+2024\s+2023")
    fcf = _row(cash, "Free cash flow")
    adjusted = _row(cash, "Adjusted free cash flow")
    if len(fcf) != 4 or len(adjusted) != 4:
        raise ValueError("Cash-flow table shape differs")
    operations = _row(cash, "Net cash provided by operating activities")
    capex = _row(cash, "Less: Purchases of property and equipment")
    timing_adjustment = _row(cash, "receivables")
    if any(len(row) != 4 for row in (operations, capex, timing_adjustment)) or any(
        op + cx != free or free + adj != result
        for op, cx, free, adj, result in zip(operations, capex, fcf, timing_adjustment, adjusted)
    ):
        raise ValueError("Cash-flow reconciliation does not balance")
    old_cash = old.split("Reconciliation of Operating Cash Flow", 1)[1]
    old_fcf, old_adjusted = _row(old_cash, "Free cash flow"), _row(old_cash, "Adjusted free cash flow")
    if [old_fcf[0], old_fcf[2]] != [fcf[1], fcf[3]] or [old_adjusted[0], old_adjusted[2]] != [adjusted[1], adjusted[3]]:
        raise ValueError("Prior-year cash-flow vintages do not reconcile")
    # The adjustment is added to reported FCF; its underlying cash contribution
    # has the opposite sign. FY includes Q4, so nine months is FY minus Q4.
    quarter_contribution = fcf[1] - adjusted[1]
    annual_contribution = fcf[3] - adjusted[3]
    comparable_growth = 100 * (current_eps / comparable_eps - 1)
    old_growth = 100 * (current_eps / original_eps - 1)
    return _finish({
        "non_gaap_eps_growth_comparable_pct": comparable_growth,
        "old_basis_eps_growth_error_pp": old_growth - comparable_growth,
        "prior_year_non_gaap_income_recast_reduction": original_income - comparable_income,
        "fy2024_free_cash_flow_growth_pct": 100 * (fcf[2] / fcf[3] - 1),
        "fy2024_adjusted_free_cash_flow_growth_pct": 100 * (adjusted[2] / adjusted[3] - 1),
        "q42023_positive_bnpl_timing_contribution": quarter_contribution,
        "first_nine_months_2023_bnpl_timing_contribution": annual_contribution - quarter_contribution,
        "q42023_timing_share_of_free_cash_flow_pct": 100 * quarter_contribution / fcf[1],
        "old_and_new_non_gaap_eps_directly_comparable": False,
        "sold_bnpl_credit_loss_rate_identified": False,
    }, {"original_eps": original_eps, "comparable_eps": comparable_eps, "current_eps": current_eps,
        "old_and_recast_income": [original_income, comparable_income], "free_cash_flow": fcf,
        "adjusted_free_cash_flow": adjusted, "timing_adjustment": timing_adjustment},
        "Exact policy-consistent EPS denominators; annual/Q4 cash bridges with opposite adjustment/contribution signs.",
        annual_timing_contribution=annual_contribution,
        credit_identification="A timing cash bridge supplies no cohort balances, defaults or recoveries; it cannot identify a cohort loss rate.")


def cisco(question, sources):
    old, new = sources["h2a-cisco-2024"], sources["h2a-cisco-2025"]
    income = new.split("See Notes", 1)[0]
    _require(income, r"July 26, 2025\s+July 27, 2024\s+July 29, 2023")
    revenue = _row(income, "Total revenue")
    net = _row(income, "NET INCOME")
    pretax = _row(income, "INCOME BEFORE PROVISION FOR INCOME TAXES")
    tax = _row(income, "Provision for income taxes")
    if any(len(row) != 3 for row in (revenue, net, pretax, tax)):
        raise ValueError("Income-statement column shape differs")
    if any(p - t != n for p, t, n in zip(pretax, tax, net)):
        raise ValueError("Tax and pretax income do not reconcile to net income")
    proforma = old.split("Pro forma Financial Information", 1)[1].split("Other Acquisitions", 1)[0]
    _require(proforma, r"July 27, 2024\s+July 29, 2023", r"not indicative", r"only the impacts of the Splunk")
    pf_revenue, pf_net = _row(proforma, "Total revenue")[0], _row(proforma, "Net income")[0]
    cash_paid = _row(old, "Cash paid for outstanding Splunk common stock")[0]
    purchase = _row(old, "Total purchase consideration")[0]
    acquired_cash = _row(old, "Cash and cash equivalents")[0]
    awards = _row(old, "Fair value of converted Splunk equity awards attributable to pre-acquisition services")[0]
    relationships = _row(old, "Settlement of pre-existing relationships")[0]
    if cash_paid + awards + relationships != purchase:
        raise ValueError("Purchase-consideration bridge does not reconcile")
    reported_growth = 100 * (revenue[0] / revenue[1] - 1)
    comparator_growth = 100 * (revenue[0] / pf_revenue - 1)
    pretax_decline, tax_reduction = pretax[1] - pretax[0], tax[1] - tax[0]
    return _finish({
        "reported_revenue_growth_2025_pct": reported_growth,
        "revenue_change_vs_2024_splunk_proforma_pct": comparator_growth,
        "reported_minus_proforma_comparator_growth_pp": reported_growth - comparator_growth,
        "fy2024_proforma_minus_reported_net_income": pf_net - net[1],
        "reported_net_income_growth_2025_pct": 100 * (net[0] / net[1] - 1),
        "pretax_decline_offset_by_tax_reduction_pct": 100 * tax_reduction / pretax_decline,
        "shareholder_cash_consideration_net_of_acquired_cash": cash_paid - acquired_cash,
        "purchase_consideration_exceeds_common_share_cash": purchase - cash_paid,
        "comparator_identifies_organic_growth": False,
        "proforma_net_income_is_realized_combined_income": False,
    }, {"reported_revenue": revenue, "reported_net_income": net, "pretax_income": pretax,
        "tax_provision": tax, "proforma_revenue": pf_revenue, "proforma_net_income": pf_net,
        "common_share_cash": cash_paid, "acquired_cash": acquired_cash, "converted_awards": awards,
        "preexisting_relationships": relationships, "purchase_consideration": purchase},
        "Exact acquisition-perimeter comparisons, a pretax/tax/net-income identity and common-share cash versus accounting consideration.",
        pretax_decline=pretax_decline, tax_reduction=tax_reduction,
        net_income_change=-pretax_decline + tax_reduction,
        interpretation="The pro-forma comparator is hypothetical and does not supply a complete acquisition/FX organic-growth bridge.")


def _date(value):
    return datetime.strptime(value, "%B %d, %Y").date()


def _interpolate(stock_price, effective_date, prices, rows):
    """Bilinear interpolation in contractual price and actual calendar days."""
    dates = sorted(rows)
    if not prices[0] <= stock_price <= prices[-1] or not dates[0] <= effective_date <= dates[-1]:
        raise ValueError("Interpolation requires dates and prices within the retained table")
    lower = max(i for i, price in enumerate(prices) if price <= stock_price)
    upper = min(i for i, price in enumerate(prices) if price >= stock_price)
    earlier = max(day for day in dates if day <= effective_date)
    later = min(day for day in dates if day >= effective_date)
    price_weight = F(0) if lower == upper else (stock_price - prices[lower]) / (prices[upper] - prices[lower])
    date_weight = F(0) if earlier == later else F((effective_date - earlier).days, (later - earlier).days)
    rate_at_earlier = rows[earlier][lower] + price_weight * (rows[earlier][upper] - rows[earlier][lower])
    rate_at_later = rows[later][lower] + price_weight * (rows[later][upper] - rows[later][lower])
    return rate_at_earlier + date_weight * (rate_at_later - rate_at_earlier), {
        "price_weight": price_weight, "date_weight": date_weight,
        "days_elapsed": (effective_date - earlier).days, "days_in_interval": (later - earlier).days,
        "earlier_price_interpolated_rate": rate_at_earlier, "later_price_interpolated_rate": rate_at_later,
    }


def boeing(question, sources):
    terms, pricing, prompt = sources["h2a-boeing-terms"], sources["h2a-boeing-pricing"], question["prompt"]
    flat = _flat(terms)
    denominator = int(_one(r"1/(\d+)th", flat))
    position = _number(_one(r"([\d,]+)-depositary-share position", prompt))
    preferred = position / denominator
    par = _number(_one(r"liquidation preference of \$([\d,]+)", flat))
    minimum = F(_one(r"Minimum Conversion Rate of ([\d.]+)", flat))
    maximum = F(_one(r"not be more than ([\d.]+) shares", flat))
    price = F(_one(r"applicable market value ([\d.]+)", prompt))
    comparison_mark = F(_one(r"with a ([\d.]+) mark", prompt))
    first_dividend = F(_one(r"first Dividend Payment Date is \$([\d.]+)", flat))
    next_dividend = F(_one(r"Each subsequent dividend is expected to be \$([\d.]+)", flat))
    _require(prompt, r"through October 15, 2025", r"no anti-dilution", r"before cash-in-lieu")
    _require(flat, r"January 15, April 15, July 15 and October 15", r"straight-line interpolation", r"365-day or 366-day")
    _require(pricing, r"volume-weighted average price", r"20 consecutive trading day", r"cash or.{0,80}shares")
    # The stipulated market value lies strictly inside the printed normal branch.
    if not minimum < par / price < maximum:
        raise ValueError("This reviewed case requires the interior mandatory-conversion branch")
    normal, optional = preferred * par / price, preferred * minimum
    table = terms.split("Effective Date", 1)[1].split("The exact stock price", 1)[0]
    prices = [F(value) for value in re.findall(r"\$(\d+\.\d+)", table.splitlines()[0])]
    rows = {}
    for day, values in re.findall(r"^(October \d+, \d{4})\s+([^\n]+)$", table, flags=re.MULTILINE):
        rows[_date(day)] = [F(value) for value in values.split()]
    if len(prices) != 12 or len(rows) != 4 or any(len(row) != len(prices) for row in rows.values()):
        raise ValueError("Preferred conversion table shape differs")
    event_price = F(_one(r"contractual stock price ([\d.]+)", prompt))
    event_date = _date(_one(r"effective date (\w+ \d+, \d{4})", prompt))
    rate, certificate = _interpolate(event_price, event_date, prices, rows)
    event_shares = preferred * rate
    return _finish({
        "preferred_shares_represented": preferred,
        "first_plus_next_three_dividends": preferred * (first_dividend + 3 * next_dividend),
        "normal_mandatory_common_share_entitlement": normal,
        "ordinary_optional_common_share_entitlement": optional,
        "normal_minus_optional_marked_value": (normal - optional) * comparison_mark,
        "fundamental_change_common_share_entitlement": event_shares,
        "fundamental_minus_ordinary_optional_shares": event_shares - optional,
        "single_close_identifies_mandatory_conversion": False,
        "dividends_contractually_guaranteed_in_cash": False,
    }, {"depositary_position": position, "receipts_per_preferred": denominator, "par": par,
        "minimum_rate": minimum, "maximum_rate": maximum, "applicable_market_value": price,
        "comparison_mark": comparison_mark,
        "first_dividend": first_dividend, "regular_dividend": next_dividend,
        "fundamental_change_stock_price": event_price, "fundamental_change_date": event_date.isoformat()},
        "Receipt-to-preferred mapping; distinct normal/optional branches; price-and-actual-day interpolation of the preferred-share table.",
        **certificate, event_conversion_rate=rate,
        interpretation="Aggregate entitlements precede fractional settlement; no dividend or make-whole value is included in conversion fields. One closing price does not specify the contractual VWAP.")


def ibm(question, sources):
    faq, tax, prompt = sources["h2b-ibm-faq"], sources["h2b-ibm-tax"], question["prompt"]
    _require(faq, r"one share.{0,60}every five shares", r"no impact on the number")
    _require(tax, r"each block", r"not bound by the approach")
    _require(prompt, r"PRINTED illustrative allocation", r"first block produces no fractional")
    blocks = [(int(n), F(price)) for n, price in re.findall(r"(\d+)(?: IBM shares acquired at| acquired at) \$(\d+(?:\.\d+)?)", prompt)]
    if len(blocks) != 2:
        raise ValueError("Reviewed two-block account differs")
    ibm_fraction, kd_fraction = (F(value) / 100 for value in _one(r"([\d.]+)%.*?IBM common stock and ([\d.]+)%", _flat(tax)))
    if ibm_fraction + kd_fraction != 1:
        raise ValueError("Printed basis fractions do not sum to one")
    entitled = [F(n, 5) for n, _ in blocks]
    fractions = [shares - shares.numerator // shares.denominator for shares in entitled]
    if fractions[0] != 0 or sum(fractions) >= 1:
        raise ValueError("Reviewed residual-fraction ownership differs")
    bases = [n * price for n, price in blocks]
    kd_bases = [basis * kd_fraction for basis in bases]
    disposed_basis = sum(basis * fraction / shares for basis, fraction, shares in zip(kd_bases, fractions, entitled))
    sale_price = F(_one(r"synthetic \$([\d.]+) per Kyndryl", prompt))
    fee = F(_one(r"charging \$([\d.]+)", prompt))
    cash = sum(fractions) * sale_price - fee
    retained_ibm, retained_kd = sum(bases) * ibm_fraction, sum(kd_bases) - disposed_basis
    if retained_ibm + retained_kd + disposed_basis != sum(bases):
        raise ValueError("Retained and disposed basis do not reconcile")
    return _finish({
        "kyndryl_economic_entitlement": sum(entitled), "retained_ibm_shares": sum(n for n, _ in blocks),
        "retained_ibm_basis": retained_ibm, "fractional_kd_basis": disposed_basis,
        "retained_kd_basis": retained_kd, "cash_in_lieu_net": cash,
        "fractional_disposal_gain": cash - disposed_basis,
        "issuer_method_is_mandatory": False, "actual_cash_known_without_sale_price": False,
    }, {"blocks": blocks, "block_original_basis": bases, "printed_ibm_fraction": ibm_fraction,
        "printed_kd_fraction": kd_fraction, "block_entitlements": entitled, "block_fractions": fractions,
        "sale_price": sale_price, "sale_fee": fee},
        "Separate acquisition-block basis allocation using printed percentages; residual-share basis, net cash and disposal gain reconcile independently.",
        total_original_basis=sum(bases), basis_conserved=True,
        identification="Different broker sale prices or fees preserve the distribution documents while changing net cash; actual cash is not identified without them.")


CHECKS = {"h2-001": paypal, "h2-002": cisco, "h2-005": boeing, "h2-007": ibm}
