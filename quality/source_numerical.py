"""Independent arithmetic from retained public evidence packets.

No answer key or scorer is imported. These checks verify the retained excerpts
and stipulated conventions, not the completeness of external source documents.
"""
from decimal import Decimal, localcontext
from datetime import date, datetime, timedelta, timezone
import re

D = Decimal


def unique(pattern, text):
    values = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    if not values or len(set(values)) != 1:
        raise ValueError("Missing or conflicting retained-source operand")
    return values[0]


def decimal(text):
    return D(text.replace(",", "").strip())


def numeric(row):
    return [decimal(value) for value in row.split("|")]


def finish(answers, operands, method):
    return ({key: float(value) if isinstance(value, D) else value for key, value in answers.items()},
            {"method": method, "operands": operands,
             "scope": "Retained-excerpt arithmetic and declared conventions; no external source revalidation, citation-grader or practitioner certification."})


def payrolls(question, sources):
    feb, mar, apr = (sources[sid] for sid in ("bls-20240202", "bls-20240308", "bls-20240405"))
    def thousands(text):
        whole, residual = divmod(int(text.replace(",", "")), 1000)
        if residual:
            raise ValueError("Payroll operand is not an exact integer thousand")
        return whole
    def first(text, month):
        return thousands(unique(r"Total nonfarm payroll employment rose by ([\d,]+) in " + month, text))
    def revision(text, month):
        direction, amount, old, new = unique(r"change(?: in total nonfarm payroll employment)? for " + month
                          + r" was revised (up|down) by ([\d,]+), from \+([\d,]+) to \+([\d,]+)", text)
        old, new, amount = thousands(old), thousands(new), thousands(amount)
        if new - old != (amount if direction.lower() == "up" else -amount):
            raise ValueError("Reported payroll revision does not reconcile")
        return old, new
    jan_first, feb_first, mar_first = first(feb, "January"), first(mar, "February"), first(apr, "March")
    dec_old, dec_mar = revision(mar, "December")
    jan_old, jan_mar = revision(mar, "January")
    jan_prev, jan_apr = revision(apr, "January")
    feb_prev, feb_apr = revision(apr, "February")
    if jan_old != jan_first or jan_prev != jan_mar or feb_prev != feb_first:
        raise ValueError("Release-vintage chain does not reconcile")
    first_sum = jan_first + feb_first + mar_first
    final_sum = jan_apr + feb_apr + mar_first
    return finish({
        "jan_first_print": jan_first, "jan_asof_mar08": jan_mar, "feb_first_print": feb_first,
        "mar08_prior_two_month_revision": dec_mar - dec_old + jan_mar - jan_old,
        "jan_asof_apr05": jan_apr, "feb_asof_apr05": feb_apr, "mar_first_print": mar_first,
        "apr05_prior_two_month_revision": jan_apr - jan_prev + feb_apr - feb_prev,
        "q1_first_print_sum": first_sum, "q1_asof_apr05_sum": final_sum,
        "q1_cumulative_revision_vs_first_prints": final_sum - first_sum,
        "apr05_increment_to_tracked_q1_change": final_sum - (jan_mar + feb_first),
    }, {"first_prints": [jan_first, feb_first, mar_first],
        "march_vintage_dec_jan_feb": [dec_mar, jan_mar, feb_first],
        "april_vintage_jan_feb_mar": [jan_apr, feb_apr, mar_first]},
        "Month-labelled release chain; distinct prior-month revision windows; Q1 sums and changing-coverage difference.")


def inflation(question, sources):
    early, late = sources["exp-cpi-20240111"], sources["exp-cpi-20240213"]
    old = numeric(unique(r"^All items \| (.+)$", early))
    new = numeric(unique(r"^All items \| (.+)$", late))
    if len(old) != 8 or len(new) != 8 or "Jun.\n2023" not in early or "Jul.\n2023" not in late:
        raise ValueError("CPI table columns differ from reviewed vintages")
    old_q4, new_q4 = old[4:7], new[3:6]
    def proxy(rates):
        factor = D(1)
        for rate in rates:
            factor *= 1 + rate / 100
        return 100 * (factor**4 - 1)
    with localcontext() as context:
        context.prec = 40
        prior, current = proxy(old_q4), proxy(new_q4)
        december_yoy = decimal(unique(r"([\d.]+)-percent increase for the 12 months ending December", late))
        expectation = decimal(unique(r"S is explicitly SYNTHETIC: ([\d.]+)%", question["prompt"]))
        answers = {
            "december_sa_revision": new[5] - old[6], "q4_annualized_proxy_jan11": prior,
            "q4_annualized_proxy_feb13": current, "q4_annualized_proxy_revision": current - prior,
            "january_sa_acceleration": new[6] - new[5], "december_nsa_yoy_revision": december_yoy - old[7],
            "january_surprise_given_expectation": new[6] - expectation,
            "surprise_identifiable_without_expectation": False, "proxy_equals_exact_index_growth": False,
        }
    return finish(answers, {"old_q4_rates": list(map(str, old_q4)), "new_q4_rates": list(map(str, new_q4)),
                            "january_sa_rate": str(new[6]), "synthetic_expectation": str(expectation)},
                  "Decimal compounding of rounded monthly proxies; same-month seasonal revision; missing expectation and unrounded levels remain unidentified.")


def output_vintages(question, sources):
    early, late = sources["exp-bea-20230830-pdf"], sources["exp-bea-20230928-pdf"]
    def contributions(text):
        table = text.split("Table 2.", 1)[1].split("Table 3.", 1)[0]
        patterns = [(2, "Personal consumption expenditures"), (26, "Gross private domestic investment"),
                    (43, "Net exports of goods and services"), (50, "Government consumption\\s+expenditures and gross investment")]
        values = []
        for row, label in patterns:
            raw = unique(r"^" + str(row) + " " + label + r" ([^\n]+)$", table).split()
            if raw[-1] != str(row):
                raise ValueError("Missing printed BEA row identity")
            values.append(decimal(raw[-2]))
        return values
    old, new = contributions(early), contributions(late)
    advance, second = map(decimal, unique(r"^Real GDP ([\d.]+) ([\d.]+)$", early))
    late_advance, late_second, third = map(decimal, unique(r"^Real GDP ([\d.]+) ([\d.]+) ([\d.]+)$", late))
    if (advance, second) != (late_advance, late_second) or "updated to 2017 from 2012" not in late:
        raise ValueError("GDP vintage/reference-year sequence differs")
    levels = unique(r"^1 Gross domestic product \(GDP\) ([^\n]+)$", late.split("Table 3.", 1)[1]).split()
    if len(levels) != 16 or levels[-1] != "1":
        raise ValueError("GDP level table shape differs")
    nominal_q2 = decimal(levels[5])
    revisions = [b - a for a, b in zip(old, new)]
    with localcontext() as context:
        context.prec = 40
        quarterly = 100 * ((1 + third / 100) ** D("0.25") - 1)
    return finish({
        "advance_to_second_revision": second - advance, "second_to_third_revision": third - second,
        "third_implied_quarterly_growth": quarterly, "pce_contribution_revision": revisions[0],
        "other_components_contribution_revision": sum(revisions[1:]), "third_component_sum": sum(new),
        "headline_minus_component_revision_gap": third - second - sum(revisions),
        "third_nominal_quarter_output": nominal_q2 / 4,
        "surprise_identifiable_without_consensus": False, "cross_base_level_ratio_is_real_growth": False,
    }, {"second_contributions": list(map(str, old)), "third_contributions": list(map(str, new)),
        "headlines": list(map(str, (advance, second, third))), "third_nominal_saar": str(nominal_q2)},
        "Disjoint contribution reconciliation retains displayed residual; fourth-root headline conversion and SAAR quarter-equivalent flow; cross-base levels are incomparable.")


def gas_storage(question, sources):
    early, late = sources["energy-gas-20210617"], sources["energy-gas-20210624"]
    def row(text, name):
        raw = unique(r"^" + name + r" \| (.+)$", text)
        return numeric(raw.replace(" C", ""))
    first, second = row(early, "Total"), row(late, "Total")
    pacific = row(early, "Pacific")
    first_end, first_start = unique(r"^Region \| ([\d-]+) \| ([\d-]+) \| change$", early)
    second_end, second_start = unique(r"^Region \| ([\d-]+) \| ([\d-]+) \| change$", late)
    if second_start != first_end:
        raise ValueError("Storage survey dates are not consecutive")
    observed_days = (date.fromisoformat(second_end) - date.fromisoformat(first_start)).days
    if observed_days <= 0:
        raise ValueError("Storage survey interval must be positive")
    reclass = decimal(unique(r"decreased working gas stocks of ([\d,]+) Bcf.*?week ending June 11, 2021", early))
    if second[1] != first[0] or first[0] - first[1] != first[2] or second[0] - second[1] != second[2]:
        raise ValueError("Storage endpoints do not reconcile")
    flow_first = first[0] - first[1] + reclass
    total_flow = flow_first + second[0] - second[1]
    days = decimal(unique(r"Over the next (\d+) days", question["prompt"]))
    extra = decimal(unique(r"with ([\d.]+) Bcf/day incremental demand", question["prompt"]))
    floor = decimal(unique(r"floor of ([\d,]+) Bcf", question["prompt"]))
    return finish({
        "june11_working_stock_change": first[0] - first[1], "june11_implied_flow": flow_first,
        "june11_pacific_implied_flow": pacific[0] - pacific[1] + reclass,
        "two_week_working_stock_change": second[0] - first[1], "two_week_implied_flow": total_flow,
        "two_week_average_implied_flow": total_flow / observed_days,
        "no_reclassification_june18_working_stock": second[0] + reclass,
        "reclassification_effect_on_total_gas": D(0),
        "scenario_terminal_working_stock": second[0] + total_flow - days * extra,
        "scenario_max_extra_demand": (second[0] + total_flow - floor) / days,
        "stock_flow_interpretation": "working_stock_change_includes_nonflow_reclassification",
    }, {"working_stocks_june4_june11_june18": list(map(str, (first[1], first[0], second[0]))),
        "reclassification_working_to_base": str(reclass), "physical_injections": str(total_flow)},
        "Endpoint stock-flow identity removes one classification event; total gas is conserved; conditional demand floor solved algebraically.")


def petroleum_balance(question, sources):
    rates = sources["energy-petroleum-table1-20240403"]
    stocks = sources["energy-petroleum-table4-20240403"]
    notes = sources["energy-petroleum-notes-20240403"]
    def rate(row):
        return decimal(unique(r"^(?:Crude Oil Supply|Other Supply) \| \(" + str(row) + r"\) [^|]+\| ([^|]+)\|", rates))
    if "(line 17) + Other Supply Production (line 18) + Net Product Imports (line 24)" not in notes:
        raise ValueError("Required product-balance convention absent")
    values = {i: rate(i) for i in (1, 4, 8, 12, 13, 14, 15, 16, 17, 18, 24, 27, 28, 29)}
    levels = numeric(unique(r"^Total Stocks \(Including SPR\) \| (.+)$", stocks))
    net_imports = values[8] - values[12]
    adjustment = values[17] - values[1] - net_imports - values[4] + values[13]
    products = values[17] + values[18] + values[24] - values[27] + values[28] + values[29]
    stock_change = levels[0] - levels[1]
    weekly_flow = (values[13] + values[27]) * 7 / 1000
    reduction = decimal(unique(r"reduce crude imports by ([\d,]+) thousand barrels/day", question["prompt"]))
    baseline_commercial = values[1] + net_imports + values[4] + values[16] - values[17] - values[15]
    if adjustment != values[16] or baseline_commercial != values[14]:
        raise ValueError("Crude accounting identity differs")
    commercial = baseline_commercial - reduction
    return finish({
        "net_crude_imports": net_imports, "reconstructed_crude_adjustment": adjustment,
        "reconstructed_products_supplied": products, "net_transfer_across_crude_and_products": values[4] + values[29],
        "weekly_stock_change_from_daily_rates": weekly_flow, "weekly_stock_change_from_levels": stock_change,
        "flow_minus_stock_rounding_gap": (weekly_flow - stock_change) * 1000,
        "adjustment_interpretation": "balancing_item_not_identified_production",
        "rounding_gap_interpretation": "rounding_consistent_not_evidence_of_physical_loss",
        "scenario_commercial_stock_change": commercial, "scenario_weekly_commercial_build": commercial * 7 / 1000,
        "maximum_import_reduction_without_commercial_draw": baseline_commercial,
    }, {"current_week_rate_rows": {str(k): str(v) for k, v in values.items()},
        "stock_levels_including_spr": list(map(str, levels[:2])), "synthetic_import_reduction": str(reduction)},
        "Six-component signed products balance, transfer cancellation, million/thousand conversion, retained rounding gap and fixed-flow commercial-stock scenario.")


def funding_scenarios(question, sources):
    prompt = question["prompt"]
    statement = sources["portfolio-fed-20230322"]
    whole, numerator, denominator, upper = unique(
        r"target range for the federal funds rate to (\d+)-(\d+)/(\d+) to (\d+) percent", statement)
    midpoint = (D(whole) + D(numerator) / D(denominator) + D(upper)) / 2
    reference = decimal(unique(r"reference midpoint is ([\d.]+)%", prompt))
    u = (midpoint - reference) * 100
    notional = decimal(unique(r"Financing pod borrows ([\d,]+) USD", prompt))
    accrual = decimal(unique(r"to a (\d+)-day accrual on ACT/360", prompt))
    funding = notional * u / 10000 * accrual / 360
    equity = tuple(map(decimal, unique(r"equity spread pod has signed DV01 at 2y ([+-]\d+) USD/bp and 10y ([+-]\d+) USD/bp", prompt)))
    treasury = tuple(map(decimal, unique(r"Treasury pod has signed DV01 at 2y ([+-]\d+) and 10y ([+-]\d+) USD/bp", prompt)))
    a2, a10, afee, amargin = map(decimal, unique(
        r"A has signed 2y DV01 ([+-]\d+) and 10y DV01 ([+-]\d+), fee (\d+), margin (\d+)", prompt))
    def scenario_tokens(text):
        return [u if t == "+u" else -u if t == "-u" else decimal(t) for t in text.split(",")]
    d2 = scenario_tokens(unique(r"d2=\(([^)]+)\) bp", prompt))
    d10 = scenario_tokens(unique(r"d10=\(([^)]+)\) bp", prompt))
    if len(d2) != 3 or len(d10) != 3:
        raise ValueError("Expected exactly three joint scenarios")
    def rate_pnl(sensitivities):
        return [-sensitivities[0] * x - sensitivities[1] * y for x, y in zip(d2, d10)]
    bpay, bfee, bmargin = unique(r"B is a fully prepaid synthetic scenario-contingent claim paying \(([^)]+)\) in H,R,T, nonrefundable premium (\d+), margin (\d+)", prompt)
    cfee, cmargin = map(decimal, unique(r"C receives exactly the incremental financing expense in every scenario, fee (\d+), margin (\d+)", prompt))
    hedges = {"A": (rate_pnl((a2, a10)), afee, amargin),
              "B": (list(map(decimal, bpay.split(","))), D(bfee), D(bmargin)),
              "C": ([funding] * 3, cfee, cmargin)}
    if len(hedges["B"][0]) != 3:
        raise ValueError("Expected exactly three claim payoffs")
    budget = decimal(unique(r"Starting free cash is (\d+)", prompt))
    epnl, tpnl = rate_pnl(equity), rate_pnl(treasury)
    base = [e + t - funding for e, t in zip(epnl, tpnl)]
    calls = [funding + max(-t, D(0)) for t in tpnl]
    plan_order = unique(r"fixed order ([A-Z|]+) and omit infeasible plans", prompt).split("|")
    if set(plan_order) != {"NONE", "A", "B", "C", "AB", "AC", "BC", "ABC"}:
        raise ValueError("Hedge-menu subset coverage differs")
    def plans(c_margin):
        rows = {}
        for plan in plan_order:
            selected = [] if plan == "NONE" else list(plan)
            fees = sum((hedges[h][1] for h in selected), D(0))
            upfront = fees + sum((c_margin if h == "C" else hedges[h][2] for h in selected), D(0))
            outflows = [max(D(0), b + sum((max(-hedges[h][0][i], D(0)) for h in selected), D(0)))
                        for i, b in enumerate(calls)]
            pnl = [b + sum((hedges[h][0][i] for h in selected), D(0)) - fees for i, b in enumerate(base)]
            rows[plan] = {"required_cash": upfront + max(outflows), "worst_pnl": min(pnl), "pnl": pnl}
        return rows
    def optimizers(rows, cash):
        feasible = [p for p in plan_order if rows[p]["required_cash"] <= cash]
        if not feasible:
            return []
        best = max(rows[p]["worst_pnl"] for p in feasible)
        return [p for p in feasible if rows[p]["worst_pnl"] == best]
    rows = plans(cmargin)
    best = optimizers(rows, budget)
    if len(best) != 1:
        raise ValueError("Fully specified stress book has no unique optimizer")
    chosen = best[0]
    next_cash = min(r["required_cash"] for r in rows.values()
                    if r["required_cash"] > budget and r["worst_pnl"] > rows[chosen]["worst_pnl"])
    zero_margin = plans(D(0))
    # Feasibility changes only at these exact margin thresholds. Evaluate both
    # the inclusive endpoint and a point on its right, covering every regime.
    thresholds = [budget - r["required_cash"] for p, r in zero_margin.items() if "C" in p]
    probes = {D(0), max([budget, *thresholds]) + 1}
    probes.update(x for t in thresholds for x in (t, t + 1) if x >= 0)
    missing_optimizers = {tuple(optimizers(plans(m), budget)) for m in probes}
    missing_determined = len(missing_optimizers) == 1 and len(next(iter(missing_optimizers))) == 1
    def compact(value):
        return format(value.normalize(), "f")
    return finish({
        "policy_midpoint": midpoint, "incremental_financing_expense": funding,
        "combined_parallel_dv01": sum(equity + treasury),
        "base_scenario_pnl": ";".join(f"{s}={compact(v)}" for s, v in zip("HRT", base)),
        "feasible_plans": "|".join(p for p in plan_order if rows[p]["required_cash"] <= budget),
        "optimal_plan": chosen, "optimal_worst_pnl": rows[chosen]["worst_pnl"],
        "optimal_minimum_cash": budget - rows[chosen]["required_cash"],
        "first_optimizer_change_budget": next_cash,
        "missing_margin_optimizer_status": "determined" if missing_determined else "insufficient_data",
    }, {"midpoint_percent": str(midpoint), "increment_bp": str(u), "funding": str(funding),
        "plan_certificates": {p: {k: list(map(str, v)) if isinstance(v, list) else str(v)
                                 for k, v in r.items()} for p, r in rows.items()},
        "missing_margin_optimizer_sets": [list(x) for x in sorted(missing_optimizers)]},
        "Enumerate all eight hedge subsets, separate terminal economics from gross timed liquidity, then inspect every nonnegative omitted-margin feasibility regime.")


def acceptance_replay(question, sources):
    prompt = question["prompt"]
    lag_policy = sources["systematic-sec-time-faq"]
    if "There is no timestamp to indicate when filing content is first available on sec.gov." not in lag_policy or "We don’t guarantee and cannot predict this lag." not in lag_policy:
        raise ValueError("Required SEC public-availability limitations absent")
    offset_h, offset_m = map(int, unique(r"EST \(UTC-(\d+):(\d+)\)", prompt))
    zone = timezone(-timedelta(hours=offset_h, minutes=offset_m))
    accepted = {}
    for asset in ("apple", "nvidia"):
        raw = unique(r"Accepted\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", sources[f"systematic-{asset}-index"])
        accepted[asset] = datetime.fromisoformat(raw).replace(tzinfo=zone).astimezone(timezone.utc)
    vendor_lag = int(unique(r"vendor arrival exactly (\d+) seconds", prompt))
    processing_lag = int(unique(r"processing another (\d+) seconds", prompt))
    ready = {a: t + timedelta(seconds=vendor_lag + processing_lag) for a, t in accepted.items()}
    first_probe, second_probe_time = unique(r"accession at (\S+) and at (\S+) using exactly", prompt)
    public_probes = (datetime.fromisoformat(first_probe),
                     datetime.fromisoformat(first_probe[:11] + second_probe_time))
    def public_status(probe):
        return "ineligible" if probe < accepted["nvidia"] else "not_established"
    first = datetime.fromisoformat(unique(r"Probe snapshots (\S+) and one second later", prompt))
    second = datetime.fromisoformat(unique(r"second trading decision is (\S+)\.", prompt))
    apple_score, nvidia_score = map(decimal, unique(r"scores are apple=([\d.]+), nvidia=([\d.]+)", prompt))
    scores = {"apple": apple_score, "nvidia": nvidia_score}
    def select(snapshot, defective=False):
        eligible = [a for a in accepted if defective or ready[a] <= snapshot]
        return max(eligible, key=lambda a: scores[a]) if eligible else "cash"
    raw_calendar = unique(r"execution calendar is ([^.]+)\.", prompt)
    year = first.year
    calendar = [datetime.strptime(f"{year} {s.strip()}", "%Y %b %d %H:%MZ").replace(tzinfo=timezone.utc)
                for s in raw_calendar.split(",")]
    if calendar != sorted(set(calendar)) or len(calendar) != 3:
        raise ValueError("Execution calendar is not three strictly ordered opens")
    returns = {}
    for start, end, a, n in re.findall(r"(Feb\d+)-open to (Feb\d+)-open: apple ([+-][\d.]+)%, nvidia ([+-][\d.]+)%", prompt):
        returns[(start, end)] = {"apple": D(a) / 100, "nvidia": D(n) / 100, "cash": D(0)}
    position = decimal(unique(r"fresh \$([\d,]+) position", prompt))
    cost = decimal(unique(r"Charge ([\d.]+) bps of actual traded dollar notional", prompt)) / 10000
    if "gains are not compounded into sizing" not in prompt or "including the changed exit value" not in prompt:
        raise ValueError("Sizing/cost convention differs")
    def replay(defective):
        total, ledger = D(0), []
        for snapshot in (first, second):
            entry = next(i for i, time in enumerate(calendar) if time > snapshot)
            exit_time = calendar[entry + 1]
            interval = (f"Feb{calendar[entry].day}", f"Feb{exit_time.day}")
            asset = select(snapshot, defective)
            rate = returns[interval][asset]
            charges = D(0) if asset == "cash" else position * cost + position * (1 + rate) * cost
            pnl = position * rate - charges
            total += pnl
            ledger.append({"asset": asset, "entry": calendar[entry].isoformat(), "exit": exit_time.isoformat(),
                           "net_profit": str(pnl), "charges": str(charges)})
        return total, ledger
    valid, valid_ledger = replay(False)
    invalid, invalid_ledger = replay(True)
    fmt = lambda t: t.isoformat().replace("+00:00", "Z")
    return finish({
        "apple_acceptance_utc": fmt(accepted["apple"]), "nvidia_acceptance_utc": fmt(accepted["nvidia"]),
        "nvidia_public_eligibility_before_acceptance": public_status(public_probes[0]),
        "nvidia_public_eligibility_at_acceptance": public_status(public_probes[1]),
        "before_ready_selected_asset": select(first), "at_ready_selected_asset": select(first + timedelta(seconds=1)),
        "first_execution_utc": fmt(next(time for time in calendar if time > first)),
        "valid_toy_net_profit": valid, "leakage_net_profit_inflation": invalid - valid,
        "empirical_performance_status": "not_established",
    }, {"accepted_utc": {a: fmt(t) for a, t in accepted.items()}, "ready_utc": {a: fmt(t) for a, t in ready.items()},
        "valid_ledger": valid_ledger, "period_end_join_ledger": invalid_ledger},
        "Timezone conversion and inclusive processing-readiness boundary; fixed decision/calendar replay; separate entry and changed-notional exit charges; acceptance alone does not certify dissemination.")


CHECKS = {"macro-001": payrolls, "macro-002": inflation, "macro-003": output_vintages,
          "energy-001": gas_storage, "energy-002": petroleum_balance,
          "portfolio-002": funding_scenarios, "systematic-001": acceptance_replay}
