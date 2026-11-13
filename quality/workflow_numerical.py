"""Independent public-packet calculations for the six executable workflows.

These functions consume candidate-visible inputs only. Cash and lot ledgers use
decimal-exact fractions; matrix fits use Cholesky solves, never matrix inversion.
The returned certificates expose reconciliation and chronological split checks.
"""
from fractions import Fraction
import math
import statistics


def number(value):
    return Fraction(str(value))


def plain(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def spd_solve(matrix, target):
    """Cholesky factorization and two triangular solves for a positive matrix."""
    n = len(target)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            residual = matrix[i][j] - dot(lower[i][:j], lower[j][:j])
            if i == j:
                if residual <= 0:
                    raise ValueError("Matrix is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    forward = []
    for i in range(n):
        forward.append((target[i] - dot(lower[i][:i], forward)) / lower[i][i])
    result = [0.0] * n
    for i in reversed(range(n)):
        result[i] = (forward[i] - sum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    return result


def q101(inputs):
    i = inputs
    security = {row["id"]: row for row in i["securities"]}
    prices = {(row["date"], row["security"]): number(row["price"]) for row in i["marks"]}
    fx = {(row["date"], row["currency"]): number(row["usd_per_unit"]) for row in i["fx"]}
    positions = {s: Fraction(0) for s in security}
    positions.update({row["security"]: number(row["quantity"]) for row in i["opening_positions"]})
    cash = {(desk, currency): Fraction(0) for desk in i["desks"] for currency in i["currencies"]}
    cash.update({(row["desk"], row["currency"]): number(row["amount"]) for row in i["opening_cash"]})

    def nav(date):
        return sum(amount * fx[date, currency] for (_, currency), amount in cash.items()) + sum(
            quantity * prices[date, s] * fx[date, security[s]["currency"]] for s, quantity in positions.items())

    prior = i["opening_date"]
    old_nav = nav(prior)
    navs, bridges, balances, returns, residuals = [], [], [], [], []
    for date in i["days"]:
        beginning_cash = cash.copy()
        flow = Fraction(0)
        for row in i["external_flows"]:
            if row["date"] == date:
                key = row["desk"], row["currency"]
                amount = number(row["amount"])
                cash[key] += amount
                beginning_cash[key] += amount
                flow += amount * fx[prior, key[1]]
        components = {desk: [Fraction(0)] * 4 for desk in i["desks"]}
        for (desk, currency), amount in beginning_cash.items():
            components[desk][3] += amount * (fx[date, currency] - fx[prior, currency])
        for s, quantity in positions.items():
            desk, currency = security[s]["desk"], security[s]["currency"]
            components[desk][0] += quantity * (prices[date, s] - prices[prior, s]) * fx[date, currency]
            components[desk][3] += quantity * prices[prior, s] * (fx[date, currency] - fx[prior, currency])
        for row in i["dividends"]:
            if row["date"] == date:
                s = row["security"]
                desk, currency = security[s]["desk"], security[s]["currency"]
                income = positions[s] * number(row["per_share"])
                cash[desk, currency] += income
                components[desk][2] += income * fx[date, currency]
        for row in i["fills"]:
            if row["date"] == date:
                s = row["security"]
                desk, currency = security[s]["desk"], security[s]["currency"]
                quantity, price, fee = map(number, (row["quantity"], row["price"], row["fee"]))
                positions[s] += quantity
                cash[desk, currency] -= quantity * price + fee
                components[desk][1] += quantity * (prices[date, s] - price) * fx[date, currency]
                components[desk][2] -= fee * fx[date, currency]
        for row in i["cash_expenses"]:
            if row["date"] == date:
                key = row["desk"], row["currency"]
                amount = number(row["amount"])
                cash[key] -= amount
                components[key[0]][2] -= amount * fx[date, key[1]]
        current = nav(date)
        bridge = [amount for desk in i["desks"] for amount in components[desk]]
        residuals.append(current - old_nav - flow - sum(bridge))
        returns.append((current - old_nav - flow) / (old_nav + flow))
        navs.append(current)
        bridges.extend(bridge)
        balances.extend(cash[desk, currency] for desk in i["desks"] for currency in i["currencies"])
        old_nav, prior = current, date
    return plain({"daily_nav_usd": navs, "daily_desk_pnl_bridge_usd": bridges,
                  "daily_cash_local": balances, "ending_quantities": [positions[s] for s in i["security_order"]],
                  "daily_returns": returns, "time_weighted_return": math.prod(1 + r for r in returns) - 1}), plain({"exact_nav_bridge_residuals": residuals})


def q102(inputs):
    i = inputs
    securities = i["security_order"]
    lots = {s: [] for s in securities}
    for row in sorted(i["opening_lots"], key=lambda row: (row["acquired"], row["lot_id"])):
        lots[row["security"]].append([number(row["quantity"]), number(row["unit_cost"])])
    marks = {(r["date"], r["security"]): number(r["price"]) for r in i["marks"]}
    cash = number(i["opening_cash"])
    claims = [[r["settlement_date"], number(r["amount"]), "trade"] for r in i["opening_unsettled"]]

    def ledger(date):
        return [[sum(q for q, _ in lots[s]), sum(q * cost for q, cost in lots[s]),
                 sum(q * (marks[date, s] - cost) for q, cost in lots[s])] for s in securities]

    previous = ledger(i["opening_date"])
    old_unrealized = sum(r[2] for r in previous)
    old_nav = cash + sum(c[1] for c in claims) + sum(r[1] + r[2] for r in previous)
    navs, bridges, balances, residuals = [], [], [], []
    for date in i["days"]:
        for row in i["splits"]:
            if row["date"] == date:
                ratio = number(row["ratio"])
                lots[row["security"]] = [[q * ratio, cost / ratio] for q, cost in lots[row["security"]]]
        income, realized, commissions = Fraction(0), Fraction(0), Fraction(0)
        for row in i["dividends"]:
            if row["ex_date"] == date:
                amount = sum(q for q, _ in lots[row["security"]]) * number(row["per_share"])
                income += amount
                claims.append([row["payment_date"], amount, "dividend"])
        for row in sorted((r for r in i["trades"] if r["trade_date"] == date), key=lambda r: (r["sequence"], r["trade_id"])):
            s = row["security"]
            quantity, price, fee = map(number, (row["quantity"], row["price"], row["fee"]))
            commissions += fee
            claims.append([row["settlement_date"], -quantity * price - fee, "trade"])
            if quantity > 0:
                lots[s].append([quantity, price])
            else:
                remaining = -quantity
                while remaining:
                    if not lots[s]:
                        raise ValueError("Sale exceeds available FIFO inventory")
                    used = min(remaining, lots[s][0][0])
                    realized += used * (price - lots[s][0][1])
                    remaining -= used
                    lots[s][0][0] -= used
                    if lots[s][0][0] == 0:
                        lots[s].pop(0)
        cash += sum(amount for due, amount, _ in claims if due <= date)
        claims = [claim for claim in claims if claim[0] > date]
        current = ledger(date)
        marked = sum(row[1] + row[2] for row in current)
        unrealized = sum(row[2] for row in current)
        receivables = sum(amount for _, amount, kind in claims if kind == "trade" and amount > 0)
        payables = -sum(amount for _, amount, kind in claims if kind == "trade" and amount < 0)
        dividends = sum(amount for _, amount, kind in claims if kind == "dividend")
        nav = cash + receivables - payables + dividends + marked
        bridge = [realized, unrealized - old_unrealized, income, -commissions]
        residuals.append(nav - old_nav - sum(bridge))
        navs.append(nav)
        bridges.extend(bridge)
        balances.extend([cash, receivables, payables, dividends, marked])
        old_nav, old_unrealized = nav, unrealized
    headroom = max(Fraction(0), cash - payables - number(i["minimum_cash_reserve"]))
    return plain({"daily_nav_usd": navs, "daily_pnl_bridge_usd": bridges,
                  "daily_balance_sheet_usd": balances, "ending_security_ledger": [v for row in current for v in row],
                  "cash_distribution_headroom_usd": headroom,
                  "can_pay_requested_distribution": number(i["requested_distribution"]) <= headroom}), plain({"exact_nav_bridge_residuals": residuals})


def fee_fixed_point(pretrade, target, cost):
    """Bracket the scalar self-financing fee equation; no target-weight drift shortcut."""
    low, high = 0.0, 0.5
    for _ in range(80):
        middle = (low + high) / 2
        notional = sum(abs((1 - middle) * w - p) for p, w in zip(pretrade, target))
        if middle < cost * notional:
            low = middle
        else:
            high = middle
    fee = (low + high) / 2
    notional = sum(abs((1 - fee) * w - p) for p, w in zip(pretrade, target))
    return fee, notional


def q103(inputs):
    i = inputs
    rows = sorted(i["rows"], key=lambda row: row["month"])
    size, side = len(i["industries"]), i["names_per_side"]
    identifiers = [industry["id"] for industry in i["industries"]]

    def run(lookback, start, end):
        pretrade = [0.0] * size
        wealth = peak = 1.0
        weights, returns, turnovers, fee_errors = [], [], [], []
        drawdown = 0.0
        for t, row in enumerate(rows):
            if not start <= row["month"] <= end:
                continue
            stop = t - i["skip_recent_months"]
            history = rows[stop - lookback:stop]
            if len(history) != lookback:
                raise ValueError("Insufficient signal history")
            signal = [math.prod(1 + h["returns"][j] for h in history) - 1 for j in range(size)]
            longs = sorted(range(size), key=lambda j: (-signal[j], identifiers[j]))[:side]
            shorts = sorted((j for j in range(size) if j not in longs), key=lambda j: (signal[j], identifiers[j]))[:side]
            target = [0.5 / side if j in longs else -0.5 / side if j in shorts else 0.0 for j in range(size)]
            fee, notional = fee_fixed_point(pretrade, target, i["one_way_cost_rate"])
            multiplier = (1 - fee) * (1 + dot(target, row["returns"]))
            pretrade = [(1 - fee) * w * (1 + r) / multiplier for w, r in zip(target, row["returns"])]
            wealth *= multiplier
            peak = max(peak, wealth)
            drawdown = max(drawdown, 1 - wealth / peak)
            returns.append(multiplier - 1)
            weights.extend(target)
            turnovers.append(notional)
            fee_errors.append(fee - i["one_way_cost_rate"] * notional)
        return weights, returns, turnovers, wealth, drawdown, fee_errors

    validation = [run(length, i["validation_start"], i["validation_end"])[1] for length in i["lookbacks"]]
    sharpes = [math.sqrt(12) * statistics.mean(r) / statistics.stdev(r) for r in validation]
    selected = min(range(len(sharpes)), key=lambda index: (-sharpes[index], i["lookbacks"][index]))
    weights, returns, turnover, wealth, drawdown, errors = run(i["lookbacks"][selected], i["test_start"], i["test_end"])
    return {"validation_sharpes": sharpes, "chosen_lookback_months": i["lookbacks"][selected],
            "test_weights": weights, "test_net_returns": returns, "test_turnover": turnover,
            "test_metrics": [12 * statistics.mean(returns), math.sqrt(12) * statistics.stdev(returns), wealth - 1, drawdown]}, {"fee_equation_residuals": errors, "test_starts_flat": True}


def q104(inputs):
    i = inputs
    returns = {(row["month"], row["security_id"]): number(row["return"]) for row in i["returns"]}
    delistings = {(row["month"], row["security_id"]): number(row["delisting_return"]) for row in i["delistings"]}
    fees = {row["security_id"]: number(row["roundtrip_bps"]) / 10000 for row in i["fees"]}
    identifiers, ratios, gross, costs, net, selected_records = [], [], [], [], [], []
    for formation in i["formations"]:
        cutoff, start, month = (formation[k] for k in ("cutoff", "holding_start", "holding_month"))
        eligible = {}
        for firm in i["firms"]:
            if not firm["active_from"] <= start < firm["active_until"]:
                continue
            records = [r for r in i["filings"] if r["firm_id"] == firm["firm_id"] and r["accepted_at"] <= cutoff and r["period_end"] <= cutoff[:10]]
            if records:
                chosen = max(records, key=lambda r: (r["period_end"], r["accepted_at"], r["record_id"]))
                eligible[firm["firm_id"]] = (firm["sector"], number(chosen["ttm_earnings"]) / number(chosen["book_equity"]), chosen["record_id"])
        month_gross, month_cost = Fraction(0), Fraction(0)
        for sector in i["sector_order"]:
            firms = [firm for firm, record in eligible.items() if record[0] == sector]
            long = min(firms, key=lambda f: (-eligible[f][1], f))
            short = min((f for f in firms if f != long), key=lambda f: (eligible[f][1], f))
            for firm, sign in ((long, 1), (short, -1)):
                mapping = [r for r in i["security_map"] if r["firm_id"] == firm and r["valid_from"] <= start < r["valid_until"]]
                if len(mapping) != 1:
                    raise ValueError("Effective security mapping is not unique")
                security = mapping[0]["security_id"]
                total_return = (1 + returns[month, security]) * (1 + delistings.get((month, security), 0)) - 1
                weight = number(i["weight_per_leg"])
                month_gross += sign * weight * total_return
                month_cost += weight * fees[security]
                identifiers.append(security)
                ratios.append(eligible[firm][1])
                selected_records.append(eligible[firm][2])
        gross.append(month_gross)
        costs.append(month_cost)
        net.append(month_gross - month_cost)
    return plain({"selected_security_ids": identifiers, "selected_roe": ratios, "monthly_gross_returns": gross,
                  "monthly_costs": costs, "monthly_net_returns": net, "terminal_wealth": math.prod(1 + r for r in net)}), {"selected_filing_record_ids": selected_records}


def covariance(rows):
    columns = list(zip(*rows))
    means = [statistics.mean(column) for column in columns]
    centered = [[value - mean for value, mean in zip(row, means)] for row in rows]
    return [[sum(row[a] * row[b] for row in centered) / (len(rows) - 1) for b in range(len(means))] for a in range(len(means))]


def minimum_variance(sample, shrinkage):
    size = len(sample)
    regularized = [[value if a == b else (1 - shrinkage) * value for b, value in enumerate(row)] for a, row in enumerate(sample)]
    direction = spd_solve(regularized, [1.0] * size)
    weights = [value / sum(direction) for value in direction]
    marginal = [dot(row, weights) for row in regularized]
    return weights, max(marginal) - min(marginal)


def q105(inputs):
    i = inputs
    rows = sorted(i["rows"], key=lambda row: row["month"])
    estimation = [row["returns"] for row in rows if i["estimation_window"]["start"] <= row["month"] <= i["estimation_window"]["end"]]
    test = [row["returns"] for row in rows if i["test_window"]["start"] <= row["month"] <= i["test_window"]["end"]]
    folds, length = i["validation"]["n_folds"], i["validation"]["validation_months"]
    grid = i["shrinkage_grid"]
    losses = [[] for _ in grid]
    sizes = []
    for k in range(folds):
        end = len(estimation) - (folds - k) * length
        sizes.append(end)
        sample = covariance(estimation[:end])
        for j, shrinkage in enumerate(grid):
            weights, _ = minimum_variance(sample, shrinkage)
            losses[j].append(statistics.variance(dot(row, weights) for row in estimation[end:end + length]))
    means = [statistics.mean(values) for values in losses]
    selected = min(range(len(grid)), key=lambda j: (means[j], -grid[j]))
    weights, stationarity = minimum_variance(covariance(estimation), grid[selected])
    annualizer = math.sqrt(i["conventions"]["annualization_factor"])
    return {"selected_shrinkage": grid[selected], "validation_variance_by_candidate": means,
            "final_weights": weights, "final_gross_leverage": sum(map(abs, weights)),
            "test_annualized_volatility": annualizer * statistics.stdev(dot(row, weights) for row in test),
            "equal_weight_test_annualized_volatility": annualizer * statistics.stdev(statistics.mean(row) for row in test)}, {
                "fold_training_sizes": sizes, "weight_sum_residual": sum(weights) - 1,
                "marginal_variance_spread": stationarity, "fold_candidate_losses": losses}


def ridge_fit(rows, penalty):
    """Dual ridge solve: beta = Z' (Z Z' + lambda I)^-1 centered_y."""
    columns = list(zip(*(row["x"] for row in rows)))
    means = [statistics.mean(column) for column in columns]
    scales = [statistics.stdev(column) for column in columns]
    target_mean = statistics.mean(row["y"] for row in rows)
    standardized = [[(x - mu) / scale for x, mu, scale in zip(row["x"], means, scales)] for row in rows]
    gram = [[dot(a, b) + (penalty if j == k else 0) for k, b in enumerate(standardized)] for j, a in enumerate(standardized)]
    dual = spd_solve(gram, [row["y"] - target_mean for row in rows])
    beta = [sum(row[j] * value for row, value in zip(standardized, dual)) for j in range(len(means))]
    raw = [value / scale for value, scale in zip(beta, scales)]
    intercept = target_mean - dot(raw, means)
    residual = [dot(row, beta) - (record["y"] - target_mean) for row, record in zip(standardized, rows)]
    gradient = [sum(row[j] * error for row, error in zip(standardized, residual)) + penalty * beta[j] for j in range(len(beta))]
    return [intercept] + raw, max(map(abs, gradient))


def q106(inputs):
    i = inputs
    rows = sorted(i["rows"], key=lambda row: row["t"])
    n, v = i["split"]["train"], i["split"]["validation"]
    train, validation, test = rows[:n], rows[n:n + v], rows[n + v:]
    if len(test) != i["split"]["test"]:
        raise ValueError("Chronological split does not partition the supplied panel")
    grid = i["lambda_grid"]

    def mse(model, observations):
        return statistics.mean((model[0] + dot(model[1:], row["x"]) - row["y"]) ** 2 for row in observations)

    fits = [ridge_fit(train, penalty) for penalty in grid]
    losses = [mse(model, validation) for model, _ in fits]
    selected = min(range(len(grid)), key=lambda j: (losses[j], -grid[j]))
    refits = [ridge_fit(train + validation, penalty) for penalty in grid]
    test_losses = [mse(model, test) for model, _ in refits]
    illicit = min(range(len(grid)), key=lambda j: (test_losses[j], -grid[j]))
    return {"validation_mse_by_lambda": losses, "selected_lambda": grid[selected],
            "final_coefficients": refits[selected][0], "test_mse_by_lambda": test_losses,
            "test_mse_selected": test_losses[selected], "illicit_lambda": grid[illicit]}, {
                "training_normal_equation_residuals": [error for _, error in fits],
                "refit_normal_equation_residuals": [error for _, error in refits],
                "selection_last_period": train[-1]["t"], "refit_last_period": validation[-1]["t"]}


SOLVERS = {f"q{n}": globals()[f"q{n}"] for n in range(101, 107)}


def audit_case(packet):
    answers, certificate = SOLVERS[packet["id"]](packet["inputs"])
    names = {field["name"] for field in packet["outputs"]}
    if set(answers) != names:
        raise ValueError("Independent calculation does not cover the declared output schema")
    return {"answers": answers, "certificate": certificate, "unsupported": [], "ambiguities": []}
