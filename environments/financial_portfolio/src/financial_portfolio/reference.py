"""Standalone public-input reference analysis for the portfolio pilot.

Copy this source to ``research.py`` beside the four public input files and run
it with Python. It writes ``policy.json`` using only those files and the Python
standard library; no evaluator package, private data or environment key is read.
"""

import csv
import json
import math
from pathlib import Path


def read_csv(filename):
    with Path(filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def public_panel(prices, features, securities, fields):
    """Resolve dated tickers and process feature releases in eligibility order."""
    intervals = {}
    for security in securities:
        intervals.setdefault(security["ticker"], []).append(
            (int(security["valid_from"]), int(security["valid_to"]), security["entity_id"])
        )
    levels = {}
    for row in prices:
        day = int(row["day"])
        identities = [entity for first, last, entity in intervals[row["ticker"]]
                      if first <= day <= last]
        if len(identities) != 1:
            raise ValueError("Price requires exactly one dated entity mapping")
        key = (identities[0], day)
        value = float(row["close"]) * float(row["shares_per_original"])
        if key in levels and levels[key] != value:
            raise ValueError("Conflicting adjusted prices")
        levels[key] = value
    revisions = {}
    for row in features:
        key = (row["entity_id"], row["metric"], int(row["observation_day"]),
               int(row["available_day"]), int(row["revision"]))
        value = float(row["value"])
        if key in revisions and revisions[key] != value:
            raise ValueError("Conflicting feature revisions")
        revisions[key] = value
    events = sorted((max(observation, available), entity, metric,
                     (observation, available, revision), value)
                    for (entity, metric, observation, available, revision), value in revisions.items())
    cursor, latest, panel = 0, {}, {}
    entities = sorted({entity for entity, _ in levels})
    for day in range(max(day for _, day in levels)):
        while cursor < len(events) and events[cursor][0] <= day:
            _, entity, metric, priority, value = events[cursor]
            key = (entity, metric)
            if key not in latest or latest[key][0] < priority:
                latest[key] = (priority, value)
            cursor += 1
        for entity in entities:
            panel[entity, day] = {
                "forward_return": levels[entity, day + 1] / levels[entity, day] - 1,
                **{field: latest.get((entity, field), (None, None))[1] for field in fields},
            }
    return panel


def training_covariance(panel, contract, entities):
    """Estimate sample covariance once, then shrink its off-diagonal entries."""
    days = range(contract["training_start"], contract["training_end"])
    count = len(days)
    if count < 2:
        raise ValueError("At least two training observations are required")
    returns = [[panel[entity, day]["forward_return"] for day in days] for entity in entities]
    means = [math.fsum(series) / count for series in returns]
    centered = [[value - mean for value in series] for series, mean in zip(returns, means)]
    matrix = [[math.fsum(left * right for left, right in zip(centered[i], centered[j])) / (count - 1)
               * (1.0 if i == j else 1 - contract["covariance_diagonal_shrinkage"])
               for j in range(len(entities))] for i in range(len(entities))]
    if any(matrix[i][i] <= 0 for i in range(len(entities))):
        raise ValueError("Training variances must be positive")
    return matrix


def annual_volatility(weights, covariance):
    variance = math.fsum(weights[i] * weights[j] * covariance[i][j]
                         for i in range(len(weights)) for j in range(len(weights)))
    return math.sqrt(252 * max(0.0, variance))


def desired_weights(day, candidate, panel, contract, entities, covariance):
    values = [panel[entity, day][candidate["feature"]] for entity in entities]
    eligible = sorted((candidate["direction"] * value, entities[index], index)
                      for index, value in enumerate(values) if value is not None)
    target = [0.0] * len(entities)
    if len(eligible) < 4:
        return target
    for sign, members in ((-1, eligible[:2]), (1, eligible[-2:])):
        scores = [1 / math.sqrt(covariance[index][index])
                  if candidate["allocation"] == "inverse_vol" else 1.0
                  for _, _, index in members]
        normalization = math.fsum(scores)
        for (_, _, index), score in zip(members, scores):
            target[index] = sign * candidate["gross"] * 0.5 * score / normalization
    gross = math.fsum(abs(value) for value in target)
    maximum = max(abs(value) for value in target)
    volatility = annual_volatility(target, covariance)
    scale = min(1.0, contract["gross_cap"] / gross, contract["per_asset_cap"] / maximum,
                contract["annual_vol_cap"] / volatility if volatility else 1.0)
    return [value * scale for value in target]


def execute_target(previous, target, contract, entities, covariance):
    requested = [new - old for new, old in zip(target, previous)]
    capacity = [contract["participation_limit"] * contract["dollar_adv"][entity]
                / contract["reference_nav"] for entity in entities]
    fraction = min([1.0] + [allowed / abs(change) for allowed, change in zip(capacity, requested)
                           if change != 0])
    # Exact target assignment when fully filled prevents residual winddown dust.
    actual = list(target) if fraction == 1 else [old + fraction * change
                                                for old, change in zip(previous, requested)]
    changes = [new - old for new, old in zip(actual, previous)]
    turnover = math.fsum(abs(change) for change in changes)
    participation = [abs(change) * contract["reference_nav"] / contract["dollar_adv"][entity]
                     for entity, change in zip(entities, changes)]
    impact = contract["impact_bps"] / 10000 * math.fsum(
        abs(change) * math.sqrt(part) for change, part in zip(changes, participation))
    short = math.fsum(-weight for weight in actual if weight < 0)
    total_cost = contract["spread_bps"] / 10000 * turnover + impact \
        + contract["borrow_bps_daily"] / 10000 * short
    return actual, {"actual_gross": math.fsum(abs(value) for value in actual),
                    "forecast_vol": annual_volatility(actual, covariance),
                    "max_participation": max(participation), "execution_fraction": fraction,
                    "turnover": turnover, "impact_cost": impact, "total_cost": total_cost}


def evaluate_candidate(candidate, panel, contract, entities, covariance):
    previous = [0.0] * len(entities)
    history = {}
    for day in range(contract["validation_start"], contract["validation_end"]):
        target = desired_weights(day, candidate, panel, contract, entities, covariance) \
            if day < contract["active_end"] else [0.0] * len(entities)
        actual, metrics = execute_target(previous, target, contract, entities, covariance)
        gross_return = math.fsum(weight * panel[entity, day]["forward_return"]
                                 for entity, weight in zip(entities, actual))
        history[day] = {**metrics, "net_return": gross_return - metrics["total_cost"]}
        previous = actual
    if any(abs(value) > 1e-12 for value in previous):
        raise ValueError("Winddown left a nonzero position")
    daily = [metrics["net_return"] for metrics in history.values()]
    count = len(daily)
    mean = math.fsum(daily) / count
    deviation = math.sqrt(math.fsum((value - mean) ** 2 for value in daily) / (count - 1))
    trial = {"id": candidate["id"], "mean_net": mean,
             "sharpe": math.sqrt(252) * mean / deviation if deviation else 0.0,
             "mean_turnover": math.fsum(metrics["turnover"] for metrics in history.values()) / count,
             # Canonicalize BEFORE selection, including constraint-equivalent policies.
             "screening_score": round(mean - 2 * deviation / math.sqrt(count), 12)}
    return trial, history


def solve():
    contract = json.loads(Path("contract.json").read_text(encoding="utf-8"))
    entities = sorted(contract["dollar_adv"])
    fields = sorted({candidate["feature"] for candidate in contract["candidates"]})
    panel = public_panel(read_csv("prices.csv"), read_csv("features.csv"),
                         read_csv("securities.csv"), fields)
    covariance = training_covariance(panel, contract, entities)
    trials, histories = [], {}
    for candidate in contract["candidates"]:
        trial, history = evaluate_candidate(candidate, panel, contract, entities, covariance)
        trials.append(trial)
        histories[candidate["id"]] = history
    audit = []
    for request in contract["audit_requests"]:
        identity, day, field = request["entity_id"], request["day"], request["field"]
        value = histories[identity][day][field] if identity in histories else panel[identity, day][field]
        audit.append({**request, "value": value})
    winner = min(trials, key=lambda trial: (-trial["screening_score"], trial["id"]))
    selected = winner["id"] if winner["screening_score"] > 0 else None
    return {"audit": audit, "trials": trials, "decision": "deploy" if selected else "abstain",
            "selected": selected}


if __name__ == "__main__":
    Path("policy.json").write_text(json.dumps(solve(), indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
