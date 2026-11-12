"""Trusted fixed-notional portfolio calculations for the experimental contract."""

import math
import statistics

from financial_research.analytics import FEATURES, audit_values, clean_public

CANDIDATES = tuple(
    {"id": f"{feature}:{direction}:{gross}:{allocation}", "feature": feature,
     "direction": direction, "gross": gross, "allocation": allocation}
    for feature in FEATURES for direction in (-1, 1) for gross in (0.25, 0.5, 1.0)
    for allocation in ("equal", "inverse_vol")
)
PORTFOLIO_FIELDS = ("actual_gross", "forecast_vol", "max_participation",
                    "execution_fraction", "turnover", "impact_cost", "net_return")


def estimate_risk(panel, contract):
    """Sample covariance, half shrunk to its diagonal, using training only."""
    entities = sorted(contract["dollar_adv"])
    indexed = {(row["entity_id"], row["day"]): row["forward_return"] for row in panel}
    days = range(contract["training_start"], contract["training_end"])
    values = [[indexed[entity, day] for day in days] for entity in entities]
    means = [statistics.mean(series) for series in values]
    n = len(days)
    if n < 2:
        raise ValueError("risk estimation needs at least two training returns")
    shrinkage = contract["covariance_diagonal_shrinkage"]
    covariance = [[sum((a - means[i]) * (b - means[j])
                       for a, b in zip(values[i], values[j])) / (n - 1)
                   * (1 if i == j else 1 - shrinkage)
                   for j in range(len(entities))] for i in range(len(entities))]
    if any(covariance[i][i] <= 0 for i in range(len(entities))):
        raise ValueError("training variances must be positive")
    return {"entities": entities, "covariance": covariance}


def forecast_vol(weights, risk):
    vector = [weights.get(entity, 0.0) for entity in risk["entities"]]
    variance = sum(vector[i] * vector[j] * risk["covariance"][i][j]
                   for i in range(len(vector)) for j in range(len(vector)))
    return math.sqrt(252 * max(0.0, variance))


def target_weights(observations, candidate, contract, risk):
    eligible = [row for row in observations if row[candidate["feature"]] is not None]
    ordered = sorted(eligible, key=lambda row: (candidate["direction"] * row[candidate["feature"]],
                                              row["entity_id"]))
    weights = {entity: 0.0 for entity in risk["entities"]}
    if len(ordered) < 4:
        return weights
    vol = {entity: math.sqrt(risk["covariance"][i][i]) for i, entity in enumerate(risk["entities"])}
    for sign, leg in ((-1, ordered[:2]), (1, ordered[-2:])):
        scores = {row["entity_id"]: 1 / vol[row["entity_id"]]
                  if candidate["allocation"] == "inverse_vol" else 1.0 for row in leg}
        total = sum(scores.values())
        for entity, score in scores.items():
            weights[entity] = sign * candidate["gross"] / 2 * score / total
    gross = sum(abs(w) for w in weights.values())
    maximum = max(abs(w) for w in weights.values())
    estimated = forecast_vol(weights, risk)
    scale = min(1.0, contract["gross_cap"] / gross, contract["per_asset_cap"] / maximum,
                contract["annual_vol_cap"] / estimated if estimated else 1.0)
    return {entity: weight * scale for entity, weight in weights.items()}


def execute(previous, target, contract, risk):
    """One common fill fraction preserves neutrality and convex risk limits."""
    requested = {entity: target[entity] - previous[entity] for entity in risk["entities"]}
    fraction = min([1.0] + [contract["participation_limit"] * contract["dollar_adv"][entity]
                           / contract["reference_nav"] / abs(delta)
                           for entity, delta in requested.items() if delta != 0])
    weights = dict(target) if fraction == 1 else {
        entity: previous[entity] + fraction * delta for entity, delta in requested.items()}
    trades = {entity: weights[entity] - previous[entity] for entity in weights}
    turnover = sum(abs(delta) for delta in trades.values())
    participation = {entity: abs(delta) * contract["reference_nav"] / contract["dollar_adv"][entity]
                     for entity, delta in trades.items()}
    impact = contract["impact_bps"] / 10000 * sum(abs(trades[e]) * math.sqrt(participation[e]) for e in trades)
    short = sum(-w for w in weights.values() if w < 0)
    cost = contract["spread_bps"] / 10000 * turnover + impact + contract["borrow_bps_daily"] / 10000 * short
    return weights, {"actual_gross": sum(abs(w) for w in weights.values()),
                     "forecast_vol": forecast_vol(weights, risk),
                     "max_participation": max(participation.values()),
                     "execution_fraction": fraction, "turnover": turnover,
                     "impact_cost": impact, "total_cost": cost}


def simulate(panel, candidate, contract, risk):
    by_day = {}
    for row in panel:
        by_day.setdefault(row["day"], []).append(row)
    previous = {entity: 0.0 for entity in risk["entities"]}
    history = []
    for day in range(contract["validation_start"], contract["validation_end"]):
        observations = by_day[day]
        active = day < contract["active_end"]
        target = target_weights(observations, candidate, contract, risk) if active else dict.fromkeys(previous, 0.0)
        weights, metrics = execute(previous, target, contract, risk)
        gross_return = sum(weights[row["entity_id"]] * row["forward_return"] for row in observations)
        history.append({"day": day, "weights": weights, **metrics, "net_return": gross_return - metrics["total_cost"]})
        previous = weights
    if any(abs(weight) > 1e-12 for weight in previous.values()):
        raise ValueError("winddown failed to liquidate the portfolio")
    returns = [row["net_return"] for row in history]
    mean, std = statistics.mean(returns), statistics.stdev(returns)
    return {"id": candidate["id"], "mean_net": mean,
            "sharpe": math.sqrt(252) * mean / std if std else 0.0,
            "mean_turnover": statistics.mean(row["turnover"] for row in history),
            "screening_score": round(mean - 2 * std / math.sqrt(len(returns)), 12)}, history


def trial_table(panel, contract, risk=None):
    risk = estimate_risk(panel, contract) if risk is None else risk
    trials, histories = [], {}
    for candidate in contract["candidates"]:
        trial, history = simulate(panel, candidate, contract, risk)
        trials.append(trial)
        histories[candidate["id"]] = history
    return trials, histories


def combined_audit(panel, histories, requests):
    source = [r for r in requests if r["entity_id"] not in histories]
    values = {(r["entity_id"], r["day"], r["field"]): r["value"] for r in audit_values(panel, source)}
    indexed = {(identity, row["day"]): row for identity, history in histories.items() for row in history}
    return [{**r, "value": indexed[r["entity_id"], r["day"]][r["field"]]
             if r["entity_id"] in histories else values[r["entity_id"], r["day"], r["field"]]} for r in requests]
