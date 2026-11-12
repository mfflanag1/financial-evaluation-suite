"""Trusted, standard-library calculations for the bounded strategy contract."""

import csv
import io
import math
import statistics

FEATURES = ("metric_a", "metric_b", "metric_c", "metric_d")
CANDIDATES = tuple(
    {"id": f"{feature}:{direction}:{gross}", "feature": feature,
     "direction": direction, "gross": gross}
    for feature in FEATURES for direction in (-1, 1) for gross in (0.5, 1.0)
)
METRIC_NAMES = ("mean_net", "sharpe", "mean_turnover", "screening_score")


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def clean_public(files):
    """Resolve dated identifiers, splits and release-time eligibility."""
    securities = rows(files["securities.csv"])
    prices = {}
    for row in rows(files["prices.csv"]):
        day = int(row["day"])
        matches = [s for s in securities if s["ticker"] == row["ticker"]
                   and int(s["valid_from"]) <= day <= int(s["valid_to"])]
        if len(matches) != 1:
            raise ValueError("ambiguous security mapping")
        key = (matches[0]["entity_id"], day)
        price = float(row["close"]) * float(row["shares_per_original"])
        if key in prices and prices[key] != price:
            raise ValueError("conflicting price duplicates")
        prices[key] = price
    history = {}
    for row in rows(files["features.csv"]):
        key = (row["entity_id"], row["metric"])
        history.setdefault(key, []).append(row)
    entities = sorted({entity for entity, _ in prices})
    panel = []
    for day in range(max(day for _, day in prices)):
        for entity in entities:
            values = {}
            for metric in FEATURES:
                eligible = [r for r in history.get((entity, metric), ())
                            if int(r["observation_day"]) <= day
                            and int(r["available_day"]) <= day]
                if eligible:
                    latest = max(eligible, key=lambda r: (int(r["observation_day"]),
                                                         int(r["available_day"]), int(r["revision"])))
                    values[metric] = float(latest["value"])
                else:
                    values[metric] = None
            panel.append({"entity_id": entity, "day": day,
                          "forward_return": prices[entity, day + 1] / prices[entity, day] - 1,
                          **values})
    return panel


def portfolio_metrics(panel, candidate, contract, start, end):
    """Flat at both boundaries, close-to-close weights, one-way L1 turnover."""
    by_day = {}
    for row in panel:
        if start <= row["day"] < end:
            by_day.setdefault(row["day"], []).append(row)
    previous = {}
    net, turnovers = [], []
    for day in range(start, end):
        observations = by_day.get(day, [])
        eligible = [r for r in observations if r[candidate["feature"]] is not None]
        ordered = sorted(eligible, key=lambda r: (candidate["direction"] * r[candidate["feature"]],
                                                r["entity_id"]))
        weights = {}
        if len(ordered) >= 4:
            for row in ordered[:2]:
                weights[row["entity_id"]] = -candidate["gross"] / 4
            for row in ordered[-2:]:
                weights[row["entity_id"]] = candidate["gross"] / 4
        turnover = sum(abs(weights.get(entity, 0) - previous.get(entity, 0))
                       for entity in weights.keys() | previous.keys())
        short = sum(-weight for weight in weights.values() if weight < 0)
        gross_return = sum(weights.get(r["entity_id"], 0) * r["forward_return"] for r in observations)
        # Charge the final liquidation to the last scored day's return.
        if day == end - 1:
            turnover += sum(abs(w) for w in weights.values())
        net.append(gross_return - contract["cost_bps"] / 10000 * turnover
                   - contract["borrow_bps_daily"] / 10000 * short)
        turnovers.append(turnover)
        previous = weights
    mean = statistics.mean(net)
    std = statistics.stdev(net) if len(net) > 1 else 0
    return {"mean_net": mean, "sharpe": math.sqrt(252) * mean / std if std > 0 else 0,
            "mean_turnover": statistics.mean(turnovers),
            "screening_score": mean - 2 * std / math.sqrt(len(net))}


def trial_table(panel, contract):
    return [{"id": c["id"], **portfolio_metrics(panel, c, contract,
            contract["validation_start"], contract["validation_end"])} for c in CANDIDATES]


def select_candidate(trials):
    best = sorted(trials, key=lambda row: (-row["screening_score"], row["id"]))[0]
    return best["id"] if best["screening_score"] > 0 else None


def audit_values(panel, requests):
    indexed = {(r["entity_id"], r["day"]): r for r in panel}
    return [{**request, "value": indexed[request["entity_id"], request["day"]][request["field"]]}
            for request in requests]
