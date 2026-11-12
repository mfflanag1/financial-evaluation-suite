"""Public-input reference analysis for the bounded financial research pilot.

Copy this file beside the four supplied data files, then run ``python research.py``.
Only the supplied CSV files and contract are read. No evaluator imports are used.
"""

import csv
import json
import math
from pathlib import Path


def read_csv(filename):
    with Path(filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assemble_panel(prices, features, securities, fields):
    """Construct adjusted entity prices and a release-time feature event stream."""
    ticker_intervals = {}
    for security in securities:
        ticker_intervals.setdefault(security["ticker"], []).append(
            (int(security["valid_from"]), int(security["valid_to"]), security["entity_id"])
        )
    levels = {}
    for price in prices:
        day = int(price["day"])
        identities = [entity for start, end, entity in ticker_intervals[price["ticker"]]
                      if start <= day <= end]
        if len(identities) != 1:
            raise ValueError("Price requires one dated entity mapping")
        key = (identities[0], day)
        adjusted = float(price["close"]) * float(price["shares_per_original"])
        if key in levels and levels[key] != adjusted:
            raise ValueError("Conflicting adjusted prices")
        levels[key] = adjusted

    # Identical source duplicates collapse before the event stream is formed.
    revisions = {}
    for feature in features:
        observation = int(feature["observation_day"])
        available = int(feature["available_day"])
        revision = int(feature["revision"])
        key = (feature["entity_id"], feature["metric"], observation, available, revision)
        value = float(feature["value"])
        if key in revisions and revisions[key] != value:
            raise ValueError("Conflicting feature revisions")
        revisions[key] = value
    events = sorted((max(observation, available), entity, metric,
                     (observation, available, revision), value)
                    for (entity, metric, observation, available, revision), value
                    in revisions.items())
    released = {}
    cursor = 0
    entities = sorted({entity for entity, _ in levels})
    panel = {}
    for day in range(max(day for _, day in levels)):
        while cursor < len(events) and events[cursor][0] <= day:
            _, entity, metric, priority, value = events[cursor]
            key = (entity, metric)
            if key not in released or released[key][0] < priority:
                released[key] = (priority, value)
            cursor += 1
        for entity in entities:
            panel[entity, day] = {
                "forward_return": levels[entity, day + 1] / levels[entity, day] - 1,
                **{field: released.get((entity, field), (None, None))[1] for field in fields},
            }
    return entities, panel


def analyze_candidate(candidate, contract, entities, panel):
    """Execute the declared rank portfolio and account for both boundary trades."""
    previous = [0.0] * len(entities)
    daily_returns, daily_turnover = [], []
    start, end = contract["validation_start"], contract["validation_end"]
    for day in range(start, end):
        ranking = sorted((candidate["direction"] * panel[entity, day][candidate["feature"]], entity)
                         for entity in entities
                         if panel[entity, day][candidate["feature"]] is not None)
        weights = dict.fromkeys(entities, 0.0)
        if len(ranking) >= 4:
            magnitude = candidate["gross"] / 4
            for _, entity in ranking[:2]:
                weights[entity] = -magnitude
            for _, entity in ranking[-2:]:
                weights[entity] = magnitude
        current = [weights[entity] for entity in entities]
        turnover = math.fsum(abs(new - old) for new, old in zip(current, previous))
        if day == end - 1:
            turnover += math.fsum(abs(weight) for weight in current)
        gross_return = math.fsum(weights[entity] * panel[entity, day]["forward_return"]
                                 for entity in entities)
        borrow = math.fsum(-weight for weight in current if weight < 0)
        daily_returns.append(gross_return - turnover * contract["cost_bps"] / 10000
                             - borrow * contract["borrow_bps_daily"] / 10000)
        daily_turnover.append(turnover)
        previous = current
    count = len(daily_returns)
    mean = math.fsum(daily_returns) / count
    variance = math.fsum((value - mean) ** 2 for value in daily_returns) / (count - 1) if count > 1 else 0
    deviation = math.sqrt(variance)
    return {"id": candidate["id"], "mean_net": mean,
            "sharpe": math.sqrt(252) * mean / deviation if deviation else 0.0,
            "mean_turnover": math.fsum(daily_turnover) / count,
            "screening_score": mean - 2 * deviation / math.sqrt(count)}


def solve():
    contract = json.loads(Path("contract.json").read_text(encoding="utf-8"))
    fields = sorted({candidate["feature"] for candidate in contract["candidates"]})
    entities, panel = assemble_panel(read_csv("prices.csv"), read_csv("features.csv"),
                                     read_csv("securities.csv"), fields)
    audit = [{**request, "value": panel[request["entity_id"], request["day"]][request["field"]]}
             for request in contract["audit_requests"]]
    trials = [analyze_candidate(candidate, contract, entities, panel)
              for candidate in contract["candidates"]]
    winner = min(trials, key=lambda row: (-row["screening_score"], row["id"]))
    selected = winner["id"] if winner["screening_score"] > 0 else None
    return {"audit": audit, "trials": trials, "decision": "deploy" if selected else "abstain",
            "selected": selected}


if __name__ == "__main__":
    Path("policy.json").write_text(json.dumps(solve(), indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
