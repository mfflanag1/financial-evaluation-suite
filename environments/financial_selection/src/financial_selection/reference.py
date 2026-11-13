"""Public-only standard-library reference; no evaluator/generator imports.

Run this source in a directory containing the five public files. It writes
policy.json. Execution and input portability require a separate runtime check.
"""

import csv
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

FILES = ("contract.json", "deployment.json", "feature_releases.csv",
         "observed_labels.csv", "universe.csv")


def _pair(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def _fraction(value):
    return Fraction(*value)


def _input_hash(public):
    result = hashlib.sha256()
    for name in sorted(FILES):
        payload = public[name].encode("utf-8")
        result.update(name.encode() + b"\0" + len(payload).to_bytes(8, "big") + payload)
    return result.hexdigest()


def _latest(rows, event, tick):
    eligible = [row for row in rows if row["event_id"] == event and int(row["published_tick"]) <= tick]
    if not eligible:
        raise ValueError("Matured population member lacks an eligible release")
    return max(eligible, key=lambda row: (int(row["published_tick"]), int(row["revision"])))


def _ceil_root(value):
    root = math.isqrt(value.numerator // value.denominator)
    return root + int(root * root * value.denominator < value.numerator)


def solve(public):
    """Solve a well-formed public packet using exact rational arithmetic."""
    contract = json.loads(public["contract.json"])
    deployment = json.loads(public["deployment.json"])
    if contract["schema"] != "selection-input-v1":
        raise ValueError("Unknown contract")
    census, features, labels = [list(csv.DictReader(io.StringIO(public[name]))) for name in
                                ("universe.csv", "feature_releases.csv", "observed_labels.csv")]
    cells = sorted(contract["cells"])
    groups = {cell: [] for cell in cells}
    audits = []
    for row in sorted(census, key=lambda item: item["event_id"]):
        event = row["event_id"]
        matured = int(row["label_maturity_tick"]) <= contract["research_cutoff"]
        audit = {"event_id": event, "population_member": matured,
                 "cell_id": None, "feature_release_id": None, "label_release_id": None}
        if matured:
            feature = _latest(features, event, int(row["decision_tick"]))
            cell = feature["regime"] + (":negative" if int(feature["signal"]) < 0 else ":nonnegative")
            observed = row["archived"] == "1"
            label = _latest(labels, event, contract["research_cutoff"]) if observed else None
            probability = None if row["pi_num"] == "" else Fraction(int(row["pi_num"]), int(row["pi_den"]))
            groups[cell].append((probability, observed, int(label["markout_bp"]) if label else None))
            audit.update(cell_id=cell, feature_release_id=feature["release_id"],
                         label_release_id=label["release_id"] if label else None)
        if event in contract["audit_event_ids"]:
            audits.append(audit)

    statistics, means, variances, eligible_cells = [], {}, {}, {}
    for cell in cells:
        members = groups[cell]
        observed = [row for row in members if row[1]]
        if not members or any(row[0] is None or row[0] == 0 for row in members):
            mean = variance = ess = None
            status = "unsupported"
        else:
            n = len(members)
            mean = sum((Fraction(y, 1) / p for p, _, y in observed), Fraction()) / n
            variance = sum(((1-p) * y*y / (p*p) for p, _, y in observed), Fraction()) / (n*n)
            weights = [1/p for p, _, _ in observed]
            ess = sum(weights)**2 / sum(w*w for w in weights) if weights else Fraction()
            status = ("eligible" if len(observed) >= contract["minimum_observations"]
                      and ess >= _fraction(contract["minimum_ess"]) else "insufficient_sample")
        means[cell], variances[cell], eligible_cells[cell] = mean, variance, status == "eligible"
        statistics.append({"cell_id": cell, "population_count": len(members), "observed_count": len(observed),
                           "status": status, "mean_bp": None if mean is None else _pair(mean),
                           "variance_bp2": None if variance is None else _pair(variance),
                           "effective_sample_size": None if ess is None else _pair(ess)})

    baskets = sorted(deployment["baskets"], key=lambda row: row["basket_id"])
    exposures, basket_rows = [], []
    for basket in baskets:
        weights = [_fraction(basket["weights"][cell]) for cell in cells]
        positive = [cell for cell, weight in zip(cells, weights) if weight]
        known = all(means[cell] is not None for cell in positive)
        eligible = all(eligible_cells[cell] for cell in positive)
        mean = sum((weight*means[cell] for cell, weight in zip(cells, weights) if weight), Fraction()) if known else None
        exposures.append([basket["principal_cents"] * weight / 10000 for weight in weights])
        basket_rows.append({"basket_id": basket["basket_id"], "eligible": eligible,
                            "mean_bp": None if mean is None else _pair(mean)})
    covariance = []
    for j, left in enumerate(exposures):
        row = []
        for k, right in enumerate(exposures):
            if basket_rows[j]["mean_bp"] is None or basket_rows[k]["mean_bp"] is None:
                row.append(None)
            else:
                row.append(_pair(sum((a*b*variances[cell] for cell, a, b in zip(cells, left, right) if a and b), Fraction())))
        covariance.append(row)

    base_cash = deployment["initial_cash_cents"]
    lending, borrowing = map(_fraction, (deployment["lending_rate"], deployment["borrowing_rate"]))
    benchmark = base_cash*(1+lending)
    penalty_coefficient = _fraction(contract["lambda"])
    best = None
    for allocation in itertools.product(*(range(basket["capacity"]+1) if row["eligible"] else range(1)
                                          for basket, row in zip(baskets, basket_rows))):
        invested = sum(q*basket["principal_cents"] for q, basket in zip(allocation, baskets))
        entry = sum(basket["entry_linear_cents"]*q + basket["entry_quadratic_cents"]*q*q
                    for q, basket in zip(allocation, baskets))
        cash = base_cash-invested-entry
        if cash < -deployment["credit_limit_cents"]:
            continue
        exit_fee = sum(q*basket["exit_linear_cents"] for q, basket in zip(allocation, baskets))
        exposure = [sum((q*values[g] for q, values in zip(allocation, exposures)), Fraction()) for g in range(len(cells))]
        gross = sum((a*means[cell] for cell, a in zip(cells, exposure) if a), Fraction())
        variance = sum((a*a*variances[cell] for cell, a in zip(cells, exposure) if a), Fraction())
        interest = cash*(lending if cash >= 0 else borrowing)
        terminal = invested+gross+cash+interest-exit_fee
        excess = terminal-benchmark
        penalty = _ceil_root((10**6*penalty_coefficient)**2*variance)
        objective = math.floor(10**6*excess)-penalty
        key = (-objective, invested, allocation)
        if best is None or key < best[0]:
            ledger = {"invested_principal_cents": invested, "entry_cost_cents": entry,
                      "exit_cost_cents": exit_fee, "pretrade_cash_cents": cash,
                      "interest_cents": _pair(interest), "estimated_gross_cents": _pair(gross),
                      "estimated_terminal_cents": _pair(terminal), "cash_benchmark_cents": _pair(benchmark),
                      "estimated_excess_cents": _pair(excess), "sampling_variance_cents2": _pair(variance),
                      "penalty_microcents": penalty, "objective_microcents": objective}
            best = (key, list(allocation), ledger)
    if best is None:
        raise ValueError("The cash allocation must always be feasible")
    return {"schema": "selection-policy-v1", "input_sha256": _input_hash(public), "audit": audits,
            "cells": statistics, "baskets": basket_rows, "basket_covariance_cents2": covariance,
            "allocation": best[1], "ledger": best[2]}


def main():
    public = {name: Path(name).read_bytes().decode("utf-8") for name in FILES}
    policy = solve(public)
    Path("policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
