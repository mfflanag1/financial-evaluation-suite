"""Reproducible outcome-blind cash baseline on explicitly public fixtures.

The policy uses release identities, census counts and inclusion probabilities.
It never uses observed markout values to calculate a statistic or allocation.
Input identity necessarily hashes the complete public files, including markouts.
This is a partial-credit control, not a solver or a private evaluation dataset.
"""
import csv
from fractions import Fraction
import io
import json
from statistics import mean

from .contract import input_digest


PUBLIC_FIXTURE_KEY = bytes(range(32))
CONDITIONS = ("mixed", "positive", "negative", "unsupported", "sparse")
FIXTURE_COUNT = 20


def _pair(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def policy(public):
    """Build an always-cash control from a valid public packet, ignoring markouts."""
    contract = json.loads(public["contract.json"])
    deployment = json.loads(public["deployment.json"])
    universe = list(csv.DictReader(io.StringIO(public["universe.csv"])))
    features = list(csv.DictReader(io.StringIO(public["feature_releases.csv"])))
    labels = list(csv.DictReader(io.StringIO(public["observed_labels.csv"])))
    populations = {cell: [] for cell in sorted(contract["cells"])}
    audits = []
    for row in sorted(universe, key=lambda record: record["event_id"]):
        event = row["event_id"]
        matured = int(row["label_maturity_tick"]) <= contract["research_cutoff"]
        audit = dict(event_id=event, population_member=matured, cell_id=None,
                     feature_release_id=None, label_release_id=None)
        if matured:
            feature = max(
                (item for item in features if item["event_id"] == event
                 and int(item["published_tick"]) <= int(row["decision_tick"])),
                key=lambda item: (int(item["published_tick"]), int(item["revision"])),
            )
            cell = feature["regime"] + (":negative" if int(feature["signal"]) < 0 else ":nonnegative")
            populations[cell].append(row)
            audit.update(cell_id=cell, feature_release_id=feature["release_id"])
            if row["archived"] == "1":
                label = max(
                    (item for item in labels if item["event_id"] == event
                     and int(item["published_tick"]) <= contract["research_cutoff"]),
                    key=lambda item: (int(item["published_tick"]), int(item["revision"])),
                )
                audit["label_release_id"] = label["release_id"]
        if event in contract["audit_event_ids"]:
            audits.append(audit)

    cells = []
    for cell, rows in populations.items():
        supported = bool(rows) and all(row["pi_num"] and int(row["pi_num"]) > 0 for row in rows)
        observed = [row for row in rows if row["archived"] == "1"]
        weights = [Fraction(int(row["pi_den"]), int(row["pi_num"])) for row in observed] if supported else []
        ess = sum(weights, Fraction()) ** 2 / sum((w*w for w in weights), Fraction()) if weights else Fraction()
        if not supported:
            status = "unsupported"
        elif len(observed) >= contract["minimum_observations"] and ess >= Fraction(*contract["minimum_ess"]):
            status = "eligible"
        else:
            status = "insufficient_sample"
        cells.append(dict(
            cell_id=cell, population_count=len(rows), observed_count=len(observed), status=status,
            mean_bp=[0, 1] if supported else None, variance_bp2=[0, 1] if supported else None,
            effective_sample_size=_pair(ess) if supported else None,
        ))
    by_cell = {row["cell_id"]: row for row in cells}
    baskets = []
    for basket in sorted(deployment["baskets"], key=lambda row: row["basket_id"]):
        required = [cell for cell, weight in basket["weights"].items() if weight[0]]
        supported = all(by_cell[cell]["status"] != "unsupported" for cell in required)
        baskets.append(dict(
            basket_id=basket["basket_id"],
            eligible=all(by_cell[cell]["status"] == "eligible" for cell in required),
            mean_bp=[0, 1] if supported else None,
        ))
    cash = deployment["initial_cash_cents"]
    interest = cash * Fraction(*deployment["lending_rate"])
    benchmark = cash + interest
    ledger = dict(
        invested_principal_cents=0, entry_cost_cents=0, exit_cost_cents=0,
        pretrade_cash_cents=cash, interest_cents=_pair(interest),
        estimated_gross_cents=[0, 1], estimated_terminal_cents=_pair(benchmark),
        cash_benchmark_cents=_pair(benchmark), estimated_excess_cents=[0, 1],
        sampling_variance_cents2=[0, 1], penalty_microcents=0, objective_microcents=0,
    )
    covariance = [
        [[0, 1] if left["mean_bp"] is not None and right["mean_bp"] is not None else None
         for right in baskets] for left in baskets
    ]
    return dict(
        schema="selection-policy-v1", input_sha256=input_digest(public), audit=audits,
        cells=cells, baskets=baskets, basket_covariance_cents2=covariance,
        allocation=[0] * len(baskets), ledger=ledger,
    )


def report():
    """Score 100 fixed public fixtures; return aggregate statistics, never policies."""
    from .generator import generate
    from .verifier import grade

    conditions = {}
    for condition in CONDITIONS:
        scores = []
        for index in range(FIXTURE_COUNT):
            public = generate(PUBLIC_FIXTURE_KEY, "dev", index, condition=condition).public_files
            scores.append(grade(public, json.dumps(policy(public))))
        conditions[condition] = dict(
            n=len(scores),
            full_policy_success_count=sum(score.get("outcome") == "correct" for score in scores),
            full_policy_success_rate=mean(score.get("outcome") == "correct" for score in scores),
            mean_reward=mean(score["reward"] for score in scores),
            minimum_reward=min(score["reward"] for score in scores),
            maximum_reward=max(score["reward"] for score in scores),
            mean_component={name: mean(score["components"][name] for score in scores)
                            for name in scores[0]["components"]},
            all_zero_optimal_count=sum(score["components"]["optimal_allocation"] == 1 for score in scores),
        )
    return dict(
        scope="Public deterministic fixture key; outcome-blind always-cash reward control, not model performance.",
        conditions=conditions,
        overall_mean_reward=mean(row["mean_reward"] for row in conditions.values()),
        overall_full_policy_success_count=sum(row["full_policy_success_count"] for row in conditions.values()),
        overall_full_policy_success_rate=mean(row["full_policy_success_rate"] for row in conditions.values()),
        fixture_identity=dict(key="bytes(range(32)); public fixture, never a private evaluation key",
                              split="dev", indices=list(range(FIXTURE_COUNT)), conditions=list(CONDITIONS)),
        not_live_model_data=True,
    )


def main():
    print(json.dumps(report(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
