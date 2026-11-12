"""Evaluator-only synthetic paths and finite experimental episode registry."""

import hashlib
import hmac
import json
import math
import random
import statistics
from functools import lru_cache

from financial_research.analytics import FEATURES, clean_public, rows
from financial_research.generator import SPLITS, _csv, _key, _raw

from .analytics import CANDIDATES, PORTFOLIO_FIELDS, combined_audit, estimate_risk, trial_table

VERSION = "portfolio-experimental-0.3.0"
FAMILIES = ("impact_sizing", "liquidity_tight", "risk_concentration", "no_signal")


def _digest(key, domain):
    return hmac.new(key.encode(), f"{VERSION}/{domain}".encode(), hashlib.sha256).hexdigest()


def _id(seed, split, family, key):
    return "fp_" + _digest(key, f"id/{split}/{seed}/{family}")[:24]


def _settings(family, rng):
    # These are synthetic mechanisms, not estimates of executable market alpha.
    if family == "impact_sizing":
        return {"alpha": rng.choice((0.003, 0.005, 0.008)), "noise": [0.006] * 8,
                "adv_ratios": [3.0] * 8, "impact_bps": 70.0,
                "spread_bps": 2.0, "participation_limit": 0.25, "annual_vol_cap": 0.20}
    if family == "liquidity_tight":
        return {"alpha": 0.006, "noise": [0.006] * 8,
                "adv_ratios": [0.35, 0.5, 0.7, 1.2, 0.45, 0.9, 0.6, 1.0], "impact_bps": 35.0,
                "spread_bps": 2.0, "participation_limit": 0.10, "annual_vol_cap": 0.20}
    if family == "risk_concentration":
        return {"alpha": 0.006, "noise": [0.004, 0.006, 0.008, 0.012, 0.020, 0.030, 0.040, 0.050],
                "adv_ratios": [4.0] * 8, "impact_bps": 2.0,
                "spread_bps": 1.0, "participation_limit": 0.25, "annual_vol_cap": 0.10}
    return {"alpha": 0.0, "noise": [0.006] * 8,
            "adv_ratios": [3.0] * 8, "impact_bps": 20.0,
            "spread_bps": 8.0, "participation_limit": 0.25, "annual_vol_cap": 0.20}


def _path(seed, active_end, winddown, feature_index, direction, settings):
    """Prices respond only to features eligible at the preceding decision."""
    files = _raw(seed, "null", active_end + winddown, feature_index, direction)
    panel = clean_public(files)
    indexed = {(row["entity_id"], row["day"]): row for row in panel}
    rng = random.Random(seed ^ int.from_bytes(b"portfolio-return-noise", "big"))
    levels = {(f"E{asset:02d}", 0): 100.0 for asset in range(8)}
    for day in range(active_end + winddown):
        common = rng.gauss(0, 0.003)
        for asset in range(8):
            entity = f"E{asset:02d}"
            signal = indexed[entity, day][FEATURES[feature_index]]
            ret = (common + rng.gauss(0, settings["noise"][asset])
                   + settings["alpha"] * direction * (signal if signal is not None else 0)) if day < active_end else 0.0
            ret = min(0.20, max(-0.20, ret))
            levels[entity, day + 1] = levels[entity, day] * (1 + ret)
    securities = rows(files["securities.csv"])
    prices = rows(files["prices.csv"])
    for row in prices:
        day = int(row["day"])
        entity = next(s["entity_id"] for s in securities if s["ticker"] == row["ticker"]
                      and int(s["valid_from"]) <= day <= int(s["valid_to"]))
        row["close"] = format(levels[entity, day] / float(row["shares_per_original"]), ".15g")
    files["prices.csv"] = _csv(prices)
    return files


RULES = [
    "All data are synthetic. Days are consecutive sessions. Exposures are fixed-reference-capital return notionals, not physical share holdings: no wealth denominator changes or weight drift. Daily P&L is divided by fixed reference_nav. No dividends or risk-free subtraction.",
    "Resolve ticker using inclusive securities validity intervals and use entity_id as identity. Adjust close by shares_per_original. Forward return at d is adjusted_close(d+1)/adjusted_close(d)-1, known at d+1.",
    "At decision d use feature rows with observation_day<=d and available_day<=d; choose maximum (observation_day,available_day,revision) for each entity/metric. Identical duplicates count once. Missing history means null, not zero. Inputs may be shuffled, missing or revised.",
    "Estimate sample covariance of training forward returns on [training_start,training_end), with denominator n-1, across entities sorted by id. Keep diagonal entries; multiply off-diagonal entries by 1-covariance_diagonal_shrinkage. All variances are positive. Freeze this matrix for validation and withheld paths. Training ends by validation_start, so all training returns are available before the first trade.",
    "Run every declared candidate. During active validation [validation_start,active_end), rank eligible assets ascending by (direction*feature,entity_id), short lowest two and long highest two. Fewer than four eligible assets means a zero target; existing exposures still unwind under execution limits.",
    "Give each leg gross/2. For equal allocation split each leg equally; for inverse_vol divide within each leg in proportion to 1/sqrt(training covariance diagonal). Normalize long and short legs separately.",
    "Scale the complete target by min(1,gross_cap/sum(abs(w)),per_asset_cap/max(abs(w)),annual_vol_cap/sqrt(252*w'Cov*w)). A zero denominator contributes 1. Never clip legs independently. The volatility limit is a model estimate, not a realized risk guarantee.",
    "Start flat. dollar_adv is known dollar volume per session and reference_nav is fixed capital. For requested trades delta_i=target_i-previous_i, set execution_fraction=min(1,min over nonzero deltas of participation_limit*dollar_adv_i/reference_nav/abs(delta_i)), where dollar_adv_i is the dictionary value for entity i. Actual weights=previous+execution_fraction*delta, using the target exactly when fraction=1. Previous means last executed fixed-notional weights.",
    "Actual trade_i=weight_i-previous_i; turnover=sum(abs(trade_i)); participation_i=abs(trade_i)*reference_nav/dollar_adv_i. spread_bps is a one-way cost. Spread cost=spread_bps/10000*turnover; impact_cost=impact_bps/10000*sum(abs(trade_i)*sqrt(participation_i)); borrow cost=borrow_bps_daily/10000*sum(abs(negative executed weights)). Charge costs to actual trades/exposures, never desired targets.",
    "net_return=sum(actual weight_i*forward_return_i)-spread cost-impact_cost-borrow cost. actual_gross=sum(abs(weights)); forecast_vol=sqrt(252*w'Cov*w); max_participation=max(participation_i). These and execution_fraction,turnover,impact_cost,net_return are portfolio audit fields.",
    "After the final active interval's return, target zero for exactly winddown_sessions=ceil(max_i(per_asset_cap/(participation_limit*dollar_adv_i/reference_nav))) extra intervals. These explicit synthetic closeout intervals have zero asset returns and unchanged fill limits, costs and borrow. Continue scoring every interval through validation_end=active_end+winddown_sessions, including already-flat cash intervals. All candidates end flat without an unconstrained final trade.",
    "mean_net is arithmetic mean across ALL validation and winddown intervals; sharpe=sqrt(252)*mean_net/sample_stdev, or zero for zero stdev; mean_turnover is the mean L1 actual turnover. screening_score=round(mean_net-2*sample_stdev/sqrt(number of scored intervals),12), Python round semantics. This canonical precision reduces numerical differences among equivalent constrained policies. It is a screening heuristic, not a significance or multiple-testing claim.",
    "Select maximum screening_score, breaking exact ties by lexicographically smallest candidate id. Deploy only if the maximum rounded score is strictly positive; otherwise abstain. The 48 labels may produce fewer distinct executed policies.",
    "Source audit requests use asset entity_id and standard feature/forward_return fields. Portfolio audit requests use candidate id as entity_id, the scored day, and a portfolio audit field.",
    "Submit only audit,trials,decision,selected in policy.json. audit has exactly one {entity_id,day,field,value} per request. trials has exactly one {id,mean_net,sharpe,mean_turnover,screening_score} per candidate. decision is deploy or abstain; selected is candidate id or null. Absolute tolerances: audit and mean_net/screening_score 1e-8; sharpe/mean_turnover 1e-6. Boolean/nonfinite numeric values are invalid.",
    "Save a nonempty regular UTF-8 research.py, at most 1 MiB, using Python standard library only. It must recreate policy.json within 30 wall seconds and 256 MiB address space in a fresh public-data container. No original output, secret, evaluator package, private data, extra files or external access is supplied. A failed or mismatched recreation earns zero reward.",
    "Reward: .25 audit accuracy +.35 trial accuracy +.20 validation decision +.20 withheld quality. Economic credit requires perfect audit/trial/decision checks. Six independent withheld paths use the same mechanism, frozen PUBLIC training risk matrix and contract, each with 64 active intervals plus the same winddown. Their screening scores use the full 64+winddown denominator and the same rounding. Hidden utility is their mean screening score.",
    "Abstention earns full economic credit if every candidate's mean withheld score is nonpositive, otherwise zero. Deployment earns max(0,min(1,chosen mean withheld score/best positive mean withheld score)), or zero if none is positive. A correct public analysis may receive less than full reward because selection and withheld paths differ."
]


def build_episode(seed, split="train", family=None):
    if type(seed) is not int or split not in SPLITS or not SPLITS[split][0] <= seed < SPLITS[split][1]:
        raise ValueError("seed is outside the requested split")
    family = FAMILIES[seed % len(FAMILIES)] if family is None else family
    if family not in FAMILIES:
        raise ValueError("unknown family")
    return _build(seed, split, family, _key())


@lru_cache(maxsize=16)
def _build(seed, split, family, key):
    identity = _id(seed, split, family, key)
    root = int(_digest(key, f"public/{identity}"), 16)
    rng = random.Random(root)
    feature_index, direction = rng.randrange(3), rng.choice((-1, 1))
    settings = _settings(family, rng)
    nav = 10_000_000.0
    adv = {f"E{i:02d}": ratio * nav for i, ratio in enumerate(settings["adv_ratios"])}
    winddown = math.ceil(max(0.35 / (settings["participation_limit"] * value / nav) for value in adv.values()))
    if not 1 <= winddown <= 16:
        raise ValueError("invalid generated liquidity limits")
    requests = [{"entity_id": f"E{asset:02d}", "day": day, "field": field}
                for asset, day in ((0, 0), (1, 60), (2, 36), (2, 37), (5, 74), (7, 119))
                for field in ("forward_return", *FEATURES)]
    requests += [{"entity_id": candidate["id"], "day": day, "field": field}
                 for candidate in CANDIDATES for day in (72, 73, 119, 120, 120 + winddown - 1)
                 for field in PORTFOLIO_FIELDS]
    # A one-session winddown makes the final two checkpoint days identical.
    requests = list({(r["entity_id"], r["day"], r["field"]): r for r in requests}.values())
    contract = {"version": VERSION, "rules": RULES, "training_start": 0, "training_end": 72,
                "validation_start": 72, "active_end": 120, "validation_end": 120 + winddown,
                "winddown_sessions": winddown, "reference_nav": nav, "dollar_adv": adv,
                "spread_bps": settings["spread_bps"], "impact_bps": settings["impact_bps"],
                "participation_limit": settings["participation_limit"], "annual_vol_cap": settings["annual_vol_cap"],
                "covariance_diagonal_shrinkage": 0.5, "borrow_bps_daily": 0.1,
                "gross_cap": 1.0, "per_asset_cap": 0.35, "net_exposure": 0.0,
                "candidates": CANDIDATES, "audit_requests": requests}
    files = _path(root, 120, winddown, feature_index, direction, settings)
    files["contract.json"] = json.dumps(contract, indent=2)
    panel = clean_public(files)
    risk = estimate_risk(panel, contract)
    trials, histories = trial_table(panel, contract, risk)
    hidden = []
    hidden_contract = {**contract, "validation_start": 0, "active_end": 64, "validation_end": 64 + winddown}
    for path in range(6):
        seed = int(_digest(key, f"hidden/{identity}/{path}"), 16)
        hidden_files = _path(seed, 64, winddown, feature_index, direction, settings)
        hidden_trials, _ = trial_table(clean_public(hidden_files), hidden_contract, risk)
        hidden.append({r["id"]: r["screening_score"] for r in hidden_trials})
    utilities = {c["id"]: statistics.mean(path[c["id"]] for path in hidden) for c in CANDIDATES}
    signatures = {identity: tuple(round(weight, 10) for row in history for weight in row["weights"].values())
                  for identity, history in histories.items()}
    ranked = sorted(trials, key=lambda row: (-row["screening_score"], row["id"]))
    distinct_runner_up = next((row for row in ranked[1:]
                               if signatures[row["id"]] != signatures[ranked[0]["id"]]), None)
    return {"id": identity,
            "prompt": "Investigate the supplied synthetic portfolio research dataset. Read /workspace/contract.json for the exact point-in-time, covariance, sizing, nonlinear cost, liquidity, risk and response rules. Analyze the three CSV files, save your analysis as /workspace/research.py and write /workspace/policy.json. Use Python and shell tools without external data. The task ends when you submit the two files.",
            "public_files": files,
            "private": {"audit": combined_audit(panel, histories, requests), "trials": trials,
                        "hidden_utilities": utilities,
                        "diagnostics": {"effective_policy_count": len(set(signatures.values())),
                                        "label_winner_gap": ranked[0]["screening_score"] - ranked[1]["screening_score"],
                                        "distinct_policy_winner_gap": ranked[0]["screening_score"] - distinct_runner_up["screening_score"]
                                        if distinct_runner_up else None}}}


@lru_cache(maxsize=1)
def _registry(key):
    return {_id(seed, split, family, key): (seed, split, family)
            for split, (lo, hi) in SPLITS.items() for seed in range(lo, hi) for family in FAMILIES}


def resolve_episode(opaque_id):
    try:
        seed, split, family = _registry(_key())[opaque_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("unrecognized task id") from exc
    return build_episode(seed, split, family)
