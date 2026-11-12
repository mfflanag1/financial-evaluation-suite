"""Finite synthetic pilot registry. Never install this module in a solver sandbox."""

import csv
import hashlib
import hmac
import io
import json
import os
import random
import statistics
from functools import lru_cache

from .analytics import CANDIDATES, FEATURES, audit_values, clean_public, trial_table

VERSION = "research-pilot-0.2.0"
FAMILIES = ("signal", "null", "cost_trap", "revision_trap")
SPLITS = {"train": (0, 128), "dev": (128, 160), "test": (160, 192)}


def _id(seed, split, family, key):
    message = f"id/{VERSION}/{split}/{seed}/{family}".encode()
    return "fr_" + hmac.new(key.encode(), message, hashlib.sha256).hexdigest()[:24]


def _key():
    key = os.environ.get("FINANCIAL_RESEARCH_KEY", "")
    if len(key) < 32:
        raise ValueError("Set FINANCIAL_RESEARCH_KEY to an evaluator-only random key of at least 32 characters")
    return key


def _csv(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _raw(seed, family, n_days, feature_index, direction):
    rng = random.Random(seed)
    securities = []
    for asset in range(8):
        # A ticker is renamed and then reused by a different entity.
        ticker = f"S{asset:02d}"
        if asset == 0:
            securities += [{"entity_id": "E00", "ticker": ticker, "valid_from": 0, "valid_to": 59},
                           {"entity_id": "E00", "ticker": "REN", "valid_from": 60, "valid_to": n_days}]
        elif asset == 1:
            securities += [{"entity_id": "E01", "ticker": ticker, "valid_from": 0, "valid_to": 59},
                           {"entity_id": "E01", "ticker": "S00", "valid_from": 60, "valid_to": n_days}]
        else:
            securities.append({"entity_id": f"E{asset:02d}", "ticker": ticker,
                               "valid_from": 0, "valid_to": n_days})
    prices, records = [], []
    levels = [100.0] * 8
    for day in range(n_days + 1):
        common = rng.gauss(0, 0.003)
        for asset in range(8):
            entity = f"E{asset:02d}"
            ticker = ("REN" if asset == 0 else "S00") if asset in (0, 1) and day >= 60 else f"S{asset:02d}"
            factor = 2 if (asset == 2 and day >= 37) or (asset == 5 and day >= 75) else 1
            prices.append({"ticker": ticker, "day": day,
                           "close": format(levels[asset] / factor, ".15g"), "shares_per_original": factor})
            values = [rng.gauss(0, 1) for _ in FEATURES]
            beta = 0.010 if family == "signal" else 0.003 if family == "cost_trap" else 0
            ret = common + rng.gauss(0, 0.006) + beta * direction * values[feature_index]
            ret = max(-0.15, min(0.15, ret))
            # metric_d measures an outcome first published after the decision.
            values[3] = ret * 100 if family == "revision_trap" else rng.gauss(0, 1)
            for index, metric in enumerate(FEATURES):
                if rng.random() < 0.04:
                    continue
                row = {"entity_id": entity, "observation_day": day,
                       "available_day": day + (1 if index == 3 else 0),
                       "revision": 0, "metric": metric, "value": format(values[index], ".15g")}
                records.append(row)
                if rng.random() < 0.02:
                    records.append(dict(row))
                if index == 1:
                    records.append({**row, "available_day": day + 2, "revision": 1,
                                    "value": format(ret * 100 if family == "revision_trap"
                                                    else values[index] + rng.gauss(0, 0.1), ".15g")})
            levels[asset] *= 1 + ret
    rng.shuffle(records)
    rng.shuffle(prices)
    return {"prices.csv": _csv(prices), "features.csv": _csv(records), "securities.csv": _csv(securities)}


def build_episode(seed, split="train", family=None):
    if type(seed) is not int or split not in SPLITS or not SPLITS[split][0] <= seed < SPLITS[split][1]:
        raise ValueError("seed is outside the requested split")
    family = FAMILIES[seed % len(FAMILIES)] if family is None else family
    if family not in FAMILIES:
        raise ValueError("unknown family")
    return _build(seed, split, family, _key())


@lru_cache(maxsize=16)
def _build(seed, split, family, key):
    # Each episode has independent observed and withheld paths under one mechanism.
    identity = _id(seed, split, family, key)
    root = int(hmac.new(key.encode(), f"public/{identity}".encode(), hashlib.sha256).hexdigest(), 16)
    rng = random.Random(root)
    feature_index, direction = rng.randrange(3), rng.choice((-1, 1))
    files = _raw(root, family, 120, feature_index, direction)
    requests = [{"entity_id": f"E{asset:02d}", "day": day, "field": field}
                for asset, day in ((0, 0), (1, 60), (2, 36), (2, 37), (5, 74), (7, 119))
                for field in ("forward_return", *FEATURES)]
    contract = {"version": VERSION, "validation_start": 72, "validation_end": 120,
                "training_start": 0, "training_end": 72, "cost_bps": 100 if family == "cost_trap" else 8,
                "borrow_bps_daily": 0.1, "gross_cap": 1.0, "per_asset_cap": 0.25,
                "net_exposure": 0.0, "candidates": CANDIDATES, "audit_requests": requests,
                "rules": [
                    "Days are consecutive trading sessions. No dividends. Feature rows may be duplicated, missing, shuffled or revised.",
                    "Use entity_id as identity. Match ticker using inclusive securities validity intervals. Adjust close by shares_per_original.",
                    "Forward return at d is adjusted_close(d+1)/adjusted_close(d)-1. Decision uses information available at d.",
                    "For each entity and metric at d, keep rows with observation_day<=d and available_day<=d; choose maximum (observation_day,available_day,revision). If none, value=null. Duplicate identical rows are one record.",
                    "Run EVERY declared candidate on validation days [72,120); training is for exploration. Missing features exclude assets.",
                    "Daily rank ascending by (direction*feature,entity_id). Short lowest two, long highest two, weight magnitude gross/4. Fewer than four eligible assets means cash.",
                    "Start validation flat. Turnover is sum absolute weight changes (no factor of 1/2); add final liquidation to the last day's turnover.",
                    "Net return=sum(weight*forward_return)-cost_bps/10000*turnover-borrow_bps_daily/10000*short_gross. No risk-free rate subtraction.",
                    "mean_net=arithmetic daily mean; sharpe=sqrt(252)*mean_net/sample_stdev (zero if stdev zero); mean_turnover=arithmetic daily mean.",
                    "screening_score=mean_net-2*sample_stdev/sqrt(48). This is a fixed screening heuristic, not a significance or multiple-testing claim.",
                    "Select maximum screening_score with lexicographically smallest candidate id breaking ties. Deploy only if that maximum is strictly positive; otherwise abstain.",
                    "Submit only keys audit,trials,decision,selected in policy.json. audit: one object per request with entity_id,day,field,value. trials: one object per candidate with id,mean_net,sharpe,mean_turnover,screening_score. decision=deploy|abstain; selected=candidate id|null.",
                    "Absolute tolerances: audit 1e-8; trial mean_net/screening_score 1e-8, sharpe/mean_turnover 1e-6. Numeric booleans and nonfinite numbers are invalid.",
                    "Save analysis source as research.py: a nonempty regular UTF-8 file no larger than 1 MiB. Native grading reruns it with Python standard library only in a fresh public-data container. It must write policy.json within 30 wall-clock seconds and 256 MiB address space. No original policy.json, evaluator code, secret key or private data is supplied. Recreated output must match the submitted policy under the same numerical tolerances; missing, failed or mismatched output yields zero reward. Do not depend on extra files or external data.",
                    "Reward: 0.25 audit accuracy +0.35 trial accuracy +0.20 correct validation decision +0.20 withheld-path portfolio quality. Economic credit requires all audit/trial/decision checks to pass. Private quality averages screening scores across six independent 64-session paths from the same episode mechanism, each starting and ending flat and using sqrt(64) instead of sqrt(48) in its screening score. No access to those paths is provided.",
                    "An abstention is fully credited economically if no candidate has positive mean withheld screening score; otherwise its economic credit is zero. Deployment economic credit is its positive withheld screening score divided by the best positive candidate score, clipped to [0,1]."
                ]}
    files["contract.json"] = json.dumps(contract, indent=2)
    panel = clean_public(files)
    trials = trial_table(panel, contract)
    hidden = []
    hidden_contract = {**contract, "validation_start": 0, "validation_end": 64}
    for path in range(6):
        hidden_seed = int(hmac.new(key.encode(), f"hidden/{identity}/{path}".encode(), hashlib.sha256).hexdigest(), 16)
        hidden_files = _raw(hidden_seed, family, 64, feature_index, direction)
        hidden.append(trial_table(clean_public(hidden_files), hidden_contract))
    utilities = {c["id"]: statistics.mean(next(r["screening_score"] for r in t if r["id"] == c["id"])
                                         for t in hidden) for c in CANDIDATES}
    return {"id": identity,
            "prompt": "Investigate the supplied synthetic financial research dataset. Read /workspace/contract.json for the exact point-in-time, strategy, validation and response contract. Analyze the three CSV files, save your analysis as /workspace/research.py, and write your final structured decision to /workspace/policy.json. You may use Python and shell tools. The task ends when you submit the two files. Do not use external data.",
            "public_files": files,
            "private": {"audit": audit_values(panel, requests), "trials": trials,
                        "hidden_utilities": utilities}}


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
