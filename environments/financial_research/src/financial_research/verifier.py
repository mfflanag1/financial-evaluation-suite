"""Deterministic grading. Candidate Python is never evaluated by this module."""

import copy
import math

from .analytics import METRIC_NAMES, select_candidate


def reference_submission(episode):
    private = episode["private"]
    selected = select_candidate(private["trials"])
    return {"audit": copy.deepcopy(private["audit"]), "trials": copy.deepcopy(private["trials"]),
            "decision": "deploy" if selected else "abstain", "selected": selected}


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _close(actual, expected, tol):
    if expected is None:
        return actual is None
    return _number(actual) and abs(actual - expected) <= tol


def _invalid():
    return {"reward": 0.0, "metrics": {"schema_valid": 0.0, "audit_accuracy": 0.0,
            "trial_accuracy": 0.0, "decision_correct": 0.0, "economic_quality": 0.0}}


def score_submission(episode, submission):
    if not isinstance(submission, dict) or set(submission) != {"audit", "trials", "decision", "selected"}:
        return _invalid()
    private = episode["private"]
    audit, trials = submission["audit"], submission["trials"]
    if not isinstance(audit, list) or not isinstance(trials, list):
        return _invalid()
    if len(audit) != len(private["audit"]) or len(trials) != len(private["trials"]):
        return _invalid()
    expected_audit = {(r["entity_id"], r["day"], r["field"]): r["value"] for r in private["audit"]}
    seen, good_audit = set(), 0
    for row in audit:
        if not isinstance(row, dict) or set(row) != {"entity_id", "day", "field", "value"}:
            return _invalid()
        if type(row["day"]) is not int or not isinstance(row["entity_id"], str) or not isinstance(row["field"], str):
            return _invalid()
        if row["value"] is not None and not _number(row["value"]):
            return _invalid()
        key = (row["entity_id"], row["day"], row["field"])
        if key in seen or key not in expected_audit:
            return _invalid()
        seen.add(key)
        good_audit += _close(row["value"], expected_audit[key], 1e-8)
    expected_trials = {r["id"]: r for r in private["trials"]}
    seen, good_trials = set(), 0
    for row in trials:
        if not isinstance(row, dict) or set(row) != {"id", *METRIC_NAMES} or not isinstance(row["id"], str):
            return _invalid()
        if row["id"] in seen or row["id"] not in expected_trials or not all(_number(row[m]) for m in METRIC_NAMES):
            return _invalid()
        seen.add(row["id"])
        for metric in METRIC_NAMES:
            good_trials += _close(row[metric], expected_trials[row["id"]][metric],
                                 1e-6 if metric in ("sharpe", "mean_turnover") else 1e-8)
    if submission["decision"] not in ("deploy", "abstain"):
        return _invalid()
    chosen = submission["selected"]
    if chosen is not None and (not isinstance(chosen, str) or chosen not in expected_trials):
        return _invalid()
    if (submission["decision"] == "abstain") != (chosen is None):
        return _invalid()
    audit_accuracy = good_audit / len(audit)
    trial_accuracy = good_trials / (len(trials) * len(METRIC_NAMES))
    decision_correct = float(chosen == select_candidate(private["trials"]))
    economics = 0.0
    if audit_accuracy == trial_accuracy == decision_correct == 1:
        utilities = private["hidden_utilities"]
        best = max(utilities.values())
        if chosen is None:
            economics = float(best <= 0)
        elif best > 0:
            economics = max(0, min(1, utilities[chosen] / best))
    return {"reward": 0.25 * audit_accuracy + 0.35 * trial_accuracy + 0.2 * decision_correct + 0.2 * economics,
            "metrics": {"schema_valid": 1.0, "audit_accuracy": audit_accuracy,
                        "trial_accuracy": trial_accuracy, "decision_correct": decision_correct,
                        "economic_quality": economics}}
