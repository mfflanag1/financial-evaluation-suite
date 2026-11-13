"""Public wire schema and units, without statistical or portfolio calculations."""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import io
import json
import re

INPUT_SCHEMA = "selection-input-v1"
OUTPUT_SCHEMA = "selection-policy-v1"
PUBLIC_FILES = ("contract.json", "deployment.json", "feature_releases.csv",
                "observed_labels.csv", "universe.csv")
HEADERS = {
    "universe.csv": ("event_id", "decision_tick", "label_maturity_tick", "archived", "pi_num", "pi_den"),
    "feature_releases.csv": ("event_id", "release_id", "published_tick", "revision", "signal", "regime"),
    "observed_labels.csv": ("event_id", "release_id", "published_tick", "revision", "markout_bp"),
}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ROWS = 4096
MAX_INTEGER = 10**12
MAX_PROBABILITY_COMPONENT = 1000
MAX_OUTPUT_DIGITS = 4096

INSTRUCTIONS = """Reconstruct every matured census event using the latest feature
published by its decision tick (then greatest revision). An archived matured
event uses its latest label published by research_cutoff (then greatest revision).
Population counts include unarchived events. Exact duplicates collapse; conflicting
keys are invalid. Cells combine regime with signal<0 (negative) or signal>=0
(nonnegative). Any zero/unknown inclusion probability or empty population makes
the cell unsupported; its estimates are null. Check every census member.
For supported cells: HT=sum(y/pi)/N; variance=sum((1-pi)*y*y/(pi*pi))/(N*N);
ESS=sum(1/pi)^2/sum(1/(pi*pi)). An empty archive gives all three zero. Below either
declared sample threshold the cell is insufficient_sample. A basket is eligible
only if every positive-weight cell is eligible. Known sampling is independent
Bernoulli; only the disclosed cell mix changes at deployment, with within-cell
outcome laws fixed. These estimates do not exactly recover missing outcomes.
For integer lots q within capacities, let a_g=sum(q_j*K_j*w_jg/10000) in cents/bp.
Estimated gross=sum(a_g*HT_g), sampling variance=sum(a_g*a_g*variance_g). Shared
cell estimates give per-lot basket covariance sum(K_j*w_jg*K_k*w_kg*variance_g/1e8).
Entry costs are linear*q+quadratic*q*q cents; exits linear*q cents. Opening cash
c=B-sum(q*K)-entry must be >=-credit_limit. Interest is c*lending_rate if c>=0,
otherwise c*borrowing_rate. Terminal wealth=sum(q*K)+gross+c+interest-exit.
Excess is terminal-B*(1+lending_rate). Rates already apply to this period.
Maximize floor(1e6*excess)-ceil(1e6*lambda*sqrt(sampling_variance)), then minimize
invested principal, then the lexicographic lot vector in sorted basket-ID order.
All-zero lots is feasible and has objective zero. Lambda is a screening heuristic,
not a post-selection confidence bound. Output exact reduced rational pairs and
integer counts/quantities under selection-policy-v1; no hidden economics enters
public calculation credit. Source reproduction is not verified by CPU scoring."""


class PacketError(ValueError):
    """The supplied public packet violates its declared input contract."""


def integer(value, *, lower=-MAX_INTEGER, upper=MAX_INTEGER):
    if type(value) is not int or not lower <= value <= upper:
        raise PacketError("invalid integer")
    return value


def ratio(value, *, nonnegative=False):
    if not isinstance(value, list) or len(value) != 2:
        raise PacketError("rational must be an integer pair")
    numerator, denominator = integer(value[0]), integer(value[1], lower=1)
    number = Fraction(numerator, denominator)
    if [number.numerator, number.denominator] != value or (nonnegative and number < 0):
        raise PacketError("rational must be reduced with the required sign")
    return number


def pair(number):
    number = Fraction(number)
    return [number.numerator, number.denominator]


def identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}", value) is None:
        raise PacketError("invalid identifier")
    return value


def keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise PacketError("unexpected object keys")


def strict_json(text):
    def unique(items):
        value = {}
        for name, item in items:
            if name in value:
                raise PacketError("duplicate JSON key")
            value[name] = item
        return value

    def reject(value):
        raise PacketError("JSON floating-point values are not allowed")

    def bounded_integer(value):
        if len(value.lstrip('-')) > MAX_OUTPUT_DIGITS:
            raise PacketError("JSON integer exceeds digit bound")
        return int(value)

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject,
                          parse_float=reject, parse_int=bounded_integer)
    except (ValueError, TypeError, RecursionError) as error:
        raise PacketError("invalid JSON") from error


def input_digest(public_files):
    """Sorted UTF-8 filename, NUL, unsigned 8-byte size, then exact UTF-8 bytes."""
    digest = hashlib.sha256()
    for name in sorted(PUBLIC_FILES):
        payload = public_files[name].encode("utf-8")
        digest.update(name.encode("utf-8") + b"\0" + len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def base_contract(cutoff, regimes=("active", "calm"), audit_ids=()):
    return {
        "schema": INPUT_SCHEMA, "research_cutoff": cutoff,
        "regimes": sorted(regimes),
        "cells": sorted(f"{regime}:{sign}" for regime in regimes for sign in ("negative", "nonnegative")),
        "minimum_observations": 4, "minimum_ess": [4, 1], "lambda": [2, 1],
        "audit_event_ids": sorted(audit_ids), "instructions": INSTRUCTIONS,
    }


def _csv(text, name):
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames != list(HEADERS[name]):
            raise PacketError("unexpected CSV header")
        rows = list(reader)
    except csv.Error as error:
        raise PacketError("invalid CSV") from error
    if len(rows) > MAX_ROWS or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise PacketError("invalid CSV shape")
    return rows


def _csv_integer(value, *, lower=-MAX_INTEGER):
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None:
        raise PacketError("invalid CSV integer")
    try:
        return integer(int(value), lower=lower)
    except ValueError as error:
        raise PacketError("invalid CSV integer") from error


def parse_public(public_files):
    """Validate raw records and units; never select releases or estimate outcomes."""
    keys(public_files, PUBLIC_FILES)
    for content in public_files.values():
        if not isinstance(content, str):
            raise PacketError("public file exceeds its text limit")
        try:
            size = len(content.encode("utf-8"))
        except UnicodeError as error:
            raise PacketError("public file is not valid UTF-8") from error
        if size > MAX_FILE_BYTES:
            raise PacketError("public file exceeds its text limit")
    contract = strict_json(public_files["contract.json"])
    keys(contract, base_contract(0))
    if contract["schema"] != INPUT_SCHEMA or contract["instructions"] != INSTRUCTIONS:
        raise PacketError("unknown financial contract")
    integer(contract["research_cutoff"], lower=0)
    regimes = contract["regimes"]
    if (not isinstance(regimes, list) or not 1 <= len(regimes) <= 4
            or any(not isinstance(item, str) for item in regimes)
            or regimes != sorted(set(regimes))):
        raise PacketError("invalid regime list")
    for regime in regimes:
        identifier(regime)
    if contract["cells"] != base_contract(0, regimes)["cells"]:
        raise PacketError("cell definitions differ from the public rule")
    integer(contract["minimum_observations"], lower=0, upper=512)
    if not 0 <= ratio(contract["minimum_ess"]) <= 512 or not 0 <= ratio(contract["lambda"]) <= 10:
        raise PacketError("invalid screening parameters")
    audit_ids = contract["audit_event_ids"]
    if (not isinstance(audit_ids, list) or len(audit_ids) > 32
            or any(not isinstance(item, str) for item in audit_ids)
            or audit_ids != sorted(set(audit_ids))):
        raise PacketError("invalid audit IDs")
    for event in audit_ids:
        identifier(event)

    universe = []
    for raw in _csv(public_files["universe.csv"], "universe.csv"):
        row = {"event_id": identifier(raw["event_id"]),
               "decision_tick": _csv_integer(raw["decision_tick"], lower=0),
               "label_maturity_tick": _csv_integer(raw["label_maturity_tick"], lower=0)}
        if row["label_maturity_tick"] <= row["decision_tick"] or raw["archived"] not in ("0", "1"):
            raise PacketError("invalid event chronology or archive flag")
        row["archived"] = raw["archived"] == "1"
        if raw["pi_num"] == raw["pi_den"] == "":
            row["probability"] = None
        else:
            probability = ratio([_csv_integer(raw["pi_num"], lower=0), _csv_integer(raw["pi_den"], lower=1)])
            if max(probability.numerator, probability.denominator) > MAX_PROBABILITY_COMPONENT:
                raise PacketError("sampling probability exceeds component bound")
            if (probability > 1 or (probability == 0 and row["archived"])
                    or (probability == 1 and not row["archived"])):
                raise PacketError("impossible sampling record")
            row["probability"] = probability
        universe.append(row)
    by_event = {row["event_id"]: row for row in universe}
    if len(by_event) != len(universe) or len(universe) > 512 or not set(audit_ids) <= set(by_event):
        raise PacketError("invalid census or audit membership")

    releases = {}
    for filename in ("feature_releases.csv", "observed_labels.csv"):
        rows, identities, versions = [], {}, {}
        for raw in _csv(public_files[filename], filename):
            event = identifier(raw["event_id"])
            if event not in by_event:
                raise PacketError("release refers to unknown event")
            row = {"event_id": event, "release_id": identifier(raw["release_id"]),
                   "published_tick": _csv_integer(raw["published_tick"], lower=0),
                   "revision": _csv_integer(raw["revision"], lower=0)}
            if filename == "feature_releases.csv":
                row.update(signal=_csv_integer(raw["signal"]), regime=raw["regime"])
                if row["regime"] not in regimes:
                    raise PacketError("undeclared regime")
            else:
                row["markout_bp"] = _csv_integer(raw["markout_bp"])
                if (not by_event[event]["archived"]
                        or row["published_tick"] < by_event[event]["label_maturity_tick"]):
                    raise PacketError("invalid label availability")
            identity = row["release_id"]
            version = (event, row["published_tick"], row["revision"])
            if (identity in identities and identities[identity] != row) or (version in versions and versions[version] != row):
                raise PacketError("conflicting release records")
            if identity not in identities:
                rows.append(row)
            identities[identity] = versions[version] = row
        releases[filename] = rows

    for event, row in by_event.items():
        if row['label_maturity_tick'] > contract['research_cutoff']:
            continue
        if not any(r['event_id'] == event and r['published_tick'] <= row['decision_tick']
                   for r in releases['feature_releases.csv']):
            raise PacketError('matured population member lacks eligible feature')
        if row['archived'] and not any(r['event_id'] == event and r['published_tick'] <= contract['research_cutoff']
                                      for r in releases['observed_labels.csv']):
            raise PacketError('archived population member lacks eligible label')

    deployment = strict_json(public_files["deployment.json"])
    keys(deployment, ("initial_cash_cents", "credit_limit_cents", "lending_rate", "borrowing_rate", "baskets"))
    integer(deployment["initial_cash_cents"], lower=0)
    integer(deployment["credit_limit_cents"], lower=0)
    lend, borrow = ratio(deployment["lending_rate"], nonnegative=True), ratio(deployment["borrowing_rate"], nonnegative=True)
    if not lend <= borrow <= 1:
        raise PacketError("invalid single-period cash rates")
    baskets = deployment["baskets"]
    if not isinstance(baskets, list) or not 1 <= len(baskets) <= 6:
        raise PacketError("invalid basket count")
    identities = set()
    for basket in baskets:
        keys(basket, ("basket_id", "weights", "principal_cents", "capacity", "entry_linear_cents", "entry_quadratic_cents", "exit_linear_cents"))
        identity = identifier(basket["basket_id"])
        if identity in identities:
            raise PacketError("duplicate basket ID")
        identities.add(identity)
        keys(basket["weights"], contract["cells"])
        if sum(ratio(weight, nonnegative=True) for weight in basket["weights"].values()) != 1:
            raise PacketError("basket weights must sum exactly to one")
        integer(basket["principal_cents"], lower=1)
        integer(basket["capacity"], lower=0, upper=3)
        for name in ("entry_linear_cents", "entry_quadratic_cents", "exit_linear_cents"):
            integer(basket[name], lower=0)
    return {"contract": contract, "deployment": deployment, "universe": universe,
            "features": releases["feature_releases.csv"], "labels": releases["observed_labels.csv"]}
