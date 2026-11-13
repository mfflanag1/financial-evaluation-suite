"""Bind public packets, derive every alternate result, then compare audit code."""
import hashlib
import importlib
import json
import math
from pathlib import Path

from . import VERSION
from .methods import CHECKS


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
CASES = {
    "q033": ("questions/legacy/quant100/questions/q033.json", "risk_numerical"),
    "q065": ("questions/legacy/quant100/questions/q065.json", "numerical"),
    "q098": ("questions/legacy/quant100/questions/q098.json", "numerical"),
    "c201": ("questions/challenges/c201.json", "core_numerical"),
    "c214": ("questions/challenges/c214.json", "core_numerical"),
}
METHODS = {
    "q033": "Complete rational bond-and-cash grid versus minimum-cash completion",
    "q065": "Analytic normal and shifted Black expectations versus payoff quadrature",
    "q098": "Every complete binary path versus an absorbing-state dynamic program",
    "c201": "Signed trade search with separate settlement ledgers versus final-holdings search",
    "c214": "Reconstructed residuals and direct score enumeration with exact pairwise comparisons",
}


class InputDriftError(ValueError):
    """A public packet no longer matches the bytes reviewed for this check."""


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _load_packets(root):
    manifest = json.loads((PACKAGE / "reviewed_inputs.json").read_text())
    if set(manifest) != {"version", "inputs"} or manifest["version"] != VERSION:
        raise ValueError("Unrecognized crosscheck input manifest")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(CASES):
        raise ValueError("Crosscheck manifest case coverage differs")
    loaded = {}
    for case_id, (relative, _) in CASES.items():
        member = inputs[case_id]
        if not isinstance(member, dict) or set(member) != {"path", "sha256"} or member["path"] != relative:
            raise ValueError(f"Invalid public packet binding: {case_id}")
        content = (root / relative).read_bytes()
        if _hash(content) != member["sha256"]:
            raise InputDriftError(f"Public packet differs from reviewed hash: {relative}")
        packet = json.loads(content)
        loaded[case_id] = (packet["question"] if case_id.startswith("c") else packet, member)
    return loaded


def _declared(case_id, question):
    if case_id.startswith("q"):
        return {field["name"]: field for field in question["outputs"]}
    # The current public core response contract states these numeric tolerances.
    return {name: dict(field, atol=1e-6, rtol=1e-8) for name, field in question["outputs"].items()}


def _primary(case_id, question):
    """Import only the arithmetic audit module, after every alternate derivation."""
    module = importlib.import_module(f"quality.{CASES[case_id][1]}")
    calculate = getattr(module, case_id)
    return calculate(question if case_id.startswith("c") else question["inputs"])[0]


def _equal(actual, expected, atol, rtol):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        try:
            return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= atol + rtol * abs(expected)
        except (OverflowError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, b, atol, rtol) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def run(root=ROOT, compare_primary=True):
    """Return deterministic evaluator-side evidence; never read stored answers.

    Complete packet hashes are checked before calculations. All alternate
    calculations finish before any primary arithmetic audit module is imported.
    ``compare_primary=False`` performs no primary implementation imports.
    """
    packets = _load_packets(Path(root))
    records = []
    for case_id in CASES:
        question, binding = packets[case_id]
        answers, certificate = CHECKS[case_id](question)
        fields = _declared(case_id, question)
        if set(answers) != set(fields):
            raise ValueError(f"Alternate output coverage differs: {case_id}")
        records.append({
            "case_id": case_id, "input": binding.copy(), "method": METHODS[case_id],
            "alternate_answers": answers, "certificate": certificate,
        })

    # This phase boundary is deliberate: no primary result can be used as an
    # input to the alternate derivations above.
    all_match = True
    if compare_primary:
        for record in records:
            case_id = record["case_id"]
            question = packets[case_id][0]
            primary = _primary(case_id, question)
            fields = _declared(case_id, question)
            coverage_matches = set(primary) == set(fields)
            comparisons = []
            for name, schema in fields.items():
                atol, rtol = schema.get("atol", 0), schema.get("rtol", 0)
                matches = name in primary and _equal(record["alternate_answers"][name], primary[name], atol, rtol)
                comparisons.append({"field": name, "atol": atol, "rtol": rtol, "matches": matches})
            record["primary_comparison"] = {
                "implementation": f"quality/{CASES[case_id][1]}.py::{case_id}",
                "implementation_sha256": _hash((ROOT / "quality" / f"{CASES[case_id][1]}.py").read_bytes()),
                "output_coverage_matches": coverage_matches,
                "fields": comparisons,
            }
            all_match &= coverage_matches and all(field["matches"] for field in comparisons)
    return {
        "version": VERSION,
        "kind": "selected_public_contract_crosschecks",
        "status": "agrees_with_primary_audit" if compare_primary and all_match else "primary_audit_disagreement" if compare_primary else "derived_only",
        "scope": "Selected assistant-authored second implementations under hash-bound synthetic contracts; not independent practitioner review or market validation",
        "case_count": len(records),
        "output_count": sum(len(record["alternate_answers"]) for record in records),
        "implementation_sha256": {f"quality/crosschecks/{name}": _hash((PACKAGE / name).read_bytes())
                                  for name in ("__init__.py", "methods.py", "runner.py", "__main__.py", "reviewed_inputs.json")},
        "records": records,
        "limits": [
            "These five cases are a subset of existing audit coverage, not additional benchmark cases.",
            "The original audit and these checks are assistant-authored; numerical agreement is not an independent human review.",
            "Exact counting and search certificates cover only the stated finite domains; decimal agreement is numerical evidence.",
            "No claim of model realism, empirical signal efficacy, hidden holdout integrity, or general-purpose input support is made.",
            "Answers and certificates are evaluator-side public benchmark material; do not include them in a solver packet.",
        ],
    }
