"""Source quote verification with one additional sufficient annual-summary route."""
from .common import FROZEN
from financial_evals.core import load_case as frozen_load_case, normalized, within_tolerance, _support_order
from pathlib import Path
from typing import Any
from copy import deepcopy


def load_case(root, case_id):
    case,gold,sources=frozen_load_case(root,case_id)
    gold=deepcopy(gold)
    if case_id=='h2-001':
        field=next(f for f in gold['fields'] if f['id']=='prior_year_non_gaap_income_recast_reduction')
        requirements=field['evidence_requirements']
        # Each alternative must include the annual heading, non-GAAP section,
        # row label and value; the independent policy/recast requirement persists.
        for sid,anchors in [
            ('h2a-paypal-2023',["FY'23 Financial Results",'FY 2023','Non-GAAP','Net income','$5,646']),
            ('h2a-paypal-2024',["FY'24 Financial Results","FY'23",'Non-GAAP','Net income','$4,240']),
        ]:
            group=next(group for group in requirements if any(
                c['source_id']==sid and 'Non-GAAP net income' in c['anchors'] for c in group))
            group.append({'source_id':sid,'anchors':anchors,'same_quote':True})
    return case,gold,sources


def grade(root: Path, case_id: str, response: Any) -> dict:
    case, gold, sources = load_case(root, case_id)
    if not isinstance(response, dict) or response.get("case_id") != case_id:
        return {"case_id": case_id, "automated_pass": False,
                "critical_errors": ["invalid_response_case_id"], "fields": []}
    answers = response.get("answers", {})
    if not isinstance(answers, dict):
        answers = {}
    refs = {f["id"]: f for f in gold["fields"]}
    results_by_id = {}
    verified_proof = {}
    for field in _support_order(case["output_fields"]):
        field_id = field["id"]
        ref = refs[field_id]
        answer = answers.get(field_id)
        failures = []
        correct = False
        if not isinstance(answer, dict) or "value" not in answer:
            failures.append("missing_answer")
            answer = {}
        else:
            value = answer["value"]
            expected = ref["expected"]
            if field["type"] == "number":
                correct = within_tolerance(value, expected, ref["atol"], ref["rtol"])
            elif field["type"] == "boolean":
                correct = isinstance(value, bool) and value is expected
            else:
                correct = isinstance(value, str) and value == expected
            if not correct:
                failures.append("incorrect_value_or_type")
        evidence = answer.get("evidence", [])
        supported = set()
        quotes_by_source = {}
        allowed_sources = set(field.get("allowed_source_ids", case["source_ids"]))
        evidence_valid = True
        if not isinstance(evidence, list):
            evidence_valid = False
            evidence = []
        for cite in evidence:
            if not isinstance(cite, dict):
                evidence_valid = False
                continue
            sid, quote = cite.get("source_id"), cite.get("quote")
            if (not isinstance(sid, str) or sid not in sources
                    or sid not in allowed_sources
                    or not isinstance(quote, str) or not quote.strip()
                    or normalized(quote) not in normalized(sources[sid]["text"])):
                evidence_valid = False
                continue
            quotes_by_source.setdefault(sid, set()).add(normalized(quote))
        support_refs = answer.get("support_refs", [])
        if (not isinstance(support_refs, list)
                or any(not isinstance(dependency, str) for dependency in support_refs)
                or len(set(support_refs)) != len(support_refs)
                or not set(support_refs).issubset(field.get("support_dependencies", []))):
            evidence_valid = False
            support_refs = []
        for dependency in support_refs:
            # Only a correct typed value with valid evidence can convey proof.
            # Actual citations, not the upstream gold anchors, are inherited.
            proof = verified_proof.get(dependency)
            if proof is None or not set(proof).issubset(allowed_sources):
                evidence_valid = False
                continue
            for sid, quotes in proof.items():
                quotes_by_source.setdefault(sid, set()).update(quotes)
        for sid, quotes in quotes_by_source.items():
            groups = ref.get("evidence_anchor_groups", {}).get(sid)
            if groups is None:
                groups = [ref.get("evidence_anchors", {}).get(sid, [])]
            # Each component needs one alternative, possibly in a separate quote.
            # Groups override the legacy OR list for this source when present.
            if all(any(normalized(anchor) in quote
                       for anchor in alternatives for quote in quotes)
                   for alternatives in groups):
                supported.add(sid)
        if field["evidence_required"]:
            evidence_valid = evidence_valid and set(ref.get("required_source_ids", [])).issubset(supported)
            # Requirements are AND; candidates are OR; a candidate needs ALL
            # its anchors, possibly across separate quotes from that source.
            evidence_valid = evidence_valid and all(
                any((any(all(normalized(anchor) in quote for anchor in candidate["anchors"])
                         for quote in quotes_by_source.get(candidate["source_id"], []))
                     if candidate.get('same_quote') else
                     all(any(normalized(anchor) in quote
                             for quote in quotes_by_source.get(candidate["source_id"], []))
                         for anchor in candidate["anchors"]))
                    for candidate in alternatives)
                for alternatives in ref.get("evidence_requirements", []))
        if not evidence_valid:
            failures.append("missing_invalid_or_unsupported_evidence")
        results_by_id[field_id] = {"id": field_id, "value_correct": correct,
                                   "evidence_valid": evidence_valid, "passed": not failures,
                                   "critical": ref["critical"], "failures": failures}
        if correct and evidence_valid:
            verified_proof[field_id] = quotes_by_source
    results = [results_by_id[field["id"]] for field in case["output_fields"]]

    def dimension_accuracy(dimension):
        selected = [results_by_id[field["id"]] for field in case["output_fields"]
                    if field.get("score_dimension", "financial") == dimension]
        return sum(result["value_correct"] for result in selected) / len(selected) if selected else None

    return {
        "case_id": case_id, "case_group": case["case_group"],
        "automated_pass": all(f["passed"] for f in results),
        "critical_errors": [f["id"] for f in results if f["critical"] and not f["passed"]],
        "value_accuracy": sum(f["value_correct"] for f in results) / len(results),
        "evidence_accuracy": sum(f["evidence_valid"] for f in results) / len(results),
        "financial_value_accuracy": dimension_accuracy("financial"),
        "uncertainty_score": dimension_accuracy("uncertainty"),
        "fields": results,
        "scope": "Declared structured fields and source anchors only; no prose or investment-skill certification.",
    }
