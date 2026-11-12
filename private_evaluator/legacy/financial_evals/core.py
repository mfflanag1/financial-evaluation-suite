"""Integrity checks, actor-pack export and bounded structured-answer grading."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An evaluation artifact violates its declared construction contract."""


def read_json(path: Path) -> Any:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ContractError(f"Non-finite JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant)


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ContractError(f"Invalid artifact identifier: {value!r}")
    return value


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ContractError(f"Expected a project-relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError(f"Path escapes project: {relative}")
    return path


def timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError(f"Invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"Timezone required: {value}")
    return parsed


def number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def normalized(text: str) -> str:
    return " ".join(text.split())


def within_tolerance(value: Any, expected: Any, atol: Any, rtol: Any) -> bool:
    if not number(value):
        return False
    decimals = [Decimal(str(x)) for x in (value, expected, atol, rtol)]
    with localcontext() as context:
        # Preserve decimal boundaries through subtraction, multiplication and addition.
        context.prec = max(32, sum(len(x.as_tuple().digits) + abs(x.as_tuple().exponent)
                                   for x in decimals) + 4)
        actual, reference, absolute, relative = decimals
        return abs(actual - reference) <= absolute + relative * abs(reference)


def _support_order(fields: list[dict]) -> list[dict]:
    """Validate the author allowlists and return dependency-first field order."""
    by_id = {field["id"]: field for field in fields}
    pending = {}
    dependents = {field_id: [] for field_id in by_id}
    for field in fields:
        field_id = field["id"]
        dependencies = field.get("support_dependencies", [])
        if (not isinstance(dependencies, list)
                or any(not isinstance(dependency, str) for dependency in dependencies)
                or len(set(dependencies)) != len(dependencies)
                or not set(dependencies).issubset(by_id)
                or field_id in dependencies):
            raise ContractError(f"Invalid support dependencies: {field_id}")
        pending[field_id] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(field_id)
    ready = [field_id for field_id in by_id if pending[field_id] == 0]
    # Iterating an expanding list avoids recursion limits on long valid chains.
    for field_id in ready:
        for dependent in dependents[field_id]:
            pending[dependent] -= 1
            if pending[dependent] == 0:
                ready.append(dependent)
    if len(ready) != len(fields):
        raise ContractError("Cyclic support dependencies")
    return [by_id[field_id] for field_id in ready]


def load_case(root: Path, case_id: str) -> tuple[dict, dict, dict]:
    identifier(case_id)
    case = read_json(inside(root, f"cases/{case_id}.json"))
    gold = read_json(inside(root, f"gold/{case_id}.json"))
    if not isinstance(case, dict) or not isinstance(gold, dict):
        raise ContractError("Case/gold artifacts must be objects")
    if case.get("id") != case_id or gold.get("case_id") != case_id:
        raise ContractError("Case/gold ID mismatch")
    for key in ("case_group", "lane", "status", "epistemic_type", "prompt"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ContractError(f"Missing case property: {key}")
    cutoff = timestamp(case.get("information_cutoff"))
    ids = case.get("source_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(sid, str) for sid in ids)
            or len(set(ids)) != len(ids)):
        raise ContractError("Case needs a nonempty unique source_ids list")
    sources = {}
    for source_id in ids:
        identifier(source_id)
        manifest = read_json(inside(root, f"sources/{source_id}/manifest.json"))
        if not isinstance(manifest, dict):
            raise ContractError(f"Source manifest must be an object: {source_id}")
        if manifest.get("id") != source_id:
            raise ContractError(f"Source ID mismatch: {source_id}")
        for key in ("url", "title", "publisher", "availability_basis", "retrieved_at",
                    "extraction_method", "redistribution_status"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                raise ContractError(f"Missing source metadata {source_id}: {key}")
        precision = manifest.get("publication_precision")
        published = None
        if precision in ("day", "second"):
            published = timestamp(manifest.get("published_at"))
            if published > cutoff:
                raise ContractError(f"Source is later than case cutoff: {source_id}")
            if precision == "day" and (published.hour, published.minute,
                                        published.second, published.microsecond) != (23, 59, 59, 0):
                raise ContractError(f"Day-precision publication must use 23:59:59 in its stated timezone: {source_id}")
        elif (precision != "unknown" or "published_at" not in manifest
              or manifest["published_at"] is not None or "available_by" not in manifest):
            raise ContractError(f"Unresolved source timing: {source_id}")
        if "available_by" in manifest:
            available_by = timestamp(manifest["available_by"])
            if published is not None and available_by < published:
                raise ContractError(f"Availability bound precedes publication: {source_id}")
            if available_by > cutoff:
                raise ContractError(f"Source availability bound is later than case cutoff: {source_id}")
        timestamp(manifest["retrieved_at"])
        for kind in ("raw", "text"):
            path = inside(root, manifest.get(f"{kind}_path"))
            if not path.is_relative_to((root / "sources" / source_id).resolve()):
                raise ContractError(f"Source path outside its source directory: {source_id}")
            content = path.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            if actual != manifest.get(f"{kind}_sha256"):
                raise ContractError(f"Source hash mismatch: {source_id}/{kind}")
            if kind == "text":
                text_bytes = content
        try:
            source_text = text_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError(f"Source text must be UTF-8: {source_id}") from exc
        sources[source_id] = {
            "manifest": manifest,
            "text": source_text,
            "text_bytes": text_bytes,
        }
    fields = case.get("output_fields")
    refs = gold.get("fields")
    if not isinstance(fields, list) or not fields or not isinstance(refs, list) or not refs:
        raise ContractError("Case/gold fields must be nonempty lists")
    for field in fields + refs:
        if not isinstance(field, dict):
            raise ContractError("Case/gold fields must be objects")
        identifier(field.get("id"))
    field_ids = [f["id"] for f in fields]
    ref_ids = [f["id"] for f in refs]
    if len(set(field_ids)) != len(fields) or len(set(ref_ids)) != len(refs):
        raise ContractError("Duplicate field IDs")
    if set(field_ids) != set(ref_ids):
        raise ContractError("Case/gold field sets differ")
    _support_order(fields)
    ref_by_id = {f["id"]: f for f in refs}
    for field in fields:
        field_id = identifier(field["id"])
        ref = ref_by_id[field_id]
        if "expected" not in ref:
            raise ContractError(f"Missing reference value: {field_id}")
        expected = ref["expected"]
        for key in ("unit", "description"):
            if not isinstance(field.get(key), str) or not field[key].strip():
                raise ContractError(f"Missing field property: {field_id}/{key}")
        if not isinstance(field.get("evidence_required"), bool):
            raise ContractError(f"Missing evidence_required flag: {field_id}")
        if field.get("score_dimension", "financial") not in ("financial", "uncertainty"):
            raise ContractError(f"Invalid score dimension: {field_id}")
        allowed = field.get("allowed_source_ids", ids)
        if (not isinstance(allowed, list) or not allowed
                or any(not isinstance(sid, str) for sid in allowed)
                or len(set(allowed)) != len(allowed) or not set(allowed).issubset(sources)):
            raise ContractError(f"Invalid field source policy: {field_id}")
        kind = field.get("type")
        if kind == "number":
            if not number(expected):
                raise ContractError(f"Invalid reference number: {field_id}")
            for tol in ("atol", "rtol"):
                if not number(ref.get(tol)) or ref[tol] < 0:
                    raise ContractError(f"Invalid tolerance: {field_id}/{tol}")
        elif kind == "string":
            if not isinstance(expected, str):
                raise ContractError(f"Invalid reference string: {field_id}")
        elif kind == "boolean":
            if not isinstance(expected, bool):
                raise ContractError(f"Invalid reference boolean: {field_id}")
        else:
            raise ContractError(f"Unsupported output type: {kind}")
        if "enum" in field:
            choices = field["enum"]
            valid_type = {"number": number, "string": lambda x: isinstance(x, str),
                          "boolean": lambda x: isinstance(x, bool)}[kind]
            if (not isinstance(choices, list) or not choices
                    or not all(valid_type(choice) for choice in choices) or expected not in choices):
                raise ContractError(f"Invalid declared enum: {field_id}")
        if not isinstance(ref.get("critical"), bool):
            raise ContractError(f"Missing critical flag: {field_id}")
        required = ref.get("required_source_ids", [])
        anchors = ref.get("evidence_anchors", {})
        anchor_groups = ref.get("evidence_anchor_groups", {})
        requirements = ref.get("evidence_requirements", [])
        if not isinstance(requirements, list):
            raise ContractError(f"Invalid evidence requirements: {field_id}")
        if requirements and not field["evidence_required"]:
            raise ContractError(f"Evidence requirements declared on optional-evidence field: {field_id}")
        if (not isinstance(required, list) or any(not isinstance(sid, str) for sid in required)
                or not set(required).issubset(allowed)):
            raise ContractError(f"Invalid required sources: {field_id}")
        if field["evidence_required"] and not required and not requirements:
            raise ContractError(f"Required evidence has no source: {field_id}")
        if not isinstance(anchors, dict) or not set(anchors).issubset(allowed):
            raise ContractError(f"Invalid anchor sources: {field_id}")
        if not isinstance(anchor_groups, dict) or not set(anchor_groups).issubset(allowed):
            raise ContractError(f"Invalid anchor group sources: {field_id}")
        for source_id in required:
            if not anchors.get(source_id) and not anchor_groups.get(source_id):
                raise ContractError(f"Missing reference anchors: {field_id}/{source_id}")

        def validate_snippets(source_id, snippets):
            if not isinstance(snippets, list) or not snippets:
                raise ContractError(f"Invalid anchor list: {field_id}/{source_id}")
            for snippet in snippets:
                if (not isinstance(snippet, str) or not snippet.strip()
                        or normalized(snippet) not in normalized(sources[source_id]["text"])):
                    raise ContractError(f"Reference anchor absent from source: {field_id}/{source_id}")
        for source_id, snippets in anchors.items():
            validate_snippets(source_id, snippets)
        for source_id, groups in anchor_groups.items():
            if not isinstance(groups, list) or not groups:
                raise ContractError(f"Invalid anchor groups: {field_id}/{source_id}")
            for alternatives in groups:
                validate_snippets(source_id, alternatives)
        for alternatives in requirements:
            if not isinstance(alternatives, list) or not alternatives:
                raise ContractError(f"Empty evidence requirement: {field_id}")
            for candidate in alternatives:
                if (not isinstance(candidate, dict)
                        or set(candidate) != {"source_id", "anchors"}
                        or not isinstance(candidate["source_id"], str)
                        or candidate["source_id"] not in allowed):
                    raise ContractError(f"Invalid evidence candidate: {field_id}")
                validate_snippets(candidate["source_id"], candidate["anchors"])
    return case, gold, sources


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
                any(all(any(normalized(anchor) in quote
                            for quote in quotes_by_source.get(candidate["source_id"], []))
                        for anchor in candidate["anchors"])
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


def export_actor(root: Path, case_id: str, destination: Path) -> dict:
    case, _, sources = load_case(root, case_id)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ContractError("Export actor pack outside the evaluator project")
    if destination.exists():
        raise ContractError("Actor destination already exists; use a fresh isolated directory")
    destination.mkdir(parents=True)
    actor = {key: case[key] for key in (
        "id", "case_group", "lane", "epistemic_type", "information_cutoff",
        "prompt", "source_ids")}
    actor["output_fields"] = [{key: field[key] for key in (
        "id", "type", "unit", "description", "evidence_required", "enum", "allowed_source_ids",
        "support_dependencies", "score_dimension") if key in field}
        for field in case["output_fields"]]
    actor["response_format"] = {"case_id": case_id, "answers": {
        "<field_id>": {"value": "<typed answer>", "evidence": [
            {"source_id": "<permitted source ID>", "quote": "<verbatim supporting passage>"}]}},
        "limitations": []}
    if any(field.get("support_dependencies") for field in case["output_fields"]):
        actor["response_format"]["answers"]["<field_id>"]["support_refs"] = [
            "<optional field ID from this output's support_dependencies>"]
    (destination / "case.json").write_text(json.dumps(actor, indent=2) + "\n", encoding="utf-8")
    (destination / "sources").mkdir()
    public_manifests = []
    for sid, source in sources.items():
        target = destination / "sources" / f"{sid}.txt"
        target.write_bytes(source["text_bytes"])
        manifest = source["manifest"]
        public_manifest = {k: manifest[k] for k in (
            "id", "url", "title", "publisher", "published_at", "publication_precision",
            "availability_basis", "text_sha256")}
        for key in ("available_by", "capture_quality", "notes", "raw_capture_type"):
            if key in manifest:
                public_manifest[key] = manifest[key]
        public_manifests.append(public_manifest)
    (destination / "sources.json").write_text(json.dumps(public_manifests, indent=2) + "\n", encoding="utf-8")
    receipt = {"case_id": case_id, "files": {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*")) if path.is_file()}}
    return receipt
