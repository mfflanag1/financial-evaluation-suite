"""Structural checkpoints from public inputs, without executing candidate code.

Checkpoint acceptance is not a financial correctness check or a reward. The
frozen final scorer and isolated reproduction check retain those responsibilities.
"""

import ast
import csv
import hashlib
import io
import json
import math


CSV_FILES = ("prices.csv", "features.csv", "securities.csv")
WINDOW_FIELDS = ("training_start", "training_end", "validation_start", "active_end",
                 "validation_end", "winddown_sessions")
TRIAL_FIELDS = ("mean_net", "sharpe", "mean_turnover", "screening_score")
STAGES = ("ingestion", "trials", "submission")
ARTIFACT_LIMITS = {
    "/workspace/research.py": 1024 * 1024,
    "/workspace/ingestion.json": 256 * 1024,
    "/workspace/trials.json": 1024 * 1024,
    "/workspace/policy.json": 2 * 1024 * 1024,
}
MAX_PUBLIC_JSON_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 32


class _Invalid(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Invalid(reason)


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def _json(payload, label, maximum_bytes):
    _require(type(payload) is bytes, f"{label}_invalid_bytes")
    _require(0 < len(payload) <= maximum_bytes, f"{label}_size_limit")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise _Invalid(f"{label}_invalid_utf8") from None
    # Bound nesting before the decoder allocates a potentially recursive object.
    depth, quoted, escaped = 0, False, False
    for char in source:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            _require(depth <= MAX_JSON_DEPTH, f"{label}_depth_limit")
        elif char in "]}":
            depth -= 1
    try:
        result = json.loads(source, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError):
        raise _Invalid(f"{label}_invalid_json") from None
    pending = [result]
    while pending:
        value = pending.pop()
        if type(value) in (int, float):
            _require(_number(value), f"{label}_nonfinite_number")
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return result


def _artifact(artifacts, filename):
    path = f"/workspace/{filename}"
    _require(path in artifacts, f"missing_{filename.split('.')[0]}_artifact")
    return artifacts[path]


def _source(artifacts):
    payload = _artifact(artifacts, "research.py")
    _require(type(payload) is bytes, "research_invalid_bytes")
    _require(0 < len(payload) <= ARTIFACT_LIMITS["/workspace/research.py"], "research_size_limit")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise _Invalid("research_invalid_utf8") from None
    _require(bool(source.strip()), "research_empty")
    try:
        tree = ast.parse(source, filename="research.py")
    except (SyntaxError, ValueError, RecursionError):
        raise _Invalid("research_invalid_syntax") from None
    _require(bool(tree.body), "research_empty")


def _public(public_files):
    _require(isinstance(public_files, dict)
             and set(public_files) == {*CSV_FILES, "contract.json"}
             and all(type(value) is str for value in public_files.values()), "invalid_public_data")
    contract = _json(public_files["contract.json"].encode("utf-8"), "public_contract", MAX_PUBLIC_JSON_BYTES)
    _require(isinstance(contract, dict), "invalid_public_data")
    candidates, requests = contract.get("candidates"), contract.get("audit_requests")
    _require(isinstance(candidates, list) and bool(candidates)
             and isinstance(requests, list) and bool(requests), "invalid_public_data")
    candidate_ids = []
    for row in candidates:
        _require(isinstance(row, dict) and type(row.get("id")) is str and bool(row["id"]),
                 "invalid_public_data")
        candidate_ids.append(row["id"])
    _require(len(set(candidate_ids)) == len(candidate_ids), "invalid_public_data")
    audit_keys = []
    for row in requests:
        _require(isinstance(row, dict) and set(row) == {"entity_id", "day", "field"}
                 and type(row["entity_id"]) is str and type(row["day"]) is int
                 and type(row["field"]) is str, "invalid_public_data")
        audit_keys.append((row["entity_id"], row["day"], row["field"]))
    _require(len(set(audit_keys)) == len(audit_keys), "invalid_public_data")
    windows = {key: contract.get(key) for key in WINDOW_FIELDS}
    _require(all(type(value) is int for value in windows.values()), "invalid_public_data")
    files, entities = {}, set()
    for filename in CSV_FILES:
        contents = public_files[filename]
        reader = csv.DictReader(io.StringIO(contents, newline=""), strict=True)
        count = 0
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()),
                     "invalid_public_data")
            count += 1
            if filename == "securities.csv":
                entity = row.get("entity_id")
                _require(type(entity) is str and bool(entity), "invalid_public_data")
                entities.add(entity)
        files[filename] = {"sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
                           "rows": count}
    _require(bool(entities), "invalid_public_data")
    return {"files": files,
            "contract_sha256": hashlib.sha256(public_files["contract.json"].encode("utf-8")).hexdigest(),
            "entity_ids": entities, "candidate_ids": set(candidate_ids), "windows": windows,
            "audit_keys": set(audit_keys)}


def stage_contract(public_files):
    """Return schemas and public counts, never expected inventory or finance values.

    Inventory digests, identifiers, row counts and window values must be read or
    calculated by the solver from its supplied files. Lists are order-independent.
    """
    public = _public(public_files)
    return {
        "version": "portfolio-stages-v1",
        "required_artifacts": {
            "ingestion": ["research.py", "ingestion.json"],
            "trials": ["research.py", "ingestion.json", "trials.json"],
            "submission": ["research.py", "ingestion.json", "trials.json", "policy.json"],
        },
        "source": {"encoding": "UTF-8", "maximum_bytes": ARTIFACT_LIMITS["/workspace/research.py"],
                   "requirement": "Nonempty syntactically valid Python; validated without execution."},
        "json": {"maximum_bytes_per_file": {path: limit for path, limit in ARTIFACT_LIMITS.items()
                                            if path.endswith(".json")},
                 "maximum_nesting": MAX_JSON_DEPTH,
                 "requirements": "Exact object keys; no duplicate keys, nonfinite numbers or Boolean numbers."},
        "ingestion_schema": {
            "files": {name: {"sha256": "SHA-256 hex digest of original UTF-8 file bytes",
                              "rows": "integer data-row count from csv.DictReader, excluding header"}
                      for name in CSV_FILES},
            "contract_sha256": "SHA-256 hex digest of original contract.json UTF-8 bytes",
            "entity_ids": "array of every distinct securities.csv entity_id exactly once",
            "candidate_ids": "array of every contract.json candidates[].id exactly once",
            "windows": {name: "integer copied from contract.json" for name in WINDOW_FIELDS},
        },
        "trials_schema": {
            "type": "array; exactly one row for each public candidate, any order",
            "row": {"id": "public candidate id", **{key: "finite non-Boolean number" for key in TRIAL_FIELDS}},
        },
        "submission_schema": {
            "audit": "array of {entity_id: string, day: integer, field: string, value: finite non-Boolean number or null}; every public audit request exactly once",
            "trials": "same rows and numerical values as current trials.json; any order",
            "decision": "deploy or abstain",
            "selected": "public candidate id when deploying; null when abstaining",
        },
        "public_counts": {"entities": len(public["entity_ids"]), "candidates": len(public["candidate_ids"]),
                          "audit_requests": len(public["audit_keys"])},
        "acceptance": "Structural checkpoint only; no financial correctness or reward is assigned. Later stages revalidate all earlier artifacts.",
    }


def inventory_for_public_files(public_files):
    """Build trusted model-free fixtures; never include this result in prompts.

    This inventory contains only byte digests, CSV row counts, identifiers and
    declared window boundaries. It performs no financial calculations.
    """
    public = _public(public_files)
    return {key: sorted(value) if isinstance(value, set) else value
            for key, value in public.items() if key != "audit_keys"}


def _identities(actual, expected):
    return (isinstance(actual, list) and all(type(value) is str for value in actual)
            and len(actual) == len(expected) and set(actual) == expected)


def _ingestion(artifacts, public):
    data = _json(_artifact(artifacts, "ingestion.json"), "ingestion", ARTIFACT_LIMITS["/workspace/ingestion.json"])
    _require(isinstance(data, dict)
             and set(data) == {"files", "contract_sha256", "entity_ids", "candidate_ids", "windows"},
             "ingestion_invalid_schema")
    files = data["files"]
    _require(isinstance(files, dict) and set(files) == set(CSV_FILES), "ingestion_invalid_schema")
    for filename in CSV_FILES:
        item = files[filename]
        _require(isinstance(item, dict) and set(item) == {"sha256", "rows"}
                 and type(item["sha256"]) is str and type(item["rows"]) is int and item["rows"] >= 0,
                 "ingestion_invalid_schema")
        _require(item == public["files"][filename], "ingestion_inventory_mismatch")
    _require(type(data["contract_sha256"]) is str
             and data["contract_sha256"] == public["contract_sha256"], "ingestion_inventory_mismatch")
    _require(_identities(data["entity_ids"], public["entity_ids"])
             and _identities(data["candidate_ids"], public["candidate_ids"]), "ingestion_identity_mismatch")
    windows = data["windows"]
    _require(isinstance(windows, dict) and set(windows) == set(WINDOW_FIELDS)
             and all(type(value) is int for value in windows.values()), "ingestion_invalid_schema")
    _require(windows == public["windows"], "ingestion_window_mismatch")


def _trials(data, public, label):
    _require(isinstance(data, list) and len(data) == len(public["candidate_ids"]), f"{label}_candidate_coverage")
    indexed = {}
    for row in data:
        _require(isinstance(row, dict) and set(row) == {"id", *TRIAL_FIELDS}
                 and type(row["id"]) is str and all(_number(row[key]) for key in TRIAL_FIELDS),
                 f"{label}_invalid_schema")
        _require(row["id"] in public["candidate_ids"] and row["id"] not in indexed,
                 f"{label}_candidate_coverage")
        indexed[row["id"]] = row
    return indexed


def _submission(artifacts, public, trials):
    policy = _json(_artifact(artifacts, "policy.json"), "policy", ARTIFACT_LIMITS["/workspace/policy.json"])
    _require(isinstance(policy, dict) and set(policy) == {"audit", "trials", "decision", "selected"},
             "policy_invalid_schema")
    policy_trials = _trials(policy["trials"], public, "policy_trials")
    _require(policy_trials == trials, "policy_trials_mismatch")
    audit = policy["audit"]
    _require(isinstance(audit, list) and len(audit) == len(public["audit_keys"]), "policy_audit_coverage")
    seen = set()
    for row in audit:
        _require(isinstance(row, dict) and set(row) == {"entity_id", "day", "field", "value"}
                 and type(row["entity_id"]) is str and type(row["day"]) is int
                 and type(row["field"]) is str and (row["value"] is None or _number(row["value"])),
                 "policy_audit_invalid_schema")
        key = (row["entity_id"], row["day"], row["field"])
        _require(key in public["audit_keys"] and key not in seen, "policy_audit_coverage")
        seen.add(key)
    _require(type(policy["decision"]) is str and policy["decision"] in ("deploy", "abstain"),
             "policy_invalid_decision")
    selected = policy["selected"]
    _require((policy["decision"] == "abstain" and selected is None)
             or (policy["decision"] == "deploy" and type(selected) is str and selected in public["candidate_ids"]),
             "policy_invalid_selection")


def validate_stage(stage: str, public_files: dict[str, str], artifacts: dict[str, bytes]) -> dict:
    """Check current artifacts using public data only; never execute source.

    All failures return a bounded stable reason without echoing candidate bytes.
    The caller is responsible for requiring sequential accepted checkpoints and
    extracting regular files safely before passing their raw bytes here.
    """
    try:
        _require(type(stage) is str and stage in STAGES, "unknown_stage")
        _require(isinstance(artifacts, dict), "invalid_artifacts")
        _source(artifacts)
        public = _public(public_files)
        _ingestion(artifacts, public)
        if stage in ("trials", "submission"):
            trials = _trials(_json(_artifact(artifacts, "trials.json"), "trials",
                                  ARTIFACT_LIMITS["/workspace/trials.json"]), public, "trials")
        if stage == "submission":
            _submission(artifacts, public, trials)
    except _Invalid as error:
        return {"valid": False, "reason": str(error)}
    except (csv.Error, UnicodeError, RecursionError):
        return {"valid": False, "reason": "invalid_public_data"}
    return {"valid": True, "reason": "stage_valid", "summary": {
        "stage": stage, "entities": len(public["entity_ids"]), "candidates": len(public["candidate_ids"]),
        "audit_requests": len(public["audit_keys"]), "financial_correctness_checked": False,
    }}
