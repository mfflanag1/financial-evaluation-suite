"""Build deterministic case-level audit records and independent numeric evidence.

Run ``python3 -B -m quality.inventory --check`` to detect audit drift. No network,
model, submitted code, or reference implementation is executed by this module.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from . import AUDIT_VERSION
from .numerical import CHECKS
from .core_numerical import CHECKS as CORE_CHECKS
from .source_numerical import CHECKS as SOURCE_CHECKS
from .allocation_numerical import CHECKS as ALLOCATION_CHECKS
from .risk_numerical import CHECKS as RISK_CHECKS
from .remaining_numerical import CHECKS as REMAINING_CHECKS

ROOT = Path(__file__).resolve().parents[1]
QUANT_MODULES = {"numerical":CHECKS,"allocation_numerical":ALLOCATION_CHECKS,
                 "risk_numerical":RISK_CHECKS,"remaining_numerical":REMAINING_CHECKS}


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member(path, pointer=None):
    entry = {"path":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)}
    if pointer is not None:
        entry["json_pointer"] = pointer
    return entry


def numeric_equal(actual, expected, atol=0.0, rtol=0.0):
    if isinstance(actual,bool) or isinstance(expected,bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual,(int,float)) and isinstance(expected,(int,float)):
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=atol+rtol*abs(expected)
    if isinstance(actual,list) and isinstance(expected,list):
        return len(actual)==len(expected) and all(numeric_equal(a,b,atol,rtol) for a,b in zip(actual,expected))
    return type(actual) is type(expected) and actual==expected


def independent_results():
    # Calculations consume only public input packets. Reference access happens
    # after each result exists, solely to compare final fields and tolerances.
    derived = {}
    quantitative = {cid:check for checks in QUANT_MODULES.values() for cid,check in checks.items()}
    if len(quantitative)!=sum(map(len,QUANT_MODULES.values())):
        raise ValueError("Duplicate quantitative derivation")
    implementations = {cid:module for module,checks in QUANT_MODULES.items() for cid in checks}
    for cid,(calculate,method) in dict(quantitative, **CORE_CHECKS).items():
        path = ROOT/f"questions/legacy/quant100/questions/{cid}.json" if cid.startswith("q") else ROOT/f"questions/challenges/{cid}.json"
        packet = read(path)
        question = packet if cid.startswith("q") else packet["question"]
        answers, certificates = calculate(question["inputs"] if cid.startswith("q") else question)
        derived[cid] = (path,question,answers,certificates,method)
    source_inputs = {}
    for cid,calculate in SOURCE_CHECKS.items():
        path = ROOT/f"questions/legacy/source_grounded/cases/{cid}/case.json"
        question = read(path)
        paths = {sid:path.parent/"sources"/f"{sid}.txt" for sid in question["source_ids"]}
        answers,certificates = calculate(question,{sid:source.read_text() for sid,source in paths.items()})
        derived[cid] = (path,question,answers,certificates,certificates["method"])
        source_inputs[cid] = {sid:member(source) for sid,source in paths.items()}
    references = {case["id"]:case for case in read(ROOT/"private_evaluator/legacy/quant100/evaluator.json")}
    references.update({case["id"]:case["stages"][0] for case in read(ROOT/"private_evaluator/challenges/cases.json")})
    references.update({cid:read(ROOT/f"private_evaluator/legacy/source_grounded/gold/{cid}.json") for cid in SOURCE_CHECKS})
    records=[]
    for cid,(path,question,answers,certificates,method) in derived.items():
        if cid in SOURCE_CHECKS:
            expected={field["id"]:field["expected"] for field in references[cid]["fields"]}
            tolerances={field["id"]:{key:field[key] for key in ("atol","rtol") if key in field} for field in references[cid]["fields"]}
            declared={field["id"]:dict(field,**tolerances[field["id"]]) for field in question["output_fields"]}
        else:
            expected=references[cid]["answer" if cid.startswith("q") else "answers"]
            declared={output["name"]:output for output in question["outputs"]} if cid.startswith("q") else {name:dict(spec,atol=1e-6,rtol=1e-8) for name,spec in question["outputs"].items()}
        if set(answers)!=set(declared) or set(expected)!=set(declared):
            raise ValueError(f"Independent check output coverage differs: {cid}")
        fields=[]
        for name,schema in declared.items():
            passed=numeric_equal(answers[name],expected[name],schema.get("atol",0),schema.get("rtol",0))
            fields.append({"field":name,"derived":answers[name],"stored_reference":expected[name],
                           "atol":schema.get("atol",0),"rtol":schema.get("rtol",0),"matches":passed})
        module=implementations.get(cid,"core_numerical")
        implementation=f"quality/source_numerical.py::{SOURCE_CHECKS[cid].__name__}" if cid in SOURCE_CHECKS else f"quality/{module}.py::{cid}"
        scope=certificates["scope"] if cid in SOURCE_CHECKS else "All declared numerical and decision outputs, under supplied synthetic conventions; no scorer, source, market-validity, or independent-practitioner certification."
        records.append({"case_id":cid,"status":"matches_stored_reference" if all(field["matches"] for field in fields) else "reference_discrepancy",
                        "method":method,"input":member(path),"implementation":implementation,"scope":scope,
                        **({"retained_source_inputs":source_inputs[cid]} if cid in SOURCE_CHECKS else {}),
                        "fields":fields,"certificates":certificates})
    return {"audit_version":AUDIT_VERSION,"kind":"independent_numerical_recomputation",
            "implementation_sha256":{name:sha256(ROOT/"quality"/name) for name in [*(module+".py" for module in QUANT_MODULES),"core_numerical.py","source_numerical.py"]},
            "case_count":len(records),"matching_cases":sum(row["status"]=="matches_stored_reference" for row in records),
            "checked_fields":sum(len(row["fields"]) for row in records),"checks":records}


def shape(value):
    if isinstance(value,bool): return "boolean"
    if isinstance(value,(int,float)): return "number"
    if isinstance(value,str): return "string"
    if isinstance(value,list): return "array"
    if isinstance(value,dict): return "object"
    if value is None: return "null"
    raise ValueError("Unsupported JSON reference value")


def stages(case,track,question):
    if track=="source_grounded":
        return [{"turn":1,"schema_origin":"public_output_fields","outputs":[{"name":f["id"],"type":f["type"],"unit":f.get("unit")} for f in question["output_fields"]]}]
    if track in ("quant100","workflows"):
        return [{"turn":1,"schema_origin":"public_outputs","outputs":[{k:f[k] for k in ("name","type","unit","atol","rtol") if k in f} for f in question["outputs"]]}]
    if track=="research58":
        return [{"turn":turn["turn"],"schema_origin":"evaluator_turn_outputs","outputs":[{k:f[k] for k in ("name","type","unit","atol","rtol") if k in f} for f in turn["outputs"]]} for turn in case["turns"]]
    return [{"turn":i,"schema_origin":"reference_template_shape; runtime output contract may impose stricter requirements",
             "outputs":[{"name":name,"type":shape(value),"description":stage.get("descriptions",{}).get(name)} for name,value in sorted(stage["answers"].items())]}
            for i,stage in enumerate(case["stages"],1)]


def build_inventory(evidence):
    catalog=read(ROOT/"questions/catalog.json")
    variants=read(ROOT/"questions/variants/catalog.json")["cases"]
    records={}; locations={}
    for track in ("quant100","workflows","research58"):
        path=ROOT/f"private_evaluator/legacy/{track}/evaluator.json"
        for index,case in enumerate(read(path)):
            records[case["id"]]=case; locations[case["id"]]=member(path,f"/{index}")
    for track in ("challenges","variants"):
        path=ROOT/f"private_evaluator/{track}/cases.json"
        for index,case in enumerate(read(path)):
            records[case["id"]]=case; locations[case["id"]]=member(path,f"/{index}")
    checks={r["case_id"]:r for r in evidence["checks"]}
    rows=[]
    for metadata in catalog+[dict(row,track="variants",path=f"variants/{row['id']}.json") for row in variants]:
        cid=metadata["id"]; track=metadata["track"]
        public_path=ROOT/"questions"/metadata["path"] if metadata.get("path") else None
        question=read(public_path) if public_path is not None else None
        if track=="source_grounded":
            path=ROOT/f"private_evaluator/legacy/source_grounded/gold/{cid}.json"
            case=read(path); reference=member(path)
        else:
            case=records[cid]; reference=locations[cid]
        schemas=stages(case,track,question)
        count=sum(len(stage["outputs"]) for stage in schemas)
        if count!=metadata["outputs"]:
            raise ValueError(f"Output inventory mismatch for {cid}: {count} != {metadata['outputs']}")
        for stage in schemas:
            names=[field["name"] for field in stage["outputs"]]
            if len(names)!=len(set(names)):
                raise ValueError(f"Duplicate output names: {cid}, turn {stage['turn']}")
        code_checks=checks.get(cid)
        rows.append({"id":cid,"bank":"related_variant" if track=="variants" else "main",
            "track":track,"area":metadata.get("area",case.get("family")),
            "title":metadata.get("title",case.get("title",question.get("case_group",cid) if question else cid)),
            "declared_split":metadata.get("split",case.get("split","not_specified")),
            "mode":metadata.get("mode",case.get("mode","single")),"turn_count":len(schemas),
            "declared_output_count":count,"related_to":metadata.get("variant_of"),
            "input_artifact":member(public_path) if public_path else None,"evaluator_artifact":reference,
            "specification":{"public_packet":"absent_from_question_directory" if question is None else "opening_turn_only" if len(schemas)>1 else "complete_single_turn",
                             "stage_schemas":schemas,"structure_checks":"catalog count, unique field names, required artifacts present"},
            "public_exposure":{"reference_answers":"published_in_repository","case_material":"published_in_repository",
                               "unseen_holdout_claim_supported":False},
            "verification":{"structural_inventory":"checked","independent_numerical_recomputation":code_checks["status"] if code_checks else "not_performed",
                            "independently_recomputed_fields":len(code_checks["fields"]) if code_checks else 0,
                            "evidence_record":f"quality/independent_checks.json#/checks/{list(checks).index(cid)}" if code_checks else None,
                            "practitioner_review":"not_documented","source_reverification":"not_performed",
                            "scorer_adversarial_review":"not_established_by_this_inventory"},
            "remaining_review":["independent practitioner review","scorer and response-contract audit"]
                +(["independent solution derivation"] if not code_checks else [])
                +(["primary-source sufficiency and redistribution review"] if track=="source_grounded" else [])})
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate case IDs")
    return {"audit_version":AUDIT_VERSION,"scope":"Tracked static collection only; generated environments and repeated attempts are not additional static cases.",
        "public_repository_warning":"Every answer key and reserved-labelled specification in this inventory is published. Directory and split labels do not provide secrecy or demonstrate contamination-free evaluation.",
        "catalog_inputs":[member(ROOT/"questions/catalog.json"),member(ROOT/"questions/variants/catalog.json")],
        "summary":{"main_cases":sum(r["bank"]=="main" for r in rows),"related_variants":sum(r["bank"]=="related_variant" for r in rows),
                   "main_output_fields":sum(r["declared_output_count"] for r in rows if r["bank"]=="main"),
                   "variant_output_fields":sum(r["declared_output_count"] for r in rows if r["bank"]=="related_variant"),
                   "by_track":dict(sorted(Counter(r["track"] for r in rows).items())),
                   "independently_recomputed_cases":len(checks),"matching_cases":evidence["matching_cases"],
                   "cases_pending_independent_recomputation":len(rows)-len(checks)},"cases":rows}


def serialized(document):
    return json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+"\n"


def evidence_equal(left, right):
    """Permit platform-level floating-point variation while binding exact structure."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(evidence_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(evidence_equal(a, b) for a, b in zip(left, right))
    if type(left) is float and type(right) is float:
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-8)
    return type(left) is type(right) and left == right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Verify checked-in records without modifying them")
    args=parser.parse_args()
    evidence=independent_results()
    inventory=build_inventory(evidence)
    artifacts={"independent_checks.json":evidence,"inventory.json":inventory}
    for name,document in artifacts.items():
        path=ROOT/"quality"/name
        generated=serialized(document)
        if args.check:
            if not path.exists() or not (evidence_equal(read(path), document) if name == "independent_checks.json" else path.read_text() == generated):
                raise SystemExit(f"Audit record is stale: quality/{name}")
        else:
            path.write_text(generated)
    print(json.dumps({"inventory":inventory["summary"],"numeric_checks":evidence["case_count"],"matching_checks":evidence["matching_cases"]},sort_keys=True))
    if evidence["matching_cases"]!=evidence["case_count"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
