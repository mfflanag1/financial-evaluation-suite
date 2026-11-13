"""Read-only reproductions of known scorer-v2 semantic-integer defects.

This command is expected to demonstrate acceptance of invalid fractional counts
by the preserved v2 implementation. It does not patch or execute simulated orders.
"""
from copy import deepcopy
import json
from .inventory import ROOT, member


def reproduce():
    from scoring_v2 import engine, research
    from scoring_v2.common import catalog
    cases=catalog()
    checks=[]
    for cid,field in [("c201","trade_quantities"),("c201","optimal_plan_count"),
                      ("c202","independent_count"),("c202","research_hypotheses")]:
        case=cases[cid][1]
        session=engine.new_session(case,"supplied");engine.public_view(session)
        stage=engine.stage(session)
        response={"question_id":cid,"turn":1,"answers":deepcopy(stage["answers"]),
                  "evidence":{k:deepcopy(routes[0]) for k,routes in stage["evidence_routes"].items()}}
        baseline=engine.score(stage,response,cid,1,session["seen"])
        original=deepcopy(response["answers"][field])
        if isinstance(original,list):response["answers"][field][0]+=5e-7
        else:response["answers"][field]+=5e-7
        result=engine.score(stage,response,cid,1,session["seen"])
        checks.append({"case_id":cid,"turn":1,"field":field,"original":original,"mutated":response["answers"][field],
                       "valid_control_accepted":baseline["all_supported_correct"],
                       "fractional_value_accepted":result["all_supported_correct"],"fractional_contract_accepted":result["response_contract_valid"]})
    for cid,turn,field in [("q138",2,"delivery_units"),("q149",1,"eligible_issue_count")]:
        case=cases[cid][1];stage=case["turns"][turn-1]
        response=research.reference_response(case,turn)
        baseline=research.grade_turn(case,turn,response)
        original=deepcopy(response["answers"][field])
        if isinstance(original,list):response["answers"][field][0]+=5e-7
        else:response["answers"][field]+=5e-7
        result=research.grade_turn(case,turn,response)
        checks.append({"case_id":cid,"turn":turn,"field":field,"original":original,"mutated":response["answers"][field],
                       "valid_control_accepted":baseline["all_correct"],"fractional_value_accepted":result["all_correct"],
                       "fractional_contract_accepted":None})
    return {"affected_version":"scoring_v2","defect_id":"discrete-output-tolerance",
            "expected_behavior":"Reject fractional values in fields explicitly requiring integer quantities or unscaled counts; preserve valid integral JSON floats.",
            "scope":"Six controlled reproductions across challenge and sequential-research tracks. This is not an exhaustive defect count.",
            "implementation_artifacts":[member(ROOT/"scoring_v2/engine.py"),member(ROOT/"scoring_v2/research.py")],"reproductions":checks}


def main():
    print(json.dumps(reproduce(),indent=2,allow_nan=False))


if __name__=="__main__":main()
