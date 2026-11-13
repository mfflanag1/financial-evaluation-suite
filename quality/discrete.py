"""Stage-aware candidate inventory for semantic discrete response fields.

Decisions use output declarations and governing text, never the numerical form
of an answer key. Approved selectors identify scalar numbers or numeric leaves;
other fields are retained as reviewed candidates or outside this specific audit.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from . import AUDIT_VERSION
from .inventory import ROOT, member, read, serialized

# A selector [] means the whole scalar; ["*"] means every array element.
# Nested selectors address mixed arrays without coercing unrelated values.
OVERRIDES = {
    "macro-001": {name:("scalar","Keep signed revision values and exact integer thousands") for name in (
        "jan_first_print","jan_asof_mar08","feb_first_print","mar08_prior_two_month_revision",
        "jan_asof_apr05","feb_asof_apr05","mar_first_print","apr05_prior_two_month_revision",
        "q1_first_print_sum","q1_asof_apr05_sum","q1_cumulative_revision_vs_first_prints",
        "apr05_increment_to_tracked_q1_change")},
    "h2-004": {"export_credit_test_precedes_new_revolver_months":("scalar","how many calendar months earlier")},
    "h2-008": {"oddlot_whole_kvue_shares":("scalar","Whole Kenvue shares")},
    "h2-009": {"hedge_contracts":("scalar","Number of whole short basis contracts")},
    "h2-010": {"delivery_days":("scalar","Inclusive number of days")},
    "q017": {"sales":("*","Sales are permitted only at the listed share counts per lot")},
    "q044": {"shares_sold":("scalar","Allowed sales are exactly")},
    "q049": {k:("*" if k=="transfers" else "scalar","All transfer amounts and F are integer millions") for k in ("external_funding","transfers")},
    "q063": {"hedge_units":("*","each a permitted nonnegative grid value")},
    "q068": {"quantity":("scalar","only integer barrels are allowed")},
    "q098": {k:("scalar","Day numbering starts at 1; use 0 for no promotion") for k in ("likelihood_stop_day","naive_fixed_test_stop_day")},
    "q091": {"schedule":("*","Each slice sells an integer number of lots")},
    "q113": {k:("*","Only integer X and Y units are allowed") for k in ("base_units","unserved_demand","expedited_b_units")},
    "q114": {"minimum_cost_order_units":("*","Quantities are integers")},
    "q123": {"nonanticipative_purchase_policy":("*","Every action is in grid0,10,20,30,40")},
    "q127": {"lots":("*","Choose the integer long/short lot plan")},
    "q131": {"deployment":("*","Allocate integer capital units")},
    "q134": {k:("*","Inventories are indivisible units") for k in ("t0_collateral_units","t1_collateral_units","repo_units","remaining_unencumbered_units")},
    "q138": {"delivery_units":("*","Unit quantities are whole contracts")},
    "q147": {k:("scalar","New draws and repurchases must each be integer multiples of5") for k in ("repurchase_cash","new_draw")},
    "q151": {"nominations":("*","Each gross nomination is a multiple of5")},
    "q154": {"selected_cash_and_calls":("*","cash units are integers")},
    "q158": {"contracts_closed":("*","Close only integer quantities")},
    "q159": {k:("scalar","Close exactly enough new contracts to meet both") for k in ("new_contracts_remaining","new_contracts_closed")},
    "q161": {"shares_sold":("scalar","using the supplied sale grid")},
    "q162": {"shares_sold":("0","Sell integer A shares")},
    "q164": {"new_order_quantity":("scalar","board lot5; no borrowing")},
    "c212": {k:("scalar","whole board lots") for k in ("confirmed_inventory","inventory_after_new_order_fill","live_buy_quantity","target_shortfall_after_new_order")},
    "c219": {"action":("*","integer lots")},
    "c220": {"action":("*","integer vector"),"diagnostics":("2/*","eligibility represented as integers 0 or 1")},
    "c221": {"action":("scalar","desired integer total hedge holding")},
    "c222": {"action":("*","nonnegative integers each at most eight")},
    "c224": {"action":("1","BUY/SELL quantity is an integer one through ten")},
}

EXCLUSIONS = {
    ("q054","stock_hedge"):"The option counts are integer-constrained; the share hedge is specified by exact delta neutrality without an explicit integer-share restriction.",
    ("q085","q_max_shares"):"The prompt expressly treats order size as continuous for the impact budget.",
    ("q089","equivalent_ewma_half_life"):"A half-life is a real-valued duration even though its unit is months.",
    ("q121","remaining_conversion_shares"):"The output is measured in millions of shares and need not be an integer in those units.",
    ("q129","maximum_common_gate"):"The investor redemption gate is fractional; only the underlying asset-sale quantities are integer constrained.",
    ("q140","post_buyback_shares"):"Million-share denominator is not itself stated to be an integer; integer authorization quantities are separate fields.",
}

# Manual dispositions apply only to the named output contracts. A unit word is
# not sufficient to extend an integer restriction from another output field.
REVIEWED_GROUPS = [
    ("Cash, valuation, loss, cost, or normalized exposure is not an unscaled item count; no additional integer restriction is stated.", {
        "c201":"cost early_cash factor_exposure final_cash",
        "c203":"chosen_expected_losses model_value_interval worst_case_expected_loss worst_case_regret",
        "c205":"expected_query_cost information_value no_information_value optimal_net_value",
        "c207":"foregone_value post_cash worst_stress_loss", "c212":"cash_balance",
        "c213":"implementation_cost", "c216":"remaining_stress_risk",
        **{f"c{i}":"constraint_violation transaction_cost" for i in range(219,225)},
        "equity-002":"recast_2023_consolidated_revenue",
        "h2-002":"shareholder_cash_consideration_net_of_acquired_cash",
        "h2-007":"retained_kd_basis", "q101":"daily_cash_local",
        "q123":"early_demand_information_cost expected_total_cost leaf_purchase_cash value_of_early_demand_information",
        "q126":"delta_hedged_pnls", "q129":"fees_paid_cash new_high_water_marks remaining_investor_nav",
        "q135":"deployment_objectives", "q146":"equity_value_per_share",
        "q158":"portfolio_delta remaining_delta", "q163":"study_net_expected_payoff",
        "systematic-003":"terminal_liquidation_cost", "vol-001":"book_strike_payment",
        "vol-003":"book_strike_payment equivalent_underlying_quote",
    }),
    ("This is a ratio, continuous statistic, coefficient, or severity value; the wording refers to units or observations without imposing an integer domain.", {
        "q015":"cash_threshold_radius", "q025":"alpha_multiplier cap_multiplier relative_entropy",
        "q031":"severity", "q054":"residual_greeks", "q082":"z_score",
        "q083":"leak_fraction_unpurged", "q103":"test_net_returns test_turnover test_weights",
        "q111":"control_coefficients issuer_coefficients", "q124":"fuel_used",
        "q133":"complete_case_gross_mean net_mean_by_completion",
    }),
    ("The field represents a contractual allocation, share-ratio entitlement, hedge exposure, or related consideration; the contract does not require rounding this field to whole shares.", {
        "event-001":"b_short_hedge_shares",
        "event-002":"c_cash_fulfillment_fraction c_holder_broadcom_shares c_holder_cash s_holder_broadcom_shares s_holder_cash s_short_hedge_shares s_stock_election_cash_fraction",
        "h2-005":"fundamental_change_common_share_entitlement fundamental_minus_ordinary_optional_shares normal_mandatory_common_share_entitlement normal_minus_optional_marked_value ordinary_optional_common_share_entitlement preferred_shares_represented",
        "h2-012":"contractual_ubs_share_delivery",
    }),
    ("Net gas quantities include the stipulated route loss. The gross nomination grid does not impose integrality on delivered or unserved volume.", {
        "q151":"delivered_by_route unserved_volume",
    }),
    ("Unfilled investor-unit requests are computed after a pro-rata gate and are not required to be whole units.", {"q157":"pending_units"}),
    ("The output is a truth value, not a numeric count.", {"c211":"state_identified"}),
    ("The mechanics declare a categorical action string; the generic output description's unit wording is not a numeric schema.", {"c223":"action"}),
]
REVIEWED = {(cid,name):reason for reason,cases in REVIEWED_GROUPS for cid,names in cases.items() for name in names.split()}

CLARIFICATION_GROUPS = [
    ("The field records shares but its public declaration does not explicitly require whole shares. Establish an exact leaf-level quantity contract before adding integral validation.", {
        "h2-007":"retained_ibm_shares", "q101":"ending_quantities",
        "q102":"ending_security_ledger", "q161":"original_p_retained q_shares_bought",
        "q164":"confirmed_inventory inventory_if_filled live_buy_quantity new_order_quantity potential_overfill remaining_target_shortfall replacement_quantity_after_clean_cancel worst_case_inventory",
    }),
    ("The declared quantity is derived from constrained decisions, but the output itself has no explicit integer declaration. A separate closure proof or clarified output contract is needed; integral reference values alone are insufficient.", {
        "q123":"leaf_ending_inventory leaf_shortfalls", "q131":"asset_flows risk_units_used unused_capital",
    }),
]
CLARIFICATION = {(cid,name):reason for reason,cases in CLARIFICATION_GROUPS for cid,names in cases.items() for name in names.split()}

CANDIDATE = re.compile(r"\binteger|\bwhole\b|\bindivisible|\bcount\b|\bcontracts\b|\bshares\b|\bunits\b|\blots\b|\bobservations\b|\bmonths\b|\bplans\b|\bpolicies\b|\bstarts\b",re.I)


def selector(code,kind):
    if code=="scalar": return []
    return [int(part) if part.isdigit() else part for part in code.split("/")]


def records():
    for metadata in read(ROOT/"questions/catalog.json"):
        cid,track=metadata["id"],metadata["track"]
        if track=="source_grounded":
            path=ROOT/"questions"/metadata["path"]; c=read(path)
            outputs=[dict(o,name=o["id"]) for o in c["output_fields"]]
            yield cid,track,1,outputs,[(member(path,"/prompt"),c["prompt"])],path,"/output_fields"
    for track in ("quant100","workflows","research58"):
        path=ROOT/f"private_evaluator/legacy/{track}/evaluator.json"
        for ci,c in enumerate(read(path)):
            context=[]
            for ti,t in enumerate(c.get("turns",[c])):
                pointer=f"/{ci}/turns/{ti}" if track=="research58" else f"/{ci}"
                for key in ("prompt","user_message"):
                    if key in t: context.append((member(path,f"{pointer}/{key}"),t[key]))
                for si,source in enumerate(t.get("sources",[])):
                    context.append((member(path,f"{pointer}/sources/{si}/text"),source["text"]))
                yield c["id"],track,ti+1,t["outputs"],list(context),path,f"{pointer}/outputs"
    for track in ("challenges","variants"):
        path=ROOT/f"private_evaluator/{track}/cases.json"
        for ci,c in enumerate(read(path)):
            for ti,t in enumerate(c["stages"]):
                context=[(member(path,f"/{ci}/documents/{di}/text"),d["text"]) for di,d in enumerate(c["documents"]) if d.get("release_stage",1)<=ti+1]
                context.append((member(path,f"/{ci}/stages/{ti}/prompt"),t["prompt"]))
                # Types are intentionally absent: classification cannot inspect reference values.
                outputs=[{"name":name,"description":description} for name,description in t["descriptions"].items()]
                yield c["id"],track,ti+1,outputs,context,path,f"/{ci}/stages/{ti}/descriptions"


def build():
    rows=[]
    seen=set()
    for cid,track,turn,outputs,context,path,pointer in records():
        for index,output in enumerate(outputs):
            name=output["name"]; desc=output.get("description",""); unit=output.get("unit","")
            origin=member(path,f"{pointer}/{name}" if track in ("challenges","variants") else f"{pointer}/{index}")
            text=desc+" "+unit
            evidence=[]; selections=[]; reason=None
            if (cid,name) in EXCLUSIONS:
                status="explicit_exclusion"; reason=EXCLUSIONS[(cid,name)]
            elif name in OVERRIDES.get(cid,{}):
                code,phrase=OVERRIDES[cid][name]
                support=[(loc,body) for loc,body in [(origin,desc)]+context if phrase in body]
                if not support:
                    # The field may occur before a governing amendment is released.
                    status="candidate_requires_review";reason="Relevant integer amendment is not yet present in this stage."
                else:
                    selections=[selector(code,output.get("type"))]
                    status="approved_integer";reason="Explicit governing restriction or integer-only permitted menu."
                    loc,body=support[0];start=body.index(phrase)
                    evidence=[{"artifact":loc,"quote":body[max(0,start-35):min(len(body),start+len(phrase)+150)]}]
            elif re.search(r"\binteger|\bwhole(?:-| )?(?:share|barrel)|\bwhole-share",text,re.I):
                status="approved_integer";reason="Output declaration explicitly requires integers or whole quantities."
                is_array=output.get("type")=="number_array" or bool(re.search(r"array|\[",desc,re.I))
                selections=[["*"]] if is_array else [[]]
                evidence=[{"artifact":origin,"quote":desc,"unit":unit}]
            elif (unit in ("count","observations","plans","policies","starts") and "fraction" not in output.get("type","")) or re.search(r"^Number of (?:admissible trades|net long futures contracts|deliverable issues|integer plans|accepted shares|post-consolidation ADS)",desc):
                status="approved_integer";reason="Output is an unscaled count of discrete items."
                selections=[[]];evidence=[{"artifact":origin,"quote":desc,"unit":unit}]
            elif CANDIDATE.search(text) and output.get("type") not in ("boolean","string","string_array"):
                if (cid,name) in REVIEWED:
                    status="reviewed_no_additional_integer_constraint";reason=REVIEWED[(cid,name)]
                elif (cid,name) in CLARIFICATION:
                    status="contract_clarification_needed";reason=CLARIFICATION[(cid,name)]
                else:
                    status="candidate_requires_review";reason="Quantity-like wording alone does not prove an integer constraint."
                evidence=[{"artifact":origin,"quote":desc,"unit":unit}]
            else:
                status="no_integer_requirement_identified";reason="No integer assertion established by this semantic audit."
            key=(cid,turn,name)
            if key in seen: raise ValueError(f"Duplicate discrete audit field {key}")
            seen.add(key)
            rows.append({"case_id":cid,"track":track,"turn":turn,"field":name,"status":status,
                         "integer_selectors":selections,"reason":reason,"evidence":evidence})
    rows.sort(key=lambda row:(row["case_id"],row["turn"],row["field"]))
    return {"audit_version":AUDIT_VERSION,"scope":"All 1,972 declared output fields across 224 main cases and six related variants; stage-aware candidate review. This inventory does not alter scoring.",
            "selector_contract":"A selector [] targets the whole scalar; '*' visits each array element; numeric selectors target mixed-array positions. Only approved_integer rows are recommendations for exact finite integral validation. JSON booleans are not numbers. Integral forms such as 2.0 may be accepted.",
            "limitations":["Contract-clarification rows have been reviewed but do not establish an enforceable integer rule; any remaining candidates require manual adjudication.","Finite noninteger grids, categorical choices, quantity bounds, and multiples require separate validators.","No field is integer-classified because its reference answer happens to be integral.","This is a semantic schema audit, not an independent financial correctness certificate."],
            "summary":{"cases":len({r['case_id'] for r in rows}),"stage_fields":len(rows),"by_status":dict(sorted(Counter(r["status"] for r in rows).items()))},"fields":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    value=build();path=ROOT/"quality/discrete_fields.json";text=serialized(value)
    if args.check:
        if not path.exists() or path.read_text()!=text: raise SystemExit("Discrete-field inventory is stale")
    else:path.write_text(text)
    print(json.dumps(value["summary"],sort_keys=True))


if __name__=="__main__":main()
