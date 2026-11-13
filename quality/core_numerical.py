"""Independent opening-packet calculations for three core cases.

Governing constants below are transcribed from the public synthetic source
contracts and bound by the input packet hash in the resulting evidence record.
No stored solution or production scorer is imported.
"""
from fractions import Fraction
import itertools
import math


def c201(question):
    inputs=question["inputs"]
    opening=inputs["opening_holdings"];prices=inputs["transaction_prices"]
    feasible=[]
    for holdings in itertools.product(range(7),repeat=4):
        factor=sum(h*f for h,f in zip(holdings,(2,1,-1,-1)))
        if sum(holdings)<16 or not -2<=factor<=2 or holdings[0]+holdings[2]<6:
            continue
        trades=tuple(h-o for h,o in zip(holdings,opening))
        cost=sum(1+Fraction(q*q,4) for q in trades if q)
        early=Fraction(inputs["opening_cash"])-cost-inputs["redemption_amount"]
        for i,(trade,price) in enumerate(zip(trades,prices)):
            if trade>0 or i in (0,2):early-=trade*price
        final=early-sum(q*prices[i] for i,q in enumerate(trades) if q<0 and i in (1,3))
        if early<0 or final<5:continue
        feasible.append((cost,holdings,trades,early,final,factor))
    best=min(feasible)
    cost,holdings,trades,early,final,factor=best
    minimizers=[candidate for candidate in feasible if candidate[0]==cost]
    return {"cost":float(cost),"early_cash":float(early),"factor_exposure":factor,"final_cash":float(final),
            "final_holdings":list(holdings),"optimal_plan_count":len(minimizers),"trade_quantities":list(trades)}, {
            "enumerated_holdings":7**4,"feasible_holdings":len(feasible),"all_optimal_holdings":[list(row[1]) for row in minimizers],
            "tie_policy":"Lexicographic holdings choose one allowed optimum; the packet permits every cost minimizer.",
            "public_contract_ids":["p_book","p_limits","p_execution","p_cash"]}


def c202(question):
    rows=question["inputs"]["observations"]
    eligible=[row for row in rows if row[3]<=row[1] and row[4]<=row[1]-2]
    retained=[];last_end=-math.inf
    for row in sorted(eligible,key=lambda row:(row[2],int(row[0][1:]))):
        if row[1]>last_end:retained.append(row);last_end=row[2]
    net=[row[5]-2 for row in retained]
    signs=[value>0 for value in net if value!=0]
    probability=Fraction(sum(math.comb(len(signs),k) for k in range(sum(signs),len(signs)+1)),2**len(signs))
    documents={d["id"]:d["text"] for d in question["sources"]}
    # Count actual policy IDs in the complete trial archive, rather than the
    # surviving dashboard or a hard-coded multiplicity from the answer key.
    policies={token for token in documents["q_trial_log"].split() if len(token)==3 and token[0]=="P" and token[1:].isdigit()}
    hypotheses=len(policies)
    adjusted=min(Fraction(1),probability*hypotheses)
    mean=Fraction(sum(net),len(net))
    return {"eligible_rows":[row[0] for row in sorted(eligible,key=lambda row:int(row[0][1:]))],
            "representative_rows":[row[0] for row in retained],"independent_count":len(retained),"mean_net_bps":float(mean),
            "raw_p_value":float(probability),"research_hypotheses":hypotheses,"adjusted_p_value":float(adjusted),
            "deploy":adjusted<=Fraction(5,100) and mean>0}, {
            "nonzero_sign_count":len(signs),"positive_sign_count":sum(signs),"raw_probability_fraction":str(probability),
            "disjoint_retained_intervals":all(after[1]>before[2] for before,after in zip(retained,retained[1:])),
            "public_contract_ids":["q_tape","q_protocol","q_trial_log"]}


def c214(question):
    documents={d["id"]:d["text"] for d in question["sources"]}
    rows=[]
    for line in documents["qt_tape"].splitlines():
        if not line.startswith("b"):continue
        fields=[field.strip() for field in line.split("|")]
        cid=fields[0];decision,publication,training,market=map(int,fields[1:5])
        gross=list(map(Fraction,fields[5].split(",")))
        if publication<=decision and training<=decision-2:
            residual=[g-Fraction(1,2)-Fraction(traded*traded,2)-beta*market
                      for g,traded,beta in zip(gross,(1,2,1),(1,0,-1))]
            rows.append((cid,residual))
    checkpoints=(3,6)
    if len(rows)!=6:raise ValueError("The published protocol requires six eligible blocks")
    def rank(values):
        total=sum(values);squares=sum(x*x for x in values)
        if total==0:return (0,Fraction(0))
        return (1,total*total/squares) if total>0 else (-1,-total*total/squares)
    def statistic(values):return float(sum(values))/math.sqrt(float(sum(x*x for x in values)))
    candidates=[]
    for policy in range(3):
        for checkpoint in checkpoints:
            values=[row[1][policy] for row in rows[:checkpoint]]
            candidates.append((rank(values),-policy,checkpoint,values))
    selected=max(candidates,key=lambda row:row[:3]);observed=selected[0];policy=-selected[1];checkpoint=selected[2]
    raw_hits=adjusted_hits=0
    for signs in itertools.product((-1,1),repeat=len(rows)):
        def signed(column,length):return [row[1][column]*sign for row,sign in zip(rows[:length],signs)]
        raw_hits+=rank(signed(policy,checkpoint))>=observed
        adjusted_hits+=max(rank(signed(column,length)) for column in range(3) for length in checkpoints)>=observed
    denominator=2**len(rows);raw=Fraction(raw_hits,denominator);adjusted=Fraction(adjusted_hits,denominator)
    early=sum(row[1][policy] for row in rows[:3])/3
    late=sum(row[1][policy] for row in rows[3:])/3
    deploy=adjusted<=Fraction(5,100) and checkpoint==6 and early>0 and late>0 and late>=early/2
    return {"deploy":deploy,"eligible_blocks":[row[0] for row in rows],"late_regime_mean":float(late),
            "selected_checkpoint":checkpoint,"selected_policy":"ABC"[policy],"selected_statistic":statistic(selected[3]),
            "selection_adjusted_p":float(adjusted),"unadjusted_p":float(raw)}, {
            "joint_sign_vectors":denominator,"raw_exceedances":raw_hits,"selection_exceedances":adjusted_hits,
            "comparison_method":"Exact signed rational squared-score ordering, including equalities.",
            "early_regime_mean":float(early),"public_contract_ids":["qt_tape","qt_cost","qt_registry","qt_gate"]}


CHECKS={
    "c201":(c201,"Enumerate all 2,401 whole-unit final holdings with exact fractional fees and separate early/late settlement cash; enumerate every tied minimum."),
    "c202":(c202,"Rebuild point-in-time eligibility and nonoverlapping intervals; exact sign-test tail and actual full-archive policy-ID count."),
    "c214":(c214,"Parse the supplied source tape, form cost/beta residuals, and enumerate all 64 common-block sign vectors across all six inspected scores with exact comparison."),
}
