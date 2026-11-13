"""Independent risk, collateral and settlement ledgers from public inputs."""
from fractions import Fraction
import itertools
import math

from .numerical import derivative, discrete_var, fraction, solve_linear, tail_expectation, variance


def es(losses,probabilities,alpha):
    return tail_expectation(losses,probabilities,1-fraction(alpha))[0]


def q028(inputs):
    old=inputs["existing_losses"];trade=inputs["trade_losses"];prob=inputs["probabilities"];alpha=inputs["alpha"]
    combined=[fraction(a)+fraction(b) for a,b in zip(old,trade)]
    before=es(old,prob,alpha);after=es(combined,prob,alpha)
    return {"new_es":float(after),"incremental_es":float(after-before),"standalone_trade_es":float(es(trade,prob,alpha))}, {
        "combined_losses":list(map(float,combined)),"prior_es":float(before),"full_trade_applied_before_tail_selection":True}


def q030(inputs):
    covariance,loading=inputs["shock_covariance"],inputs["loss_loadings"]
    direction=[sum(c*x for c,x in zip(row,loading)) for row in covariance]
    norm_squared=sum(x*y for x,y in zip(loading,direction));multiplier=inputs["loss_barrier"]/norm_squared
    shock=[multiplier*x for x in direction];precision_shock=solve_linear(covariance,shock)
    severity=math.sqrt(sum(x*y for x,y in zip(shock,precision_shock)))
    return {"shock":shock,"severity":severity,"loss":sum(x*y for x,y in zip(shock,loading))}, {
        "global_cauchy_schwarz_lower_bound":inputs["loss_barrier"]/math.sqrt(norm_squared),"stationarity_residuals":[a-multiplier*b for a,b in zip(precision_shock,loading)]}


def q031(inputs):
    c=inputs["loss_coefficients"];severity=inputs["severity_weights"];candidates=[]
    for x,y in itertools.product(inputs["x_grid"],inputs["y_grid"]):
        loss=c["x"]*x+c["y"]*y+c["xy"]*x*y+c["x_squared"]*x*x
        if loss>=inputs["loss_barrier"]:candidates.append((severity[0]*x*x+severity[1]*y*y,x,y,loss))
    selected=min(candidates);value,x,y,loss=selected
    return {"shock":[x,y],"severity":value,"loss":loss}, {"finite_points_searched":len(inputs["x_grid"])*len(inputs["y_grid"]),"feasible_points":len(candidates),"runner_up":list(sorted(candidates)[1])}


def q032(inputs):
    balance=fraction(inputs["opening_cash"])+fraction(inputs["existing_line"]);balances=[]
    for event in sorted(inputs["event_cashflows"],key=lambda event:event["time"]):
        balance+=fraction(event["flow"]);balances.append(balance)
    extra=max(0,-min(balances))
    return {"balances":list(map(float,balances)),"extra_buffer":float(extra),"required_line":float(fraction(inputs["existing_line"])+extra)}, {
        "balances_with_minimum_buffer":[float(balance+extra) for balance in balances],"negative_pro_forma_means_payment_failure":True}


def q033(inputs):
    availability,haircuts,costs,steps=(list(map(fraction,inputs[key])) for key in ("availability","haircuts","annual_cost_rates","face_block_size"))
    candidates=[];searched=0
    for treasury,corporate in itertools.product(range(int(availability[0]/steps[0])+1),range(int(availability[1]/steps[1])+1)):
        searched+=1;face=[treasury*steps[0],corporate*steps[1]]
        recognized=[face[i]*(1-haircuts[i]) for i in (0,1)]
        if recognized[1]>fraction(inputs["corporate_recognized_cap"]):continue
        deficit=max(0,fraction(inputs["required_recognized"])-sum(recognized))
        cash_blocks=math.ceil(deficit/(steps[2]*(1-haircuts[2])));cash=cash_blocks*steps[2]
        if cash>availability[2]:continue
        cost=sum(f*c for f,c in zip(face,costs))+cash*costs[2]
        candidates.append((cost,cash,treasury,corporate,sum(recognized)+cash*(1-haircuts[2])))
    cost,cash,treasury,corporate,recognized=min(candidates)
    return {"bond_blocks":[treasury,corporate],"cash":float(cash),"annual_cost":float(cost),"recognized_value":float(recognized)}, {
        "bond_menu_points":searched,"feasible_minimum_cash_completions":len(candidates),"cash_reduction_proof":"Positive cash cost makes the smallest eligible cash-grid completion optimal for every bond pair."}


def q034(inputs):
    gross=Fraction(0);by_counterparty={};by_set={}
    for row in inputs["netting_sets"]:
        marks=list(map(fraction,row["marks"]));gross+=sum(max(0,mark) for mark in marks)
        uncovered=max(0,sum(marks)-fraction(row["collateral"]));by_set[row["id"]]=float(uncovered)
        by_counterparty[row["counterparty"]]=by_counterparty.get(row["counterparty"],0)+uncovered
    return {"gross_receivables":float(gross),"total_ead":float(sum(by_counterparty.values())),"counterparty_ead":[float(by_counterparty.get(name,0)) for name in ("A","B")]}, {"legal_set_exposures":by_set,"collateral_reallocated_across_sets":False}


def q035(inputs):
    joint=[fraction(state)*fraction(default) for state,default in zip(inputs["state_probabilities"],inputs["default_probability_given_state"])]
    default=sum(joint);conditional=[mass/default for mass in joint]
    losses=[fraction(value)*fraction(inputs["unsecured_loss_given_default"]) for value in inputs["positive_receivables"]]
    alpha=inputs["conditional_tail_alpha"];tail=es(losses,conditional,alpha)
    return {"default_probability":float(default),"state_probabilities_given_default":list(map(float,conditional)),"conditional_var95":float(discrete_var(losses,conditional,alpha)),"conditional_es95":float(tail),"buffer_inadequate":tail>fraction(inputs["workout_buffer"])}, {
        "state_and_default_joint_probabilities":list(map(float,joint)),"conditional_probability_sum":float(sum(conditional)),"conditional_loss_states":list(map(float,losses))}


def q036(inputs):
    realized=[fraction(inputs["current_collateral_market_value"])*(1-fraction(decline))*(1-fraction(inputs["liquidation_cost_rate"])) for decline in inputs["collateral_price_declines"]]
    ead=[max(0,fraction(inputs["initial_mark"])+fraction(change)-collateral) for change,collateral in zip(inputs["mark_changes"],realized)]
    losses=[value*fraction(inputs["unsecured_lgd"]) for value in ead]
    expected=sum(value*fraction(prob) for value,prob in zip(losses,inputs["probabilities"]))
    return {"closeout_ead":list(map(float,ead)),"conditional_expected_loss":float(expected),"conditional_es95":float(es(losses,inputs["probabilities"],.95))}, {
        "collateral_realization":list(map(float,realized)),"recognition_haircut_reapplied_at_closeout":False}


def q037(inputs):
    rows=inputs["desk_factor_loadings"];weights=inputs["desk_weights"];covariance=inputs["factor_covariance"]
    def aggregate(ws):return [sum(w*row[j] for w,row in zip(ws,rows)) for j in range(len(covariance))]
    def volatility(ws):return math.sqrt(variance(aggregate(ws),covariance))
    vol=volatility(weights);contributions=[];errors=[]
    for i in range(len(weights)):
        def scaled(scale):
            changed=weights[:];changed[i]*=scale;return volatility(changed)
        contribution,error=derivative(scaled,1,1e-3);contributions.append(contribution);errors.append(error)
    loading=aggregate(weights);share=loading[0]*sum(c*w for c,w in zip(covariance[0],loading))/vol**2
    removed=weights[:];removed[2]=0
    return {"volatility":vol,"desk_contributions":contributions,"factor1_variance_share":share,"remove_c_vol_change":volatility(removed)-vol}, {
        "aggregate_factor_loadings":loading,"euler_contribution_sum_residual":sum(contributions)-vol,"derivative_mesh_changes":errors}


def q038(inputs):
    exposures=inputs["asset_usd_values"][:];exposures[0]-=inputs["eur_short_hedge_usd"]
    vols=inputs["fx_return_volatilities"];cross=math.prod(vols)*inputs["fx_correlation"]
    covariance=[[vols[0]**2,cross],[cross,vols[1]**2]]
    return {"net_fx_notional":sum(exposures),"residual_volatility":math.sqrt(variance(exposures,covariance)),"stress_loss":-sum(w*ret for w,ret in zip(exposures,inputs["stress_fx_returns"]))}, {
        "signed_currency_exposures":exposures,"absolute_currency_exposure":sum(map(abs,exposures)),"covariance_determinant":covariance[0][0]*covariance[1][1]-cross*cross}


def q039(inputs):
    states=[]
    for name,regime,losses,probabilities in zip(inputs["regime_names"],inputs["regime_probabilities"],inputs["conditional_losses"],inputs["conditional_probabilities"]):
        states.extend((name,fraction(loss),fraction(regime)*fraction(prob)) for loss,prob in zip(losses,probabilities))
    losses=[row[1] for row in states];prob=[row[2] for row in states];var=discrete_var(losses,prob,.95)
    exceed=sum(p for _,loss,p in states if loss>var);crisis=sum(p for name,loss,p in states if loss>var and name=="crisis")
    calm=inputs["regime_names"].index("calm")
    return {"unconditional_var95":float(var),"unconditional_es95":float(es(losses,prob,.95)),"calm_es95":float(es(inputs["conditional_losses"][calm],inputs["conditional_probabilities"][calm],.95)),"crisis_given_var_exceedance":float(crisis/exceed)}, {
        "unconditional_probabilities":list(map(float,prob)),"strict_exceedance_probability":float(exceed),"probability_at_var":float(sum(p for loss,p in zip(losses,prob) if loss==var))}


def q041(inputs):
    drawdowns=[]
    for path in inputs["paths"]:
        timeline=[inputs["initial_nav"]]+path
        drawdowns.append(max(timeline[i]-timeline[j] for i in range(len(timeline)) for j in range(i,len(timeline))))
    terminal=[inputs["initial_nav"]-path[-1] for path in inputs["paths"]];prob=inputs["path_probabilities"]
    trigger=sum(fraction(p) for drawdown,p in zip(drawdowns,prob) if drawdown>=inputs["drawdown_trigger"])
    return {"maximum_drawdowns":drawdowns,"drawdown_es75":float(es(drawdowns,prob,inputs["alpha"])),"terminal_loss_es75":float(es(terminal,prob,inputs["alpha"])),"trigger_probability":float(trigger)}, {
        "terminal_losses":terminal,"method":"Enumerate every ordered peak/later-observation pair, including the opening NAV."}


def q042(inputs):
    if inputs["guarantee_valid_when"]!="s<4":raise ValueError("Unreviewed guarantee eligibility contract")
    loans=[min(inputs["loan_loss_cap"],inputs["loan_loss_slope"]*s) for s in inputs["stress_levels"]]
    reimbursements=[min(inputs["guarantee_cap"],loss) if s<4 else 0 for s,loss in zip(inputs["stress_levels"],loans)]
    net=[loan-reimb+inputs["other_loss_slope"]*s for loan,reimb,s in zip(loans,reimbursements,inputs["stress_levels"])]
    breach=min(s for s,loss in zip(inputs["stress_levels"],net) if loss>inputs["capital_limit"])
    return {"net_losses":net,"es90":float(es(net,inputs["probabilities"],.9)),"first_breach_level":breach,"expected_loss":float(sum(fraction(p)*loss for p,loss in zip(inputs["probabilities"],net)))}, {
        "gross_loan_losses":loans,"guarantee_reimbursements":reimbursements,"eligibility_checked_before_reimbursement":True}


def q044(inputs):
    evaluated=[]
    for quantity in sorted(inputs["permitted_sale_quantities"]):
        price=fraction(inputs["price_intercept_usd"])-fraction(inputs["price_impact_usd_per_share"])*quantity
        debt=fraction(inputs["debt_usd"])-quantity*price;assets=(inputs["shares"]-quantity)*price
        feasible=debt>=0 and assets>0 and debt<=fraction(inputs["max_ltv"])*assets
        evaluated.append((quantity,price,debt,assets,feasible))
    selected=next(row for row in evaluated if row[-1]);quantity,price,debt,assets,_=selected
    return {"shares_sold":quantity,"clearing_price":float(price),"final_ltv":float(debt/assets),"final_equity":float(assets-debt)}, {
        "all_smaller_menu_sales_infeasible":all(not row[-1] for row in evaluated if row[0]<quantity),"remaining_debt":float(debt),"remaining_marked_assets":float(assets),"equity_uniform_mark_identity_error":float(assets-debt-(inputs["shares"]*price-fraction(inputs["debt_usd"])))}


def q046(inputs):
    by_counterparty={};exposures=[];balances=[];cash=fraction(inputs["opening_cash"])+fraction(inputs["existing_line"]);default_exposure=Fraction(0)
    for event in sorted(inputs["schedule"],key=lambda event:event["time"]):
        name=event["counterparty"];amount=fraction(event["paid_minus_received"])
        by_counterparty[name]=by_counterparty.get(name,0)+amount;cash-=amount
        exposures.append(sum(max(0,value) for value in by_counterparty.values()));balances.append(cash)
        if event["time"]<inputs["default_time"]:default_exposure=by_counterparty.get(inputs["default_counterparty"],0)
    return {"peak_principal_exposure":float(max(exposures)),"extra_line":float(max(0,-min(balances))),"a_default_loss":float(default_exposure*(1-fraction(inputs["recovery"])))}, {
        "performing_cash_timeline":list(map(float,balances)),"principal_exposure_timeline":list(map(float,exposures)),"specified_default_exposure":float(default_exposure)}


def q048(inputs):
    retained=[];reimbursements=[]
    for losses in inputs["annual_occurrence_losses"]:
        occurrence=[min(inputs["per_occurrence_reimbursement_cap"],max(0,loss-inputs["per_occurrence_deductible"])) for loss in losses]
        reimb=min(inputs["annual_reimbursement_cap"],sum(occurrence));reimbursements.append(reimb);retained.append(sum(losses)-reimb)
    return {"retained_losses":retained,"expected_retained_loss":float(sum(fraction(p)*loss for p,loss in zip(inputs["probabilities"],retained))),"retained_es80":float(es(retained,inputs["probabilities"],inputs["alpha"]))}, {
        "annual_reimbursements":reimbursements,"annual_cap_applied_after_occurrence_terms":True}


def q049(inputs):
    if (inputs["transfer_arrivals"]!={"A_to_P":"10:30","A_to_B":"09:00","P_to_B":"09:00"}
            or inputs["external_funding_to_P_at"]!="09:00"
            or {name:call["time"] for name,call in inputs["calls"].items()}!={"P":"11:00","A":"10:00","B":"10:00"}):
        raise ValueError("Cash-state certificate requires the reviewed transfer and payment schedule")
    caps=inputs["transfer_caps"];candidates=[];tested=0
    for funding,ap,ab,pb in itertools.product(range(inputs["max_external_funding"]+1),range(caps["A_to_P"]+1),range(caps["A_to_B"]+1),range(caps["P_to_B"]+1)):
        tested+=1;cash=inputs["opening_cash"].copy();cash["P"]+=funding
        cash["A"]-=ap+ab;cash["P"]-=pb;cash["B"]+=ab+pb
        timeline=[cash.copy()]
        cash["A"]-=inputs["calls"]["A"]["amount"];cash["B"]-=inputs["calls"]["B"]["amount"];timeline.append(cash.copy())
        cash["P"]+=ap;timeline.append(cash.copy())
        cash["P"]-=inputs["calls"]["P"]["amount"];timeline.append(cash.copy())
        if all(value>=0 for snapshot in timeline for value in snapshot.values()):candidates.append((funding,ap,ab,pb,cash["A"],timeline))
    funding,ap,ab,pb,trapped,timeline=min(candidates,key=lambda row:row[:4])
    deficit=max(0,sum(row["amount"] for row in inputs["calls"].values())-sum(inputs["opening_cash"].values()))
    return {"external_funding":funding,"transfers":[ap,ab,pb],"trapped_cash_a":trapped,"naive_pooled_deficit":deficit}, {
        "finite_plans_tested":tested,"feasible_plans":len(candidates),"selected_cash_timeline":timeline,"pending_ap_transfer_credited_only_at_1030":True}


CHECKS={
    "q028":(q028,"Apply the full hedge scenario by scenario before threshold-tail optimization; compare with the old and standalone tails."),
    "q030":(q030,"Covariance-metric Cauchy-Schwarz yields an attained global severity bound, verified by precision-space stationarity."),
    "q031":(q031,"Enumerate all permitted nonlinear grid shocks and apply the stated severity and lexicographic ordering."),
    "q032":(q032,"Chronological cash-prefix ledger identifies the intraday deficit before later proceeds can settle."),
    "q033":(q033,"Enumerate indivisible bond allocations and complete each with its exact least-cost cash-grid requirement; apply eligibility before cost ranking."),
    "q034":(q034,"Net and collateralize each legal set independently before counterparty aggregation; preserve gross positive trade marks separately."),
    "q035":(q035,"Form exact joint state/default masses, condition by Bayes, then apply quantile and partial-atom ES to conditional losses."),
    "q036":(q036,"Scenario closeout ledger liquidates actual collateral once, floors unsecured exposure, then computes conditional loss statistics."),
    "q037":(q037,"Aggregate factor loadings and differentiate desk-weighted volatility with two meshes; separately reprice the finite desk removal."),
    "q038":(q038,"Retain separate signed currency exposures, propagate the FX covariance, and compute the stated opposing-currency stress."),
    "q039":(q039,"Expand the regime tree into exact unconditional atoms, preserve the strict VaR-exceedance event, and separate conditional calm risk."),
    "q041":(q041,"Enumerate ordered peak/trough pairs along each complete path, then separately aggregate drawdown and terminal-loss tails."),
    "q042":(q042,"Evaluate guarantee eligibility in each stress state before reimbursement and net loss; distinguish strict capital breach from equality."),
    "q044":(q044,"Exhaust the permitted uniform-price auction quantities using rational debt and asset ledgers; certify every smaller sale fails."),
    "q046":(q046,"Maintain separate intraday principal receivables and funded cash prefixes; apply recovery only to the exposure open at the specified default time."),
    "q048":(q048,"Apply occurrence deductibles and limits separately before the annual cap, then compute retained-loss expectation and tail."),
    "q049":(q049,"Enumerate all integer funding/transfer plans with departure, payment and delayed-arrival cash states; apply the full lexicographic tie rule."),
}
