"""Independent allocation and financing checks from synthetic public inputs.

Finite menus, active-set enumeration, primal constraints and dual certificates
are used before any separate stored-reference comparison.
"""
import itertools
import math

from .numerical import fraction, minimize_interval, solve_linear, variance


def q001(inputs):
    equity,long,short=map(fraction,[inputs["equity_usd_m"],inputs["long_usd_m"],inputs["short_usd_m"]])
    debt=max(0,long-equity)
    ledger={"long_total_return":long*fraction(inputs["long_total_return"]),
            "short_security_return":-short*fraction(inputs["short_security_total_return"]),
            "restricted_cash_rebate":short*fraction(inputs["short_collateral_rebate"]),
            "unsecured_funding":-debt*fraction(inputs["unsecured_funding_rate"])}
    return {"borrowing":float(debt),"pnl":float(sum(ledger.values())),"gross_leverage":float((long+short)/equity)}, {
        "pnl_ledger":{name:float(value) for name,value in ledger.items()},"restricted_collateral_used_to_fund_longs":0}


def q003(inputs):
    matrix=inputs["hedge_factor_loadings"];exposure=inputs["existing_factor_exposure"]
    hedge=solve_linear(matrix,[-value for value in exposure])
    return {"hedge":hedge,"net_notional":sum(hedge),"gross_notional":sum(map(abs,hedge))}, {
        "remaining_factor_exposures":[original+sum(loading*position for loading,position in zip(row,hedge)) for row,original in zip(matrix,exposure)]}


def q004(inputs):
    vols=inputs["annual_volatility"];cross=math.prod(vols)*inputs["correlation"]
    covariance=[[vols[0]**2,cross],[cross,vols[1]**2]]
    def contributions(weight):
        weights=[weight,1-weight]
        return [w*sum(c*z for c,z in zip(row,weights)) for w,row in zip(weights,covariance)]
    lo,hi=0.0,1.0
    for _ in range(100):
        mid=(lo+hi)/2;rc=contributions(mid)
        if rc[0]<rc[1]:lo=mid
        else:hi=mid
    equal=(lo+hi)/2;capped=min(equal,inputs["asset_a_weight_cap"]);rc=contributions(capped)
    return {"unconstrained_a":equal,"capped_a":capped,"a_risk_share":rc[0]/sum(rc)}, {
        "equal_weight_contributions":contributions(equal),"capped_contributions":rc,"contribution_sum_equals_variance":sum(rc)-variance([capped,1-capped],covariance)}


def q005(inputs):
    active=inputs["active_stock_weights"]
    beta=sum(w*b for w,b in zip(active,inputs["market_betas"]));overlay=-beta
    idio=sum(w*w*v for w,v in zip(active,inputs["idiosyncratic_variance"]))
    mean=sum(w*a for w,a in zip(active,inputs["stock_alphas"]))+(beta+overlay)*inputs["market_expected_return"]
    return {"overlay":overlay,"tracking_error":math.sqrt(idio+(beta+overlay)**2*inputs["market_variance"]),"active_return":mean}, {
        "active_beta_after_overlay":beta+overlay,"active_stock_weights_sum":sum(active),"idiosyncratic_variance":idio}


def q007(inputs):
    corners=list(itertools.product(*[(mean-radius,mean+radius) for mean,radius in zip(inputs["nominal_mean"],inputs["mean_box_radius"])]))
    results={}
    for name,weights in inputs["menu"].items():
        risk=inputs["risk_aversion"]*variance(weights,inputs["covariance"])/2
        cost=inputs["linear_cost"]*sum(abs(w-old) for w,old in zip(weights,inputs["initial_weights"]))
        utility=lambda means:sum(w*mean for w,mean in zip(weights,means))-risk-cost
        corner_values=[utility(corner) for corner in corners]
        results[name]={"worst":min(corner_values),"nominal":utility(inputs["nominal_mean"]),"worst_corner":list(corners[corner_values.index(min(corner_values))])}
    robust=min(results,key=lambda name:(-results[name]["worst"],name));nominal=min(results,key=lambda name:(-results[name]["nominal"],name))
    return {"robust_choice":robust,"robust_utility":results[robust]["worst"],"nominal_choice":nominal}, {"menu_results":results,"corners_evaluated_per_menu":len(corners)}


def q008(inputs):
    def utility(weight):
        return sum(w*mean for w,mean in zip((weight,1-weight),inputs["mean_returns"]))-inputs["risk_aversion"]/2*sum(w*w*v for w,v in zip((weight,1-weight),inputs["diagonal_covariance"]))
    initial=inputs["initial_weight_a"];cap=inputs["total_absolute_weight_change_cap"]
    def optimize(capacity):
        low=max(0,initial-capacity/2);high=min(1,initial+capacity/2)
        weight=minimize_interval(lambda w:-utility(w),low,high)
        return weight,utility(weight)
    weight,value=optimize(cap);step=1e-4
    right=(optimize(cap+step)[1]-value)/step;fine=(optimize(cap+step/2)[1]-value)/(step/2)
    shadow=2*fine-right
    return {"weight_a":weight,"turnover":2*abs(weight-initial),"turnover_shadow_price":shadow}, {
        "utility":value,"right_derivative_mesh_change":fine-right,"turnover_interval":[max(0,initial-cap/2),min(1,initial+cap/2)]}


def proportional_choices(initial,alpha,risk,cost):
    """Stationary points and endpoints of both smooth pieces of the utility."""
    candidates={0.0,1.0,initial}
    sell=(alpha+cost)/risk;buy=(alpha-cost)/risk
    if 0<=sell<=initial:candidates.add(sell)
    if initial<=buy<=1:candidates.add(buy)
    return {weight:alpha*weight-risk*weight*weight/2-cost*abs(weight-initial) for weight in candidates}


def q009(inputs):
    initial,risk,cost=inputs["risky_weight_before"],inputs["quadratic_risk_coefficient"],inputs["proportional_one_way_cost"]
    choices=[proportional_choices(initial,inputs[key],risk,cost) for key in ("base_alpha","updated_alpha")]
    best=[max(menu,key=lambda w:(menu[w],w==initial)) for menu in choices]
    return {"base_weight":best[0],"updated_weight":best[1],"no_trade_alpha_interval":[risk*initial-cost,risk*initial+cost]}, {
        "enumerated_stationary_candidates":[[{"weight":w,"utility":v} for w,v in sorted(menu.items())] for menu in choices]}


def q010(inputs):
    benchmark=inputs["benchmark"][0];scale=math.sqrt(variance([1,-1],inputs["covariance"]))
    radius=inputs["tracking_error_limit"]/scale;low=max(0,benchmark-radius)
    bounds={"concentration":inputs["relaxed_a_cap"],"tracking_error":benchmark+radius,"long_only":1}
    direction=inputs["expected_return"][0]-inputs["expected_return"][1]
    if direction<=0:raise ValueError("This case certificate assumes the supplied positive A return premium")
    base=min(inputs["base_a_cap"],benchmark+radius,1);relaxed=min(bounds.values())
    return {"base_a":base,"relaxed_a":relaxed,"base_tracking_error":abs(base-benchmark)*scale,"relaxed_binding":min(bounds,key=bounds.get)}, {
        "tracking_error_feasible_a_interval":[low,min(1,benchmark+radius)],"active_return_slope":direction}


def bounded_diagonal_minimum(variances,lower,upper):
    candidates=[]
    for status in itertools.product(("free","lower","upper"),repeat=len(variances)):
        weights=[None if state=="free" else lower[i] if state=="lower" else upper[i] for i,state in enumerate(status)]
        free=[i for i,state in enumerate(status) if state=="free"]
        remaining=1-sum(value for value in weights if value is not None)
        if free:
            inverse=sum(1/variances[i] for i in free)
            for i in free:weights[i]=remaining/variances[i]/inverse
        elif abs(remaining)>1e-12:continue
        if any(w<lo-1e-12 or w>hi+1e-12 for w,lo,hi in zip(weights,lower,upper)):continue
        candidates.append((sum(w*w*v for w,v in zip(weights,variances)),weights,status))
    return min(candidates,key=lambda row:row[0]),len(candidates)


def q011(inputs):
    variances=inputs["diagonal_covariance"];lower=[0,inputs["asset_b_lower"],0];upper=[inputs["asset_a_upper"],1,1]
    (value,weights,status),count=bounded_diagonal_minimum(variances,lower,upper)
    free=[i for i,w in enumerate(weights) if lower[i]+1e-10<w<upper[i]-1e-10]
    common=2*variances[free[0]]*weights[free[0]]
    multiplier=max(0,common-2*variances[0]*weights[0]) if abs(weights[0]-upper[0])<1e-10 else 0.0
    return {"weights":weights,"variance":value,"cap_multiplier":multiplier}, {
        "feasible_active_set_candidates":count,"selected_active_set":list(status),"free_coordinate_stationarity":[2*variances[i]*weights[i]-common for i in free]}


def q012(inputs):
    variances=inputs["diagonal_covariance"];rows=[[1.0]*len(variances),inputs["factor_loadings"],inputs["signal_loadings"]]
    gram=[[sum(a*b/v for a,b,v in zip(left,right,variances)) for right in rows] for left in rows]
    targets=[0,0,inputs["required_signal_exposure"]];dual=solve_linear(gram,targets)
    weights=[sum(rows[j][i]*dual[j] for j in range(3))/v for i,v in enumerate(variances)]
    return {"weights":weights,"variance":sum(w*w*v for w,v in zip(weights,variances)),"gross":sum(map(abs,weights))}, {
        "constraint_residuals":[sum(w*a for w,a in zip(weights,row))-target for row,target in zip(rows,targets)],
        "stationarity_residuals":[2*v*w-2*sum(rows[j][i]*dual[j] for j in range(3)) for i,(v,w) in enumerate(zip(variances,weights))],"covariance_diagonal_positive":all(v>0 for v in variances)}


def q014(inputs):
    initial=inputs["initial_weight"]
    menu=proportional_choices(initial,inputs["alpha"],inputs["risk_coefficient"],inputs["linear_cost"])
    hold=menu[initial];trade=max((w for w in menu if w!=initial),key=menu.get);threshold=menu[trade]-hold
    def optimize(fixed):return trade if threshold>fixed else initial
    base=optimize(inputs["fixed_cost"])
    return {"base_weight":base,"higher_cost_weight":optimize(inputs["higher_fixed_cost"]),"incremental_utility":max(0,threshold-inputs["fixed_cost"]),"indifference_fixed_cost":threshold}, {
        "best_nonzero_trade_before_ticket":trade,"hold_utility":hold,"best_trade_utility_before_ticket":menu[trade],"tie_prefers_no_trade":optimize(threshold)==initial}


def q015(inputs):
    variances=inputs["diagonal_covariance"];means=inputs["mean_returns"];risk=inputs["risk_aversion"];radius=inputs["ellipsoid_radius"]
    norm=math.sqrt(sum(mu*mu/v for mu,v in zip(means,variances)))
    scale=max(0,(norm-radius)/risk)
    scale=min(scale,inputs["volatility_cap"])
    weights=[scale*mu/v/norm for mu,v in zip(means,variances)]
    if sum(weights)>1+1e-12:raise ValueError("Certificate does not cover a binding total-risky-weight constraint")
    utility=scale*(norm-radius)-risk*scale*scale/2
    gradient=[mu-(risk+radius/scale)*v*w if scale else 0.0 for mu,v,w in zip(means,variances,weights)]
    return {"weights":weights,"robust_utility":utility,"cash_threshold_radius":norm,"larger_radius_all_cash":inputs["larger_radius"]>=norm}, {
        "risky_weight_sum":sum(weights),"volatility":scale,"unconstrained_stationarity_residuals":gradient,
        "global_bound":"Cauchy-Schwarz in whitened risk coordinates bounds mean by norm times volatility; these positive weights attain that bound and satisfy funding."}


def q016(inputs):
    results=[];objectives=[];vols=inputs["annual_volatilities"]
    for correlation in inputs["completion_correlations"]:
        cross=correlation*math.prod(vols);covariance=[[vols[0]**2,cross],[cross,vols[1]**2]]
        objective=lambda w:inputs["risk_aversion"]/2*variance([w,1-w],covariance)-sum(z*mean for z,mean in zip((w,1-w),inputs["mean_returns"]))
        weight=minimize_interval(objective,0,1);results.append(weight);objectives.append(-objective(weight))
    same=math.isclose(results[0],results[1],rel_tol=0,abs_tol=1e-8)
    return {"identification":"identified" if same else "not_identified","completion_a_weights":results,"same_decision":same}, {"completion_optimal_utilities":objectives,"scope":"The supplied distinct optimizers disprove identification; agreement at two points would not by itself prove agreement over a general interval."}


def q022(inputs):
    endpoint=1+inputs["cheap_borrowing_capacity"];risk=inputs["quadratic_risk_coefficient"]
    low,high=inputs["low_funding_rate"],inputs["high_funding_rate"]
    def utility(leverage,alpha):
        debt=max(leverage-1,0);funding=min(debt,inputs["cheap_borrowing_capacity"])*low+max(0,debt-inputs["cheap_borrowing_capacity"])*high
        return alpha*leverage-risk*leverage*leverage/2-funding
    def optimize(alpha):
        candidates={0,1,endpoint,inputs["leverage_cap"]}
        for lower,upper,rate in ((0,1,0),(1,endpoint,low),(endpoint,inputs["leverage_cap"],high)):
            point=(alpha-rate)/risk
            if lower<=point<=upper:candidates.add(point)
        feasible=[x for x in candidates if 0<=x<=inputs["leverage_cap"]]
        return max(feasible,key=lambda x:(utility(x,alpha),-x))
    base=optimize(inputs["base_expected_excess_return"]);updated=optimize(inputs["updated_expected_excess_return"])
    threshold=risk*endpoint+high
    return {"base_leverage":base,"updated_leverage":updated,"base_utility":utility(base,inputs["base_expected_excess_return"]),"high_tier_entry_alpha":threshold}, {"cheap_tier_endpoint":endpoint,"optimum_at_entry_threshold":optimize(threshold)}


def q024(inputs):
    covariance=inputs["covariance"];budgets=inputs["risk_budgets"];x=[1.0]*len(budgets)
    for iteration in range(10000):
        previous=x[:]
        for i,budget in enumerate(budgets):
            cross=sum(covariance[i][j]*x[j] for j in range(len(x)) if j!=i)
            x[i]=2*budget/(cross+math.sqrt(cross*cross+4*covariance[i][i]*budget))
        if max(abs(a-b) for a,b in zip(x,previous))<1e-14:break
    weights=[value/sum(x) for value in x];vol=math.sqrt(variance(weights,covariance))
    margin=sum(w*m for w,m in zip(weights,inputs["margin_fractions"]));leverage=inputs["margin_cap"]/margin
    contributions=[w*sum(c*z for c,z in zip(row,weights))/vol**2 for w,row in zip(weights,covariance)]
    return {"weights":weights,"max_leverage":leverage,"attainable_volatility":min(inputs["target_volatility"],leverage*vol),"target_reachable":leverage*vol>=inputs["target_volatility"]}, {
        "normalized_risk_contributions":contributions,"risk_budget_residuals":[a-b for a,b in zip(contributions,budgets)],"coordinate_iterations":iteration+1,"composition_volatility":vol,"margin_at_max_leverage":leverage*margin,
        "convex_certificate":"Positive x minimizes x' Sigma x/2 minus sum(b_i log x_i); its stationarity fixes each Euler contribution before normalization."}


def q025(inputs):
    benchmark=inputs["benchmark"];alpha=inputs["expected_alpha"];cap=inputs["asset_a_cap"];target=inputs["alpha_target"]
    extreme=max(alpha[1:]);maximum=max(extreme,cap*alpha[0]+(1-cap)*extreme)
    if target>=maximum:raise ValueError("Finite dual certificate requires a feasible target below the maximum alpha")
    def tilted(multiplier):
        log_values=[math.log(b)+multiplier*a for b,a in zip(benchmark,alpha)];offset=max(log_values)
        unscaled=[math.exp(value-offset) for value in log_values];weights=[value/sum(unscaled) for value in unscaled]
        if weights[0]>cap:
            remaining=sum(unscaled[1:]);weights=[cap]+[(1-cap)*value/remaining for value in unscaled[1:]]
        return weights
    expected=lambda weights:sum(w*a for w,a in zip(weights,alpha))
    low,high=0.0,1.0
    while expected(tilted(high))<target:high*=2
    for _ in range(120):
        mid=(low+high)/2
        if expected(tilted(mid))<target:low=mid
        else:high=mid
    multiplier=(low+high)/2 if expected(tilted(0))<target else 0.0;weights=tilted(multiplier)
    cap_multiplier=max(0,multiplier*(alpha[0]-alpha[2])-math.log((weights[0]/benchmark[0])/(weights[2]/benchmark[2]))) if abs(weights[0]-cap)<1e-12 else 0.0
    gradient=[math.log(w/b)+1-multiplier*a+(cap_multiplier if i==0 else 0) for i,(w,b,a) in enumerate(zip(weights,benchmark,alpha))]
    return {"weights":weights,"relative_entropy":sum(w*math.log(w/b) for w,b in zip(weights,benchmark)),"alpha_multiplier":multiplier,"cap_multiplier":cap_multiplier,"higher_target_feasible":inputs["higher_alpha_target"]<=maximum}, {
        "expected_alpha":expected(weights),"stationarity_differences":[value-gradient[-1] for value in gradient],"maximum_feasible_alpha":maximum,"alpha_complementarity":multiplier*(target-expected(weights)),"cap_complementarity":cap_multiplier*(weights[0]-cap)}


CHECKS={
    "q001":(q001,"Rational long, short, segregated-collateral and unsecured-borrowing ledger without reuse of restricted proceeds."),
    "q003":(q003,"Solve the hedge loading system and verify both residual exposures independently of signed/gross notionals."),
    "q004":(q004,"Bracket equal Euler contributions, impose the weight cap, and reconcile contribution sums to variance."),
    "q005":(q005,"Decompose active return into common market and orthogonal idiosyncratic components before applying the futures overlay."),
    "q007":(q007,"Enumerate every independent mean-box vertex for every permitted portfolio, including the short-position worst-case direction."),
    "q008":(q008,"Turn both-leg turnover into a feasible interval; optimize the concave utility and estimate the right shadow price with two meshes."),
    "q009":(q009,"Enumerate stationary points and endpoints on both sides of the proportional-cost kink; retain the original holding as a separate candidate."),
    "q010":(q010,"Reduce active covariance to the A-minus-B direction and intersect tracking-error, concentration and long-only intervals."),
    "q011":(q011,"Enumerate all bounded diagonal-quadratic active sets; recover the cap multiplier from free-coordinate stationarity."),
    "q012":(q012,"Solve the equality-constrained risk-space Gram system and verify primal neutrality, signal exposure and quadratic stationarity."),
    "q014":(q014,"Keep the discontinuous no-trade point separate from the best smooth trade; derive the fixed-charge indifference threshold."),
    "q015":(q015,"Use whitened-coordinate Cauchy-Schwarz to establish an attainable global bound, with explicit funding/volatility feasibility and cash threshold."),
    "q016":(q016,"Independently optimize two admissible covariance completions; different optimal portfolios witness nonidentification."),
    "q022":(q022,"Enumerate piecewise funding-tier stationary points and kink endpoints; compare fully debited utilities."),
    "q024":(q024,"Coordinate minimization of a strictly convex risk-budget potential, normalized Euler checks, then separate margin scaling."),
    "q025":(q025,"Solve the capped exponential-family dual by bisection and certify the convex KL optimum with primal/dual feasibility and complementarity."),
}
