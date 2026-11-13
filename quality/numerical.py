"""Independent calculations from supplied question inputs, without scorer imports.

Each function returns candidate answers and numerical certificates. Comparison
with stored reference answers is deliberately a separate step in inventory.py.
These checks establish calculations under the question's stipulated model;
they do not establish that the model is an adequate description of markets.
"""
from fractions import Fraction
import itertools
import math
from statistics import NormalDist


def fraction(value):
    return Fraction(str(value))


def variance(weights, covariance):
    return math.fsum(wi * wj * covariance[i][j]
                     for i, wi in enumerate(weights)
                     for j, wj in enumerate(weights))


def minimize_interval(function, lower, upper, iterations=160):
    """Derivative-free scalar minimization for the convex objectives used here."""
    ratio = (math.sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    for _ in range(iterations):
        if function(left) <= function(right):
            upper, right = right, left
            left = upper - ratio * (upper - lower)
        else:
            lower, left = left, right
            right = lower + ratio * (upper - lower)
    return min((lower, upper, (lower + upper) / 2), key=function)


def minimum_variance(covariance):
    objective = lambda w: variance([w, 1 - w], covariance)
    weight = minimize_interval(objective, 0.0, 1.0)
    gradient = 2 * (weight * (covariance[0][0] - covariance[0][1])
                    + (1 - weight) * (covariance[0][1] - covariance[1][1]))
    return weight, objective(weight), gradient


def tail_expectation(losses, probabilities, mass):
    """Use the convex CVaR threshold objective over all loss breakpoints."""
    losses = list(map(fraction, losses))
    probabilities = list(map(fraction, probabilities))
    mass = fraction(mass)
    if not 0 < mass <= 1 or sum(probabilities) != 1:
        raise ValueError("Invalid tail probability")
    if any(p < 0 for p in probabilities):
        raise ValueError("Negative scenario probability")
    value, threshold = min((z + sum(p * max(loss - z, 0)
                            for p, loss in zip(probabilities, losses)) / mass, z)
                           for z in set(losses))
    above = sum(p for p, loss in zip(probabilities, losses) if loss > threshold)
    boundary = mass - above
    tied = sum(p for p, loss in zip(probabilities, losses) if loss == threshold)
    selected = [p if loss > threshold else p * boundary / tied if loss == threshold else Fraction(0)
                for p, loss in zip(probabilities, losses)]
    assert sum(selected) == mass
    assert sum(p * loss for p, loss in zip(selected, losses)) / mass == value
    return value, threshold, selected


def q002(inputs):
    vols = inputs["annual_volatility"]
    covariance = [[vols[0] ** 2, math.prod(vols) * inputs["correlation"]],
                  [math.prod(vols) * inputs["correlation"], vols[1] ** 2]]
    w, value, gradient = minimum_variance(covariance)
    return {"weight_a": w, "variance": value,
            "volatility_reduction": w * vols[0] + (1-w) * vols[1] - math.sqrt(value)}, {
            "variance_gradient": gradient, "weights_sum": 1.0,
            "covariance_determinant": covariance[0][0]*covariance[1][1]-covariance[0][1]**2}


def q006(inputs):
    sample, target, a = (inputs[k] for k in ("sample_covariance", "diagonal_target", "target_weight"))
    estimate = [[(1-a)*sample[i][j]+a*target[i][j] for j in range(2)] for i in range(2)]
    w, estimated, gradient = minimum_variance(estimate)
    _, sample_min, _ = minimum_variance(sample)
    sample_value = variance([w, 1-w], sample)
    return {"weight_a": w, "estimation_variance": estimated, "sample_variance": sample_value,
            "sample_regret": sample_value-sample_min}, {
            "estimation_gradient": gradient, "sample_optimal_variance": sample_min,
            "nonnegative_regret": sample_value >= sample_min - 1e-14}


def q021(inputs):
    probabilities = list(map(fraction, inputs["probabilities"]))
    candidates = []
    for a in map(fraction, inputs["permitted_a_weights"]):
        returns = [a*fraction(x)+(1-a)*fraction(y) for x,y in zip(inputs["asset_a_returns"], inputs["asset_b_returns"])]
        mean = sum(p*r for p,r in zip(probabilities, returns))
        if mean >= fraction(inputs["minimum_expected_return"]):
            es, _, _ = tail_expectation([-r for r in returns], probabilities, inputs["tail_mass"])
            candidates.append((es, a, mean))
    es, a, mean = min(candidates)
    return {"weight_a": float(a), "expected_shortfall": float(es), "expected_return": float(mean)}, {
        "feasible_choices": len(candidates), "runner_up_tail_gap": float(sorted(candidates)[1][0]-es)}


def q026(inputs):
    losses, ps, alpha = inputs["losses"], inputs["probabilities"], fraction(inputs["alpha"])
    es, _, selected = tail_expectation(losses, ps, 1-alpha)
    cumulative = Fraction(0)
    for loss, p in sorted(zip(losses, map(fraction, ps))):
        cumulative += p
        if cumulative >= alpha:
            var = loss
            break
    boundary = sum(p for p, loss in zip(selected, losses) if loss == var)
    return {"var": var, "es": float(es), "boundary_mass": float(boundary)}, {
        "selected_probability_mass": float(sum(selected)), "es_at_least_var": es >= var}


def q027(inputs):
    sd, corr = inputs["desk_standard_deviations"], inputs["loss_correlation"]
    diagonal = sum(x*x for x in sd)
    cross = 2*sum(sd[i]*sd[j]*corr[i][j] for i in range(len(sd)) for j in range(i))
    vol = math.sqrt(diagonal+cross)
    return {"volatility": vol, "var95": inputs["normal_quantile"]*vol, "cross_variance": cross}, {
        "standalone_variance": diagonal, "portfolio_variance": diagonal+cross}


def q040(inputs):
    a, b = inputs["desk_a_losses"], inputs["desk_b_losses"]
    total = [x+y for x,y in zip(a,b)]
    mass = 1-fraction(inputs["alpha"])
    es, threshold, selected = tail_expectation(total, inputs["probabilities"], mass)
    allocations = [sum(p*fraction(loss) for p,loss in zip(selected, desk))/mass for desk in (a,b)]
    return {"portfolio_es": float(es), "attributions": list(map(float, allocations)),
            "selected_tie_mass": float(sum(p for p,loss in zip(selected,total) if loss == threshold))}, {
            "allocation_sum_equals_portfolio": sum(allocations) == es,
            "selected_scenario_probabilities": list(map(float, selected))}


def q051(inputs):
    pv_strike = inputs["strike"]*math.exp(-inputs["continuous_rate"]*inputs["expiry_years"])
    pv_dividend = inputs["cash_dividend"]*math.exp(-inputs["continuous_rate"]*inputs["dividend_time"])
    fair = inputs["put_quote"]+inputs["spot"]-pv_strike-pv_dividend
    cash = fair-inputs["call_quote"]
    package = "buy_call_sell_put_short_stock" if cash >= 0 else "sell_call_buy_put_long_stock"
    terminal_errors = []
    for terminal_spot in (0, inputs["strike"]/2, inputs["strike"], inputs["strike"]*2):
        terminal_errors.append(max(terminal_spot-inputs["strike"],0)-max(inputs["strike"]-terminal_spot,0)
                               -terminal_spot+inputs["strike"])
    return {"fair_call": fair, "initial_arbitrage_cash": abs(cash), "package": package}, {
        "terminal_replication_errors": terminal_errors, "prefunded_dividend": pv_dividend}


def call_price(inputs, vol):
    spot, strike, time = inputs["spot"], inputs["strike"], inputs["T"]
    r, dividend = inputs["continuous_rate"], inputs["continuous_dividend_yield"]
    d1 = (math.log(spot/strike)+(r-dividend+vol*vol/2)*time)/(vol*math.sqrt(time))
    d2 = d1-vol*math.sqrt(time)
    normal = NormalDist()
    return spot*math.exp(-dividend*time)*normal.cdf(d1)-strike*math.exp(-r*time)*normal.cdf(d2)


def q053(inputs):
    lower, upper = 1e-9, 10.0
    for _ in range(100):
        middle = (lower+upper)/2
        if call_price(inputs,middle) < inputs["call_quote"]:
            lower = middle
        else:
            upper = middle
    vol = (lower+upper)/2
    put = inputs["call_quote"]-inputs["spot"]*math.exp(-inputs["continuous_dividend_yield"]*inputs["T"])
    put += inputs["strike"]*math.exp(-inputs["continuous_rate"]*inputs["T"])
    return {"call_iv": vol, "consistent_put": put, "joint_fit_possible": math.isclose(put,inputs["put_quote"],abs_tol=1e-10)}, {
        "call_repricing_error": call_price(inputs,vol)-inputs["call_quote"],
        "put_parity_discrepancy": inputs["put_quote"]-put}


def q054(inputs):
    candidates = []
    initial, first, second = inputs["initial_greeks"], inputs["A"], inputs["B"]
    for a,b in itertools.product(range(13), repeat=2):
        delta = initial["delta"]+a*first["delta"]+b*second["delta"]
        stock = -delta/inputs["stock_delta"]
        cash = a*first["premium"]+b*second["premium"]+abs(stock)*inputs["stock_collateral_per_share"]
        if cash > inputs["cash_limit"]:
            continue
        gamma = initial["gamma"]+a*first["gamma"]+b*second["gamma"]
        vega = initial["vega"]+a*first["vega"]+b*second["vega"]
        losses = [-(gamma*spot*spot/2+vega*vol) for spot,vol in itertools.product(inputs["spot_moves"], inputs["vol_percentage_point_moves"])]
        candidates.append((max(0,*losses),cash,a,b,stock,gamma,vega))
    worst,cash,a,b,stock,gamma,vega = min(candidates)
    return {"contracts":[a,b],"stock_hedge":stock,"worst_loss":worst,"cash_used":cash,"residual_greeks":[gamma,vega]}, {
        "enumerated_choices":169,"feasible_choices":len(candidates),
        "residual_delta":initial["delta"]+a*first["delta"]+b*second["delta"]+stock*inputs["stock_delta"]}


def q057(inputs):
    if inputs["steps"] != 2:
        raise ValueError("This independent check enumerates exactly two steps")
    s,k,u,d,g = (inputs[x] for x in ("S0","K","up_factor","down_factor","cash_growth_per_step"))
    p = (g-d)/(u-d)
    put = lambda spot: max(k-spot,0)
    policies=[]
    for exercise_up,exercise_down in itertools.product((False,True),repeat=2):
        value=0
        for first,first_p,exercise in ((u,p,exercise_up),(d,1-p,exercise_down)):
            if exercise:
                value += first_p*put(s*first)/g
            else:
                value += sum(first_p*second_p*put(s*first*second)/(g*g)
                             for second,second_p in ((u,p),(d,1-p)))
        policies.append((value,exercise_up,exercise_down))
    european=next(x[0] for x in policies if x[1:] == (False,False))
    best=max(policies)
    american=max(put(s),best[0])
    return {"american_put":american,"exercise_premium":american-european,"exercise_first_down":best[2]}, {
        "stopping_policies_enumerated":4,"risk_neutral_probability":p,
        "dominates_immediate_and_european":american >= max(put(s),european)}


def q060(inputs):
    low,mid,high=inputs["terminal_states"]
    # Parameterize the martingale simplex by probability t of the high state.
    slope=(high-mid)/(mid-low)
    intercept=(mid-inputs["spot"])/(mid-low)
    lower=max(0.0,-intercept/slope)
    upper=min(1.0,(1-intercept)/(1+slope))
    weights=lambda t:[intercept+slope*t,1-intercept-(1+slope)*t,t]
    prices=lambda t:[sum(p*max(s-k,0) for p,s in zip(weights(t),inputs["terminal_states"])) for k in inputs["strikes"]]
    base,one=prices(0.0),prices(1.0)
    directions=[b-a for a,b in zip(base,one)]
    stationary=sum(d*(q-b) for d,q,b in zip(directions,inputs["quotes"],base))/sum(d*d for d in directions)
    t=min(upper,max(lower,stationary))
    repaired=prices(t)
    error=sum((p-q)**2 for p,q in zip(repaired,inputs["quotes"]))
    probabilities=weights(t)
    return {"probabilities":probabilities,"repaired_calls":repaired,"squared_error":error,"exact_fit":error < 1e-20}, {
        "probability_sum":sum(probabilities),"martingale_mean":sum(p*s for p,s in zip(probabilities,inputs["terminal_states"])),
        "feasible_parameter_interval":[lower,upper],"objective_second_derivative":2*sum(d*d for d in directions)}


def q076(inputs):
    sr=inputs["mean_monthly_excess_return"]/inputs["std_monthly_return"]
    variance=1+sr*sr/2
    se=math.sqrt(variance/inputs["months"])
    minimum=math.ceil(inputs["critical_t"]**2*variance/(sr*sr))
    statistic=lambda months:sr*math.sqrt(months/variance)
    return {"sharpe_annual":sr*math.sqrt(12),"se_sharpe_annual":se*math.sqrt(12),
            "t_stat":sr/se,"min_months":minimum}, {
            "minimum_months_statistic":statistic(minimum),"previous_month_statistic":statistic(minimum-1),
            "annualization_preserves_statistic":math.isclose(sr/se,sr*math.sqrt(12)/(se*math.sqrt(12)))}


def q078(inputs):
    y,x=inputs["strategy_excess_returns"],inputs["factor_excess_returns"]
    n=len(x); sx=math.fsum(x); sy=math.fsum(y)
    sxx=math.fsum(v*v for v in x); sxy=math.fsum(a*b for a,b in zip(x,y))
    determinant=n*sxx-sx*sx
    alpha=(sxx*sy-sx*sxy)/determinant
    beta=(n*sxy-sx*sy)/determinant
    residuals=[a-alpha-beta*b for a,b in zip(y,x)]
    residual_variance=math.fsum(v*v for v in residuals)/(n-2)
    intercept_se=math.sqrt(residual_variance*sxx/determinant)
    return {"beta":beta,"alpha_monthly_pct":100*alpha,"alpha_t_stat":alpha/intercept_se,
            "info_ratio_annual":alpha/math.sqrt(residual_variance)*math.sqrt(12)}, {
            "residual_sum":math.fsum(residuals),"factor_residual_inner_product":math.fsum(a*b for a,b in zip(x,residuals)),
            "residual_degrees_of_freedom":n-2}


def q079(inputs):
    rho=lambda lag:1.0 if lag==0 else inputs["autocorrelations"].get(f"lag{lag}",inputs["autocorrelations"]["beyond"])
    sd,mean,n,lag=(inputs[k] for k in ("sd_monthly_pct","mean_monthly_pct","months","bartlett_lag"))
    # The Bartlett factor is also variance of an L+1 block sum divided by its length.
    block=lag+1
    factor=sum(rho(abs(i-j)) for i in range(block) for j in range(block))/block
    naive=mean*math.sqrt(n)/sd
    annual_variance=sum(sd*sd*rho(abs(i-j)) for i in range(12) for j in range(12))
    return {"t_naive":naive,"t_hac":naive/math.sqrt(factor),"long_run_variance_factor":factor,
            "annual_vol_autocorr_pct":math.sqrt(annual_variance)}, {
            "annual_covariance_entries":144,"annual_sum_variance":annual_variance,"bartlett_block_length":block}


def expected_maximum(trials,gamma):
    normal=NormalDist()
    return (1-gamma)*normal.inv_cdf(1-1/trials)+gamma*normal.inv_cdf(1-1/(trials*math.e))


def q080(inputs):
    p=math.erfc(abs(inputs["best_t_stat"])/math.sqrt(2))
    corrected=-math.expm1(inputs["trials"]*math.log1p(-p))
    return {"expected_max_sharpe":inputs["sd_of_trial_sharpes"]*expected_maximum(inputs["trials"],inputs["euler_mascheroni"]),
            "p_single":p,"p_sidak":corrected,"significant_fwer_5pct":corrected < inputs["fwer_level"]}, {
            "family_p_at_least_single":corrected>=p,"family_p_at_most_bonferroni":corrected<=min(1,inputs["trials"]*p)}


def q083(inputs):
    start,end=inputs["test_fold_indices"]; horizon=inputs["label_horizon_days"]
    test=set(range(start,end+1)); training=set(range(inputs["n_obs"]))-test
    test_days=set().union(*(set(range(i,i+horizon+1)) for i in test))
    purged={i for i in training if test_days.intersection(range(i,i+horizon+1))}
    last=max(i for i in purged if i>end)
    embargo=set(range(last+1,last+1+inputs["embargo_obs"])) & training
    removed=purged|embargo
    return {"n_train":len(training-removed),"n_purged_before_test":len([i for i in purged if i<start]),
            "n_removed_after_test":len([i for i in removed if i>end]),"leak_fraction_unpurged":len(purged)/len(training)}, {
            "last_purged_after_test":last,"embargo_indices":sorted(embargo),"no_overlap_after_removal":all(not test_days.intersection(range(i,i+horizon+1)) for i in training-removed)}


def q084(inputs):
    prices,lookback=inputs["closes"],inputs["signal_lookback_days"]
    signal=lambda t:(prices[t]>prices[t-lookback])-(prices[t]<prices[t-lookback])
    correct=lookahead=1.0
    for end in range(lookback+1,len(prices)):
        ret=prices[end]/prices[end-1]-1
        correct*=1+signal(end-1)*ret
        lookahead*=1+signal(end)*ret
    return {"correct_cum_return_pct":100*(correct-1),"lookahead_cum_return_pct":100*(lookahead-1),
            "inflation_pct_points":100*(lookahead-correct)}, {
            "holding_periods":len(prices)-lookback-1,"first_correct_signal_day":lookback,"last_return_end_day":len(prices)-1}


def q085(inputs):
    impact=lambda order:inputs["impact_coefficient_Y"]*inputs["daily_vol"]*math.sqrt(order/inputs["adv_shares"])
    dollars=lambda order:order*inputs["price_usd"]*impact(order)
    quantity=inputs["order_shares"]; epsilon=1e-4
    elasticity=(math.log(dollars(quantity*math.exp(epsilon)))-math.log(dollars(quantity*math.exp(-epsilon))))/(2*epsilon)
    target=inputs["impact_budget_bps"]/10000
    lower,upper=0.0,inputs["adv_shares"]
    while impact(upper)<target: upper*=2
    for _ in range(100):
        middle=(lower+upper)/2
        if impact(middle)<=target: lower=middle
        else: upper=middle
    return {"impact_bps":10000*impact(quantity),"total_cost_usd":dollars(quantity)+quantity*inputs["price_usd"]*inputs["half_spread_bps"]/10000,
            "impact_dollar_elasticity":elasticity,"q_max_shares":lower}, {
            "impact_at_maximum_bps":10000*impact(lower),"impact_above_maximum_bps":10000*impact(lower+1)}


def q087(inputs):
    sr=inputs["sharpe_monthly"]
    benchmark=math.sqrt(inputs["variance_of_trial_sharpes"])*expected_maximum(inputs["trials"],inputs["euler_mascheroni"])
    scale=math.sqrt((inputs["months"]-1)/(1-inputs["skewness"]*sr+(inputs["kurtosis_raw"]-1)*sr*sr/4))
    psr=lambda baseline:NormalDist().cdf((sr-baseline)*scale)
    return {"sr0_benchmark":benchmark,"dsr_probability":psr(benchmark),"psr_undeflated":psr(0),"reject_null_95":psr(benchmark)>=0.95}, {
        "deflation_does_not_increase_probability":psr(benchmark)<=psr(0),"monthly_sharpe_used":sr}


def q088(inputs):
    ps=sorted(map(fraction,inputs["p_values"])); count=len(ps)
    q,alpha=fraction(inputs["fdr_q"]),fraction(inputs["fwer_alpha"])
    ranks=[i for i,p in enumerate(ps,1) if p<=i*q/count]
    bh=max(ranks,default=0)
    holm=0
    for i,p in enumerate(ps):
        if p>alpha/(count-i): break
        holm+=1
    bonferroni=sum(p<=alpha/count for p in ps)
    return {"n_bh_discoveries":bh,"bh_pvalue_cutoff":float(ps[bh-1]) if bh else 0.0,
            "n_bonferroni":bonferroni,"n_holm":holm}, {
            "holm_at_least_bonferroni":holm>=bonferroni,"bh_last_rank":bh,"exact_rational_thresholds":True}


CHECKS = {
    "q002": (q002,"Derivative-free convex minimization of the covariance quadratic; gradient and covariance determinant certificate."),
    "q006": (q006,"Separate derivative-free minimizations under estimation and sample matrices; nonnegative regret check."),
    "q021": (q021,"Enumerate every permitted portfolio with exact rational arithmetic; CVaR threshold objective over loss breakpoints."),
    "q026": (q026,"Exact rational quantile and CVaR threshold calculation; partial boundary mass conservation."),
    "q027": (q027,"Pairwise signed covariance aggregation with a separate standalone-variance decomposition."),
    "q040": (q040,"Exact rational CVaR and proportional tied-atom allocation; allocations must add to portfolio ES."),
    "q051": (q051,"Cash-flow replication with separately discounted strike and dividend; terminal payoff cancellation."),
    "q053": (q053,"Black-Scholes bisection and call repricing; matching put computed from parity independent of volatility."),
    "q054": (q054,"Complete enumeration of 169 integer hedge choices, cash feasibility and all five stress states."),
    "q057": (q057,"Enumerate all four first-step stopping policies and initial exercise; compare to European continuation."),
    "q060": (q060,"Reduce martingale probability simplex to one dimension; solve the constrained quadratic and verify probability/mean identities."),
    "q076": (q076,"Stipulated Sharpe standard error and annualization; verify integer minimum by bracketing the Wald threshold."),
    "q078": (q078,"Solve two-by-two OLS normal equations; independently verify both residual orthogonality identities."),
    "q079": (q079,"Construct covariance matrices of four- and twelve-month sums; derive Bartlett factor and exact annual variance by entry sums."),
    "q080": (q080,"Stipulated expected-maximum approximation; normal tail via erfc and numerically stable Sidak correction."),
    "q083": (q083,"Explicit inclusive label-day sets, overlap removal and post-purge embargo; verify no remaining intersection."),
    "q084": (q084,"Compound the same eight realized holding periods with lagged and same-period signals."),
    "q085": (q085,"Evaluate execution cash flows; estimate elasticity by log finite differences and invert impact budget by bisection."),
    "q087": (q087,"Stipulated non-normal probabilistic Sharpe formula in monthly units; compare deflated and undeflated probabilities."),
    "q088": (q088,"Exact rational multiple-testing thresholds with separate BH, Bonferroni and Holm rejection rules."),
}


def discrete_var(losses,probabilities,alpha):
    cumulative=Fraction(0)
    for loss,p in sorted(zip(losses,map(fraction,probabilities))):
        cumulative+=p
        if cumulative>=fraction(alpha):return loss
    raise ValueError("Distribution does not reach requested quantile")


def q029(inputs):
    p=fraction(inputs["default_probability"]); loss=inputs["default_loss"]
    losses=[0,loss,2*loss];probabilities=[(1-p)**2,2*p*(1-p),p*p]
    single=discrete_var([0,loss],[1-p,p],inputs["alpha"])
    total=discrete_var(losses,probabilities,inputs["alpha"])
    es,_,_=tail_expectation(losses,probabilities,1-fraction(inputs["alpha"]))
    return {"single_name_var":single,"portfolio_var":total,"portfolio_es":float(es),"var_subadditive_here":total<=2*single}, {
        "joint_probability_sum":float(sum(probabilities)),"double_default_probability":float(p*p)}


def q043(inputs):
    p,q=map(fraction,inputs["name_default_probabilities"])
    a,b=map(fraction,inputs["name_default_losses"])
    if a!=b:raise ValueError("This sharp-bound proof uses equal positive losses")
    lower,upper=max(Fraction(0),p+q-1),min(p,q)
    tail=1-fraction(inputs["alpha"])
    def es(joint):return tail_expectation([0,a,b,a+b],[1-p-q+joint,p-joint,q-joint,joint],tail)[0]
    lo,hi=es(lower),es(upper)
    # With equal losses, moving mass from one-default states to the two extreme
    # states is a mean-preserving spread. Upper-tail CVaR cannot decrease.
    independent,common=es(p*q),es(upper)
    limit=inputs["es_limit"]
    decision="certainly_met" if hi<=limit else "certainly_breached" if lo>limit else "not_identified"
    return {"sharp_es_bounds":[float(lo),float(hi)],"completion_es":[float(independent),float(common)],"limit_decision":decision}, {
        "feasible_joint_default_interval":[float(lower),float(upper)],"bound_basis":"Equal-loss Bernoulli coupling ordered by mean-preserving spread."}


def q047(inputs):
    exposures=list(map(fraction,inputs["exposures"]));lgd=fraction(inputs["lgd"])
    regimes=list(map(fraction,inputs["regime_probabilities"]));pds=list(map(fraction,inputs["conditional_name_pd"]))
    cap=fraction(inputs["detachment"])-fraction(inputs["attachment"])
    losses=[];probabilities=[];independent=[]
    marginal=sum(p*w for p,w in zip(pds,regimes))
    for bits in itertools.product((0,1),repeat=len(exposures)):
        loss=min(cap,max(Fraction(0),sum(e*b*lgd for e,b in zip(exposures,bits))-fraction(inputs["attachment"])))
        pdmass=lambda p:math.prod(p if bit else 1-p for bit in bits)
        losses.append(loss);probabilities.append(sum(w*pdmass(p) for w,p in zip(regimes,pds)));independent.append(pdmass(marginal))
    expected=sum(p*l for p,l in zip(probabilities,losses))
    independent_expected=sum(p*l for p,l in zip(independent,losses))
    es,_,_=tail_expectation(losses,probabilities,1-fraction(inputs["alpha"]))
    return {"expected_layer_loss":float(expected),"exhaustion_probability":float(sum(p for p,l in zip(probabilities,losses) if l==cap)),
            "layer_es95":float(es),"independent_expected_layer_loss":float(independent_expected)}, {
            "default_states_enumerated":len(losses),"probability_sum":float(sum(probabilities)),"matched_marginal_pd":float(marginal)}


def q050(inputs):
    vars=[];shortfalls=[];moments=[]
    for key in ("completion_a","completion_b"):
        data=inputs[key];losses=data["losses"];ps=data["probabilities"]
        vars.append(discrete_var(losses,ps,inputs["alpha"]))
        es,_,_=tail_expectation(losses,ps,1-fraction(inputs["alpha"]))
        shortfalls.append(float(es))
        mean=math.fsum(p*l for p,l in zip(ps,losses))
        moments.append([mean,math.fsum(p*(l-mean)**2 for p,l in zip(ps,losses))])
    disagreements=(shortfalls[0]<=inputs["es_limit"]) != (shortfalls[1]<=inputs["es_limit"])
    if not disagreements:raise ValueError("These completions do not establish nonidentification")
    return {"completion_vars":vars,"completion_es":shortfalls,"limit_identification":"not_identified"}, {
        "completion_mean_variance":moments,"two_completions_are_not_global_bounds":True}


def q058(inputs):
    paths=[inputs["path_A"],inputs["path_B"]]
    duration=inputs["times_years"][-1]-inputs["times_years"][0]
    variances=[math.fsum(math.log(b/a)**2 for a,b in zip(path,path[1:]))/duration for path in paths]
    payoffs=[max(path[-1]-inputs["terminal_call_strike"],0) for path in paths]
    return {"realized_variances":variances,"call_payoff_difference":abs(payoffs[0]-payoffs[1]),
            "identification":"not_identified" if variances[0]!=variances[1] else "identified"}, {
            "same_initial_and_terminal":paths[0][0]==paths[1][0] and paths[0][-1]==paths[1][-1],"observation_intervals":len(paths[0])-1}


def q059(inputs):
    if inputs["steps"]!=2 or inputs["rate"]!=0:raise ValueError("Check is scoped to the stated two-step zero-rate tree")
    up,down=inputs["u"],inputs["d"];p=(1-down)/(up-down)
    outcomes=[]
    for moves in itertools.product((up,down),repeat=2):
        spot=inputs["S0"];knocked=False;probability=1.0
        for move in moves:
            spot*=move;knocked|=spot>=inputs["barrier"];probability*=p if move==up else 1-p
        payoff=0.0 if knocked else max(spot-inputs["K"],0)
        outcomes.append((moves,spot,probability,payoff))
    central=[row[3] for row in outcomes if math.isclose(row[1],inputs["S0"],abs_tol=1e-10)]
    spread=max(central)-min(central)
    return {"knockout_value":sum(row[2]*row[3] for row in outcomes),"same_terminal_payoff_range":spread,"terminal_only_replication":spread==0}, {
        "path_count":4,"probability_sum":sum(row[2] for row in outcomes),"same_terminal_payoffs":central}


def q071(inputs):
    probabilities=[];survival=1.0
    for pd in inputs["conditional_default_probabilities"]:
        probabilities.append(survival*pd);survival*=1-pd
    probabilities.append(survival)
    protection=annuity=0.0;periods=len(probabilities)-1
    for event,p in enumerate(probabilities):
        if event<periods:
            protection+=p*inputs["notional"]*(1-inputs["recovery_fraction"])*math.exp(-inputs["rate"]*inputs["default_times"][event])
        premium=sum(math.exp(-inputs["rate"]*time) for time in inputs["scheduled_premium_times"][:event])
        if event<periods:premium+=inputs["default_accrual"]*math.exp(-inputs["rate"]*inputs["default_times"][event])
        annuity+=p*premium
    spread=protection/(inputs["notional"]*annuity)
    return {"protection_pv":protection,"risky_annuity":annuity,"par_spread":spread*10000,
            "upfront":protection-inputs["notional"]*inputs["running_coupon"]*annuity}, {
            "default_or_survival_probability_sum":sum(probabilities),"event_count":len(probabilities),
            "par_contract_net_pv":protection-inputs["notional"]*spread*annuity}


def q074(inputs):
    p,q=map(fraction,inputs["marginal_default_probabilities"])
    lower,upper=max(Fraction(0),p+q-1),min(p,q)
    value=lambda joint:float((p+q-joint)*fraction(inputs["discount_factor"])*fraction(inputs["payout_once"])-fraction(inputs["upfront"]))
    lo,hi=value(upper),value(lower)
    decision="buy_for_all" if lo>0 else "reject_for_all" if hi<0 else "not_identified"
    return {"net_value_bounds":[lo,hi],"extremal_joint_defaults":[float(upper),float(lower)],"decision":decision}, {
        "event_probability_bounds":[float(p+q-upper),float(p+q-lower)],"extrema_attainable":True}


def q075(inputs):
    def call(strike):
        return call_price({"spot":inputs["S0"],"strike":strike,"T":inputs["T"],"continuous_rate":inputs["continuous_rate"],"continuous_dividend_yield":0},inputs["volatility"])
    narrow,wide=inputs["widths"];cash=inputs["digital_cash"];strike=inputs["K"]
    spread=lambda width:cash/(2*width)*(call(strike-width)-call(strike+width))
    premium=(4*spread(narrow)-spread(wide))/3
    def payoff(spot,width):return cash/(2*width)*(max(spot-strike+width,0)-max(spot-strike-width,0))
    payoffs=[(4*payoff(spot,narrow)-payoff(spot,wide))/3 for spot in inputs["terminal_spot_menu"]]
    return {"extrapolated_price":premium,"minimum_payoff":min(payoffs),"maximum_payoff":max(payoffs),"bounded_on_menu":all(0<=p<=cash for p in payoffs)}, {
        "menu_payoffs":payoffs,"scope":"Only the expressly supplied terminal spot menu."}


def q097(inputs):
    n=inputs["population_size"];counts=inputs["archived_counts"];returns=inputs["archived_annual_returns"]
    weighted_counts=[fraction(c)/fraction(p) for c,p in zip(counts,inputs["inclusion_probabilities"])]
    total=sum(c*fraction(r) for c,r in zip(weighted_counts,returns))
    archive=sum(fraction(c)*fraction(r) for c,r in zip(counts,returns))/sum(counts)
    ht=total/n;hajek=total/sum(weighted_counts);net=ht-fraction(inputs["additional_cost_decimal"])
    return {"archive_mean":float(archive),"horvitz_thompson_mean":float(ht),"hajek_mean":float(hajek),"cost_adjusted_ht":float(net),"sign_reversal":archive*net<0}, {
        "known_population_size":n,"estimated_population_size":float(sum(weighted_counts)),"inverse_probability_total":float(total)}


def q098(inputs):
    null,alternative,alpha=map(fraction,[inputs["null_success_probability"],inputs["alternative_success_probability"],inputs["alpha"]])
    boundary=1/alpha;ratio=Fraction(1);likelihood_stop=naive_stop=0;at_stop=None;successes=0
    for day,bit in enumerate(inputs["observations"],1):
        successes+=bit;ratio*=alternative/null if bit else (1-alternative)/(1-null)
        if not likelihood_stop and ratio>=boundary:likelihood_stop=day;at_stop=ratio
        tail=sum(Fraction(math.comb(day,k))*null**k*(1-null)**(day-k) for k in range(successes,day+1))
        if not naive_stop and tail<=alpha:naive_stop=day
    if at_stop is None:at_stop=ratio
    # Absorbing dynamic program keeps only paths that have not previously promoted.
    surviving={0:Fraction(1)};absorbed=Fraction(0)
    for day in range(1,inputs["horizon_days"]+1):
        next_states={}
        for count,mass in surviving.items():
            for bit,p in ((0,1-null),(1,null)):
                k=count+bit
                likelihood=(alternative/null)**k*((1-alternative)/(1-null))**(day-k)
                if likelihood>=boundary:absorbed+=mass*p
                else:next_states[k]=next_states.get(k,Fraction(0))+mass*p
        surviving=next_states
    return {"likelihood_stop_day":likelihood_stop,"likelihood_at_stop":float(at_stop),"naive_fixed_test_stop_day":naive_stop,
            "finite_horizon_false_promotion":float(absorbed),"anytime_false_promotion_bound":float(alpha)}, {
            "probability_conservation":float(absorbed+sum(surviving.values())),"exact_false_promotion_fraction":str(absorbed),"bound_satisfied":absorbed<=alpha}


def q100(inputs):
    cash=fraction(inputs["initial_nav"]);risky=Fraction(0);turnover=interest=fees=Fraction(0);leverages=[];ledger=[]
    for forecast,ret in zip(inputs["forecasts_available_before_period"],inputs["risky_simple_returns"]):
        nav=cash+risky
        leverage=min(fraction(inputs["leverage_cap"]),fraction(inputs["annual_volatility_target"])/fraction(forecast))
        target=nav*leverage;trade=target-risky;fee=abs(trade)*fraction(inputs["fee_per_dollar_traded"])
        cash-=trade+fee;risky=target;turnover+=abs(trade);fees+=fee
        period_interest=cash*fraction(inputs["lending_rate_per_period"] if cash>=0 else inputs["borrowing_rate_per_period"])
        interest+=period_interest;cash+=period_interest;risky*=1+fraction(ret)
        leverages.append(float(leverage));ledger.append({"pretrade_nav":float(nav),"trade":float(trade),"fee":float(fee),"interest":float(period_interest),"closing_nav":float(cash+risky)})
    exit_fee=abs(risky)*fraction(inputs["fee_per_dollar_traded"]);turnover+=abs(risky);fees+=exit_fee;cash+=risky-exit_fee
    return {"opening_leverages":leverages,"total_turnover":float(turnover),"cash_interest":float(interest),"final_nav":float(cash),"net_simple_return":float(cash/fraction(inputs["initial_nav"])-1)}, {
        "period_ledger":ledger,"total_fees":float(fees),"exit_fee":float(exit_fee),"uses_future_realized_volatility":False}


CHECKS.update({
    "q029":(q029,"Exact Bernoulli convolution with quantile and partial-atom ES; compare portfolio VaR with two standalone VaRs."),
    "q043":(q043,"Frechet coupling interval and equal-loss mean-preserving-spread ordering establish sharp ES endpoints."),
    "q047":(q047,"Enumerate all eight default subsets under a common regime and a matched-marginal independent completion; exact rational probabilities."),
    "q050":(q050,"Exact-probability quantiles, tail averages and separate moment checks for two contrasting admissible completions."),
    "q058":(q058,"Direct pathwise log-increment quadratic variation and terminal call comparison demonstrate nonidentification."),
    "q059":(q059,"Enumerate all four monitored tree paths; retain path history despite terminal-state recombination."),
    "q071":(q071,"Enumerate each mutually exclusive default year and full survival; sum scenario coupon, accrual and protection cash flows."),
    "q074":(q074,"Frechet joint-default bounds and union probability establish attainable extremal contract values."),
    "q075":(q075,"Price and enumerate payoff of the actual Richardson call portfolio; explicitly inspect overshoot on the stated spot menu."),
    "q097":(q097,"Exact inverse-inclusion totals with separate known and estimated population denominators; apply the population-wide cost afterward."),
    "q098":(q098,"Exact rational path likelihoods and an absorbing probability dynamic program for first boundary crossing; no terminal-only approximation."),
    "q100":(q100,"Exact rational signed cash-and-risky-asset ledger, pre-fee targets, drift, asymmetric financing and final liquidation."),
})


def q045(inputs):
    a,b=inputs["corr_ab"],inputs["corr_ac"]
    residual=math.sqrt((1-a*a)*(1-b*b))
    correlations=[a*b-residual,a*b+residual]
    variances=[];gram_matrices=[]
    for rho in correlations:
        corr=[[1,a,b],[a,1,rho],[b,rho,1]]
        covariance=[[corr[i][j]*math.sqrt(inputs["variances"][i]*inputs["variances"][j]) for j in range(3)] for i in range(3)]
        variances.append(variance(inputs["portfolio_loadings"],covariance));gram_matrices.append(corr)
    bounds=[math.sqrt(max(0,value)) for value in sorted(variances)]
    limit=inputs["volatility_limit"]
    decision="certainly_met" if bounds[1]<=limit else "certainly_breached" if bounds[0]>limit else "not_identified"
    return {"correlation_bounds":correlations,"volatility_bounds":bounds,"limit_decision":decision}, {
        "endpoint_gram_matrices":gram_matrices,"endpoint_determinants":[1+2*a*b*c-a*a-b*b-c*c for c in correlations],
        "construction":"Project B and C on unit A. Their orthogonal residuals attain correlation minus/plus one, so both endpoints are feasible."}


def derivative(function, point, step):
    coarse=(function(point+step)-function(point-step))/(2*step)
    fine=(function(point+step/2)-function(point-step/2))/step
    return (4*fine-coarse)/3,abs(fine-coarse)


def q055(inputs):
    def price(spot,vol):
        return call_price({"spot":spot,"strike":inputs["K"],"T":inputs["T"],"continuous_rate":inputs["r"],"continuous_dividend_yield":inputs["q"]},vol)
    spot,vol=inputs["S0"],inputs["sigma0"]
    smile=lambda s:vol+inputs["eta"]*math.log(s/spot)
    frozen,frozen_error=derivative(lambda s:price(s,vol),spot,spot*1e-4)
    total,total_error=derivative(lambda s:price(s,smile(s)),spot,spot*1e-4)
    new=inputs["new_spot"];pnl=price(new,smile(new))-price(spot,vol)-total*(new-spot)
    return {"frozen_delta":frozen,"smile_delta":total,"new_volatility":smile(new),"hedged_pnl":pnl}, {
        "derivative_mesh_changes":[frozen_error,total_error],"hedge_shares":-total,"elapsed_time":0}


def normal_integral(function, lower=-12.0, upper=12.0, intervals=2400):
    """Composite Simpson expectation; payoff kink is an integration endpoint."""
    if lower>=upper:return 0.0
    step=(upper-lower)/intervals
    density=NormalDist().pdf
    values=[function(lower+i*step)*density(lower+i*step) for i in range(intervals+1)]
    return step/3*(values[0]+values[-1]+4*math.fsum(values[1:-1:2])+2*math.fsum(values[2:-1:2]))


def lognormal_call_expectation(forward,strike,log_sd):
    threshold=(math.log(strike/forward)+log_sd**2/2)/log_sd
    return normal_integral(lambda z:forward*math.exp(log_sd*z-log_sd**2/2)-strike,max(threshold,-12.0))


def q065(inputs):
    forward,strike=inputs["forward"],inputs["strike"]
    sd=inputs["normal_volatility"]*math.sqrt(inputs["T"])
    scale=inputs["notional"]*inputs["accrual"]*inputs["discount_factor"]
    normal=scale*normal_integral(lambda z:forward+sd*z-strike,max((strike-forward)/sd,-12.0))
    shifted=scale*lognormal_call_expectation(forward+inputs["shift"],strike+inputs["shift"],inputs["shifted_black_volatility"]*math.sqrt(inputs["T"]))
    pv01=scale*NormalDist().cdf((forward-strike)/sd)*1e-4
    return {"normal_price":normal,"shifted_black_price":shifted,"normal_forward_pv01":pv01,
            "unshifted_black_admissible":forward>0 and strike>0}, {
        "pricing_method":"Integrate positive terminal payoff under the stipulated normal and shifted-lognormal distributions.",
        "normal_exercise_probability":pv01/(scale*1e-4),"normal_truncation_z":12}


def q066(inputs):
    probabilities,annuities,rates=(list(map(fraction,inputs[k])) for k in ("state_probabilities","time1_annuities","time1_par_rates"))
    mean=sum(p*r for p,r in zip(probabilities,rates));annuity=sum(p*a for p,a in zip(probabilities,annuities))
    pv=lambda k:sum(p*a*(k-r) for p,a,r in zip(probabilities,annuities,rates))*fraction(inputs["notional"])*fraction(inputs["discount_to_time1"])
    fair=-pv(0)/(pv(1)-pv(0))
    return {"fair_strike":float(fair),"arithmetic_rate":float(mean),"receiver_pv_at_arithmetic_rate":float(pv(mean))}, {
        "fair_strike_pv":float(pv(fair)),"annuity_weighted_probabilities":[float(p*a/annuity) for p,a in zip(probabilities,annuities)]}


def q067(inputs):
    probabilities,spots,growth=(list(map(fraction,inputs[k])) for k in ("probabilities","terminal_spot","state_cash_growth_1_to_2"))
    discounts=[1/g for g in growth]
    forward=sum(p*d*s for p,d,s in zip(probabilities,discounts,spots))/sum(p*d for p,d in zip(probabilities,discounts))
    future=sum(p*s for p,s in zip(probabilities,spots));wealth=[(s-future)*g for s,g in zip(spots,growth)]
    return {"forward":float(forward),"futures":float(future),"basis":float(future-forward),"expected_terminal_wealth":float(sum(p*w for p,w in zip(probabilities,wealth)))}, {
        "forward_time_zero_value":float(sum(p*d*(s-forward) for p,d,s in zip(probabilities,discounts,spots))),
        "discounted_futures_wealth":float(sum(p*d*w for p,d,w in zip(probabilities,discounts,wealth))),"terminal_wealth_by_state":list(map(float,wealth))}


def q069(inputs):
    def price(rho):
        drift=inputs["eur_rate"]-inputs["dividend_yield"]-rho*inputs["equity_vol"]*inputs["fx_vol"]
        forward=inputs["foreign_equity_spot"]*math.exp(drift*inputs["T"])
        payoff=lognormal_call_expectation(forward,inputs["foreign_strike"],inputs["equity_vol"]*math.sqrt(inputs["T"]))
        return inputs["fixed_usd_per_eur"]*math.exp(-inputs["usd_rate"]*inputs["T"])*payoff
    rho=inputs["correlation"];sensitivity,error=derivative(price,rho,1e-3)
    return {"quanto_price":price(rho),"zero_correlation_price":price(0),"correlation_derivative":sensitivity,"spot_fx_delta":0.0}, {
        "derivative_mesh_change":error,"conversion_is_contractually_fixed":True,"payoff_integration_z_limit":12}


def q070(inputs):
    eur,usd,accrual=map(fraction,[inputs["eur_notional"],inputs["usd_notional"],inputs["accrual"]])
    discounts=list(map(fraction,inputs["usd_discount_factors"]));forwards=list(map(fraction,inputs["usd_per_eur_forwards"]))
    received=[eur*accrual*fraction(inputs["eur_coupon"])]*len(discounts)
    paid=[usd*accrual*fraction(inputs["usd_coupon"])]*len(discounts)
    received[-1]+=eur;paid[-1]+=usd
    cashflows=[r*f-p for r,f,p in zip(received,forwards,paid)]
    pv=sum(d*c for d,c in zip(discounts,cashflows));annuity=sum(discounts)*usd*accrual
    basis=pv/annuity
    return {"pv":float(pv),"usd_basis":float(10000*basis),"eur_discount_1y":float(discounts[-1]*forwards[-1]/fraction(inputs["spot"]))}, {
        "converted_net_cashflows":list(map(float,cashflows)),"spread_adjusted_pv":float(pv-basis*annuity),"spread_sign":"A positive USD spread increases payments and reduces receiver PV."}


def q072(inputs):
    terminal_value=inputs["bond_price"]*math.exp(inputs["continuous_rate"]*inputs["T"])
    default=(inputs["solvent_cashflow"]-terminal_value)/(inputs["solvent_cashflow"]-inputs["default_cashflow"])
    if math.isclose(terminal_value,inputs["solvent_cashflow"],rel_tol=0,abs_tol=1e-12):default=0.0
    if not 0<=default<=1:raise ValueError("Inadmissible credit price")
    protection=inputs["cds_default_payment"]*default
    premium=protection/(inputs["face"]*(1-default))
    repriced=math.exp(-inputs["continuous_rate"]*inputs["T"])*((1-default)*inputs["solvent_cashflow"]+default*inputs["default_cashflow"])
    return {"default_probability":default,"cds_spread":premium*10000,"expected_principal_loss":default*(inputs["face"]-inputs["default_cashflow"])}, {
        "bond_repricing_error":repriced-inputs["bond_price"],"cds_undiscounted_leg_difference":protection-premium*inputs["face"]*(1-default)}


def q081(inputs):
    prior_var=inputs["prior_sd_pct"]**2;noise_var=inputs["forecast_noise_sd_pct"]**2
    precision=1/prior_var+1/noise_var;var=1/precision
    mean=var*(inputs["prior_mean_pct"]/prior_var+inputs["raw_forecast_pct"]/noise_var)
    sd=math.sqrt(var)
    return {"shrinkage_factor":mean/inputs["raw_forecast_pct"],"posterior_mean_pct":mean,"posterior_sd_pct":sd,"prob_positive":NormalDist().cdf(mean/sd)}, {
        "posterior_precision":precision,"quadratic_log_density_derivative_at_mean":mean*precision-inputs["prior_mean_pct"]/prior_var-inputs["raw_forecast_pct"]/noise_var}


def q086(inputs):
    prior=fraction(inputs["prior_high"]);informed=fraction(inputs["prob_informed"]);uninformed_buy=fraction(inputs["uninformed_buy_prob"])
    buy_high=informed+(1-informed)*uninformed_buy;buy_low=(1-informed)*uninformed_buy
    def posterior(buys,sells):
        high=prior*buy_high**buys*(1-buy_high)**sells
        low=(1-prior)*buy_low**buys*(1-buy_low)**sells
        return high/(high+low)
    def quote(prob):return prob*fraction(inputs["value_high"])+(1-prob)*fraction(inputs["value_low"])
    ask,bid=quote(posterior(1,0)),quote(posterior(0,1))
    return {"ask_initial":float(ask),"bid_initial":float(bid),"spread_initial":float(ask-bid),"ask_after_two_buys":float(quote(posterior(3,0)))}, {
        "posterior_after_one_two_three_buys":[float(posterior(i,0)) for i in (1,2,3)],"third_buy_is_conditioned":True}


def solve_linear(matrix, vector):
    rows=[list(map(float,row))+[float(value)] for row,value in zip(matrix,vector)];n=len(rows)
    for column in range(n):
        pivot=max(range(column,n),key=lambda i:abs(rows[i][column]));rows[column],rows[pivot]=rows[pivot],rows[column]
        if abs(rows[column][column])<1e-15:raise ValueError("Singular system")
        scale=rows[column][column];rows[column]=[value/scale for value in rows[column]]
        for i in range(n):
            if i!=column:
                scale=rows[i][column];rows[i]=[left-scale*right for left,right in zip(rows[i],rows[column])]
    return [row[-1] for row in rows]


def q089(inputs):
    observations=inputs["observations"];n=len(observations)+1;q,r=inputs["state_noise_var"],inputs["obs_noise_var"]
    # Construct the full joint Gaussian precision from the prior, state increments
    # and observations. Its final marginal equals the terminal filtered state.
    precision=[[0.0]*n for _ in range(n)];linear=[0.0]*n
    precision[0][0]=1/inputs["prior_var"];linear[0]=inputs["prior_mean"]/inputs["prior_var"]
    for i,y in enumerate(observations,1):
        precision[i-1][i-1]+=1/q;precision[i][i]+=1/q+1/r
        precision[i-1][i]-=1/q;precision[i][i-1]-=1/q;linear[i]+=y/r
    means=solve_linear(precision,linear);column=solve_linear(precision,[0.0]*(n-1)+[1.0])
    variance_ss=inputs["prior_var"]
    for _ in range(10000):
        prediction=variance_ss+q;updated=1/(1/prediction+1/r)
        if abs(updated-variance_ss)<1e-16:break
        variance_ss=updated
    gain=(variance_ss+q)/(variance_ss+q+r)
    residual=max(abs(math.fsum(a*b for a,b in zip(row,means))-target) for row,target in zip(precision,linear))
    return {"filtered_mean":means[-1],"filtered_variance":column[-1],"steady_state_gain":gain,"equivalent_ewma_half_life":math.log(.5)/math.log1p(-gain)}, {
        "joint_precision_normal_equation_residual":residual,"steady_variance_fixed_point_error":updated-variance_ss,"joint_state_count":n}


def q094(inputs):
    sample,target=inputs["training_error_covariance_bps_squared"],inputs["shrinkage_diagonal_target_bps_squared"]
    amount=inputs["target_weight"];fitted=[[(1-amount)*sample[i][j]+amount*target[i][j] for j in range(2)] for i in range(2)]
    objective=lambda w:variance([w,1-w],fitted)
    a=(objective(1)+objective(-1)-2*objective(0))/2;b=(objective(1)-objective(-1))/2
    weight=-b/(2*a);weights=[weight,1-weight]
    intercept=-sum(w*bias for w,bias in zip(weights,inputs["forecast_error_bias_bps"]))
    evaluated=variance(weights,inputs["evaluation_error_covariance_bps_squared"])
    comparison=inputs["evaluation_error_covariance_bps_squared"][1][1]
    return {"weights":weights,"intercept":intercept,"fitting_mse":objective(weight),"evaluation_mse":evaluated,"evaluation_choice":"combined" if evaluated<comparison else "forecast_b"}, {
        "unrestricted_stationarity_residual":2*a*weight+b,"fitting_curvature":2*a,"evaluation_benchmark_mse":comparison,"combined_error_mean":intercept+sum(w*bias for w,bias in zip(weights,inputs["forecast_error_bias_bps"]))}


def q095(inputs):
    observed=inputs["variance_observed_factor"];latent=observed-inputs["known_measurement_noise_variance"]
    cov=inputs["covariance_observed_factor_strategy"];total=inputs["variance_strategy_return"]
    beta=cov/latent;slope=cov/observed;alpha=inputs["mean_strategy_return"]-beta*inputs["mean_observed_factor"]
    residual=total-beta*beta*latent
    mse=lambda b:total+b*b*observed-2*b*cov
    return {"structural_beta":beta,"structural_alpha":alpha,"residual_variance":residual,"predictive_slope":slope,"mse_penalty_using_structural_slope":mse(beta)-mse(slope)}, {
        "latent_factor_variance":latent,"reconstructed_strategy_variance":beta*beta*latent+residual,"predictive_gradient":2*(slope*observed-cov),"penalty_square_identity":(beta-slope)**2*observed}


def q096(inputs):
    rw=inputs["random_walk_increment_variance"];x,y=inputs["residual_X"],inputs["residual_Y"]
    varx=rw+2*x["stationary_variance"]*(1-x["ar_coefficient"])
    vary=.25*rw+2*y["stationary_variance"]*(1-y["ar_coefficient"]);cov=.5*rw
    h=2;regression=cov/vary
    ux=inputs["current_X"]-100-inputs["current_W"]
    vy=inputs["current_Y"]-50-.5*inputs["current_W"]
    expected=x["ar_coefficient"]*ux-h*y["ar_coefficient"]*vy
    current=inputs["current_X"]-h*inputs["current_Y"];count=inputs["short_X_shares"]
    gross=count*(current-expected)
    borrow=count*inputs["current_X"]*inputs["annual_stock_borrow_rate"]/inputs["days_per_year"]
    funding=count*h*inputs["current_Y"]*inputs["annual_long_funding_rate"]/inputs["days_per_year"]
    execution=count*(inputs["round_trip_cost_per_X_share"]+h*inputs["round_trip_cost_per_Y_share"])
    return {"cointegrating_hedge":h,"daily_return_correlation":cov/math.sqrt(varx*vary),"regression_hedge_trend_loading":1-.5*regression,"next_expected_spread":expected,"net_expected_pnl":gross-borrow-funding-execution}, {
        "increment_covariance":[[varx,cov],[cov,vary]],"current_residuals":[ux,vy],"gross_expected_pnl":gross,"costs":{"stock_borrow":borrow,"long_funding":funding,"execution":execution},"cointegrating_trend_loading":1-.5*h}


def q099(inputs):
    favorable,fill=map(fraction,[inputs["favorable_markout_probability"],inputs["fill_probability"]])
    bounds=[max(0,favorable+fill-1),min(favorable,fill)]
    good,bad,fee=map(fraction,[inputs["favorable_markout_bp"],inputs["adverse_markout_bp"],inputs["fee_per_filled_notional_bp"]])
    tables=[[j,fill-j,favorable-j,1-favorable-fill+j] for j in bounds]
    submitted=[row[0]*(good-fee)+row[1]*(bad-fee) for row in tables]
    filled=[value/fill for value in submitted]
    return {"filled_alpha_bounds":list(map(float,filled)),"submitted_alpha_bounds":list(map(float,submitted)),"joint_favorable_fill_bounds":list(map(float,bounds)),"independence_alpha":float(favorable*good+(1-favorable)*bad-fee),"alpha_sign":"positive_for_all" if min(filled)>0 else "negative_for_all" if max(filled)<0 else "not_identified"}, {
        "attaining_joint_tables":[list(map(float,row)) for row in tables],"joint_table_order":["filled_favorable","filled_adverse","unfilled_favorable","unfilled_adverse"],"all_masses_nonnegative":all(mass>=0 for row in tables for mass in row)}


CHECKS.update({
    "q045":(q045,"Gram-vector residual geometry gives sharp PSD correlation bounds and attainable portfolio-risk endpoints."),
    "q055":(q055,"Richardson central differences of frozen and smile-dependent call prices, followed by exact model repricing with the initial share hedge."),
    "q065":(q065,"Integrate normal and shifted-lognormal terminal payoff distributions directly; exercise probability gives the forward derivative."),
    "q066":(q066,"Solve the two-state discounted affine receiver cash-flow equation with rational arithmetic; separate annuity-weighted and ordinary probabilities."),
    "q067":(q067,"Price terminal forward cash flows with state discounting; accumulate earlier futures variation cash at state-specific growth rates."),
    "q069":(q069,"Integrate the stipulated foreign-equity terminal distribution, discount in payoff currency, and differentiate correlation numerically with mesh comparison."),
    "q070":(q070,"Rational coupon/principal ledger converts each payment with its own forward and solves the signed USD-spread cash-flow equation."),
    "q072":(q072,"Invert the two terminal bond payoff states, then balance survival-only CDS premiums against default protection."),
    "q081":(q081,"Complete the Gaussian posterior precision quadratic and integrate the positive half-line."),
    "q086":(q086,"Exact rational likelihood multiplication for sequential buys; the updated ask conditions on a third incoming buy."),
    "q089":(q089,"Solve the full joint Gaussian precision system for terminal mean/variance; independently iterate the steady-state variance fixed point."),
    "q094":(q094,"Fit the unrestricted scalar quadratic after intercept bias cancellation, then evaluate the frozen coefficients under the separate covariance regime."),
    "q095":(q095,"Reconstruct latent moments and residual variance, solve the noisy-proxy prediction normal equation, and compare full quadratic MSEs."),
    "q096":(q096,"Derive AR increment covariances and residual mean reversion, distinguish trend cancellation from regression hedging, and debit separate funding/cost legs."),
    "q099":(q099,"Construct both extremal nonnegative 2x2 fill/markout tables and compute submitted and conditional filled P&L separately."),
})
