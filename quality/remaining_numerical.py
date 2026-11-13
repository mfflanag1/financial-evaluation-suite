"""Independent finite policies, execution, rates and capacity calculations."""
from fractions import Fraction
import itertools
import math

from .numerical import derivative, fraction, solve_linear


def q013(inputs):
    candidates=[]
    for a,b in itertools.product(inputs['permitted_integer_units'],repeat=2):
        capital=3*a+2*b+2*abs(a-fraction(.8)*b)
        if a+b<=inputs['gross_unit_cap']:
            value=fraction(inputs['desk_a_profit_usd_m'][a])+fraction(inputs['desk_b_profit_usd_m'][b])-fraction(inputs['capital_charge_rate'])*capital
            candidates.append((value,a,b,capital))
    def choose(limit):return min((row for row in candidates if row[3]<=fraction(limit)),key=lambda row:(-row[0],row[1],row[2]))
    base=choose(inputs['capital_base_usd_m']);tight=choose(inputs['capital_tight_usd_m'])
    return {'base_units':list(base[1:3]),'tight_units':list(tight[1:3]),'base_objective':float(base[0]),'objective_loss':float(base[0]-tight[0])}, {
        'base_capital':float(base[3]),'tight_capital':float(tight[3]),'gross_feasible_candidates':len(candidates)}


def q017(inputs):
    candidates=[]
    for counts in itertools.product(inputs['permitted_sales_per_lot'],repeat=2):
        if any(count>available for count,available in zip(counts,inputs['shares_by_tax_lot'])):continue
        shares=sum(counts);tax=sum(count*max(0,fraction(inputs['sale_price_usd'])-fraction(basis))*fraction(inputs['tax_rate']) for count,basis in zip(counts,inputs['basis_per_share_usd']))
        execution=shares*fraction(inputs['per_share_execution_cost_usd'])+(fraction(inputs['fixed_sale_ticket_usd']) if shares else 0)
        cash=fraction(inputs['starting_cash_usd'])+shares*fraction(inputs['sale_price_usd'])-tax-execution
        if cash>=fraction(inputs['required_cash_usd']):candidates.append((shares,tax+execution,counts[0],counts,cash,tax,execution))
    _,cost,_,counts,cash,tax,execution=min(candidates,key=lambda row:row[:3])
    return {'sales':list(counts),'ending_cash':float(cash),'tax_and_cost':float(cost)}, {'tax':float(tax),'execution':float(execution),'feasible_plans':len(candidates),'single_order_ticket':True}


def rebalance(cash,risky,target,cost):
    nav=cash+risky;gap=target*nav-risky
    trade=gap/(1+target*cost) if gap>=0 else gap/(1-target*cost)
    fee=abs(trade)*cost
    next_cash,next_risky=cash-trade-fee,risky+trade
    if next_cash<0 or next_risky<0:raise ValueError('Rebalance violates no-borrowing contract')
    assert next_risky==target*(next_cash+next_risky)
    return next_cash,next_risky,fee


def q018(inputs):
    cost=fraction(inputs['one_way_trading_cost']);menu=list(map(fraction,inputs['permitted_risky_weights']));candidates=[]
    for policy in itertools.product(menu,repeat=3):
        opening_cash,opening_risky,entry_fee=rebalance(fraction(inputs['initial_wealth']),Fraction(0),policy[0],cost)
        wealth=[];probabilities=[];branch_fees=[]
        for branch,first_return in enumerate(inputs['first_period_returns']):
            cash,risky,fee=rebalance(opening_cash,opening_risky*(1+fraction(first_return)),policy[branch+1],cost)
            first_prob=fraction(inputs['initial_up_probability']) if branch==0 else 1-fraction(inputs['initial_up_probability'])
            branch_fees.append(fee)
            for terminal,second_return in enumerate(inputs['second_period_returns'][branch]):
                wealth.append(cash+risky*(1+fraction(second_return)))
                conditional=fraction(inputs['second_period_up_probability'][branch]);probabilities.append(first_prob*(conditional if terminal==0 else 1-conditional))
        if min(wealth)<fraction(inputs['terminal_wealth_floor']):continue
        value=math.fsum(float(prob)*math.log(float(nav)) for prob,nav in zip(probabilities,wealth))
        candidates.append((value,policy,wealth,probabilities,entry_fee,branch_fees))
    value,policy,wealth,probabilities,entry_fee,branch_fees=min(candidates,key=lambda row:(-row[0],row[1]))
    return {'policy':list(map(float,policy)),'expected_log_wealth':value,'worst_terminal_wealth':float(min(wealth))}, {
        'feasible_nonanticipating_policies':len(candidates),'terminal_wealth':list(map(float,wealth)),'terminal_probabilities':list(map(float,probabilities)),'entry_fee':float(entry_fee),'branch_fees':list(map(float,branch_fees)),'target_weight_denominator':'post-fee NAV'}


def q019(inputs):
    def choose(cross):
        candidates=[]
        for first,second in itertools.product(inputs['permitted_step_counts'],repeat=2):
            a,b=fraction(inputs['active_step'])*first,fraction(inputs['active_step'])*second;active=[a,-a,b,-b]
            if any(fraction(weight)+change<0 for weight,change in zip(inputs['benchmark'],active)):continue
            risk=fraction(inputs['pair_variances'][0])*a*a+fraction(inputs['pair_variances'][1])*b*b+2*fraction(cross)*a*b
            if risk>fraction(inputs['tracking_error_cap'])**2:continue
            objective=sum(fraction(alpha)*w for alpha,w in zip(inputs['alpha'],active))-fraction(inputs['linear_cost'])*sum(map(abs,active))
            candidates.append((objective,a,b,risk))
        return min(candidates,key=lambda row:(-row[0],row[1],row[2])),len(candidates)
    base,nbase=choose(inputs['base_pair_covariance']);stress,nstress=choose(inputs['stressed_pair_covariance'])
    return {'base_pair_weights':list(map(float,base[1:3])),'stressed_pair_weights':list(map(float,stress[1:3])),'net_objectives':[float(base[0]),float(stress[0])]}, {
        'selected_tracking_variances':[float(base[3]),float(stress[3])],'feasible_counts':[nbase,nstress],'sector_active_sums':[0,0]}


def q020(inputs):
    candidates=[]
    for units in itertools.product(inputs['permitted_bond_units'],repeat=len(inputs['bond_prices'])):
        cash=Fraction(0);balances=[]
        for date,liability in enumerate(inputs['liability_cashflows']):
            cash+=sum(unit*fraction(flows[date]) for unit,flows in zip(units,inputs['bond_cashflows']))-fraction(liability);balances.append(cash)
        if min(balances)<0:continue
        cost=sum(unit*fraction(price) for unit,price in zip(units,inputs['bond_prices']));candidates.append((cost,units,balances))
    cost,units,balances=min(candidates,key=lambda row:(row[0],row[1]))
    return {'bond_units':list(units),'purchase_cost':float(cost),'minimum_cash_balance':float(min(balances))}, {
        'end_of_date_cash_balances':list(map(float,balances)),'feasible_integer_ladders':len(candidates),'terminal_surplus_credited_against_purchase_cost':False}


def q023(inputs):
    gross_max=inputs['number_of_desks']*(inputs['each_desk_long_usd_m']+inputs['each_desk_short_usd_m'])
    gross_min=abs(inputs['number_of_desks']*(inputs['each_desk_long_usd_m']-inputs['each_desk_short_usd_m']))
    unconstrained=inputs['net_objective_linear_usd_m']/(2*inputs['net_objective_quadratic_usd_m'])
    scales=[min(unconstrained,inputs['capital_usd_m']/(gross*inputs['margin_fraction_of_netted_gross'])) for gross in (gross_min,gross_max)]
    return {'gross_bounds':[gross_min,gross_max],'completion_scales':scales,'robust_scale':scales[-1],'optimal_scale_identified':scales[0]==scales[1]}, {
        'unconstrained_objective_optimum':unconstrained,'minimum_gross_certificate':'Absolute total signed exposure bounds gross below; the supplied crossed-security completion attains it.','maximum_gross_certificate':'Triangle inequality bounds netted gross above by pre-net gross; four distinct securities attain it.'}


def q052(inputs):
    usd_discount=math.exp(-inputs['usd_rate']*inputs['expiry_years']);eur_discount=math.exp(-inputs['eur_rate']*inputs['expiry_years'])
    spot=inputs['usd_per_eur_spot'];forward=spot*eur_discount/usd_discount;locked=forward*inputs['eur_receivable'];present=locked*usd_discount
    return {'forward':forward,'locked_usd':locked,'present_value':present}, {'spot_converted_discounted_eur_receivable':spot*eur_discount*inputs['eur_receivable'],'two_currency_replication_residual':present-spot*eur_discount*inputs['eur_receivable']}


def q056(inputs):
    if len(inputs['permitted_shares'])!=2:raise ValueError('Reviewed input gives the inclusive integer-share bounds')
    candidates=[]
    for shares in range(inputs['permitted_shares'][0],inputs['permitted_shares'][1]+1):
        liabilities=[inputs['calls_owed']*max(0,fraction(spot)-fraction(inputs['strike'])) for spot in inputs['terminal_spots']]
        cash=max((liability-shares*fraction(bid))/fraction(inputs['cash_growth']) for liability,bid in zip(liabilities,inputs['terminal_stock_bids']))
        outlay=shares*fraction(inputs['initial_stock_ask'])+cash
        surplus=[shares*fraction(bid)+cash*fraction(inputs['cash_growth'])-liability for bid,liability in zip(inputs['terminal_stock_bids'],liabilities)]
        candidates.append((outlay,shares,cash,surplus))
    outlay,shares,cash,surplus=min(candidates,key=lambda row:(row[0],row[1]))
    return {'shares':shares,'cash_account':float(cash),'minimum_outlay':float(outlay)}, {'terminal_superhedge_surplus':list(map(float,surplus)),'integer_positions_searched':len(candidates),'negative_cash_is_funded_debt':cash<0}


def q061(inputs):
    known={.5:inputs['discount_0_5']};times=sorted({time for key in ('bond_A_cashflows','bond_B_cashflows') for time,_ in inputs[key]}-set(known));matrix=[];prices=[]
    for label in ('A','B'):
        cashflows=inputs[f'bond_{label}_cashflows'];matrix.append([sum(value for time,value in cashflows if time==target) for target in times])
        prices.append(inputs[f'bond_{label}_price']-sum(value*known[time] for time,value in cashflows if time in known))
    discounts=solve_linear(matrix,prices);forward=(discounts[0]/discounts[1]-1)/inputs['forward_accrual']
    all_discounts=known|dict(zip(times,discounts))
    residual=[sum(value*all_discounts[time] for time,value in inputs[f'bond_{label}_cashflows'])-inputs[f'bond_{label}_price'] for label in ('A','B')]
    return {'discounts':discounts,'forward_rate':forward}, {'cashflow_repricing_residuals':residual,'unknown_discount_dates':times}


def q062(inputs):
    n=inputs['notional'];accruals=inputs['accruals'];discounts=inputs['discount_factors'];forwards=inputs['projection_forwards'];fixed=inputs['fixed_rate'];bp=inputs['basis_point']
    payments=[n*accrual*(forward-fixed) for accrual,forward in zip(accruals,forwards)]
    pv=sum(cash*discount for cash,discount in zip(payments,discounts));annuity=sum(accrual*discount for accrual,discount in zip(accruals,discounts))
    par=sum(accrual*discount*forward for accrual,discount,forward in zip(accruals,discounts,forwards))/annuity
    projection=n*annuity*bp;discount_derivative=-sum(time*cash*discount for time,cash,discount in zip(inputs['payment_times'],payments,discounts))*bp
    numeric,mesh=derivative(lambda rate:sum(cash*discount*math.exp(-rate*time) for time,cash,discount in zip(inputs['payment_times'],payments,discounts)),0,1e-4)
    return {'pv':pv,'par_rate':par,'projection_pv01':projection,'discount_pv01':discount_derivative}, {'net_coupon_payments':payments,'discount_derivative_check':numeric*bp,'discount_derivative_mesh_change':mesh,'par_coupon_repricing':sum(n*accrual*(forward-par)*discount for accrual,forward,discount in zip(accruals,forwards,discounts))}


def q063(inputs):
    candidates=[]
    for units in itertools.product(inputs['unit_grid'],repeat=3):
        collateral=sum(unit*cost for unit,cost in zip(units,inputs['collateral_per_unit']))
        if collateral>inputs['cash_limit']:continue
        sensitivity=[base+sum(unit*row[j] for unit,row in zip(units,inputs['instrument_rows'])) for j,base in enumerate(inputs['base_sensitivity'])]
        losses=[-sum(delta*move for delta,move in zip(sensitivity,shock)) for shock in inputs['stress_moves_bp']]+[0]
        candidates.append((max(losses),collateral,units,sensitivity,losses))
    loss,collateral,units,sensitivity,losses=min(candidates,key=lambda row:(row[0],row[1],row[2]))
    return {'hedge_units':list(units),'worst_loss':loss,'collateral':collateral,'residual_sensitivity':sensitivity}, {'cash_feasible_grid_plans':len(candidates),'scenario_losses_including_zero':losses,'collateral_deducted_from_pnl':False}


def q064(inputs):
    def price(yield_rate):return sum(cash/(1+yield_rate)**time for cash,time in zip(inputs['annual_cashflows'],inputs['times_years']))
    y=inputs['initial_yield'];dy=inputs['new_yield']-y;initial=price(y)
    first,first_error=derivative(price,y,1e-3)
    second=lambda step:(price(y+step)-2*initial+price(y-step))/step**2
    coarse,fine=second(1e-3),second(5e-4);curvature=(4*fine-coarse)/3
    duration=first*dy;convexity=duration+curvature*dy**2/2
    return {'exact_pnl':price(inputs['new_yield'])-initial,'duration_pnl':duration,'convexity_pnl':convexity}, {'initial_price':initial,'first_derivative_mesh_change':first_error,'second_derivative_mesh_change':fine-coarse,'estimated_price_convexity':curvature}


def q068(inputs):
    initial=fraction(inputs['spot_per_barrel'])+fraction(inputs['prepaid_storage_per_barrel']);bound=float(initial)*math.exp(inputs['borrowing_rate']*inputs['T'])+inputs['terminal_delivery_fee']
    capacity=min(inputs['tank_capacity_barrels'],math.floor(fraction(inputs['initial_borrow_limit'])/initial))
    quantity=capacity if inputs['forward_bid']>bound else 0
    return {'carry_bound':bound,'quantity':quantity,'terminal_profit':quantity*(inputs['forward_bid']-bound)}, {'initial_borrowing':float(initial*quantity),'funded_capacity':capacity,'forward_is_executable_bid':True}


def q073(inputs):
    probabilities=list(map(fraction,inputs['state_probabilities']));defaults=list(map(fraction,inputs['conditional_default_probabilities']))
    exposures=list(map(fraction,inputs['positive_exposures']));threshold,cap=map(fraction,[inputs['collateral_threshold'],inputs['collateral_cap']])
    collateral=[min(max(value-threshold,0),cap) for value in exposures];residual=[value-secured for value,secured in zip(exposures,collateral)]
    factor=fraction(inputs['loss_given_default'])*fraction(inputs['discount_factor']);joint=sum(p*d*e for p,d,e in zip(probabilities,defaults,residual))*factor
    marginal_default=sum(p*d for p,d in zip(probabilities,defaults));mean_exposure=sum(p*e for p,e in zip(probabilities,residual));independent=marginal_default*mean_exposure*factor
    return {'residual_exposures':list(map(float,residual)),'joint_cva':float(joint),'independent_cva':float(independent),'understatement':float(joint-independent)}, {'collateral_by_state':list(map(float,collateral)),'marginal_default_probability':float(marginal_default),'mean_residual_exposure':float(mean_exposure)}


def q077(inputs):
    turnover=2*fraction(inputs['one_way_turnover_pct_of_nav'])/100
    cost_percent=turnover*fraction(inputs['cost_per_dollar_traded_bps'])/100
    net=fraction(inputs['gross_excess_return_pct'])-cost_percent-fraction(inputs['management_fee_pct'])
    breakeven=(fraction(inputs['gross_excess_return_pct'])-fraction(inputs['management_fee_pct']))*100/turnover
    return {'annual_cost_pct':float(cost_percent),'net_return_pct':float(net),'breakeven_cost_bps':float(breakeven),'net_sharpe':float(net/fraction(inputs['volatility_pct']))}, {'total_traded_notional_per_nav':float(turnover),'cost_and_fee_leave_volatility_unchanged':True}


def q082(inputs):
    phi=1+inputs['b'];mean=-inputs['a']/inputs['b'];variance=inputs['residual_sd']**2/(1-phi**2)
    forecast=inputs['current_spread']
    for _ in range(inputs['horizon_days']):forecast=inputs['a']+phi*forecast
    half=math.log(.5)/math.log(phi)
    return {'half_life_days':half,'long_run_mean':mean,'stationary_sd':math.sqrt(variance),'z_score':(inputs['current_spread']-mean)/math.sqrt(variance),'expected_spread_5d':forecast}, {'stationary_mean_residual':mean-inputs['a']-phi*mean,'stationary_variance_residual':variance-phi**2*variance-inputs['residual_sd']**2,'half_life_decay':phi**half}


def exp_square_integral(upper):
    terms=[];power=upper;factorial=1
    for k in range(1000):
        term=power/(factorial*(2*k+1));terms.append(term)
        if abs(term)<1e-16:break
        power*=upper*upper;factorial*=k+1
    return math.fsum(terms),len(terms),term


def q090(inputs):
    theta,sigma=inputs['theta_per_day'],inputs['sigma_per_sqrt_day'];results=[]
    for threshold in inputs['threshold_grid']:
        u=threshold*math.sqrt(theta)/sigma;integral,terms,last=exp_square_integral(u)
        cycle=math.sqrt(math.pi)/theta*integral
        results.append(((threshold-inputs['round_trip_cost_points'])/cycle,threshold,cycle,terms,last))
    rate,threshold,cycle,terms,last=max(results,key=lambda row:(row[0],-row[1]))
    return {'best_threshold':threshold,'best_profit_rate_per_day':rate,'cycle_days_at_best':cycle,'stationary_sd':sigma/math.sqrt(2*theta)}, {'grid_profit_rates':[row[0] for row in results],'positive_series_terms_at_best':terms,'last_series_term_at_best':last,'identity':'The stipulated entry and return integrands add to exp(x^2) because erf(x)+erfc(x)=1.'}


def q091(inputs):
    candidates=[]
    for schedule in itertools.product(*(range(maximum+1) for maximum in inputs['slice_maximum_lots'])):
        if sum(schedule)!=inputs['initial_inventory_lots']:continue
        temporary=sum(fraction(cost)*quantity**2 for cost,quantity in zip(inputs['temporary_cost_usd_per_lot_squared'],schedule))
        linear=fraction(inputs['linear_cost_usd_per_lot'])*sum(schedule);permanent=fraction(inputs['permanent_cost_coefficient_usd_per_lot_squared'])*sum(schedule)**2/2
        remaining=inputs['initial_inventory_lots'];interval_risk=[];inventory=[]
        for quantity,sd in zip(schedule,inputs['price_increment_sd_usd_per_share']):
            remaining-=quantity;inventory.append(remaining);interval_risk.append((remaining*inputs['shares_per_lot']*fraction(sd))**2)
        variance=sum(interval_risk);cost=temporary+linear+permanent;objective=cost+fraction(inputs['variance_penalty_per_usd'])*variance
        candidates.append((objective,schedule,cost,variance,inventory,interval_risk))
    objective,schedule,cost,variance,inventory,interval_risk=min(candidates,key=lambda row:(row[0],row[1]))
    return {'schedule':list(schedule),'expected_cost':float(cost),'cost_variance':float(variance),'objective':float(objective)}, {'complete_liquidation_schedules':len(candidates),'post_slice_inventory_bearing_risk':inventory,'interval_variances':list(map(float,interval_risk)),'price_intervals':len(interval_risk)}


def q092(inputs):
    offsets=inputs['permitted_quote_offsets_ticks'];probabilities=dict(zip(offsets,map(fraction,inputs['fill_probability_by_offset'])))
    tick=fraction(inputs['tick_usd']);mid=fraction(inputs['midprice_usd']);shares=inputs['shares_per_inventory_block'];adverse=fraction(inputs['adverse_mark_ticks']);penalty=fraction(inputs['terminal_inventory_penalty_usd_per_block_squared'])
    def optimize(initial):
        candidates=[]
        for bid,ask in itertools.product(offsets,repeat=2):
            outcomes=[];expected=Fraction(0);inventory=Fraction(0)
            for buy,sell in itertools.product((0,1),repeat=2):
                probability=(probabilities[bid] if buy else 1-probabilities[bid])*(probabilities[ask] if sell else 1-probabilities[ask])
                terminal_inventory=initial+buy-sell;terminal_mid=mid+tick*adverse*(sell-buy)
                cash=shares*(sell*(mid+ask*tick)-buy*(mid-bid*tick))
                value=cash+shares*(terminal_inventory*terminal_mid-initial*mid)-penalty*terminal_inventory**2
                expected+=probability*value;inventory+=probability*terminal_inventory
                outcomes.append({'buy':buy,'sell':sell,'probability':float(probability),'net_value':float(value)})
            candidates.append((expected,bid,ask,inventory,outcomes))
        return min(candidates,key=lambda row:(-row[0],row[1],row[2]))
    actual=optimize(inputs['initial_inventory_blocks']);flat=optimize(0)
    return {'inventory_quote_offsets':list(actual[1:3]),'flat_quote_offsets':list(flat[1:3]),'inventory_objective':float(actual[0]),'expected_final_inventory':float(actual[3])}, {'selected_quote_fill_states':actual[4],'joint_law':'Independent bid and ask Bernoulli fills; terminal mark responds to their realized difference.'}


def bisect_root(function,low,high):
    flo=function(low)
    if flo*function(high)>0:raise ValueError('Root is not bracketed')
    for _ in range(120):
        middle=(low+high)/2;fm=function(middle)
        if fm==0:return middle
        if flo*fm<=0:high=middle
        else:low=middle;flo=fm
    return (low+high)/2


def q093(inputs):
    spread=inputs['gross_alpha_per_year']-inputs['linear_drag_per_year'];impact=inputs['impact_coefficient'];fixed=inputs['annual_fixed_cost_usd_million'];maximum=inputs['maximum_aum_usd_million']
    operating=lambda capital:spread*capital-impact*capital**1.5-fixed
    point=min(maximum,(2*spread/(3*impact))**2)
    capital=point if operating(point)>0 else 0
    if not 0<point<maximum or operating(point)<=0:raise ValueError('Two interior break-even roots require the stated profitable finite-capacity regime')
    roots=[bisect_root(operating,0,point),bisect_root(operating,point,maximum)]
    return_maximum=min(maximum,(2*fixed/impact)**(2/3))
    return {'profit_maximizing_aum':capital,'maximum_annual_profit':operating(capital) if capital else 0,'positive_break_even_aum':roots,'return_maximizing_aum':return_maximum}, {'break_even_profit_residuals':[operating(root) for root in roots],'dollar_profit_derivative':spread-1.5*impact*math.sqrt(capital),'return_objective_derivative':-impact/(2*math.sqrt(return_maximum))+fixed/return_maximum**2,'shutdown_avoids_fixed_cost':True}


CHECKS={
 'q013':(q013,'Enumerate the complete integer desk menu with nonlinear shared capital charge, gross cap and lexicographic objective ordering.'),
 'q017':(q017,'Enumerate tax-lot sales, charge positive gains and one combined execution ticket, then rank feasible cash plans by shares and economic cost.'),
 'q018':(q018,'Enumerate all nonanticipating contingent policies; solve each post-fee-NAV trade exactly and propagate all four terminal cash/risky-asset states.'),
 'q019':(q019,'Rational finite sector-neutral pair grid with nonnegative full holdings and exact tracking-variance inequalities under both covariance completions.'),
 'q020':(q020,'Enumerate all integer bond ladders with receipt-before-liability cash prefixes and no terminal-surplus credit against purchase funding.'),
 'q023':(q023,'Triangle-inequality gross bounds are attained by the two disclosed identity completions; constrain the same concave scaling objective separately.'),
 'q052':(q052,'Replicate the locked currency receipt by discounting the EUR receivable, converting at spot, and investing in USD.'),
 'q056':(q056,'For every integer share hedge, solve the maximum terminal-state cash requirement, then minimize initial ask purchase plus signed cash account.'),
 'q061':(q061,'Solve the triangular coupon-pricing linear system, reprice both bonds, and convert adjacent discounts to a simple forward.'),
 'q062':(q062,'Separate projected coupon cash flows from discount factors; compute signed projection and discount sensitivities with a numerical derivative cross-check.'),
 'q063':(q063,'Enumerate every collateral-feasible key-rate hedge grid, price all joint stresses including zero, and apply the full tie rule.'),
 'q064':(q064,'Reprice the full annual-discount cash-flow sum and obtain first/second Taylor terms from Richardson central differences.'),
 'q068':(q068,'Finance purchase and prepaid storage together; add the terminal delivery fee and enforce both whole-barrel funding and tank bounds.'),
 'q073':(q073,'Apply collateral thresholds and caps by state, then compare the exact joint default/exposure expectation with the matched-marginal product.'),
 'q077':(q077,'Convert one-way turnover to both traded legs and reconcile percent, basis-point, fee and net-Sharpe units.'),
 'q082':(q082,'Solve stationary AR moments and recursively propagate conditional expectations; verify mean, variance and half-life fixed-point identities.'),
 'q090':(q090,'Add the stipulated first-passage integrands analytically and integrate exp(x^2) with a convergent positive power series across the complete threshold menu.'),
 'q091':(q091,'Enumerate complete liquidation schedules with interval risk attached to post-slice inventory, separate permanent/temporary costs and exact objective ordering.'),
 'q092':(q092,'Enumerate all independent bid/ask fill states for each quote pair, keep cash and marked inventory separately, and average the nonlinear inventory penalty.'),
 'q093':(q093,'Differentiate separate dollar-profit and return-on-AUM objectives, bracket both positive operating break-even roots, and preserve the shutdown discontinuity.'),
}
