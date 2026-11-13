"""Independent arithmetic for remaining fixed public decision contracts.

Only candidate-visible inputs and released documents enter these calculations.
Certificates distinguish fixed snapshots from state-transition validation.
No production scorer, solution table or simulator algorithm is imported.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import itertools
import json


SOURCE_HASHES = {
    "c203:1": "55c85ff0ed820bf15fc5628c349d3ca0766d995a3e7770e2ea944e64b6e78eae",
    "c204:1": "3a77ac929b0d1d5efde0ce4cdc2a2db1db72b54bcea7223709c30ca4d1fb17b9",
    "c207:1": "918e8ee39f9329c3eed87be76090c1f5be01155fdb2e1c11640be0efe6d5e1b0",
    "c207:2": "2c0b538c79fb4a0459b6303d4bdc143c3b046c09c3c303eb1ba3cc18150f0242",
    "c207:3": "73c2fdacf4e6a061db5b75b9d486c8d356a762e32e69c8701a0853049d238ec3",
    "c207:4": "ff6fc79fc6daeefb4c9220702e64f32909fdca5279ec564e148d65626fa5bd8f",
    "c208:1": "25e8d028d65f5c5033861e66f7571918da642cd916cc87a6f964dcaf3c19d512",
    "c208:2": "4b7cd4ae8d6471c92e0cd72473d22c58a8dd6d2ff80eb5c86351686eb71abbf5",
    "c208:3": "cadeba8077fdf45770e7c973c8240f1a3a57a3dffe1c938a6f37348de4d25d2e",
    "c208:4": "94220f1a5099f83d777ebf7377be1b539fd139bebc98bc26f6fc44600d2e95de",
    "c209:1": "a0b084b31e8de87bbdccf2d41155631356b9aa21130b1bcb06bf19c9155d054a",
    "c209:2": "13c44f8968bffa3216f7d3fe4af7793e9bc1b6be2736257c262534ca995ad247",
    "c209:3": "043ba01afd3feaa113352072c7144a6bd59bf49ba9fb8723f078ecef8bc7247c",
    "c209:4": "c8cc265e0bc03473fc1be26f57249b418256de6531212ecc172cec5a01098a69",
    "c210:1": "77327a85c540cf3b199c6e452cf57bce938f84937bbce2e02150c5e950e8dc72",
    "c210:2": "940aa0161b2ce2ea926868a3cb201a5b17c13a3d7ba0591b9d38332fcf812b4d",
    "c210:3": "fa3a07e96e284d7c2f9c892185d7335408589b3373758a66af70382d546b8fb5",
    "c210:4": "cc8d9620774ecd5f063e799b5ce1450c899efc52b909a3c948a8620c10382e26",
    "c211:1": "df5a8741269de0ec254a3ab69c23f65185c207ace841eb79656d1d126c3a11aa",
    "c211:2": "917eca503b692c3a9c7574b89cc9eac83b8b178d2fc2896e4e4b3b6d4c50cc4f",
    "c211:3": "aae7aac40da1bfa5cd3a3ea9e6d583dce10f86b8b6367c7eeecfe39a8a0c0b9f",
    "c211:4": "4b0ca90c02b55b791f78078c39557c41587d8cb145e2d1e18df693aefc6aebe9",
    "c212:1": "3584ceea196175c2b35711d8864921d187e4693c5b7ab9723c17f74b69642bfa",
    "c212:2": "dacf0718be7cf75a5350939e61624839c46c48c2567cd9401b6634b5129dd53f",
    "c212:3": "2223d7d5c38a2e3cc0d883ef5583b97ea68507287f7204911a9ce97171979b62",
    "c212:4": "16045c73ac3fb64a88733e7ec046ee453d2f7b8ce8b5e4265fa91b67296484f6",
    "c215:1": "a3efb0c7c84555102675788a34d8a4bcd85c167f8d7fc443cdeeaa653ff3fad9",
    "c216:1": "185264596cfaabd267e876ab4052a3c8b8370528996fcd66eefd6088084278cc",
    "c219:1": "08a86e3552d244ae1babab9d18c34ccb014f697965ce879c1f7ea5e56a13c252",
    "c219:2": "4cab62c10e187029f95dade581cf1c4fe8165f9ad8c6593ea5790b1671da0caa",
    "c219:3": "38e4987eea970532320a6fd3ee58b13b7c9c0e6ce88aeb7b174f4d8ece1c83f9",
    "c219:4": "eac6c279451f7684ac68fb20eebde71895e68f763fa7cc3222ad5e11b457456b",
    "c220:1": "ac4dfa58f6559d2ca643ca1e7cdf38f3ade2e57e94e877da63f803a42472eaf4",
    "c220:2": "4f2710a7b1fd17c4d9146e63f4f5d99473fdd38aeb6425b7bdd693b92b1d0a3f",
    "c220:3": "34820acf45e89e08e792e0808b6dbb54ccb329cc2d6a9fa620bc31efa0e958e5",
    "c220:4": "0c54f4e2fa6699c546e007a01a9df4ff354c9043bd7078b314068384df52b313",
    "c221:1": "d9174c82f4ab466118fe30e973f1caac5f454ef07a890d1b050c6653dfbd7d6b",
    "c221:2": "55bb0316dbfdae34b2ffbcacc1e6236c6de377aa940242bbef233a7e1ebd3cdb",
    "c221:3": "aae0e9d682562774b0ca56fc10eb81286e81bd927122d11c3073b4785e2810f5",
    "c221:4": "7a86405137f90a1d248e55e460fcf1dee9b19e0ee27315debd228624f57bc863",
    "c222:1": "bebd6c713ee7483d22fb46c1e054d44088aeae5438ea433febc4f0aba415a197",
    "c222:2": "5ae7c47ce5a7ae81231a4799273bdac18e881e48d3b26d54b2b4b0d9968cd350",
    "c222:3": "44c48197ae178d829ec401d1c056649db2ebccc2b257fdc80f33722a4bad5131",
    "c222:4": "e3facb4763b4e117d3fabc8993750dfcf631648916a52c34d07ca1d7b2ddb3d1",
    "c223:1": "e799e81bc3121da4cb020c0e6045c8db77c506d82d0cddbc0024c2795de01d0f",
    "c223:2": "bb6ae76fa58c819121a2fa16932ac26167257bf1bfa85f31bf44a718c47e9499",
    "c223:3": "68d57fff30a2e69345388381a6ef52853efc96182be54be0e50a78574288b73a",
    "c223:4": "9818bac9323557ab7e2d978f13db56ebf43b5f513e0cf13175fd98abfb7e3824",
    "c224:1": "1b3b542b3f1100a85f6df001ceb8eb53f376f396cda5704dba782440e44501d8",
    "c224:2": "9dabc9bc5aada0d1637e253142247e9c822bfbab21b16c4ea9533236e73aa480",
    "c224:3": "79a83b75d41952a260fd4520324472c7a8575e12f622fc468d0ace76004ed14f",
    "c224:4": "495da0e44c7815ffc8db3538af003a7becd821849017eac627988b38ad116876"
}


def scope(question):
    cid = question['question_id']
    digest = hashlib.sha256(json.dumps(question['sources'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != SOURCE_HASHES[f'{cid}:{question["turn"]}']:
        raise ValueError('Public source contract differs from reviewed text')
    return {'public_source_sha256': digest, 'turn': question['turn'],
            'scope': 'All declared outputs for the supplied public snapshot; no transition, arbitrary-trajectory, source-validity or practitioner certification.'}


def f(value):
    return F(str(value))


def tail_mean(losses, probabilities, mass):
    """Signed upper-tail mean, splitting atoms; no zero truncation."""
    residual = mass
    total = F(0)
    for loss, probability in sorted(zip(losses, probabilities), reverse=True):
        portion = min(residual, probability)
        total += loss * portion
        residual -= portion
        if residual == 0:
            return total / mass
    raise ValueError('Tail exceeds total probability')


def c203(question):
    certificate = scope(question)
    inputs = question['inputs']
    terminal = list(map(f, inputs['terminal_asset_values']))
    price, premium = f(inputs['purchase_price']), f(inputs['insurance_premium'])
    laws = [list(map(f, inputs['endpoint_probability_laws'][name])) for name in ('cautious', 'constructive')]
    pnl = {'pass': [F(0)] * 3, 'unprotected_purchase': [v - price for v in terminal],
           'insured_purchase': [max(price, v) - price - premium for v in terminal]}
    loss = {action: [2 * max(-z, 0) - max(z, 0) for z in values] for action, values in pnl.items()}
    expected = {action: [sum(p * z for p, z in zip(law, losses)) for law in laws] for action, losses in loss.items()}
    minima = [min(values[index] for values in expected.values()) for index in range(2)]
    regrets = {action: max(value - minimum for value, minimum in zip(values, minima)) for action, values in expected.items()}
    priority = {'pass': 0, 'insured_purchase': 1, 'unprotected_purchase': 2}
    chosen = min(expected, key=lambda action: (regrets[action], priority[action]))
    values = sorted(sum(p * v for p, v in zip(law, terminal)) for law in laws)
    certificate.update(endpoint_expected_losses={a: list(map(str, v)) for a, v in expected.items()},
                       endpoint_minima=list(map(str, minima)), worst_regrets={a: str(v) for a, v in regrets.items()},
                       mixture_argument='Regret is a maximum of affine action-loss differences, hence convex; its maximum over the interval is attained at an endpoint.')
    return {'action': chosen, 'chosen_expected_losses': list(map(float, expected[chosen])),
            'model_value_interval': list(map(float, values)), 'selected_method': 'minimax_expected_regret',
            'worst_case_expected_loss': float(max(expected[chosen])), 'worst_case_regret': float(regrets[chosen])}, certificate


def treasury_facilities(opening, transfers, early_inflow, release, short_cap=F(20), term_cap=F(25)):
    """Solve the continuous two-facility LP by exact intersection vertices."""
    need = transfers - opening
    # Every inequality a*s+b*t<=c, including the 11:00 short repayment.
    constraints = [(-1, 0, 0), (0, -1, 0), (1, 0, short_cap), (0, 1, term_cap),
                   (-1, -1, -need), (0, -1, early_inflow - need),
                   (F(1, 100), F(4, 100), opening - transfers + early_inflow + release)]
    feasible = []
    for left, right in itertools.combinations(constraints, 2):
        a, b, c = map(f, left); d, e, g = map(f, right)
        determinant = a * e - b * d
        if determinant == 0:
            continue
        short, term = (c * e - b * g) / determinant, (a * g - c * d) / determinant
        if all(a * short + b * term <= c for a, b, c in constraints):
            fee = short / 100 + term / 25
            feasible.append((fee, short + term, short, term))
    if not feasible:
        raise ValueError('No feasible facility funding')
    fee, total, short, term = min(feasible)
    final = opening - transfers + early_inflow + release - fee
    return short, term, fee, final, min(row[1] for row in feasible), feasible


def c204(question):
    certificate = scope(question); inputs = question['inputs']
    opening = inputs['opening_cash']
    calls = {row['account']: f(row['amount']) for row in inputs['cash_calls']}
    transfers = [calls[account] - f(opening[account]) for account in ('A', 'B')]
    receipts = {row['account']: f(row['amount']) for row in inputs['cash_receipts']}
    short, term, fee, final, minimum, vertices = treasury_facilities(f(opening['treasury']), sum(transfers), receipts['treasury'], receipts['A'] + receipts['B'])
    certificate.update(feasible_vertices=[[str(v) for v in row] for row in vertices],
                       cash_after_0900=str(f(opening['treasury']) + short + term - sum(transfers)),
                       cash_after_1100=str(f(opening['treasury']) + term - sum(transfers) + receipts['treasury']),
                       principal_cancels_from_final_cash=True)
    return {'early_desk_transfers': list(map(float, transfers)), 'facility_draws': [float(short), float(term)],
            'final_unrestricted_treasury_cash': float(final), 'minimum_external_draw': float(minimum),
            'net_directional_contracts': sum(inputs['directional_contracts'].values()), 'total_fee': float(fee)}, certificate


def dependence_es(losses, default_a=F(1, 4), default_b=F(1, 2), mass=F(1, 5)):
    """Enumerate all breakpoints of ES as the joint-default probability moves."""
    lower, upper = max(F(0), default_a + default_b - 1), min(default_a, default_b)
    constants = (F(0), default_a, default_b, 1 - default_a - default_b)
    slopes = (1, -1, -1, 1)
    order = sorted(range(4), key=lambda index: losses[index], reverse=True)
    points = {lower, upper}; intercept = F(0); slope = 0
    for index in order:
        intercept += constants[index]; slope += slopes[index]
        if slope:
            breakpoint = (mass - intercept) / slope
            if lower <= breakpoint <= upper:
                points.add(breakpoint)
    values = [(tail_mean(losses, [constant + coefficient * p for constant, coefficient in zip(constants, slopes)], mass), p)
              for p in sorted(points)]
    return max(value for value, p in values), values


def c215(question):
    certificate = scope(question); feasible = []
    for a, b, protection in itertools.product(range(3), repeat=3):
        cost = 80 * a + 50 * b + 3 * protection
        if cost > 200:
            continue
        losses = [F(cost - payoff) for payoff in (40*a + 20*b + 30*protection, 40*a + 100*b, 100*a + 20*b, 100*a + 100*b)]
        es, breakpoints = dependence_es(losses)
        profits = [-(p*losses[0] + (F(1,4)-p)*losses[1] + (F(1,2)-p)*losses[2] + (F(1,4)+p)*losses[3]) for p in (F(0), F(1,4))]
        if es <= 55:
            feasible.append((min(profits), cost, (a,b,protection), losses, es, sorted(profits), breakpoints))
    value, cost, action, losses, es, profits, points = min(feasible, key=lambda row: (-row[0], row[1], row[2]))
    certificate.update(enumerated_actions=27, feasible_actions=len(feasible), es_breakpoints=[[str(p), str(v)] for v,p in points],
                       dependence_argument='Four state masses are affine in joint-default probability. Tail order is fixed; extrema occur at endpoints or cumulative-tail-mass breakpoints.')
    return {'action': list(action), 'expected_profit_interval': list(map(float, profits)),
            'joint_default_probability_interval': [0.0, 0.25], 'minimum_expected_profit': float(value),
            'purchase_cost': cost, 'state_losses': list(map(float, losses)), 'worst_expected_shortfall': float(es)}, certificate


def cascade_plan(action):
    a,b,c,draw = action
    cap = max(0, min(15, 3*(6-a)-2*b)); haircut = 8 if a >= 3 else 0
    early = 20 + 8*a + 6*c + draw - 50 - 5*b - haircut
    final = F(early + 12*b + 30 - 12 - draw) - F(draw,2)
    risk = 6*(6-a) + 4*(5-b) + 6*c
    cost = 4*a + 3*b + 9*c + F(draw,2) + haircut
    return {'cap':cap,'haircut':haircut,'early':early,'final':final,'risk':risk,'cost':cost,
            'feasible':draw <= cap and early >= 0 and final >= 8 and risk <= 34}


def c216(question):
    certificate = scope(question)
    rows = [(action,cascade_plan(action)) for action in itertools.product(range(7),range(6),range(5),range(21))]
    feasible = [(action,result) for action,result in rows if result['feasible']]
    action, result = min(feasible,key=lambda row:(row[1]['cost'],row[0]))
    certificate.update(enumerated_actions=len(rows),feasible_actions=len(feasible),
                       exact_cost=str(result['cost']),next_cost=str(sorted(set(row[1]['cost'] for row in feasible))[1]))
    return {'action':list(action),'borrowing_base':result['cap'],'early_cash':result['early'],
            'economic_cost':float(result['cost']),'final_cash':float(result['final']),
            'haircut_call':result['haircut'],'remaining_stress_risk':result['risk']}, certificate


def worst_stress(holdings, rows):
    return max([F(0)] + [sum(f(coefficient)*quantity for coefficient,quantity in zip(row,holdings)) for row in rows])


def c207(question):
    certificate=scope(question);book=question['inputs']['current_book'];terms=question['inputs']['terms']
    a0,b0,h0=book['positions'];options=[]
    for a,b,h in itertools.product(range(a0+1),range(b0+1),range(-h0,terms['hedge_cap']-h0+1)):
        fee=F(a*a,5)+F(b*b,10)+sum(value!=0 for value in (a,b,h))
        cash=f(book['cash'])+a*f(terms['sell_prices'][0])-max(h,0)*f(terms['hedge_buy'])+max(-h,0)*f(terms['hedge_sell'])-fee-f(terms['liability'])-f(book['unpaid'])
        positions=[a0-a,b0-b,h0+h];risk=worst_stress(positions,terms['stress_loss_per_lot'])
        if cash<f(terms['reserve']) or risk>f(terms['risk_cap']):continue
        objective=2*a+b+2*max(h,0)+max(-h,0)+fee
        options.append((objective,(a,b,h),cash,positions,risk))
    certificate['feasible_plans']=len(options)
    if not options:
        return {'decision':'NO_FEASIBLE_PLAN','foregone_value':0,'post_cash':book['cash'],
                'post_positions':book['positions'],'trade_lots':[0,0,0],
                'worst_stress_loss':float(worst_stress(book['positions'],terms['stress_loss_per_lot']))},certificate
    objective,action,cash,positions,risk=min(options)
    certificate.update(all_optimal_actions=[list(row[1]) for row in options if row[0]==objective],
                       unsettled_current_sale_proceeds=str(action[1]*f(terms['sell_prices'][1])),
                       scope='All outputs for this supplied authored account snapshot. Current rules are applied independently; commit, advance and alternate trajectories are not validated.')
    return {'decision':'EXECUTE','foregone_value':float(objective),'post_cash':float(cash),
            'post_positions':positions,'trade_lots':list(action),'worst_stress_loss':float(risk)},certificate


def c212(question):
    certificate=scope(question);inputs=question['inputs'];book=inputs['current_book'];lot=inputs['board_lot']
    quantity=0;inventory=book['inventory'];cash=f(book['cash']);live=book['live'];target=book['target']
    committed=cash-live*f(book['price']);post=inventory
    if live:
        action='CANCEL' if inventory+live>target else 'WAIT'
    elif inventory>target:
        quantity=(inventory-target)//lot*lot;action='SELL' if quantity else 'WAIT'
        post-=quantity;committed+=quantity*f(inputs['sell_price'])
    else:
        affordable=max(0,(cash-f(inputs['cash_reserve']))//(lot*f(inputs['current_new_buy_price'])))
        quantity=min((target-inventory)//lot,affordable)*lot;action='BUY' if quantity else 'WAIT'
        post+=quantity;committed-=quantity*f(inputs['current_new_buy_price'])
    return {'action':action,'cash_after_commitments':float(committed),'cash_balance':float(cash),
            'confirmed_inventory':inventory,'inventory_after_new_order_fill':post,'live_buy_quantity':live,
            'quantity':quantity,'target_shortfall_after_new_order':target-post},certificate


def observations(question):
    return {row['source_id']: deepcopy(row['data']) for row in question['inputs']['observations']}


def c208(question):
    certificate = scope(question); data = observations(question)
    policy, sample = data['q_policy'], data['q_sample']
    groups = sample['groups']
    if 'q_clock' in data:
        correction = data['q_clock']
        for row in groups:
            if row['group'] == correction['group']:
                row['availability'] = correction['feature_availability']
    eligible = [row for row in groups if row['availability'] < policy['decision_time']]
    names = {row['group'] for row in eligible}
    count = sum(row['trades'] for row in eligible)
    gross_sum = sum(f(row['gross_return_sum_pp']) for row in eligible)
    p_value = next(f(row['p_value']) for row in sample['audited_tests'] if set(row['groups']) == names)
    trials = data.get('q_trials', sample)['recorded_pipeline_count']
    cutoff = f(policy['familywise_alpha']) / trials
    status = 'provisional' if p_value <= cutoff else 'reject'
    result = {'admissible_count': count, 'deploy': False, 'gross_mean': float(gross_sum/count),
              'p_value': float(p_value), 'significance_cutoff': float(cutoff), 'status': status}
    if 'q_execution' in data:
        execution = data['q_execution']
        if execution['holdout_outcomes_released']:
            raise ValueError('This fixed audit has no rule for an unspecified released holdout outcome')
        net = (gross_sum - f(execution['execution_cost_sum_pp'])) / count
        result.update(net_mean=float(net), holdout_status='unobserved')
        result.pop('p_value')
        if net <= 0:
            result['status'] = 'reject'
    certificate.update(eligible_groups=sorted(names), statistical_scope='Audited p-values are supplied sufficient statistics, not independently reconstructed from unavailable returns.')
    return result, certificate


def c209(question):
    certificate = scope(question); data = observations(question)
    settlement = {row['state']: f(row['cash']) for row in data.get('v_recovery', data['v_contract'])['settlement']}
    probability = {name: f(value) for name, value in data.get('v_risk', data['v_belief'])['physical_probabilities'].items()}
    physical = sum(probability[state] * cash for state,cash in settlement.items())
    if 'v_market' not in data:
        return {'fair_value':'unidentified', 'identified':False, 'lower_bound':float(min(settlement.values())),
                'upper_bound':float(max(settlement.values())), 'physical_expected_payoff':float(physical),
                'method':'no_unique_price'}, certificate
    quotes = {row['instrument']: f(row['price']) for row in data['v_market']['instruments']}
    # One recovery cash position plus the payoff difference in event certificates.
    price = settlement['not_E'] * quotes['cash'] + (settlement['E']-settlement['not_E']) * quotes['event_certificate']
    certificate.update(replication_cash=str(settlement['not_E']),replication_event_certificates=str(settlement['E']-settlement['not_E']))
    if 'v_risk' not in data:
        return {'fair_value':float(price),'identified':True,'lower_bound':float(price),'upper_bound':float(price),
                'physical_expected_payoff':float(physical),'method':'state_replication'}, certificate
    states = sorted(settlement); losses = [price-settlement[state] for state in states]
    probabilities = [probability[state] for state in states]; confidence=f(data['v_risk']['tail_confidence'])
    cumulative=F(0)
    for loss,p in sorted(zip(losses,probabilities)):
        cumulative += p
        if cumulative >= confidence:
            var = loss; break
    es = tail_mean(losses,probabilities,1-confidence)
    return {'es_80':float(es),'fair_value':float(price),'loss_probability':float(sum(p for loss,p in zip(losses,probabilities) if loss>0)),
            'method':'state_replication_and_physical_tail','physical_expected_profit':float(physical-price),'var_80':float(var)}, certificate


def c210(question):
    certificate = scope(question); data = observations(question)
    opening=data['f_open']; cash={row['entity']:f(row['cash']) for row in opening['opening_cash']}
    projects={row['project']:row for row in opening['projects']}
    inventory=next(row for row in opening['inventory'] if row['entity']=='X')
    encumbered=f(inventory['encumbered_value']); transfer_time='00:00'
    if 'f_profit' in data:
        amendment=data['f_profit'];projects[amendment['project']]['net_profit']=amendment['net_profit']
    if 'f_constraints' in data:
        amendment=data['f_constraints'];encumbered=f(amendment['encumbered_inventory_value']);transfer_time=amendment['transfer_settlement']
    if 'f_final' in data:
        amendment=data['f_final']
        for receipt in amendment['cash_receipts']:
            if receipt['time'] > min(row['cutoff'] for row in projects.values()):
                raise ValueError('A late receipt needs a separate deadline cash ledger')
            cash[receipt['entity']] += f(receipt['cash'])
        for item in amendment['segregated_cash']:
            cash[item['entity']] -= f(item['cash'])
        cutoff=amendment['project_cutoff'];projects[cutoff['project']]['cutoff']=cutoff['cutoff']
    capacity=f(data['f_rules']['collateral_advance_rate'])*(f(inventory['value'])-encumbered)
    options=[];names=sorted(projects)
    for mask in itertools.product((False,True),repeat=len(names)):
        chosen=[projects[name] for name,included in zip(names,mask) if included]
        x=sum(f(row['required_cash']) for row in chosen if row['entity']=='X')
        y=sum(f(row['required_cash']) for row in chosen if row['entity']=='Y')
        transfer=max(F(0),y-cash['Y']);borrow=max(F(0),x+transfer-cash['X'])
        if borrow>capacity or (transfer and any(transfer_time>row['cutoff'] for row in chosen if row['entity']=='Y')):
            continue
        profit=sum(f(row['net_profit']) for row in chosen)
        options.append((profit,borrow,transfer,[row['project'] for row in chosen],cash['X']+borrow-x-transfer,cash['Y']+transfer-y))
    profit,borrow,transfer,names,cash_x,cash_y=min(options,key=lambda row:(-row[0],row[1],row[2],row[3]))
    certificate.update(enumerated_project_subsets=2**len(projects),feasible_project_subsets=len(options))
    return {'borrow_X':float(borrow),'cash_X':float(cash_x),'cash_Y':float(cash_y),'net_profit':float(profit),
            'projects':names,'repo_capacity':float(capacity),'transfer_X_to_Y':float(transfer)},certificate


def report_value(prior, good_payoff, bad_payoff, positive_good, positive_bad, fee):
    """One-query expected payoff, integrating states before any division."""
    joint_positive=(prior*positive_good,(1-prior)*positive_bad)
    joint_negative=(prior*(1-positive_good),(1-prior)*(1-positive_bad))
    outcomes=[good_payoff*g+bad_payoff*b for g,b in (joint_positive,joint_negative)]
    actions=['invest' if value>0 else 'abstain' for value in outcomes]
    return sum(max(F(0),value) for value in outcomes)-fee,actions


def c211(question):
    certificate=scope(question);data=observations(question);terms=data['i_terms'];prior=f(terms['prior_probability_G'])
    payoffs=data.get('i_loss',terms)['investment_payoffs'];good,bad=map(f,(payoffs['G'],payoffs['not_G']))
    reports={row['report']:row for row in data['i_menu']['reports']};budget=f(terms['research_budget'])
    if 'i_observed' in data:
        actual=data['i_observed'];r=reports['R']
        numerator=prior*f(r['positive_probability_G'])
        prior=numerator/(numerator+(1-prior)*f(r['positive_probability_not_G']))
        reports={row['report']:row for row in actual['reports']};budget=f(actual['research_budget'])
    if 'i_terminal' in data:
        report=reports['T'];numerator=prior*(1-f(report['positive_probability_G']))
        prior=numerator/(numerator+(1-prior)*(1-f(report['positive_probability_not_G'])))
        investment=prior*good+(1-prior)*bad;sale=f(data['i_terminal']['resale_payoff'])
        value,action=max(((F(0),'abstain'),(investment,'invest'),(sale,'sell')),key=lambda row:row[0])
        return {'action':action,'expected_net':float(value),'investment_value':float(investment),
                'posterior_G':float(prior),'query':'none','remaining_budget':data['i_terminal']['research_budget'],
                'state_identified':prior in (0,1)},certificate
    immediate=prior*good+(1-prior)*bad;baseline=max(F(0),immediate)
    baseline_action='invest' if immediate>0 else 'abstain'
    options=[(baseline,'none',[baseline_action]*2,F(0))];increments={}
    for name,report in reports.items():
        fee=f(report['fee'])
        if fee>budget:continue
        if name=='D':
            value,actions=baseline-fee,[baseline_action]*2
        else:
            value,actions=report_value(prior,good,bad,f(report['positive_probability_G']),f(report['positive_probability_not_G']),fee)
        increments[name]=value-baseline;options.append((value,name,actions,fee))
    value,query,actions,fee=min(options,key=lambda row:(-row[0],row[1]!='none',row[1]))
    certificate.update(query_incremental_values={name:str(value) for name,value in increments.items()},posterior_fraction=str(prior),sunk_fees_excluded=True)
    if question['turn']==2:
        return {'D_incremental_value':float(increments['D']),'T_incremental_value':float(increments['T']),
                'action':baseline_action,'expected_net':float(value),'posterior_G':float(prior),'query':query},certificate
    answer={'baseline_value':float(baseline),'expected_net':float(value),'negative_action':actions[1],
            'positive_action':actions[0],'query':query,'remaining_budget':float(budget-fee)}
    if question['turn']==3:answer.update(incremental_value=float(value-baseline),posterior_G=float(prior))
    return answer,certificate


def round_result(question, choices, certificate):
    """choices contain (objective, action, cost, violation, diagnostics)."""
    objective,action,cost,violation,diagnostics=min(choices,key=lambda row:(row[0],row[1]))
    def plain(value):
        if isinstance(value,F):return float(value)
        if isinstance(value,(list,tuple)):return [plain(item) for item in value]
        return value
    certificate.update(mechanically_legal_actions=len(choices),
                       all_optimal_actions=[plain(row[1]) for row in choices if row[0]==objective],
                       exact_objective=str(objective),
                       scope='All outputs for this supplied authored round snapshot; no commit, advance, hidden-realization or arbitrary-trajectory validation. Lexicographic order selects an audit representative among equal optima.')
    return {'action':plain(action),'constraint_violation':plain(violation),'diagnostics':plain(diagnostics),
            'objective':plain(objective),'state_round':question['inputs']['current_state']['stage'],
            'transaction_cost':plain(cost)},certificate


def c219(question):
    certificate=scope(question);s=question['inputs']['current_state'];t=question['inputs']['current_terms'];choices=[]
    for a,b,h in itertools.product(range(s['positions'][0]+1),range(s['positions'][1]+1),range(-s['hedges'],5-s['hedges'])):
        fee=f(t['joint_impact'])*(a+b)**2+F(3,10)*h*h+sum(x!=0 for x in (a,b,h))
        available=f(s['cash'])+a*f(t['prices'][0])-max(h,0)*f(t['hedge_price'])+max(-h,0)*f(t['hedge_salvage'])-fee
        if available<0:continue
        obligation=f(t['due'])+f(s['arrears']);cash=max(F(0),available-obligation);arrears=max(F(0),obligation-available)
        risk=worst_stress((s['positions'][0]-a,s['positions'][1]-b,s['hedges']+h),t['stress'])
        violation=max(F(0),f(t['reserve'])-cash)+max(F(0),risk-f(t['risk_cap']))+arrears
        objective=fee+2*a+F(3,2)*b+F(7,10)*abs(h)+1000*violation
        choices.append((objective,(a,b,h),fee,violation,[cash,risk,arrears]))
    return round_result(question,choices,certificate)


def c220(question):
    certificate=scope(question);s=question['inputs']['current_state'];t=question['inputs']['current_terms'];choices=[]
    eligible=[int(not leaked and regime and f(p)*trials<=f(t['alpha']))
              for leaked,regime,p,trials in zip(t['leakage'],t['regime_ok'],t['p'],t['trials'])]
    for allocation in itertools.product(range(t['max_units']+1),repeat=3):
        fee=f(t['turnover_cost'])*sum(abs(new-old) for new,old in zip(allocation,s['allocations']))
        capital=f(t['capital_per_unit'])*sum(allocation)
        if capital+fee>f(s['capital']):continue
        edge=sum(q*(f(gross)-f(cost)) for q,gross,cost in zip(allocation,t['gross_edge'],t['unit_cost']))-fee
        violation=sum(q*(1-good) for q,good in zip(allocation,eligible))
        choices.append((-edge+1000*violation,allocation,fee,violation,[edge,capital,eligible]))
    certificate['adjusted_p_values']=[str(f(p)*count) for p,count in zip(t['p'],t['trials'])]
    return round_result(question,choices,certificate)


def c221(question):
    certificate=scope(question);s=question['inputs']['current_state'];t=question['inputs']['current_terms'];choices=[]
    for hedge in range(t['limit']+1):
        fee=abs(hedge-s['hedge'])*f(t['fee'])
        if fee>f(s['cash']):continue
        losses=[f(t['exposure'])*f(slope)-hedge*f(loading) for interval,loading in zip(t['scenario_slopes'],t['hedge_loadings']) for slope in interval]
        lower,upper=min(losses),max(losses);violation=max(F(0),upper-f(t['max_worst_loss']))
        choices.append((upper+fee+1000*violation,hedge,fee,violation,[lower,upper,f(s['cash'])-fee]))
    certificate['interval_argument']='Each model loss is affine in its slope; both range endpoints suffice within each closed interval.'
    return round_result(question,choices,certificate)


def c222(question):
    certificate=scope(question);s=question['inputs']['current_state'];t=question['inputs']['current_terms'];choices=[]
    for a,b,borrow in itertools.product(range(9),repeat=3):
        remaining_a=f(s['a'])-a*f(t['a_unit']);remaining_b=f(s['b'])-b*f(t['b_unit'])
        # Existing deficits do not make a zero release mechanically illegal.
        if (a and remaining_a<f(t['a_floor'])) or (b and remaining_b<f(t['b_floor'])) or f(s['debt'])+borrow>f(t['borrow_cap']):continue
        proceeds_a=a*f(t['a_unit'])*(1-f(t['a_haircut']))
        pending=f(s['pending'])+b*f(t['b_unit'])*(1-f(t['b_haircut']))
        available=f(s['central'])+proceeds_a+borrow*(1-f(t['borrow_fee']))
        obligation=f(t['due'])+f(s['arrears']);cash=max(F(0),available-obligation);arrears=max(F(0),obligation-available)
        cost=a*f(t['a_unit'])*f(t['a_haircut'])+b*f(t['b_unit'])*f(t['b_haircut'])+borrow*f(t['borrow_fee'])
        violation=max(F(0),f(t['a_floor'])-remaining_a)+max(F(0),f(t['b_floor'])-remaining_b)+max(F(0),f(t['central_floor'])-cash)+arrears
        if s['stage']<4:
            violation+=max(F(0),f(t['next_due'])+f(t['central_floor'])-f(t['next_inflow'])-cash-pending)
        objective=1000*violation+cost+F(1,10)*(f(s['debt'])+borrow)
        choices.append((objective,(a,b,borrow),cost,violation,[cash,arrears,pending]))
    return round_result(question,choices,certificate)


def information_tree(worlds, costs, budget, rounds, observations, committed=False):
    """Exact recursive information-set tree, with observations filtering worlds."""
    possible=[row for row in worlds if all(row[name.lower()]==bit for name,bit in observations.items())]
    mass=sum(f(row['probability']) for row in possible)
    if mass<=0:raise ValueError('Observed history has zero probability')
    mean=sum(f(row['probability'])*f(row['payoff']) for row in possible)/mass
    if committed:return [('WAIT',F(0),F(0))],mean
    choices=[('ABSTAIN',F(0),F(0)),('INVEST',mean,F(0))]
    if rounds>1:
        for name,cost in costs.items():
            cost=f(cost)
            if name in observations or cost>budget:continue
            continuation=F(0)
            for bit in (0,1):
                probability=sum(f(row['probability']) for row in possible if row[name.lower()]==bit)/mass
                if probability:
                    descendants,_=information_tree(worlds,costs,budget-cost,rounds-1,dict(observations,**{name:bit}))
                    continuation+=probability*max(row[1] for row in descendants)
            choices.append(('READ_'+name,continuation-cost,cost))
    return choices,mean


def c223(question):
    certificate=scope(question);s=question['inputs']['current_state'];t=question['inputs']['current_terms']
    actions,mean=information_tree(t['worlds'],t['costs'],f(s['budget']),t['rounds_left'],s['observations'],s['decision']!='OPEN')
    choices=[(-value,action,cost,F(0),[mean,value,f(s['budget'])-cost]) for action,value,cost in actions]
    certificate['conditional_action_values']={action:str(value) for action,value,cost in actions}
    return round_result(question,choices,certificate)


def execution_choices(state,terms):
    """One-round order choices; pending proceeds never finance current orders."""
    s,t=state,terms;actions=[('WAIT',0)]
    if s['live']:
        if not s['cancel']:actions.append(('CANCEL',0))
    else:
        for quantity in range(1,t['lot_limit']+1):
            if f(s['cash'])-f(t['fee'])-quantity*f(t['buy_price'])>=f(t['reserve']):actions.append(('BUY',quantity))
            if quantity<=s['inventory']:actions.append(('SELL',quantity))
    choices=[]
    for verb,quantity in actions:
        cash=f(s['cash']);inventory=s['inventory'];live=s['live'];pending=f(s['pending']);cancel=s['cancel'];price=f(s['price']);fee=F(0)
        if verb=='CANCEL':cancel=True
        elif verb=='BUY':
            fee=f(t['fee']);cash-=fee;price=f(t['buy_price'])
            if t['terminal']:inventory+=quantity;cash-=quantity*price
            else:live=quantity
        elif verb=='SELL':
            fee=f(t['fee']);inventory-=quantity;pending+=quantity*f(t['sell_price'])-fee
        upper=inventory+(min(2,live) if cancel else live)
        deviation=max(abs(inventory-s['target']),abs(upper-s['target']))
        available=cash-live*price;violation=max(F(0),f(t['reserve'])-available)
        objective=10*deviation+fee+F(live,10)+1000*violation
        choices.append((objective,(verb,quantity),fee,violation,[available,deviation,pending]))
    return choices


def c224(question):
    return round_result(question,execution_choices(question['inputs']['current_state'],question['inputs']['current_terms']),scope(question))


CHECKS = {cid: (globals()[cid], 'Exact finite-state, conditional information-set or polyhedral public-contract calculation.')
          for cid in ('c203','c204','c207','c208','c209','c210','c211','c212','c215','c216','c219','c220','c221','c222','c223','c224')}
