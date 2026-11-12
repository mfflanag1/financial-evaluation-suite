"""Correlated diagnostic variants of one adaptive research decision.

Exact rational reference construction and action-based economic scoring. These
six related instances are a sensitivity panel, not six independent samples.
"""
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
from case_contract import validate

BASE_WEIGHTS = [[2,3,4,5,3,8,9,16],[20,8,7,3,5,3,3,1]]
BASE_PAY = {'invest':[30,-40], 'hedge':[10,-8], 'pass':[0,0]}
FIELDS = {'first_query':'String A,B,C or none.', 'second_queries':'Two strings, continuation decisions after first bit 0,1.', 'terminal_actions':'Four strings invest,hedge or pass under the documented branch order.', 'expected_query_cost':'Number, ex ante total query fees.', 'expected_gross_payoff':'Number, prior expected terminal payoff before fees.', 'optimal_net_value':'Number, prior payoff net of query fees.', 'no_information_value':'Number, best prior payoff with no query.', 'incremental_information_value':'Number, optimal net value minus no-query value.'}


def parameters(costs=('1/2','1/2','3/2'), budget='4', scale=1, permutation=None):
    weights = [list(w) for w in BASE_WEIGHTS]
    pay = {a:[x*scale for x in ps] for a,ps in BASE_PAY.items()}
    if permutation is not None:
        # permutation[new index] is its old signal index; reorder both law and costs.
        weights = [[0]*8 for _ in range(2)]
        for s in range(2):
            for i,w in enumerate(BASE_WEIGHTS[s]):
                old=[(i>>2)&1,(i>>1)&1,i&1]
                j=sum(old[permutation[k]] << (2-k) for k in range(3))
                weights[s][j]=w
        costs=tuple(costs[k] for k in permutation)
    return {'weights':weights,'denominator':100,'costs':dict(zip('ABC',costs)), 'budget':str(budget),'payoffs':pay}


def _decode(p):
    costs={q:F(v) for q,v in p['costs'].items()}
    return p['weights'],F(p['denominator']),costs,F(p['budget']),p['payoffs']


def policy_economics(p, first, continuation, actions):
    weights,denom,costs,budget,pay=_decode(p)
    bad=[]
    if type(first) is not str or first not in ('none','A','B','C') or type(continuation) is not list or len(continuation)!=2 or type(actions) is not list or len(actions)!=4 or any(type(a) is not str or a not in pay for a in actions) or any(type(q) is not str for q in continuation):
        return {'violations':['Malformed observation policy.']}
    if first=='none':
        if continuation!=['stop','stop'] or len(set(actions))!=1:bad.append('No-query policy must stop and repeat its immediate action.')
    else:
        if costs[first]>budget:bad.append('First query exceeds path budget.')
        for b,q in enumerate(continuation):
            if q not in ('stop','A','B','C') or q==first:bad.append('Repeated or unknown second query.')
            elif q!='stop' and costs[first]+costs[q]>budget:bad.append('Branch query-cost budget exceeded.')
            if q=='stop' and actions[2*b]!=actions[2*b+1]:bad.append('Stopped branch actions must agree.')
    if bad:return {'violations':bad}
    gross=F(0);cost=F(0)
    for s,ws in enumerate(weights):
        for i,w in enumerate(ws):
            bits=[(i>>2)&1,(i>>1)&1,i&1]
            if first=='none':a=actions[0];fee=F(0)
            else:
                b=bits['ABC'.index(first)];q=continuation[b];fee=costs[first]
                bit=0 if q=='stop' else bits['ABC'.index(q)]
                if q!='stop':fee+=costs[q]
                a=actions[2*b+bit]
            gross+=F(w,denom)*pay[a][s];cost+=F(w,denom)*fee
    return {'violations':[],'gross':gross,'cost':cost,'net':gross-cost,'objective':cost-gross}


def _key(p):
    import json
    return json.dumps(p,sort_keys=True)


@lru_cache(None)
def _enumerate(key):
    import json
    p=json.loads(key);rows=[]
    for a in sorted(p['payoffs']):
        e=policy_economics(p,'none',['stop','stop'],[a]*4)
        rows.append((e['objective'],'none',('stop','stop'),(a,)*4,e))
    for f in 'ABC':
        options=['stop']+[q for q in 'ABC' if q!=f]
        for qs in product(options,repeat=2):
            for acts in product(sorted(p['payoffs']),repeat=4):
                e=policy_economics(p,f,list(qs),list(acts))
                if not e['violations']:rows.append((e['objective'],f,qs,acts,e))
    return tuple(sorted(rows,key=lambda r:r[:4]))


def enumerate_reference(p):
    return _enumerate(_key(p))


def backward_reference(p, max_queries=2):
    """Independent conditional optimization: no policy_economics calls or trees.

    Branch values use unnormalized joint mass, preserving correlation. Summing
    those values produces an exact prior value without division by branch mass.
    """
    weights,denom,costs,budget,pay=_decode(p)
    def mass_value(action,tests):
        return sum(F(weights[s][i],denom)*pay[action][s]
                   for s in range(2) for i in range(8)
                   if all(((i>>(2-j))&1)==v for j,v in tests))
    def mass(tests):
        return sum(F(weights[s][i],denom) for s in range(2) for i in range(8)
                   if all(((i>>(2-j))&1)==v for j,v in tests))
    def best_terminal(tests):
        return min((-mass_value(a,tests),a) for a in pay)
    val,a=best_terminal([])
    candidates=[(val,'none',('stop','stop'),(a,)*4)]
    if max_queries==0:return candidates[0]
    for f in 'ABC':
        if costs[f]>budget:continue
        branch=[]
        for b in range(2):
            tests=[('ABC'.index(f),b)];v,a=best_terminal(tests)
            opts=[(v,'stop',(a,a))]
            if max_queries>1:
                for q in 'ABC':
                    if q==f or costs[f]+costs[q]>budget:continue
                    va,aa=best_terminal(tests+[('ABC'.index(q),0)])
                    vb,ab=best_terminal(tests+[('ABC'.index(q),1)])
                    opts.append((va+vb+mass(tests)*costs[q],q,(aa,ab)))
            branch.append(min(opts))
        # Lexical first query and continuation precede terminal actions. Each
        # branch objective is independent, hence local minima compose globally.
        b0,b1=branch
        candidates.append((costs[f]+b0[0]+b1[0],f,(b0[1],b1[1]),b0[2]+b1[2]))
    return min(candidates)


def _answers(p):
    o,f,qs,acts,e=enumerate_reference(p)[0]
    baseline=-backward_reference(p,max_queries=0)[0]
    return {'first_query':f,'second_queries':list(qs),'terminal_actions':list(acts),'expected_query_cost':float(e['cost']),'expected_gross_payoff':float(e['gross']),'optimal_net_value':float(-o),'no_information_value':float(baseline),'incremental_information_value':float(-o-baseline)}


# Fixed authoring menu, selected only by exact mathematics before solver use.
# Labels describe private diagnostic roles and never enter public problem text.
SPECS = [
    ('c225','larger continuation benefit',parameters(scale=4)),
    ('c226','cost-driven early stop',parameters(costs=('2','2','6'))),
    ('c227','optimal no query',parameters(costs=('20','20','20'),budget='40')),
    ('c228','changed first-query identity',parameters(costs=('2','1/10','3/2'))),
    ('c229','binding path budget',parameters(budget='1/2')),
    ('c230','semantic isomorphism',parameters(permutation=(1,2,0))),
]


def scalar_routes(p):
    """Sufficient source routes; scalar values need no representation tie rule.

    Gross may be derived by optimizing the economic law without a format source
    only when every net-optimal policy has the same gross payoff.
    """
    ranked=enumerate_reference(p)
    gross_values={r[4]['gross'] for r in ranked if r[0]==ranked[0][0]}
    routes={'no_information_value':[['in_joint','in_actions']],
            'expected_query_cost':[['in_joint','in_query']],
            'expected_gross_payoff':[['in_joint','in_actions','in_format']],
            'optimal_net_value':[['in_joint','in_actions','in_query']],
            'incremental_information_value':[['in_joint','in_actions','in_query']]}
    if len(gross_values)==1:
        routes['expected_gross_payoff'].append(['in_joint','in_actions','in_query'])
    return routes


def build_cases():
    result=[]
    for cid,role,p in SPECS:
        ans=_answers(p);w,denom,costs,budget,pay=_decode(p)
        texts=[
            ('in_joint','Full correlated noisy-signal law',f'State is good or bad. Binary signal triple [A,B,C] is fixed once for the decision. In lexicographic triple order [000,001,010,011,100,101,110,111], JOINT probabilities for good are {w[0]}/{int(denom)} and for bad are {w[1]}/{int(denom)}. These 16 joint cells sum to 1. Queries reveal the selected bit, not a fresh signal. Bits are not assumed independent, even conditional on state. All 16 joint cells are possible.'),
            ('in_actions','Terminal financial opportunities','Exactly one terminal action is allowed: '+ '; '.join(f'{a} pays {ps[0]} in good and {ps[1]} in bad' for a,ps in pay.items())+'. These terminal payoffs exclude query fees. No combinations, shorting, scaling, or later switching are allowed. Maximize prior expected payoff minus purchased query fees. Risk preferences are linear in cash and no wealth constraint applies apart from the research-cost budget.'),
            ('in_query','Adaptive query costs and budget',f'A query costs {costs["A"]} for A, {costs["B"]} for B, and {costs["C"]} for C, regardless of its bit. At most two DISTINCT queries may be purchased. Total fees on EACH possible path must be at most {budget}, including the first query. Stopping before any query or after the first is permitted. The second query identity or stopping may depend on the first observed bit. Terminal action may depend on purchased bits. There is no fee for information not purchased.'),
            ('in_format','Policy representation and deterministic ties','first_query is A,B,C or none. second_queries is a two-element array in first-bit order [0,1], each A,B,C or stop. terminal_actions has four entries [first0/second0,first0/second1,first1/second0,first1/second1]. On a stopped branch repeat its action in both slots. With first_query none use second_queries [stop,stop] and repeat one action in all four slots. Among policies with equal expected net value, use lexicographically smallest tuple (first_query,second_queries,terminal_actions) under ordinary string order, comparing arrays entry by entry.')]
        docs=[{'id':i,'title':t,'text':s,'release_stage':1} for i,t,s in texts];ids=[d['id'] for d in docs]
        routes={k:[ids] for k in ans};routes.update(scalar_routes(p))
        stage={'prompt':'Choose the globally optimal adaptive research and investment policy under the specified joint signal law, available actions, query fees and path budgets. Determine which query to buy first, whether and what to buy next on each result, and the terminal action at each reachable history. Report prior expected economics and apply the documented deterministic tie rule.',
               'inputs':{'scope':'All financial amounts, probabilities, and rules are synthetic and fully specified in the supplied documents.'},'answers':ans,'descriptions':dict(FIELDS),'evidence_routes':routes,'allowed_evidence':{k:ids for k in ans},
               'solution':'Enumerate all represented policies using exact rational joint-probability weights and reject repeat queries, impossible first purchases and path budget breaches. Independently optimize each conditional branch by backward induction, preserving the joint dependence structure. Rank by negative expected net value then the complete stated lexical policy key.',
               'checks':[{'name':'joint mass conservation','passed':sum(map(sum,w))==denom},{'name':'independent exact optimizer agreement','passed':enumerate_reference(p)[0][:4]==backward_reference(p)}],
               'wrong_answers':[{'label':'repeat an observation','answers':{'first_query':'A','second_queries':['A','A']},'reason':'Repeated queries are forbidden and do not draw new evidence.'},{'label':'self-report unattainable value','answers':{'optimal_net_value':ans['optimal_net_value']+100},'reason':'Economic value must be computed from a feasible policy and fees.'}]}
        case={'id':cid,'title':'Adaptive research decision '+cid[-3:],'family':'information_value','split':'development','mode':'single','variant_of':'c217','diagnostic_role':role,'information_parameters':p,'documents':docs,'oracle_document_ids':ids,'stages':[stage]}
        result.append(validate(case))
    return result


def economic_outcome(case,answers):
    if type(answers) is not dict:e={'violations':['Expected answer object.']}
    else:e=policy_economics(case['information_parameters'],answers.get('first_query'),answers.get('second_queries'),answers.get('terminal_actions'))
    good=not e['violations']
    best=enumerate_reference(case['information_parameters'])[0][0]
    return {'action_feasible':good,'violations':e['violations'],'objective':float(e['objective']) if good else None,'decision_regret':float(e['objective']-best) if good else None,'avoidable_cost':None,'objective_units':'negative prior expected net payoff','rejection_semantics':'Infeasible hypothetical plans are rejected; no state mutation or credit is awarded.'}


if __name__=='__main__':
    import json
    print(json.dumps([{'id':c['id'],'role':c['diagnostic_role'],'answers':c['stages'][0]['answers'],'two_query_gain':float(backward_reference(c['information_parameters'],1)[0]-backward_reference(c['information_parameters'])[0])} for c in build_cases()],indent=2))
