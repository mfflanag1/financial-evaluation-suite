"""Finite synthetic economic worlds with observable action-dependent recovery."""
from copy import deepcopy
from itertools import product
from functools import lru_cache
import math

IDS={'c219','c220','c221','c222','c223','c224'}

def initial(cid):
    base={'stage':1,'rejected':0,'total_cost':0.0}
    data={
      'c219':dict(positions=[4,3],hedges=0,cash=19.,receivable=0.,arrears=0.),
      'c220':dict(allocations=[0,0,0],capital=12.,spent=0.),
      'c221':dict(hedge=0,cash=18.,spent=0.),
      'c222':dict(a=16.,b=17.,central=3.,debt=0.,arrears=0.,pending=0.),
      'c223':dict(observations={},budget=9.,decision='OPEN',realized_payoff=0.),
      'c224':dict(inventory=12,cash=1000.,pending=0.,live=8,price=40.,cancel=False,target=15,bust_units=0)}
    if cid not in data:raise ValueError(cid)
    return dict(base,**data[cid])

def terms(cid,n):
    if cid=='c219':return dict(prices=[[9,8],[8,9],[9,7],[10,8]][n-1],due=[18,16,15,8][n-1],reserve=4,risk_cap=[26,22,15,9][n-1],stress=[[8,3,-6],[-2,9,-3]],hedge_price=4,hedge_salvage=2,lot_caps=[4,3],joint_impact=.4)
    if cid=='c220':
        # Sample selection evidence changes admissibility; returns are estimates, not realized P&L.
        return dict(gross_edge=[[3.4,2.8,1.4],[3.0,2.6,1.5],[2.9,2.7,1.7],[2.6,2.5,1.8]][n-1],p=[.0007,.003,.021],trials=[8,8,1],leakage=[[False,False,False],[True,False,False],[True,False,False],[True,False,False]][n-1],regime_ok=[[True,True,True],[True,True,True],[True,False,True],[True,False,True]][n-1],alpha=.05,unit_cost=[.8,.6,.2],turnover_cost=.3,capital_per_unit=2,max_units=3)
    if cid=='c221':return dict(exposure=[7,8,6,5][n-1],scenario_slopes=[[[-4,7],[-1,3]],[[-5,8],[-2,4]],[[-2,5],[0,2]],[[-1,4],[1,3]]][n-1],hedge_loadings=[3,-2],fee=1.,limit=4,max_worst_loss=[43,38,27,20][n-1])
    if cid=='c222':return dict(a_floor=[8,8,6,5][n-1],b_floor=[10,10,8,7][n-1],due=[10,8,10,10][n-1],a_haircut=.2,b_haircut=.1,borrow_cap=8,borrow_fee=.25,central_floor=1,a_unit=2,b_unit=2,next_due=[8,10,10,0][n-1],next_inflow=[5,6,3,0][n-1])
    if cid=='c223':return dict(worlds=[{'a':0,'b':0,'probability':.3,'payoff':100},{'a':0,'b':1,'probability':.2,'payoff':-60},{'a':1,'b':0,'probability':.2,'payoff':-60},{'a':1,'b':1,'probability':.3,'payoff':20}],costs={'A':3.,'B':4.},rounds_left=5-n)
    return dict(buy_price=[40,42,39,41][n-1],sell_price=[39,41,38,40][n-1],reserve=150,fee=2,lot_limit=10,terminal=n==4)

def public_state(cid,s):
    return {k:deepcopy(v) for k,v in s.items() if k not in ('bust_units',)}

def stress(s,t):return max(0,*[s['positions'][0]*r[0]+s['positions'][1]*r[1]+s['hedges']*r[2] for r in t['stress']])

def _admissible(t):
    return [not t['leakage'][i] and t['regime_ok'][i] and t['p'][i]*t['trials'][i]<=t['alpha'] for i in range(3)]

def _posterior(obs):
    worlds=terms('c223',1)['worlds'];keep=[w for w in worlds if all(w[k.lower()]==v for k,v in obs.items())];z=sum(w['probability'] for w in keep)
    return [(w,w['probability']/z) for w in keep]

@lru_cache(None)
def _signal_values(observations,budget,rounds):
    obs=dict(observations);post=_posterior(obs);mean=sum(w['payoff']*p for w,p in post)
    vals={'INVEST':mean,'ABSTAIN':0.}
    if rounds>1:
        for name,cost in terms('c223',1)['costs'].items():
            if name in obs or cost>budget:continue
            expected=-cost
            for bit in (0,1):
                pr=sum(p for w,p in post if w[name.lower()]==bit)
                if pr:
                    nxt=tuple(sorted(dict(obs,**{name:bit}).items()))
                    expected+=pr*max(dict(_signal_values(nxt,budget-cost,rounds-1)).values())
            vals['READ_'+name]=expected
    return tuple(sorted(vals.items()))

def actions(cid,s):
    if cid=='c219':return [list(a) for a in product(range(s['positions'][0]+1),range(s['positions'][1]+1),range(-s['hedges'],5-s['hedges']))]
    if cid=='c220':return [list(a) for a in product(range(4),repeat=3)]
    if cid=='c221':return list(range(5))
    if cid=='c222':return [list(a) for a in product(range(9),range(9),range(9))]
    if cid=='c223':return ['WAIT'] if s['decision']!='OPEN' else list(dict(_signal_values(tuple(sorted(s['observations'].items())),s['budget'],5-s['stage'])))
    ans=[['WAIT',0]]
    if s['live'] and not s['cancel']:ans.append(['CANCEL',0])
    if not s['live']:
        ans += [['BUY',q] for q in range(1,11)]+[['SELL',q] for q in range(1,min(10,s['inventory'])+1)]
    return ans

def _exact_integer_quantities(value):
    """Align executable integral quantities with numeric JSON grading, never round."""
    if type(value) is float and math.isfinite(value) and value.is_integer():return int(value)
    if isinstance(value,list):return [_exact_integer_quantities(v) for v in value]
    return value

def evaluate(cid,s,action):
    """Return legal execution state and its private instantaneous economic assessment."""
    action=_exact_integer_quantities(action)
    t=terms(cid,s['stage']);z=deepcopy(s);cost=0.;violation=0.;obj=0.
    if cid=='c219':
        if not isinstance(action,list) or len(action)!=3 or any(type(v)is not int for v in action):return None
        a,b,h=action
        if not(0<=a<=s['positions'][0] and 0<=b<=s['positions'][1] and 0<=s['hedges']+h<=4):return None
        cost=.4*(a+b)**2+.3*h*h+sum(v!=0 for v in action)
        z.update(positions=[s['positions'][0]-a,s['positions'][1]-b],hedges=s['hedges']+h)
        z['cash']+=a*t['prices'][0]-4*max(h,0)+2*max(-h,0)-cost
        z['receivable']+=b*t['prices'][1]
        owed=t['due']+s['arrears'];paid=min(max(z['cash'],0),owed);z['cash']-=paid;z['arrears']=owed-paid
        if z['cash']<0:return None
        risk=stress(z,t);violation=max(0,t['reserve']-z['cash'])+max(0,risk-t['risk_cap'])+z['arrears']
        obj=cost+2*a+1.5*b+.7*abs(h)+1000*violation
        detail=[z['cash'],risk,z['arrears']]
    elif cid=='c220':
        if not isinstance(action,list) or len(action)!=3 or any(type(v)is not int or v<0 or v>3 for v in action):return None
        turnover=sum(abs(a-b) for a,b in zip(action,s['allocations']));cost=.3*turnover
        if 2*sum(action)+cost>s['capital']+1e-9:return None
        valid=_admissible(t);violation=sum(a for a,v in zip(action,valid) if not v)
        expected=sum(a*(e-c) for a,e,c in zip(action,t['gross_edge'],t['unit_cost']))-cost
        obj=-expected+1000*violation
        z.update(allocations=action[:],capital=s['capital']-cost,spent=s['spent']+cost)
        detail=[expected,2*sum(action),[int(v) for v in valid]]
    elif cid=='c221':
        if type(action)is not int or not 0<=action<=4:return None
        cost=abs(action-s['hedge'])*t['fee']
        if cost>s['cash']:return None
        # Each row is an interval of slopes; evaluate every endpoint, never midpoint.
        losses=[t['exposure']*end-action*load for interval,load in zip(t['scenario_slopes'],t['hedge_loadings']) for end in interval]
        lo=min(losses);hi=max(losses);violation=max(0,hi-t['max_worst_loss']);obj=hi+cost+1000*violation
        z.update(hedge=action,cash=s['cash']-cost,spent=s['spent']+cost);detail=[lo,hi,z['cash']]
    elif cid=='c222':
        if not isinstance(action,list) or len(action)!=3 or any(type(v)is not int or v<0 for v in action):return None
        a,b,borrow=action
        if a>8 or b>8 or borrow>8 or 2*a>max(0,s['a']-t['a_floor']) or 2*b>max(0,s['b']-t['b_floor']) or s['debt']+borrow>8:return None
        # A collateral releases same day; B sale proceeds remain unsettled until next stage.
        z['a']-=2*a;z['b']-=2*b;z['central']+=1.6*a+borrow-.25*borrow;z['pending']+=1.8*b;z['debt']+=borrow
        cost=.4*a+.2*b+.25*borrow
        due=t['due']+s['arrears'];paid=min(max(z['central'],0),due);z['central']-=paid;z['arrears']=due-paid
        violation=max(0,t['a_floor']-z['a'])+max(0,t['b_floor']-z['b'])+max(0,1-z['central'])+z['arrears']
        violation+=max(0,t['next_due']+1-t['next_inflow']-z['central']-z['pending']) if s['stage']<4 else 0
        obj=1000*violation+cost+.1*z['debt'];detail=[z['central'],z['arrears'],z['pending']]
    elif cid=='c223':
        if action not in actions(cid,s):return None
        mean=sum(w['payoff']*p for w,p in _posterior(s['observations']))
        vals={'WAIT':0.} if s['decision']!='OPEN' else dict(_signal_values(tuple(sorted(s['observations'].items())),s['budget'],5-s['stage']))
        obj=-vals[action]
        if action.startswith('READ_'):
            name=action[-1];cost=t['costs'][name];z['budget']-=cost
            # Fixed hidden realized world (A=0,B=1); only purchased signals become visible.
            z['observations'][name]=0 if name=='A' else 1
        elif action in ('INVEST','ABSTAIN'):
            z['decision']=action;z['realized_payoff']=-60. if action=='INVEST' else 0.
        detail=[mean,vals[action],s['budget']-cost]
    else:
        if not isinstance(action,list) or len(action)!=2:return None
        verb,q=action
        if type(q)is not int or action not in actions(cid,s):return None
        if verb=='CANCEL':z['cancel']=True
        if verb in ('BUY','SELL'):
            cost=2.
            if verb=='BUY':
                if q*t['buy_price']+2>s['cash']-150:return None
                z.update(live=q,price=t['buy_price']);z['cash']-=2
                if t['terminal']:z['inventory']+=q;z['cash']-=q*t['buy_price'];z['live']=0
            else:z['inventory']-=q;z['pending']+=q*t['sell_price']-2
        lo=z['inventory'];hi=z['inventory']+z['live'];worst=max(abs(lo-z['target']),abs(hi-z['target']))
        violation=max(0,150-(z['cash']-z['live']*z['price']))
        # Pending cancellation caps future fills at two lots; acknowledgement occurs only at advance.
        if z['cancel']:hi=z['inventory']+min(2,z['live']);worst=max(abs(lo-z['target']),abs(hi-z['target']))
        obj=10*worst+cost+.1*z['live']+1000*violation;detail=[z['cash']-z['live']*z['price'],worst,z['pending']]
    z['total_cost']+=cost
    return z,{'objective':round(obj,10),'constraint_violation':round(violation,10),'cost':round(cost,10),'detail':detail}

def gold(cid,s):
    rows=[]
    for a in actions(cid,s):
        out=evaluate(cid,s,a)
        if out:rows.append((out[1]['objective'],a,out[1]))
    if not rows:raise AssertionError('At least one legal action required')
    best=min(r[0] for r in rows)
    return [answer(cid,s,a,m) for o,a,m in rows if abs(o-best)<1e-8]

def answer(cid,s,action,m):
    return {'action':deepcopy(action),'objective':m['objective'],'constraint_violation':m['constraint_violation'],'transaction_cost':m['cost'],'diagnostics':deepcopy(m['detail']),'state_round':s['stage']}

def economic_outcome(cid,state,answers):
    actual=evaluate(cid,state,answers.get('action'));choices=gold(cid,state);opt=choices[0]['objective']
    scope='conditional_continuation' if cid=='c223' else 'current_round_objective'
    if actual is None:return {'accepted':False,'economic_feasible':False,'objective':None,'decision_regret':None,'penalized_objective_gap':None,'constraint_violation':None,'excess_transaction_cost_vs_objective_optimum':None,'recovery':False,'constraint_violations':['nonexecutable_action'],'regret_scope':scope}
    z,m=actual;gap=max(0.,round(m['objective']-opt,10));feasible=m['constraint_violation']==0
    optimal_cost=min(a['transaction_cost'] for a in choices)
    return {'accepted':True,'economic_feasible':feasible,'objective':m['objective'],'decision_regret':gap if feasible and choices[0]['constraint_violation']==0 else None,'penalized_objective_gap':gap,'constraint_violation':m['constraint_violation'],'excess_transaction_cost_vs_objective_optimum':max(0.,m['cost']-optimal_cost),'recovery':feasible and (state.get('arrears',0)>0 or state['rejected']>0),'constraint_violations':[] if feasible else ['reported_policy_deficit'],'regret_scope':scope}

def commit(cid,state,answers):
    out=evaluate(cid,state,answers.get('action'))
    if out is None:
        z=deepcopy(state);z['rejected']+=1
        # A rejected operational decision still leaves the current payment due.
        if cid in ('c219','c222'):z['arrears']+=terms(cid,z['stage'])['due']
        return z,{'status':'rejected','reason':'Action is not executable under public mechanical rules','actual_state':public_state(cid,z)}
    z,_=out
    return z,{'status':'executed','actual_state':public_state(cid,z)}

def advance(cid,state):
    z=deepcopy(state);n=z['stage']+1
    if n>4:return z
    if cid=='c219':
        z['cash']+=z['receivable'];z['receivable']=0.
        if n==3:z['cash']+=12.  # New external cash receipt disclosed only at stage three.
    elif cid=='c220':
        # Capital cashflows are stipulated scenario events, not edge-estimate realization.
        z['capital']+=[0,2.,-1.,1.][n-1]
    elif cid=='c221':z['cash']+=[0,0.,2.,0.][n-1]
    elif cid=='c222':
        z['central']+=z['pending']+[0,5.,6.,3.][n-1];z['pending']=0.
        if n==3:z['a']+=8.;z['b']+=6.
    elif cid=='c224':
        z['cash']+=z['pending'];z['pending']=0.
        fill=min(z['live'],2 if z['cancel'] else [0,4,3,2][n-1]);z['inventory']+=fill;z['cash']-=fill*z['price'];z['live']-=fill
        if n==2:z['bust_units']=min(fill,2)
        if z['cancel']:z['live']=0;z['cancel']=False
        if n==3:z['target']=13
        if n==4:
            z['inventory']-=z['bust_units'];z['cash']+=40*z['bust_units'];z['target']=16
    z['stage']=n;return z

def view(case,n,state):
    t=deepcopy(case['stages'][n-1]);t['inputs']={'current_state':public_state(case['id'],state),'current_terms':terms(case['id'],n)}
    choices=gold(case['id'],state);t['answers']=choices[0];t['alternative_answers']=choices[1:]
    return t

def terminal_ok(cid,state):
    if cid=='c219':return state['arrears']==0 and state['cash']>=4 and stress(state,terms(cid,4))<=9
    if cid=='c220':return all(not a or v for a,v in zip(state['allocations'],_admissible(terms(cid,4))))
    if cid=='c221':return evaluate(cid,state,state['hedge'])[1]['constraint_violation']==0
    if cid=='c222':return state['arrears']==0 and state['central']>=1 and state['a']>=5 and state['b']>=7
    if cid=='c223':return state['decision']!='OPEN' and state['budget']>=0
    return state['inventory']==state['target'] and not state['live'] and state['cash']>=150
