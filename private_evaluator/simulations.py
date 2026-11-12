"""Deterministic simulated books. Never sends orders to an external service."""
from copy import deepcopy
from itertools import product


def initial(cid):
    if cid=='c207':return {'positions':[6,4,0],'cash':18.,'receivable':0.,'unpaid':0.,'stage':1}
    if cid=='c212':return {'inventory':40,'cash':7000,'live':60,'price':110,'cancel_pending':False,'order_id':'O1','target':80,'stage':1,'bust_eligible':0}
    raise ValueError('Unknown simulated case')


def portfolio_terms(n):
    return {'sell_prices':[[10,10],[8,9],[9,9],[9,10]][n-1],'liability':[20,12,8,0][n-1],
            'reserve':10,'risk_cap':[35,24,16,10][n-1],'hedge_buy':5,'hedge_sell':4,'hedge_cap':3,
            'stress_loss_per_lot':[[8,5,-6],[-2,7,-2]],'max_action_lots':6}


def portfolio_trade(state,action):
    if not isinstance(action,list) or len(action)!=3 or any(type(x) is not int for x in action):return None
    a,b,h=action;pos=state['positions'];t=portfolio_terms(state['stage'])
    if not (0<=a<=pos[0] and 0<=b<=pos[1] and 0<=pos[2]+h<=3):return None
    fees=.2*a*a+.1*b*b+sum(x!=0 for x in action)
    positions=[pos[0]-a,pos[1]-b,pos[2]+h]
    cash=state['cash']+a*t['sell_prices'][0]-5*max(h,0)+4*max(-h,0)-fees-t['liability']-state['unpaid']
    if cash< -1e-8:return None
    stress=max(0,*[sum(x*y for x,y in zip(positions,row)) for row in t['stress_loss_per_lot']])
    loss=2*a+b+2*max(h,0)+max(-h,0)+fees
    return {'trade_lots':action,'post_cash':cash,'post_positions':positions,'worst_stress_loss':stress,'foregone_value':loss,'decision':'EXECUTE'},b*t['sell_prices'][1]


def portfolio_gold(state):
    t=portfolio_terms(state['stage']);choices=[]
    for a,b,h in product(range(state['positions'][0]+1),range(state['positions'][1]+1),range(-state['positions'][2],4-state['positions'][2])):
        v=portfolio_trade(state,[a,b,h])
        if v and v[0]['post_cash']>=10-1e-8 and v[0]['worst_stress_loss']<=t['risk_cap']+1e-8:choices.append(v[0])
    if not choices:
        stress=max(0,*[sum(x*y for x,y in zip(state['positions'],row)) for row in t['stress_loss_per_lot']])
        return [{'trade_lots':[0,0,0],'post_cash':state['cash'],'post_positions':state['positions'][:],'worst_stress_loss':stress,'foregone_value':0,'decision':'NO_FEASIBLE_PLAN'}]
    best=min(c['foregone_value'] for c in choices)
    return sorted([c for c in choices if abs(c['foregone_value']-best)<1e-8],key=lambda c:c['trade_lots'])


def execution_gold(state):
    gap=state['target']-state['inventory'];live=state['live'];price=100 if state['stage']<4 else 105
    if live:action='CANCEL' if state['inventory']+live>state['target'] else 'WAIT';quantity=0
    elif gap<0:
        quantity=(-gap)//5*5;action='SELL' if quantity else 'WAIT'
    else:
        quantity=min(max(gap,0)//5,max(0,int((state['cash']-1000)//(5*price))))*5
        action='BUY' if quantity else 'WAIT'
    projected=state['inventory']+(quantity if action=='BUY' else -quantity if action=='SELL' else 0)
    commitment=live*state['price']+(quantity*price if action=='BUY' else 0)
    return {'action':action,'quantity':quantity,'confirmed_inventory':state['inventory'],'live_buy_quantity':live,
            'cash_balance':state['cash'],'cash_after_commitments':state['cash']-commitment+(quantity*(price-1) if action=='SELL' else 0),
            'inventory_after_new_order_fill':projected,'target_shortfall_after_new_order':state['target']-projected}


def commit(cid,state,answers):
    """Commit a declared simulated action once; rejected actions never change the book."""
    s=deepcopy(state)
    if cid=='c207':
        if answers.get('decision')=='NO_FEASIBLE_PLAN':
            s['unpaid']+=portfolio_terms(s['stage'])['liability'];return s,{'status':'no_trade','unpaid_liabilities':s['unpaid']}
        trade=portfolio_trade(s,answers.get('trade_lots'))
        if trade is None or answers.get('decision')!='EXECUTE':
            s['unpaid']+=portfolio_terms(s['stage'])['liability'];return s,{'status':'rejected','reason':'Invalid lots, holdings or cash for liability payment','unpaid_liabilities':s['unpaid']}
        a,receivable=trade;s.update(positions=a['post_positions'],cash=a['post_cash'],receivable=s['receivable']+receivable,unpaid=0.)
        return s,{'status':'executed','positions':s['positions'],'cash':s['cash'],'receivable':s['receivable']}
    action=answers.get('action');qty=answers.get('quantity');price=100 if s['stage']<4 else 105
    if action not in ('WAIT','CANCEL','BUY','SELL') or type(qty) is not int or qty<0 or qty%5:return s,{'status':'rejected','reason':'Invalid action or board lot'}
    if action in ('WAIT','CANCEL') and qty:return s,{'status':'rejected','reason':'No quantity with wait/cancel'}
    if action=='CANCEL':
        if not s['live'] or s['cancel_pending']:return s,{'status':'rejected','reason':'No cancellable live order'}
        s['cancel_pending']=True
    elif action=='BUY':
        if s['live'] or qty<=0 or qty*price>s['cash']-1000:return s,{'status':'rejected','reason':'Outstanding order or insufficient cash'}
        s.update(live=qty,price=price,order_id='N'+str(s['stage']))
        if s['stage']==4:s['inventory']+=qty;s['cash']-=qty*price;s['live']=0
    elif action=='SELL':
        if s['live'] or qty<=0 or qty>s['inventory']:return s,{'status':'rejected','reason':'Outstanding order or insufficient inventory'}
        s['inventory']-=qty;s['cash']+=qty*(price-1)
    return s,{'status':'accepted','inventory':s['inventory'],'cash':s['cash'],'live':s['live'],'cancel_pending':s['cancel_pending']}


def advance(cid,state):
    s=deepcopy(state);n=s['stage']+1
    if n>4:return s
    if cid=='c207':
        s['cash']+=s['receivable'];s['receivable']=0
        if n==2:s['cash']-=2*s['positions'][2]
    else:
        if n==2:
            fill=min(s['live'],10 if s['cancel_pending'] else 30);s['bust_eligible']=min(fill,10)
        elif n==3:fill=min(s['live'],15)
        else:fill=min(s['live'],5)
        s['inventory']+=fill;s['cash']-=fill*s['price'];s['live']-=fill
        if s['cancel_pending']:s['live']=0;s['cancel_pending']=False
        if n==3:s['target']=65
        if n==4:
            bust=s['bust_eligible'];s['inventory']-=bust;s['cash']+=110*bust;s['target']=75
    s['stage']=n;return s


def view(case,n,state):
    t=deepcopy(case['stages'][n-1]);s=deepcopy(state)
    t['inputs']={'current_book':s,'terms':portfolio_terms(n)} if case['id']=='c207' else {'current_book':{k:v for k,v in s.items() if k!='bust_eligible'},'current_new_buy_price':100 if n<4 else 105,'sell_price':99 if n<4 else 104,'cash_reserve':1000,'board_lot':5}
    values=portfolio_gold(s) if case['id']=='c207' else [execution_gold(s)]
    t['answers']=values[0];t['alternative_answers']=values[1:]
    return t


def terminal_ok(cid,state):
    if cid=='c207':
        t=portfolio_terms(4);risk=max(0,*[sum(x*y for x,y in zip(state['positions'],r)) for r in t['stress_loss_per_lot']])
        return state['cash']>=10-1e-8 and risk<=10+1e-8 and state['unpaid']==0
    return state['inventory']==state['target'] and state['live']==0 and state['cash']>=1000
