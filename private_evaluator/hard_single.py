"""Synthetic advanced financial decisions; all economic inputs are public case facts.

The outcome scorer recomputes policy economics, ignoring self-reported objectives.
Holdout constructors and mathematical tests do not constitute solver evaluation.
"""
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
from math import sqrt, isfinite
from case_contract import validate


def _doc(i,t,s): return {'id':i,'title':t,'text':s,'release_stage':1}
def _case(cid,title,family,split,docs,ans,desc,prompt,solution,checks,wrong,routes=None):
    ids=[d['id'] for d in docs]
    stage={'prompt':prompt,'inputs':{'scope':'All amounts and rules are synthetic and fully specified in the current documents.'},'answers':ans,'descriptions':desc,
           'evidence_routes':routes or {k:[ids] for k in ans},'allowed_evidence':{k:ids for k in ans},
           'solution':solution,'checks':[{'name':n,'passed':bool(b)} for n,b in checks], 'wrong_answers':wrong}
    return validate({'id':cid,'title':title,'family':family,'split':split,'mode':'single','documents':docs,'oracle_document_ids':ids,'stages':[stage]})
def _wrong(label,a,reason):return {'label':label,'answers':a,'reason':reason}
def _whole_units(v):
    """Normalize exact whole-unit JSON numbers; never round or accept booleans."""
    if not isinstance(v,list):return v
    return [int(x) if type(x) is float and isfinite(x) and x.is_integer() else x for x in v]
def _intvec(v,n,lo,hi):return isinstance(v,list) and len(v)==n and all(type(x) is int and lo<=x<=hi for x in v)
def _bad(msg):return {'action_feasible':False,'violations':[msg],'objective':None,'decision_regret':None,'avoidable_cost':None}
def _out(v,obj,opt,cost=None):return {'action_feasible':not v,'violations':v,'objective':float(obj) if not v else None,'decision_regret':float(obj-opt) if not v else None,'avoidable_cost':None if v or cost is None else float(cost)}


def portfolio_economics(a):
    a=_whole_units(a)
    if not _intvec(a,5,0,8) or a[3]>5 or a[4]>4:return _bad('Expected [A,B,C,hedge,draw] integers within the stated domain.')
    h=a[:3];hedge,draw=a[3:];old=[7,6,4];q=[h[i]-old[i] for i in range(3)]
    # Aggregate execution per issuer: first two sale units get full price; depth thereafter costs 2 each.
    sale=[max(-x,0) for x in q];buy=[max(x,0) for x in q]
    impact=sum(2*max(x-2,0) for x in sale)+sum(F(x*x,2) for x in buy)
    tickets=sum(x!=0 for x in q)+(hedge>0)+(draw>0)
    cost=impact+tickets+F(hedge,2)+F(draw,2)
    early=20+10*sale[0]-sum(10*x for x in buy)-cost-2*hedge+5*draw-40
    # The same positions and financing must survive each joint market/liquidity scenario.
    losses=[5*h[0]+2*h[1]-h[2]-6*hedge, -h[0]+5*h[1]+3*h[2]-2*hedge]
    collateral=[2*h[0]+h[2], h[0]+2*h[2]]
    final=[early+10*(sale[1]+sale[2])+2*hedge-5*draw-max(0,loss)-3 for loss in losses]
    violations=[]
    if sum(h)<10:violations.append('minimum invested units')
    if h[0]+h[1]>11:violations.append('issuer-group cap')
    if abs(h[0]-h[2])>3:violations.append('factor band')
    if sum(abs(x) for x in q)+hedge>13:violations.append('turnover limit')
    if early<0:violations.append('early cash deficit')
    if any(5*draw>c for c in collateral):violations.append('scenario collateral borrowing base')
    if any(x<0 for x in final):violations.append('stressed final cash deficit')
    objective=cost+F(3,10)*max(losses)+F(1,5)*(17-sum(h))
    return {'violations':violations,'cost':cost,'early':early,'final':final,'losses':losses,'objective':objective}
@lru_cache(None)
def portfolio_reference():
    rows=[]
    for a in product(range(9),range(9),range(9),range(6),range(5)):
        e=portfolio_economics(list(a))
        if not e['violations']:rows.append((e['objective'],a,e))
    rows.sort(key=lambda x:(x[0],x[1]));return rows

def _portfolio():
    rows=portfolio_reference();o,a,e=rows[0]
    ans={'action':list(a),'implementation_cost':float(e['cost']),'early_cash':float(e['early']),'scenario_cash':list(map(float,e['final'])),'scenario_losses':e['losses'],'objective':float(o),'next_best_objective':float(rows[1][0])}
    docs=[_doc('rp_book','Opening holdings and executable depth','Issuer order is [A,B,C]. Opening units are [7,6,4], each marked at10; cash20. Final positions must be integers0..8. One net trade per issuer; no shorting or round trips. A sale of s units realizes 10s minus 2*max(s-2,0). A buy of b units costs 10b plus b^2/2. Every nonzero issuer trade pays one ticket of1. All execution impact and all tickets are paid at the early checkpoint, so sale receipts at each settlement are booked at gross10 per unit; do not subtract impact twice.'),
          _doc('rp_hedge','Funding and hedge menu','Choose integer hedge units H in0..5 and integer funding units D in0..4. H costs premium H/2 and one ticket if H>0, posts refundable collateral2H at the early checkpoint, and releases that2H before final stress payments. D credits5D cash early, repays5D at final, and costs fee D/2 plus one ticket if D>0, paid early. Funding is secured by final A,C holdings: its principal must be no more than 2A+C in scenario1 and A+2C in scenario2. These are simultaneous borrowing-base requirements, not alternative facilities.'),
          _doc('rp_limits','Common policy under joint stress','Use one common [A,B,C,H,D] in both scenarios. Final A+B+C must be at least10; A+B at most11; |A-C| at most3. Sum of absolute issuer net trades plus H must be at most13. Scenario cash losses, possibly negative, are L1=5A+2B-C-6H and L2=-A+5B+3C-2H. Final stress cash payment is max(0,Ls); a negative loss is not a cash receipt. Do not probability-weight scenarios or choose different trades after learning the scenario.'),
          _doc('rp_cash','Timeline and robust economic mandate','At the early checkpoint A-sale gross receipts arrive; all issuer purchases, all implementation cost, and hedge collateral are paid; funding credits arrive; redemption40 is paid. B,C-sale gross receipts arrive only afterward. Final hedge collateral is released, funding principal repaid, stress payment made, and operating charge3 paid. Cash must be nonnegative at both checkpoints in both scenarios. Within a checkpoint receipts can precede payments. Implementation cost K includes depth impact, buy impact, tickets, hedge premium and funding fee, excluding principal, collateral, redemption and operating charge. Minimize J=K+0.3*max(L1,L2)+0.2*(17-A-B-C). Break exact ties by lexicographically smallest [A,B,C,H,D]. Next-best objective means the second distinct feasible action in ascending (J,lexicographic action) order, even if tied.')]
    return _case('c213','Common repair under joint market and funding stress','portfolio_repair','development',docs,ans,
        {'action':'Five integer quantities [final A,final B,final C,H,D].','implementation_cost':'Number K, cash units.','early_cash':'Number, cash after the early checkpoint.','scenario_cash':'Two numbers, final unrestricted cash in scenario order1,2.','scenario_losses':'Two signed numbers [L1,L2].','objective':'Number, minimum J.','next_best_objective':'Number, J of the next feasible action in the defined ordering.'},
        'Choose the globally optimal common repair and funding action under both joint stress scenarios. Respect executable depth, timing, all portfolio restrictions and scenario borrowing bases. Report its cash path, economic objective, and the next-best feasible objective under the documented ordering.',
        'Exhaustive finite-domain certification over 9^3*6*5 common actions applies every borrowing-base and cash constraint before ranking. The best action and next feasible objective are computed from exact rational cash and cost arithmetic; scenario-specific reoptimization is prohibited.',
        [('feasible domain nonempty',len(rows)>1),('unique optimizer',rows[1][0]>o)],
        [_wrong('ignore funding collateral',{'action':[3,4,3,5,4]},'A draw of20 exceeds both scenario borrowing bases9, even if a pooled cash calculation appears funded.'),_wrong('report a low cost without implementing it',{'objective':0},'Reported objective does not change the economic action or its true cost.')])


# Six independent eligible blocks plus two audit exclusions. Net residual vectors used below
# are also independently reconstructed from source gross returns by the tests.
_Q_NET=[[9,5,6],[8,4,5],[7,6,4],[6,-2,2],[5,-3,1],[-4,-4,-1]]
_Q_IDS=['b1','b2','b4','b5','b7','b8']
def quant_reference():
    n=_Q_NET; specs=[(j,k) for j in range(3) for k in (3,6)]
    def stat(j,k,signs):return sum(signs[i]*n[i][j] for i in range(k))/sqrt(sum(n[i][j]**2 for i in range(k)))
    obs={(j,k):stat(j,k,[1]*6) for j,k in specs};sel=max(specs,key=lambda jk:(obs[jk],-jk[0],jk[1]))
    threshold=obs[sel]
    count=sum(max(stat(j,k,s) for j,k in specs)>=threshold-1e-12 for s in product((-1,1),repeat=6))
    raw=sum(stat(*sel,s)>=threshold-1e-12 for s in product((-1,1),repeat=6))
    return sel,threshold,F(count,64),F(raw,64),count

def _quant():
    sel,t,p,raw,count=quant_reference();j,k=sel
    ans={'eligible_blocks':_Q_IDS,'selected_policy':['A','B','C'][j],'selected_checkpoint':k,'selected_statistic':t,'unadjusted_p':float(raw),'selection_adjusted_p':float(p),'late_regime_mean':float(F(sum(r[j] for r in _Q_NET[3:]),3)),'deploy':False}
    # Gross = residual + beta*market + spread + (units^2)/2.
    markets=[2,-1,3,-2,1,-3];units=[1,2,1];betas=[1,0,-1];spread=F(1,2)
    lines=[]
    for idx,(bid,row,m) in enumerate(zip(_Q_IDS,_Q_NET,markets)):
        day=10+idx*5;gross=[F(row[z])+betas[z]*m+spread+F(units[z]**2,2) for z in range(3)]
        lines.append(f'{bid} | {day} | {day-1} | {day-3} | {m} | '+','.join(str(float(x)) for x in gross))
    lines.insert(2,'b3 | 17 | 18 | 14 | 4 | 40,35,30')
    lines.insert(5,'b6 | 27 | 26 | 26 | -4 | 30,25,20')
    docs=[_doc('qt_tape','Timestamped independent-block audit','Columns are id | decision day | first feature publication day | latest training-label end day | market bps | gross bps [A,B,C]. Blocks have disjoint future P&L windows and independent sign shocks if eligible. No rows beyond this complete tape exist. Rows are ordered by numeric block id; publication is effective at start of its day.\n'+'\n'.join(lines)),
          _doc('qt_cost','Frozen residual-return transformation','Eligible blocks require feature publication no later than decision, and latest training-label end no later than decision minus2. If a block fails, remove it for all policies before forming either checkpoint. For each eligible policy/block subtract spread0.5 bps plus half the square of traded units; traded units [A,B,C]=[1,2,1]. Then subtract beta times market bps using prespecified betas[1,0,-1]. These betas were frozen before this sample. The resulting residual net returns are the inputs for inference and regime diagnostics, not gross returns.'),
          _doc('qt_registry','Complete selection universe and joint null','Exactly policies A,B,C were inspected, each at two checkpoints: first3 eligible blocks and all6 eligible blocks. All six scores were inspected including archived scores. At checkpoint k, score is sum(x_i)/sqrt(sum(x_i^2)). The published candidate is the highest of all six scores; exact ties use alphabetical policy then longer checkpoint. Under the stipulated complete null, conditional on the absolute block vectors, all2^6 vectors of independent signs on whole blocks are equally likely. A block sign multiplies all three policies together; policy columns must not get independent signs. The selection-adjusted one-sided p is the fraction of these signs for which the maximum of all six recomputed scores is at least the observed selected score. The unadjusted p compares only the selected policy/checkpoint statistic. Include equality and all signs, with no Monte Carlo or plus-one correction.'),
          _doc('qt_gate','Deployment and stability rule','Deployment requires selection-adjusted p at most0.05, selected checkpoint equal to the full6 eligible blocks, and selected policy mean residual net return strictly positive separately in the first3 and last3 eligible blocks, with the late-regime mean at least half the early-regime mean. All gates bind. The late-regime mean always uses last3 eligible blocks, even if the selected checkpoint was first3. The regime diagnostic is an economic stability gate, not another p-value. Report the selected score without rounding it before comparisons.')]
    return _case('c214','Audit a correlated research search with regime gates','quant_validity','development',docs,ans,
        {'eligible_blocks':'Array of block ids in numeric order.','selected_policy':'String A, B or C.','selected_checkpoint':'Integer3 or6.','selected_statistic':'Number, selected standardized score.','unadjusted_p':'Number, exact selected-statistic sign-randomization probability.','selection_adjusted_p':'Number, exact search-adjusted probability.','late_regime_mean':'Number, selected policy late-regime residual net bps.','deploy':'Boolean under all deployment gates.'},
        'Determine whether the published research winner can be deployed. Reconstruct point-in-time eligible net residual returns, account for every inspected policy and checkpoint with their shared block dependence, and apply the independent regime gates. Return the selected candidate and both exact inference probabilities.',
        f'Exclude b3 for unavailable features and b6 for training-label leakage. Reconstruct residual rows {_Q_NET}. Evaluate all six registered scores, then enumerate64 shared block-sign vectors. Selected {sel}, observed score{t}, adjusted tail count{count}/64, raw p{raw}. Apply the full-checkpoint and regime gates independently of statistical significance.',
        [('eligible selection has six blocks',len(_Q_IDS)==6),('search correction cannot improve p',p>=raw)],
        [_wrong('use gross winning rows despite publication dates',{'eligible_blocks':['b1','b2','b3','b4','b5','b6','b7','b8']},'Blocks b3 and b6 fail point-in-time validity.'),_wrong('treat winning checkpoint as a fresh test',{'selection_adjusted_p':float(raw/2),'deploy':True},'All six inspected scores belong to the null maximum and the deployment gates also apply.')])


def valuation_economics(a):
    a=_whole_units(a)
    if not _intvec(a,3,0,2):return _bad('Expected integer [A bonds,B bonds,joint-default protection] in0..2.')
    A,B,H=a;cost=80*A+50*B+3*H
    pay=[40*A+20*B+30*H,40*A+100*B,100*A+20*B,100*A+100*B]
    loss=[F(cost-x) for x in pay]
    # Each state's probability is affine in q; check every tail-composition breakpoint.
    slopes=[1,-1,-1,1];intercepts=[F(0),F(1,4),F(1,2),F(1,4)]
    order=sorted(range(4),key=lambda i:(-loss[i],i));points={F(0),F(1,4)};s=0;b=F(0)
    for i in order:
        s+=slopes[i];b+=intercepts[i]
        if s:
            q=(F(1,5)-b)/s
            if 0<=q<=F(1,4):points.add(q)
    def tail(q):
        remaining=F(1,5);total=F(0)
        for i in order:
            mass=min(remaining,intercepts[i]+slopes[i]*q);total+=mass*loss[i];remaining-=mass
            if not remaining:break
        return total*5
    worst=max(tail(q) for q in points)
    means=[sum((intercepts[i]+slopes[i]*q)*(pay[i]-cost) for i in range(4)) for q in (F(0),F(1,4))]
    violations=[]
    if cost>200:violations.append('purchase budget')
    if worst>55:violations.append('robust expected shortfall limit')
    return {'violations':violations,'cost':cost,'loss':loss,'worst_tail':worst,'mean_interval':[min(means),max(means)],'points':sorted(points),'objective':-min(means)}
@lru_cache(None)
def valuation_reference():
    rows=[]
    for a in product(range(3),repeat=3):
        e=valuation_economics(list(a))
        if not e['violations']:rows.append((e['objective'],e['cost'],a,e))
    return sorted(rows,key=lambda r:r[:3])
def _valuation():
    o,c,a,e=valuation_reference()[0]
    ans={'action':list(a),'purchase_cost':c,'joint_default_probability_interval':[0,0.25],'expected_profit_interval':list(map(float,e['mean_interval'])),'worst_expected_shortfall':float(e['worst_tail']),'state_losses':list(map(float,e['loss'])),'minimum_expected_profit':float(-o)}
    docs=[_doc('va_marginals','Prices and identified marginal default risks','Each bond pays100 if its issuer survives. A recovery is40 and B recovery20. A default probability is exactly0.25 and B default probability exactly0.50. There is no evidence about their dependence. Every joint distribution with these marginals is admissible; no independence, calibration weighting, or preferred law is authorized. A bond costs80 and B bond50 now. Prices and recoveries are cash units; there is no discounting or financing.'),
          _doc('va_protection','Executable joint-default protection','A protection unit costs3 now and pays30 exactly when BOTH issuers default, otherwise0. It cannot default. Choose integers A,B,H each0..2; naked protection is permitted. Buy nothing is permitted. Total purchase expenditure may not exceed200. The four state order for losses is [both default,A only defaults,B only defaults,neither defaults]. Cash retained rather than spent earns0 and does not change profit.'),
          _doc('va_risk','Binding risk measure under underidentification','Loss is total purchase expenditure minus total terminal payoff, and may be negative. Expected shortfall means the mean loss in the worst20% probability mass, splitting a state proportionally at the cutoff if necessary. It is not a loss quantile and negative tail losses must not be truncated. For a feasible action, expected shortfall must be at most55 under EVERY admissible joint law. No marginal-default approximation or independence assumption may replace this requirement.'),
          _doc('va_mandate','Robust economic decision and tie-breaks','Among actions meeting the budget and robust expected-shortfall limit, maximize the minimum expected terminal profit across all admissible joint laws. Break ties by lower purchase expenditure, then lexicographically smaller [A,B,H]. Report the full attainable interval of expected profit for the chosen action and its largest expected shortfall across the ambiguity set. The joint default probability interval describes all admissible laws, whether or not they are worst for the selected action. Do not infer uniqueness of a valuation from the marginal probabilities.')]
    routes={k:[[d['id'] for d in docs]] for k in ans};routes['joint_default_probability_interval']=[['va_marginals']];routes['purchase_cost']=[['va_marginals','va_protection','va_risk','va_mandate']]
    return _case('c215','Portfolio choice under unidentified default dependence','valuation_choice','holdout',docs,ans,
        {'action':'Integer array [A bonds,B bonds,H protection units].','purchase_cost':'Number, upfront expenditure.','joint_default_probability_interval':'Array [lowest,highest] admissible joint default probability.','expected_profit_interval':'Array [lowest,highest] expected profit for the selected action.','worst_expected_shortfall':'Number, maximum worst20%-tail mean loss across admissible laws.','state_losses':'Four signed numbers in the documented state order.','minimum_expected_profit':'Number, robust objective achieved.'},
        'Choose the robustly optimal portfolio using only identified default information. Evaluate the risk constraint over the entire admissible dependence set and distinguish an expected-shortfall tail average from a quantile. Report the portfolio, identified ranges, state losses and economic objective.',
        'Write q=P(both defaults). The four state probabilities are [q,.25-q,.5-q,.25+q], q in[0,.25]. For each finite action, expected profit is affine in q. Its tail mean is piecewise affine, with breakpoints wherever cumulative worst-state mass equals .2, so endpoints and those breakpoints certify the entire ambiguity set. Rank only actions satisfying the budget and worst-tail cap.',
        [('dependence set nonempty',e['mean_interval'][0]<=e['mean_interval'][1]),('chosen action meets tail cap',e['worst_tail']<=55)],
        [_wrong('assume independent defaults are identified',{'joint_default_probability_interval':[0.125,0.125]},'The marginals only imply the Frechet interval0..0.25.'),_wrong('ignore the robust tail limit',{'action':[1,2,0]},'The superficially high mean-profit portfolio has an inadmissible joint-default tail.')],routes)


def funding_economics(a):
    a=_whole_units(a)
    if not _intvec(a,4,0,20) or a[0]>6 or a[1]>5 or a[2]>4:return _bad('Expected integer [A sold,B sold,C sold,draw] within the menu.')
    x,y,z,b=a;cap=max(0,min(15,3*(6-x)-2*y));haircut=8 if x>=3 else 0
    early=20+8*x+6*z+b-50-5*y-haircut
    final=early+12*y+30-b-F(b,2)-12
    risk=6*(6-x)+4*(5-y)+6*z
    cost=4*x+3*y+9*z+F(b,2)+haircut
    v=[]
    if b>cap:v.append('pledged collateral borrowing base')
    if early<0:v.append('early deadline deficit')
    if final<8:v.append('final reserve deficit')
    if risk>34:v.append('remaining stressed risk limit')
    return {'violations':v,'cap':cap,'haircut':haircut,'early':early,'final':final,'risk':risk,'objective':cost,'cost':cost}
@lru_cache(None)
def funding_reference():
    rows=[]
    for a in product(range(7),range(6),range(5),range(21)):
        e=funding_economics(list(a))
        if not e['violations']:rows.append((e['objective'],a,e))
    return sorted(rows,key=lambda r:(r[0],r[1]))
def _funding():
    rows=funding_reference();o,a,e=rows[0]
    ans={'action':list(a),'borrowing_base':e['cap'],'haircut_call':e['haircut'],'early_cash':float(e['early']),'final_cash':float(e['final']),'remaining_stress_risk':e['risk'],'economic_cost':float(o)}
    docs=[_doc('fu_book','Cross-strategy opening ledger','Treasury cash is20. Strategy assets available for sale are6 A,5 B,4 C units. A sells at8 per unit and C at6 with immediate early receipts; B sells at12 with receipts only at the final checkpoint. Sales are irrevocable whole units. Treasury owes50 early and12 final. A separate unrestricted30 inflow arrives final. No sale costs apply except the economic opportunity costs and calls separately specified.'),
          _doc('fu_cascade','Collateral and hedge feedback','Remaining A units secure the ONLY facility. After all chosen sales its principal cap is max(0,min(15,3*(6-A_sold)-2*B_sold)). Selling B closes a cross-strategy hedge and causes an additional early nonrefundable margin payment5 per B sold; that payment is already a funded loss and is NOT part of the opportunity-cost objective. If A_sold is at least3, concentration deterioration causes an additional early nonrefundable haircut call8, otherwise0. This call is included in the economic-cost objective. Selling C also removes a hedge; stressed remaining risk is6*(6-A_sold)+4*(5-B_sold)+6*C_sold and must not exceed34.'),
          _doc('fu_line','Facility timing and admissible actions','Choose a nonnegative integer facility draw no greater than20 and no greater than its post-sale borrowing base. Its principal is credited early, repaid final; a fee equal to half the draw is paid final. Early sale receipts and draw can be received before early payments; late sale receipts and the30 inflow arrive before final payments. No other borrowing, netting future receipts, staged sales, or second draw is permitted. Early cash must be nonnegative; final cash after all repayments and payments must be at least8.'),
          _doc('fu_objective','Liquidation mandate with full economic cost','Minimize J=4*A_sold+3*B_sold+9*C_sold+0.5*draw+haircut_call. The per-unit coefficients are supplied opportunity costs beyond the cash ledger; do not substitute sale proceeds for them or add the B margin payment again. Exact objective ties use lexicographically smaller [A_sold,B_sold,C_sold,draw]. All cash figures are common treasury cash units and no future recoveries occur within the horizon. Both deadline feasibility and remaining stress risk must hold before comparing economic costs.')]
    return _case('c216','Liquidation feedback into collateral and hedge funding','cross_strategy_funding','holdout',docs,ans,
        {'action':'Integer array [A sold,B sold,C sold,facility draw].','borrowing_base':'Number, post-sale facility principal cap.','haircut_call':'Number, triggered concentration call.','early_cash':'Number, unrestricted cash after early obligations.','final_cash':'Number, unrestricted cash after final obligations, debt and fees.','remaining_stress_risk':'Number, remaining stress-risk units.','economic_cost':'Number, minimum J.'},
        'Find the least-cost executable liquidation and funding policy after accounting for the feedback of sales into borrowing capacity, concentration calls and cross-strategy hedges. Check both cash deadlines and the remaining risk limit before ranking plans. Return the complete policy and its actual economics.',
        'Enumerate the finite sale and draw menu, recomputing the post-sale line cap and concentration discontinuity before checking early cash. Late B proceeds cannot fund its own early margin effect. Repayment and fees constrain final reserve. Compare opportunity cost plus fees and concentration call only among fully feasible policies.',
        [('feasible liquidation exists',len(rows)>0),('post-sale collateral holds',a[3]<=e['cap'])],
        [_wrong('freeze the borrowing base before liquidation',{'action':[4,1,0,15]},'Selling A and B reduces the executable line cap to4; the opening cap cannot finance the chosen liquidation.'),_wrong('pool late proceeds at the first deadline',{'action':[2,5,0,0]},'B sale proceeds arrive after the50 call and its own early hedge margin.')])


_I_WEIGHTS=[[2,3,4,5,3,8,9,16],[20,8,7,3,5,3,3,1]]
_I_COST={'A':F(1,2),'B':F(1,2),'C':F(3,2)}
_I_PAY={'invest':(30,-40),'hedge':(10,-8),'pass':(0,0)}
def information_economics(first,continuation,actions):
    if type(first) is not str or first not in ('none','A','B','C') or not isinstance(continuation,list) or len(continuation)!=2 or not isinstance(actions,list) or len(actions)!=4 or any(type(x) is not str or x not in _I_PAY for x in actions) or any(type(x) is not str for x in continuation):return _bad('Malformed observation policy.')
    if first=='none':
        if continuation!=['stop','stop'] or len(set(actions))!=1:return _bad('A no-query policy must stop and repeat its immediate action.')
    else:
        for b,q in enumerate(continuation):
            if q not in ('stop','A','B','C') or q==first:return _bad('Repeated or unknown second query.')
            if q=='stop' and actions[2*b]!=actions[2*b+1]:return _bad('Stopped branch actions must agree.')
            if q!='stop' and _I_COST[first]+_I_COST[q]>4:return _bad('Branch query-cost budget exceeded.')
    gross=F(0);cost=F(0)
    for state,weights in enumerate(_I_WEIGHTS):
        for index,w in enumerate(weights):
            bits=[(index>>2)&1,(index>>1)&1,index&1]
            if first=='none':action=actions[0];c=0
            else:
                b=bits['ABC'.index(first)];q=continuation[b];c=_I_COST[first]
                bit=0 if q=='stop' else bits['ABC'.index(q)]
                if q!='stop':c+=_I_COST[q]
                action=actions[2*b+bit]
            gross+=F(w,100)*_I_PAY[action][state];cost+=F(w,100)*c
    return {'violations':[],'gross':gross,'cost':cost,'net':gross-cost,'objective':cost-gross}
@lru_cache(None)
def information_reference():
    rows=[]
    for action in _I_PAY:
        e=information_economics('none',['stop','stop'],[action]*4);rows.append((e['objective'],'none',('stop','stop'),(action,)*4,e))
    for first in 'ABC':
        options=['stop']+[q for q in 'ABC' if q!=first]
        for continuation in product(options,repeat=2):
            for actions in product(_I_PAY,repeat=4):
                e=information_economics(first,list(continuation),list(actions))
                if 'objective' in e and not e['violations']:rows.append((e['objective'],first,continuation,actions,e))
    return sorted(rows,key=lambda r:r[:4])
def _information():
    rows=information_reference();o,f,cs,acts,e=rows[0];baseline=max(F(sum(_I_WEIGHTS[s][i]*pay[s] for s in range(2) for i in range(8)),100) for pay in _I_PAY.values())
    ans={'first_query':f,'second_queries':list(cs),'terminal_actions':list(acts),'expected_query_cost':float(e['cost']),'expected_gross_payoff':float(e['gross']),'optimal_net_value':float(-o),'no_information_value':float(baseline),'incremental_information_value':float(-o-baseline)}
    docs=[_doc('in_joint','Full correlated noisy-signal law','State is good or bad. Binary signal triple[A,B,C] is fixed once for the decision. In lexicographic triple order[000,001,010,011,100,101,110,111], JOINT probabilities for good are[2,3,4,5,3,8,9,16]/100 and for bad are[20,8,7,3,5,3,3,1]/100. These16 joint cells sum to1. Queries reveal the selected bit; they do not draw a fresh signal. Every bit is imperfect evidence about state and bits need not be independent either unconditionally or conditional on state.'),
          _doc('in_actions','Terminal financial opportunities','Exactly one terminal action is allowed: invest pays30 in good and -40 in bad; hedge pays10 in good and -8 in bad; pass pays0 in either state. These are terminal payoffs before query costs. No combinations, shorting, scaling or later switching are allowed. Maximize prior expected payoff minus all purchased query fees. Risk preferences are linear in cash and no wealth constraint applies apart from the stated research-cost budget.'),
          _doc('in_query','Adaptive query costs and budget','A query costs0.5 for A,0.5 for B,1.5 for C, regardless of bit. At most two DISTINCT queries may be purchased. Total fees on each possible path must be at most4. Stopping before any query or after the first is permitted. The second query identity, or stopping, may depend on the first observed bit; terminal action may depend on all purchased bits. There is no charge for observations not purchased.'),
          _doc('in_format','Policy representation and deterministic ties','first_query is A,B,C or none. second_queries is a two-element array in first-bit order[0,1], each A,B,C or stop. terminal_actions has four entries[first0/second0,first0/second1,first1/second0,first1/second1]; on a stopped branch repeat its terminal action in both corresponding slots. With first_query none, second_queries must be[stop,stop] and repeat one immediate action in all four slots. Among policies with equal expected net value, use lexicographically smallest tuple(first_query,second_queries,terminal_actions) under ordinary string order, comparing arrays entry by entry. All16 joint cells are possible; no absent branch can be ignored.')]
    return _case('c217','Adaptive paid research with noisy dependent observations','information_value','development',docs,ans,
        {'first_query':'String A,B,C or none.','second_queries':'Two strings, continuation decisions after first bit0,1.','terminal_actions':'Four strings invest,hedge or pass under the documented branch order.','expected_query_cost':'Number, ex ante total query fees.','expected_gross_payoff':'Number, prior expected terminal payoff before fees.','optimal_net_value':'Number, prior payoff net of query fees.','no_information_value':'Number, best prior payoff with no query.','incremental_information_value':'Number, optimal net value minus no-query value.'},
        'Choose the optimal adaptive research and investment policy under the explicit noisy joint signal law. Determine which query to buy first, whether and what to buy next on each result, and the final investment at each reachable history. Apply branch budgets and charge only purchased information; report full prior economics.',
        'Evaluate each allowable two-query decision tree under the16 joint cells, weighting both payoffs and purchased fees. The conditional posterior after one bit must preserve state-dependent signal correlation. Ranking exact rational expected net values and the stated lexical tie rule certifies the selected policy.',
        [('positive value of information',-o>baseline),('nontrivial adaptive policy',f!='none' and len(set(cs))>1)],
        [_wrong('buy the most expensive pair without a branch budget',{'first_query':'C','second_queries':['C','C']},'Repeated observations are forbidden and reveal no independent evidence.'),_wrong('treat gross information payoff as net',{'optimal_net_value':float(e['gross'])},'Every purchased query has a fee regardless of its realized bit.')])


def execution_economics(action):
    action=_whole_units(action)
    # At cutoff: corrected B purchase12 at10 settles T1; purchases15+5 at11 settle T2.
    # S fill10 at12 settles T1; canceled remainder remains reserved until accepted cancellation.
    if not _intvec(action,3,0,100) or action[0]%5 or action[1]%5:return _bad('Expected [sell now,buy T2,bridge] with share lots5 and integer cash bridge.')
    sell,buy,bridge=action;v=[]
    # Opening settled cash600; opening settled shares70. Delivery25 occurs T1.
    # Existing S is live20, corrected/executed shares are accounting-owned but not settled.
    if sell>50:v.append('sale exceeds settled unreserved inventory')
    if buy>80:v.append('new purchase cap')
    if bridge>100:v.append('bridge cap')
    t0cash=600+9*sell+bridge-10*buy
    t1cash=t0cash-120+120-400
    t2cash=t1cash-220-bridge-F(bridge,20)
    # Immediate sells settle T0; new buys prepaid T0 but delivered T2.
    t1shares=70-sell+12-10-25
    finalshares=t1shares+20+buy
    if t0cash<0:v.append('T0 cash deficit')
    if t1cash<50:v.append('T1 cash reserve')
    if t2cash<100:v.append('T2 cash reserve')
    if t1shares<20:v.append('T1 live sell reservation')
    if finalshares<20:v.append('final live sell reservation')
    # Maximize FINAL owned shares, not the gross purchase leg. Fees break economic ties.
    obj=-finalshares
    return {'violations':v,'t0':t0cash,'t1':t1cash,'t2':t2cash,'t1shares':t1shares,'finalshares':finalshares,'objective':obj,'cost':F(bridge,20)}
@lru_cache(None)
def execution_reference():
    rows=[]
    for sell in range(0,51,5):
        for buy in range(0,81,5):
            for bridge in range(101):
                a=[sell,buy,bridge];e=execution_economics(a)
                if not e['violations']:rows.append((e['objective'],tuple(a),e))
    return sorted(rows,key=lambda r:(r[0],r[2]['cost'],r[1]))
def _execution():
    rows=execution_reference();o,a,e=rows[0]
    ans={'owned_inventory_at_cutoff':92,'settled_inventory_at_cutoff':70,'live_sell_reservation':20,'old_trade_cash_by_day':[0,-220],'action':list(a),'cash_path':[float(e['t0']),float(e['t1']),float(e['t2'])],'inventory_after_T1':e['t1shares'],'final_owned_inventory':e['finalshares'],'bridge_fee':float(e['cost'])}
    docs=[_doc('ex_open','Opening balances and accepted orders','At T0 before any tape events: settled inventory70 shares, cash600. Accepted B buys80 at limit10; accepted S sells30 at12. No fills yet. Mandatory delivery of25 owned shares occurs T1 with no proceeds; operating cash payment400 occurs T1. Cash reserve requirements are specified solely in the action mandate. There are no other cash flows or positions. Holdings owned but not yet settled cannot fund a T0 sale.'),
          _doc('ex_tape','Exchange events and correction chain','Events in exchange order:1 B fill F1 quantity20 at9, settles T1;2 accepted B replacement sets TOTAL quantity60 at11, including prior filled quantity;3 B fill F2 quantity15 at11, settles T2;4 local cancel request for B;5 B fill F3 quantity5 at11, settles T2;6 B cancel acknowledgement closes its remaining20;7 correction of F1 replaces that entire execution with quantity12 at10, still settles T1;8 S fill F4 quantity10 at12 settles T1;9 duplicate delivery of event8 with the same event and execution ids;10 local cancel request for S with no acknowledgement. Every event is received by cutoff; no settlement has occurred yet. Corrections are replacements, not extra executions; duplicates are idempotent. A post-cancellation fill correction does not reopen B.'),
          _doc('ex_rules','Settlement and commitment conventions','Economic ownership changes when a trade fills or is corrected. Cash and settled shares change only on their dated settlement. S remains live for20 shares after its unacknowledged cancellation and reserves20 settled shares at every checkpoint. B is terminally canceled and reserves no future buy cash. At a checkpoint incoming settlements occur before mandatory delivery and no new S fill arrives during the planned horizon. Existing trade cash by day means net contractual cash [T1,T2] from old fills only, excluding action, delivery, operating payment and facility.'),
          _doc('ex_action','Repair action and economic objective','Choose at most one immediate sale at9 from0,5,...,50 shares, and at most one new purchase at10 from0,5,...,80 shares. Zero sale or zero purchase is permitted. New purchase cash is prepaid T0 and shares delivered T2. Sale proceeds and optional integer cash bridge in0..100 arrive T0 before purchase payment. Bridge principal plus5% fee is repaid T2. T0 sales use only currently settled unreserved inventory; no re-selling new receipts. At T1 maintain live S share reservation20 and cash at least50; at T2 maintain reservation20 and cash at least100. Cash cannot be negative at T0. Maximize final owned shares AFTER mandatory delivery, then minimize bridge fee, then lexicographically minimize[sell,buy,bridge]. No external execution is requested. Report cash inclusive of required reserves.')]
    return _case('c218','Reconcile corrected fills before a settlement-constrained repair','execution_state','development',docs,ans,
        {'owned_inventory_at_cutoff':'Integer shares owned economically after all events, before settlement or action.','settled_inventory_at_cutoff':'Integer currently settled shares before action.','live_sell_reservation':'Integer shares reserved by remaining accepted S.','old_trade_cash_by_day':'Two numbers, net old-trade cash at[T1,T2].','action':'Three integers [immediate sell shares,new buy shares,bridge cash].','cash_path':'Three numbers, cash after all obligations at[T0,T1,T2].','inventory_after_T1':'Integer settled shares after T1 settlements and delivery.','final_owned_inventory':'Integer total owned shares at T2 after action and delivery.','bridge_fee':'Number, bridge financing fee.'},
        'Reconstruct economic ownership, settled resources and surviving commitments from the corrected execution tape. Choose the best repair under each dated cash and share constraint. Earlier accepted replacement, later correction and unacknowledged cancellation must all be reflected in the executable state; gross purchase size is not the economic objective.',
        'Correct F1 from20@9 to12@10, retain F2+F3 purchases20@11, and count F4 sale10 only once. Owned shares=70+12+20-10=92; settled remain70. B stays closed, S reserves20. Old cash is0 atT1 and-220 atT2. Apply a new sale before any later settlements and honor the25-share T1 delivery. Rank feasible actions by final owned shares, bridge fee, then lexical order.',
        [('ownership conservation',70+12+15+5-10==92),('feasible optimal action',len(rows)>0 and not e['violations'])],
        [_wrong('count correction as an additional fill',{'owned_inventory_at_cutoff':112,'old_trade_cash_by_day':[-180,-220]},'F1 correction replaces the original fill and its cash, rather than adding a purchase.'),_wrong('finance from unacknowledged sell cancellation',{'live_sell_reservation':0,'action':[45,0,0]},'S remains live and T1 delivery leaves insufficient shares for its reservation.')])


def build_cases():
    cases=[_portfolio(),_quant(),_valuation(),_funding(),_information(),_execution()]
    # Source sufficiency is conditional on the reported action. Relevant extra context
    # remains allowed, but mechanically deriving a quantity need not cite every mandate.
    narrow={
      'c213':{'implementation_cost':['rp_book','rp_hedge'],'early_cash':['rp_book','rp_hedge','rp_cash'],'scenario_losses':['rp_limits']},
      'c214':{'eligible_blocks':['qt_tape','qt_cost'],**{k:['qt_tape','qt_cost','qt_registry'] for k in ['selected_policy','selected_checkpoint','selected_statistic','unadjusted_p','selection_adjusted_p','late_regime_mean']}},
      'c215':{'purchase_cost':['va_marginals','va_protection'],'state_losses':['va_marginals','va_protection'],'expected_profit_interval':['va_marginals','va_protection'],'minimum_expected_profit':['va_marginals','va_protection'],'worst_expected_shortfall':['va_marginals','va_protection','va_risk']},
      'c216':{'borrowing_base':['fu_cascade'],'haircut_call':['fu_cascade'],'remaining_stress_risk':['fu_cascade'],'economic_cost':['fu_cascade','fu_objective']},
      'c217':{'no_information_value':['in_joint','in_actions']},
      'c218':{k:['ex_open','ex_tape','ex_rules'] for k in ['owned_inventory_at_cutoff','settled_inventory_at_cutoff','live_sell_reservation','old_trade_cash_by_day']}}
    for c in cases:
        if c['id'] in ('c213','c215','c216','c218'):
            c['stages'][0]['prompt']+=' Whole-unit action quantities encoded as JSON3 or3.0 are equivalent; fractional, Boolean and nonfinite action quantities are invalid.'
        for key,ids in narrow[c['id']].items():c['stages'][0]['evidence_routes'][key]=[ids]
        if c['id']=='c214':c['stages'][0]['evidence_routes']['late_regime_mean']=[['qt_tape','qt_cost','qt_gate']]
        if c['id']=='c218':
            c['stages'][0]['evidence_routes']['old_trade_cash_by_day']=[['ex_tape','ex_rules']]
            c['stages'][0]['evidence_routes']['live_sell_reservation']=[['ex_rules']]
        validate(c)
    return cases

def economic_outcome(case,answers):
    """Score the submitted economic decision; malformed/infeasible policies earn no numeric regret.

    Synthetic plans are hypothetical, so rejection has no simulated cash mutation.
    Numeric objectives are minimization losses under the SAME prior information.
    Self-reported calculations never alter the decoded action or economics.
    """
    if not isinstance(answers,dict):return _bad('Expected answer object.')
    cid=case['id'] if isinstance(case,dict) else case
    if cid=='c213':
        e=portfolio_economics(answers.get('action'));rows=portfolio_reference();best=rows[0][0]
    elif cid=='c214':
        if type(answers.get('deploy')) is not bool:return _bad('Expected Boolean deployment action.')
        if answers['deploy']:return _bad('Deployment violates the frozen full-checkpoint, inference or stability gates.')
        return _out([],0,0)
    elif cid=='c215':e=valuation_economics(answers.get('action'));best=valuation_reference()[0][0]
    elif cid=='c216':e=funding_economics(answers.get('action'));best=funding_reference()[0][0]
    elif cid=='c217':
        e=information_economics(answers.get('first_query'),answers.get('second_queries'),answers.get('terminal_actions'));best=information_reference()[0][0]
    elif cid=='c218':e=execution_economics(answers.get('action'));best=execution_reference()[0][0]
    else:raise ValueError('Unknown case id')
    if 'action_feasible' in e:return e
    # Report avoidable cash cost only where the objective is an actual cost,
    # or where an explicit equal-inventory financing counterfactual is available.
    result=_out(e['violations'],e['objective'],best)
    result['avoidable_cost']=float(e['objective']-best) if cid == 'c216' and not e['violations'] else None
    result['objective_units']={'c213':'mandated risk-cost score','c215':'negative robust expected profit','c216':'mandated economic cost','c217':'negative prior expected net payoff','c218':'negative final shares'}[cid]
    if cid=='c218' and not e['violations']:
        same=[r[2]['cost'] for r in execution_reference() if r[2]['finalshares']==e['finalshares']]
        result['avoidable_cost']=float(e['cost']-min(same))
        result['avoidable_cost_scope']='Excess bridge fee versus a feasible policy with the same final inventory.'
    result['rejection_semantics']='Infeasible hypothetical plans are rejected; no state mutation or credit is awarded.'
    return result

if __name__=='__main__':
    import json
    print(json.dumps([{ 'id':c['id'],'answers':c['stages'][0]['answers'],'outcome':economic_outcome(c,c['stages'][0]['answers'])} for c in build_cases()],indent=2))
