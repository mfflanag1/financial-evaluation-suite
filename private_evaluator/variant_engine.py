"""Pure staged evaluation with bounded document tools and simulated state."""
from copy import deepcopy
import math
import re
import combined_simulations as simulations
import hard_single
import information_variants
import hard_simulations


def number(x):return type(x) in (int,float) and math.isfinite(x)
def matches(x,y):
    if type(y) is bool:return type(x) is bool and x==y
    if number(y):return number(x) and abs(x-y)<=1e-6+1e-8*abs(y)
    if isinstance(y,list):return isinstance(x,list) and len(x)==len(y) and all(matches(a,b) for a,b in zip(x,y))
    return type(x) is type(y) and x==y

def strict_score(t,response,cid,n,seen):
    valid=isinstance(response,dict) and response.get('question_id')==cid and type(response.get('turn')) is int and response['turn']==n
    a=response.get('answers',{}) if valid else {};e=response.get('evidence',{}) if valid else {}
    a=a if isinstance(a,dict) else {};e=e if isinstance(e,dict) else {}
    options=[t['answers']]+t.get('alternative_answers',[])
    def exact_action(x,y):
        if type(y) is bool:return type(x) is bool and x==y
        if number(y):return number(x) and x==y
        if isinstance(y,list):return isinstance(x,list) and len(x)==len(y) and all(exact_action(a,b) for a,b in zip(x,y))
        return type(x) is type(y) and x==y
    def field_match(k,x,y):
        return exact_action(x,y) if cid.startswith('c') and int(cid[1:])>=213 and k=='action' else matches(x,y)
    chosen=max(options,key=lambda ref:sum(field_match(k,a.get(k),v) for k,v in ref.items()))
    fields=[]
    for k,v in chosen.items():
        ids=e.get(k);routes=t['evidence_routes'][k]
        allowed=set(t.get('allowed_evidence',{}).get(k,[i for route in routes for i in route]))
        support=isinstance(ids,list) and all(type(i) is str for i in ids) and len(ids)==len(set(ids)) and set(ids)<=allowed&set(seen) and any(set(route)<=set(ids) for route in routes)
        fields.append({'name':k,'value_correct':field_match(k,a.get(k),v),'source_correct':bool(support)})
    coherent=any(all(field_match(k,a.get(k),v) for k,v in ref.items()) for ref in options)
    return {'fields':fields,'all_values_correct':coherent,'all_supported_correct':coherent and all(f['source_correct'] for f in fields),'expected_fields':len(fields),'value_correct_fields':sum(f['value_correct'] for f in fields),'source_correct_fields':sum(f['source_correct'] for f in fields),'coherent_alternative_match':coherent}


def score(t,response,cid,n,seen):
    result=strict_score(t,response,cid,n,seen)
    identity=isinstance(response,dict) and response.get('question_id')==cid and type(response.get('turn')) is int and response['turn']==n
    contract=identity and isinstance(response.get('answers'),dict) and isinstance(response.get('evidence'),dict)
    normalized=deepcopy(response) if isinstance(response,dict) else {}
    normalized.update(question_id=cid,turn=n)
    content=strict_score(t,normalized,cid,n,seen)
    result.update(response_identity_valid=identity,response_contract_valid=contract,
                  financial_content_correct=content['all_values_correct'],
                  financial_content_correct_fields=content['value_correct_fields'],
                  independent_source_correct_fields=content['source_correct_fields'],
                  content_and_support_correct=content['all_supported_correct'])
    return result


def new_session(case,condition):
    if condition not in ('supplied','retrieval'):raise ValueError('Unknown condition')
    return {'case':deepcopy(case),'condition':condition,'turn':1,'state':simulations.initial(case['id']) if case.get('simulator') else None,
            'history':[],'seen':[],'tool_calls':0,'scores':[],'actions':[],'economic_outcomes':[],'finished':False,'presented_turn':0}


def stage(session):
    c=session['case'];n=session['turn']
    return simulations.view(c,n,session['state']) if c.get('simulator') else deepcopy(c['stages'][n-1])


def documents(session):return [deepcopy(d) for d in session['case']['documents'] if d.get('release_stage',1)<=session['turn']]


def public_view(session):
    if session['finished']:return {'finished':True}
    c=session['case'];t=stage(session);docs=documents(session)
    visible=[d for d in docs if d['id'] in c['oracle_document_ids']] if session['condition']=='supplied' else []
    session['seen']=list(dict.fromkeys(session['seen']+[d['id'] for d in visible]))
    type_of=lambda v:'boolean' if type(v) is bool else 'number' if number(v) else 'array' if isinstance(v,list) else 'string'
    value={'question':{'question_id':c['id'],'title':c['title'],'turn':session['turn'],'total_turns':len(c['stages']),
            'prompt':t['prompt'],'inputs':deepcopy(t['inputs']),'sources':visible,
            'document_catalog':[{'id':d['id'],'title':d['title']} for d in docs],
            'outputs':{k:{'type':type_of(v),'description':t['descriptions'][k]} for k,v in t['answers'].items()},
            'response_contract':{'question_id':c['id'],'turn':session['turn'],'answers':'All declared typed fields','evidence':'For each field, a list of IDs of viewed supporting documents. Each field must have its own sufficient support; references elsewhere do not carry over. Valid alternative sufficient routes and relevant corroboration accepted. Numeric tolerance1e-6 absolute+1e-8 relative.'},
            'tools':{'read':'{"tool":"read","ids":["id",...]}; at most3 IDs per call','search':'{"tool":"search","query":"text"}; deterministic ranked matches among released documents'},
            'remaining_document_calls':max(0,6-session['tool_calls'])},'history':deepcopy(session['history'])}
    value['question']['final_response_template']={
        'question_id':c['id'],'turn':session['turn'],
        'answers':{k:'<typed value>' for k in t['answers']},
        'evidence':{k:['<viewed supporting document ID>'] for k in t['answers']}}
    value['question']['required_final_keys_note']='Include all four top-level keys: question_id, integer turn, answers and evidence, even for single-turn cases. Replace every placeholder; this template contains no financial answer.'
    if session['presented_turn']!=session['turn']:
        session['history'].append({'role':'user','content':deepcopy(value['question'])});session['presented_turn']=session['turn']
    value['history']=deepcopy(session['history'])
    return value


def step(session,response):
    if session['finished']:raise ValueError('Session already complete')
    if isinstance(response,dict) and 'tool' in response:
        session['tool_calls']+=1
        if session['tool_calls']>6:
            result={'error':'Document-call budget exhausted; submit final JSON now.'}
            session['history'].extend([{'role':'assistant','content':deepcopy(response)},{'role':'tool','content':result}]);return result
        docs=documents(session);byid={d['id']:d for d in docs}
        if response.get('tool')=='read' and set(response)=={'tool','ids'} and isinstance(response['ids'],list) and 1<=len(response['ids'])<=3 and all(type(i) is str for i in response['ids']):
            result={'documents':[byid[i] for i in response['ids'] if i in byid],'unavailable_ids':[i for i in response['ids'] if i not in byid]}
            session['seen']=list(dict.fromkeys(session['seen']+[d['id'] for d in result['documents']]))
        elif response.get('tool')=='search' and set(response)=={'tool','query'} and isinstance(response['query'],str) and 0<len(response['query'])<=200:
            words=set(re.findall(r'[a-z0-9]+',response['query'].lower()))
            ranked=sorted(docs,key=lambda d:(-sum(w in (d['title']+' '+d['text']).lower() for w in words),d['id']))[:3]
            result={'matches':[{'id':d['id'],'title':d['title']} for d in ranked]}
        else:result={'error':'Invalid tool request. Use declared read or search schemas.'}
        session['history'].extend([{'role':'assistant','content':deepcopy(response)},{'role':'tool','content':result}]);return result
    t=stage(session);c=session['case'];n=session['turn'];public=public_view(session)['question'];g=score(t,response,c['id'],n,session['seen']);session['scores'].append(g)
    session['history'].append({'role':'assistant','content':deepcopy(response)})
    action_answers=response.get('answers',{}) if g['response_identity_valid'] else {}
    action_answers=action_answers if isinstance(action_answers,dict) else {}
    if c.get('variant_of')=='c217':
        session['economic_outcomes'].append(information_variants.economic_outcome(c,action_answers))
    elif 213<=int(c['id'][1:])<=218:
        session['economic_outcomes'].append(hard_single.economic_outcome(c,action_answers))
    elif int(c['id'][1:])>=219:
        session['economic_outcomes'].append(hard_simulations.economic_outcome(c['id'],deepcopy(session['state']),action_answers))
    if c.get('simulator'):
        good_identity=isinstance(response,dict) and response.get('question_id')==c['id'] and type(response.get('turn')) is int and response['turn']==n
        answers=response.get('answers',{}) if good_identity else {};answers=answers if isinstance(answers,dict) else {}
        session['state'],receipt=simulations.commit(c['id'],session['state'],answers);session['actions'].append(receipt)
        session['history'].append({'role':'tool','content':{'simulated_execution_ack':receipt}})
        if n<len(c['stages']):session['state']=simulations.advance(c['id'],session['state'])
    session['finished']=n==len(c['stages'])
    if not session['finished']:session['turn']+=1;session['tool_calls']=0
    return {'status':'complete' if session['finished'] else 'next_turn','turn':session['turn']}


def report(session):
    c=session['case'];count=sum(len(t['answers']) for t in c['stages']);scores=session['scores']
    terminal=simulations.terminal_ok(c['id'],session['state']) if c.get('simulator') else None
    if c.get('simulator'):terminal=bool(session['finished'] and terminal)
    return {'case_id':c['id'],'condition':session['condition'],'completed':session['finished'],'expected_fields':count,
            'value_correct_fields':sum(s['value_correct_fields'] for s in scores),'source_correct_fields':sum(s['source_correct_fields'] for s in scores),
            'all_values_correct':len(scores)==len(c['stages']) and all(s['all_values_correct'] for s in scores),
            'all_supported_correct':len(scores)==len(c['stages']) and all(s['all_supported_correct'] for s in scores),
            'response_contract_valid':len(scores)==len(c['stages']) and all(s['response_contract_valid'] for s in scores),
            'financial_content_correct':len(scores)==len(c['stages']) and all(s['financial_content_correct'] for s in scores),
            'financial_content_correct_fields':sum(s['financial_content_correct_fields'] for s in scores),
            'independent_source_correct_fields':sum(s['independent_source_correct_fields'] for s in scores),
            'economic_outcomes':session.get('economic_outcomes',[]),
            'terminal_state_pass':terminal,'final_state':session['state'],'actions':session['actions'],'stages':scores}
