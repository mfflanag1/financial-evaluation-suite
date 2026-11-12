"""Contracts and deterministic scoring for staged institutional research tasks."""
from copy import deepcopy
import json
import math
import re


INSTRUCTIONS = (
    'Use only the supplied synthetic information and the conversation history. '
    'Return strict JSON with question_id, turn, answers and evidence. answers maps every '
    'declared output to its typed value; evidence maps each output to a list of supporting '
    'source IDs available so far. Cite all and only the sources supporting that output; '
    'unsupported, unknown, repeated or future source IDs lose citation credit. Numbers use the declared units and '
    'tolerances; do not round intermediate calculations. Arrays require exact length and '
    'the requested order; strings use exact labels and booleans must be JSON true/false. Optional rationale is not automatically '
    'graded. Preserve prior constraints unless the user explicitly changes them. '
    'Distinguish stipulated scenarios from observed facts or forecasts. No trading, '
    'network access or proprietary query execution is required.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def typed(value, kind):
    if kind == 'number':
        return number(value)
    if kind == 'number_array':
        return isinstance(value, list) and bool(value) and all(number(v) for v in value)
    if kind == 'string_array':
        return isinstance(value, list) and all(type(v) is str and v for v in value)
    if kind == 'boolean':
        return type(value) is bool
    return kind == 'string' and type(value) is str and bool(value)


def match(value, expected, field):
    if not typed(value, field['type']):
        return False
    if field['type'] == 'number':
        return abs(value-expected) <= field['atol'] + field['rtol']*abs(expected)
    if field['type'] == 'number_array':
        return len(value) == len(expected) and all(
            abs(v-e) <= field['atol']+field['rtol']*abs(e) for v,e in zip(value,expected))
    return value == expected


def validate_case(case):
    require(isinstance(case,dict), 'Case must be an object')
    cid = case.get('id')
    require(isinstance(cid,str) and re.fullmatch(r'q\d{3}',cid), 'Invalid case ID')
    for key in ('title','domain','decision_relevance'):
        require(isinstance(case.get(key),str) and bool(case[key].strip()), f'{cid}: missing {key}')
    require(case.get('data_origin') == 'synthetic', f'{cid}: invalid data origin')
    require(case.get('mode') in ('single','multi'), f'{cid}: invalid mode')
    for key,minimum in (('capabilities',3),('complexity',4),('limitations',2)):
        values=case.get(key)
        require(isinstance(values,list) and len(values)>=minimum and all(type(v) is str and v.strip() for v in values)
                and len(set(values))==len(values), f'{cid}: invalid {key}')
    turns=case.get('turns')
    require(isinstance(turns,list) and len(turns)==(1 if case['mode']=='single' else 4), f'{cid}: turn count')
    available=set()
    for index,t in enumerate(turns,1):
        require(isinstance(t,dict) and type(t.get('turn')) is int and t['turn']==index, f'{cid}: turn order')
        label=f'{cid}/{index}'
        for key in ('user_message','solution'):
            require(isinstance(t.get(key),str) and len(t[key].strip())>=40, f'{label}: missing {key}')
        require(isinstance(t.get('inputs'),dict),f'{label}: inputs')
        require(isinstance(t.get('sources'),list),f'{label}: sources')
        for s in t['sources']:
            require(isinstance(s,dict) and isinstance(s.get('id'),str)
                    and re.fullmatch(r'[a-zA-Z0-9_-]+',s['id']) and s['id'] not in available
                    and isinstance(s.get('text'),str) and len(s['text'])>=30, f'{label}: invalid source')
            available.add(s['id'])
        require(len(available)>=2, f'{label}: need at least two source packets')
        require(isinstance(t.get('state_contract'),list) and t['state_contract']
                and all(type(v) is str and v for v in t['state_contract']), f'{label}: state contract')
        outputs=t.get('outputs')
        minimum=5 if case['mode']=='single' else 3
        require(isinstance(outputs,list) and minimum<=len(outputs)<=12, f'{label}: output count')
        names=[]
        for f in outputs:
            require(isinstance(f,dict) and isinstance(f.get('name'),str)
                    and re.fullmatch(r'[a-z][a-z0-9_]*',f['name']),f'{label}: field identifier')
            require(f.get('type') in ('number','number_array','string','string_array','boolean'),f'{label}: output type')
            require(all(isinstance(f.get(k),str) and f[k] for k in ('unit','description')),f'{label}: field metadata')
            if f['type'] in ('number','number_array'):
                require(number(f.get('atol')) and f['atol']>=0 and number(f.get('rtol'))
                        and 0<=f['rtol']<=.001,f'{label}: tolerance')
            names.append(f['name'])
        require(len(set(names))==len(names),f'{label}: duplicate field')
        for key in ('answer','evidence'):
            require(isinstance(t.get(key),dict) and set(t[key])==set(names),f'{label}: {key} keys')
        require(isinstance(t.get('allowed_evidence',{}),dict)
                and set(t.get('allowed_evidence',{}))<=set(names),f'{label}: allowed evidence map')
        for f in outputs:
            name=f['name']
            require(typed(t['answer'][name],f['type']),f'{label}/{name}: reference type')
            ids=t['evidence'][name]
            require(isinstance(ids,list) and bool(ids) and all(type(v) is str for v in ids)
                    and len(set(ids))==len(ids) and set(ids)<=available,f'{label}/{name}: reference evidence unavailable')
            allowed=t.get('allowed_evidence',{}).get(name,ids)
            require(isinstance(allowed,list) and all(type(v) is str for v in allowed)
                    and len(set(allowed))==len(allowed) and set(ids)<=set(allowed)<=available,
                    f'{label}/{name}: invalid allowed evidence')
        critical=t.get('critical_outputs')
        require(isinstance(critical,list) and critical and len(set(critical))==len(critical)
                and set(critical)<=set(names),f'{label}: critical outputs')
        checks=t.get('checks')
        require(isinstance(checks,list) and len(checks)>=2,f'{label}: missing construction checks')
        for c in checks:
            require(isinstance(c,dict) and c.get('name'),f'{label}: check shape')
            expected=c.get('expected')
            spec={'type':'boolean' if type(expected) is bool else 'number',
                  'atol':c.get('atol',1e-8),'rtol':c.get('rtol',1e-8)}
            require(typed(expected,spec['type']) and number(spec['atol']) and number(spec['rtol'])
                    and spec['atol']>=0 and 0<=spec['rtol']<=.001,f'{label}: check tolerance/type')
            require(match(c.get('actual'),expected,spec),f'{label}: construction check failed: {c["name"]}')
        wrongs=t.get('wrong_answers')
        require(isinstance(wrongs,list) and len(wrongs)>=2,f'{label}: missing misconception controls')
        for w in wrongs:
            require(isinstance(w,dict) and w.get('label') and w.get('reason'),f'{label}: control metadata')
            patch=w.get('answers')
            require(isinstance(patch,dict) and patch and set(patch)<=set(names),f'{label}: control shape')
            fs={f['name']:f for f in outputs}
            require(all(typed(v,fs[k]['type']) for k,v in patch.items()),f'{label}: control types')
            require(any(not match(v,t['answer'][k],fs[k]) for k,v in patch.items()),f'{label}: ineffective control {w["label"]}')
    json.dumps(case,allow_nan=False)
    return case


def public_turn(case, number=1):
    require(type(number) is int and 1<=number<=len(case['turns']),'Invalid turn')
    t=case['turns'][number-1]
    return deepcopy({'question_id':case['id'],'title':case['title'],'mode':case['mode'],
                     'data_origin':case['data_origin'],'turn':number,'total_turns':len(case['turns']),
                     'instructions':INSTRUCTIONS,'user_message':t['user_message'],
                     'inputs':t['inputs'],'sources':t['sources'],'outputs':t['outputs']})


def reference_response(case, number):
    t=case['turns'][number-1]
    return deepcopy({'question_id':case['id'],'turn':number,'answers':t['answer'],'evidence':t['evidence']})


def grade_turn(case, number, response):
    require(type(number) is int and 1<=number<=len(case['turns']),'Invalid turn')
    t=case['turns'][number-1]
    valid=isinstance(response,dict) and response.get('question_id')==case['id'] and type(response.get('turn')) is int and response['turn']==number
    answers=response.get('answers') if valid else None
    evidence=response.get('evidence') if valid else None
    available={s['id'] for stage in case['turns'][:number] for s in stage['sources']}
    fields=[]
    for f in t['outputs']:
        key=f['name']
        value_ok=isinstance(answers,dict) and key in answers and match(answers[key],t['answer'][key],f)
        ids=evidence.get(key) if isinstance(evidence,dict) else None
        allowed=set(t.get('allowed_evidence',{}).get(key,t['evidence'][key]))
        source_ok=(isinstance(ids,list) and bool(ids) and all(type(s) is str for s in ids)
                   and len(set(ids))==len(ids) and set(ids)<=available
                   and set(t['evidence'][key])<=set(ids)<=allowed)
        fields.append({'name':key,'value_correct':bool(value_ok),'source_ids_correct':bool(source_ok),
                       'correct':bool(value_ok and source_ok)})
    total=len(fields)
    return {'question_id':case['id'],'turn':number,'expected_fields':total,
            'correct_fields':sum(f['correct'] for f in fields),
            'value_accuracy':sum(f['value_correct'] for f in fields)/total,
            'source_id_accuracy':sum(f['source_ids_correct'] for f in fields)/total,
            'field_accuracy':sum(f['correct'] for f in fields)/total,
            'all_correct':all(f['correct'] for f in fields),
            'critical_pass':all(f['correct'] for f in fields if f['name'] in t['critical_outputs']),
            'fields':fields,
            'unknown_answer_fields':sorted(set(answers)-{f['name'] for f in t['outputs']}) if isinstance(answers,dict) else [],
            'interpretation':'Typed answer correctness and authored source-ID coverage/precision only; not prose quality, quote entailment, external retrieval or execution.'}


def grade_episode(case,responses):
    require(isinstance(responses,list),'Responses must be an ordered list')
    require(len(responses)<=len(case['turns']),'Too many responses')
    grades=[grade_turn(case,i,responses[i-1] if i<=len(responses) else None) for i in range(1,len(case['turns'])+1)]
    total=sum(r['expected_fields'] for r in grades)
    return {'question_id':case['id'],'expected_turns':len(grades),'submitted_turns':len(responses),
            'all_turns_correct':all(r['all_correct'] for r in grades),
            'final_turn_correct':grades[-1]['all_correct'],
            'critical_path_pass':all(r['critical_pass'] for r in grades),
            'field_accuracy_micro':sum(r['correct_fields'] for r in grades)/total,
            'turn_accuracy_mean':sum(r['field_accuracy'] for r in grades)/len(grades),
            'expected_fields':total,'turns':grades}


def control_results(case):
    results=[]
    for t in case['turns']:
        n=t['turn']
        gold=reference_response(case,n)
        results.append(grade_turn(case,n,gold)['all_correct'])
        for w in t['wrong_answers']:
            response=deepcopy(gold)
            response['answers'].update(w['answers'])
            results.append(not grade_turn(case,n,response)['all_correct'])
        for bad in (None,{},dict(gold,turn=n+1),dict(gold,evidence={})):
            result=grade_turn(case,n,bad)
            results.append(not result['all_correct'] and result['expected_fields']==len(t['outputs']))
        bad=deepcopy(gold)
        bad['evidence']={f['name']:['unavailable_source'] for f in t['outputs']}
        results.append(grade_turn(case,n,bad)['source_id_accuracy']==0)
    return results
