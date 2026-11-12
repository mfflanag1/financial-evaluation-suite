"""Neutral challenge authoring contract; contains no provider integration."""
import math

def finite_json(x):
    if x is None or type(x) in (bool,str,int):return True
    if type(x) is float:return math.isfinite(x)
    if isinstance(x,list):return all(finite_json(v) for v in x)
    if isinstance(x,dict):return all(type(k) is str and finite_json(v) for k,v in x.items())
    return False

def validate(c):
    assert finite_json(c), 'Nonfinite or non-JSON case value'
    assert c['id'].startswith('c') and c['mode'] in ('single','multi')
    assert c['split'] in ('development','holdout') and c['family'] and c['title']
    assert len(c['stages'])==(1 if c['mode']=='single' else 4)
    ids={d['id'] for d in c['documents']};assert len(ids)==len(c['documents'])>=4
    assert len(c['oracle_document_ids'])>=2 and set(c['oracle_document_ids'])<=ids
    for d in c['documents']:assert len(d['text'])>30 and d['title'] and 1<=d.get('release_stage',1)<=len(c['stages'])
    for n,t in enumerate(c['stages'],1):
        assert len(t['prompt'])>80 and isinstance(t['inputs'],dict)
        assert 4<=len(t['answers'])<=10 and set(t['answers'])==set(t['descriptions'])==set(t['evidence_routes'])
        available={d['id'] for d in c['documents'] if d.get('release_stage',1)<=n}
        for name,routes in t['evidence_routes'].items():
            assert routes and all(route and len(route)==len(set(route)) and set(route)<=available for route in routes)
            required={d for route in routes for d in route}
            allowed=set(t.get('allowed_evidence',{}).get(name,required))
            assert required<=allowed<=available
            assert required<=set(c['oracle_document_ids'])
        assert len(t['checks'])>=2 and all(v['passed'] is True for v in t['checks'])
        assert len(t['wrong_answers'])>=2 and all(any(t['answers'].get(k)!=v for k,v in w['answers'].items()) for w in t['wrong_answers'])
        assert t['solution'] and all(set(a)==set(t['answers']) for a in t.get('alternative_answers',[]))
    return c
