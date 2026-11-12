#!/usr/bin/env python3
"""Private collection entry point. Never executes submitted code."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key')
        out[k]=v
    return out
def read(path):
    value=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    json.dumps(value,allow_nan=False)
    return value
def verify_tree(root,kind):
    if root.is_symlink() or (root/'MANIFEST.json').is_symlink():raise ValueError('Symlinked package')
    m=read(root/'MANIFEST.json')
    if m.get('package')!=kind or not isinstance(m.get('files'),dict):raise ValueError('Invalid manifest')
    for rel,expected in m['files'].items():
        if not isinstance(rel,str) or Path(rel).is_absolute() or any(x in ('','.','..') for x in rel.split('/')) or '\\' in rel:raise ValueError('Invalid manifest path')
        p=root
        for part in rel.split('/'):
            p=p/part
            if p.is_symlink():raise ValueError('Symlinked manifest member')
        if not p.is_file() or digest(p)!=expected:raise ValueError('Package drift: '+rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=root/'MANIFEST.json'}
    if actual!=set(m['files']):raise ValueError('Unlisted or missing package file')
    return m
def verify():
    m=verify_tree(ROOT,'private_evaluator');q=ROOT.parent/'questions'
    if q.exists():
        if digest(q/'MANIFEST.json')!=m['solver_manifest_sha256']:raise ValueError('Solver manifest binding differs')
        verify_tree(q,'questions')
    return m
def bank():
    verify()
    import case_contract
    cases=read(ROOT/'challenges/cases.json')
    if [c['id'] for c in cases]!=[f'c{i}' for i in range(201,225)]:raise ValueError('Unexpected challenge IDs')
    for c in cases:case_contract.validate(c)
    return {c['id']:c for c in cases}
def controls(cases):
    import challenge_engine as e
    count=0
    for c in cases.values():
        z=e.new_session(c,'supplied')
        while not z['finished']:
            e.public_view(z);t=e.stage(z)
            r={'question_id':c['id'],'turn':z['turn'],'answers':t['answers'],'evidence':{k:v[0] for k,v in t['evidence_routes'].items()}}
            assert e.score(t,r,c['id'],z['turn'],z['seen'])['all_supported_correct'];count+=1
            for a in t.get('alternative_answers',[]):
                assert e.score(t,{**r,'answers':a},c['id'],z['turn'],z['seen'])['all_supported_correct'];count+=1
            for w in t['wrong_answers']:
                assert not e.score(t,{**r,'answers':{**r['answers'],**w['answers']}},c['id'],z['turn'],z['seen'])['all_values_correct'];count+=1
            e.step(z,r)
        assert e.report(z)['all_supported_correct'] and e.report(z)['terminal_state_pass'] is not False
    return count
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','controls','grade']);p.add_argument('case_id',nargs='?');p.add_argument('response',nargs='?');p.add_argument('--turn',type=int,default=1);p.add_argument('--variant');a=p.parse_args()
    cases=bank()
    if a.action in ('validate','controls'):
        old=subprocess.run([sys.executable,'-B',str(ROOT/'legacy/grade.py'),a.action],capture_output=True,text=True)
        if old.returncode:raise ValueError(old.stderr or old.stdout)
        print(json.dumps({'legacy':json.loads(old.stdout),'challenge_cases':len(cases),'challenge_fields':sum(len(t['answers']) for c in cases.values() for t in c['stages']),'challenge_controls':controls(cases) if a.action=='controls' else None}));return
    if a.case_id not in cases:
        args=[sys.executable,'-B',str(ROOT/'legacy/grade.py'),'grade',a.case_id or '',a.response or '','--turn',str(a.turn)]
        if a.variant:args+=['--variant',a.variant]
        raise SystemExit(subprocess.run(args).returncode)
    c=cases[a.case_id]
    if c['mode']=='multi':raise ValueError('Use sessions.py for multi-turn challenges; grading needs actual history and state')
    if a.turn!=1 or a.variant:raise ValueError('Invalid single-turn invocation')
    import challenge_engine as e
    try:r=read(a.response)
    except (ValueError,UnicodeError,RecursionError):r=None
    z=e.new_session(c,'supplied');e.public_view(z);e.step(z,r);print(json.dumps(e.report(z),indent=2))
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,TypeError,KeyError,AssertionError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
