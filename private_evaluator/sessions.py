#!/usr/bin/env python3
"""Trusted local coordinator. Private state never belongs in a solver workspace."""
from pathlib import Path
import argparse,json,os,sys
sys.dont_write_bytecode=True
import grade

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)
def publish(session,out,core):
    public=core.public_view(session['engine']);public['session_revision']=session['revision']
    out.mkdir(parents=True,exist_ok=True);save(out/'current.json',public)
def run():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','submit','resume','report']);p.add_argument('case_id',nargs='?');p.add_argument('--session',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--condition',choices=['supplied','retrieval'],default='retrieval');p.add_argument('--response',type=Path);p.add_argument('--revision',type=int);p.add_argument('--allow-holdout',action='store_true');a=p.parse_args()
    cases=grade.bank()
    import challenge_engine as core
    folder=a.session.resolve()
    if folder.is_relative_to(grade.ROOT) or folder.is_relative_to(grade.ROOT.parent/'questions'):raise ValueError('Session state must be outside package directories')
    if a.out:
        out=a.out.resolve()
        if out.is_relative_to(folder) or folder.is_relative_to(out) or out.is_relative_to(grade.ROOT) or grade.ROOT.is_relative_to(out) or out.is_relative_to(grade.ROOT.parent/'questions') or (grade.ROOT.parent/'questions').is_relative_to(out):raise ValueError('Private session and public output must be separate and outside packages')
    elif a.action!='report':raise ValueError('--out is required')
    folder.mkdir(parents=True,exist_ok=True);lock=folder/'coordinator.lock'
    with lock.open('x') as handle:handle.write('Single coordinator active\n')
    try:
        path=folder/'state.json';binding=grade.digest(grade.ROOT/'MANIFEST.json')
        if a.action=='start':
            if path.exists():raise ValueError('Session already exists')
            c=cases[a.case_id]
            if c['split']=='holdout' and not a.allow_holdout:raise ValueError('Holdout release requires explicit --allow-holdout; it was unrun in calibration')
            session={'bank_binding':binding,'revision':0,'engine':core.new_session(c,a.condition)}
        else:
            session=grade.read(path)
            if session['bank_binding']!=binding:raise ValueError('Session bank/runtime version differs')
        if a.action=='submit':
            if a.revision is None or a.revision!=session['revision']:raise ValueError('Stale or absent revision; do not replay a committed response')
            try:response=grade.read(a.response)
            except (ValueError,UnicodeError,RecursionError):response=None
            core.step(session['engine'],response);session['revision']+=1
        if a.action=='report':print(json.dumps(core.report(session['engine']),indent=2));return
        # Record the public question in private history before persisting state.
        core.public_view(session['engine']);save(path,session);publish(session,out,core)
        print(json.dumps({'status':'complete' if session['engine']['finished'] else 'released','revision':session['revision']}))
    finally:lock.unlink()
if __name__=='__main__':
    try:run()
    except (OSError,ValueError,TypeError,KeyError,AssertionError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
