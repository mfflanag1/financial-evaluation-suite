#!/usr/bin/env python3
"""Grade a supplied-evidence diagnostic response; never execute its contents."""
import argparse,json,sys
from pathlib import Path
import grade
import variant_engine as engine

def main():
 p=argparse.ArgumentParser();p.add_argument('case_id',choices=[f'c{i}' for i in range(225,231)]);p.add_argument('response',type=Path);a=p.parse_args()
 grade.verify();c=next(c for c in grade.read(grade.ROOT/'variants/cases.json') if c['id']==a.case_id)
 try:r=grade.read(a.response)
 except (ValueError,UnicodeError,RecursionError):r=None
 if isinstance(r,dict) and 'tool' in r:raise ValueError('Submit a final response; this entry point uses supplied evidence.')
 s=engine.new_session(c,'supplied');engine.public_view(s);engine.step(s,r);print(json.dumps(engine.report(s),indent=2,allow_nan=False))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,StopIteration) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
