#!/usr/bin/env python3
"""Release staged user turns only after committing each preceding response."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode=True
import grade


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path,value):
    if path.is_symlink():
        raise ValueError('Refusing to overwrite a symlink')
    data=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def outside(private,public):
    private,public=private.resolve(),public.resolve()
    if private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError('Controller state and solver output directories must be separate')
    if public.is_relative_to(grade.ROOT) or grade.ROOT.is_relative_to(public):
        raise ValueError('Solver outputs must be outside evaluator files')
    if private.is_relative_to(grade.ROOT) or grade.ROOT.is_relative_to(private):
        raise ValueError('Session state must be outside the packaged evaluator')


def select(cid):
    catalog=grade.load_catalog()
    if cid not in catalog or catalog[cid][0]!='research58':
        raise ValueError('Select one of the research cases q107 through q164')
    return catalog[cid][1]


def state_and_case(directory):
    catalog=grade.load_catalog()
    state=grade.read_value(directory/'session.json')
    if not isinstance(state,dict) or state.get('schema_version')!=1:
        raise ValueError('Invalid session state')
    cid=state.get('question_id')
    if cid not in catalog or catalog[cid][0]!='research58':
        raise ValueError('Invalid session question')
    case=catalog[cid][1]
    if state.get('bank_sha256')!=sha((grade.ROOT/'research58/evaluator.json').read_bytes()):
        raise ValueError('Session bank has changed')
    responses=state.get('responses')
    if not isinstance(responses,list) or len(responses)>len(case['turns']):
        raise ValueError('Invalid committed responses')
    for i,entry in enumerate(responses,1):
        name=f'response_{i:02}.raw'
        path=directory/name
        if not isinstance(entry,dict) or entry.get('turn')!=i or entry.get('file')!=name:
            raise ValueError('Nonsequential session state')
        if path.is_symlink() or sha(path.read_bytes())!=entry.get('sha256'):
            raise ValueError('Committed response changed')
    if state.get('next_turn')!=len(responses)+1:
        raise ValueError('Session progression differs from committed responses')
    return state,case


def parse_response(path):
    try:
        return grade.read_value(path)
    except (ValueError,UnicodeError,RecursionError):
        return None


def publish(case,state,private,public):
    from financial_evals import research
    messages=[]
    for n in range(1,min(state['next_turn'],len(case['turns']))+1):
        messages.append({'role':'user','content':research.public_turn(case,n)})
        if n<=len(state['responses']):
            messages.append({'role':'assistant','content':(private/f'response_{n:02}.raw').read_bytes().decode('utf-8',errors='replace')})
    save(public/'conversation.json',{'question_id':case['id'],'messages':messages})
    if state['next_turn']<=len(case['turns']):
        save(public/'current_turn.json',research.public_turn(case,state['next_turn']))
        return {'status':'ready','question_id':case['id'],'turn':state['next_turn']}
    save(public/'current_turn.json',{'question_id':case['id'],'status':'complete'})
    return {'status':'complete','question_id':case['id'],'committed_turns':len(state['responses'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    start=commands.add_parser('start')
    start.add_argument('question_id')
    start.add_argument('--session',type=Path,required=True)
    start.add_argument('--out',type=Path,required=True)
    submit=commands.add_parser('submit')
    submit.add_argument('--session',type=Path,required=True)
    submit.add_argument('--response',type=Path,required=True)
    submit.add_argument('--out',type=Path,required=True)
    report=commands.add_parser('report')
    report.add_argument('--session',type=Path,required=True)
    resume=commands.add_parser('resume')
    resume.add_argument('--session',type=Path,required=True)
    resume.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        private=args.session.resolve()
        if args.command=='start':
            public=args.out.resolve()
            outside(private,public)
            if private.exists() or public.exists():
                raise ValueError('Start with fresh controller and solver directories')
            case=select(args.question_id)
            private.mkdir(parents=True)
            public.mkdir(parents=True)
            state={'schema_version':1,'question_id':case['id'],'next_turn':1,'responses':[],
                   'bank_sha256':sha((grade.ROOT/'research58/evaluator.json').read_bytes())}
            save(private/'session.json',state)
            result=publish(case,state,private,public)
        else:
            state,case=state_and_case(private)
            if args.command in ('submit','resume'):
                public=args.out.resolve()
                outside(private,public)
                if not public.is_dir():
                    raise ValueError('Solver output directory is missing')
                if args.command=='resume':
                    result=publish(case,state,private,public)
                    print(json.dumps(result,indent=2,allow_nan=False))
                    return 0
                n=state['next_turn']
                if n>len(case['turns']):
                    raise ValueError('Session is already complete')
                raw=args.response.read_bytes()
                name=f'response_{n:02}.raw'
                path=private/name
                if path.is_symlink():
                    raise ValueError('Response path cannot be a symlink')
                if path.exists():
                    if path.read_bytes()!=raw:
                        raise ValueError('An interrupted commit left a different response; retry the original response')
                else:
                    path.write_bytes(raw)
                state['responses'].append({'turn':n,'file':name,'sha256':sha(raw)})
                state['next_turn']=n+1
                save(private/'session.json',state)
                result=publish(case,state,private,public)
            else:
                from financial_evals import research
                responses=[parse_response(private/e['file']) for e in state['responses']]
                result=research.grade_episode(case,responses)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,KeyError,TypeError,RecursionError) as exc:
        parser.exit(2,f'Controller error: {exc}\n')


if __name__=='__main__':
    sys.exit(main())
