import argparse
import json
import sys
from .core import fit,predict
from .demo import run_demo,run_optimization_demo
from .acquisition import suggest
from .response import inspect
from .response_export import saved

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError("duplicate JSON key: "+k)
        out[k]=v
    return out



def main(argv=None):
    parser=argparse.ArgumentParser(description='KernelLab exact fixed-parameter RBF GP regression')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('fit','predict','suggest','response'):
        cmd=sub.add_parser(name);cmd.add_argument('--input',required=True)
    sub.add_parser('demo');sub.add_parser('optimize-demo');sub.add_parser('response-report')
    args=parser.parse_args(argv)
    try:
        if args.command=='response-report':result=saved()
        elif args.command=='demo':result=run_demo()
        elif args.command=='optimize-demo':result=run_optimization_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin,object_pairs_hook=pairs)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source,object_pairs_hook=pairs)
            if args.command=='response':
                if not isinstance(payload,dict) or set(payload) not in ({'model','row'},{'model','row','deltas'}):raise ValueError('response requires model/row and optional deltas only')
                result=inspect(payload['model'],payload['row'],payload.get('deltas'))
            elif args.command=='fit':result=fit(payload)
            elif args.command=='suggest':
                if not isinstance(payload,dict) or set(payload) not in ({'model','candidates'},{'model','candidates','xi'}):
                    raise ValueError('suggest requires model/candidates and optional xi only')
                result=suggest(payload['model'],payload['candidates'],payload.get('xi',0.))
            else:
                if not isinstance(payload,dict) or set(payload) not in ({'model','rows'},{'model','rows','full_covariance'}):
                    raise ValueError('prediction requires model/rows and optional full_covariance only')
                result=predict(payload['model'],payload['rows'],payload.get('full_covariance',False))
        text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OverflowError,ZeroDivisionError,OSError) as error:
        print('error: '+str(error),file=sys.stderr);return 2
    print(text);return 0
