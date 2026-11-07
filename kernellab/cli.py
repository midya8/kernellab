import argparse
import json
import sys
from .core import fit,predict
from .demo import run_demo,run_optimization_demo
from .acquisition import suggest


def main(argv=None):
    parser=argparse.ArgumentParser(description='KernelLab exact fixed-parameter RBF GP regression')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('fit','predict','suggest'):
        cmd=sub.add_parser(name);cmd.add_argument('--input',required=True)
    sub.add_parser('demo');sub.add_parser('optimize-demo')
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        elif args.command=='optimize-demo':result=run_optimization_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            if args.command=='fit':result=fit(payload)
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
