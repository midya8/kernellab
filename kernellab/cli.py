import argparse
import json
import sys
from .core import fit,predict
from .demo import run_demo


def main(argv=None):
    parser=argparse.ArgumentParser(description='KernelLab exact fixed-parameter RBF GP regression')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('fit','predict'):
        cmd=sub.add_parser(name);cmd.add_argument('--input',required=True)
    sub.add_parser('demo')
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':result=run_demo()
        else:
            if args.input=='-':payload=json.load(sys.stdin)
            else:
                with open(args.input,encoding='utf-8') as source:payload=json.load(source)
            if args.command=='fit':result=fit(payload)
            else:
                if not isinstance(payload,dict) or set(payload) not in ({'model','rows'},{'model','rows','full_covariance'}):
                    raise ValueError('prediction requires model/rows and optional full_covariance only')
                result=predict(payload['model'],payload['rows'],payload.get('full_covariance',False))
        text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    except (ValueError,TypeError,OverflowError,ZeroDivisionError,OSError) as error:
        print('error: '+str(error),file=sys.stderr);return 2
    print(text);return 0
