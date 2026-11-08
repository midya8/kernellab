"""Reproduce original regression fixture; retain every original result."""
import json,math
from pathlib import Path
from .core import fit,predict
from .demo import fixture,run_demo,run_optimization_demo
from .response import inspect

def build():
    rows,targets,query,test,truth=fixture();arms={}
    for name,length in [('length_0_8',.8),('length_5',5.)]:
        report=fit(dict(rows=rows,targets=targets,kernel=dict(length_scale=length,signal_variance=1.,noise_variance=.12**2,jitter=1e-10,prior_mean=0.)))
        refs=[inspect(report['model'],q) for q in query]
        edits=[]
        for j in range(15):
            deltas=[0.]*15;deltas[j]=1.
            edits.append(predict(dict(report['model'],targets=[y+d for y,d in zip(targets,deltas)]),query))
        arms[name]=dict(report=report,references=refs,unit_target_edit_curves=edits)
    return dict(version=1,arms=arms,query=query,report_only_test=test,report_only_truth=truth,original_demo=run_demo(),original_optimization_demo=run_optimization_demo())

def assert_reproduces(actual,expected,path='root'):
    """Exact structure/non-numeric data, 1e-12 abs/rel binary64 portability."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path
        for key in expected:assert_reproduces(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):assert_reproduces(a,b,path+'['+str(i)+']')
    elif type(expected) is float:
        assert type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(path,actual,expected)
    else:assert type(actual) is type(expected) and actual==expected,(path,actual,expected)

def encoded(x):return json.dumps(x,separators=(',',':'),allow_nan=False)+'\n'
def export(root):
    root=Path(root);(root/'kernellab/artifacts').mkdir(exist_ok=True);(root/'docs').mkdir(exist_ok=True);r=build()
    for name,a in r['arms'].items():
        (root/('kernellab/artifacts/response-'+name+'.json')).write_text(encoded({k:v for k,v in a.items() if k!='references'}))
        for k in range(4):(root/('kernellab/artifacts/response-'+name+'-refs-'+str(k)+'.json')).write_text(encoded(a['references'][k*26:(k+1)*26]))
    meta={k:v for k,v in r.items() if k!='arms'};(root/'kernellab/artifacts/response-meta.json').write_text(encoded(meta))
    browser=dict(meta,arms={k:dict(model=a['report']['model'],report=a['report']) for k,a in r['arms'].items()})
    (root/'docs/data.js').write_text('globalThis.KERNEL_DATA = '+encoded(browser).rstrip()+';\n');return dict(arms=2,queries=202,target_edit_predictions=3030)
def saved():
    root=Path(__file__).parent/'artifacts';r=json.loads((root/'response-meta.json').read_text());r['arms']={}
    for name in ['length_0_8','length_5']:
        a=json.loads((root/('response-'+name+'.json')).read_text());a['references']=[p for k in range(4) for p in json.loads((root/('response-'+name+'-refs-'+str(k)+'.json')).read_text())];r['arms'][name]=a
    return r
if __name__=='__main__':print(json.dumps(export(Path(__file__).resolve().parents[1])))
