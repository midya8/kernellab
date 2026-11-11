#!/usr/bin/env bash
set -euo pipefail
installed=${1}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,kernellab,json
from kernellab.response_export import build,saved,assert_reproduces
mods=list(pkgutil.walk_packages(kernellab.__path__,kernellab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==8
r=saved();assert_reproduces(build(),r)
m=r['arms']['length_0_8']['report']['model'];json.dump(dict(model=m,row=[.125]),open('response.json','w'));json.dump(dict(model=m,rows=[[.125]],full_covariance=True),open('predict.json','w'));json.dump({k:m[k] for k in ['rows','targets','kernel']},open('train.json','w'))
zero=dict(m,kernel=dict(m['kernel'],noise_variance=0.));json.dump(dict(model=zero,candidates=[[.125],[.625]],xi=.01),open('suggest.json','w'))
print('all9modules,2original regression fits,3030targetresponses,202witnesses and full original optimization demo reproduce outside checkout')
PY
python -m kernellab fit --input train.json > fit.json
python -m kernellab predict --input predict.json > predict-out.json
python -m kernellab suggest --input suggest.json > suggest-out.json
python -m kernellab demo > demo.json
python -m kernellab optimize-demo > optimize.json
python -m kernellab response --input response.json > response-out.json
python -m kernellab response-report > response-report.json
python - <<'PY'
import json
from kernellab.response_export import saved,assert_reproduces
r=saved();assert_reproduces(json.load(open('demo.json')),r['original_demo']);assert_reproduces(json.load(open('optimize.json')),r['original_optimization_demo']);assert_reproduces(json.load(open('response-report.json')),r)
assert len(json.load(open('response-out.json'))['weights'])==15
assert_reproduces(json.load(open('fit.json')),r['arms']['length_0_8']['report'])
assert len(json.load(open('predict-out.json'))['latent_covariance'])==1
assert json.load(open('suggest-out.json'))['status']=='suggested'
print('all7main commands and exporter API reproduction pass')
PY

python - <<'PY2'
import json,subprocess,sys
from pathlib import Path
import kernellab
from kernellab.joint import audit
files=sorted((Path(kernellab.__file__).parent/'artifacts').glob('joint-*.json'))
assert len(files)==12
for f in files:
 c=json.loads(f.read_text());assert audit(c['input'])==c['report']
 Path('joint-input.json').write_text(json.dumps(c['input']))
 p=subprocess.run([sys.executable,'-m','kernellab.joint','joint-input.json'],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 assert json.loads(p.stdout)==c['report']
 bad=dict(c['input'],weights=[[11]*len(c['input']['queries'])])
 p=subprocess.run([sys.executable,'-m','kernellab.joint','-'],input=json.dumps(bad),capture_output=True,text=True)
 assert p.returncode==2 and not p.stdout
print('12complete joint controls, new CLI and12invalid weight refusals reproduce from installed package outside checkout')
PY2
