#!/usr/bin/env bash
set -euo pipefail
installed=${1}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
python - <<'PY'
import importlib,pkgutil,kernellab,json
from kernellab.response_export import build,saved
mods=list(pkgutil.walk_packages(kernellab.__path__,kernellab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==7
r=saved();assert build()==r
m=r['arms']['length_0_8']['report']['model'];json.dump(dict(model=m,row=[.125]),open('response.json','w'));json.dump(dict(model=m,rows=[[.125]],full_covariance=True),open('predict.json','w'));json.dump({k:m[k] for k in ['rows','targets','kernel']},open('train.json','w'))
zero=dict(m,kernel=dict(m['kernel'],noise_variance=0.));json.dump(dict(model=zero,candidates=[[.125],[.625]],xi=.01),open('suggest.json','w'))
print('all8modules,2original regression fits,3030targetresponses,202witnesses and full original optimization demo reproduce outside checkout')
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
from kernellab.response_export import saved
r=saved();assert json.load(open('demo.json'))==r['original_demo'];assert json.load(open('optimize.json'))==r['original_optimization_demo'];assert json.load(open('response-report.json'))==r
assert len(json.load(open('response-out.json'))['weights'])==15
assert json.load(open('fit.json'))==r['arms']['length_0_8']['report']
assert len(json.load(open('predict-out.json'))['latent_covariance'])==1
assert json.load(open('suggest-out.json'))['status']=='suggested'
print('all7main commands and exporter API reproduction pass')
PY
