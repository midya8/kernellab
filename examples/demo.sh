#!/usr/bin/env bash
set -eu
python -m kernellab demo
python -m kernellab fit --input examples/train.json > /tmp/kernellab-fit.json
python - <<'PY'
import json
from kernellab import fit,predict
with open('/tmp/kernellab-fit.json') as source:r=json.load(source)
with open('/tmp/kernellab-request.json','w') as dest:
    json.dump({'model':r['model'],'rows':[[.3],[1.4]],'full_covariance':True},dest)
assert len(r['loo']['means'])==3
p={'rows':[[0],[0]],'targets':[1,1],
   'kernel':{'length_scale':1.,'signal_variance':1.,'noise_variance':0.,'jitter':0.,'prior_mean':0.}}
try:fit(p)
except ValueError as error:assert 'pivot' in str(error)
else:raise AssertionError('singular noiseless training must fail')
p['kernel']['jitter']=1e-8
assert fit(p)['diagnostics']['training_diagonal_addition']==1e-8
print('singular covariance rejected; caller-specified jitter succeeds')
PY
python -m kernellab predict --input /tmp/kernellab-request.json > /tmp/kernellab-prediction.json
python - <<'PY'
import json
with open('/tmp/kernellab-prediction.json') as source:p=json.load(source)
assert len(p['means'])==2
for latent,observation in zip(p['latent_variances'],p['observation_variances']):
    assert abs(observation-latent-.2)<1e-12
assert p['latent_covariance'][0][1]==p['latent_covariance'][1][0]
print('serialized GP predicts means, latent uncertainty and noisy observations')
PY
PYTHONPATH=. python examples/render_demo.py
git diff --exit-code -- assets/posterior.svg 2>/dev/null || { test ! -d .git; }
