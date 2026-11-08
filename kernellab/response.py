"""Fixed-kernel target response, not causal influence or hyperparameter refitting."""
import math
from .core import predict,_training,_factor,_solve,rbf,_number

def inspect(model,row,deltas=None):
    baseline=predict(model,[row])  # The owning inference surface validates the complete model and query.
    rows,targets,p=_training({k:model[k] for k in ('rows','targets','kernel')});n=len(rows)
    if deltas is None:deltas=[0.]*n
    if not isinstance(deltas,list) or len(deltas)!=n:raise ValueError('deltas require one finite value per training target')
    deltas=[_number(v,-1e6,1e6,'target delta') for v in deltas]
    changed=[_number(y+d,-1e6,1e6,'edited target') for y,d in zip(targets,deltas)]
    lower,_,_=_factor(rows,targets,p);cross=[rbf(row,r,p['length_scale'],p['signal_variance']) for r in rows];weights=_solve(lower,cross)
    a=[[rbf(u,v,p['length_scale'],p['signal_variance'])+(p['noise_variance']+p['jitter'] if i==j else 0.) for j,v in enumerate(rows)] for i,u in enumerate(rows)]
    residual=max(abs(math.fsum(x*w for x,w in zip(r,weights))-k) for r,k in zip(a,cross))
    scale=max(sum(abs(v) for v in r) for r in a)*max(abs(v) for v in weights)+max(abs(v) for v in cross)
    normalized=residual/scale if scale else 0.
    if not all(math.isfinite(v) for v in weights+[normalized]) or normalized>1e-10:raise ValueError('target-response solve residual exceeds numerical gate')
    contributions=[w*(y-p['prior_mean']) for w,y in zip(weights,targets)]
    mean=p['prior_mean']+math.fsum(contributions);shift=math.fsum(w*d for w,d in zip(weights,deltas));edited=predict(dict(model,targets=changed),[row])
    tolerance=1e-9*(1+sum(abs(v) for v in contributions)+abs(shift))
    if abs(mean-baseline['means'][0])>tolerance or abs(mean+shift-edited['means'][0])>tolerance:raise ValueError('adjoint target-response mean disagrees with direct conditioning')
    return dict(version=1,row=list(map(float,row)),weights=weights,centered_target_terms=contributions,deltas=deltas,edited_targets=changed,
                original_mean=baseline['means'][0],reconstructed_mean=mean,mean_shift=shift,edited_mean=edited['means'][0],
                latent_variance=baseline['latent_variances'][0],observation_variance=baseline['observation_variances'][0],
                edited_latent_variance=edited['latent_variances'][0],edited_observation_variance=edited['observation_variances'][0],
                training_diagonal_addition=p['noise_variance']+p['jitter'],noise_variance=p['noise_variance'],jitter=p['jitter'],
                solve_residual=residual,normalized_solve_residual=normalized,mean_roundoff_disagreement=mean-baseline['means'][0],
                negative_variance_tolerance=baseline['diagnostics']['negative_variance_tolerance'],
                scope='Fixed observed rows, isotropic RBF parameters, supplied mean, iid noise and explicit training jitter. Signed target sensitivities, not feature or causal attribution; editing targets does not refit hyperparameters. Covariance is conditional model variance, not calibrated confidence. Numerical witnesses, not exact or forward-error certificates.')
