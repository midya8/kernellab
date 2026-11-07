"""Finite-candidate expected improvement for deterministic maximization only."""
import math
from .core import _number,_rows,predict


def expected_improvement(mean,variance,incumbent,xi=0.):
    """E[max(F-incumbent-xi,0)] under the supplied scalar Gaussian."""
    mean=_number(mean,-1e12,1e12,'mean');incumbent=_number(incumbent,-1e12,1e12,'incumbent')
    variance=_number(variance,0,1e12,'variance');xi=_number(xi,0,1e6,'xi')
    delta=mean-incumbent-xi
    if variance==0:return max(0.,delta)
    sigma=math.sqrt(variance);z=delta/sigma
    cdf=.5*math.erfc(-z/math.sqrt(2))
    pdf=math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    # Cancellation in a very small negative-tail EI can round below zero.
    return max(0.,delta*cdf+sigma*pdf)


def suggest(model,candidates,xi=0.):
    """Score a finite pool without executing evaluations or fitting parameters."""
    xi=_number(xi,0,1e6,'xi')
    # predict owns strict artifact validation. No caller-supplied alpha/factors.
    posterior=predict(model,candidates)
    if model['kernel']['noise_variance']!=0:
        raise ValueError('suggest requires noise_variance=0 for deterministic objectives; noisy EI is unsupported')
    rows=_rows(candidates,len(model['rows'][0]));keys=[tuple(row) for row in rows]
    if len(set(keys))!=len(keys):raise ValueError('candidate pool must have unique rows after binary64 conversion')
    known={tuple(row) for row in _rows(model['rows'])}
    incumbent=max(model['targets']);scored=[];best_index=None;best_score=-1.
    for i,(row,mean,variance) in enumerate(zip(rows,posterior['means'],posterior['latent_variances'])):
        excluded=tuple(row) in known
        score=None if excluded else expected_improvement(mean,variance,incumbent,xi)
        scored.append({'index':i,'row':row,'mean':mean,'latent_variance':variance,
                       'expected_improvement':score,'excluded_already_evaluated':excluded})
        if score is not None and score>best_score:
            best_index=i;best_score=score
    return {'suggestion':None if best_index is None else dict(scored[best_index]),
            'status':'exhausted' if best_index is None else 'suggested',
            'incumbent':incumbent,'xi':xi,'candidates':scored,
            'prediction_diagnostics':posterior['diagnostics'],
            'scope':'Deterministic maximization only; EI uses latent variance and best observed target. '
                    'Fixed hyperparameters and caller-owned candidate pool; exact ties retain first eligible index. '
                    'Already evaluated rows are excluded by exact binary64 coordinate equality, not a distance tolerance. '
                    'Training jitter perturbs the posterior. Even zero maximum EI returns a candidate, not a convergence claim. '
                    'No objective evaluation, batch optimization, continuous-domain guarantee, noisy-objective method or external action.'}
