"""Exact small-data Gaussian-process regression with explicit fixed assumptions."""
import math


def _number(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(name + ' must be finite in [' + str(low) + ',' + str(high) + ']')
    return float(value)


def _rows(value, width=None):
    if not isinstance(value,list) or not 1 <= len(value) <= 128:
        raise ValueError('rows must contain 1-128 rows')
    if width is None:
        if not isinstance(value[0],list):raise ValueError('each row must be a list')
        width=len(value[0])
    if not 1 <= width <= 8:raise ValueError('rows require 1-8 features')
    result=[]
    for row in value:
        if not isinstance(row,list) or len(row)!=width:raise ValueError('inconsistent feature width')
        result.append([_number(v,-1e6,1e6,'coordinate') for v in row])
    return result


def _params(value):
    keys={'length_scale','signal_variance','noise_variance','jitter','prior_mean'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('kernel requires '+','.join(sorted(keys)))
    return {'length_scale':_number(value['length_scale'],1e-6,1e6,'length_scale'),
            'signal_variance':_number(value['signal_variance'],1e-12,1e6,'signal_variance'),
            'noise_variance':_number(value['noise_variance'],0,1e6,'noise_variance'),
            'jitter':_number(value['jitter'],0,1e-2,'jitter'),
            'prior_mean':_number(value['prior_mean'],-1e6,1e6,'prior_mean')}


def rbf(a,b,length_scale,signal_variance):
    return signal_variance*math.exp(-.5*math.fsum(((u-v)/length_scale)**2 for u,v in zip(a,b)))


def _dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def _cholesky(a):
    n=len(a);lower=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            remainder=a[i][j]-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if not math.isfinite(remainder):raise ValueError('nonfinite Cholesky arithmetic')
            if i==j:
                if remainder<=0:raise ValueError('nonpositive Cholesky pivot; revise noise or explicit jitter')
                lower[i][j]=math.sqrt(remainder)
            else:lower[i][j]=remainder/lower[j][j]
    return lower


def _forward(lower,b):
    x=[]
    for i,row in enumerate(lower):x.append((b[i]-_dot(row[:i],x))/row[i])
    return x


def _backward(lower,b):
    n=len(b);x=[0.]*n
    for i in reversed(range(n)):
        x[i]=(b[i]-math.fsum(lower[j][i]*x[j] for j in range(i+1,n)))/lower[i][i]
    return x


def _solve(lower,b):return _backward(lower,_forward(lower,b))


def _training(payload):
    if not isinstance(payload,dict) or set(payload)!={'rows','targets','kernel'}:
        raise ValueError('training requires rows, targets and kernel only')
    rows=_rows(payload['rows']);targets=payload['targets'];params=_params(payload['kernel'])
    if not isinstance(targets,list) or len(targets)!=len(rows):raise ValueError('one target per training row required')
    targets=[_number(y,-1e6,1e6,'target') for y in targets]
    return rows,targets,params


def _factor(rows,targets,p):
    n=len(rows);a=[[rbf(u,v,p['length_scale'],p['signal_variance']) for v in rows] for u in rows]
    for i in range(n):a[i][i]+=p['noise_variance']+p['jitter']
    lower=_cholesky(a);centered=[y-p['prior_mean'] for y in targets];alpha=_solve(lower,centered)
    lml=-.5*_dot(centered,alpha)-math.fsum(math.log(lower[i][i]) for i in range(n))-.5*n*math.log(2*math.pi)
    if any(not math.isfinite(v) for v in alpha+[lml]):raise ValueError('nonfinite GP solve')
    return lower,alpha,lml


def fit(payload):
    """Condition a fixed GP; do not fit hyperparameters or estimate a mean."""
    rows,targets,p=_training(payload);lower,alpha,lml=_factor(rows,targets,p)
    # Only diagonal inverse entries are needed. Each is ||L^-1 e_i||^2.
    diagonal=[]
    for i in range(len(rows)):
        basis=[float(j==i) for j in range(len(rows))]
        v=_forward(lower,basis);diagonal.append(_dot(v,v))
    means=[y-a/d for y,a,d in zip(targets,alpha,diagonal)]
    variances=[1/d for d in diagonal]
    if any(not math.isfinite(v) or v<=0 for v in variances):raise ValueError('nonfinite LOO variance')
    nll=[.5*(math.log(2*math.pi*v)+(y-m)**2/v) for y,m,v in zip(targets,means,variances)]
    if any(not math.isfinite(v) for v in means+nll):raise ValueError('nonfinite LOO arithmetic')
    return {'model':{'version':'kernellab_v1','rows':rows,'targets':targets,'kernel':p},
            'log_marginal_likelihood':lml,
            'loo':{'means':means,'effective_observation_variances':variances,'negative_log_densities':nll,
                   'mean_negative_log_density':math.fsum(nll)/len(nll),
                   'scope':'Fixed-parameter row leave-one-out under training covariance including noise AND jitter. '
                           'No hyperparameter refit, fitted empirical mean, group/time protection or calibrated-coverage guarantee.'},
            'diagnostics':{'train_rows':len(rows),'features':len(rows[0]),
                           'training_diagonal_addition':p['noise_variance']+p['jitter'],
                           'min_cholesky_diagonal':min(lower[i][i] for i in range(len(rows)))},
            'scope':'Fixed isotropic RBF GP with supplied constant mean and iid Gaussian observation noise. '
                    'Exact dense conditioning up to binary64 arithmetic, not exact scientific assumptions. '
                    'Jitter perturbs training covariance and marginal likelihood; no silent adaptive jitter or hyperparameter optimization.'}


def predict(model,rows,full_covariance=False):
    if not isinstance(model,dict) or set(model)!={'version','rows','targets','kernel'} or model['version']!='kernellab_v1':
        raise ValueError('invalid kernellab_v1 model schema')
    if type(full_covariance) is not bool:raise ValueError('full_covariance must be boolean')
    train,targets,p=_training({k:model[k] for k in ('rows','targets','kernel')})
    query=_rows(rows,len(train[0]));lower,alpha,_=_factor(train,targets,p)
    cross=[[rbf(u,v,p['length_scale'],p['signal_variance']) for v in train] for u in query]
    solved=[_forward(lower,k) for k in cross]
    means=[p['prior_mean']+_dot(k,alpha) for k in cross]
    variance=[];clamped=0;tolerance=max(1e-12,p['signal_variance']*1e-10)
    for v in solved:
        raw=p['signal_variance']-_dot(v,v)
        if raw < -tolerance:raise ValueError('posterior variance negative beyond numerical tolerance')
        if raw<0:clamped+=1
        variance.append(max(0.,raw))
    observation=[v+p['noise_variance'] for v in variance]
    if any(not math.isfinite(v) for v in means+variance+observation):raise ValueError('nonfinite posterior arithmetic')
    result={'means':means,'latent_variances':variance,'observation_variances':observation,
            'diagnostics':{'clamped_latent_diagonals':clamped,'negative_variance_tolerance':tolerance},
            'scope':'Observation variances add noise_variance, not training-only jitter. Pointwise Gaussian model distributions, '
                    'not distribution-free confidence/coverage guarantees. Input scaling and hyperparameters are caller-owned.'}
    if full_covariance:
        covariance=[[0.]*len(query) for _ in query]
        for i,u in enumerate(query):
            covariance[i][i]=variance[i]
            for j in range(i):
                v=rbf(u,query[j],p['length_scale'],p['signal_variance'])-_dot(solved[i],solved[j])
                if not math.isfinite(v):raise ValueError('nonfinite posterior covariance')
                covariance[i][j]=covariance[j][i]=v
        result['latent_covariance']=covariance
        result['observation_covariance']=[list(row) for row in covariance]
        for i in range(len(query)):result['observation_covariance'][i][i]+=p['noise_variance']
    return result
