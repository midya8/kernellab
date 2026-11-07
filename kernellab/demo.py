"""Deterministic development fixture, not an untouched external benchmark."""
import math
import random
from .core import fit,predict


def fixture():
    rng=random.Random(19)
    train=[[-3+6*i/14] for i in range(15)]
    truth=lambda x:math.sin(1.5*x)+.15*x
    targets=[truth(x[0])+rng.gauss(0,.12) for x in train]
    query=[[-5+10*i/100] for i in range(101)]
    test=[truth(x[0])+rng.gauss(0,.12) for x in query]
    return train,targets,query,test,[truth(x[0]) for x in query]


def run_demo():
    train,targets,query,test,truth=fixture();arms={}
    for name,length in [('length_0_8',.8),('length_5',5.)]:
        result=fit({'rows':train,'targets':targets,'kernel':{'length_scale':length,'signal_variance':1.,
                     'noise_variance':.12**2,'jitter':1e-10,'prior_mean':0.}})
        pred=predict(result['model'],query)
        means=pred['means'];variances=pred['observation_variances']
        nll=[.5*(math.log(2*math.pi*v)+(y-m)**2/v) for y,m,v in zip(test,means,variances)]
        inner=[i for i,x in enumerate(query) if -3<=x[0]<=3];outer=[i for i,x in enumerate(query) if abs(x[0])>3]
        arms[name]={'length_scale':length,'test_observation_rmse':math.sqrt(math.fsum((y-m)**2 for y,m in zip(test,means))/len(test)),
                    'mean_test_observation_nll':math.fsum(nll)/len(nll),
                    'test_pointwise_2sd_coverage':sum(abs(y-m)<=2*math.sqrt(v) for y,m,v in zip(test,means,variances))/len(test),
                    'inner_mean_latent_variance':math.fsum(pred['latent_variances'][i] for i in inner)/len(inner),
                    'outer_mean_latent_variance':math.fsum(pred['latent_variances'][i] for i in outer)/len(outer),
                    'log_marginal_likelihood':result['log_marginal_likelihood'],
                    'loo_mean_nll':result['loo']['mean_negative_log_density']}
    prior_nll=math.fsum(.5*(math.log(2*math.pi*1.0144)+y*y/1.0144) for y in test)/len(test)
    return {'seed':19,'train_rows':15,'test_rows':101,'arms':arms,
            'unconditioned_prior':{'observation_rmse':math.sqrt(math.fsum(y*y for y in test)/len(test)),
                                   'mean_observation_nll':prior_nll,'mean':0,'observation_variance':1.0144},
            'scope':'Constructed smooth sinusoid plus linear trend with iid Gaussian noise SD .12. Two manually specified '
                    'length scales on development data, no parameter search or test-based selection routine. '
                    'Test points span [-5,5]; training [-3,3]. 2-SD coverage is descriptive for one correlated function realization, '
                    'not calibration proof or a general GP ranking. Wrong kernel/mean can be confidently wrong.'}
