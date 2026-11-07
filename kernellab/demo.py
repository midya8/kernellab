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


def run_optimization_demo():
    """Equal-budget policies on a fixed, public deterministic grid objective."""
    from .acquisition import suggest
    objective=lambda x:math.sin(2*x)+.4*math.cos(5*x)-.05*x*x
    grid=[[-4+i/10] for i in range(81)];arms={}
    for policy in ('expected_improvement','farthest_point','uniform_random_seed29'):
        rng=random.Random(29)
        indices=[0,40,80];targets=[objective(grid[i][0]) for i in indices];trace=[]
        for _ in range(6):
            if policy=='expected_improvement':
                model=fit({'rows':[grid[i] for i in indices],'targets':targets,
                    'kernel':{'length_scale':.7,'signal_variance':1.,'noise_variance':0.,'jitter':1e-8,'prior_mean':0.}})['model']
                report=suggest(model,grid,xi=.01);choice=report['suggestion'];index=choice['index']
                policy_score=choice['expected_improvement']
            elif policy=='uniform_random_seed29':
                available=[i for i in range(len(grid)) if i not in indices]
                index=rng.choice(available);policy_score=None
            else:
                available=[i for i in range(len(grid)) if i not in indices]
                # Same grid and budget, deterministic space-filling policy independent of objective values.
                index=max(available,key=lambda i:min(abs(grid[i][0]-grid[j][0]) for j in indices))
                policy_score=min(abs(grid[index][0]-grid[j][0]) for j in indices)
            assert index not in indices
            target=objective(grid[index][0]);indices.append(index);targets.append(target)
            trace.append({'evaluation':len(indices),'grid_index':index,'x':grid[index][0],
                          'observed_target':target,'best_observed':max(targets),'policy_score':policy_score})
        arms[policy]={'evaluations':len(indices),'initial_grid_indices':[0,40,80],
                      'evaluated_grid_indices':indices,'observed_targets':list(targets),'best_observed':max(targets),
                      'best_x':grid[indices[max(range(len(targets)),key=targets.__getitem__)]][0],'trace':trace}
    # Full-grid oracle is computed for reporting only, after all policies finish.
    oracle_index=max(range(len(grid)),key=lambda i:objective(grid[i][0]));oracle=objective(grid[oracle_index][0])
    for arm in arms.values():arm['finite_grid_simple_regret']=oracle-arm['best_observed']
    return {'objective':'sin(2*x)+0.4*cos(5*x)-0.05*x*x (maximize)',
            'grid':{'rows':81,'lower':-4,'upper':4,'step':.1},'budget_including_initial':9,'arms':arms,
            'report_only_grid_oracle':{'x':grid[oracle_index][0],'value':oracle},
            'scope':'Selected deterministic development objective, not a benchmark or universal optimizer ranking. '
                    'All policies start at grid indices0/40/80 and get6 more evaluations; same81-point pool. '
                    'Farthest-point uses first-index ties and no objective values for selection. '
                    'EI uses fixed length.7/signal1/noise0/jitter1e-8/mean0/xi.01, no parameter sweep. '
                    'Only the built-in offline demo evaluates the objective; suggest itself never does. '
                    'Seed29 random samples without replacement, no objective values for selection. Oracle grid values are report-only after all loops; no continuous-global-optimum guarantee.'}
