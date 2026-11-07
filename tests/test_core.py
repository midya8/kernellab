import copy
import json
import math
import unittest
from kernellab import fit,predict
from kernellab.core import _cholesky,_solve,rbf
from unittest.mock import patch


def payload():
    return {'rows':[[0],[1],[2]],'targets':[1,-.5,.7],
            'kernel':{'length_scale':.8,'signal_variance':1.3,'noise_variance':.2,'jitter':1e-9,'prior_mean':.3}}


def inverse(a):
    """Independent tiny Gauss-Jordan reference, never used by production code."""
    n=len(a);aug=[list(row)+[float(i==j) for j in range(n)] for i,row in enumerate(a)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(aug[j][i]));aug[i],aug[pivot]=aug[pivot],aug[i]
        divisor=aug[i][i];aug[i]=[v/divisor for v in aug[i]]
        for j in range(n):
            if j!=i:
                multiplier=aug[j][i];aug[j]=[x-multiplier*y for x,y in zip(aug[j],aug[i])]
    return [row[n:] for row in aug]


class TestCore(unittest.TestCase):
    def test_rbf_diagonal_symmetry_and_length(self):
        self.assertEqual(rbf([1,2],[1,2],.4,2),2)
        self.assertAlmostEqual(rbf([0],[1],1,2),2*math.exp(-.5))
        self.assertEqual(rbf([1,2],[2,0],2,3),rbf([2,0],[1,2],2,3))
        self.assertGreater(rbf([0],[1],2,1),rbf([0],[1],.5,1))

    def test_cholesky_reconstruct_and_solve(self):
        a=[[4,2,0],[2,5,1],[0,1,2]];l=_cholesky(a)
        for i in range(3):
            for j in range(3):self.assertAlmostEqual(sum(l[i][k]*l[j][k] for k in range(3)),a[i][j])
        b=[1,2,3];x=_solve(l,b)
        for i in range(3):self.assertAlmostEqual(sum(a[i][j]*x[j] for j in range(3)),b[i])

    def test_single_point_analytic_mean_variance_lml(self):
        d=payload();d.update(rows=[[0]],targets=[2]);p=d['kernel'];r=fit(d);q=predict(r['model'],[[0]])
        total=1.3+.2+1e-9;center=2-.3
        self.assertAlmostEqual(q['means'][0],.3+1.3/total*center)
        self.assertAlmostEqual(q['latent_variances'][0],1.3-1.3**2/total)
        expected=-.5*(center**2/total+math.log(total)+math.log(2*math.pi))
        self.assertAlmostEqual(r['log_marginal_likelihood'],expected)
        self.assertAlmostEqual(r['loo']['means'][0],.3)
        self.assertAlmostEqual(r['loo']['effective_observation_variances'][0],total)

    def test_independent_dense_inverse_prediction_reference(self):
        d=payload();p=d['kernel'];r=fit(d);q=[[.3],[1.4]];actual=predict(r['model'],q,True)
        k=lambda a,b:rbf(a,b,p['length_scale'],p['signal_variance'])
        a=[[k(u,v)+(p['noise_variance']+p['jitter'] if i==j else 0) for j,v in enumerate(d['rows'])] for i,u in enumerate(d['rows'])]
        inv=inverse(a);center=[y-p['prior_mean'] for y in d['targets']]
        alpha=[sum(v*y for v,y in zip(row,center)) for row in inv]
        cross=[[k(u,v) for v in d['rows']] for u in q]
        for i,u in enumerate(q):
            self.assertAlmostEqual(actual['means'][i],p['prior_mean']+sum(v*a for v,a in zip(cross[i],alpha)),places=12)
            for j,v in enumerate(q):
                correction=sum(cross[i][a]*inv[a][b]*cross[j][b] for a in range(3) for b in range(3))
                self.assertAlmostEqual(actual['latent_covariance'][i][j],k(u,v)-correction,places=12)

    def test_leave_one_out_matches_explicit_refit_with_jitter(self):
        d=payload();r=fit(d)
        for i in range(3):
            subset=copy.deepcopy(d)
            subset['rows'].pop(i);subset['targets'].pop(i)
            pred=predict(fit(subset)['model'],[d['rows'][i]])
            self.assertAlmostEqual(r['loo']['means'][i],pred['means'][0],places=12)
            self.assertAlmostEqual(r['loo']['effective_observation_variances'][i],pred['observation_variances'][0]+d['kernel']['jitter'],places=12)

    def test_leave_one_out_own_target_independence(self):
        a=payload();b=copy.deepcopy(a);b['targets'][1]=1234
        x,y=fit(a),fit(b)
        self.assertAlmostEqual(x['loo']['means'][1],y['loo']['means'][1],places=10)
        self.assertEqual(x['loo']['effective_observation_variances'],y['loo']['effective_observation_variances'])

    def test_observation_noise_not_jitter_added_at_prediction(self):
        d=payload();d['kernel']['jitter']=.01;q=predict(fit(d)['model'],[[.5],[.5]],True)
        for i in range(2):self.assertAlmostEqual(q['observation_variances'][i]-q['latent_variances'][i],.2)
        self.assertEqual(q['observation_covariance'][0][1],q['latent_covariance'][0][1])
        self.assertAlmostEqual(q['observation_covariance'][0][0]-q['latent_covariance'][0][0],.2)

    def test_duplicate_noiseless_failure_and_explicit_jitter_recovery(self):
        d=payload();d.update(rows=[[0],[0]],targets=[1,1]);d['kernel'].update(noise_variance=0,jitter=0)
        with self.assertRaisesRegex(ValueError,'pivot'):fit(d)
        d['kernel']['jitter']=1e-8;r=fit(d)
        self.assertEqual(r['diagnostics']['training_diagonal_addition'],1e-8)
        self.assertGreater(predict(r['model'],[[0]])['latent_variances'][0],0)

    def test_noiseless_training_interpolates(self):
        d=payload();d['kernel'].update(noise_variance=0,jitter=0);p=predict(fit(d)['model'],d['rows'])
        for expected,actual in zip(d['targets'],p['means']):self.assertAlmostEqual(actual,expected,places=12)
        for v in p['latent_variances']:self.assertLess(v,1e-12)

    def test_far_away_returns_supplied_prior(self):
        d=payload();p=predict(fit(d)['model'],[[1e6]])
        self.assertEqual(p['means'],[.3]);self.assertEqual(p['latent_variances'],[1.3])

    def test_variance_independent_of_target_values(self):
        a=payload();b=copy.deepcopy(a);b['targets']=[100,-20,1]
        x=predict(fit(a)['model'],[[.5],[4]],True);y=predict(fit(b)['model'],[[.5],[4]],True)
        self.assertEqual(x['latent_covariance'],y['latent_covariance']);self.assertNotEqual(x['means'],y['means'])

    def test_permutation_invariance(self):
        a=payload();b=copy.deepcopy(a);b['rows'].reverse();b['targets'].reverse()
        x,y=fit(a),fit(b)
        self.assertAlmostEqual(x['log_marginal_likelihood'],y['log_marginal_likelihood'],places=12)
        p,q=predict(x['model'],[[.3]]),predict(y['model'],[[.3]])
        self.assertAlmostEqual(p['means'][0],q['means'][0],places=12)

    def test_serialized_model_and_no_input_mutation(self):
        d=payload();before=copy.deepcopy(d);r=fit(d);model=copy.deepcopy(r['model'])
        p=predict(r['model'],[[.2]],True);q=predict(json.loads(json.dumps(r['model'])),[[.2]],True)
        self.assertEqual(p,q);self.assertEqual(d,before);self.assertEqual(r['model'],model)

    def test_strict_training_and_artifact_schema(self):
        for key in ('test','extra'):
            d=payload();d[key]=[]
            with self.assertRaises(ValueError):fit(d)
        model=fit(payload())['model']
        for key in ('targets','version','kernel'):
            d=copy.deepcopy(model);d.pop(key)
            with self.assertRaises(ValueError):predict(d,[[1]])
        for version in ('v2',True,None):
            d=copy.deepcopy(model);d['version']=version
            with self.assertRaises(ValueError):predict(d,[[1]])
        with self.assertRaises(ValueError):predict(model,[[1]],1)

    def test_bad_targets_and_numeric_values(self):
        for bad in (True,math.nan,math.inf,10**1000,'1',None):
            for place in ('target','coordinate'):
                d=payload()
                if place=='target':d['targets'][0]=bad
                else:d['rows'][0][0]=bad
                with self.subTest(place=place,bad=type(bad)),self.assertRaises(ValueError):fit(d)
        d=payload();d['targets'].pop()
        with self.assertRaises(ValueError):fit(d)

    def test_bounds_and_dimensions(self):
        for rows in ([],[[0]]*129,[[0]*9],[[0],[0,1]],[[]],[[False]]):
            d=payload();d['rows']=rows
            with self.subTest(rows=rows[:1]),self.assertRaises(ValueError):fit(d)
        model=fit(payload())['model']
        for rows in ([],[[0,1]],[[math.nan]],[[0]]*129):
            with self.assertRaises(ValueError):predict(model,rows)

    def test_kernel_parameter_bounds_and_no_defaults(self):
        for key,bad in [('length_scale',0),('signal_variance',0),('noise_variance',-1),('jitter',.02),('prior_mean',1e7),('noise_variance',True)]:
            d=payload();d['kernel'][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):fit(d)
        d=payload();d['kernel'].pop('jitter')
        with self.assertRaises(ValueError):fit(d)

    def test_full_covariance_symmetric_and_diagonal_matches(self):
        r=predict(fit(payload())['model'],[[0],[.2],[1]],True)
        for name,var in [('latent_covariance','latent_variances'),('observation_covariance','observation_variances')]:
            c=r[name]
            for i in range(3):
                self.assertEqual(c[i][i],r[var][i])
                for j in range(3):self.assertEqual(c[i][j],c[j][i])
            for v in ([1,-1,0],[1,2,3],[-1,1,4]):
                self.assertGreaterEqual(sum(v[i]*c[i][j]*v[j] for i in range(3) for j in range(3)),-1e-10)

    def test_multidimensional_rows_supported(self):
        d=payload();d['rows']=[[0,1],[1,0],[2,1]]
        r=fit(d);self.assertEqual(r['diagnostics']['features'],2)
        self.assertEqual(len(predict(r['model'],[[.2,.5]])['means']),1)

    def test_roundoff_negative_variance_is_counted(self):
        model=fit(payload())['model']
        # Isolate variance guard with a deterministic simulated solve roundoff.
        with patch('kernellab.core._factor',return_value=([[1]], [0], 0)), \
             patch('kernellab.core._forward',return_value=[math.sqrt(1.3+1e-12)]):
            p=predict(model,[[0]])
        self.assertEqual(p['latent_variances'],[0])
        self.assertEqual(p['diagnostics']['clamped_latent_diagonals'],1)

    def test_material_negative_variance_is_not_silently_repaired(self):
        model=fit(payload())['model']
        with patch('kernellab.core._factor',return_value=([[1]], [0], 0)), \
             patch('kernellab.core._forward',return_value=[2]):
            with self.assertRaisesRegex(ValueError,'negative beyond'):predict(model,[[0]])
