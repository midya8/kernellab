import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from kernellab import fit,suggest,expected_improvement
from kernellab.cli import main
from kernellab.demo import run_optimization_demo
from test_core import payload


def model():
    d=payload();d['kernel']['noise_variance']=0
    return fit(d)['model']


class TestAcquisition(unittest.TestCase):
    def test_analytic_equal_mean_and_degenerate_limits(self):
        self.assertAlmostEqual(expected_improvement(2,1,2),1/math.sqrt(2*math.pi))
        self.assertEqual(expected_improvement(4,0,2),2)
        self.assertEqual(expected_improvement(1,0,2),0)
        self.assertEqual(expected_improvement(4,0,2,xi=.5),1.5)

    def test_independent_gaussian_quadrature_reference(self):
        for mu,var,best,xi in [(0,1,0,0),(1,.25,.4,.1),(-1,2,.5,0),(.2,.09,.4,.05)]:
            step=.001;sigma=math.sqrt(var)
            # Midpoint integral of max(mu+sigma*z-best-xi,0) phi(z).
            reference=math.fsum(max(mu+sigma*(-10+(i+.5)*step)-best-xi,0)*
                math.exp(-.5*(-10+(i+.5)*step)**2)/math.sqrt(2*math.pi)*step for i in range(20000))
            self.assertAlmostEqual(expected_improvement(mu,var,best,xi),reference,delta=4e-8)

    def test_ei_translation_scaling_and_exploration_offset(self):
        base=expected_improvement(.3,.7,.5,.1)
        self.assertAlmostEqual(expected_improvement(10.3,.7,10.5,.1),base)
        self.assertAlmostEqual(expected_improvement(.6,2.8,1,.2),2*base)
        self.assertGreater(base,expected_improvement(.3,.7,.5,.2))

    def test_extreme_gaussian_tails_remain_finite_nonnegative(self):
        self.assertEqual(expected_improvement(-1e6,1,1e6),0)
        self.assertEqual(expected_improvement(1e6,1,-1e6),2e6)
        for mu in (-40,-12,-8,-2,0,8,40):
            self.assertGreaterEqual(expected_improvement(mu,1,0),0)
        self.assertEqual(expected_improvement(2,5e-324,0),2)

    def test_scalar_parameter_validation(self):
        for args in [(True,1,0),(.2,-1,0),(.2,1,math.nan),(.2,1,0,-1),(.2,1,0,math.inf),(.2,1e20,0)]:
            with self.assertRaises(ValueError):expected_improvement(*args)

    def test_known_points_excluded_and_no_input_mutation(self):
        m=model();pool=[[0],[.5],[1],[1.5],[2]];before=copy.deepcopy((m,pool))
        r=suggest(m,pool,.01)
        self.assertEqual([x['index'] for x in r['candidates'] if x['excluded_already_evaluated']],[0,2,4])
        self.assertTrue(all(r['candidates'][i]['expected_improvement'] is None for i in (0,2,4)))
        self.assertIn(r['suggestion']['index'],(1,3));self.assertEqual((m,pool),before)
        expected=max((row for row in r['candidates'] if row['expected_improvement'] is not None),key=lambda row:row['expected_improvement'])
        self.assertEqual(r['suggestion'],expected)

    def test_exact_ties_and_zero_ei_still_choose_first_eligible(self):
        d=payload();d.update(rows=[[0]],targets=[0]);d['kernel'].update(prior_mean=0,noise_variance=0)
        m=fit(d)['model'];pool=[[0],[-1],[1]]
        for xi in (0,1e6):
            r=suggest(m,pool,xi);self.assertEqual(r['suggestion']['index'],1)
            self.assertEqual(r['candidates'][1]['expected_improvement'],r['candidates'][2]['expected_improvement'])
            self.assertEqual(r['status'],'suggested')

    def test_exhausted_pool_is_not_false_next_action(self):
        r=suggest(model(),[[0],[2]])
        self.assertIsNone(r['suggestion']);self.assertEqual(r['status'],'exhausted')

    def test_noisy_model_and_duplicate_candidate_pool_rejected(self):
        with self.assertRaisesRegex(ValueError,'noise_variance=0'):suggest(fit(payload())['model'],[[.5]])
        for pool in ([[.5],[.5]],[[0],[False]],[[0,1]]):
            with self.assertRaises(ValueError):suggest(model(),pool)
        with self.assertRaises(ValueError):suggest(model(),[[.5]],xi=True)

    def test_suggest_cli_and_error_no_partial_json(self):
        for request,expected in [({'model':model(),'candidates':[[.5],[1.5]],'xi':.01},0),
                                 ({'model':model(),'candidates':[[0]],'extra':1},2)]:
            out,err=io.StringIO(),io.StringIO()
            with patch('sys.stdin',io.StringIO(json.dumps(request))),redirect_stdout(out),redirect_stderr(err):
                self.assertEqual(main(['suggest','--input','-']),expected)
            if expected==0:self.assertIsNotNone(json.loads(out.getvalue())['suggestion'])
            else:self.assertEqual(out.getvalue(),'');self.assertTrue(err.getvalue().startswith('error:'))

    def test_optimization_demo_budget_control_and_actual_regret(self):
        r=run_optimization_demo();self.assertEqual(r,run_optimization_demo())
        oracle=r['report_only_grid_oracle']['value']
        objective=lambda x:math.sin(2*x)+.4*math.cos(5*x)-.05*x*x
        self.assertAlmostEqual(oracle,max(objective(-4+i/10) for i in range(81)))
        for name,arm in r['arms'].items():
            self.assertEqual(arm['evaluations'],9);self.assertEqual(len(set(arm['evaluated_grid_indices'])),9)
            self.assertEqual(arm['evaluated_grid_indices'][:3],[0,40,80])
            self.assertEqual(len(arm['trace']),6)
            for index,target in zip(arm['evaluated_grid_indices'],arm['observed_targets']):
                self.assertEqual(target,objective(-4+index/10))
            self.assertAlmostEqual(arm['finite_grid_simple_regret'],oracle-arm['best_observed'])
        self.assertAlmostEqual(r['arms']['expected_improvement']['finite_grid_simple_regret'],.00016260599426676237)
        self.assertAlmostEqual(r['arms']['farthest_point']['finite_grid_simple_regret'],.07294389131786527)
        self.assertAlmostEqual(r['arms']['uniform_random_seed29']['finite_grid_simple_regret'],.3994734394488937)
