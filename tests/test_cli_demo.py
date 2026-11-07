import io
import json
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from kernellab.cli import main
from kernellab.demo import run_demo
from test_core import payload


class TestCLI(unittest.TestCase):
    def invoke(self,args,data):
        out,err=io.StringIO(),io.StringIO()
        with patch('sys.stdin',io.StringIO(data)),redirect_stdout(out),redirect_stderr(err):
            code=main(args)
        return code,out.getvalue(),err.getvalue()

    def test_fit_predict_cli_roundtrip(self):
        code,out,err=self.invoke(['fit','--input','-'],json.dumps(payload()))
        self.assertEqual(code,0);self.assertEqual(err,'');model=json.loads(out)['model']
        request={'model':model,'rows':[[.2],[1]],'full_covariance':True}
        code,out,err=self.invoke(['predict','--input','-'],json.dumps(request))
        self.assertEqual(code,0);self.assertIn('latent_covariance',json.loads(out))

    def test_errors_no_partial_report(self):
        for data in ('{}','{bad',json.dumps({'model':{},'rows':[[1]],'extra':3})):
            code,out,err=self.invoke(['predict','--input','-'],data)
            self.assertEqual(code,2);self.assertEqual(out,'');self.assertTrue(err.startswith('error:'))

    def test_singular_fit_cli_is_actionable(self):
        d=payload();d['rows']=[[0],[0],[0]];d['kernel'].update(noise_variance=0,jitter=0)
        code,out,err=self.invoke(['fit','--input','-'],json.dumps(d))
        self.assertEqual(code,2);self.assertEqual(out,'');self.assertIn('explicit jitter',err)

    def test_demo_deterministic_and_measured(self):
        a,b=run_demo(),run_demo();self.assertEqual(a,b)
        good,bad=a['arms']['length_0_8'],a['arms']['length_5']
        self.assertAlmostEqual(good['test_observation_rmse'],.6242366399119823)
        self.assertAlmostEqual(bad['mean_test_observation_nll'],12.13090604407389)
        self.assertGreater(good['outer_mean_latent_variance'],good['inner_mean_latent_variance'])
        self.assertLess(bad['test_pointwise_2sd_coverage'],.3)
        self.assertGreater(good['test_pointwise_2sd_coverage'],.95)
        self.assertIn('not calibration proof',a['scope'])
