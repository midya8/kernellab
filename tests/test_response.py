import copy,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal as D,localcontext
from kernellab.core import fit,predict,rbf
from kernellab.response import inspect
from kernellab.response_export import saved,build

class ResponseTests(unittest.TestCase):
    def test_archive_original_reproduction(self):self.assertEqual(build(),saved())
    def test_202queries_3030weight_edits(self):
        r=saved()
        for a in r['arms'].values():
            m=a['report']['model'];self.assertEqual(len(a['references']),101)
            for i,(row,ref) in enumerate(zip(r['query'],a['references'])):
                self.assertEqual(inspect(m,row),ref)
                for j,curve in enumerate(a['unit_target_edit_curves']):
                    self.assertAlmostEqual(ref['original_mean']+ref['weights'][j],curve['means'][i],places=11)
                    self.assertEqual(ref['latent_variance'],curve['latent_variances'][i]);self.assertEqual(ref['observation_variance'],curve['observation_variances'][i])
    def test_high_precision_independent_gauss_jordan(self):
        with localcontext() as ctx:
            ctx.prec=70
            for n in range(1,7):
                for length in [.35,1.7]:
                    m=fit(dict(rows=[[i/3] for i in range(n)],targets=[math.sin(i) for i in range(n)],kernel=dict(length_scale=length,signal_variance=1.3,noise_variance=.07,jitter=1e-9,prior_mean=.2)))['model']
                    for q in [-1.,.125,1.25,3.]:
                        p=m['kernel'];A=[[D.from_float(rbf(u,v,p['length_scale'],p['signal_variance'])+((p['noise_variance']+p['jitter']) if i==j else 0.)) for j,v in enumerate(m['rows'])] for i,u in enumerate(m['rows'])];b=[D.from_float(rbf([q],v,p['length_scale'],p['signal_variance'])) for v in m['rows']]
                        for i in range(n):
                            pivot=A[i][i];A[i]=[v/pivot for v in A[i]];b[i]/=pivot
                            for j in range(n):
                                if j==i:continue
                                c=A[j][i];A[j]=[v-c*w for v,w in zip(A[j],A[i])];b[j]-=c*b[i]
                        actual=inspect(m,[q]);self.assertLess(actual['normalized_solve_residual'],1e-10)
                        for w,v in zip(actual['weights'],b):self.assertAlmostEqual(w,float(v),places=12)
    def test_signed_weights_and_nonbarycentric_sum(self):
        a=saved()['arms']['length_0_8'];weights=[r['weights'] for r in a['references']]
        self.assertTrue(any(min(w)<-.01 for w in weights));self.assertTrue(any(abs(sum(w)-1)>.1 for w in weights))
    def test_multiple_edits_and_finite_differences(self):
        r=saved()
        for a in r['arms'].values():
            m=a['report']['model'];delta=[(-1)**j*(j+1)/30 for j in range(15)]
            for row in [[-5.],[-1.25],[0.],[2.6],[5.]]:
                out=inspect(m,row,delta);self.assertEqual(out['latent_variance'],out['edited_latent_variance']);self.assertEqual(out['observation_variance'],out['edited_observation_variance'])
                self.assertAlmostEqual(out['original_mean']+out['mean_shift'],out['edited_mean'],places=11)
                for j in [0,7,14]:
                    d=[0.]*15;d[j]=.01;u=inspect(m,row,d);self.assertAlmostEqual((u['edited_mean']-u['original_mean'])/.01,out['weights'][j],places=10)
    def test_validation_and_nonmutation(self):
        m=saved()['arms']['length_0_8']['report']['model'];old=copy.deepcopy(m);row=[.1];delta=[0.]*15;inspect(m,row,delta);self.assertEqual(m,old);self.assertEqual(row,[.1]);self.assertEqual(delta,[0.]*15)
        for d in [[],[True]*15,[float('nan')]*15,['1']*15,[1e6]*15]:
            with self.assertRaises(ValueError):inspect(m,row,d)
        with self.assertRaises(ValueError):inspect(dict(m,extra=1),row)
        with self.assertRaises(ValueError):inspect(m,[False])
    def test_cli_duplicate_key_and_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'request.json';p.write_text('{"model":{},"row":[],"row":[]}');r=subprocess.run([sys.executable,'-m','kernellab','response','--input',str(p)],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertIn('duplicate',r.stderr);self.assertEqual(r.stdout,'')
            m=saved()['arms']['length_0_8']['report']['model'];p.write_text(json.dumps(dict(model=m,row=[0.])));r=subprocess.run([sys.executable,'-m','kernellab','response','--input',str(p)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len(json.loads(r.stdout)['weights']),15)
