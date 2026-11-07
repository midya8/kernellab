"""Regenerate the public SVG from actual model predictions, no graphics package."""
import math
from pathlib import Path
from kernellab import fit,predict
from kernellab.demo import fixture,run_demo


def render():
    train,targets,query,_,truth=fixture();measured=run_demo();parts=[]
    def add(s):parts.append(s)
    add('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="940" viewBox="0 0 900 940" role="img" aria-labelledby="title desc">')
    add('<title id="title">KernelLab: uncertainty depends on model assumptions</title>')
    add('<desc id="desc">Two GP length scales on the same noisy sinusoidal fixture. Mean predictions, pointwise observation bands and latent truth. Both panels share axes.</desc>')
    add('<rect width="900" height="940" fill="#fff"/>')
    add('<g font-family="Arial,sans-serif" fill="#251f21">')
    def text(x,y,s,size=16,weight='400'):
        add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{s}</text>')
    text(40,40,'Uncertainty depends on the model',28,'600')
    text(40,67,'Same 15 training observations. Fixed RBF kernels. No parameter search.',16)
    for panel,length in enumerate((.8,5.)):
        top=130+panel*370;left=80;w=760;h=255
        key='length_0_8' if panel==0 else 'length_5';metrics=measured['arms'][key]
        r=predict(fit({'rows':train,'targets':targets,'kernel':{'length_scale':length,'signal_variance':1.,
                  'noise_variance':.0144,'jitter':1e-10,'prior_mean':0.}})['model'],query)
        xs=lambda v:left+(v+5)/10*w
        ys=lambda v:top+(4-v)/8*h
        text(40,top-32,f'Length scale {length:g}',21,'600')
        text(310,top-32,f'Observation RMSE {metrics["test_observation_rmse"]:.3f}  |  mean NLL {metrics["mean_test_observation_nll"]:.3f}',16)
        for y in (-4,-2,0,2,4):
            add(f'<line x1="{left}" x2="{left+w}" y1="{ys(y)}" y2="{ys(y)}" stroke="#eae9ea"/>')
            text(49,ys(y)+5,str(y),14)
        for x in (-5,-3,0,3,5):
            text(xs(x)-8,top+h+24,str(x),14)
        for edge in (-3,3):add(f'<line x1="{xs(edge)}" x2="{xs(edge)}" y1="{top}" y2="{top+h}" stroke="#c0bebf" stroke-dasharray="4 4"/>')
        band=[(q[0],m+2*math.sqrt(v)) for q,m,v in zip(query,r['means'],r['observation_variances'])]
        band+=list(reversed([(q[0],m-2*math.sqrt(v)) for q,m,v in zip(query,r['means'],r['observation_variances'])]))
        add('<polygon points="'+' '.join(f'{xs(x):.2f},{ys(y):.2f}' for x,y in band)+'" fill="#73a89a" fill-opacity=".25"/>')
        def line(values,color,dash=''):
            add('<polyline points="'+' '.join(f'{xs(q[0]):.2f},{ys(y):.2f}' for q,y in zip(query,values))+f'" fill="none" stroke="{color}" stroke-width="2.4" stroke-dasharray="{dash}"/>')
        line(truth,'#585254','6 4');line(r['means'],'#34715f')
        for q,y in zip(train,targets):add(f'<circle cx="{xs(q[0]):.2f}" cy="{ys(y):.2f}" r="3.5" fill="#251f21"/>')
        text(400,top+h+47,'Input x (unitless)',14)
    text(40,843,'Teal: posterior mean and +/- 2 observation SD. Dashed: latent truth.',16)
    text(40,869,'Dots: training observations. Vertical guides: training domain [-3, 3].',16)
    text(40,897,'Pointwise model bands, not calibrated coverage guarantees. Development fixture, seed 19.',15)
    text(40,920,'The long length scale is a misspecification example, not a universal GP ranking.',15)
    add('</g></svg>')
    return '\n'.join(parts)+'\n'


if __name__=='__main__':
    Path('assets/posterior.svg').write_text(render(),encoding='utf-8')
