"""Regenerate the equal-budget optimizer comparison from measured traces."""
from pathlib import Path
from kernellab.demo import run_optimization_demo


def render():
    result=run_optimization_demo();parts=[]
    def add(s):parts.append(s)
    add('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="610" viewBox="0 0 900 610" role="img" aria-labelledby="title desc">')
    add('<title id="title">KernelLab finite-grid optimization: same evaluation budget</title>')
    add('<desc id="desc">Best observed deterministic objective by total evaluations, including three shared initial points. EI, farthest-point and one seeded random policy. Grid oracle is report-only.</desc>')
    add('<rect width="900" height="610" fill="#fff"/><g font-family="Arial,sans-serif" fill="#251f21">')
    def text(x,y,s,size=16,weight='400'):add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{s}</text>')
    text(40,40,'Same grid. Same evaluation budget.',28,'600')
    text(40,68,'Maximize sin(2x) + 0.4 cos(5x) - 0.05x^2 on 81 fixed candidates.',16)
    left,top,w,h=80,125,755,270
    xs=lambda n:left+(n-3)/6*w
    ys=lambda y:top+(1.2-y)/1.0*h
    for y in (.2,.4,.6,.8,1.,1.2):
        add(f'<line x1="{left}" x2="{left+w}" y1="{ys(y)}" y2="{ys(y)}" stroke="#eae9ea"/>')
        text(43,ys(y)+5,f'{y:.1f}',14)
    for n in range(3,10):text(xs(n)-5,top+h+22,str(n),14)
    text(300,top+h+48,'Total objective evaluations (including 3 shared initial)',15)
    text(40,104,'Best observed objective (unitless)',15)
    oracle=result['report_only_grid_oracle']['value']
    add(f'<line x1="{left}" x2="{left+w}" y1="{ys(oracle)}" y2="{ys(oracle)}" stroke="#585254" stroke-dasharray="4 4"/>')
    text(85,ys(oracle)-10,f'Report-only grid oracle {oracle:.6f}',14)
    labels={'expected_improvement':('Expected improvement','#34715f',''),
            'farthest_point':('Farthest-point','#251f21','6 3'),
            'uniform_random_seed29':('Uniform random, seed 29','#827b7d','3 3')}
    for i,(key,(label,color,dash)) in enumerate(labels.items()):
        arm=result['arms'][key];values=[max(arm['observed_targets'][:3])]+[row['best_observed'] for row in arm['trace']]
        add('<polyline points="'+' '.join(f'{xs(j+3):.2f},{ys(value):.2f}' for j,value in enumerate(values))+f'" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
        for j,value in enumerate(values):add(f'<circle cx="{xs(j+3):.2f}" cy="{ys(value):.2f}" r="3.5" fill="{color}"/>')
        add(f'<line x1="40" x2="67" y1="{477+i*26}" y2="{477+i*26}" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
        text(80,482+i*26,f'{label}: best {arm["best_observed"]:.6f}; finite-grid regret {arm["finite_grid_simple_regret"]:.6f}',15,'600' if i==0 else '400')
    text(40,567,'One development objective and one random seed, not a general optimizer ranking.',15)
    text(40,590,'EI: fixed kernel, latent variance, zero observation noise. Oracle never enters selection.',15)
    add('</g></svg>')
    return '\n'.join(parts)+'\n'


if __name__=='__main__':Path('assets/optimization.svg').write_text(render(),encoding='utf-8')
