import json,csv,cmath,math,importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
s=importlib.util.spec_from_file_location('local_analysis',R/'analyze.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
with (R/'per_geometry_normal_pairs.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
lookup={(int(r['seed']),r['method'],r['geometry_id'],r['region'],float(r['gate']),float(r['eta'])):r for r in rows};coverage={r['geometry_id']:r for r in read(R/'geometry_coverage.json')};records=[r for r in a.read(a.OLD/'geometry_manifest.json')['records'] if r['split']=='test']
def bilinear(field,points,x,y):
 xx=(points[:,0]-x[0])/(x[1]-x[0]);yy=(points[:,1]-y[0])/(y[1]-y[0]);i=np.floor(xx).astype(int);j=np.floor(yy).astype(int);u=xx-i;v=yy-j
 assert (i>=0).all() and (i+1<field.shape[1]).all() and (j>=0).all() and (j+1<field.shape[0]).all()
 return (1-u)*(1-v)*field[j,i]+u*(1-v)*field[j,i+1]+(1-u)*v*field[j+1,i]+u*v*field[j+1,i+1]
checks=0;maxphase=0.;maxcoherence=0.
for rec in [records[i] for i in [0,49,99]]:
 with np.load(a.OLD/'data'/rec['geometry_id']/'sample.npz',allow_pickle=False) as z:data={k:z[k] for k in ['x_um','y_um']};e=z['Ez'].astype(np.complex128)
 poly,bend=a.G['centerline_from_params'](rec['params']);plus,minus,angles=a.paircoords(rec,data,poly,bend,.5);maxamp=coverage[rec['geometry_id']]['bend_core_max_amp'];ep=bilinear(e,plus,**{'x':data['x_um'],'y':data['y_um']});em=bilinear(e,minus,data['x_um'],data['y_um']);good=(abs(ep)>=.05*maxamp)&(abs(em)>=.05*maxamp);ct=ep[good]*em[good].conj()
 for seed in a.SEEDS:
  for method in a.METHODS:
   with np.load(a.predpath(seed,method),allow_pickle=False) as z:ii=z['geometry_id'].tolist().index(rec['geometry_id']);p=z['prediction'][ii].astype(np.complex128)
   pp=bilinear(p,plus,data['x_um'],data['y_um']);pm=bilinear(p,minus,data['x_um'],data['y_um']);cp=pp[good]*pm[good].conj();weights=[abs(z) for z in ct];errors=[abs(cmath.phase(q/z)) if abs(q)>1e-24*maxamp**2 else math.pi for q,z in zip(cp,ct)];phase=sum(w*v for w,v in zip(weights,errors))/sum(weights)*180/math.pi;coh=math.sqrt(sum(abs(q-z)**2 for q,z in zip(cp,ct))/sum(abs(z)**2 for z in ct));r=lookup[seed,method,rec['geometry_id'],'normal_pair_all',.05,.5];maxphase=max(maxphase,abs(phase-float(r['normal_relative_phase_wmae_deg'])));maxcoherence=max(maxcoherence,abs(coh-float(r['coherence_complex_rel_l2'])));assert int(r['pairs'])==int(good.sum());checks+=1
assert maxphase<1e-9 and maxcoherence<1e-10
out=dict(passed=True,geometry_count=3,seed_method_checks=checks,manual_bilinear_plus_scalar_complex_phase=True,max_normal_phase_difference_deg=maxphase,max_coherence_relative_error_difference=maxcoherence)
(R/'normal_pair_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
