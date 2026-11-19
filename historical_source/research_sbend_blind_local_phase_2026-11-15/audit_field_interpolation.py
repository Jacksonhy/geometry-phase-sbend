"""Supplemental field interpolation sensitivity; fixed18 grids and predictions."""
import numpy as np
from experiment import R,SEEDS,METHODS,read,save,csvsave
from evaluate_blind import RECORDS,CFG,predictions,load,A,interp,incoming
P=predictions();rows=[];truth=[]
for i,r in enumerate(RECORDS):
 gid=r['geometry_id']
 if gid not in CFG['fine_geometry_ids']:continue
 a0=load(R/'data'/gid/'sample.npz');af=load(R/'fine'/gid/'res40/sample.npz');poly,bend,masks=A.geometry(r,a0);X,Y=np.meshgrid(a0['x_um'],a0['y_um']);points=np.stack([X,Y],axis=-1);el=interp(af['Ez'],af['x_um'],af['y_um'],points,1);ec=interp(af['Ez'],af['x_um'],af['y_um'],points,3);ma=float(abs(ec[masks['bend_core']]).max());v=A.fieldmetric(el,ec,masks['bend_core'],.05,ma);truth.append(dict(geometry_id=gid,linear_vs_cubic_truth_phase_wmae_deg=v['phase_wmae_deg'],linear_vs_cubic_truth_complex_rel_l2=v['complex_rel_l2']))
 q0,_=incoming(a0);qf,_=incoming(af)
 for conv in ['raw','incident_normalized']:
  etl=el if conv=='raw' else el/qf;etc=ec if conv=='raw' else ec/qf
  for s in SEEDS:
   for m in METHODS:
    p=P[s,m][i].astype(np.complex128);p=p if conv=='raw' else p/q0;v1=A.fieldmetric(p,etl,masks['bend_core'],.05,float(abs(etl[masks['bend_core']]).max()));v3=A.fieldmetric(p,etc,masks['bend_core'],.05,float(abs(etc[masks['bend_core']]).max()));rows.append(dict(geometry_id=gid,seed=s,method=m,convention=conv,phase_wmae_linear=v1['phase_wmae_deg'],phase_wmae_cubic=v3['phase_wmae_deg'],metric_difference_deg=v3['phase_wmae_deg']-v1['phase_wmae_deg']))
paired=[]
for conv in ['raw','incident_normalized']:
 for s in SEEDS:
  a={r['geometry_id']:r for r in rows if r['seed']==s and r['method']=='local_tangent' and r['convention']==conv};b={r['geometry_id']:r for r in rows if r['seed']==s and r['method']=='arc_only' and r['convention']==conv};ids=sorted(set(a)&set(b));d1=np.mean([a[g]['phase_wmae_linear']-b[g]['phase_wmae_linear'] for g in ids]);d3=np.mean([a[g]['phase_wmae_cubic']-b[g]['phase_wmae_cubic'] for g in ids]);paired.append(dict(seed=s,convention=conv,n=len(ids),delta_LT_minus_Arc_linear=float(d1),delta_LT_minus_Arc_cubic=float(d3),delta_change_deg=float(d3-d1)))
csvsave(R/'field_interpolation_sensitivity.csv',rows);save(R/'field_interpolation_sensitivity.json',dict(geometries=18,truth_resolution=40,truth_interpolation_comparison=truth,paired_deltas=paired,mean_absolute_phase_metric_difference_deg=float(np.mean([abs(r['metric_difference_deg']) for r in rows])),max_absolute_phase_metric_difference_deg=float(max(abs(r['metric_difference_deg']) for r in rows)),no_gate_or_method_changes=True));print('FIELD_INTERPOLATION_AUDITED',paired,flush=True)
