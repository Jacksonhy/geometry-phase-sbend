"""Additional independent scalar metric and input-contract verification."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,csv,copy,math
from pathlib import Path
import numpy as np
from predict import R,BASE,read,save,records,field_free_guard
B=BASE/'research_sbend_blind_local_phase_2026-11-15'
def run():
 rr=records(R/'blind_geometries.json');rows=[]
 with (R/'endpoints_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 # Scalar real/imag error accumulation, separate from original vector metric code.
 checks=[]
 for index in [0,35,71]:
  rec=rr[index];gid=rec['geometry_id']
  with np.load(B/'data'/gid/'sample.npz') as z:e=z['Ez']
  for method in ['linear','arc_only','local_tangent','path_v2']:
   with np.load(R/'replays'/f'seed2026_{method}.npz') as z:p=z['prediction'][index]
   numerator=math.fsum((float(a.real)-float(b.real))**2+(float(a.imag)-float(b.imag))**2 for a,b in zip(p.flat,e.flat));denominator=math.fsum(float(a.real)**2+float(a.imag)**2 for a in e.flat);v=math.sqrt(numerator/denominator)
   expected=next(float(r['value']) for r in rows if r['geometry_id']==gid and r['seed']=='2026' and r['method']==method and r['endpoint']=='whole_rel_l2');diff=abs(v-expected);assert diff<1e-12;checks.append(dict(geometry_id=gid,method=method,scalar_whole_rel_l2=v,difference=diff))
 # Guard verified only after audit truth accesses above; forbidden path is never read.
 opened=field_free_guard();blocked=False
 try:
  with (B/'data'/rr[0]['geometry_id']/'sample.npz').open('rb'):pass
 except RuntimeError as exc:blocked='Field-free path guard blocked' in str(exc)
 assert blocked
 errors=[]
 for key,value in [('input_port',3),('width',.6),('offset',1.0)]:
  a=copy.deepcopy(rr[0]);a['input_port']=value if key=='input_port' else a['input_port']
  if key!='input_port':a['params']['wg_width_um' if key=='width' else 'lateral_offset_um']=value
  path=R/('invalid_'+key+'.json');save(path,a)
  try:records(path)
  except ValueError:errors.append(key)
 assert len(errors)==3
 save(R/'independent_delivery_verification.json',dict(passed=True,scalar_checks=checks,truth_path_guard_blocks_access=blocked,invalid_inputs_rejected=errors))
if __name__=='__main__':run()
