"""Independent reconstruction/prediction checks; this audit alone reads old fields."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,csv,platform,subprocess
from pathlib import Path
import numpy as np
from predict import R,BASE,OLD,METHODS,SEEDS,read,save,digest,records,weight
B=BASE/'research_sbend_blind_local_phase_2026-11-15'
def freeze():
 rr=records(R/'blind_geometries.json');paths=set()
 for p in read(B/'protected_inputs.json'):paths.add(Path(p))
 for name in ['geometry_manifest.json','config.json','blind_per_geometry.csv','blind_paired.csv','blind_summary.csv','base_costs.json','fine_costs.json','REPORT.md','prediction_freeze.json','feature_costs.json','inference_costs.json']:
  paths.add(B/name)
 for s in SEEDS:
  for m in METHODS:paths.add(B/'predictions'/f'seed{s}_{m}.npz');paths.add(weight(s,m).with_name('summary.json'))
 for p in [OLD/'cache_complete.json',OLD/'data_complete.json',OLD/'fdtd_environment.json',OLD/'generate_data.py',OLD/'calibration.py',OLD/'prepare_training.py',OLD/'train.py'] :paths.add(p)
 for r in read(OLD/'geometry_manifest.json')['records']:
  paths.add(OLD/'data'/r['geometry_id']/'result.json')
  resource=OLD/'logs'/f"fdtd_{read(OLD/'geometry_manifest.json')['records'].index(r):03d}_resource.json"
  if resource.exists():paths.add(resource)
 for r in rr:paths.add(B/'data'/r['geometry_id']/'sample.npz');paths.add(B/'cache'/(r['geometry_id']+'.npz'))
 p=R/'protected_inputs.json'
 if p.exists():raise FileExistsError(p)
 save(p,{str(p):digest(p) for p in sorted(paths)})
 save(R/'freeze.json',dict(unix=time.time(),protocol_sha256=digest(R/'PROTOCOL.md'),predict_initial_sha256=digest(R/'predict.py'),no_new_training=True,no_new_FDTD=True,protected_files=len(paths)))
 print('FROZEN',len(paths),flush=True)
def check_inputs(folder='inputs',limit=None):
 out=[]
 for r in records(R/'blind_geometries.json')[:limit]:
  gid=r['geometry_id']
  with np.load(R/folder/(gid+'.npz'),allow_pickle=False) as a,np.load(B/'data'/gid/'sample.npz',allow_pickle=False) as e,np.load(B/'cache'/(gid+'.npz'),allow_pickle=False) as c:
   q=dict(geometry_id=gid)
   for k in ['eps','src_mask','x_um','y_um','source_center_um']:q[k+'_exact']=bool(np.array_equal(a[k],e[k]));q[k+'_max_abs_diff']=float(np.max(abs(a[k]-e[k])))
   for k in ['x']+['carrier_'+m for m in METHODS]:q[k+'_exact']=bool(np.array_equal(a[k],c[k]));q[k+'_max_abs_diff']=float(np.max(abs(a[k]-c[k])))
   out.append(q)
   assert all(v for k,v in q.items() if k.endswith('_exact')),q
 save(R/(folder+'_verification.json'),dict(passed=True,n=len(out),checks=out,verification_only_reads_truth_metadata=True,field_values_used_for_input=False))
 print('INPUTS_EXACT',len(out),flush=True)
def check_prediction(seed,method):
 p=R/'replays'/f'seed{seed}_{method}.npz'
 with np.load(p) as a,np.load(B/'predictions'/p.name) as b:
  assert np.array_equal(a['geometry_id'],b['geometry_id']) and np.array_equal(a['input_port'],b['input_port'])
  x=a['prediction'];y=b['prediction'];d=float(abs(x-y).max());rel=d/float(abs(y).max());assert rel<=1e-5,(seed,method,rel)
  return dict(seed=seed,method=method,bitwise_equal=bool(np.array_equal(x,y)),max_absolute_difference=d,max_relative_to_peak=rel,passed=True)
def verify():
 for p,h in read(R/'protected_inputs.json').items():assert digest(p)==h,p
 assert digest(R/'PROTOCOL.md')==read(R/'freeze.json')['protocol_sha256']
 save(R/'preservation_check.json',dict(passed=True,protected_files=len(read(R/'protected_inputs.json')),history_unchanged=True,training_runs=0,FDTD_runs=0,phase_or_amplitude_calibrations=0))
 print('PROTECTED_INPUTS_UNCHANGED',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='inputs':check_inputs(sys.argv[2] if len(sys.argv)>2 else 'inputs',int(sys.argv[3]) if len(sys.argv)>3 else None)
 elif sys.argv[1]=='predictions':save(R/'prediction_replay_verification.json',dict(passed=True,checks=[check_prediction(s,m) for s in SEEDS for m in METHODS]))
 elif sys.argv[1]=='verify':verify()
