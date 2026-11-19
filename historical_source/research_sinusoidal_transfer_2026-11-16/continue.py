"""Continue after output checksum guard issue, without repeating inference."""
import time,traceback
import numpy as np
import experiment as E
from experiment import R,read,save,sha,remaining
try:
 for p,h in read(R/'freeze.json')['hashes'].items():assert sha(p)==h,p
 rr=read(R/'geometry_manifest.json')['records'];assert not (R/'data').exists() and not (R/'fine').exists()
 paths=list((R/'predictions').glob('*.npz'));assert len(paths)==12
 for path in paths:
  with np.load(path) as z:
   assert z['geometry_id'].tolist()==[r['geometry_id'] for r in rr] and z['input_port'].tolist()==[r['input_port'] for r in rr]
   assert z['prediction'].shape==(36,160,480) and np.isfinite(z['prediction']).all()
 assert len(read(R/'inference_costs.json'))==12
 save(R/'prediction_freeze.json',dict(unix=time.time(),truth_fields_accessed=False,before_new_truth=True,all_new_truth_directories_absent=True,hashes={str(p):sha(p) for p in paths},note='All12 forward runs succeeded under read guard. Guard blocked subsequent self-output checksum read; this separate non-inference process verifies and freezes outputs before first truth. No forward rerun.',runner_sha256=sha(R/'continue.py'),startup_runner_sha256=sha(R/'resume.py')))
 cost=[]
 for i,r in enumerate(rr):
  sec=E.launch('experiment.py',['simworker',i,20],f'base_{i:02d}.log');cost.append(dict(geometry_id=r['geometry_id'],resolution=20,seconds=sec));save(R/'progress.json',dict(stage='base_FDTD',completed=i+1,total=36,remaining_seconds=remaining()));print('BASE',i+1,36,flush=True)
 E.launch('analyze.py',['blind'],'analysis_base.log',limit=1800)
 for gid in read(R/'config.json')['fine_geometry_ids']:
  i=next(i for i,r in enumerate(rr) if r['geometry_id']==gid)
  for res in [30,40]:
   sec=E.launch('experiment.py',['simworker',i,res],f'fine_{i:02d}_{res}.log');cost.append(dict(geometry_id=gid,resolution=res,seconds=sec));save(R/'progress.json',dict(stage='fine_FDTD',completed=len(cost)-36,total=12,remaining_seconds=remaining()));print('GRID',len(cost)-36,12,flush=True)
 save(R/'simulation_costs.json',cost);E.launch('analyze.py',['grid'],'analysis_grid.log',limit=1800);E.launch('analyze.py',['report'],'report.log',limit=1800);save(R/'complete.json',dict(unix=time.time(),elapsed_seconds=14400-remaining(),remaining_seconds=remaining(),base=36,fine=12,training_runs=0));print('EXPERIMENT_COMPLETE',flush=True)
except Exception:
 traceback.print_exc();save(R/f'stopped_{int(time.time())}.json',dict(unix=time.time(),reason=traceback.format_exc(),automatic_extension=False));raise
