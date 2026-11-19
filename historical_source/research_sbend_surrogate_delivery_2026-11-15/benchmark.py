"""Finite synchronized benchmarks. No training, old truth or old prediction reads."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import time,sys,subprocess,platform
from pathlib import Path
import numpy as np
from predict import R,BASE,MEEP,METHODS,records,save,read,field_free_guard,load_net,forward_records
GPU=sys.executable

def bench():
 import torch
 field_free_guard();rr=records(R/'blind_geometries.json');selected=[]
 for st in ['low','middle','high']:
  for ip in [1,2]:selected.append(next(r for r in rr if r['angle_stratum']==st and r['input_port']==ip))
 save(R/'benchmark_geometries.json',dict(records=selected,seed=2026,selection='first input1/2 in low/middle/high stratum; frozen protocol'))
 env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cpu=platform.processor(),platform=platform.platform(),torch_threads=4,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),cudnn_benchmark=False,tf32=True)
 save(R/'benchmark_environment.json',env);cost=[]
 for m in METHODS:
  net,load=load_net(2026,m)
  # Warm GPU using same shape; exclude warmup from steady-state timings, retain its cost.
  tw=time.perf_counter();forward_records(net,rr[:2],R/'inputs',m,2);warmup=time.perf_counter()-tw
  for batch in [1,2]:
   repeats=[]
   for rep in range(3):
    t=time.perf_counter();p,parts,peak=forward_records(net,rr,R/'inputs',m,batch);infer_seconds=time.perf_counter()-t;ts=time.perf_counter()
    dest=R/'benchmark_outputs'/f'{m}_b{batch}_r{rep}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,prediction=p,geometry_id=np.array([r['geometry_id'] for r in rr]));export=time.perf_counter()-ts
    repeats.append(dict(repeat=rep,infer_including_input_io_seconds=infer_seconds,output_export_seconds=export,total_seconds=infer_seconds+export,parts=parts,peak_allocated_MiB=peak))
   cost.append(dict(method=m,mode='batch',batch_size=batch,samples=72,load=load,warmup_seconds=warmup,repeats=repeats))
   save(R/'benchmark_progress.json',cost);print('BATCH_DONE',m,batch,flush=True)
  singles=[]
  for j,r in enumerate(selected):
   man=R/'benchmark_inputs'/f'{m}_{j}.json';save(man,r);folder=man.with_suffix('');t=time.perf_counter()
   with (R/'logs'/f'warm_{m}_{j}.log').open('w') as log:subprocess.run([MEEP,'-B',str(R/'predict.py'),'--geometry-json',str(man),'--prepare-only','--output',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
   prep=time.perf_counter()-t;p,parts,peak=forward_records(net,[r],folder,m,1);ts=time.perf_counter();dest=R/'benchmark_outputs'/f'{m}_single{j}.npz';np.savez_compressed(dest,prediction=p);export=time.perf_counter()-ts
   singles.append(dict(geometry_id=r['geometry_id'],total_seconds=time.perf_counter()-t,prepare_process_seconds=prep,parts=parts,export_seconds=export,peak_allocated_MiB=peak))
  cost.append(dict(method=m,mode='resident_model_single',samples=1,load=load,warmup_seconds=warmup,repeats=singles));save(R/'benchmark_progress.json',cost)
  del net,p;torch.cuda.empty_cache();print('WARM_SINGLE_DONE',m,flush=True)
 save(R/'benchmark_resident.json',cost)
 cold=[]
 for m in METHODS:
  for rep in range(3):
   dest=R/'cold_outputs'/f'{m}_{rep}.npz';dest.parent.mkdir(exist_ok=True);t=time.perf_counter()
   with (R/'logs'/f'cold_{m}_{rep}.log').open('w') as log:subprocess.run([GPU,'-B',str(R/'predict.py'),'--geometry-json',str(R/'example_geometry.json'),'--method',m,'--seed','2026','--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
   cold.append(dict(method=m,repeat=rep,total_process_seconds=time.perf_counter()-t,record=read(dest.with_suffix('.json')),workflow=read(dest.with_suffix('.workflow.json'))));save(R/'benchmark_cold.json',cold)
  print('COLD_DONE',m,flush=True)
 save(R/'benchmarks_complete.json',dict(passed=True,unmeasured_energy=True,hardware_specific_wall_time=True,training_runs=0,FDTD_runs=0))
if __name__=='__main__':bench()
