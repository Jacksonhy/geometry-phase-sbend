"""Time-only replay of frozen training preparation. No deployed calibration changes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,copy
from pathlib import Path
import numpy as np
from predict import R,BASE,OLD,read,save,digest
sys.path.insert(0,str(OLD))
def replay():
 import features as F
 from calibration import extract
 cal=read(OLD/'calibration.json');before=digest(OLD/'calibration.json');rr=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split'] in ['train','val']];tr=[r for r in rr if r['split']=='train'];cal_times=[]
 for rep in range(3):
  t=time.perf_counter();xx=[];yy=[];energy=0.;count=0
  for r in tr:
   with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
    e=z['Ez'];v=extract(e,z['x_um'],z['y_um'],r['params'],r['input_port']);xx.append(float(z['eps'].max()));yy.append(v['beta_rad_um']);energy+=float(np.sum(abs(e)**2));count+=e.size
  coef=np.polyfit(xx,yy,1);scale=float(np.sqrt(energy/count));coef_diff=float(np.max(abs(coef-np.array(cal['coefficients_rad_um']))));scale_diff=abs(scale-cal['scale']);print('CALIBRATION_FLOAT_DIFFERENCE',coef_diff,scale_diff,flush=True);assert coef_diff<1e-10 and scale_diff<1e-12;cal_times.append(time.perf_counter()-t)
  print('CALIBRATION_TIME_REPLAY',rep,cal_times[-1],flush=True)
 # Match original raw input reads and original compressed cache payload, without recomputing geometry.
 dest=R/'offline_cache_replay';dest.mkdir(exist_ok=False);io=[]
 for r in rr:
  t=time.perf_counter()
  with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:eps=z['eps'];src=z['src_mask'];x=z['x_um'];y=z['y_um'];source=z['source_center_um']
  raw_read=time.perf_counter()-t
  with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:payload={k:z[k] for k in z.files}
  t=time.perf_counter();np.savez_compressed(dest/(r['geometry_id']+'.npz'),**payload);write_seconds=time.perf_counter()-t;io.append(dict(geometry_id=r['geometry_id'],raw_input_read_seconds=raw_read,cache_write_seconds=write_seconds,total_seconds=raw_read+write_seconds))
  # These caches are timing artifacts, hashes enable selective cleanup later.
 print('CACHE_IO_REPLAY',sum(q['total_seconds'] for q in io),flush=True)
 import torch,model
 model.configure();assemble=[]
 for rep in range(3):
  t=time.perf_counter();xx=[];yy=[];cc={m:[] for m in F.METHODS}
  for r in rr:
   with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
    xx.append(z['x'])
    for m in cc:cc[m].append(z['carrier_'+m])
   with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:yy.append((z['Ez'][None]/cal['scale']).astype(np.complex64))
  data={'x':torch.from_numpy(np.stack(xx)),'y':torch.from_numpy(np.stack(yy)),'c':{m:torch.from_numpy(np.stack(a)) for m,a in cc.items()}}
  assemble.append(time.perf_counter()-t);del data,xx,yy,cc;print('ASSEMBLY_REPLAY',rep,assemble[-1],flush=True)
 setup=[]
 for rep in range(3):
  torch.cuda.empty_cache();t=time.perf_counter();net=model.new();h=model.state_hash(net);b=model.batches();opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2400,eta_min=.0001);torch.cuda.synchronize();setup.append(time.perf_counter()-t);del net,opt,sch,b
 assert digest(OLD/'calibration.json')==before
 save(R/'offline_overhead_replay.json',dict(passed=True,calibration_replay_seconds=cal_times,training80_assembly_seconds=assemble,model_optimizer_setup_seconds=setup,cache80_io=io,calibration_reproduced_within_float_tolerance=True,coefficient_max_abs_difference=coef_diff,scale_abs_difference=scale_diff,frozen_calibration_unchanged=True,fit_outputs_discarded=True,training_steps=0,source='present-day timing-only replay, not historical wall measurement',scope='first-use imports/OS cold-cache effects may differ; original feature compute timer combined with replay IO'))
if __name__=='__main__':replay()

