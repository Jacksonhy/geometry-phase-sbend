"""Geometry-only S-bend inference. CLI runs in existing picflow-gpu Python.
No FDTD propagation, sample.npz, stored truth, historical caches or predictions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,subprocess,argparse,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;BASE=R.parent;OLD=BASE/'research_sbend_formal'
MEEP='/home/cudauser/.conda/envs/picflow-meep/bin/python'
METHODS=['linear','arc_only','local_tangent','path_v2'];SEEDS=[42,2026,2027]
PROCESS_START=time.perf_counter()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def records(path):
 a=read(path);a=a['records'] if isinstance(a,dict) and 'records' in a else [a] if isinstance(a,dict) else a
 for i,r in enumerate(a):
  r.setdefault('geometry_id',f'query_{i:04d}');p=r['params']
  if r['input_port'] not in (1,2):raise ValueError('input_port must be 1 or 2')
  for k,lo,hi in [('wg_width_um',.4,.575),('lateral_offset_um',2.,5.5),('R_min_um',3.,7.)]:
   if not np.isfinite(p[k]) or not lo-1e-10<=p[k]<=hi+1e-10:raise ValueError(f'{k} outside evaluated support')
  if any(abs(p[k]/.025-round(p[k]/.025))>1e-8 for k in p):raise ValueError('Only established 0.025um parameter lattice is supported')
  if Path(r['geometry_id']).name!=r['geometry_id']:raise ValueError('geometry_id must be a basename')
 if len({r['geometry_id'] for r in a})!=len(a):raise ValueError('Duplicate geometry_id')
 return a

def field_free_guard():
 """Fail closed for old sample/caches/predictions, and log read dependencies."""
 opened=[]
 def audit(event,args):
  if event!='open':return
  try:p=os.fspath(args[0])
  except TypeError:return
  if not isinstance(p,str):return
  q=p.replace('\\','/')
  if '/mnt/d/game/' not in q:return
  if q.endswith('/sample.npz') or any(s in q for s in ['/data/','/cache/','/predictions/']):raise RuntimeError('Field-free path guard blocked '+q)
  opened.append(q)
 sys.addaudithook(audit);return opened

def prepare(manifest,out):
 t0=time.perf_counter();rr=records(manifest);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 opened=field_free_guard();sys.path.insert(0,str(OLD));import fdtd_worker as fw
 import features as F;import common as C
 mp=fw.mp;mp.verbosity(0)
 def forbidden(*a,**k):raise RuntimeError('Time-domain run/step is forbidden in input preparation')
 mp.Simulation.run=forbidden;mp.Simulation.step=forbidden
 cal=read(OLD/'calibration.json');cost=[]
 for r in rr:
  gid=r['geometry_id'];dest=out/(gid+'.npz')
  if dest.exists():raise FileExistsError(dest)
  t=time.perf_counter();p=r['params'];ip=r['input_port']
  dev=fw.EulerSBend2D(wg_width=p['wg_width_um'],lateral_offset_um=p['lateral_offset_um'],R_min_um=p['R_min_um'],wavelength_um=1.55,resolution=20,dpml=1,cell_x=26,cell_y=10)
  sim,mons,dft,_=dev._build_sim_single(input_port=ip);sim.init_sim()
  assert sim.meep_time()==0
  ep=sim.get_epsilon().T[20:-20,20:-20].astype(np.float32)
  meta=sim.get_array_metadata(dft_cell=dft);x=np.array(meta.x[20:-20]);y=np.array(meta.y[20:-20])
  src=sim.sources[0];source=np.array([src.center.x,src.center.y,src.center.z]);half=float(src.size.y)/2
  X,Y=np.meshgrid(x,y);sm=((X-source[0])**2+(Y-np.clip(Y,source[1]-half,source[1]+half))**2<=.15**2+1e-14).astype(np.float32)
  assert ep.shape==sm.shape==(160,480)
  material_seconds=time.perf_counter()-t;sim.reset_meep();tf=time.perf_counter()
  beta=float(np.polyval(cal['coefficients_rad_um'],float(ep.max())))
  d,_,f=F.geometry(p,x,y,source,ip);inp,carriers=F.inputs(ep,sm,x,y,source,ip,beta,d)
  poly=f['polyline_um'];_,_,arc,_=C.v1.curve_info(poly);ss=float(C.v1.project_points(source[None,:2],poly)['s'][0])
  da=(1 if ip==1 else -1)*np.broadcast_to(np.interp(x,poly[:,0],arc)-ss,ep.shape)
  carriers['arc_only']=np.exp(1j*beta*da).astype(np.complex64)[None]
  feature_seconds=time.perf_counter()-tf;ts=time.perf_counter()
  np.savez_compressed(dest,x=inp,**{'carrier_'+m:carriers[m] for m in METHODS},eps=ep,src_mask=sm,x_um=x,y_um=y,source_center_um=source,input_port=np.int32(ip),geometry_id=np.array(gid),beta_rad_um=beta)
  cost.append(dict(geometry_id=gid,material_init_seconds=material_seconds,feature_seconds=feature_seconds,export_seconds=time.perf_counter()-ts,total_seconds=time.perf_counter()-t,simulation_time=0.))
  if len(cost)%12==0:print('PREPARED',len(cost),len(rr),flush=True)
 save(out/'preparation.json',dict(records=rr,costs=cost,total_seconds=time.perf_counter()-t0,all_four_carriers_built=True,all_12_common_real_features_preserved=True,FDTD_runs=0,simulation_time=0,truth_files_read=0,read_dependencies=sorted(set(opened)),calibration_sha256=digest(OLD/'calibration.json')))
 print('PREPARATION_COMPLETE',len(rr),flush=True)

def weight(seed,method):
 if seed==2026:return (BASE/'research_sbend_fourmethod_5seed/runs/arc_only' if method=='arc_only' else OLD/'runs'/method)/'best.pt'
 return BASE/'research_sbend_fourmethod_5seed'/f'seed{seed}'/'runs'/method/'best.pt'

def load_net(seed,method):
 import torch
 sys.path.insert(0,str(OLD));import model
 model.configure();t=time.perf_counter();net=model.new();ck=torch.load(weight(seed,method),map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);net.eval();torch.cuda.synchronize()
 return net,dict(model_load_seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in net.parameters()),checkpoint_sha256=digest(weight(seed,method)))

def forward_records(net,rr,inputs,method,batch=2):
 import torch,model
 cal=read(OLD/'calibration.json');result=[];parts=[]
 torch.cuda.reset_peak_memory_stats()
 for start in range(0,len(rr),batch):
  tt=time.perf_counter();xx=[];cc=[]
  for r in rr[start:start+batch]:
   with np.load(Path(inputs)/(r['geometry_id']+'.npz'),allow_pickle=False) as z:xx.append(z['x']);cc.append(z['carrier_'+method])
  load_seconds=time.perf_counter()-tt;t=time.perf_counter();xt=torch.from_numpy(np.stack(xx)).cuda();ct=torch.from_numpy(np.stack(cc)).cuda();torch.cuda.synchronize();h2d=time.perf_counter()-t;t=time.perf_counter()
  with torch.inference_mode():v=model.forward(net,xt,ct)[:,0]*cal['scale']
  torch.cuda.synchronize();gpu=time.perf_counter()-t;t=time.perf_counter();result.append(v.cpu().numpy());d2h=time.perf_counter()-t
  parts.append(dict(samples=len(xx),load_seconds=load_seconds,h2d_seconds=h2d,forward_and_carrier_seconds=gpu,d2h_seconds=d2h))
 return np.concatenate(result),parts,float(torch.cuda.max_memory_allocated()/1024**2)

def infer(manifest,inputs,method,seed,output,batch=2):
 opened=field_free_guard();t=time.perf_counter();rr=records(manifest);contract_t=time.perf_counter();prepared=read(Path(inputs)/'preparation.json');saved={r['geometry_id']:r for r in prepared['records']}
 for r in rr:
  if r['geometry_id'] not in saved or saved[r['geometry_id']]['params']!=r['params'] or saved[r['geometry_id']]['input_port']!=r['input_port']:raise ValueError('Prepared inputs do not match requested geometry/port')
 if prepared['calibration_sha256']!=digest(OLD/'calibration.json'):raise ValueError('Prepared inputs use a different training calibration')
 contract_seconds=time.perf_counter()-contract_t;net,meta=load_net(seed,method)
 pred,parts,peak=forward_records(net,rr,inputs,method,batch);ts=time.perf_counter()
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 with np.load(Path(inputs)/(rr[0]['geometry_id']+'.npz')) as z:x=z['x_um'];y=z['y_um']
 np.savez_compressed(output,prediction=pred,x_um=x,y_um=y,geometry_id=np.array([r['geometry_id'] for r in rr]),input_port=np.array([r['input_port'] for r in rr]),method=np.array(method),seed=np.int32(seed),wavelength_um=1.55,dx_um=.05,source_amplitude=1.,source_phase_rad=0.)
 save(output.with_suffix('.json'),dict(method=method,seed=seed,samples=len(rr),batch_size=batch,**meta,input_contract_check_seconds=contract_seconds,parts=parts,export_seconds=time.perf_counter()-ts,total_seconds=time.perf_counter()-t,peak_allocated_MiB=peak,truth_files_read=0,read_dependencies=sorted(set(opened)),reference='fixed original Gaussian EigenModeSource convention; raw frequency-domain Ez; no truth alignment',scope='2D effective-index, wavelength1.55um, resolution20 surrogate; no continuous-solution certification'))
 print('INFERENCE_COMPLETE',method,seed,len(rr),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry-json',required=True);p.add_argument('--method',choices=METHODS,default='local_tangent');p.add_argument('--seed',type=int,choices=SEEDS,default=2026);p.add_argument('--output',required=True);p.add_argument('--prepared-inputs');p.add_argument('--batch-size',type=int,choices=[1,2],default=2);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 if a.prepare_only:return prepare(a.geometry_json,a.output)
 opened=field_free_guard();t=time.perf_counter();out=Path(a.output);inputs=Path(a.prepared_inputs) if a.prepared_inputs else out.parent/(out.stem+'_inputs')
 if not a.prepared_inputs:
  subprocess.run([MEEP,'-B',str(R/'predict.py'),'--geometry-json',a.geometry_json,'--prepare-only','--output',str(inputs)],check=True)
 infer(a.geometry_json,inputs,a.method,a.seed,a.output,a.batch_size)
 save(out.with_suffix('.workflow.json'),dict(geometry_json=str(a.geometry_json),method=a.method,seed=a.seed,precomputed_inputs=a.prepared_inputs is not None,total_seconds_after_imports=time.perf_counter()-t,process_seconds_after_standard_imports=time.perf_counter()-PROCESS_START,truth_files_read=0,FDTD_runs=0,training_runs=0))
if __name__=='__main__':main()

