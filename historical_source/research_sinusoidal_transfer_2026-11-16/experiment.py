"""Frozen-model transfer to explicitly defined raised-sine S-bends. No training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MPLBACKEND','Agg');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,hashlib,csv,subprocess,signal,traceback,importlib.util,contextlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;BASE=R.parent;OLD=BASE/'research_sbend_formal'
MEEP='/home/cudauser/.conda/envs/picflow-meep/bin/python';GPU='/home/cudauser/.venvs/picflow-gpu/bin/python'
METHODS=['linear','arc_only','local_tangent','path_v2'];SEEDS=[42,2026,2027]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def csvsave(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for a in rows for k in a))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def deployment():return module('frozen_deployment',BASE/'research_sbend_surrogate_delivery_2026-11-15/predict.py')
def weight(s,m):return deployment().weight(s,m)
def remaining():return read(R/'budget.json')['deadline_unix']-time.time()
def sine_points(offset,length,n_pts=256):
 t=np.linspace(0.,1.,n_pts);return length*(t-.5),offset*(t-np.sin(2*np.pi*t)/(2*np.pi)-.5)
def centerline(params,x_bounds=(-12.,12.)):
 x,y=sine_points(params['lateral_offset_um'],params['length_um']);bend=np.column_stack([x,y]);assert x_bounds[0]<x[0]<x[-1]<x_bounds[1]
 return np.vstack([[x_bounds[0],y[0]],bend,[x_bounds[1],y[-1]]]),bend
def metrics():
 a=module('frozen_local_metrics',BASE/'research_sbend_local_phase_2026-11-15/analyze.py');a.G['centerline_from_params']=centerline;return a
@contextlib.contextmanager
def feature_geometry():
 sys.path.insert(0,str(OLD));import features as F
 old=F.v1.centerline_from_params;F.v1.centerline_from_params=centerline
 try:yield F
 finally:F.v1.centerline_from_params=old
def device(r,res):
 sys.path.insert(0,str(OLD));import fdtd_worker as fw
 mod=sys.modules[fw.EulerSBend2D.__module__];original=mod.euler_sbend_points;p=r['params']
 mod.euler_sbend_points=lambda lateral_offset_um,R_min_um,n_pts=256:sine_points(lateral_offset_um,p['length_um'],n_pts)
 try:d=fw.EulerSBend2D(wg_width=p['wg_width_um'],lateral_offset_um=p['lateral_offset_um'],R_min_um=5.,wavelength_um=1.55,resolution=res,dpml=1,cell_x=26,cell_y=10)
 finally:mod.euler_sbend_points=original
 _,bend=centerline(p);assert np.array_equal(d._sbend_pts,bend)
 return d,fw
def descriptors(b):
 d=np.diff(b,axis=0);arc=float(np.linalg.norm(d,axis=1).sum());t=np.gradient(b[:,1],b[:,0]);curv=np.gradient(t,b[:,0])/(1+t*t)**1.5
 return dict(axial_length_um=float(np.ptp(b[:,0])),offset_um=float(np.ptp(b[:,1])),bend_arc_length_um=arc,max_angle_deg=float(np.degrees(np.arctan(abs(t))).max()),discrete_min_radius_um=float(1/abs(curv).max()))
def freeze():
 assert not (R/'freeze.json').exists();t=time.time();save(R/'budget.json',dict(start_unix=t,deadline_unix=t+14400,max_hours=4,automatic_extension=False,includes_preparation_and_analysis=True))
 a=metrics();pool=read(BASE/'research_sbend_blind_local_phase_2026-11-15/geometry_manifest.json')['records'];candidates=[]
 for old in pool:
  ep,eb=a.G['official_euler_function']()(old['params']['lateral_offset_um'],old['params']['R_min_um'])
  eb=np.column_stack([ep-.5*(ep.min()+ep.max()),eb-.5*(eb.min()+eb.max())]);p=dict(wg_width_um=old['params']['wg_width_um'],lateral_offset_um=float(np.ptp(eb[:,1])),length_um=float(np.ptp(eb[:,0])))
  _,b=centerline(p);r=dict(geometry_id='sine_'+sha_text(json.dumps(p,sort_keys=True))[:16],params=p,input_port=old['input_port'],split='transfer',matched_euler_geometry_id=old['geometry_id'],matched_euler_params=old['params'],max_local_angle_deg=float(np.degrees(np.arctan(2*p['lateral_offset_um']/p['length_um']))),sine=descriptors(b),euler=descriptors(eb))
  assert np.max(abs(b[:,0]))+p['wg_width_um']/2+.5<12 and np.max(abs(b[:,1]))+p['wg_width_um']/2+.5<4
  r['same_fraction_centerline_rms_difference_um']=float(np.sqrt(np.mean(np.sum((b-eb)**2,axis=1))));candidates.append(r)
 edges=np.quantile([r['max_local_angle_deg'] for r in candidates],[1/3,2/3]);records=[]
 for st,label in enumerate(['low','middle','high']):
  for w in [.4,.425,.475,.5,.55,.575]:
   for ip in [1,2]:
    rr=[r for r in candidates if abs(r['params']['wg_width_um']-w)<1e-8 and r['input_port']==ip and np.searchsorted(edges,r['max_local_angle_deg'],side='right')==st]
    r=sorted(rr,key=lambda r:r['geometry_id'])[0];r['angle_stratum']=label;records.append(r)
 assert len(records)==len({r['geometry_id'] for r in records})==36
 fine=[]
 for st,w in [('low',.4),('middle',.5),('high',.575)]:
  fine.extend(r['geometry_id'] for r in records if r['angle_stratum']==st and abs(r['params']['wg_width_um']-w)<1e-8)
 cfg=dict(methods=METHODS,seeds=SEEDS,count=36,selection='geometry-only from all72 archived Euler geometries; sine angle tertiles, six widths, two directions; lexicographic newID tie-break; no field/prediction selection',angle_edges_deg=edges.tolist(),widths_um=[.4,.425,.475,.5,.55,.575],fine_geometry_ids=fine,fine_resolutions=[20,30,40],centerline='y=l*(t-sin(2*pi*t)/(2*pi)-0.5), x=L*(t-0.5), t in [0,1]',length_offset='actual endpoint span of matched archived Euler polyline; NOT nominal Euler offset parameter',n_points=256,geometry='original rotated Blocks,1.05 segment overlap, original leads/source/PML/material',physical_model='2D effective index,1.55um,cell26x10,PML1,decay1e-6,original Gaussian eigenmode source',prediction_before_truth=True,training_runs=0,test_calibration=False,phase_alignment=False,primary='LT minus ArcOnly, per seed geometry-paired10000 bootstrap; bend phase and normal pair phase; no pooled108 independence',secondary='whole and bend complex RelL2, transverse coherence RelL2; local angle/normal position; matched Euler descriptive comparison',phase_gate=.05,normal_pair_eta=.5,sensitivity_gates=[.01,.1],spatial_angle_edges_deg=[0,10,20,35,90],normal_eta_edges=[-3,-1,-.5,0,.5,1,3],mesh='common physical coarse coordinates, complex linear interpolation; cubic pair sensitivity; raw plus incident-normalized diagnostic; no fitted phase alignment',decision='report all signs/CIs and adverse cases; no training or application automatically follows; no continuum claim',max_hours=4)
 save(R/'geometry_manifest.json',dict(records=records,distribution='shape-family transfer with matched Euler interfaces; angle stratified, not iid or cross topology'));save(R/'config.json',cfg)
 paths=[OLD/n for n in ['calibration.json','features.py','common.py','fdtd_worker.py','model.py']]+[weight(s,m) for s in SEEDS for m in METHODS]+[BASE/'research_sbend_path/path_geometry.py',BASE/'research_sbend_path_v2/path_v2.py',BASE/'research_sbend_local_phase_2026-11-15/analyze.py',BASE/'research_sbend_blind_local_phase_2026-11-15/evaluate_blind.py',BASE/'research_sbend_surrogate_delivery_2026-11-15/predict.py',BASE/'research_sbend_blind_local_phase_2026-11-15/geometry_manifest.json']
 paths+=list((BASE/'PIC-Flow-新研究资料').rglob('FDTD/sbend/sbend.py'))
 save(R/'protected_inputs.json',{str(p):sha(p) for p in paths});save(R/'freeze.json',dict(unix=time.time(),before_FDTD=True,before_predictions=True,hashes={str(p):sha(p) for p in [R/'experiment.py',R/'analyze.py',R/'PROTOCOL.md',R/'config.json',R/'geometry_manifest.json',R/'protected_inputs.json']}));print('FROZEN',len(records),'fine',len(fine),'angle edges',edges,flush=True)
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
def prepare():
 D=deployment();opened=D.field_free_guard();records=read(R/'geometry_manifest.json')['records'];cost=[]
 for i,r in enumerate(records):
  t=time.perf_counter();dev,fw=device(r,20);mp=fw.mp;mp.verbosity(0)
  def forbidden(*a,**k):raise RuntimeError('No time evolution in input preparation')
  mp.Simulation.run=forbidden;mp.Simulation.step=forbidden
  sim,mons,dft,_=dev._build_sim_single(input_port=r['input_port']);sim.init_sim();assert sim.meep_time()==0
  e=sim.get_epsilon().T[20:-20,20:-20].astype(np.float32);meta=sim.get_array_metadata(dft_cell=dft);x=np.array(meta.x[20:-20]);y=np.array(meta.y[20:-20]);src=sim.sources[0];source=np.array([src.center.x,src.center.y,0.]);half=float(src.size.y)/2;X,Y=np.meshgrid(x,y);sm=((X-source[0])**2+(Y-np.clip(Y,source[1]-half,source[1]+half))**2<=.15**2+1e-14).astype(np.float32);sim.reset_meep();beta=float(np.polyval(read(OLD/'calibration.json')['coefficients_rad_um'],float(e.max())))
  with feature_geometry() as F:
   d,_,f=F.geometry(r['params'],x,y,source,r['input_port']);inp,c=F.inputs(e,sm,x,y,source,r['input_port'],beta,d);_,_,arc,_=F.v1.curve_info(f['polyline_um']);ss=float(F.v1.project_points(source[None,:2],f['polyline_um'])['s'][0]);da=(1 if r['input_port']==1 else -1)*np.broadcast_to(np.interp(x,f['polyline_um'][:,0],arc)-ss,e.shape);c['arc_only']=np.exp(1j*beta*da).astype(np.complex64)[None]
  assert inp.shape==(7,160,480) and all(v.shape==(1,160,480) for v in c.values())
  np.savez_compressed(R/'inputs'/(r['geometry_id']+'.npz'),x=inp,eps=e,src_mask=sm,x_um=x,y_um=y,source_center_um=source,**{'carrier_'+m:c[m] for m in METHODS})
  cost.append(dict(geometry_id=r['geometry_id'],seconds=time.perf_counter()-t));print('PREPARED',i+1,36,flush=True)
 save(R/'preparation.json',dict(costs=cost,truth_files_read=0,FDTD_runs=0,dependencies=sorted(set(opened))))
def infer():
 import torch
 D=deployment();opened=D.field_free_guard();rr=read(R/'geometry_manifest.json')['records'];cost=[]
 for s in SEEDS:
  for m in METHODS:
   t=time.perf_counter();net,meta=D.load_net(s,m);p,parts,peak=D.forward_records(net,rr,R/'inputs',m,2)
   assert p.shape==(36,160,480) and np.isfinite(p).all();np.savez_compressed(R/'predictions'/f'seed{s}_{m}.npz',prediction=p,geometry_id=np.array([r['geometry_id'] for r in rr]),input_port=np.array([r['input_port'] for r in rr]));cost.append(dict(seed=s,method=m,seconds=time.perf_counter()-t,**meta,peak_MiB=peak));del net,p;torch.cuda.empty_cache();print('INFER',s,m,flush=True)
 save(R/'inference_costs.json',cost);save(R/'prediction_freeze.json',dict(unix=time.time(),truth_fields_accessed=False,before_new_truth=True,dependencies=sorted(set(opened)),hashes={str(p):sha(p) for p in (R/'predictions').glob('*.npz')}))
def simworker(index,res):
 r=read(R/'geometry_manifest.json')['records'][int(index)];res=int(res);t=time.perf_counter();dev,fw=device(r,res);mp=fw.mp;mp.verbosity(0);ip=r['input_port'];dest=R/('data' if res==20 else 'fine')/r['geometry_id'];dest=dest if res==20 else dest/f'res{res}';dest.mkdir(parents=True,exist_ok=False)
 sim,mons,dft,_=dev._build_sim_single(input_port=ip);dh=sim.add_dft_fields([mp.Hx,mp.Hy],1/1.55,0,1,center=dev.full_plane.center,size=dev.full_plane.size);sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6));arrays={};alpha=[];modes=[]
 for j,mon in enumerate(mons,1):
  aa,_,mm=fw.mode_audit(sim,mon,1/1.55,j,arrays);alpha.append(aa);modes.append(mm)
 alpha=np.array(alpha);S=alpha[np.arange(2),[1,0]]/alpha[ip-1,0 if ip==1 else 1];meta=sim.get_array_metadata(dft_cell=dft);E=sim.get_dft_array(dft,mp.Ez,0).T;Hx=sim.get_dft_array(dh,mp.Hx,0).T;Hy=sim.get_dft_array(dh,mp.Hy,0).T;eps=sim.get_epsilon().T;src=sim.sources[0];source=np.array([src.center.x,src.center.y,0.]);sl=np.s_[res:-res,res:-res]
 assert E.shape==(10*res,26*res) and np.isfinite(E).all() and np.sum(abs(S)**2)<1.5
 arrays.update(Ez=E[sl],Hx=Hx[sl],Hy=Hy[sl],eps=eps[sl].astype(np.float32),x_um=meta.x[res:-res],y_um=meta.y[res:-res],source_center_um=source,input_port=np.int32(ip),geometry_id=np.array(r['geometry_id']),resolution=np.int32(res),alpha_native=alpha,S_native=S)
 if res==20:
  with np.load(R/'inputs'/(r['geometry_id']+'.npz')) as z:
   assert np.array_equal(z['eps'],arrays['eps']) and np.array_equal(z['x_um'],arrays['x_um']) and np.array_equal(z['y_um'],arrays['y_um']) and np.array_equal(z['source_center_um'],source)
   arrays['src_mask']=z['src_mask']
 np.savez_compressed(dest/'sample.npz',**arrays);save(dest/'result.json',dict(record=r,resolution=res,wall_seconds=time.perf_counter()-t,simulation_time=sim.meep_time(),decay_tol=1e-6,source_center_um=source.tolist(),monitors=modes,guided_power_sum=float(np.sum(abs(S)**2)),passed=True));sim.reset_meep();print('FDTD',index,res,flush=True)
def launch(script,args,log,py=MEEP,limit=1800):
 assert remaining()>90,'Hard4h budget reached';t=time.perf_counter()
 with (R/'logs'/log).open('x',encoding='utf-8') as f:
  p=subprocess.Popen([py,'-B','-u',str(R/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,max(1,remaining()-45)))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('Hard budget or task timeout '+log)
 if code:raise RuntimeError('Task failed '+log)
 return time.perf_counter()-t
def run():
 freeze();launch('experiment.py',['prepare'],'prepare.log',limit=2400);launch('experiment.py',['infer'],'inference.log',GPU,2400);rr=read(R/'geometry_manifest.json')['records'];cost=[]
 for i,r in enumerate(rr):
  sec=launch('experiment.py',['simworker',i,20],f'base_{i:02d}.log');cost.append(dict(geometry_id=r['geometry_id'],resolution=20,seconds=sec));save(R/'progress.json',dict(stage='base_FDTD',completed=i+1,total=36,remaining_seconds=remaining()));print('BASE',i+1,36,flush=True)
 launch('analyze.py',['blind'],'analysis_base.log',limit=1800)
 for gid in read(R/'config.json')['fine_geometry_ids']:
  i=next(i for i,r in enumerate(rr) if r['geometry_id']==gid)
  for res in [30,40]:
   sec=launch('experiment.py',['simworker',i,res],f'fine_{i:02d}_{res}.log');cost.append(dict(geometry_id=gid,resolution=res,seconds=sec));save(R/'progress.json',dict(stage='fine_FDTD',completed=len(cost)-36,total=12,remaining_seconds=remaining()));print('GRID',len(cost)-36,12,flush=True)
 save(R/'simulation_costs.json',cost);launch('analyze.py',['grid'],'analysis_grid.log',limit=1800);launch('analyze.py',['report'],'report.log',limit=1800);save(R/'complete.json',dict(unix=time.time(),elapsed_seconds=14400-remaining(),remaining_seconds=remaining(),base=36,fine=12,training_runs=0));print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  a=sys.argv[1]
  if a=='run':run()
  elif a=='prepare':prepare()
  elif a=='infer':infer()
  elif a=='simworker':simworker(*sys.argv[2:])
 except Exception:
  traceback.print_exc()
  if sys.argv[1]=='run':save(R/'stopped.json',dict(unix=time.time(),reason=traceback.format_exc(),automatic_extension=False))
  raise
