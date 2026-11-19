"""Finite prospective S-bend local phase validation; historical inputs read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MPLBACKEND','Agg');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,hashlib,subprocess,signal,importlib.util,csv,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;BASE=R.parent;OLD=BASE/'research_sbend_formal'
MEEP='/home/cudauser/.conda/envs/picflow-meep/bin/python';GPU='/home/cudauser/.venvs/picflow-gpu/bin/python'
METHODS=['linear','arc_only','local_tangent','path_v2'];SEEDS=[42,2026,2027]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def metrics():return module('localmetrics',BASE/'research_sbend_local_phase_2026-11-15/analyze.py')
def weight(s,m):
 if s==2026:return (BASE/'research_sbend_fourmethod_5seed/runs/arc_only' if m=='arc_only' else OLD/'runs'/m)/'best.pt'
 return BASE/'research_sbend_fourmethod_5seed'/f'seed{s}'/'runs'/m/'best.pt'
def csvsave(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def remaining():return read(R/'budget.json')['deadline_unix']-time.time()
def launch(args,log,py=MEEP,limit=1800):
 assert remaining()>90,'Hard four-hour budget reached'
 t=time.perf_counter();log=R/'logs'/log;log.parent.mkdir(exist_ok=True)
 with log.open('x',encoding='utf-8') as f:
  p=subprocess.Popen([py,'-B','-u',str(R/'experiment.py'),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,remaining()-60))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('Budget/task timeout: '+str(args))
 if code:raise RuntimeError('Failed: '+str(log))
 return time.perf_counter()-t

def prepare():
 assert not (R/'freeze.json').exists()
 now=time.time();save(R/'budget.json',dict(start_unix=now,deadline_unix=now+14400,max_hours=4,automatic_extension=False))
 A=metrics();G=A.G
 paths=[OLD/'geometry_manifest.json',BASE/'research_meep_validation/sbend_index.json']
 paths+=list((BASE/'research_sbend_phase_selection_2026-11-15').rglob('geometry_manifest.json'))
 paths+=list((BASE/'output/pdf').rglob('geometry_manifest.json'))
 old=[]
 for p in paths:
  v=read(p);old+=v.get('records',[]) if isinstance(v,dict) else v
 old=[a for a in old if 'params' in a and 'R_min_um' in a['params']]
 uniq={tuple(round(a['params'][k],10) for k in ['wg_width_um','lateral_offset_um','R_min_um']):a for a in old};old=list(uniq.values())
 oldpoly=[]
 for a in old:
  poly,b=G['centerline_from_params'](a['params']);oldpoly.append((a['params']['wg_width_um'],b))
 rng=np.random.default_rng(2026111502);pool=[]
 for j in range(6000):
  wi=int(rng.integers(16,24));p=dict(wg_width_um=wi*.025,lateral_offset_um=int(rng.integers(80,221))*.025,R_min_um=int(rng.integers(120,281))*.025)
  poly,b=G['centerline_from_params'](p);margin=p['wg_width_um']/2+.5
  if abs(b[:,0]).max()+margin>=12 or abs(b[:,1]).max()+margin>=4:continue
  angle=float(np.degrees(np.arctan2(abs(G['smooth_tangent_at_x'](b[:,0],poly)[:,1]),G['smooth_tangent_at_x'](b[:,0],poly)[:,0])).max());pool.append((wi,p,b,angle))
 edges=np.quantile([v[3] for v in pool],[1/3,2/3]).tolist();records=[];seen=[]
 for st in range(3):
  for wi in range(16,24):
   candidates=[v for v in pool if v[0]==wi and int(np.searchsorted(edges,v[3],side='right'))==st];rng.shuffle(candidates);n=0
   for _,p,b,angle in candidates:
    previous=oldpoly+seen
    distances=[float(np.sqrt(np.mean(np.sum((b-bb)**2,axis=1)))) for w,bb in previous if abs(w-p['wg_width_um'])<1e-9]
    if distances and min(distances)<.035:continue
    seen.append((p['wg_width_um'],b));gid='blindB_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:16]
    ip=1 if (wi+n)%2==0 else 2
    records.append(dict(geometry_id=gid,params=p,input_port=ip,split='blind',angle_stratum=['low','middle','high'][st],max_local_angle_deg=angle,min_same_width_centerline_rms_distance_um=min(distances) if distances else None));n+=1
    if n==3:break
   assert n==3,(st,wi,n)
 assert len(records)==72
 fine=[];reverse=[]
 for st in ['low','middle','high']:
  rr=[a for a in records if a['angle_stratum']==st]
  for ii,wi in enumerate([16,17,19,20,22,23]):
   a=[r for r in rr if round(r['params']['wg_width_um']/.025)==wi];fine.append(next((r for r in a if r['input_port']==1+ii%2),a[0])['geometry_id'])
  reverse +=[rr[i]['geometry_id'] for i in [0,3,6,9,12,15]]
 reverse +=[records[20]['geometry_id'],records[44]['geometry_id']]
 cfg=dict(seeds=SEEDS,methods=METHODS,new_geometries=72,geometry_only_rng_seed=2026111502,geometry_proposal_count=6000,max_angle_stratum_edges_deg=edges,stratum_count=24,width_each_stratum=3,input_each_stratum=12,physical_fit_gate='original constructor margin; within original parameter support',novelty_exclusion='all located historical manifests deduplicated; same-width centerline RMS>=0.035um; material-grid reflection hashes checked after simulation',fine_geometry_ids=fine,fine_resolutions=[20,30,40],conditional_res60_ids=[fine[0],fine[6],fine[12]],conditional_reverse_geometry_ids=reverse,phase_gate=.05,sensitivity_gates=[.01,.1],spatial_angle_edges_deg=[0,10,20,35,90],normal_eta_edges=[-3,-1,-.5,0,.5,1,3],normal_pair_eta=[.5,1.],main='LT minus ArcOnly; each seed separately; geometry-paired bootstrap10000; no truth phase fitting',fine_comparison='coarse resolution20 physical grid and fixed normal pairs; complex linear interpolation; cubic sensitivity; no refitting of models',input_reference='raw primary; additionally divide truth by physical incoming Ez profile at fixed input monitor x=+/-11.5 and source center y; prediction divided by SAME res20 scalar; no fitted alignment',grid_gate=dict(normal_truth_30_40_wmae_max_mean_deg=.25,input_normalized_bend_truth_30_40_wmae_max_mean_deg=.5,interpolation_pair_metric_difference_max_mean_deg=.1,LT_phase_deltas_negative_all_seeds_at_res20_30_40=True),decision='reverse20 only if both primary blind phase CIs below zero in all three seeds, grid gate passes, and at least20min remain; otherwise skip; no coherent application',conditional_refine='if grid gate fails, run exactly preselected3 res60 only if >=20min remain; no additional geometries or tuning',no_training=True,no_test_calibration=True,no_global_phase_alignment=True)
 save(R/'config.json',cfg);save(R/'geometry_manifest.json',dict(records=records,proposal_angle_range_deg=[min(v[3] for v in pool),max(v[3] for v in pool)],historical_unique_geometries=len(old),historical_manifests=[str(p) for p in paths],distribution='prospective angle-stratified within original parameter support; not iid historical test'))
 ps=paths+[weight(s,m) for s in SEEDS for m in METHODS]+[OLD/n for n in ['calibration.json','features.py','model.py','fdtd_worker.py','common.py']]+[BASE/'research_sbend_path/path_geometry.py',BASE/'research_sbend_path_v2/path_v2.py',BASE/'research_sbend_local_phase_2026-11-15/analyze.py',BASE/'PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py']
 save(R/'protected_inputs.json',{str(p):sha(p) for p in ps});save(R/'freeze.json',dict(unix=time.time(),before_new_FDTD=True,before_new_predictions=True,hashes={str(p):sha(p) for p in [R/'config.json',R/'geometry_manifest.json',R/'PROTOCOL.md',R/'experiment.py',R/'protected_inputs.json']}));print('FROZEN',len(old),edges,len(records),len(fine),flush=True)

def baseworker(index):
 sys.path.insert(0,str(OLD));import fdtd_worker as fw
 fw.OUT=R;fw.mp.verbosity(0);fw.run_device('sbend',int(index))

def fineworker(index,res,reverse=False):
 sys.path.insert(0,str(OLD));import fdtd_worker as fw
 mp=fw.mp;mp.verbosity(0);r=read(R/'geometry_manifest.json')['records'][int(index)];p=r['params'];res=int(res);ip=3-r['input_port'] if reverse else r['input_port'];dest=R/('reverse' if reverse else 'fine')/r['geometry_id']/f'res{res}';dest.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
 dev=fw.EulerSBend2D(wg_width=p['wg_width_um'],lateral_offset_um=p['lateral_offset_um'],R_min_um=p['R_min_um'],wavelength_um=1.55,resolution=res,dpml=1,cell_x=26,cell_y=10)
 sim,mons,dft,_=dev._build_sim_single(input_port=ip);dft_h=sim.add_dft_fields([mp.Hx,mp.Hy],1/1.55,0,1,center=dev.full_plane.center,size=dev.full_plane.size)
 sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6));arrays={};alpha=[];modes=[]
 for j,mon in enumerate(mons,1):
  a,b,m=fw.mode_audit(sim,mon,1/1.55,j,arrays);alpha.append(a);modes.append(m)
 alpha=np.array(alpha);ind=ip-1;idir=0 if ip==1 else 1;S=alpha[np.arange(2),[1,0]]/alpha[ind,idir];meta=sim.get_array_metadata(dft_cell=dft)
 E=sim.get_dft_array(dft,mp.Ez,0).T;H=sim.get_dft_array(dft_h,mp.Hy,0).T;Hx=sim.get_dft_array(dft_h,mp.Hx,0).T;ep=sim.get_epsilon().T
 arrays.update(Ez=E,Hy=H,Hx=Hx,eps=ep,x_um=meta.x,y_um=meta.y,input_port=ip,geometry_id=np.array(r['geometry_id']),resolution=res,source_center_um=np.array([sim.sources[0].center.x,sim.sources[0].center.y,0]),alpha_native=alpha,S_native=S)
 assert E.shape==(10*res,26*res) and np.isfinite(E).all() and np.sum(abs(S)**2)<1.5
 np.savez_compressed(dest/'sample.npz',**arrays);save(dest/'result.json',dict(record=r,input_port=ip,resolution=res,source_center_um=arrays['source_center_um'].tolist(),shape=list(E.shape),wall_seconds=time.perf_counter()-t,simulation_time=sim.meep_time(),decay_tol=1e-6,monitors=modes,passed=True));sim.reset_meep();print('FINE_DONE',r['geometry_id'],res,flush=True)

def covariates(r,path):
 sys.path.insert(0,str(OLD));import features as F;import common as C
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in ['eps','src_mask','source_center_um','x_um','y_um']}
 ip=r['input_port'];beta=float(np.polyval(read(OLD/'calibration.json')['coefficients_rad_um'],float(a['eps'].max())))
 d,t,f=F.geometry(r['params'],a['x_um'],a['y_um'],a['source_center_um'],ip);x,c=F.inputs(a['eps'],a['src_mask'],a['x_um'],a['y_um'],a['source_center_um'],ip,beta,d)
 poly=f['polyline_um'];_,_,arc,_=C.v1.curve_info(poly);ss=float(C.v1.project_points(a['source_center_um'][None,:2],poly)['s'][0]);da=(1 if ip==1 else -1)*np.broadcast_to(np.interp(a['x_um'],poly[:,0],arc)-ss,a['eps'].shape);c['arc_only']=np.exp(1j*beta*da).astype(np.complex64)[None]
 return x,c

def infer():
 import torch
 sys.path.insert(0,str(OLD));import model
 model.configure();cal=read(OLD/'calibration.json');records=read(R/'geometry_manifest.json')['records'];checks=[]
 hist=read(OLD/'geometry_manifest.json')['records']
 for ip in [1,2]:
  r=next(a for a in hist if a['input_port']==ip);x,c=covariates(r,OLD/'data'/r['geometry_id']/'sample.npz')
  with np.load(OLD/'cache'/(r['geometry_id']+'.npz')) as z:
   assert np.array_equal(x,z['x'])
   for m in ['linear','local_tangent','path_v2']:assert np.array_equal(c[m],z['carrier_'+m])
  checks.append(dict(input_port=ip,x_and_carriers_exact=True,geometry_id=r['geometry_id']))
 save(R/'feature_replay.json',checks)
 (R/'cache').mkdir();(R/'predictions').mkdir();features=[]
 for i,r in enumerate(records):
  t=time.perf_counter();x,c=covariates(r,R/'data'/r['geometry_id']/'sample.npz');np.savez_compressed(R/'cache'/(r['geometry_id']+'.npz'),x=x,**{'carrier_'+m:c[m] for m in METHODS});features.append(dict(geometry_id=r['geometry_id'],seconds=time.perf_counter()-t))
  if (i+1)%12==0:print('FEATURES',i+1,72,flush=True)
 save(R/'feature_costs.json',features);cost=[]
 for s in SEEDS:
  for m in METHODS:
   t=time.perf_counter();net=model.new();ck=torch.load(weight(s,m),map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);net.eval();pred=[]
   for start in range(0,72,2):
    xx=[];cc=[]
    for r in records[start:start+2]:
     with np.load(R/'cache'/(r['geometry_id']+'.npz')) as z:xx.append(z['x']);cc.append(z['carrier_'+m])
    with torch.no_grad():v=model.forward(net,torch.from_numpy(np.stack(xx)).cuda(),torch.from_numpy(np.stack(cc)).cuda())[:,0]*cal['scale']
    pred.append(v.cpu().numpy())
   np.savez_compressed(R/'predictions'/f'seed{s}_{m}.npz',prediction=np.concatenate(pred),geometry_id=np.array([r['geometry_id'] for r in records]),input_port=np.array([r['input_port'] for r in records]));cost.append(dict(seed=s,method=m,seconds=time.perf_counter()-t));del net,ck,pred,v;torch.cuda.empty_cache();print('INFER',s,m,flush=True)
 save(R/'inference_costs.json',cost);save(R/'prediction_freeze.json',dict(unix=time.time(),truth_fields_accessed=False,hashes={str(p):sha(p) for p in (R/'predictions').glob('*.npz')}))

def materialaudit():
 hist=read(OLD/'geometry_manifest.json')['records'];D=read(BASE/'research_meep_validation/sbend_index.json');paths=[OLD/'data'/r['geometry_id']/'sample.npz' for r in hist]+[BASE/'research_meep_validation'/r['sample'] for r in D]
 for stage in ['development','formal']:
  root=BASE/'research_sbend_phase_selection_2026-11-15'/stage
  paths +=[root/'data'/r['geometry_id']/'sample.npz' for r in read(root/'geometry_manifest.json')['records']]
 hashes=set()
 def ah(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float32).tobytes()).hexdigest()
 for p in paths:
  with np.load(p) as z:e=z['eps']
  hashes.update(ah(a) for a in [e,e[::-1],e[:,::-1],e[::-1,::-1]])
 for r in read(R/'geometry_manifest.json')['records']:
  with np.load(R/'data'/r['geometry_id']/'sample.npz') as z:e=z['eps']
  assert ah(e) not in hashes,'Material duplicate!'
  hashes.update(ah(a) for a in [e,e[::-1],e[:,::-1],e[::-1,::-1]])
 save(R/'novelty_verification.json',dict(passed=True,historical_material_grids=len(paths),new_material_grids=72,reflections_checked=True,truth_field_values_accessed=False))

def run():
 prepare();records=read(R/'geometry_manifest.json')['records'];cost=[]
 for i,r in enumerate(records):
  s=launch(['baseworker',i],f'base_{i:02d}.log');cost.append(dict(geometry_id=r['geometry_id'],seconds=s));save(R/'progress.json',dict(stage='base_FDTD',completed=i+1,total=72,remaining_budget_seconds=remaining()))
  if (i+1)%6==0:print('BASE',i+1,72,flush=True)
 save(R/'base_costs.json',cost);materialaudit();launch(['infer'],'inference.log',py=GPU,limit=3600)
 cfg=read(R/'config.json');cost=[]
 for gid in cfg['fine_geometry_ids']:
  i=next(j for j,r in enumerate(records) if r['geometry_id']==gid)
  for res in [30,40]:
   s=launch(['fineworker',i,res],f'fine_{i:02d}_{res}.log');cost.append(dict(geometry_id=gid,resolution=res,seconds=s));save(R/'progress.json',dict(stage='fine_FDTD',completed=len(cost),total=36,remaining_budget_seconds=remaining()));print('GRID',len(cost),36,flush=True)
 save(R/'fine_costs.json',cost);save(R/'simulations_complete.json',dict(unix=time.time(),base=72,fine=36,remaining_budget_seconds=remaining(),analysis_pending=True));print('SIMULATIONS_COMPLETE',flush=True)

if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='run':run()
  elif action=='baseworker':baseworker(sys.argv[2])
  elif action=='fineworker':fineworker(sys.argv[2],sys.argv[3],len(sys.argv)>4)
  elif action=='infer':infer()
 except Exception:
  traceback.print_exc()
  if len(sys.argv)>1 and sys.argv[1]=='run':save(R/'stopped.json',dict(unix=time.time(),traceback=traceback.format_exc(),automatic_extension=False))
  raise
