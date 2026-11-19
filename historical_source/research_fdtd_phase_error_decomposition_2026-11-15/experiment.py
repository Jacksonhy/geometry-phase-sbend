"""Finite FDTD phase-error decomposition. No networks, no test phase correction."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MPLBACKEND','Agg');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,subprocess,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;BASE=R.parent;B=BASE/'research_sbend_blind_local_phase_2026-11-15';OLD=BASE/'research_sbend_formal';PY='/home/cudauser/.conda/envs/picflow-meep/bin/python'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
def remaining():return read(R/'budget.json')['deadline_unix']-time.time()
def fwimport():
 sys.path.insert(0,str(OLD));import fdtd_worker as fw;fw.mp.verbosity(0);return fw
def prior(gid,res):return B/'data'/gid/'sample.npz' if res==20 else B/'fine'/gid/f'res{res}/sample.npz'
def geom():
 p=BASE/'research_sbend_local_phase_2026-11-15/analyze.py';s=importlib.util.spec_from_file_location('old_metrics',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare():
 assert not (R/'freeze.json').exists();t=time.time();save(R/'budget.json',dict(start_unix=t,deadline_unix=t+14400,hard_hours=4,no_extension=True))
 rr=read(B/'geometry_manifest.json')['records'];fine=read(B/'config.json')['fine_geometry_ids'];records=[]
 for st in ['low','middle','high']:
  for w in [.4,.5,.575]:records.append(next(r for r in rr if r['angle_stratum']==st and abs(r['params']['wg_width_um']-w)<1e-9 and r['geometry_id'] in fine))
 controls=[records[0]['geometry_id'],records[4]['geometry_id'],records[8]['geometry_id']]
 cfg=dict(wavelength_um=1.55,widths_um=[.4,.5,.575],resolutions=[20,30,40,60],diagnostic_records=records,control_ids=controls,new60_ids=[r['geometry_id'] for r in records if not prior(r['geometry_id'],60).exists()],conditional80_ids=controls,conditional80_rule='after60, same fixed control3 only if any40to60 incident-normalized bend phase>0.5deg or normal phase>0.25deg and >=40min remain',base_settings=dict(cell_um=[26,10],PML_um=1,decay_tol=1e-6,Courant=.5,eps_averaging=True),straight='3 centered widths x4 resolutions plus nine geometry-matched input-y/input-direction references x4; same source, span and fixed x=+/-11.5 reference planes',beta='MPB default and eig_resolution=4*res; FDTD projected forward-mode longitudinal unwrapped phase slopes with multiple fixed windows plus centerline; no network calibration',propagation_diagnostic='Delta beta from matched straight references times signed centerline arclength from fixed input monitor; circular residual reported; no fitted correction',relative_reference='physical modal Ez transmission and native t divided by same-width same-input-y straight t at each grid; diagnostic only',material='initialize original Block geometry at20/30/40/60; sample actual Ez inverse-epsilon zz, trace-average epsilon, exact continuous union, fixed normal profiles and fractional widths; eps_averaging off and same-material Block order reversal at20/40 on fixed controls',sensitivity='control3 atres40: baseline tol1e-6, continue+100, tighten tol1e-9 minimum100 maximum500; PML2 and cell28x12 keep source/ports/geometry/non-PML physical limits; eps-off and reversed-block-order; centered straight40 Courant0.25',no_network=True,no_training=True,no_phase_fit=True,no_smooth_geometry_replacement=True,finite_budget='4h hard deadline; do not extend; preserve partial data and report missing tasks')
 save(R/'config.json',cfg);paths=[B/'geometry_manifest.json',B/'config.json',B/'evaluate_blind.py',BASE/'research_sbend_local_phase_2026-11-15/analyze.py',OLD/'fdtd_worker.py',BASE/'PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py']
 for r in records:
  for res in [20,30,40,60]:
   p=prior(r['geometry_id'],res)
   if p.exists():paths.append(p)
 save(R/'protected_inputs.json',{str(p):sha(p) for p in paths});save(R/'freeze.json',dict(unix=time.time(),before_new_FDTD=True,hashes={str(p):sha(p) for p in [R/'config.json',R/'PROTOCOL.md',R/'experiment.py',R/'protected_inputs.json']}));print('FROZEN',[(r['angle_stratum'],r['params']['wg_width_um'],r['input_port']) for r in records],flush=True)

def spec(job):
 cfg=read(R/'config.json')
 if job['kind']=='centered':
  w=job['width'];r=dict(geometry_id=f'centered_w{w:.3f}',params=dict(wg_width_um=w,lateral_offset_um=3.,R_min_um=5.),input_port=1);cy=0
 else:r=next(r for r in cfg['diagnostic_records'] if r['geometry_id']==job['geometry_id']);cy=None
 return r,cy

def build(job):
 fw=fwimport();mp=fw.mp;r,cy=spec(job);p=r['params'];res=job['resolution'];dev=fw.EulerSBend2D(wg_width=p['wg_width_um'],lateral_offset_um=p['lateral_offset_um'],R_min_um=p['R_min_um'],wavelength_um=1.55,resolution=res,dpml=1,cell_x=26,cell_y=10);ip=r['input_port'];old,_,_,_=dev._build_sim_single(ip)
 sources=old.sources;geometry=dev.geometry;ports=[dev.port_in,dev.port_out]
 if job['kind'] in ['straight','centered']:
  cy=float(sources[0].center.y) if cy is None else cy;span=p['wg_width_um']+1
  sources=[mp.EigenModeSource(src=mp.GaussianSource(1/1.55,fwidth=.1/1.55),center=mp.Vector3(-11.9 if ip==1 else 11.9,cy),size=mp.Vector3(0,span),eig_band=1,eig_parity=mp.NO_PARITY,eig_match_freq=True,eig_kpoint=mp.Vector3(1 if ip==1 else -1))]
  geometry=[mp.Block(size=mp.Vector3(mp.inf,p['wg_width_um'],mp.inf),center=mp.Vector3(0,cy),material=dev.core_medium)];ports=[mp.Volume(center=mp.Vector3(x,cy),size=mp.Vector3(0,span)) for x in [-11.5,11.5]]
 if job.get('reverse_order'):geometry=[geometry[0],*geometry[1:-1][::-1],geometry[-1]]
 pml=job.get('pml',1);cell=mp.Vector3(26+2*(pml-1),10+2*(pml-1));sim=mp.Simulation(cell_size=cell,resolution=res,boundary_layers=[mp.PML(pml)],geometry=geometry,default_material=dev.clad_medium,sources=sources,eps_averaging=job.get('eps_averaging',True),Courant=job.get('Courant',.5))
 mons=[sim.add_mode_monitor(1/1.55,0,1,mp.ModeRegion(volume=p)) for p in ports];dft=sim.add_dft_fields([mp.Ez],1/1.55,0,1,center=mp.Vector3(),size=mp.Vector3(26,10));dh=sim.add_dft_fields([mp.Hx,mp.Hy],1/1.55,0,1,center=mp.Vector3(),size=mp.Vector3(26,10))
 return fw,r,dev,sim,mons,dft,dh

def snapshot(job,dest,state):
 fw,r,dev,sim,mons,dft,dh=state;mp=fw.mp;dest.mkdir(parents=True,exist_ok=False);a={};modes=[];alpha=[]
 for j,mon in enumerate(mons,1):
  aa,bb,mm=fw.mode_audit(sim,mon,1/1.55,j,a);alpha.append(aa);modes.append(mm)
  md=sim.get_eigenmode(1/1.55,mp.X,mon.where,1,mp.Vector3(),resolution=4*job['resolution'],parity=mp.NO_PARITY);mm['high_eig_resolution']=4*job['resolution'];mm['high_resolution_mode_k']=fw.vec(md.k)
 alpha=np.asarray(alpha);ip=r['input_port'];S=alpha[np.arange(2),[1,0]]/alpha[ip-1,0 if ip==1 else 1];meta=sim.get_array_metadata(dft_cell=dft);E=sim.get_dft_array(dft,mp.Ez,0).T
 a.update(Ez=E,Hx=sim.get_dft_array(dh,mp.Hx,0).T,Hy=sim.get_dft_array(dh,mp.Hy,0).T,x_um=meta.x,y_um=meta.y,eps=sim.get_array(component=mp.Dielectric,center=mp.Vector3(),size=mp.Vector3(26,10)).T,source_center_um=np.array([sim.sources[0].center.x,sim.sources[0].center.y,0]),input_port=ip,resolution=job['resolution'],geometry_id=np.array(r['geometry_id']),alpha_native=alpha,S_native=S)
 assert E.shape==(len(meta.y),len(meta.x)) and abs(len(meta.y)-10*job['resolution'])<=2 and abs(len(meta.x)-26*job['resolution'])<=2 and np.isfinite(E).all() and np.sum(abs(S)**2)<1.5,(E.shape,len(meta.y),len(meta.x),float(np.sum(abs(S)**2)))
 # Actual inverse zz sampled along input straight transverse line; not trace epsilon.
 ys=np.arange(-399,400)/100+float(sim.sources[0].center.y);ys=ys[(ys>-4.8)&(ys<4.8)];xx=-10.5 if ip==1 else 10.5
 a['epszz_profile_y']=ys;a['epszz_profile']=np.array([1/complex(sim.fields.get_inveps(mp.Ez,mp.Z,mp.vec(xx,float(y)))).real for y in ys])
 np.savez_compressed(dest/'sample.npz',**a);save(dest/'result.json',dict(job=job,record=r,source_center_um=a['source_center_um'].tolist(),simulation_time=float(sim.meep_time()),monitors=modes,passed=True,eps_averaging=sim.eps_averaging,subpixel_tol=sim.subpixel_tol,subpixel_maxeval=sim.subpixel_maxeval,Courant=sim.Courant,cell_um=[26+2*(job.get('pml',1)-1),10+2*(job.get('pml',1)-1)]))

def worker(jobid):
 job=read(R/'jobs.json')[int(jobid)];t=time.perf_counter();state=build(job);sim=state[3];mp=state[0].mp
 sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6));snapshot(job,R/'runs'/job['id'],state)
 if job.get('continuation'):
  sim.run(until=100);snapshot(job,R/'runs'/(job['id']+'_plus100'),state)
  strict_start=float(sim.meep_time());sim.run(until=mp.stop_when_dft_decayed(tol=1e-9,minimum_run_time=strict_start+100,maximum_run_time=strict_start+500));snapshot(job,R/'runs'/(job['id']+'_strict'),state)
 sim.reset_meep();save(R/'costs'/(job['id']+'.json'),dict(job=job,wall_seconds=time.perf_counter()-t));print('DONE',job['id'],time.perf_counter()-t,flush=True)

def material(jobid):
 job=read(R/'material_jobs.json')[int(jobid)];t=time.perf_counter();fw,r,dev,sim,mons,dft,dh=build(job);mp=fw.mp;sim.init_sim();A=geom();poly,bend=A.G['centerline_from_params'](r['params']);idx=np.unique(np.linspace(1,len(bend)-2,129).astype(int));centers=bend[idx];tangent=A.G['smooth_tangent_at_x'](centers[:,0],poly);norm=np.column_stack([-tangent[:,1],tangent[:,0]]);eta=np.linspace(-1.8,1.8,145);pts=centers[:,None]+norm[:,None]*eta[None,:,None]*r['params']['wg_width_um']/2;flat=pts.reshape(-1,2)
 ez=np.array([1/complex(sim.fields.get_inveps(mp.Ez,mp.Z,mp.vec(float(x),float(y)))).real for x,y in flat]).reshape(pts.shape[:-1]);trace=np.array([sim.get_epsilon_point(mp.Vector3(float(x),float(y))) for x,y in flat]).reshape(ez.shape)
 occ=np.zeros(len(flat),bool)
 for block in dev.geometry:
  delta=flat-np.array([block.center.x,block.center.y]);u=delta@np.array([block.e1.x,block.e1.y]);v=delta@np.array([block.e2.x,block.e2.y]);occ|=(abs(u)<=block.size.x/2+1e-12)&(abs(v)<=block.size.y/2+1e-12)
 actual=occ.reshape(ez.shape);y_um=np.linspace(-4.8,4.8,961);xx=-10.5 if r['input_port']==1 else 10.5;profile=np.array([1/complex(sim.fields.get_inveps(mp.Ez,mp.Z,mp.vec(xx,float(y)))).real for y in y_um]);dest=R/'materials'/job['id'];dest.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(dest/'samples.npz',epszz=ez,epsilon_trace_average=trace,exact_union_inside=actual,points_um=pts,eta=eta,centers_um=centers,centerline_vertex_indices=idx,straight_profile_y_um=y_um,straight_profile_epszz=profile,eps_core=dev.n_core**2,eps_clad=dev.n_clad**2)
 save(dest/'result.json',dict(job=job,geometry_id=r['geometry_id'],eps_averaging=sim.eps_averaging,subpixel_tol=sim.subpixel_tol,subpixel_maxeval=sim.subpixel_maxeval,overlap_same_material=True,geometry_objects=len(dev.geometry),wall_seconds=time.perf_counter()-t,source='fields.get_inveps(Ez,Z,physical_point), interpolated actual inverse tensor; get_epsilon_point is trace-average diagnostic; exact union analytically tested against unchanged Blocks'))
 sim.reset_meep();print('MATERIAL',job['id'],flush=True)

def launch(action,index,jid,limit=1800):
 assert remaining()>120,'Hard deadline near; preserve partial results';dest=R/'logs'/f'{jid}.log';dest.parent.mkdir(exist_ok=True)
 if dest.exists():dest=dest.with_name(dest.stem+'_resume_'+str(int(time.time()))+'.log')
 with dest.open('x') as f:
  p=subprocess.Popen([PY,'-B','-u',str(R/'experiment.py'),action,str(index)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,remaining()-90))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('Task/deadline timeout '+jid)
 if code:raise RuntimeError('Failed '+str(dest))

def run():
 prepare();c=read(R/'config.json');jobs=[]
 for w in c['widths_um']:
  for res in c['resolutions']:jobs.append(dict(id=f'centered_w{w:.3f}_r{res}',kind='centered',width=w,resolution=res))
 for r in c['diagnostic_records']:
  for res in c['resolutions']:jobs.append(dict(id=f'straight_{r["geometry_id"]}_r{res}',kind='straight',geometry_id=r['geometry_id'],resolution=res))
 for gid in c['new60_ids']:jobs.append(dict(id=f'sbend_{gid}_r60',kind='sbend',geometry_id=gid,resolution=60))
 for gid in c['control_ids']:
  jobs +=[dict(id=f'control_{gid}_r40',kind='sbend',geometry_id=gid,resolution=40,continuation=True),dict(id=f'pml2_{gid}_r40',kind='sbend',geometry_id=gid,resolution=40,pml=2),dict(id=f'noavg_{gid}_r40',kind='sbend',geometry_id=gid,resolution=40,eps_averaging=False),dict(id=f'reordered_{gid}_r40',kind='sbend',geometry_id=gid,resolution=40,reverse_order=True)]
 for w in c['widths_um']:jobs.append(dict(id=f'courant025_w{w:.3f}_r40',kind='centered',width=w,resolution=40,Courant=.25))
 save(R/'jobs.json',jobs);save(R/'jobs_freeze.json',dict(unix=time.time(),jobs_sha256=sha(R/'jobs.json'),before_jobs=True))
 for i,j in enumerate(jobs):
  launch('worker',i,j['id']);save(R/'progress.json',dict(stage='FDTD',completed=i+1,total=len(jobs),last=j['id'],remaining_budget_seconds=remaining()));print('FDTD',i+1,len(jobs),j['id'],flush=True)
 mj=[]
 for r in c['diagnostic_records']:
  for res in c['resolutions']:mj.append(dict(id=f'original_{r["geometry_id"]}_r{res}',kind='sbend',geometry_id=r['geometry_id'],resolution=res))
 for gid in c['control_ids']:
  for res in [20,40]:
   mj +=[dict(id=f'noavg_{gid}_r{res}',kind='sbend',geometry_id=gid,resolution=res,eps_averaging=False),dict(id=f'reordered_{gid}_r{res}',kind='sbend',geometry_id=gid,resolution=res,reverse_order=True)]
 save(R/'material_jobs.json',mj)
 for i,j in enumerate(mj):
  launch('material',i,'material_'+j['id']);save(R/'progress.json',dict(stage='material_initialization',completed=i+1,total=len(mj),remaining_budget_seconds=remaining()))
 save(R/'primary_complete.json',dict(unix=time.time(),fdtd_jobs=len(jobs),material_jobs=len(mj),remaining_budget_seconds=remaining(),conditional80_pending=True));print('PRIMARY_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='run':run()
  elif sys.argv[1]=='worker':worker(sys.argv[2])
  elif sys.argv[1]=='material':material(sys.argv[2])
 except Exception:
  traceback.print_exc()
  if sys.argv[1]=='run':save(R/'stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()))
  raise



