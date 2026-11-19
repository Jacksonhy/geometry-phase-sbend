"""Saved-array, paired local phase diagnostic. No solver, inference or training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import ast,json,csv,time,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent;BASE=R.parent;OLD=BASE/'research_sbend_formal'
SEEDS=[42,2026,2027];METHODS=['linear','arc_only','local_tangent','path_v2']
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
def csvsave(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predpath(seed,m):
 if seed==2026:return (BASE/'research_sbend_fourmethod_5seed/predictions' if m=='arc_only' else OLD/'predictions')/(m+'_best.npz')
 return BASE/'research_sbend_fourmethod_5seed'/f'seed{seed}'/'predictions'/(m+'_best.npz')
def functions():
 p=BASE/'PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py';t=ast.parse(p.read_text(encoding='utf-8-sig'));node=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='euler_sbend_points');e={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),e);f=e['euler_sbend_points']
 p=BASE/'research_sbend_path/path_geometry.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'));names=['centerline_from_params','curve_info','project_points','smooth_tangent_at_x'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];e['official_euler_function']=lambda:f;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),e);return e
G=functions()
def geometry(rec,a):
 x,y=a['x_um'],a['y_um'];poly,bend=G['centerline_from_params'](rec['params']);X,Y=np.meshgrid(x,y);projection=G['project_points'](np.stack([X,Y],axis=-1),poly)
 p=projection['closest'];tx=G['smooth_tangent_at_x'](p[...,0],poly);angle=np.degrees(np.arctan2(abs(tx[...,1]),tx[...,0]));eta=2*projection['normal_offset']/rec['params']['wg_width_um'];is_bend=(X>=bend[0,0])&(X<=bend[-1,0]);core=projection['distance']<=rec['params']['wg_width_um']/2
 masks={'whole':np.ones(core.shape,bool),'bend_core':is_bend&core,'straight_core':~is_bend&core,'bend_near_cladding':is_bend&(abs(eta)>1)&(abs(eta)<=3)}
 ab=[0.,10.,20.,35.,90.];eb=[-3.,-1.,-.5,0.,.5,1.,3.]
 for k in range(len(ab)-1):
  am=(angle>=ab[k])&(angle<ab[k+1]);masks[f'angle{k}_core']=is_bend&core&am
  for j in range(len(eb)-1):masks[f'cell_a{k}_n{j}']=is_bend&am&(eta>=eb[j])&(eta<eb[j+1])
 for j in range(len(eb)-1):masks[f'normal{j}']=is_bend&(eta>=eb[j])&(eta<eb[j+1])
 return poly,bend,masks

def fieldmetric(p,e,mask,gate,maxamp):
 n=int(mask.sum())
 if n<10:return None
 power=abs(e)**2;den=float(power[mask].sum())
 if den<=0:return None
 err=abs(p-e)**2;delta=np.angle(p*e.conj());delta=np.where(abs(p)>1e-12*maxamp,delta,np.pi)
 good=mask&(abs(e)>=gate*maxamp);pd=float(power[good].sum());ng=int(good.sum())
 if ng<10 or pd<=0:return None
 amp=abs(p)-abs(e);phasepart=2*abs(p)*abs(e)*(1-np.cos(delta));decomp=float((amp[mask]**2+phasepart[mask]).sum())
 assert abs(decomp-float(err[mask].sum()))<1e-8*max(den,1.)
 return dict(pixels=n,phase_pixels=ng,true_energy_fraction=float(den/power.sum()),complex_rel_l2=float(np.sqrt(err[mask].sum()/den)),phase_wmae_deg=float(np.degrees(np.sum(power[good]*abs(delta[good]))/pd)),phase_phasor_l2=float(np.sqrt(np.sum(power[good]*abs(np.exp(1j*delta[good])-1)**2)/pd)),amplitude_rel_l2=float(np.sqrt(np.sum(amp[mask]**2)/den)),phase_term_normalized_mse=float(np.sum(phasepart[mask])/den))

def samples(field,x,y,points):
 coords=np.stack([(points[:,1]-y[0])/(y[1]-y[0]),(points[:,0]-x[0])/(x[1]-x[0])]);return map_coordinates(field,coords,order=1,mode='nearest',prefilter=False)
def paircoords(rec,a,poly,bend,eta):
 x=a['x_um'];xx=x[(x>=bend[0,0]+.1)&(x<=bend[-1,0]-.1)];yc=np.interp(xx,poly[:,0],poly[:,1]);t=G['smooth_tangent_at_x'](xx,poly);normal=np.column_stack([-t[:,1],t[:,0]]);base=np.column_stack([xx,yc]);distance=eta*rec['params']['wg_width_um']/2
 return base+distance*normal,base-distance*normal,np.degrees(np.arctan2(abs(t[:,1]),t[:,0]))
def pairmetric(p,e,a,plus,minus,angle,gate,maxamp,anglebin):
 ep=samples(e,a['x_um'],a['y_um'],plus);em=samples(e,a['x_um'],a['y_um'],minus);pp=samples(p,a['x_um'],a['y_um'],plus);pm=samples(p,a['x_um'],a['y_um'],minus)
 good=(abs(ep)>=gate*maxamp)&(abs(em)>=gate*maxamp)
 if anglebin is not None:
  bounds=[0,10,20,35,90];good&=(angle>=bounds[anglebin])&(angle<bounds[anglebin+1])
 if good.sum()<8:return None
 ct=ep[good]*em[good].conj();cp=pp[good]*pm[good].conj();phase=np.angle(cp*ct.conj());phase=np.where(abs(cp)>1e-24*maxamp**2,phase,np.pi);w=abs(ct)
 return dict(pairs=int(good.sum()),normal_relative_phase_wmae_deg=float(np.degrees(np.sum(w*abs(phase))/w.sum())),coherence_complex_rel_l2=float(np.sqrt(np.sum(abs(cp-ct)**2)/np.sum(abs(ct)**2))),normal_relative_phase_phasor_l2=float(np.sqrt(np.sum(w*abs(np.exp(1j*phase)-1)**2)/w.sum())))

def prepare():
 assert not (R/'freeze.json').exists()
 cfg=dict(seeds=SEEDS,methods=METHODS,checkpoint='best',evaluation='historical100_retrospective',main_comparison='local_tangent_minus_arc_only',phase_amplitude_gates=[.05,.01,.10],primary_gate=.05,angle_edges_deg=[0,10,20,35,90],normal_coordinate='2*nearest_centerline_signed_normal_offset/width',normal_edges=[-3,-1,-.5,0,.5,1,3],normal_pair_eta=[.5,1.],primary_pair_eta=.5,min_roi_pixels=10,min_pairs=8,bootstrap_replicates=10000,bootstrap_seed=20261115,statistics='geometry-paired conditional-on-fixed-model percentile CI, separate seeds; no pixel or pooled300 bootstrap',no_phase_alignment=True,no_amplitude_fit=True,no_FDTD=True,no_inference=True,no_training=True,coherence_metric='E_plus*conj(E_minus) at normal pairs; transverse coherence diagnostic, NOT bidirectional port interference',bidirectional_requirement='same geometry, both source directions with shared coherent source convention; no reflection/conjugation surrogate',decision='descriptive consistent local phase advantage requires smaller bend-core absolute phase WMAE and normal-pair relative phase WMAE in all three seeds; complex error separately; CI crossing zero weakens evidence; no complex application simulation follows automatically')
 save(R/'config.json',cfg)
 records=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split']=='test'];assert len(records)==100
 paths=[OLD/'geometry_manifest.json',BASE/'research_sbend_path/path_geometry.py',BASE/'PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py']+[predpath(s,m) for s in SEEDS for m in METHODS]+[OLD/'data'/r['geometry_id']/'sample.npz' for r in records]
 save(R/'sources_before.json',{str(p):sha(p) for p in paths});save(R/'freeze.json',dict(time_unix=time.time(),config_sha256=sha(R/'config.json'),protocol_sha256=sha(R/'PROTOCOL.md'),script_sha256=sha(R/'analyze.py'),sources_sha256=sha(R/'sources_before.json'),metric_results_computed=False));print('FROZEN',len(paths),'inputs',flush=True)

def inventory(records):
 manifests=[('formal',read(OLD/'geometry_manifest.json')['records']),('development',read(BASE/'research_meep_validation/sbend_index.json'))]
 phase=BASE/'research_sbend_phase_selection_2026-11-15'
 for stage in ['development','formal']:
  p=phase/stage/'geometry_manifest.json'
  if p.exists():manifests.append(('phase_selection_'+stage,read(p)['records']))
 group={}
 for tag,rr in manifests:
  for r in rr:
   key=tuple(round(r['params'][k],12) for k in ['wg_width_um','lateral_offset_um','R_min_um']);group.setdefault(key,[]).append((tag,r['geometry_id'],r['input_port']))
 pairs=[v for v in group.values() if len({q[2] for q in v})==2];asym=[]
 for r in records:
  with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:ep=z['eps'];asym.append(dict(geometry_id=r['geometry_id'],max_abs_eps_minus_rotated=float(abs(ep-ep[::-1,::-1]).max()),different_eps_pixels=int(np.count_nonzero(ep!=ep[::-1,::-1]))))
 save(R/'bidirectional_inventory.json',dict(manifests_counts={tag:len(rr) for tag,rr in manifests},same_parameter_opposite_input_pairs=pairs,formal_eval_directions={str(i):sum(r['input_port']==i for r in records) for i in [1,2]},formal_eval_exact_centro_symmetric=sum(r['different_eps_pixels']==0 for r in asym),rotation_checks=asym,can_evaluate_true_two_port_interference=bool(pairs),substitution_applied=False))

def run():
 cfg=read(R/'config.json');records=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split']=='test'];preds={};rows=[];prows=[];coverage=[]
 for seed in SEEDS:
  for m in METHODS:
   with np.load(predpath(seed,m),allow_pickle=False) as z:
    ids=z['geometry_id'].tolist();ips=z['input_port'].tolist();pr=z['prediction'];assert len(set(ids))==100
    preds[seed,m]={gid:(pr[i],int(ips[i])) for i,gid in enumerate(ids)}
 inventory(records)
 for index,r in enumerate(records):
  gid=r['geometry_id']
  with np.load(OLD/'data'/gid/'sample.npz',allow_pickle=False) as z:a={k:z[k] for k in ['Ez','eps','x_um','y_um']};assert str(z['geometry_id'])==gid and int(z['input_port'])==r['input_port']
  e=a['Ez'].astype(np.complex128);poly,bend,masks=geometry(r,a);maxamp=float(abs(e[masks['bend_core']]).max());coords={eta:paircoords(r,a,poly,bend,eta) for eta in [.5,1.]}
  for seed in SEEDS:
   for m in METHODS:
    p,ip=preds[seed,m][gid];assert ip==r['input_port'];p=p.astype(np.complex128)
    for gate in cfg['phase_amplitude_gates']:
     for region,mask in masks.items():
      if gate!=.05 and region not in ['bend_core','straight_core']:continue
      v=fieldmetric(p,e,mask,gate,maxamp)
      if v is not None:rows.append(dict(seed=seed,method=m,geometry_id=gid,input_port=ip,region=region,gate=gate,**v))
     for eta,(plus,minus,ang) in coords.items():
      for ab in [None,0,1,2,3]:
       if gate!=.05 and (eta!=.5 or ab is not None):continue
       v=pairmetric(p,e,a,plus,minus,ang,gate,maxamp,ab)
       if v is not None:prows.append(dict(seed=seed,method=m,geometry_id=gid,input_port=ip,region='normal_pair_all' if ab is None else f'normal_pair_angle{ab}',gate=gate,eta=eta,**v))
  coverage.append(dict(geometry_id=gid,bend_core_pixels=int(masks['bend_core'].sum()),bend_core_max_amp=maxamp))
  if (index+1)%10==0:print('ANALYZED',index+1,100,flush=True)
 csvsave(R/'per_geometry_fields.csv',rows);csvsave(R/'per_geometry_normal_pairs.csv',prows);save(R/'geometry_coverage.json',coverage)
 print('SAVED',len(rows),'field rows',len(prows),'normal pair rows',flush=True)
 summarize(rows,prows)

def summarize(fields,pairs):
 sums=[];comparisons=[]
 for typ,rows,metrics in [('field',fields,['complex_rel_l2','phase_wmae_deg','phase_phasor_l2','amplitude_rel_l2','phase_term_normalized_mse']),('normal_pair',pairs,['normal_relative_phase_wmae_deg','coherence_complex_rel_l2','normal_relative_phase_phasor_l2'])]:
  keys=sorted({(r['region'],r['gate'],r.get('eta',0)) for r in rows});lookup={(r['seed'],r['method'],r['region'],r['gate'],r.get('eta',0),r['geometry_id']):r for r in rows}
  for seed in SEEDS:
   for region,gate,eta in keys:
    for port in [0,1,2]:
     subset=[r for r in rows if r['seed']==seed and r['region']==region and r['gate']==gate and r.get('eta',0)==eta and (port==0 or r['input_port']==port)]
     for m in METHODS:
      a=[r for r in subset if r['method']==m]
      if not a:continue
      sums.append(dict(kind=typ,seed=seed,method=m,region=region,gate=gate,eta=eta,input_port=port,n=len(a),**{k:float(np.mean([r[k] for r in a])) for k in metrics}))
     for baseline in ['arc_only','linear','path_v2']:
      a={r['geometry_id']:r for r in subset if r['method']=='local_tangent'};b={r['geometry_id']:r for r in subset if r['method']==baseline};ids=sorted(set(a)&set(b))
      if not ids:continue
      # One set of paired geometry bootstrap indices reused across these metrics.
      draw=np.random.default_rng(20261115).integers(0,len(ids),(10000,len(ids)))
      for metric in metrics:
       aa=np.array([a[g][metric] for g in ids]);bb=np.array([b[g][metric] for g in ids]);d=aa-bb;ci=np.quantile(d[draw].mean(axis=1),[.025,.975])
       comparisons.append(dict(kind=typ,seed=seed,baseline=baseline,candidate='local_tangent',region=region,gate=gate,eta=eta,input_port=port,metric=metric,n=len(ids),baseline_mean=float(bb.mean()),local_tangent_mean=float(aa.mean()),delta_LT_minus_baseline=float(d.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),improved_geometries=int(sum(d<0)),worsened_geometries=int(sum(d>0)),ties=int(sum(d==0))))
 csvsave(R/'summary_per_seed.csv',sums);csvsave(R/'paired_comparisons.csv',comparisons);save(R/'paired_comparisons.json',comparisons)
 primary=[r for r in comparisons if r['baseline']=='arc_only' and r['input_port']==0 and r['gate']==.05 and ((r['region']=='bend_core' and r['metric'] in ['complex_rel_l2','phase_wmae_deg']) or (r['region']=='normal_pair_all' and r['eta']==.5 and r['metric'] in ['normal_relative_phase_wmae_deg','coherence_complex_rel_l2']))]
 gate={}
 for metric in ['phase_wmae_deg','normal_relative_phase_wmae_deg','complex_rel_l2','coherence_complex_rel_l2']:
  rr=[r for r in primary if r['metric']==metric];assert len(rr)==3;gate[metric]=dict(all_three_means_improve=all(r['delta_LT_minus_baseline']<0 for r in rr),all_three_geometry_CI_below_zero=all(r['ci95_high']<0 for r in rr),per_seed=[dict(seed=r['seed'],baseline_mean=r['baseline_mean'],LT_mean=r['local_tangent_mean'],delta=r['delta_LT_minus_baseline'],ci95=[r['ci95_low'],r['ci95_high']]) for r in rr])
 save(R/'decision.json',dict(primary_LT_vs_ArcOnly=gate,consistent_phase_mean_advantage=gate['phase_wmae_deg']['all_three_means_improve'] and gate['normal_relative_phase_wmae_deg']['all_three_means_improve'],new_application_simulation_authorized_or_launched=False,bidirectional_interference='not evaluable without both-direction same-geometry truth; transverse coherence is separately labelled',scope='retrospective fixed100 geometries and three existing best checkpoints'))
 for r in primary:print(json.dumps(r),flush=True)

def verify():
 for p,h in read(R/'sources_before.json').items():assert sha(p)==h,p
 f=read(R/'freeze.json');assert sha(R/'analyze.py')==f['script_sha256'];assert sha(R/'config.json')==f['config_sha256'];assert sha(R/'PROTOCOL.md')==f['protocol_sha256']
 with (R/'per_geometry_fields.csv').open(encoding='utf-8-sig') as z:rows=list(csv.DictReader(z))
 # Whole-domain saved fields must reproduce historical means, independently of bins.
 expected={42:{'linear':.400205,'arc_only':.166673,'local_tangent':.148650,'path_v2':.149312},2026:{'linear':.337387,'arc_only':.161969,'local_tangent':.153715,'path_v2':.153420},2027:{'linear':.300163,'arc_only':.166983,'local_tangent':.154842,'path_v2':.154553}};replays=[]
 for seed in SEEDS:
  for method in METHODS:
   a=[float(r['complex_rel_l2']) for r in rows if int(r['seed'])==seed and r['method']==method and r['region']=='whole' and float(r['gate'])==.05];assert len(a)==100;mean=float(np.mean(a));diff=abs(mean-expected[seed][method]);assert diff<5.1e-7,(seed,method,mean);replays.append(dict(seed=seed,method=method,mean=mean,difference_from_rounded_history=diff))
 save(R/'verification.json',dict(passed=True,protected_files=len(read(R/'sources_before.json')),all_protected_inputs_unchanged=True,whole_domain_replays=replays,training_runs=0,model_inference_runs=0,FDTD_runs=0,phase_or_amplitude_fits=0,geometry_ID_and_input_port_checked=True))
 print('VERIFIED',flush=True)
if __name__=='__main__':
 t=time.perf_counter();prepare();run();verify();save(R/'complete.json',dict(completed=True,wall_seconds=time.perf_counter()-t));print('LOCAL_PHASE_COMPLETE',flush=True)
