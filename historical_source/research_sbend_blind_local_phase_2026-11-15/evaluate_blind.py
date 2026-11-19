"""Apply preregistered metric definitions to frozen prospective data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MPLBACKEND','Agg')
import sys,time,json,csv,cmath,math
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from experiment import R,BASE,OLD,SEEDS,METHODS,read,save,sha,csvsave,metrics,remaining
A=metrics();CFG=read(R/'config.json');RECORDS=read(R/'geometry_manifest.json')['records']

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def predictions():
 out={}
 for s in SEEDS:
  for m in METHODS:
   with np.load(R/'predictions'/f'seed{s}_{m}.npz') as z:
    assert z['geometry_id'].tolist()==[r['geometry_id'] for r in RECORDS];assert z['input_port'].tolist()==[r['input_port'] for r in RECORDS];out[s,m]=z['prediction']
 return out

def interp(e,x,y,points,order=1):
 coords=np.stack([(points[...,1]-y[0])/(y[1]-y[0]),(points[...,0]-x[0])/(x[1]-x[0])]);assert coords.min()>=-1e-6 and coords[0].max()<len(y) and coords[1].max()<len(x)
 return map_coordinates(e,coords,order=order,mode='nearest',prefilter=order>1)

def incoming(a):
 ip=int(a['input_port']);pre=f'port{ip}/';cy=float(a['source_center_um'][1]);y=a[pre+'y_um'];em=a[pre+'mode_Ez'];hm=a[pre+'mode_Hy'];w=a[pre+'weights'];P=float(-np.real(np.sum(w*em.conj()*hm)));idir=0 if ip==1 else 1
 alpha=a[pre+'alpha_EH_grid'][idir];native=a[pre+'alpha_native'][idir];center=np.interp(cy,y,em.real)+1j*np.interp(cy,y,em.imag);q=alpha*center/np.sqrt(P)
 assert abs(q)>1e-8 and P>0
 return q,dict(input_port=ip,monitor_x_um=float(a[pre+'x_um'][0]),center_y_um=cy,incident_Ez_real=float(q.real),incident_Ez_imag=float(q.imag),EH_vs_native_amplitude_relative=float(abs(alpha/native-1)),EH_vs_native_phase_deg=float(np.degrees(np.angle(alpha/native))),normalization='independent EH-grid incoming coefficient times unit-power-mode Ez at physical center; fixed input monitor')

def perfield(p,e,masks,gate,maxamp,regions):
 out=[]
 for name in regions:
  v=A.fieldmetric(p,e,masks[name],gate,maxamp)
  if v is not None:out.append(dict(region=name,gate=gate,eta=0.,**v))
 return out

def aggregate(rows,prefix):
 # Long-form metric rows; all finite; equal geometry weights; paired by geometry.
 metrics_by_kind={'field':['phase_wmae_deg','complex_rel_l2'],'pair':['normal_relative_phase_wmae_deg','coherence_complex_rel_l2']}
 summary=[];paired=[];lookup={}
 for r in rows:
  key=(r['kind'],r.get('truth_resolution',20),r.get('convention','raw'),r['seed'],r['method'],r['region'],r['gate'],r['eta']);lookup.setdefault(key,[]).append(r)
 for key,rr in lookup.items():
  kind,res,conv,s,m,region,gate,eta=key
  for group in ['all','low','middle','high','input1','input2']:
   aa=[r for r in rr if group=='all' or r['angle_stratum']==group or group=='input'+str(r['input_port'])]
   if not aa:continue
   for met in metrics_by_kind[kind]:summary.append(dict(kind=kind,truth_resolution=res,convention=conv,seed=s,method=m,region=region,gate=gate,eta=eta,group=group,metric=met,n=len(aa),mean=float(np.mean([r[met] for r in aa]))))
   if m!='local_tangent':continue
   primaryregion=(kind=='field' and region=='bend_core') or (kind=='pair' and region=='normal_pair_all' and eta==.5)
   # Spatial-cell comparisons are descriptive means; primary/angle bins retain paired CIs.
   if region.startswith('cell_') or (group!='all' and not primaryregion):continue
   for baseline in ['arc_only','linear','path_v2']:
    bb=lookup.get((kind,res,conv,s,baseline,region,gate,eta),[]);b={r['geometry_id']:r for r in bb};a={r['geometry_id']:r for r in aa};ids=sorted(set(a)&set(b))
    if not ids:continue
    draw=np.random.default_rng(20261115).integers(0,len(ids),(10000,len(ids)))
    for met in metrics_by_kind[kind]:
     av=np.array([a[g][met] for g in ids]);bv=np.array([b[g][met] for g in ids]);d=av-bv;ci=np.quantile(d[draw].mean(axis=1),[.025,.975]);paired.append(dict(kind=kind,truth_resolution=res,convention=conv,seed=s,baseline=baseline,region=region,gate=gate,eta=eta,group=group,metric=met,n=len(ids),baseline_mean=float(bv.mean()),LT_mean=float(av.mean()),delta=float(d.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),improved=int(np.sum(d<0))))
 csvsave(R/f'{prefix}_summary.csv',summary);csvsave(R/f'{prefix}_paired.csv',paired);save(R/f'{prefix}_paired.json',paired);save(R/f'{prefix}_summary.json',summary)
 return paired

def blind():
 P=predictions();rows=[];coverage=[];indchecks=[]
 for i,r in enumerate(RECORDS):
  with np.load(R/'data'/r['geometry_id']/'sample.npz') as z:a={k:z[k] for k in ['Ez','x_um','y_um']}
  e=a['Ez'].astype(np.complex128);poly,bend,masks=A.geometry(r,a);maxamp=float(abs(e[masks['bend_core']]).max());coords={eta:A.paircoords(r,a,poly,bend,eta) for eta in [.5,1.]}
  coverage.append(dict(geometry_id=r['geometry_id'],angle_stratum=r['angle_stratum'],max_angle_deg=r['max_local_angle_deg'],input_port=r['input_port'],**{name:int(mask.sum()) for name,mask in masks.items()}))
  for s in SEEDS:
   for m in METHODS:
    p=P[s,m][i].astype(np.complex128);base=dict(seed=s,method=m,geometry_id=r['geometry_id'],input_port=r['input_port'],angle_stratum=r['angle_stratum'],truth_resolution=20,convention='raw')
    for gate in [.05,.01,.1]:
     regions=list(masks) if gate==.05 else ['bend_core','straight_core']
     rows.extend(dict(**base,kind='field',**v) for v in perfield(p,e,masks,gate,maxamp,regions))
     for eta,(plus,minus,angle) in coords.items():
      for ab in [None,0,1,2,3]:
       if gate!=.05 and (eta!=.5 or ab is not None):continue
       v=A.pairmetric(p,e,a,plus,minus,angle,gate,maxamp,ab)
       if v is not None:rows.append(dict(**base,kind='pair',region='normal_pair_all' if ab is None else f'normal_pair_angle{ab}',gate=gate,eta=eta,**v))
    if i in [0,35,71]:
     mask=masks['bend_core']&(abs(e)>=.05*maxamp);ee=e[mask];pp=p[mask];ww=abs(ee)**2;val=sum(float(w)*abs(cmath.phase(complex(pv)/complex(ev))) if abs(pv)>1e-12*maxamp else float(w)*math.pi for w,pv,ev in zip(ww,pp,ee))/float(ww.sum())*180/math.pi
     ref=A.fieldmetric(p,e,masks['bend_core'],.05,maxamp)['phase_wmae_deg'];assert abs(val-ref)<1e-10;indchecks.append(abs(val-ref))
  if (i+1)%6==0:print('BLIND_ANALYZED',i+1,72,flush=True)
 csvsave(R/'blind_per_geometry.csv',rows);save(R/'blind_coverage.json',coverage);paired=aggregate(rows,'blind');primary=[r for r in paired if r['baseline']=='arc_only' and r['group']=='all' and r['gate']==.05 and ((r['region']=='bend_core' and r['metric']=='phase_wmae_deg') or (r['region']=='normal_pair_all' and r['eta']==.5 and r['metric']=='normal_relative_phase_wmae_deg'))];assert len(primary)==6
 save(R/'blind_primary.json',primary);save(R/'blind_verification.json',dict(scalar_independent_checks=len(indchecks),max_phase_difference_deg=max(indchecks),geometry_ids_ports_checked=True,full_predictions=12));print(json.dumps(primary),flush=True)

def directpair(p,e,a0,af,plus,minus,gate,maxamp,order=1):
 pp=interp(p,a0['x_um'],a0['y_um'],plus);pm=interp(p,a0['x_um'],a0['y_um'],minus);ep=interp(e,af['x_um'],af['y_um'],plus,order);em=interp(e,af['x_um'],af['y_um'],minus,order);good=(abs(ep)>=gate*maxamp)&(abs(em)>=gate*maxamp)
 if good.sum()<8:return None
 ct=ep[good]*em[good].conj();cp=pp[good]*pm[good].conj();phase=np.angle(cp*ct.conj());phase=np.where(abs(cp)>1e-24*maxamp**2,phase,np.pi);w=abs(ct)
 return dict(pairs=int(good.sum()),normal_relative_phase_wmae_deg=float(np.degrees(np.sum(w*abs(phase))/w.sum())),coherence_complex_rel_l2=float(np.sqrt(np.sum(abs(cp-ct)**2)/np.sum(abs(ct)**2))))

def grid(resolutions=[20,30,40],only_ids=None,prefix='grid'):
 P=predictions();rows=[];truthrows=[];qrows=[];interp_rows=[]
 for i,r in enumerate(RECORDS):
  gid=r['geometry_id']
  if gid not in (only_ids or CFG['fine_geometry_ids']):continue
  a0=load(R/'data'/gid/'sample.npz');poly,bend,masks=A.geometry(r,a0);X,Y=np.meshgrid(a0['x_um'],a0['y_um']);points=np.stack([X,Y],axis=-1);plus,minus,ang=A.paircoords(r,a0,poly,bend,.5);truth={};coarse={};q={}
  for res in resolutions:
   a=a0 if res==20 else load(R/'fine'/gid/f'res{res}/sample.npz');truth[res]=a;e=a['Ez'].astype(np.complex128);coarse[res]=e if res==20 else interp(e,a['x_um'],a['y_um'],points);q[res],qr=incoming(a);qrows.append(dict(geometry_id=gid,resolution=res,**qr));assert np.max(abs(a['source_center_um']-a0['source_center_um']))<1e-12
   maxamp=float(abs(coarse[res][masks['bend_core']]).max())
   for s in SEEDS:
    for m in METHODS:
     pp=P[s,m][i].astype(np.complex128)
     for conv in ['raw','incident_normalized']:
      en=coarse[res] if conv=='raw' else coarse[res]/q[res];pn=pp if conv=='raw' else pp/q[20];ma=float(abs(en[masks['bend_core']]).max());base=dict(seed=s,method=m,geometry_id=gid,input_port=r['input_port'],angle_stratum=r['angle_stratum'],truth_resolution=res,convention=conv)
      for gate in [.05,.01,.1]:
       v=A.fieldmetric(pn,en,masks['bend_core'],gate,ma);rows.append(dict(**base,kind='field',region='bend_core',gate=gate,eta=0.,**v))
      # Relative phase identical under incident normalization; cross-term amplitude is not.
      ef=e if conv=='raw' else e/q[res];af=a;v=directpair(pn,ef,a0,af,plus,minus,.05,ma)
      if v is not None:rows.append(dict(**base,kind='pair',region='normal_pair_all',gate=.05,eta=.5,**v))
     if res in [30,40,60]:
      vl=directpair(pp,e,a0,a,plus,minus,.05,maxamp,1);vc=directpair(pp,e,a0,a,plus,minus,.05,maxamp,3)
      interp_rows.append(dict(geometry_id=gid,resolution=res,seed=s,method=m,pair_metric_linear_deg=vl['normal_relative_phase_wmae_deg'],pair_metric_cubic_deg=vc['normal_relative_phase_wmae_deg'],absolute_metric_difference_deg=abs(vl['normal_relative_phase_wmae_deg']-vc['normal_relative_phase_wmae_deg'])))
  for lo,hi in zip(resolutions[:-1],resolutions[1:]):
   al,ah=truth[lo],truth[hi];el=al['Ez'];eh=ah['Ez'];ma=float(abs(coarse[hi][masks['bend_core']]).max())
   for conv in ['raw','incident_normalized']:
    pl=coarse[lo] if conv=='raw' else coarse[lo]/q[lo];th=coarse[hi] if conv=='raw' else coarse[hi]/q[hi];v=A.fieldmetric(pl,th,masks['bend_core'],.05,float(abs(th[masks['bend_core']]).max()));truthrows.append(dict(geometry_id=gid,low_resolution=lo,high_resolution=hi,convention=conv,region='bend_core',**v))
   v=directpair(el,eh,al,ah,plus,minus,.05,ma);truthrows.append(dict(geometry_id=gid,low_resolution=lo,high_resolution=hi,convention='raw',region='normal_pair_all',**v))
  print('GRID_ANALYZED',gid,resolutions,flush=True)
 csvsave(R/f'{prefix}_per_geometry.csv',rows);csvsave(R/f'{prefix}_truth_changes.csv',truthrows);save(R/f'{prefix}_truth_changes.json',truthrows);csvsave(R/f'{prefix}_incoming_reference.csv',qrows);save(R/f'{prefix}_incoming_reference.json',qrows);csvsave(R/f'{prefix}_interpolation.csv',interp_rows);save(R/f'{prefix}_interpolation.json',interp_rows);aggregate(rows,prefix)

def decision():
 blind=read(R/'blind_primary.json');gp=read(R/'grid_paired.json');tc=read(R/'grid_truth_changes.json');ic=read(R/'grid_interpolation.json')
 mean=lambda a:float(np.mean(a))
 normal=mean([r['normal_relative_phase_wmae_deg'] for r in tc if r['low_resolution']==30 and r['region']=='normal_pair_all']);phase=mean([r['phase_wmae_deg'] for r in tc if r['low_resolution']==30 and r['region']=='bend_core' and r['convention']=='incident_normalized']);inter=mean([r['absolute_metric_difference_deg'] for r in ic if r['resolution']==40])
 primary=[r for r in gp if r['baseline']=='arc_only' and r['group']=='all' and r['gate']==.05 and r['metric'] in ['phase_wmae_deg','normal_relative_phase_wmae_deg']];assert len(primary)==36
 check=dict(blind_both_CIs_below_zero_all_seeds=all(r['ci95_high']<0 for r in blind),grid_phase_deltas_negative_all_seeds_resolutions_conventions=all(r['delta']<0 for r in primary),normal_truth30_40_mean_deg=normal,normal_truth_gate_pass=normal<=.25,incident_normalized_truth30_40_mean_deg=phase,incident_phase_gate_pass=phase<=.5,interpolation_metric_difference_res40_mean_deg=inter,interpolation_gate_pass=inter<=.1)
 passed=all(check[k] for k in ['blind_both_CIs_below_zero_all_seeds','grid_phase_deltas_negative_all_seeds_resolutions_conventions','normal_truth_gate_pass','incident_phase_gate_pass','interpolation_gate_pass'])
 save(R/'decision.json',dict(checks=check,application_readiness_gate_passed=passed,reverse20_eligible=passed,remaining_budget_seconds=remaining(),no_coherent_application=True));print(json.dumps(check),flush=True)

def verify():
 for p,h in read(R/'protected_inputs.json').items():assert sha(p)==h,p
 for p,h in read(R/'freeze.json')['hashes'].items():assert sha(p)==h,p
 for p,h in read(R/'prediction_freeze.json')['hashes'].items():assert sha(p)==h,p
 save(R/'preservation_check.json',dict(passed=True,protected_inputs=len(read(R/'protected_inputs.json')),all_input_hashes_unchanged=True,freeze_unchanged=True,predictions_unchanged=True))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='blind':blind()
 elif action=='grid':grid();decision()
 elif action=='refined':grid([20,30,40,60],CFG['conditional_res60_ids'],'refined')
 elif action=='verify':verify()
