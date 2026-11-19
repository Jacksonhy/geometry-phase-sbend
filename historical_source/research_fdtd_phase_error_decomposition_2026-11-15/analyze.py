"""Frozen-definition analysis for FDTD phase diagnostics; no neural imports."""
import sys,csv,cmath,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import map_coordinates
from experiment import R,BASE,B,read,save,sha,prior,geom,remaining
A=geom();CFG=read(R/'config.json');RECORDS=CFG['diagnostic_records']
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvsave(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def interpolate(a,points,key='Ez'):
 xy=np.stack([(points[...,1]-a['y_um'][0])/(a['y_um'][1]-a['y_um'][0]),(points[...,0]-a['x_um'][0])/(a['x_um'][1]-a['x_um'][0])]);assert xy[0].min()>=-1e-8 and xy[1].min()>=-1e-8 and xy[0].max()<=len(a['y_um'])-1+1e-8 and xy[1].max()<=len(a['x_um'])-1+1e-8
 return map_coordinates(a[key],xy,order=1,mode='nearest',prefilter=False)
def port_E(a,j,direction,cy):
 pre=f'port{j}/';em=a[pre+'mode_Ez'];hm=a[pre+'mode_Hy'];power=float(-np.real(np.sum(a[pre+'weights']*em.conj()*hm)));center=np.interp(cy,a[pre+'y_um'],em.real)+1j*np.interp(cy,a[pre+'y_um'],em.imag);return a[pre+'alpha_EH_grid'][direction]*center/np.sqrt(power)
def incident(a):return port_E(a,int(a['input_port']),0 if int(a['input_port'])==1 else 1,float(a['source_center_um'][1]))
def physical_t(a,r=None,straight=False):
 ip=int(a['input_port']);op=3-ip
 if straight:cy=float(a['source_center_um'][1])
 else:
  poly,_=A.G['centerline_from_params'](r['params']);cy=float(poly[0 if op==1 else -1,1])
 return port_E(a,op,1 if op==1 else 0,cy)/incident(a)
def sbend(gid,res):
 p=prior(gid,res)
 if not p.exists():p=R/'runs'/f'sbend_{gid}_r{res}'/'sample.npz'
 if not p.exists():p=R/'runs'/f'res80_{gid}'/'sample.npz'
 return load(p)
def slope(x,z,sig,window):
 good=(x>=window[0])&(x<=window[1]);xx=sig*x[good];ph=np.unwrap(np.angle(z[good]));w=abs(z[good])**2;center=np.sum(w*xx)/w.sum();pc=np.sum(w*ph)/w.sum();beta=float(np.sum(w*(xx-center)*(ph-pc))/np.sum(w*(xx-center)**2));intercept=pc-beta*center;resid=ph-(beta*xx+intercept)
 return dict(beta_rad_um=beta,residual_rms_deg=float(np.degrees(np.sqrt(np.sum(w*resid**2)/w.sum()))),n=int(good.sum()),phase_span_rad=float(np.ptp(ph)),window_um=window)
def beta_one(path):
 a=load(path/'sample.npz');result=read(path/'result.json');j=result['job'];ip=int(a['input_port']);sig=1 if ip==1 else -1;pre=f'port{ip}/';cy=float(a['source_center_um'][1]);yp=a[pre+'y_um'];ym=(a['y_um']>=min(yp))&(a['y_um']<=max(yp));y=a['y_um'][ym]
 em=np.interp(y,yp,a[pre+'mode_Ez'].real)+1j*np.interp(y,yp,a[pre+'mode_Ez'].imag);hm=np.interp(y,yp,a[pre+'mode_Hy'].real)+1j*np.interp(y,yp,a[pre+'mode_Hy'].imag);ww=np.full(len(y),float(a['y_um'][1]-a['y_um'][0]));ww[[0,-1]]*=.5;power=-np.real(np.sum(ww*em.conj()*hm));eh=-np.sum((ww*em.conj())[:,None]*a['Hy'][ym],axis=0);he=-np.sum((ww*hm.conj())[:,None]*a['Ez'][ym],axis=0);aa=(eh+he)/(2*np.sqrt(power)) if ip==1 else (he-eh)/(2*np.sqrt(power));back=(he-eh)/(2*np.sqrt(power)) if ip==1 else (eh+he)/(2*np.sqrt(power));x=a['x_um'];center=interpolate(a,np.column_stack([x,np.full(len(x),cy)]));windows=[[-9,9],[-7,7],[-9,-1],[1,9]];slopes=[slope(x,aa,sig,w) for w in windows];cl=slope(x,center,sig,windows[0]);mm=result['monitors'][ip-1];beta_mpb=2*np.pi*abs(mm['independent_mode_k'][0]);beta_high=2*np.pi*abs(mm['high_resolution_mode_k'][0]);w=j.get('width',next((r['params']['wg_width_um'] for r in RECORDS if r['geometry_id']==j.get('geometry_id')),0));nc=float(np.sqrt(a['eps'].max()));ncl=1.444;k0=2*np.pi/1.55;V=k0*w/2*np.sqrt(nc**2-ncl**2);u=brentq(lambda u:u*np.tan(u)-np.sqrt(max(0,V*V-u*u)),1e-12,min(V,np.pi/2)-1e-10);continuum=float(np.sqrt((nc*k0)**2-(2*u/w)**2));mask=(abs(x)<9)
 return dict(job_id=j['id'],kind=j['kind'],geometry_id=j.get('geometry_id','centered'),width_um=w,input_port=ip,input_y_um=cy,resolution=j['resolution'],Courant=j.get('Courant',.5),beta_FDTD=slopes[0]['beta_rad_um'],beta_centerline=cl['beta_rad_um'],beta_MPB=beta_mpb,beta_MPB_high=beta_high,beta_analytic_slab=continuum,modal_slope_residual_rms_deg=slopes[0]['residual_rms_deg'],window_beta_span=float(np.ptp([v['beta_rad_um'] for v in slopes])),backward_to_forward_rms=float(np.sqrt(np.sum(abs(back[mask])**2)/np.sum(abs(aa[mask])**2))),MPB_default_minus_FDTD=beta_mpb-slopes[0]['beta_rad_um'],slopes=slopes,centerline_slope=cl,physical_t_real=float(physical_t(a,straight=True).real),physical_t_imag=float(physical_t(a,straight=True).imag),native_t_real=float(a['S_native'][2-ip].real),native_t_imag=float(a['S_native'][2-ip].imag))
def beta():
 rows=[]
 for p in sorted((R/'runs').glob('*')):
  if (p/'result.json').exists() and read(p/'result.json')['job']['kind'] in ['straight','centered']:rows.append(beta_one(p))
 save(R/'beta.json',rows);csvsave(R/'beta.csv',[{k:v for k,v in r.items() if k not in ['slopes','centerline_slope']} for r in rows]);print('BETA_ANALYZED',len(rows),flush=True)

def phase_compare(r,lo,hi,beta_lo=None,beta_hi=None):
 al=sbend(r['geometry_id'],lo);ah=sbend(r['geometry_id'],hi);a0=load(prior(r['geometry_id'],20));poly,bend,masks=A.geometry(r,a0);X,Y=np.meshgrid(a0['x_um'],a0['y_um']);pts=np.stack([X,Y],axis=-1);el=interpolate(al,pts)/incident(al);eh=interpolate(ah,pts)/incident(ah);mask=masks['bend_core']&(abs(eh)>=.05*abs(eh[masks['bend_core']]).max());wt=abs(eh[mask])**2;phase=np.angle(eh[mask]*el[mask].conj());proj=A.G['project_points'](pts[mask],poly);ip=r['input_port'];sig=1 if ip==1 else -1;source=[-11.5 if ip==1 else 11.5,float(al['source_center_um'][1])];ss=A.G['project_points'](np.asarray(source)[None],poly)['s'][0];L=sig*(proj['s']-ss);raw=float(np.degrees(np.sum(wt*abs(phase))/wt.sum()));out=dict(geometry_id=r['geometry_id'],width_um=r['params']['wg_width_um'],angle_stratum=r['angle_stratum'],input_port=ip,low_resolution=lo,high_resolution=hi,bend_phase_change_wmae_deg=raw,bend_signed_circular_mean_deg=float(np.degrees(np.angle(np.sum(wt*np.exp(1j*phase))))),mean_arc_length_from_input_um=float(np.sum(wt*L)/wt.sum()))
 if beta_lo is not None:
  pred=(beta_hi-beta_lo)*L;residual=np.angle(np.exp(1j*(phase-pred)));res=float(np.degrees(np.sum(wt*abs(residual))/wt.sum()));out.update(delta_beta_rad_um=beta_hi-beta_lo,predicted_propagation_wmae_deg=float(np.degrees(np.sum(wt*abs(np.angle(np.exp(1j*pred))))/wt.sum())),residual_after_delta_beta_wmae_deg=res,descriptive_relative_WMAE_reduction=float(1-res/raw) if raw>0 else None)
 plus,minus,ang=A.paircoords(r,a0,poly,bend,.5);lp=interpolate(al,plus);lm=interpolate(al,minus);hp=interpolate(ah,plus);hm=interpolate(ah,minus);ct=hp*hm.conj();cl=lp*lm.conj();good=(abs(hp)>=.05*abs(interpolate(ah,pts)[masks['bend_core']]).max())&(abs(hm)>=.05*abs(interpolate(ah,pts)[masks['bend_core']]).max());w=abs(ct[good]);out['normal_phase_change_wmae_deg']=float(np.degrees(np.sum(w*abs(np.angle(ct[good]*cl[good].conj())))/w.sum()))
 if beta_lo is not None:
  sp=A.G['project_points'](plus[good],poly)['s'];sm=A.G['project_points'](minus[good],poly)['s'];pred=(beta_hi-beta_lo)*sig*(sp-sm);res=np.angle(np.exp(1j*(np.angle(ct[good]*cl[good].conj())-pred)));out['normal_residual_after_delta_beta_wmae_deg']=float(np.degrees(np.sum(w*abs(res))/w.sum()))
 return out

def decompose(res80=False):
 bet=read(R/'beta.json');lookup={r['job_id']:r for r in bet};rows=[];ports=[]
 for r in RECORDS:
  gid=r['geometry_id'];resolutions=[20,30,40,60]
  if res80 and gid in CFG['conditional80_ids'] and (R/'runs'/f'res80_{gid}/sample.npz').exists():resolutions.append(80)
  for lo,hi in zip(resolutions[:-1],resolutions[1:]):
   bl=lookup[f'straight_{gid}_r{lo}']['beta_FDTD'];bh=lookup.get(f'straight_{gid}_r{hi}')
   row=phase_compare(r,lo,hi,bl,bh['beta_FDTD']) if bh else phase_compare(r,lo,hi);rows.append(row)
  for res in resolutions:
   a=sbend(gid,res);pp=R/'runs'/f'straight_{gid}_r{res}/sample.npz'
   if not pp.exists():continue
   st=load(pp);ts=physical_t(a,r);tr=physical_t(st,straight=True);tn=a['S_native'][2-r['input_port']];rn=st['S_native'][2-r['input_port']];poly,bend=A.G['centerline_from_params'](r['params']);p1=np.array([[-11.5,poly[0,1]],[11.5,poly[-1,1]]]);sd=A.G['project_points'](p1,poly)['s'];length=float(sd[1]-sd[0]);b=lookup[f'straight_{gid}_r{res}']['beta_FDTD'];ports.append(dict(geometry_id=gid,resolution=res,physical_t_phase_deg=float(np.degrees(np.angle(ts))),native_t_phase_deg=float(np.degrees(np.angle(tn))),straight_physical_phase_deg=float(np.degrees(np.angle(tr))),extra_physical_phase_deg=float(np.degrees(np.angle(ts/tr))),extra_native_phase_deg=float(np.degrees(np.angle(tn/rn))),native_vs_physical_extra_phase_deg=float(np.degrees(np.angle((tn/rn)/(ts/tr)))),port_arc_length_um=length,delta_length_um=length-23,beta_deltaL_predicted_deg=float(np.angle(np.exp(1j*b*(length-23))))*180/np.pi,physical_extra_real=float((ts/tr).real),physical_extra_imag=float((ts/tr).imag)))
 save(R/'decomposition.json',rows);csvsave(R/'decomposition.csv',rows);save(R/'port_reference.json',ports);csvsave(R/'port_reference.csv',ports);print('DECOMPOSITION',len(rows),flush=True)

def materials():
 rows=[];order=[]
 for r in RECORDS:
  gid=r['geometry_id'];orig={}
  for res in [20,30,40,60]:
   p=R/'materials'/f'original_{gid}_r{res}/samples.npz';a=load(p);orig[res]=a;fraction=(a['epszz']-a['eps_clad'])/(a['eps_core']-a['eps_clad']);widths=np.trapezoid(fraction,a['eta']*r['params']['wg_width_um']/2,axis=1);exact=np.trapezoid(a['exact_union_inside'].astype(float),a['eta']*r['params']['wg_width_um']/2,axis=1);rows.append(dict(geometry_id=gid,width_um=r['params']['wg_width_um'],resolution=res,effective_normal_width_mean_um=float(widths.mean()),effective_width_span_um=float(np.ptp(widths)),sampled_union_width_mean_um=float(exact.mean()),epsilonzz_vs_trace_max_abs=float(np.max(abs(a['epszz']-a['epsilon_trace_average']))),intermediate_epsilon_fraction=float(np.mean((fraction>1e-5)&(fraction<1-1e-5)))))
  for lo,hi in [(20,30),(30,40),(40,60)]:
   a,b=orig[lo],orig[hi];diff=(b['epszz']-a['epszz'])/(a['eps_core']-a['eps_clad']);rows.append(dict(geometry_id=gid,width_um=r['params']['wg_width_um'],resolution=f'{lo}->{hi}',epsilonzz_difference_normalized_RMS=float(np.sqrt(np.mean(diff**2))),epsilonzz_difference_normalized_max=float(abs(diff).max())))
  if gid in CFG['control_ids']:
   for res in [20,40]:
    a=orig[res]
    for typ in ['noavg','reordered']:
     b=load(R/'materials'/f'{typ}_{gid}_r{res}/samples.npz');assert np.array_equal(a['exact_union_inside'],b['exact_union_inside']);diff=(b['epszz']-a['epszz'])/(a['eps_core']-a['eps_clad']);order.append(dict(geometry_id=gid,resolution=res,variant=typ,same_continuous_union=True,epsilonzz_difference_normalized_RMS=float(np.sqrt(np.mean(diff**2))),epsilonzz_difference_normalized_max=float(abs(diff).max()),changed_points=int(np.count_nonzero(abs(diff)>1e-10))))
 save(R/'material_summary.json',rows);csvsave(R/'material_summary.csv',rows);save(R/'material_controls.json',order);csvsave(R/'material_controls.csv',order);print('MATERIALS_ANALYZED',flush=True)

def compare_arrays(r,a,b):
 a0=load(prior(r['geometry_id'],20));poly,bend,masks=A.geometry(r,a0);X,Y=np.meshgrid(a0['x_um'],a0['y_um']);pts=np.stack([X,Y],axis=-1);ea=interpolate(a,pts)/incident(a);eb=interpolate(b,pts)/incident(b);ma=float(abs(eb[masks['bend_core']]).max());v=A.fieldmetric(ea,eb,masks['bend_core'],.05,ma);pp,pm,ang=A.paircoords(r,a0,poly,bend,.5);cp=interpolate(a,pp)*interpolate(a,pm).conj();ct=interpolate(b,pp)*interpolate(b,pm).conj();w=abs(ct);normal=float(np.degrees(np.sum(w*abs(np.angle(cp*ct.conj())))/w.sum()));tp=physical_t(a,r);tq=physical_t(b,r)
 return dict(bend_phase_wmae_deg=v['phase_wmae_deg'],bend_complex_relL2=v['complex_rel_l2'],normal_phase_wmae_deg=normal,physical_t_phase_difference_deg=float(np.degrees(np.angle(tq/tp))),physical_t_relative_amplitude_difference=float(abs(abs(tq)/abs(tp)-1)))
def sensitivity():
 while not (R/'strict_validated_complete.json').exists():
  assert remaining()>120,'hard deadline waiting for strict controls'
  if (R/'strict_validation_stopped.json').exists():raise RuntimeError('strict validation failed')
  time.sleep(15)
 rows=[]
 for gid in CFG['control_ids']:
  r=next(r for r in RECORDS if r['geometry_id']==gid);base=load(R/'runs'/f'control_{gid}_r40/sample.npz');old=sbend(gid,40);rows.append(dict(geometry_id=gid,variant='cold_baseline_replay',**compare_arrays(r,old,base)))
  for pre,suffix in [('control','_plus100'),('control','_strict'),('pml2',''),('noavg',''),('reordered','')]:
   key=f'strictvalidated_{gid}_r40' if suffix=='_strict' else f'{pre}_{gid}_r40{suffix}';a=load(R/'runs'/key/'sample.npz');result=read(R/'runs'/key/'result.json');rows.append(dict(geometry_id=gid,variant=pre+suffix,simulation_time=result['simulation_time'],source_coordinate_difference_um=float(abs(a['source_center_um']-base['source_center_um']).max()),**compare_arrays(r,base,a)))
 save(R/'sensitivity.json',rows);csvsave(R/'sensitivity.csv',rows);print('SENSITIVITY_ANALYZED',flush=True)

def verify():
 for p,h in read(R/'protected_inputs.json').items():assert sha(p)==h,p
 f=read(R/'freeze.json')
 for p,h in f['hashes'].items():
  if Path(p).name=='experiment.py':assert sha(R/'experiment_initial_frozen.py')==h and sha(R/'experiment.py')==read(R/'IMPLEMENTATION_REPAIR.json')['corrected_sha256']
  else:assert sha(p)==h,p
 assert sha(R/'jobs.json')==read(R/'jobs_freeze.json')['jobs_sha256'];save(R/'verification.json',dict(passed=True,protected_files=len(read(R/'protected_inputs.json')),history_unchanged=True,frozen_jobs_unchanged=True,documented_geometry_import_repair=True,no_network=True,no_phase_correction_to_predictions=True))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='beta':beta()
 elif action=='all':beta();decompose();materials();sensitivity();verify()
 elif action=='final':beta();decompose(True);verify()





