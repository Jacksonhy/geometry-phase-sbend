"""Read-only train/development analysis. No networks, FDTD or new carrier."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ['MPLBACKEND']='Agg'
import sys,json,csv,hashlib,time,platform,datetime
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from scipy.signal.windows import tukey
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
ROOT=Path('/mnt/d/game'); OUT=ROOT/'research_phase_theory'; FORM=ROOT/'research_sbend_formal'
# Load the unchanged formal common first; historical geometry modules import it.
sys.path.insert(0,str(FORM))
import common as frozen_common
import features as frozen_features
V1=frozen_common.v1
METHODS=frozen_common.METHODS

CONFIG={
 'seed':2026,'datasets':['development16','train64'],'formal_test_fields_allowed':False,
 'shape':[160,480],'dx_dy_um':.05,'spectrum':'orthonormal complex fft2, equal geometry weights',
 'fields':'original complex128 Ez / frozen common training RMS scale',
 'carriers':'unchanged formal features.geometry; exp(i beta d) rounded to complex64 as formal cache',
 'beta':'unchanged formal calibration.json; no fit','global_phase_alignment':False,
 'rectangles_y_x_counts':[[10,18],[20,35],[40,70],[60,105]],'retention_rule':'abs(my)<ny and abs(mx)<nx',
 'radial_cutoffs_rad_um':[4.,8.,12.,16.,24.,40.],
 'windows':['full','tukey','core_soft','bend_soft','halo_soft','far_soft'],
 'tukey_alpha':.2,'region_gaussian_sigma_um':.10,'halo_width_um':.5,
 'curvature_ds_um':.05,'curvature_geometry_smoothing_um':.10,
 'slope_windows_um':[.15,.30,.60],'slope_amplitude_gates':[.01,.05],
 'normal_samples_width_fraction':[-.25,0.,.25],
 'bend_interior_margin_um':.30,'bootstrap_geometry_replicates':5000,
 'modal_box_halfwidths_um':[1.5,2.,2.5],'modal_mesh_um':[.01,.005],
 'modal_kappa_per_um':[0.,.05,.10,.20,.25,-.20],
 'modal_scope':'closed Dirichlet scalar Ez constant-curvature diagnostic only, not open bent Maxwell modes',
 'counterexamples':'analytic periodic demodulation raises tail; geometry-dependent unitary can increase POD rank',
 'example_rule':'save both extremal v2-minus-linear and v2-minus-v1 full spectra, never select algorithms',
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
 return float(np.corrcoef(a[ok],b[ok])[0,1]) if ok.sum()>3 and np.std(a[ok])>1e-12 and np.std(b[ok])>1e-12 else None
def stats(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
 return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),std=float(a.std())) if len(a) else None
def dataset():
 dev=read(ROOT/'research_meep_validation/sbend_index.json')
 formal=read(FORM/'geometry_manifest.json')['records']
 records=[]
 for r in dev:
  records.append({**r,'subset':'development16','path':str(ROOT/'research_meep_validation'/r['sample'])})
 for r in formal:
  if r['split']=='train':records.append({**r,'subset':'train64','path':str(FORM/'data'/r['geometry_id']/'sample.npz')})
 assert len(records)==80 and sum(r['subset']=='train64' for r in records)==64
 return records

def snapshot(records):
 paths={Path(r['path']) for r in records}
 for r in records:
  if r['subset']=='development16':paths.add(Path(r['path']).with_name('result.json'))
 for directory in ['research_sbend_path','research_sbend_path_v2','research_sbend_path_v3','research_sbend_formal','research_mechanism']:
  paths.update(p for p in (ROOT/directory).iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.csv'])
 paths.add(ROOT/'研究交接_给下一位Agent.md')
 paths.add(ROOT/'references/NeurOLight-source/NeurOLight-main/core/models/layers/neurolight_conv2d.py')
 paths.add(frozen_common.SRC/'FDTD/sbend/sbend.py')
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def spectral(records,cal):
 rows=[];checks=[];means={};saved={};t0=time.monotonic()
 ny,nx=CONFIG['shape'];mx=np.fft.fftfreq(nx)*nx;my=np.fft.fftfreq(ny)*ny
 kx=2*np.pi*np.fft.fftfreq(nx,.05);ky=2*np.pi*np.fft.fftfreq(ny,.05)
 KX,KY=np.meshgrid(kx,ky);kr=np.hypot(KX,KY)
 radial_edges=np.linspace(0,float(kr.max())+1e-9,97)
 np.savez_compressed(OUT/'frequency_axes.npz',kx_rad_um=kx,ky_rad_um=ky,mx=mx,my=my,radial_edges_rad_um=radial_edges)
 for i,r in enumerate(records):
  gid=r['geometry_id'];sub=r['subset'];p=r['params'];ip=r['input_port']
  with np.load(r['path'],allow_pickle=False) as z:
   E=z['Ez'];eps=z['eps'];x=z['x_um'];y=z['y_um'];src=z['source_center_um']
  assert E.shape==(160,480) and E.dtype==np.complex128
  assert np.max(abs(np.diff(x)-.05))<1e-10 and np.max(abs(np.diff(y)-.05))<1e-10
  beta=float(np.polyval(cal['coefficients_rad_um'],eps.max()))
  d,cost,f=frozen_features.geometry(p,x,y,src,ip)
  carriers={m:np.exp(1j*beta*d[m]).astype(np.complex64) for m in METHODS}
  cache_error=0.
  if sub=='train64':
   with np.load(FORM/'cache'/f'{gid}.npz',allow_pickle=False) as z:
    cache_error=max(float(abs(z['carrier_'+m][0]-carriers[m]).max()) for m in METHODS)
   assert cache_error==0
  X,Y=np.meshgrid(x,y);dist=f['projection']['distance'];w=p['wg_width_um'];bend=f['bend_um']
  bcore=(dist<=w/2)&(X>=bend[0,0])&(X<=bend[-1,0])
  masks={'core_soft':dist<=w/2,'bend_soft':bcore,'halo_soft':(dist>w/2)&(dist<=w/2+.5),'far_soft':dist>w/2+.5}
  edge=tukey(ny,.2)[:,None]*tukey(nx,.2)[None,:]
  windows={'full':np.ones_like(E.real),'tukey':edge}
  windows.update({k:edge*gaussian_filter(v.astype(float),2.,mode='constant') for k,v in masks.items()})
  targets={m:(E/cal['scale'])*carriers[m].conj() for m in METHODS}
  energy=np.sum(abs(E/cal['scale'])**2)
  checks.append(dict(geometry_id=gid,subset=sub,input_port=ip,beta_rad_um=beta,carrier_cache_max_error=cache_error,
    carrier_modulus_error=max(float(abs(abs(c)-1).max()) for c in carriers.values()),
    demod_energy_relative_error=max(float(abs(np.sum(abs(a)**2)/energy-1)) for a in targets.values()),**cost))
  for wn,W in windows.items():
   mass=float(np.sum(abs(E/cal['scale']*W)**2))
   for m,A in targets.items():
    B=A*W;ft=np.fft.fft2(B,norm='ortho');power=abs(ft)**2;total=float(power.sum());P=power/total
    key=(sub,wn,m);means[key]=means.get(key,0)+P
    if wn=='full':saved[(gid,m)]=P
    gx=np.diff(B,axis=1)/.05;gy=np.diff(B,axis=0)/.05
    J=float((np.sum(abs(gx)**2)+np.sum(abs(gy)**2))/np.sum(abs(B)**2))
    seam=float((np.sum(abs(B[:,0]-B[:,-1])**2)+np.sum(abs(B[0]-B[-1])**2))/np.sum(abs(B)**2))
    for cy,cx in CONFIG['rectangles_y_x_counts']:
     inside=(abs(my[:,None])<cy-1e-6)&(abs(mx[None,:])<cx-1e-6)
     tail=float(P[~inside].sum());xtail=float(P[:,abs(mx)>=cx-1e-6].sum());ytail=float(P[abs(my)>=cy-1e-6,:].sum())
     rows.append(dict(geometry_id=gid,subset=sub,input_port=ip,width_um=w,R_min_um=p['R_min_um'],method=m,window=wn,ny_count=cy,nx_count=cx,
      tail_fraction=tail,reconstruction_rel_l2=float(np.sqrt(tail)),x_tail_fraction=xtail,y_tail_fraction=ytail,
      retained_fraction=1-tail,window_energy_fraction=mass/energy,common_scale_energy=mass,finite_difference_J_per_um2=J,periodic_seam_energy_ratio=seam,
      **{f'radial_tail_k{k:g}':float(P[kr>k].sum()) for k in CONFIG['radial_cutoffs_rad_um']}))
    # Check one actual inverse transform, not only a Parseval proxy.
    if wn=='full' and i==0:
     rect=(abs(my[:,None])<40-1e-6)&(abs(mx[None,:])<70-1e-6)
     rec=np.fft.ifft2(ft*rect,norm='ortho')
     measured=np.linalg.norm(rec-B)/np.linalg.norm(B)
     expected=np.sqrt(P[~rect].sum())
     assert abs(measured-expected)<1e-12
  if sub=='development16':
   np.savez_compressed(OUT/'dev_geometry'/f'{gid}.npz',x_um=x,y_um=y,polyline_um=f['polyline_um'],bend_um=bend,
      **{'d_'+m:d[m] for m in METHODS},beta_rad_um=beta)
  print(f'FFT {i+1}/80 {gid} {time.monotonic()-t0:.1f}s',flush=True)
 csvout('spectral_per_geometry.csv',rows);csvout('carrier_checks.csv',checks)
 mean_curves=[]
 for (sub,wn,m),P in means.items():
  P=P/(16 if sub=='development16' else 64)
  radial=np.histogram(kr,bins=radial_edges,weights=P)[0]
  np.savez_compressed(OUT/'mean_spectra'/f'{sub}__{wn}__{m}.npz',power_normalized=np.fft.fftshift(P),
    kx_rad_um=np.fft.fftshift(kx),ky_rad_um=np.fft.fftshift(ky),x_marginal=np.fft.fftshift(P.sum(axis=0)),
    y_marginal=np.fft.fftshift(P.sum(axis=1)),radial_edges_rad_um=radial_edges,radial_power=radial,radial_cdf=np.cumsum(radial))
  for j,v in enumerate(radial):mean_curves.append(dict(subset=sub,window=wn,method=m,k_lower=radial_edges[j],k_upper=radial_edges[j+1],power=v,cdf=float(radial[:j+1].sum())))
 csvout('mean_radial_spectra.csv',mean_curves)
 summary=[]
 for sub in ['development16','train64']:
  for wn in CONFIG['windows']:
   for cy,cx in CONFIG['rectangles_y_x_counts']:
    for m in METHODS:
     rr=[r for r in rows if (r['subset'],r['window'],r['ny_count'],r['nx_count'],r['method'])==(sub,wn,cy,cx,m)]
     summary.append(dict(subset=sub,window=wn,ny_count=cy,nx_count=cx,method=m,n=len(rr),
       **{k:float(np.mean([r[k] for r in rr])) for k in ['tail_fraction','reconstruction_rel_l2','x_tail_fraction','y_tail_fraction','window_energy_fraction','finite_difference_J_per_um2','periodic_seam_energy_ratio']}))
 csvout('spectral_summary.csv',summary)
 pairs=[];rng=np.random.default_rng(2026)
 for sub in ['development16','train64']:
  for wn in CONFIG['windows']:
   for cy,cx in CONFIG['rectangles_y_x_counts']:
    rr=[r for r in rows if (r['subset'],r['window'],r['ny_count'],r['nx_count'])==(sub,wn,cy,cx)]
    for base in ['direct','linear','local_tangent','path_v1']:
     a={r['geometry_id']:r for r in rr if r['method']==base};b={r['geometry_id']:r for r in rr if r['method']=='path_v2'}
     for group in ['all','input1','input2']:
      ids=sorted(k for k in a if group=='all' or group==f"input{a[k]['input_port']}")
      dif=np.array([a[k]['tail_fraction']-b[k]['tail_fraction'] for k in ids])
      boots=np.mean(dif[rng.integers(0,len(ids),(5000,len(ids)))],axis=1)
      pairs.append(dict(subset=sub,window=wn,ny_count=cy,nx_count=cx,baseline=base,target='path_v2',group=group,n=len(ids),
        mean_tail_difference=float(dif.mean()),improved=int(sum(dif>0)),worsened=int(sum(dif<0)),ci_lower=float(np.quantile(boots,.025)),ci_upper=float(np.quantile(boots,.975))))
 csvout('spectral_paired.csv',pairs)
 # Deterministic extremal examples retain both positive and negative differences.
 rr=[r for r in rows if r['subset']=='train64' and r['window']=='full' and r['ny_count']==40]
 examples=[]
 for base in ['linear','path_v1']:
  a={r['geometry_id']:r for r in rr if r['method']==base};b={r['geometry_id']:r for r in rr if r['method']=='path_v2'}
  ordered=sorted(a,key=lambda k:a[k]['tail_fraction']-b[k]['tail_fraction'])
  for label,gid in [('least_favorable',ordered[0]),('most_favorable',ordered[-1])]:
   np.savez_compressed(OUT/'examples'/f'{base}_{label}_{gid}.npz',**{m:np.fft.fftshift(saved[(gid,m)]) for m in METHODS})
   examples.append(dict(baseline=base,label=label,geometry_id=gid,baseline_tail=a[gid]['tail_fraction'],v2_tail=b[gid]['tail_fraction']))
 write('examples.json',examples)
 # Read only existing method/group summaries, never test fields or predictions.
 old=[r for r in read(FORM/'test_summary.json') if r['checkpoint']=='best' and r['group']=='all']
 rel={r['method']:r['rel_l2'] for r in old};ranks=[]
 for wn in CONFIG['windows']:
  for cy,cx in CONFIG['rectangles_y_x_counts']:
   sr={r['method']:r for r in summary if r['subset']=='train64' and r['window']==wn and r['ny_count']==cy}
   ranks.append(dict(window=wn,ny_count=cy,nx_count=cx,
      five_method_spearman=float(spearmanr([sr[m]['tail_fraction'] for m in METHODS],[rel[m] for m in METHODS]).statistic),
      geometric_three_spearman=float(spearmanr([sr[m]['tail_fraction'] for m in METHODS[2:]],[rel[m] for m in METHODS[2:]]).statistic),
      frequency_ranking=sorted(METHODS,key=lambda m:sr[m]['tail_fraction']),prediction_ranking=sorted(METHODS,key=rel.get)))
 write('ordering_vs_saved_predictions.json',ranks)
 return checks

def curvature(records,cal):
 rows=[];profiles=[];mode_checks=[]
 for r in records:
  if r['subset']!='development16':continue
  gid=r['geometry_id'];p=r['params'];ip=r['input_port'];direction=1 if ip==1 else -1
  with np.load(r['path'],allow_pickle=False) as z:
   E=z['Ez'];Hx=z['Hx'];Hy=z['Hy'];eps=z['eps'];x=z['x_um'];y=z['y_um'];src=z['source_center_um']
   S=z['S_native'];Sp=z['S_Ez_proxy'];Seh=z['S_EH_grid'];alpha=z['alpha_native']
   assert np.isfinite(Hx).all() and np.isfinite(Hy).all()
   # Check modal completeness and existing E/H overlap; no phase fitting.
   m=read(Path(r['path']).with_name('result.json'))['monitors']
   betampb=float(np.mean([2*np.pi*a['kpoints'][0][0] for a in m]))
   out=1 if ip==1 else 0;inc=0 if ip==1 else 1
   mode_checks.append(dict(geometry_id=gid,input_port=ip,beta_straight_MPB=betampb,
     through_absS=float(abs(S[out])),reflection_absS=float(abs(S[inc])),
     through_Ez_proxy_error=float(abs(Sp[out]-S[out])),reflection_Ez_proxy_error=float(abs(Sp[inc]-S[inc])),
     through_EH_error=float(abs(Seh[out]-S[out])),H_finite=True))
  with np.load(OUT/'dev_geometry'/f'{gid}.npz',allow_pickle=False) as g:
   poly=g['polyline_um'];bend=g['bend_um'];beta=float(g['beta_rad_um'])
  _,lens,arc,tangent=V1.curve_info(poly);ss=np.arange(.25,arc[-1]-.25,.05)
  center,tt=V1.points_on_arc(ss,poly);normal=np.stack([-tt[:,1],tt[:,0]],axis=-1)
  # Geometry-only curvature estimate on a fixed physical smoothing scale.
  angles=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]));mid=(arc[:-1]+arc[1:])/2
  angle=gaussian_filter1d(np.interp(ss,mid,angles),2.,mode='nearest')
  kap=np.gradient(angle,.05);kprime=np.gradient(kap,.05)
  inside=(ss>=arc[1]+.30)&(ss<=arc[-2]-.30)
  lead=((ss<arc[1]-.75)|(ss>arc[-2]+.75))&(abs(center[:,0])<10.5)
  downstream=lead&((center[:,0]>bend[-1,0]) if ip==1 else (center[:,0]<bend[0,0]))
  ns=np.array([-.25,0.,.25])*p['wg_width_um'];pts=center[:,None,:]+ns[None,:,None]*normal[:,None,:]
  field=RegularGridInterpolator((y,x),E,bounds_error=False,fill_value=np.nan)(pts[...,[1,0]])
  phase=np.unwrap(np.angle(field),axis=0);am=abs(field)
  for j,n in enumerate(ns):
   for win in CONFIG['slope_windows_um']:
    half=int(round(win/.05/2));slope=np.full(len(ss),np.nan)
    for q in range(half,len(ss)-half):
     zz=phase[q-half:q+half+1,j];v=ss[q-half:q+half+1]-ss[q];weights=am[q-half:q+half+1,j]**2
     vm=np.average(v,weights=weights);zm=np.average(zz,weights=weights)
     slope[q]=direction*np.sum(weights*(v-vm)*(zz-zm))/np.sum(weights*(v-vm)**2)
    straight=float(np.nanmedian(slope[downstream]));db=slope-straight
    for gate in CONFIG['slope_amplitude_gates']:
     ok=inside&np.isfinite(slope)&(am[:,j]>gate*am[:,j].max())
     kk=kap[ok];dd=db[ok];dc=slope[ok]-beta
     # A descriptive within-geometry fit only. Never construct a new carrier.
     X=np.stack([np.ones(len(kk)),kk**2],axis=1);coef=np.linalg.lstsq(X,dd,rcond=None)[0]
     resid=dd-X@coef;var=np.sum((dd-dd.mean())**2)
     rows.append(dict(geometry_id=gid,input_port=ip,width_um=p['wg_width_um'],R_min_um=p['R_min_um'],normal_width_fraction=float(n/p['wg_width_um']),
       window_um=win,gate=gate,count=int(ok.sum()),beta_frozen=beta,beta_straight_field=straight,beta_straight_MPB=betampb,
       bend_beta_mean=float(slope[ok].mean()),bend_delta_mean=float(dd.mean()),bend_delta_rms=float(np.sqrt(np.mean(dd**2))),
       frozen_residual_rms=float(np.sqrt(np.mean(dc**2))),max_abs_kappa=float(abs(kk).max()),
       corr_kappa=corr(kk,dd),corr_kappa_squared=corr(kk**2,dd),corr_directional_kprime=corr(direction*kprime[ok],dd),
       descriptive_c2_um=float(coef[1]),descriptive_fit_R2=float(1-np.sum(resid**2)/var) if var>0 else 0.,
       bend_net_delta_phase_rad=float(np.trapezoid(db[inside],ss[inside])),
       frozen_net_residual_phase_rad=float(np.trapezoid((slope-beta)[inside],ss[inside]))))
    if j==1 and win==.30:
     for q in range(len(ss)):
      profiles.append(dict(geometry_id=gid,input_port=ip,s_um=float(ss[q]),x_um=float(center[q,0]),y_um=float(center[q,1]),
       kappa_per_um=float(kap[q]),kappa_prime_per_um2=float(kprime[q]),amplitude=float(am[q,j]),
       field_beta_rad_um=float(slope[q]) if np.isfinite(slope[q]) else None,
       beta_frozen=beta,beta_straight_field=straight,beta_straight_MPB=betampb,bend_interior=bool(inside[q])))
  print('curvature '+gid,flush=True)
 csvout('curvature_per_geometry.csv',rows);csvout('curvature_profiles.csv',profiles);csvout('existing_modal_checks.csv',mode_checks)
 main=[r for r in rows if r['normal_width_fraction']==0 and r['window_um']==.30 and r['gate']==.01]
 summary={k:stats([r[k] for r in main if r[k] is not None]) for k in ['beta_straight_field','bend_delta_mean','bend_delta_rms','frozen_residual_rms','corr_kappa','corr_kappa_squared','corr_directional_kprime','descriptive_c2_um','descriptive_fit_R2','bend_net_delta_phase_rad','frozen_net_residual_phase_rad']}
 summary['positive_c2']=sum(r['descriptive_c2_um']>0 for r in main)
 summary['negative_c2']=sum(r['descriptive_c2_um']<0 for r in main)
 summary['geometry_count']=len(main)
 summary['not_a_mode_beta']='FDTD phase contains interference, finite-resolution effects and nonadiabatic evolution; diagnostic c2 is never a carrier.'
 # Width-conditioned between-geometry association, not pointwise pseudoreplication.
 widths=sorted(set(r['width_um'] for r in main))
 a=[];b=[]
 for w in widths:
  rr=[r for r in main if r['width_um']==w]
  aa=np.array([1/r['R_min_um']**2 for r in rr]);bb=np.array([r['bend_delta_mean'] for r in rr]);a.extend(aa-aa.mean());b.extend(bb-bb.mean())
 summary['width_centered_geometry_corr']=corr(a,b)
 write('curvature_summary.json',summary)

def theory_checks():
 rng=np.random.default_rng(2026);n=256;x=2*np.pi*np.arange(n)/n
 E=rng.normal(size=n)+1j*rng.normal(size=n);H=rng.normal(size=n)+1j*rng.normal(size=n);phi=.7*np.sin(x)
 err=abs(np.linalg.norm((H-E)*np.exp(-1j*phi))-np.linalg.norm(H-E))
 rho=2+.2*np.cos(x);rp=-.2*np.sin(x);theta=3*x+.3*np.sin(2*x);tp=3+.6*np.cos(2*x);pp=.7*np.cos(x)
 A=rho*np.exp(1j*(theta-phi));grad=(rp+1j*rho*(tp-pp))*np.exp(1j*(theta-phi))
 ident=float(abs(abs(grad)**2-(rp**2+rho**2*(tp-pp)**2)).max())
 modes=np.fft.fftfreq(n)*n;P=abs(np.fft.fft(E,norm='ortho'))**2;K=20.
 tail=float(P[abs(modes)>K].sum());bound=float(np.sum(modes**2*P)/K**2)
 counter_direct=np.ones(n,dtype=complex);counter_demod=np.exp(-1j*40*x)
 counter=lambda f:float((abs(np.fft.fft(f,norm='ortho'))**2)[abs(modes)>20].sum()/np.sum(abs(f)**2))
 # Two original identical snapshots have rank one; geometry-dependent phases make rank two.
 rank_before=int(np.linalg.matrix_rank(np.stack([np.ones(n),np.ones(n)])))
 rank_after=int(np.linalg.matrix_rank(np.stack([np.ones(n),np.exp(-1j*x)])))
 # Exact analytic 1D source-free Helmholtz product check for E=exp(ikx).
 k=5.;ph=.4*np.sin(x);ph1=.4*np.cos(x);ph2=-.4*np.sin(x)
 B=np.exp(1j*(k*x-ph));B1=1j*(k-ph1)*B;B2=(-1j*ph2-(k-ph1)**2)*B
 helm=B2+2j*ph1*B1+(1j*ph2-ph1**2+k*k)*B
 write('theory_numerical_checks.json',dict(l2_isometry_abs_error=err,gradient_identity_max_error=ident,
    finite_DFT_tail=tail,spectral_derivative_bound=bound,bound_pass=tail<=bound,
    counterexample_raw_tail=counter(counter_direct),counterexample_demod_tail=counter(counter_demod),
    snapshot_rank_before=rank_before,snapshot_rank_after=rank_after,analytic_Helmholtz_identity_max_error=float(abs(helm).max())))

def main():
 OUT.mkdir(exist_ok=True)
 for name in ['mean_spectra','dev_geometry','examples','figures']:(OUT/name).mkdir(exist_ok=True)
 if (OUT/'analysis_complete.json').exists():raise RuntimeError('Completed outputs exist; do not overwrite')
 frozen_common.verify_freeze();records=dataset();cal=read(FORM/'calibration.json')
 write('analysis_config.json',CONFIG)
 before=snapshot(records);write('input_hashes_before.json',before)
 write('diagnostic_freeze.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(__file__),
  'config_sha256':sha(OUT/'analysis_config.json'),'input_hashes_sha256':sha(OUT/'input_hashes_before.json'),
  'allowed_samples':[dict(geometry_id=r['geometry_id'],subset=r['subset'],path=r['path']) for r in records]})
 theory_checks();checks=spectral(records,cal);curvature(records,cal)
 after=snapshot(records);assert before==after;write('historical_verification.json',{'file_count':len(before),'all_input_hashes_unchanged':True,'formal_test_fields_loaded':False,'network_runs':0,'FDTD_runs':0})
 write('analysis_complete.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'samples':80,'seed':2026,'checks':len(checks)})
if __name__=='__main__':main()
