"""Fixed diagnostics aggregation, E/H checks, independent FFT verification and figures."""
from analyze_phase import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

COLORS=['#777777','#d68422','#15846b','#ab527a','#3c67bc']
LABELS=['Direct','Linear','LocalTangent','Path v1','Path v2']
def loadcsv(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def numbers(r):
 d={}
 for k,v in r.items():
  try:d[k]=float(v)
  except (ValueError,TypeError):d[k]=v
 return d
def readnums(name):return [numbers(r) for r in loadcsv(name)]
def finish(fig,name):
 fig.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def checks():
 rows=readnums('spectral_per_geometry.csv');cal=read(FORM/'calibration.json');errs=[]
 # First train geometry per width: frozen cache, no expensive geometry reconstruction.
 tr=[r for r in dataset() if r['subset']=='train64'];chosen={}
 for r in tr:chosen.setdefault(r['params']['wg_width_um'],r)
 for r in chosen.values():
  with np.load(r['path']) as z:E=z['Ez']/cal['scale']
  with np.load(FORM/'cache'/f"{r['geometry_id']}.npz") as z:C={m:z['carrier_'+m][0] for m in METHODS}
  mx=np.fft.fftfreq(480)*480;my=np.fft.fftfreq(160)*160;mask=(abs(my[:,None])<39.999999)&(abs(mx[None,:])<69.999999)
  for m in METHODS:
   A=E*C[m].conj();ft=np.fft.fft2(A,norm='ortho');recon=np.fft.ifft2(ft*mask,norm='ortho');rel=np.linalg.norm(recon-A)/np.linalg.norm(A)
   old=next(v for v in rows if v['geometry_id']==r['geometry_id'] and v['method']==m and v['window']=='full' and v['ny_count']==40)
   A32=E.astype(np.complex64)*C[m].conj();P32=abs(np.fft.fft2(A32,norm='ortho'))**2
   tail32=float(P32[~mask].sum()/P32.sum())
   errs.append(dict(geometry_id=r['geometry_id'],method=m,reconstruction_formula_error=abs(rel-old['reconstruction_rel_l2']),
      original128_vs_training64_tail_difference=abs(tail32-old['tail_fraction'])))
 assert max(r['reconstruction_formula_error'] for r in errs)<1e-12
 csvout('independent_spectrum_verification.csv',errs)
 write('spectrum_verification.json',{'sample_count':len(chosen),'method_checks':len(errs),
   'max_reconstruction_formula_error':max(r['reconstruction_formula_error'] for r in errs),
   'max_complex64_vs128_tail_difference':max(r['original128_vs_training64_tail_difference'] for r in errs)})

def eh_diagnostic():
 rows=[]
 for r in dataset():
  if r['subset']!='development16':continue
  with np.load(r['path']) as z:E=z['Ez'];Hx=z['Hx'];Hy=z['Hy'];x=z['x_um'];y=z['y_um']
  with np.load(OUT/'dev_geometry'/f"{r['geometry_id']}.npz") as z:poly=z['polyline_um']
  _,_,arc,_=V1.curve_info(poly);ss=np.arange(arc[1]+.3,arc[-2]-.3,.05);pts,T=V1.points_on_arc(ss,poly);N=np.stack([-T[:,1],T[:,0]],axis=-1)
  values=[RegularGridInterpolator((y,x),f)(pts[:,[1,0]]) for f in [E,Hx,Hy]];e,hx,hy=values
  S=np.stack([-.5*np.real(e*hy.conj()),.5*np.real(e*hx.conj())],axis=-1)
  direct=1 if r['input_port']==1 else -1;st=direct*np.sum(S*T,axis=-1);sn=np.sum(S*N,axis=-1)
  gate=abs(e)>.01*abs(e).max();angle=np.arctan2(sn,st)
  rows.append(dict(geometry_id=r['geometry_id'],input_port=r['input_port'],
   amplitude_weighted_poynting_angle_rms_deg=float(np.degrees(np.sqrt(np.average(angle[gate]**2,weights=abs(e[gate])**2)))),
   normal_flux_abs_over_forward_abs=float(np.sum(abs(sn[gate]))/np.sum(abs(st[gate]))),
   backward_flux_points=int(sum(st[gate]<0)),sampled_points=int(gate.sum()),
   scope='interpolated same-run E/H local flux; not eigenmode decomposition'))
 csvout('EH_local_flux_diagnostic.csv',rows)

def summarize_curvature():
 rows=readnums('curvature_per_geometry.csv');main=[r for r in rows if r['normal_width_fraction']==0 and r['window_um']==.3 and r['gate']==.01]
 scales={r['geometry_id']:r for r in readnums('closed_scalar_phase_scale.csv')}
 bootstrap=[];rng=np.random.default_rng(2026)
 for metric in ['bend_delta_mean','bend_net_delta_phase_rad','corr_kappa_squared','descriptive_c2_um']:
  v=np.array([r[metric] for r in main]);b=np.mean(v[rng.integers(0,len(v),(5000,len(v)))],axis=1)
  bootstrap.append(dict(metric=metric,mean=float(v.mean()),lower=float(np.quantile(b,.025)),upper=float(np.quantile(b,.975)),unit_of_resampling='geometry'))
 csvout('curvature_geometry_bootstrap.csv',bootstrap)
 sensitivity=[]
 for win in CONFIG['slope_windows_um']:
  for gate in CONFIG['slope_amplitude_gates']:
   for n in [-.25,0.,.25]:
    rr=[r for r in rows if r['window_um']==win and r['gate']==gate and abs(r['normal_width_fraction']-n)<1e-10]
    sensitivity.append(dict(window_nominal_um=win,window_actual_span_um=2*int(round(win/.05/2))*.05,gate=gate,normal_width_fraction=n,
     mean_bend_delta_rms=float(np.mean([r['bend_delta_rms'] for r in rr])),mean_correlation_kappa2=float(np.mean([r['corr_kappa_squared'] for r in rr])),
     mean_c2=float(np.mean([r['descriptive_c2_um'] for r in rr])),negative_c2=int(sum(r['descriptive_c2_um']<0 for r in rr)),
     mean_integrated_delta_phase=float(np.mean([r['bend_net_delta_phase_rad'] for r in rr]))))
 csvout('curvature_sensitivity.csv',sensitivity)
 between=[]
 for group in ['all','input1','input2']:
  rr=[r for r in main if group=='all' or group==f"input{int(r['input_port'])}"]
  a=np.array([scales[r['geometry_id']]['integral_kappa_squared_per_um'] for r in rr]);b=np.array([r['bend_net_delta_phase_rad'] for r in rr])
  between.append(dict(group=group,n=len(rr),corr_integral_kappa2_vs_delta_phase=corr(a,b),
    mean_delta_beta=float(np.mean([r['bend_delta_mean'] for r in rr])),mean_delta_phase=float(b.mean()),
    beta_field_minus_MPB_mean=float(np.mean([r['beta_straight_field']-r['beta_straight_MPB'] for r in rr])),
    beta_field_minus_frozen_rms=float(np.sqrt(np.mean([(r['beta_straight_field']-r['beta_frozen'])**2 for r in rr])))))
 csvout('curvature_direction_summary.csv',between)

def figures():
 summary=readnums('spectral_summary.csv');rows=readnums('spectral_per_geometry.csv')
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for ax,wn in zip(axes.ravel(),['full','tukey','core_soft','far_soft']):
  for m,c,l in zip(METHODS,COLORS,LABELS):
   rr=[r for r in summary if r['subset']=='train64' and r['window']==wn and r['method']==m]
   ax.semilogy([r['nx_count'] for r in rr],[r['tail_fraction'] for r in rr],'o-',color=c,label=l)
  ax.set(title=f'Train64: {wn}',xlabel='Retained x frequency count (paired y counts: 10,20,40,60)',ylabel='Mean tail energy fraction');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();finish(fig,'cutoff_and_window_effects')
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,wn in zip(axes,['tukey','core_soft','far_soft']):
  for m,c,l in zip(METHODS,COLORS,LABELS):
   z=np.load(OUT/'mean_spectra'/f'train64__{wn}__{m}.npz');edges=z['radial_edges_rad_um']
   ax.semilogy(edges[1:],np.maximum(1-z['radial_cdf'],1e-12),color=c,label=l)
  ax.set(xlim=(0,40),ylim=(1e-8,1),title=wn,xlabel='Radial angular wavenumber [rad/um]',ylabel='Cumulative tail fraction');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.tight_layout();finish(fig,'radial_tail_curves')
 fig,axes=plt.subplots(2,5,figsize=(16,8),layout='constrained')
 for j,m in enumerate(METHODS):
  for i,wn in enumerate(['full','tukey']):
   z=np.load(OUT/'mean_spectra'/f'train64__{wn}__{m}.npz');ax=axes[i,j]
   im=ax.pcolormesh(z['kx_rad_um'],z['ky_rad_um'],np.log10(np.maximum(z['power_normalized'],1e-12)),vmin=-10,vmax=-2,cmap='magma',shading='auto')
   ax.set(xlim=(-25,25),ylim=(-40,40),title=f'{LABELS[j]} / {wn}',xlabel='kx [rad/um]')
   ax.axvline(2*np.pi*70/24,color='cyan',lw=.5);ax.axvline(-2*np.pi*70/24,color='cyan',lw=.5)
   ax.axhline(2*np.pi*40/8,color='cyan',lw=.5);ax.axhline(-2*np.pi*40/8,color='cyan',lw=.5)
   if j==0:ax.set_ylabel('ky [rad/um]')
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='log10 mean normalized power');finish(fig,'mean_2d_power')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for m,c,l in zip(METHODS,COLORS,LABELS):
  z=np.load(OUT/'mean_spectra'/f'train64__tukey__{m}.npz')
  for ax,axis in zip(axes,['x','y']):ax.semilogy(z[f'k{axis}_rad_um'],z[f'{axis}_marginal'],label=l,color=c)
 for ax,axis in zip(axes,['x','y']):ax.set(xlim=(-40,40),ylim=(1e-9,1),xlabel=f'k{axis} [rad/um]',ylabel='Mean marginal power',title=f'Train64 Tukey: {axis}');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.tight_layout();finish(fig,'directional_marginals')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,wn in zip(axes,['full','tukey']):
  vals=[[r['tail_fraction'] for r in rows if r['subset']=='train64' and r['window']==wn and r['ny_count']==40 and r['method']==m] for m in METHODS]
  ax.boxplot(vals,tick_labels=LABELS,showfliers=True);ax.set(title=f'40/70 rectangle, {wn}',ylabel='Tail energy fraction');ax.tick_params(axis='x',labelrotation=20)
 fig.tight_layout();finish(fig,'per_geometry_distributions')
 # All predeclared extremal examples, including cases where v2 loses.
 ex=read(OUT/'examples.json');fig,axes=plt.subplots(2,2,figsize=(12,8))
 axes2=np.load(OUT/'frequency_axes.npz');kx=np.fft.fftshift(axes2['kx_rad_um'])
 for ax,r in zip(axes.ravel(),ex):
  z=np.load(OUT/'examples'/f"{r['baseline']}_{r['label']}_{r['geometry_id']}.npz")
  for m,c,l in zip(METHODS,COLORS,LABELS):ax.semilogy(kx,z[m].sum(axis=0),color=c,label=l)
  ax.set(xlim=(-35,35),ylim=(1e-9,1),xlabel='kx [rad/um]',ylabel='Power',title=f"{r['label']} vs {r['baseline']}\n{r['geometry_id']}")
 axes[0,0].legend(fontsize=8);fig.tight_layout();finish(fig,'positive_and_negative_examples')
 cr=readnums('curvature_per_geometry.csv');main=[r for r in cr if r['normal_width_fraction']==0 and r['window_um']==.3 and r['gate']==.01]
 scales={r['geometry_id']:r for r in readnums('closed_scalar_phase_scale.csv')}
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for ip,marker in [(1,'o'),(2,'s')]:
  rr=[r for r in main if r['input_port']==ip];x=[scales[r['geometry_id']]['integral_kappa_squared_per_um'] for r in rr]
  axes[0].scatter(x,[r['bend_net_delta_phase_rad'] for r in rr],marker=marker,label=f'input {ip}')
  axes[1].scatter([r['width_um'] for r in rr],[r['descriptive_c2_um'] for r in rr],marker=marker,label=f'input {ip}')
  axes[2].scatter([scales[r['geometry_id']]['closed_model_phase_scale_rad'] for r in rr],[r['bend_net_delta_phase_rad'] for r in rr],marker=marker)
 axes[0].set(xlabel='Integral kappa^2 ds [1/um]',ylabel='Integrated field phase excess [rad]',title='Development16: observational association')
 axes[1].set(xlabel='Width [um]',ylabel='Descriptive fitted c2 [um]',title='Within-geometry fits (not carriers)');axes[1].axhline(0,color='k',lw=.7)
 axes[2].set(xlabel='Closed-box small-kappa scale [rad]',ylabel='FDTD phase excess [rad]',title='Different physical quantities')
 for ax in axes:ax.grid(alpha=.2)
 axes[0].legend();fig.tight_layout();finish(fig,'curvature_evidence_and_limits')
 prof=readnums('curvature_profiles.csv');first=main[0]['geometry_id'];negative=next(r['geometry_id'] for r in main if r['descriptive_c2_um']<0)
 fig,axes=plt.subplots(2,1,figsize=(11,7))
 for ax,gid in zip(axes,[first,negative]):
  rr=[r for r in prof if r['geometry_id']==gid and r['bend_interior']=='True'];x=[r['s_um'] for r in rr]
  ax.plot(x,[r['field_beta_rad_um']-r['beta_straight_field'] for r in rr],color='#555555',label='Local FDTD slope - downstream field slope')
  sc=scales[gid];ax.plot(x,[sc['small_curvature_c2_um']*r['kappa_per_um']**2 for r in rr],color='#3c67bc',label='Closed scalar small-curvature scale (not fit)')
  ax.set(title=gid,xlabel='Arc length [um]',ylabel='Phase slope excess [rad/um]');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();finish(fig,'curvature_local_profiles')

def main():
 checks();eh_diagnostic();summarize_curvature();figures()
 write('postprocess_complete.json',{'script_sha256':sha(__file__),'figures':len(list((OUT/'figures').glob('*.png'))),'networks':0,'FDTD':0})
 print('Verification, E/H diagnostics, summaries and figures complete.',flush=True)
if __name__=='__main__':main()
