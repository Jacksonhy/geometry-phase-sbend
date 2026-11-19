"""Saved-data integration and unbiased gallery selection; no model execution."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import csv,json,sys,importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from predict import R,BASE,METHODS,SEEDS,read,save,digest,records
B=BASE/'research_sbend_blind_local_phase_2026-11-15'
ENDPOINTS=[('whole_rel_l2','field','whole',0.,'complex_rel_l2'),('bend_rel_l2','field','bend_core',0.,'complex_rel_l2'),('bend_phase_deg','field','bend_core',0.,'phase_wmae_deg'),('normal_phase_deg','pair','normal_pair_all',.5,'normal_relative_phase_wmae_deg'),('cross_rel_l2','pair','normal_pair_all',.5,'coherence_complex_rel_l2'),('bend_amplitude_rel_l2','field','bend_core',0.,'amplitude_rel_l2')]
LABELS={'linear':'Linear','arc_only':'ArcOnly','local_tangent':'LocalTangent','path_v2':'Path v2'}
def csvsave(name,rows):
 with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def tables():
 rows=csvread(B/'blind_per_geometry.csv');rr=records(R/'blind_geometries.json');ids=[r['geometry_id'] for r in rr];lookup={};main=[];quant=[];paired=[];values={};retained=[]
 for name,kind,region,eta,metric in ENDPOINTS:
  for seed in SEEDS:
   for method in METHODS:
    a=[r for r in rows if int(r['seed'])==seed and r['method']==method and r['kind']==kind and r['region']==region and float(r['gate'])==.05 and float(r['eta'])==eta and r['convention']=='raw' and int(r['truth_resolution'])==20]
    d={r['geometry_id']:float(r[metric]) for r in a};assert set(d)==set(ids),(name,seed,method,len(d));v=np.array([d[g] for g in ids]);values[name,seed,method]=v
    main.append(dict(endpoint=name,seed=seed,method=method,n=72,mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(v.max())))
    for g,x in zip(ids,v):retained.append(dict(endpoint=name,seed=seed,method=method,geometry_id=g,value=float(x)))
   for baseline in ['linear','arc_only','path_v2']:
    a=values[name,seed,'local_tangent'];b=values[name,seed,baseline];delta=a-b;delta=delta[np.argsort(ids)];idx=np.random.default_rng(20261115).integers(0,72,(10000,72));ci=np.quantile(delta[idx].mean(axis=1),[.025,.975]);paired.append(dict(endpoint=name,seed=seed,baseline=baseline,n=72,baseline_mean=float(b.mean()),LT_mean=float(a.mean()),delta=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),improved=int((delta<0).sum())))
  for method in METHODS:
   v=np.array([values[name,s,method].mean() for s in SEEDS]);quant.append(dict(endpoint=name,method=method,n_geometries=72,n_seeds=3,mean_of_seed_means=float(v.mean()),sample_sd_of_seed_means=float(v.std(ddof=1))))
 csvsave('main_table_per_seed.csv',main);csvsave('main_table_three_seed_summary.csv',quant);csvsave('paired_differences.csv',paired);csvsave('endpoints_per_geometry.csv',retained)
 # Check imported numbers independently against existing published aggregate table.
 previous=csvread(B/'blind_summary.csv');diffs=[]
 for q in main:
  spec=next(t for t in ENDPOINTS if t[0]==q['endpoint']);_,kind,region,eta,metric=spec
  if metric=='amplitude_rel_l2':continue
  h=[p for p in previous if int(p['seed'])==q['seed'] and p['method']==q['method'] and p['kind']==kind and p['region']==region and float(p['gate'])==.05 and float(p['eta'])==eta and p['group']=='all' and p['convention']=='raw' and int(p['truth_resolution'])==20 and p['metric']==metric];assert len(h)==1
  diff=abs(q['mean']-float(h[0]['mean']));assert diff<1e-12;diffs.append(diff)
 save(R/'table_reconstruction_verification.json',dict(passed=True,aggregate_checks=len(diffs),max_mean_difference=max(diffs),no_new_truth_fitting=True))
 # Geometry joins and angle/width/direction descriptive strata.
 strata=[]
 for name,_,_,_,_ in ENDPOINTS:
  for seed in SEEDS:
   for method in METHODS:
    for typ,groups in [('angle',['low','middle','high']),('input',[1,2]),('width',sorted({r['params']['wg_width_um'] for r in rr}))]:
     for group in groups:
      idx=[i for i,r in enumerate(rr) if (r['angle_stratum'] if typ=='angle' else r['input_port'] if typ=='input' else r['params']['wg_width_um'])==group];v=values[name,seed,method][idx];strata.append(dict(endpoint=name,seed=seed,method=method,stratum_type=typ,stratum=str(group),n=len(idx),mean=float(v.mean())))
 csvsave('stratified_metrics.csv',strata)
 success=[]
 for seed in SEEDS:
  for method in METHODS:
   v=values['whole_rel_l2',seed,method]
   for threshold in [.10,.15,.20]:success.append(dict(seed=seed,method=method,criterion='whole_rel_l2',threshold=threshold,n=72,passes=int((v<=threshold).sum()),fraction=float((v<=threshold).mean())))
   good=(values['bend_rel_l2',seed,method]<=.15)&(values['bend_phase_deg',seed,method]<=5)
   success.append(dict(seed=seed,method=method,criterion='joint_bend_rel_l2<=.15_and_phase<=5deg',threshold=0.,n=72,passes=int(good.sum()),fraction=float(good.mean())))
 csvsave('illustrative_tolerance_rates.csv',success)
 # Fixed case selection, independent of seed superiority.
 cases=[]
 for st in ['low','middle','high']:
  pool=[r for r in rr if r['angle_stratum']==st];med=np.median([r['max_local_angle_deg'] for r in pool]);r=min(pool,key=lambda r:(abs(r['max_local_angle_deg']-med),r['geometry_id']));cases.append(dict(role=st+'_median_angle',record=r))
 worst=int(np.argmax(values['bend_rel_l2',2026,'local_tangent']));regress=int(np.argmax(values['bend_rel_l2',2026,'local_tangent']-values['bend_rel_l2',2026,'arc_only']))
 for role,i in [('worst_LT_bend_error',worst),('largest_LT_minus_ArcOnly_regression',regress)]:cases.append(dict(role=role,record=rr[i]))
 save(R/'gallery_selection.json',dict(seed=2026,predeclared_rule=True,cases=cases,repeated_geometries_retained=True,uses_error_for_two_explicit_failure_cases=True))
 save(R/'integrated_tables_complete.json',dict(passed=True,rows=len(retained),geometries=72,models=12))
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'pdf.fonttype':42})
 fig,axs=plt.subplots(2,3,figsize=(12,6.4),layout='constrained');colors=['#818181','#d28e23','#137d99','#7853a0'];titles=['Whole complex RelL2','Bend-core complex RelL2','Bend-core phase WMAE (deg)','Normal-pair phase WMAE (deg)','Transverse cross-term RelL2','Bend-core amplitude RelL2']
 for ax,spec,title in zip(axs.flat,ENDPOINTS,titles):
  name=spec[0]
  for j,m in enumerate(METHODS):
   v=[values[name,s,m].mean() for s in SEEDS];ax.scatter(j+np.array([-.10,0,.10]),v,c=colors[j],s=25,zorder=3);ax.plot([j-.17,j+.17],[np.mean(v)]*2,c=colors[j],lw=3)
  ax.set_xticks(range(4),['Linear','ArcOnly','Local\nTangent','Path v2']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
 fig.suptitle('72 independent angle-stratified geometries | each dot: one fixed training seed\nAll values relative to resolution20 reference; no fitted phase alignment',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(R/f'blind_main_metrics.{ext}',dpi=180)
 plt.close(fig)
 return values

def gallery():
 cfg=read(R/'gallery_selection.json');cases=cfg['cases'];seed=cfg['seed'];rr=records(R/'blind_geometries.json');ids=[r['geometry_id'] for r in rr];P={}
 for method in METHODS:
  with np.load(B/'predictions'/f'seed{seed}_{method}.npz') as z:assert z['geometry_id'].tolist()==ids;P[method]=z['prediction']
 plt.rcParams.update({'font.size':8,'axes.titlesize':9,'pdf.fonttype':42})
 for variant in ['real_field','phase_error','complex_error']:
  fig,axs=plt.subplots(5,5,figsize=(17,8.8));fig.subplots_adjust(left=.055,right=.95,top=.85,bottom=.055,hspace=.55,wspace=.12)
  for i,case in enumerate(cases):
   rec=case['record'];gid=rec['geometry_id'];idx=ids.index(gid)
   with np.load(B/'data'/gid/'sample.npz') as z:e=z['Ez'];x=z['x_um'];y=z['y_um'];eps=z['eps']
   fields=[e]+[P[m][idx] for m in METHODS];extent=[x[0]-.025,x[-1]+.025,y[0]-.025,y[-1]+.025];level=(eps.min()+eps.max())/2
   realmax=max(float(abs(a.real).max()) for a in fields);strong=abs(e)>=.05*float(abs(e).max());peak=float(abs(e).max())
   for j,(ax,a) in enumerate(zip(axs[i],fields)):
    ax.set_facecolor('#eeeeee')
    if variant=='real_field':im=ax.imshow(a.real,origin='lower',extent=extent,cmap='RdBu_r',vmin=-realmax,vmax=realmax,interpolation='nearest')
    elif variant=='phase_error':
     v=np.degrees(np.angle(e)) if j==0 else np.degrees(np.angle(a*e.conj()));v=np.where(strong,v,np.nan);im=ax.imshow(v,origin='lower',extent=extent,cmap='twilight' if j==0 else 'RdBu_r',vmin=-180 if j==0 else -60,vmax=180 if j==0 else 60,interpolation='nearest')
    else:
     v=abs(e)/peak if j==0 else abs(a-e)/peak;im=ax.imshow(v,origin='lower',extent=extent,cmap='viridis' if j==0 else 'magma',vmin=0,vmax=1 if j==0 else .5,interpolation='nearest')
    ax.contour(x,y,eps,levels=[level],colors='black',linewidths=.25,alpha=.65);ax.set_xlim(-12,12);ax.set_ylim(-4,4);ax.set_xticks([-10,0,10]);ax.set_yticks([-3,0,3]);ax.tick_params(labelsize=6)
    if j:ax.set_yticklabels([])
    if i==0:fig.text((ax.get_position().x0+ax.get_position().x1)/2,.905,(['FDTD reference']+[LABELS[m] for m in METHODS])[j],ha='center',fontsize=10)
    if i==4:ax.set_xlabel('x (um)',fontsize=7)
    if j==0:ax.set_ylabel('y (um)',fontsize=7)
    if j==4:fig.colorbar(im,ax=ax,location='right',fraction=.035,pad=.015)
   title=case['role'].replace('_',' ')+f" | w={rec['params']['wg_width_um']:.3f} um, angle={rec['max_local_angle_deg']:.2f} deg, input={rec['input_port']} | {gid}"
   axs[i,0].text(0,1.16,title,transform=axs[i,0].transAxes,fontsize=8,ha='left',clip_on=False)
  caption={'real_field':'Raw Re(Ez), original source convention; symmetric scale shared within each row','phase_error':'Truth: phase [-180,180] deg. Predictions: phase error [-60,60] deg (clipped); gray: truth amplitude <5% of peak','complex_error':'Truth: normalized |Ez| [0,1]. Predictions: |pred-truth| / truth peak [0,0.5] (clipped)'}[variant]
  fig.suptitle('Full-field surrogate: geometry-selected representatives and explicit failure cases\n'+caption+'\nFixed seed2026; resolution20 reference; no fitted alignment',fontsize=11,y=.98)
  for ext in ['png','pdf']:fig.savefig(R/f'field_gallery_{variant}.{ext}',dpi=180)
  plt.close(fig)
 save(R/'figures_complete.json',dict(passed=True,gallery_cases=5,figure_sets=4,seed=2026))
if __name__=='__main__':tables();gallery()



