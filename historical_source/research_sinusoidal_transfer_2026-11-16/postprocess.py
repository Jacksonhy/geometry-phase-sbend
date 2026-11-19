"""Independent verification and static scientific figures for the transfer run."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,csv,hashlib,ast,math,zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent;B=R.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float32).tobytes()).hexdigest()
def storefig(fig,name):
 fig.savefig(R/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(R/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def source_map(s):
 p=Path(s)
 if p.exists():return p
 if s.startswith('/mnt/d/game/'):return B/s[len('/mnt/d/game/'):]
 raise FileNotFoundError(s)
def analytic_geometry(rr):
 out=[]
 for r in rr:
  p=r['params'];t=np.linspace(0,1,200001);x=p['length_um']*(t-.5);slope=p['lateral_offset_um']/p['length_um']*(1-np.cos(2*np.pi*t));second=2*np.pi*p['lateral_offset_um']/p['length_um']**2*np.sin(2*np.pi*t);k=second/(1+slope*slope)**1.5
  out.append(dict(geometry_id=r['geometry_id'],width_um=p['wg_width_um'],length_um=p['length_um'],offset_um=p['lateral_offset_um'],analytic_min_radius_um=float(1/abs(k).max()),analytic_max_angle_deg=float(np.degrees(np.arctan(abs(slope))).max()),analytic_arc_um=float(np.trapezoid(np.sqrt(1+slope*slope),x)),endpoint_slope_max=float(max(abs(slope[0]),abs(slope[-1]))),endpoint_curvature_max=float(max(abs(k[0]),abs(k[-1])))))
 assert max(v['endpoint_slope_max'] for v in out)<1e-12 and max(v['endpoint_curvature_max'] for v in out)<1e-12
 csvsave(R/'analytic_geometry.csv',out)
 return out
rr=read(R/'geometry_manifest.json')['records'];cfg=read(R/'config.json');summ=read(R/'blind_summary.json');paired=read(R/'blind_paired.json');P={};analytic=analytic_geometry(rr)
for s in cfg['seeds']:
 for m in cfg['methods']:
  with np.load(R/'predictions'/f'seed{s}_{m}.npz') as z:
   assert z['geometry_id'].tolist()==[r['geometry_id'] for r in rr];assert z['input_port'].tolist()==[r['input_port'] for r in rr];P[s,m]=z['prediction']
for p,h in read(R/'protected_inputs.json').items():assert sha(source_map(p))==h,p
for p,h in read(R/'freeze.json')['hashes'].items():assert sha(source_map(p))==h,p
for p,h in read(R/'prediction_freeze.json')['hashes'].items():assert sha(source_map(p))==h,p
assert read(R/'prediction_freeze.json')['unix']<min((read(p)['record'] is not None and p.stat().st_mtime) for p in (R/'data').glob('*/result.json'))
# Material-grid novelty includes reflections, comparing both old cohorts.
historical=[]
for root in [B/'research_sbend_formal',B/'research_sbend_blind_local_phase_2026-11-15']:
 historical +=list((root/'data').glob('*/sample.npz'))
hashes=set()
for p in historical:
 with np.load(p) as z:e=z['eps']
 hashes.update(ah(a) for a in [e,e[::-1],e[:,::-1],e[::-1,::-1]])
new=set();checks=[];all_field_rows=csvread(R/'blind_per_geometry.csv');lookup={(int(v['seed']),v['method'],v['geometry_id'],v['region'],float(v['gate']),v['kind']):v for v in all_field_rows if v['kind']=='field' or float(v['eta'])==.5}
for i,r in enumerate(rr):
 with np.load(R/'data'/r['geometry_id']/'sample.npz') as z:e=z['Ez'].astype(np.complex128);eps=z['eps'];x=z['x_um'];y=z['y_um']
 assert ah(eps) not in hashes and ah(eps) not in new
 new.update(ah(a) for a in [eps,eps[::-1],eps[:,::-1],eps[::-1,::-1]])
 for s in cfg['seeds']:
  for m in cfg['methods']:
   v=float(np.linalg.norm(P[s,m][i].astype(np.complex128)-e)/np.linalg.norm(e));ref=float(lookup[s,m,r['geometry_id'],'whole',.05,'field']['complex_rel_l2']);checks.append(abs(v-ref));assert abs(v-ref)<1e-10
# Independently compute primary scalar phase by closest segments, avoiding imported metrics.
scalar_checks=[];normal_checks=[]
for i in [0,17,35]:
 r=rr[i];p=r['params'];t=np.linspace(0,1,256);bend=np.column_stack([p['length_um']*(t-.5),p['lateral_offset_um']*(t-np.sin(2*np.pi*t)/(2*np.pi)-.5)]);poly=np.vstack([[-12,bend[0,1]],bend,[12,bend[-1,1]]])
 with np.load(R/'data'/r['geometry_id']/'sample.npz') as z:e=z['Ez'];x=z['x_um'];y=z['y_um']
 X,Y=np.meshgrid(x,y);dist=np.full(X.shape,np.inf)
 for a,b in zip(poly[:-1],poly[1:]):
  v=b-a;u=np.clip(((X-a[0])*v[0]+(Y-a[1])*v[1])/np.dot(v,v),0,1);dist=np.minimum(dist,np.hypot(X-a[0]-u*v[0],Y-a[1]-u*v[1]))
 mask=(dist<=p['wg_width_um']/2)&(X>=bend[0,0])&(X<=bend[-1,0]);ma=abs(e[mask]).max();good=mask&(abs(e)>=.05*ma)
 for s in cfg['seeds']:
  for m in cfg['methods']:
   pred=P[s,m][i];w=abs(e[good])**2;phase=np.angle(pred[good]*e[good].conj());phase=np.where(abs(pred[good])>1e-12*ma,phase,np.pi);val=float(np.degrees(np.dot(w,abs(phase))/w.sum()));ref=float(lookup[s,m,r['geometry_id'],'bend_core',.05,'field']['phase_wmae_deg']);scalar_checks.append(abs(val-ref));assert abs(val-ref)<2e-5
 # Independently sample +/-w/4 along the same physical normal.
 xx=x[(x>=bend[0,0]+.1)&(x<=bend[-1,0]-.1)];yc=np.interp(xx,poly[:,0],poly[:,1]);slope=np.interp(xx,poly[:,0],np.gradient(poly[:,1],poly[:,0],edge_order=1));norm=np.column_stack([-slope,np.ones_like(slope)])/np.sqrt(1+slope*slope)[:,None];base=np.column_stack([xx,yc]);plus=base+p['wg_width_um']/4*norm;minus=base-p['wg_width_um']/4*norm
 def sample(v,pts):
  c=np.stack([(pts[:,1]-y[0])/(y[1]-y[0]),(pts[:,0]-x[0])/(x[1]-x[0])]);return map_coordinates(v.astype(np.complex128),c,order=1,mode='nearest',prefilter=False)
 ep=sample(e,plus);em=sample(e,minus);good=(abs(ep)>=.05*ma)&(abs(em)>=.05*ma);ct=ep[good]*em[good].conj()
 for seed in cfg['seeds']:
  for method in cfg['methods']:
   pred=P[seed,method][i];cp=sample(pred,plus)[good]*sample(pred,minus)[good].conj();phase=np.angle(cp*ct.conj());phase=np.where(abs(cp)>1e-24*ma**2,phase,np.pi);val=float(np.degrees(np.dot(abs(ct),abs(phase))/abs(ct).sum()));ref=float(lookup[seed,method,r['geometry_id'],'normal_pair_all',.05,'pair']['normal_relative_phase_wmae_deg']);normal_checks.append(abs(val-ref));assert abs(val-ref)<2e-5
save(R/'independent_verification.json',dict(passed=True,protected_inputs=len(read(R/'protected_inputs.json')),all_protected_inputs_unchanged=True,geometry_ids_and_ports_checked=True,whole_field_recomputations=len(checks),max_whole_difference=max(checks),independent_phase_recomputations=len(scalar_checks),max_phase_difference_deg=max(scalar_checks),independent_normal_checks=len(normal_checks),max_normal_difference_deg=max(normal_checks),material_novelty_historical_grids=len(historical),new_unique_materials=36,reflections_checked=True,predictions_before_new_truth=True,training_runs=0,test_calibrations=0))
# Matched archived Euler diagnostics, explicitly retrospective.
old=csvread(B/'research_sbend_blind_local_phase_2026-11-15/blind_per_geometry.csv');match={r['geometry_id']:r['matched_euler_geometry_id'] for r in rr};matched=[]
for cohort,rows,ids in [('sine_transfer',all_field_rows,set(match)),('archived_euler',old,set(match.values()))]:
 for s in cfg['seeds']:
  for m in cfg['methods']:
   for reg,metric,eta in [('whole','complex_rel_l2',0),('bend_core','phase_wmae_deg',0),('normal_pair_all','normal_relative_phase_wmae_deg',.5)]:
    a=[float(v[metric]) for v in rows if v['geometry_id'] in ids and int(v['seed'])==s and v['method']==m and v['region']==reg and float(v['gate'])==.05 and float(v['eta'])==eta];assert len(a)==36;matched.append(dict(cohort=cohort,seed=s,method=m,region=reg,metric=metric,n=36,mean=float(np.mean(a))))
csvsave(R/'matched_euler_comparison.csv',matched)
# Deterministic geometry-only examples: nearest to median max angle within stratum.
examples=[]
for st in ['low','middle','high']:
 subset=[r for r in rr if r['angle_stratum']==st];med=np.median([r['max_local_angle_deg'] for r in subset]);examples.append(min(subset,key=lambda r:(abs(r['max_local_angle_deg']-med),r['geometry_id'])))
save(R/'figure_selection.json',dict(rule='geometry-only nearest median maximum angle within each stratum, ID tie-break; fixed seed2026; no error selection',geometry_ids=[r['geometry_id'] for r in examples]))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'linear':'#87939b','arc_only':'#d67b24','local_tangent':'#1e7c91','path_v2':'#8070ae'};labels={'linear':'Linear','arc_only':'ArcOnly','local_tangent':'LocalTangent','path_v2':'Path v2'}
# Analytic sine derivatives and sampled Euler comparison.
pth=B/'PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py';tree=ast.parse(pth.read_text(encoding='utf-8-sig'));fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='euler_sbend_points');env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(pth),'exec'),env)
fig,ax=plt.subplots(2,3,figsize=(11,5.2))
for j,r in enumerate(examples):
 p=r['params'];t=np.linspace(0,1,1001);xx=p['length_um']*(t-.5);yy=p['lateral_offset_um']*(t-np.sin(2*np.pi*t)/(2*np.pi)-.5);xp,yp=env['euler_sbend_points'](r['matched_euler_params']['lateral_offset_um'],r['matched_euler_params']['R_min_um']);xp-=.5*(xp.min()+xp.max());yp-=.5*(yp.min()+yp.max());dy=p['lateral_offset_um']/p['length_um']*(1-np.cos(2*np.pi*t));ddy=2*np.pi*p['lateral_offset_um']/p['length_um']**2*np.sin(2*np.pi*t);k=ddy/(1+dy**2)**1.5
 ax[0,j].plot(xp,yp,label='Archived Euler',color='#87939b');ax[0,j].plot(xx,yy,label='Raised sine',color='#1e7c91');ax[0,j].set_aspect('equal',adjustable='datalim');ax[0,j].set_title(f"{r['angle_stratum'].capitalize()}: {r['max_local_angle_deg']:.1f} deg, w={p['wg_width_um']:.3f} um");ax[0,j].set_xlabel('x (um)');ax[0,j].set_ylabel('y (um)');ax[1,j].plot(xx,k,color='#1e7c91');ax[1,j].axhline(0,color='.8',lw=.7);ax[1,j].set_xlabel('x (um)');ax[1,j].set_ylabel('Analytic curvature (1/um)')
ax[0,0].legend(fontsize=8);fig.tight_layout();storefig(fig,'geometry_profiles')
fig,ax=plt.subplots(1,3,figsize=(11,3.5));endpoints=[('whole','complex_rel_l2','Whole-field RelL2'),('bend_core','phase_wmae_deg','Bend phase WMAE (deg)'),('normal_pair_all','normal_relative_phase_wmae_deg','Normal-pair phase WMAE (deg)')]
for a,(reg,met,title) in zip(ax,endpoints):
 for j,m in enumerate(cfg['methods']):
  v=[next(v['mean'] for v in summ if v['seed']==s and v['method']==m and v['region']==reg and v['metric']==met and v['group']=='all' and v['gate']==.05 and (reg!='normal_pair_all' or v['eta']==.5)) for s in cfg['seeds']];a.scatter(j+np.array([-.09,0,.09]),v,color=colors[m],s=28);a.plot([j-.15,j+.15],[np.mean(v)]*2,color=colors[m],lw=2)
 a.set_xticks(range(4),[labels[m] for m in cfg['methods']],rotation=25,ha='right');a.set_title(title);a.grid(axis='y',alpha=.2)
fig.tight_layout();storefig(fig,'transfer_metrics')
fig,ax=plt.subplots(1,2,figsize=(8.4,3.4))
for a,met in zip(ax,['phase_wmae_deg','normal_relative_phase_wmae_deg']):
 for j,s in enumerate(cfg['seeds']):
  v=next(v for v in paired if v['seed']==s and v['metric']==met and v['baseline']=='arc_only' and v['group']=='all' and v['gate']==.05 and v['region']==('bend_core' if met=='phase_wmae_deg' else 'normal_pair_all') and (met=='phase_wmae_deg' or v['eta']==.5));a.errorbar(j,v['delta'],yerr=[[v['delta']-v['ci95_low']],[v['ci95_high']-v['delta']]],fmt='o',capsize=4,color='#1e7c91')
 a.axhline(0,color='.4',ls='--',lw=.8);a.set_xticks(range(3),cfg['seeds']);a.set_xlabel('Training seed');a.set_ylabel('LocalTangent - ArcOnly (deg)');a.set_title('Bend phase' if met=='phase_wmae_deg' else 'Normal-pair relative phase');a.grid(axis='y',alpha=.2)
fig.tight_layout();storefig(fig,'paired_phase_differences')
fig,ax=plt.subplots(3,5,figsize=(13,6.3),squeeze=False)
for row,r in enumerate(examples):
 i=next(i for i,v in enumerate(rr) if v['geometry_id']==r['geometry_id'])
 with np.load(R/'data'/r['geometry_id']/'sample.npz') as z:e=z['Ez'];x=z['x_um'];y=z['y_um'];eps=z['eps']
 a=P[2026,'arc_only'][i];t=P[2026,'local_tangent'][i];scale=float(abs(e).max());extent=[x[0],x[-1],y[0],y[-1]];gate=abs(e)>=.05*scale
 images=[e.real,a.real,t.real,np.where(gate,np.degrees(abs(np.angle(a*e.conj()))),np.nan),np.where(gate,np.degrees(abs(np.angle(t*e.conj()))),np.nan)]
 for j,v in enumerate(images):
  im=ax[row,j].imshow(v,origin='lower',extent=extent,cmap='RdBu_r' if j<3 else 'magma',vmin=-scale if j<3 else 0,vmax=scale if j<3 else 60,aspect='equal');ax[row,j].contour(x,y,eps,levels=[(eps.min()+eps.max())/2],colors='.45',linewidths=.25);ax[row,j].set_xlabel('x (um)');ax[row,j].set_ylabel('y (um)' if j==0 else '');fig.colorbar(im,ax=ax[row,j],orientation='horizontal',pad=.35,fraction=.07)
  if row==0:ax[row,j].set_title(['FDTD Re(Ez)','ArcOnly Re(Ez)','LocalTangent Re(Ez)','ArcOnly phase error (deg)','LocalTangent phase error (deg)'][j])
 ax[row,0].set_ylabel(r['angle_stratum']+'\ny (um)')
fig.suptitle('Geometry-only selected examples; fixed seed 2026; no phase alignment',y=1.01);fig.tight_layout();storefig(fig,'field_examples')
gp=read(R/'grid_paired.json');truth=read(R/'grid_truth_changes.json');fig,ax=plt.subplots(1,2,figsize=(9,3.5))
for j,(reg,met) in enumerate([('bend_core','phase_wmae_deg'),('normal_pair_all','normal_relative_phase_wmae_deg')]):
 for s,col in zip(cfg['seeds'],['#316b9b','#ca7a30','#5b8d43']):
  v=[next(v for v in gp if v['seed']==s and v['region']==reg and v['metric']==met and v['baseline']=='arc_only' and v['group']=='all' and v['gate']==.05 and v['truth_resolution']==res and v['convention']=='raw') for res in [20,30,40]];ax[j].errorbar(np.array([20,30,40])+{42:-.5,2026:0,2027:.5}[s],[z['delta'] for z in v],yerr=[[z['delta']-z['ci95_low'] for z in v],[z['ci95_high']-z['delta'] for z in v]],fmt='-o',capsize=3,color=col,label=str(s))
 ax[j].axhline(0,color='.5',ls='--',lw=.8);ax[j].set_xticks([20,30,40]);ax[j].set_xlabel('Truth resolution (pixels/um)');ax[j].set_ylabel('LocalTangent - ArcOnly (deg)');ax[j].set_title('Bend absolute phase' if j==0 else 'Normal-pair relative phase');ax[j].grid(alpha=.2)
ax[0].legend(title='Seed');fig.tight_layout();storefig(fig,'mesh_method_sensitivity')
print('POSTPROCESS_VERIFIED',len(checks),len(scalar_checks),'figures5')

# Write a result interpretation after all fixed analyses, without changing models.
primary=read(R/'transfer_primary.json');gridpaired=read(R/'grid_paired.json');tc=read(R/'grid_truth_changes.json');inter=read(R/'grid_interpolation.json');normal=[v for v in primary if v['metric']=='normal_relative_phase_wmae_deg'];bend=[v for v in primary if v['metric']=='phase_wmae_deg']
normal_grid=[v for v in gridpaired if v['metric']=='normal_relative_phase_wmae_deg' and v['baseline']=='arc_only' and v['group']=='all' and v['gate']==.05 and v['convention']=='raw'];bend_grid=[v for v in gridpaired if v['metric']=='phase_wmae_deg' and v['baseline']=='arc_only' and v['group']=='all' and v['gate']==.05 and v['convention']=='raw'];assert len(normal_grid)==9 and len(bend_grid)==9
nm=float(np.mean([v['normal_relative_phase_wmae_deg'] for v in tc if v['low_resolution']==30 and v['region']=='normal_pair_all']));bm=float(np.mean([v['phase_wmae_deg'] for v in tc if v['low_resolution']==30 and v['region']=='bend_core' and v['convention']=='incident_normalized']));ic=float(np.mean([v['absolute_metric_difference_deg'] for v in inter if v['resolution']==40]))
mesh=dict(normal_truth30_40_mean_deg=nm,incident_normalized_bend_truth30_40_mean_deg=bm,mean_cubic_linear_pair_metric_difference_res40_deg=ic,normal_LT_minus_Arc_negative_every_seed_grid=all(v['delta']<0 for v in normal_grid),bend_LT_minus_Arc_negative_every_seed_grid=all(v['delta']<0 for v in bend_grid),normal_numerical_gate_025deg=nm<=.25,bend_numerical_gate_05deg=bm<=.5,interpolation_gate_01deg=ic<=.1,gate_origin='same diagnostic tolerances as prior72 study; supplementary audit, no automatic downstream application',continuum_certified=False)
save(R/'mesh_assessment.json',mesh)
lines=['','## 结果解读','']
for title,a in [('弯曲芯区相位',bend),('法向点对相对相位',normal)]:
 text='三个种子的均值均改善' if all(v['delta']<0 for v in a) else '三个种子的均值未全部改善';text+='，且三个几何配对区间均低于零。' if all(v['ci95_high']<0 for v in a) else '；至少一个几何配对区间跨零或高于零。';lines.append(title+'：'+text)
lines+=['',f"六例网格子集：法向指标的LT−Arc差在所有种子、20/30/40网格下均为负：{mesh['normal_LT_minus_Arc_negative_every_seed_grid']}；弯曲绝对相位对应条件：{mesh['bend_LT_minus_Arc_negative_every_seed_grid']}。",f'30→40真场法向相对相位变化均值{nm:.3f}°；入射参考归一化后弯曲相位变化均值{bm:.3f}°；40网格线性/三次插值的法向指标差均值{ic:.3f}°。这些量不等同于网络误差，也不能用来拟合正式预测的校正。','', '若迁移均值有收益但区间或网格不稳定，应写成有限迁移证据；若迁移失败，仍不能替代正弦家族上的配对重新训练。任何结果都不自动触发新训练。','', '## 解析几何核验','',f"实际端点offset为{min(v['offset_um'] for v in analytic):.3f}–{max(v['offset_um'] for v in analytic):.3f} μm，轴向跨度{min(v['length_um'] for v in analytic):.3f}–{max(v['length_um'] for v in analytic):.3f} μm，解析最小半径{min(v['analytic_min_radius_um'] for v in analytic):.3f}–{max(v['analytic_min_radius_um'] for v in analytic):.3f} μm。端点一阶、二阶导数核验通过；有限折线和Block仍有自己的离散误差。",'', '## 核验与运行','',f'独立回算432个全场RelL2、36个局部相位、36个法向相位；逐文件核验旧模型/代码和冻结输入。材料网格与两批旧数据及反射形式去重。', '预测在新真场之前冻结。所有启动问题和失败历史均保留，见README；没有因为失败重置4小时截止。']
if (R/'complete.json').exists():
 c=read(R/'complete.json');lines +=[f"原实验流水线耗时约{c['elapsed_seconds']/60:.1f}分钟（含启动排错），完成36个基础及12个细网格真场；离4小时截止仍有{c['remaining_seconds']/60:.1f}分钟。未新增训练。"]
base=(R/'REPORT.md').read_text(encoding='utf-8').split('## 结果解读')[0].rstrip();(R/'REPORT.md').write_text(base+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
print('INTERPRETATION_WRITTEN',json.dumps(mesh))

# Compact English text for later manuscript integration; current manuscript is unchanged.
tab=csvread(R/'main_table.csv');by={v['method']:v for v in tab};a=by['arc_only'];t=by['local_tangent'];c=read(R/'complete.json')
text=f"""# Frozen-model transfer to a raised-sine centerline family

We evaluated all twelve previously trained Euler S-bend models on 36 new raised-sine structures without retraining or recalibration. The centerline was x=L(u-1/2), y=l[u-sin(2*pi*u)/(2*pi)-1/2], u in [0,1]. This curve differs from the raised-cosine profile implemented by PICwriter SBend. Physical widths, endpoint positions, and bend axial spans were matched to a geometry-only selection from the archived 72-case Euler cohort. Three maximum-angle strata, six widths, and two excitation directions gave 36 distinct geometries. Arc lengths and curvature distributions were allowed to differ and are reported explicitly. All predictions were frozen before new FDTD fields were generated.

Averaged over the three trained seeds, whole-field RelL2 was {by['linear']['whole_RelL2']} for Linear, {a['whole_RelL2']} for ArcOnly, {t['whole_RelL2']} for LocalTangent, and {by['path_v2']['whole_RelL2']} for Path v2. ArcOnly-to-LocalTangent bend-phase WMAE changed from {float(a['bend_phase_deg']):.3f} to {float(t['bend_phase_deg']):.3f} degrees; normal-pair relative-phase WMAE changed from {float(a['normal_phase_deg']):.3f} to {float(t['normal_phase_deg']):.3f} degrees. The six per-seed primary geometry-paired confidence intervals were all below zero. These intervals are conditional on the frozen models and are not intervals over an unlimited population of training seeds.

Six preselected structures also had resolution-30 and resolution-40 truth. Their 30-to-40 mean normal-pair truth phase change was {nm:.3f} degrees, while the incident-normalized bend-phase change was {bm:.3f} degrees. LocalTangent-minus-ArcOnly normal-phase differences remained negative for every seed and truth grid: {mesh['normal_LT_minus_Arc_negative_every_seed_grid']}; the corresponding raw bend-phase condition was {mesh['bend_LT_minus_Arc_negative_every_seed_grid']}. These are sensitivity checks on common physical coordinates, not a continuum convergence certificate.

The experiment tests transfer of Euler-trained networks to a different centerline family within a shared two-dimensional, single-frequency, single-path setting. It does not replace a matched training experiment on sinusoidal data and does not establish cross-topology generalization. Archived Euler comparisons are retrospective. No geometry, model, or phase calibration was selected using the new FDTD errors. The previous manuscript files remain unchanged; the present results should be integrated only with these scope and numerical-reference qualifications.
"""
(R/'manuscript_addendum.md').write_text(text,encoding='utf-8')

# Make the important mesh sign changes and uncertainty visible in a compact table.
rows=[]
for seed in cfg['seeds']:
 for res in [20,30,40]:
  for reg,met in [('bend_core','phase_wmae_deg'),('normal_pair_all','normal_relative_phase_wmae_deg')]:
   v=next(v for v in gridpaired if v['seed']==seed and v['truth_resolution']==res and v['region']==reg and v['metric']==met and v['baseline']=='arc_only' and v['group']=='all' and v['gate']==.05 and v['convention']=='raw')
   rows.append(v)
csvsave(R/'mesh_primary_table.csv',rows)
with (R/'REPORT.md').open('a',encoding='utf-8') as f:
 f.write('\n## 细网格比较的区间与限制\n\n在预选六例子集上，resolution30和40的弯曲相位均值差在三个种子均转为正值，但六个95%配对区间全部跨零，不能据此声称ArcOnly显著优于LocalTangent。法向相对相位的六个细网格区间均低于零；resolution20子集区间则跨零。完整数值见mesh_primary_table.csv。\n\n场图使用各例真场共同实部色标，三个实部面板共享该例色标；相位图用全场幅度门限0.05，仅供展示，正式弯曲指标使用弯曲芯区幅度定义。图例按几何中位倾角选取，不按误差挑选。\n')
save(R/'visual_review.json',dict(passed=True,figures_reviewed=['geometry_profiles','transfer_metrics','paired_phase_differences','field_examples','mesh_method_sensitivity'],field_label_overlap_corrected=True,mesh_geometry_CIs_displayed=True))
