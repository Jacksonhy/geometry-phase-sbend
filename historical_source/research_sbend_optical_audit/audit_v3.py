"""Retrospective S-bend optical audit from frozen predictions. No training/FDTD."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MPLBACKEND']='Agg'
import ast,csv,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
B=Path('/mnt/d/game');O=Path(__file__).resolve().parent;S=B/'research_sbend_formal';F=B/'research_sbend_fourmethod_5seed'
METHODS=['linear','arc_only','local_tangent','path_v2'];SEEDS=[42,2026,2027]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(n,obj):(O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvout(n,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (O/n).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def predpath(seed,m,ck):
    if seed!=2026:return F/f'seed{seed}'/'predictions'/f'{m}_{ck}.npz'
    return (F if m=='arc_only' else S)/'predictions'/f'{m}_{ck}.npz'
env=dict(np=np,eigh_tridiagonal=eigh_tridiagonal)
p=B/'research_official_audit/validate_sparams.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fd_directional_proxy')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),env);extract=env['fd_directional_proxy']
manifest=read(S/'geometry_manifest.json')['records'];records=[r for r in manifest if r['split']=='test']
files=[S/'geometry_manifest.json',p,Path(__file__),O/'PROTOCOL.md']
files += [predpath(s,m,c) for s in SEEDS for m in METHODS for c in ['best','last']]
files += [S/'data'/r['geometry_id']/'sample.npz' for r in manifest]
if not (O/'freeze_v3.json').exists():
    save('freeze_v3.json',dict(time_utc=time.time(),seeds=SEEDS,methods=METHODS,phase_native_amplitude_gate=.05,
       retrospective=True,no_refit=True,hashes={str(p):sha(p) for p in files}))
freeze=read(O/'freeze_v3.json')
for p,h in freeze['hashes'].items():assert sha(p)==h,p
truth={};native_rows=[];extractor=[];labelcheck=[];sources=[]
for r in manifest:
    with np.load(S/'data'/r['geometry_id']/'sample.npz') as z:
        gid=r['geometry_id'];ip=r['input_port'];ids=z['port_ids'];al=z['alpha_native']
        native=al[np.arange(2),[1,0]]/al[ip-1,0 if ip==1 else 1]
        assert np.max(abs(native-z['S_native']))<1e-14
        proxy=extract(z['Ez'],z['eps'],z['port_masks'],ids,ip,1.55,.05)
        labelcheck.append(float(np.max(abs(proxy-z['S_Ez_proxy']))));assert labelcheck[-1]<1e-7
        if r['split']=='test':
            truth[gid]=dict(E=z['Ez'],eps=z['eps'],pm=z['port_masks'],ids=ids,ip=ip,native=native,proxy=proxy)
            for j in range(2):
                a=proxy[j];t=native[j]
                extractor.append(dict(geometry_id=gid,input_port=ip,output_port=j+1,role='reflection' if j+1==ip else 'transmission',
                  proxy_real=float(a.real),proxy_imag=float(a.imag),native_real=float(t.real),native_imag=float(t.imag),
                  complex_error=float(abs(a-t)),power_error=float(abs(abs(a)**2-abs(t)**2)),
                  phase_error_deg=float(abs(np.angle(a*t.conjugate()))*180/np.pi) if abs(t)>=.05 else None,
                  EH_grid_complex_error=float(abs(z['S_EH_grid'][j]-t))))
        out=1 if ip==1 else 0;t=native[out];refl=native[ip-1]
        native_rows.append(dict(geometry_id=gid,split=r['split'],input_port=ip,width_um=r['params']['wg_width_um'],
             T=float(abs(t)**2),R=float(abs(refl)**2),T_plus_R=float(np.sum(abs(native)**2)),phase_deg=float(np.angle(t)*180/np.pi)))
        sources.append(dict(geometry_id=gid,source_x=float(z['source_center_um'][0]),input_port=ip,
          source_expected_x=-11.9 if ip==1 else 11.9,source_mask_weighted_x=float(np.sum(z['src_plane_interpolation_weights']*z['x_um'][None])/np.sum(z['src_plane_interpolation_weights']))))
csvout('native_labels.csv',native_rows);csvout('extractor_bias.csv',extractor);csvout('source_reference_checks.csv',sources)
rows=[];replay=0.;historical={}
for seed in SEEDS:
    root=F/f'seed{seed}' if seed!=2026 else F
    paths=[root/'port_coefficients.csv'] if seed!=2026 else [S/'port_coefficients.csv',F/'arc_only_port_coefficients.csv']
    for path in paths:
      with path.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['method'] in METHODS:
                historical[(seed,r['method'],r['checkpoint'],r['geometry_id'],int(r['output_port']))]=complex(float(r['estimate_real']),float(r['estimate_imag']))
    for method in METHODS:
      for ck in ['best','last']:
        with np.load(predpath(seed,method,ck)) as z:
            assert len(z['geometry_id'])==100
            for gid,pr,port in zip(z['geometry_id'],z['prediction'],z['input_port']):
                gid=str(gid);q=truth[gid];assert port==q['ip']
                a=extract(pr,q['eps'],q['pm'],q['ids'],q['ip'],1.55,.05)
                for j in range(2):
                    replay=max(replay,abs(a[j]-historical[(seed,method,ck,gid,j+1)]))
                    nat=q['native'][j];tp=q['proxy'][j];pred=a[j]
                    parts=dict(model=pred-tp,extractor=tp-nat,total=pred-nat)
                    assert abs(parts['total']-parts['model']-parts['extractor'])<1e-14
                    for component,error in parts.items():
                        target=tp if component=='model' else nat
                        estimate=tp if component=='extractor' else pred
                        # fixed native gate keeps phase comparisons on same geometry subset
                        gate=abs(nat)>=.05
                        rows.append(dict(seed=seed,method=method,checkpoint=ck,geometry_id=gid,input_port=int(port),output_port=j+1,
                         role='reflection' if j+1==port else 'transmission',component=component,
                         estimate_real=float(estimate.real),estimate_imag=float(estimate.imag),target_real=float(target.real),target_imag=float(target.imag),
                         complex_error=float(abs(error)),power_error=float(abs(abs(estimate)**2-abs(target)**2)),
                         signed_power_error=float(abs(estimate)**2-abs(target)**2),
                         phase_error_deg=float(abs(np.angle(estimate*target.conjugate()))*180/np.pi) if gate else None,
                         phase_gate=bool(gate),il_error_db=float(abs(-10*np.log10(max(abs(estimate)**2,1e-12))+10*np.log10(max(abs(target)**2,1e-12)))),
                         native_power=float(abs(nat)**2)))
        print('AUDIT',seed,method,ck,flush=True)
assert replay<1e-7,replay
csvout('per_geometry_optics.csv',rows)
summary=[]
for seed in SEEDS:
 for method in METHODS:
  for ck in ['best','last']:
   for role in ['transmission','reflection']:
    for component in ['model','extractor','total']:
     for port in [0,1,2]:
        rr=[r for r in rows if r['seed']==seed and r['method']==method and r['checkpoint']==ck and r['role']==role and r['component']==component and (port==0 or r['input_port']==port)]
        phase=[r['phase_error_deg'] for r in rr if r['phase_gate']]
        summary.append(dict(seed=seed,method=method,checkpoint=ck,role=role,component=component,input_port=port,n=len(rr),
          complex_mae=float(np.mean([r['complex_error'] for r in rr])),complex_rmse=float(np.sqrt(np.mean([r['complex_error']**2 for r in rr]))),
          power_mae=float(np.mean([r['power_error'] for r in rr])),power_bias=float(np.mean([r['signed_power_error'] for r in rr])),
          phase_mae_deg=float(np.mean(phase)) if phase else None,phase_count=len(phase),
          il_mae_db=float(np.mean([r['il_error_db'] for r in rr]))))
csvout('summary_per_seed.csv',summary)
grand=[]
for m in METHODS:
 for ck in ['best','last']:
  for role in ['transmission','reflection']:
   for component in ['model','extractor','total']:
    rr=[r for r in summary if r['method']==m and r['checkpoint']==ck and r['role']==role and r['component']==component and r['input_port']==0]
    gr=dict(method=m,checkpoint=ck,role=role,component=component,seeds=SEEDS,n_seeds=3,n_geometries=100)
    for key in ['complex_mae','power_mae','phase_mae_deg','il_mae_db']:
        vals=[r[key] for r in rr if r[key] is not None];gr[key]=float(np.mean(vals)) if vals else None
        gr[key+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    grand.append(gr)
save('three_seed_summary.json',grand);csvout('three_seed_summary.csv',grand)
paired=[];rng=np.random.default_rng(1411)
for seed in SEEDS:
 for base,cand in [('linear','local_tangent'),('arc_only','local_tangent'),('local_tangent','path_v2')]:
  for role in ['transmission','reflection']:
   for metric in ['complex_error','power_error','phase_error_deg']:
    aa=sorted([r for r in rows if r['seed']==seed and r['method']==base and r['checkpoint']=='best' and r['component']=='total' and r['role']==role],key=lambda r:r['geometry_id'])
    bb=sorted([r for r in rows if r['seed']==seed and r['method']==cand and r['checkpoint']=='best' and r['component']=='total' and r['role']==role],key=lambda r:r['geometry_id'])
    vv=np.array([a[metric]-b[metric] for a,b in zip(aa,bb) if a[metric] is not None and b[metric] is not None])
    if len(vv):
        ix=rng.integers(0,len(vv),(5000,len(vv)));paired.append(dict(seed=seed,baseline=base,candidate=cand,role=role,metric=metric,n=len(vv),
         baseline_minus_candidate=float(vv.mean()),ci95=np.quantile(vv[ix].mean(1),[.025,.975]).tolist()))
save('paired_geometry_statistics.json',paired)
# No new model selection; labels for train/val distributions only, no device screening on test.
dist={}
for split in ['train','val','test']:
    rr=[r for r in native_rows if r['split']==split]
    dist[split]={k:dict(min=float(min(r[k] for r in rr)),median=float(np.median([r[k] for r in rr])),max=float(max(r[k] for r in rr))) for k in ['T','R','T_plus_R']}
save('native_distribution.json',dist)
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for ax,key,title in zip(axes,['complex_mae','phase_mae_deg','power_mae'],['Complex transmission error','Transmission phase error (deg)','Transmission power MAE']):
    for s in SEEDS:
        vals=[next(r[key] for r in summary if r['seed']==s and r['method']==m and r['checkpoint']=='best' and r['role']=='transmission' and r['component']=='total' and r['input_port']==0) for m in METHODS]
        ax.plot(METHODS,vals,'o-',label=f'seed{s}')
    ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=25)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(O/'optical_summary.png',dpi=180);plt.close(fig)
lines=['# S-bend 三种子光学标签与提取器审计','','数据角色：现有100测试几何上的回顾性分析；三种子42/2026/2027复用已保存best/last预测，无训练、无FDTD、无测试场相位对齐。','',
'复数层面严格满足 total=(预测proxy−真场proxy)+(真场proxy−原生)。误差绝对值、功率误差和相位误差不能按此相加。phase gate使用固定原生系数幅度>=.05；反射通常低于此值，因此不强行报告反射相位。IL计算下限功率1e-12，线性功率MAE为优先指标。','',
'| 角色 | 方法 | 分量 | 复系数MAE | 功率MAE | 相位MAE deg |','|---|---|---|---:|---:|---:|']
for r in grand:
    if r['checkpoint']=='best':
        ph='NA' if r['phase_mae_deg'] is None else f"{r['phase_mae_deg']:.5f}"
        lines.append(f"| {r['role']} | {r['method']} | {r['component']} | {r['complex_mae']:.6f} | {r['power_mae']:.6f} | {ph} |")
lines+=['','各表为先逐种子求100几何均值再跨3种子平均；种子标准差另存CSV。几何配对CI是固定模型条件下的不确定性，不将3×100视为300独立实验。','',
'原生T/R分布：',json.dumps(dist,ensure_ascii=False,indent=2),'',
'保存标签的归一化为设备仿真中输出/输入模态振幅之比；源与端口平面独立，未去嵌入。未将1−T−R直接解释为已认证辐射损耗。原历史网格收敛不因本轮指标审计而自动得到认证。','',
'![optical summary](optical_summary.png)']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
for p,h in freeze['hashes'].items():assert sha(p)==h,p
save('verification.json',dict(passed=True,historical_coefficient_replay_max_error=float(replay),stored_proxy_export_max_error=max(labelcheck),coefficients_replayed=4800,metric_rows=len(rows),
    protected_files=len(freeze['hashes']),unchanged=True,no_training=True,no_new_fdtd=True,retrospective=True))
print('AUDIT COMPLETE',flush=True)
