"""Report and independent scalar arithmetic replay for saved-field diagnostics."""
import csv,json,hashlib,importlib.util,cmath,math,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def loadcsv(name):
 with (R/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fmt(x,n=3):return f'{float(x):.{n}f}'
# Replay three fixed geometry ranks, all seeds/methods, local phase by scalar atan2.
spec=importlib.util.spec_from_file_location('local_analysis',R/'analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
rows=loadcsv('per_geometry_fields.csv');lookup={(int(r['seed']),r['method'],r['geometry_id'],r['region'],float(r['gate'])):r for r in rows};records=[r for r in a.read(a.OLD/'geometry_manifest.json')['records'] if r['split']=='test'];maxdiff=0.;checks=0
for rec in [records[i] for i in [0,49,99]]:
 with np.load(a.OLD/'data'/rec['geometry_id']/'sample.npz',allow_pickle=False) as z:truth=z['Ez'].astype(np.complex128);data={k:z[k] for k in ['x_um','y_um']}
 poly,bend,masks=a.geometry(rec,data);mask=masks['bend_core'];ampmax=float(abs(truth[mask]).max());good=mask&(abs(truth)>=.05*ampmax);ids=np.flatnonzero(good);ev=truth.ravel()[ids]
 for seed in a.SEEDS:
  for method in a.METHODS:
   with np.load(a.predpath(seed,method),allow_pickle=False) as z:ii=z['geometry_id'].tolist().index(rec['geometry_id']);p=z['prediction'][ii].astype(np.complex128)
   pv=p.ravel()[ids];weights=[abs(z)**2 for z in ev];errs=[abs(cmath.phase(q/z)) if abs(q)>1e-12*ampmax else math.pi for q,z in zip(pv,ev)];v=sum(w*e for w,e in zip(weights,errs))/sum(weights)*180/math.pi;expected=float(lookup[seed,method,rec['geometry_id'],'bend_core',.05]['phase_wmae_deg']);maxdiff=max(maxdiff,abs(v-expected));checks+=1
assert maxdiff<1e-10
ind=dict(passed=True,geometries=3,seed_method_phase_checks=checks,independent_scalar_phase_max_difference_deg=maxdiff,source_verification='verification.json')
(R/'independent_verification.json').write_text(json.dumps(ind,indent=2)+'\n',encoding='utf-8')
comparisons=read(R/'paired_comparisons.json');sums=loadcsv('summary_per_seed.csv');decision=read(R/'decision.json');inventory=read(R/'bidirectional_inventory.json');seeds=[42,2026,2027];methods=['linear','arc_only','local_tangent','path_v2'];labels={'linear':'Linear','arc_only':'ArcOnly','local_tangent':'LocalTangent','path_v2':'Path v2'}
def summary(seed,method,region,gate=.05,eta=0.,port=0):
 return next(r for r in sums if int(r['seed'])==seed and r['method']==method and r['region']==region and float(r['gate'])==gate and float(r['eta'])==eta and int(r['input_port'])==port)
def compare(seed,region,metric,eta=0.,gate=.05,port=0,baseline='arc_only'):
 return next(r for r in comparisons if r['seed']==seed and r['baseline']==baseline and r['region']==region and r['metric']==metric and r['eta']==eta and r['gate']==gate and r['input_port']==port)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbfc','axes.facecolor':'#fafbfc','savefig.facecolor':'#fafbfc'})
fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained')
plots=[('bend_core','phase_wmae_deg',0,'Bend-core absolute phase','Intensity-weighted MAE (deg)'),('normal_pair_all','normal_relative_phase_wmae_deg',.5,'Normal-pair relative phase','Weighted MAE (deg)'),('bend_core','complex_rel_l2',0,'Bend-core complex field','Relative L2 error')]
for k,(region,metric,eta,title,ylabel) in enumerate(plots):
 xx=np.arange(3)
 for off,m,color in [(-.22,'arc_only','#d99735'),(0,'local_tangent','#277b86'),(.22,'path_v2','#8964a8')]:
  vals=[float(summary(seed,m,region,eta=eta)[metric]) for seed in seeds];ax[0,k].bar(xx+off,vals,width=.21,color=color,label=labels[m])
 ax[0,k].set(title=title,ylabel=ylabel,xticks=xx,xticklabels=[f'seed{s}' for s in seeds]);ax[0,k].grid(axis='y',alpha=.2,zorder=0)
 if k==0:ax[0,k].legend(fontsize=8)
matrices=[];counts=[]
for seed in seeds:
 mat=np.full((4,6),np.nan);cnt=np.zeros((4,6),int)
 for i in range(4):
  for j in range(6):
   rr=[r for r in comparisons if r['seed']==seed and r['baseline']=='arc_only' and r['input_port']==0 and r['gate']==.05 and r['region']==f'cell_a{i}_n{j}' and r['metric']=='phase_wmae_deg']
   if rr:mat[i,j]=rr[0]['delta_LT_minus_baseline'];cnt[i,j]=rr[0]['n']
 matrices.append(mat);counts.append(cnt)
lim=max(1.,max(float(np.nanmax(abs(m))) for m in matrices));norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
for k,(seed,mat,cnt) in enumerate(zip(seeds,matrices,counts)):
 im=ax[1,k].imshow(mat,cmap='RdBu_r',norm=norm,aspect='auto')
 for i in range(4):
  for j in range(6):
   if np.isfinite(mat[i,j]):ax[1,k].text(j,i,f'{mat[i,j]:+.1f}\nn={cnt[i,j]}',ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>.65*lim else '#172f45')
 ax[1,k].set(title=f'seed{seed}: LT - ArcOnly phase MAE',xlabel='Signed normal position eta = 2n / width',ylabel='Local tangent angle (deg)',xticks=np.arange(6),xticklabels=['[-3,-1)','[-1,-.5)','[-.5,0)','[0,.5)','[.5,1)','[1,3)'],yticks=np.arange(4),yticklabels=['0-10','10-20','20-35','35-90']);ax[1,k].tick_params(axis='x',labelrotation=35)
fig.colorbar(im,ax=list(ax[1,:]),label='Phase MAE difference (deg); negative favors LocalTangent',shrink=.8)
fig.suptitle('Local phase diagnostic | 100 saved S-bends, three independent training seeds\nPaired geometries, frozen checkpoints; normal-pair coherence is NOT two-port interference.',fontsize=15,fontweight='bold')
fig.savefig(R/'local_phase_diagnostics.png',dpi=170);fig.savefig(R/'local_phase_diagnostics.pdf');plt.close(fig)
consistent=decision['consistent_phase_mean_advantage'];verdict='在原100几何上，LocalTangent的两项主相位指标在三个种子均改善，六个几何配对条件95%区间均低于零；局部收益不是seed42独有。双端口干涉仍缺同几何双向真场，本轮不据此自动追加复杂应用仿真。' if consistent else 'LocalTangent未在两项主相位指标上建立三个种子一致优势；按用户判断条件，不支持为它继续投入更复杂应用仿真。'
lines=['# S-bend 三种子局部相位诊断','',verdict,'','本轮只读取现有100个评估几何真场及12份best复场预测，无训练、推理或FDTD。主比较是LocalTangent对ArcOnly；该固定基准已反复研究，本轮属于回顾性诊断。实验A的新144候选只保存预测传输系数，未保存完整预测场；本报告不把旧100几何的局部改善当作新144候选局部误差的直接解释。','', '## 主指标：逐种子配对','', '每项先逐几何计算，再对100几何等权平均。Δ=LocalTangent−ArcOnly，负数为改善。95%区间为固定模型条件下10000次几何配对bootstrap区间，不是训练种子置信区间。','', '| 指标 | 种子 | ArcOnly | LocalTangent | Δ | 条件95%区间 | 改善几何数/有效数 |','|---|---:|---:|---:|---:|---|---|']
metrics=[('弯曲芯区相位WMAE/度','bend_core','phase_wmae_deg',0),('法向点对相对相位WMAE/度','normal_pair_all','normal_relative_phase_wmae_deg',.5),('弯曲芯区复场RelL2','bend_core','complex_rel_l2',0),('横向相干交叉项RelL2','normal_pair_all','coherence_complex_rel_l2',.5)]
for name,region,metric,eta in metrics:
 for seed in seeds:
  r=compare(seed,region,metric,eta=eta);lines.append(f"| {name} | {seed} | {r['baseline_mean']:.5f} | {r['local_tangent_mean']:.5f} | {r['delta_LT_minus_baseline']:+.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['improved_geometries']}/{r['n']} |")
lines+=['','绝对相位误差为Arg(Epred·Etrue*)，没有逐样本相位或振幅拟合。用真场强度加权，在真幅度≥弯曲芯区最大幅度5%的像素上计算。复场RelL2使用完整区域；两个指标的像素集合不同。','', '法向点对在同一弯曲位置取±w/4两点，对比它们的相位差。公共相位偏差自然相消，没有拟合对齐。交叉项C=E_plus·E_minus*兼顾横向相对相位和振幅；它是局部横向相干性诊断，不是两个端口同时激励的器件干涉误差。','', '## 四方法背景','', '| 种子 | 方法 | 弯曲芯区相位WMAE/度 | 弯曲芯区复场RelL2 | 法向点对相对相位WMAE/度 |','|---|---|---:|---:|---:|']
for seed in seeds:
 for m in methods:
  s=summary(seed,m,'bend_core');p=summary(seed,m,'normal_pair_all',eta=.5);lines.append(f"| {seed} | {labels[m]} | {fmt(s['phase_wmae_deg'])} | {fmt(s['complex_rel_l2'],5)} | {fmt(p['normal_relative_phase_wmae_deg'])} |")
lines+=['','## 门限与方向敏感性','', '| 种子 | 真幅度门限 | 弯曲芯区相位Δ/度 | 法向点对相对相位Δ/度 |','|---|---:|---:|---:|']
for seed in seeds:
 for gate in [.01,.05,.1]:
  x=compare(seed,'bend_core','phase_wmae_deg',gate=gate);y=compare(seed,'normal_pair_all','normal_relative_phase_wmae_deg',eta=.5,gate=gate);lines.append(f"| {seed} | {gate*100:.0f}% | {x['delta_LT_minus_baseline']:+.5f} | {y['delta_LT_minus_baseline']:+.5f} |")
lines+=['','| 种子 | 激励端口 | 弯曲芯区相位Δ/度 | 法向点对相对相位Δ/度 | 几何数 |','|---|---:|---:|---:|---:|']
for seed in seeds:
 for port in [1,2]:
  x=compare(seed,'bend_core','phase_wmae_deg',port=port);y=compare(seed,'normal_pair_all','normal_relative_phase_wmae_deg',eta=.5,port=port);lines.append(f"| {seed} | {port} | {x['delta_LT_minus_baseline']:+.5f} | {y['delta_LT_minus_baseline']:+.5f} | {x['n']} |")
lines+=['','## 局部斜率与横向位置','', '斜率角分箱为0–10、10–20、20–35、35–90度；横向坐标eta=2n/w，n为最近中心线的有符号法向距离。|eta|≤1为芯区，1<|eta|≤3为近包层。图中给出每格相位Δ和有效几何数n；蓝色为LocalTangent改善，红色为退化。分箱区间为描述性诊断，无多重比较显著性主张。', '', '斜率分组显示额外收益在较大方向角处更明显：20–35度芯区的相位误差下降依种子42/2026/2027为2.132/1.402/0.952度；35–90度组为3.349/2.007/1.774度，后者仅52个有效几何。这里的角度是局部方向角，不是曲率。seed2026在0–10度组均值微退化0.029度、区间跨零；10–20度组的改善区间也跨零，不能声称各位置均改善。横向芯区四个法向位置带的汇总相位误差在三种子均下降；弱场门限1%/5%/10%及左右方向分组不改变主结论。不同角度组不完全共享几何，不能将组间差距解释成孤立因果效应。\n\n分箱完整复场误差、相位误差、振幅误差和相位项分解见summary_per_seed.csv及paired_comparisons.csv；逐几何原表见per_geometry_fields.csv、per_geometry_normal_pairs.csv。', '', '## 双端口干涉：现有证据不支持计算','', f"检查过的主要清单：{inventory['manifests_counts']}。精确几何参数相同、激励方向相反的真场配对数量为{len(inventory['same_parameter_opposite_input_pairs'])}。原评估集合左右端口各50几何，每个几何仅一个方向。100个材料网格中严格中心对称者为{inventory['formal_eval_exact_centro_symmetric']}。", '', '因此不能把单向场翻转或共轭作为另一方向真场，也不能把不同几何的场组合冒充同一器件相干干涉。两端同时激励的有效参考需要同一几何两种单端激励真场及一致源幅相约定；本轮没有这些配对证据，故该项标为未评价。用户已明确指两个端口同时相干激励，而非单端反射。', '', '## 决策与解释边界','',verdict,'', '局部相位改善、复场改善和设计成功率是不同证据。若只有复场RelL2稳定下降，不能据此宣称相位敏感应用必然更好；横向相对相位改善也不能替代双端口干涉真值验证。即便均值同向而部分条件区间跨零，也只能表述为同向趋势。', '', 'Path v2在本次局部相位指标上三种子均值略优于LocalTangent：弯曲芯区约0.026–0.045度，法向相对相位约0.007–0.060度，但部分配对区间跨零。保留这一局部细小收益，不把全场无稳定优势误写为所有局部指标均无收益。\n\n本轮不扩展复杂应用，不重训、不重新校准、不修改既有论文。旧100几何的网格收敛局限仍然存在；这里评价相对于保存数值真场的误差。', '', '## 核验','', f"115个受保护输入哈希前后不变；12个全场RelL2均重放原历史均值至四舍五入容差内。三个固定几何、全部种子与方法共{checks}次独立标量相位核验，最大差{maxdiff:.3g}度。未把像素或种子×几何作为独立训练重复。", '', '文件：PROTOCOL.md、config.json、freeze.json、sources_before.json、verification.json、independent_verification.json、bidirectional_inventory.json、decision.json及完整CSV原表。']
lines+=['', '汇总导出首次因相位/复场与点对指标列不同而失败；保留summary_per_seed_partial_failed.csv，以resume_summary.py仅修复CSV列并从原始指标表重汇总，未重算场或改变指标/统计定义。详见SUMMARY_EXPORT_REPAIR.json。', '', '横向点对以手写双线性插值和标量复相位独立核验36次，相位最大差1.07e-14度，交叉项RelL2最大差2.78e-16；见normal_pair_verification.json。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest=[]
for p in R.glob('*'):
 if p.is_file() and p.name!='OUTPUT_MANIFEST.json':manifest.append(dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('REPORT_COMPLETE',verdict)
print(json.dumps(decision,ensure_ascii=False,indent=2))


