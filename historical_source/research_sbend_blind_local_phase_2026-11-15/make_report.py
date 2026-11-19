"""Scientific report from frozen saved metrics; no model or solver changes."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import time,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from experiment import R,read,save,sha,SEEDS,METHODS
labels=dict(linear='Linear',arc_only='ArcOnly',local_tangent='LocalTangent',path_v2='Path v2')
cfg=read(R/'config.json');records=read(R/'geometry_manifest.json')['records'];bp=read(R/'blind_paired.json');bs=read(R/'blind_summary.json');gp=read(R/'grid_paired.json');gs=read(R/'grid_summary.json');tc=read(R/'grid_truth_changes.json');dec=read(R/'decision.json');refined=read(R/'conditional_refinement.json');reverse=read(R/'reverse_stage.json')
def summ(rows,s,m,metric,region,res=20,conv='raw',group='all',gate=.05,eta=0):
 return next(r for r in rows if r['seed']==s and r['method']==m and r['metric']==metric and r['region']==region and r['truth_resolution']==res and r['convention']==conv and r['group']==group and r['gate']==gate and r['eta']==eta)
def comp(rows,s,metric,region,res=20,conv='raw',group='all',gate=.05,eta=0,baseline='arc_only'):
 return next(r for r in rows if r['seed']==s and r['baseline']==baseline and r['metric']==metric and r['region']==region and r['truth_resolution']==res and r['convention']==conv and r['group']==group and r['gate']==gate and r['eta']==eta)
plots=[('phase_wmae_deg','bend_core',0,'Bend-core phase WMAE (deg)'),('normal_relative_phase_wmae_deg','normal_pair_all',.5,'Normal-pair relative phase WMAE (deg)'),('complex_rel_l2','bend_core',0,'Bend-core complex RelL2')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained');xx=np.arange(3)
for k,(met,region,eta,title) in enumerate(plots):
 for off,m,color in [(-.24,'arc_only','#c68b36'),(0,'local_tangent','#237e83'),(.24,'path_v2','#8763a9')]:
  axs[0,k].bar(xx+off,[summ(bs,s,m,met,region,eta=eta)['mean'] for s in SEEDS],.22,label=labels[m],color=color)
 axs[0,k].set(title=title,xticks=xx,xticklabels=[str(s) for s in SEEDS],xlabel='Training seed');axs[0,k].grid(axis='y',alpha=.2)
 if k==0:axs[0,k].legend(fontsize=8)
for k,s in enumerate(SEEDS):
 for met,region,eta,color in [('phase_wmae_deg','bend_core',0,'#237e83'),('normal_relative_phase_wmae_deg','normal_pair_all',.5,'#c68b36')]:
  vals=[];los=[];his=[]
  for res in [20,30,40]:
   v=comp(gp,s,met,region,res=res,conv='incident_normalized',eta=eta);vals.append(v['delta']);los.append(v['delta']-v['ci95_low']);his.append(v['ci95_high']-v['delta'])
  axs[1,k].errorbar([20,30,40],vals,yerr=[los,his],marker='o',label='Bend phase' if eta==0 else 'Normal-pair phase',color=color,capsize=3)
 axs[1,k].axhline(0,color='black',lw=.8);axs[1,k].set(title=f'seed{s}: sensitivity to truth grid',xlabel='Truth resolution (pixels/um)',ylabel='LT - ArcOnly (deg)',xticks=[20,30,40]);axs[1,k].grid(alpha=.2)
 if k==0:axs[1,k].legend(fontsize=8)
fig.suptitle('Prospective S-bend blind validation: 72 geometries, 3 seeds\nBottom: 18 preselected geometries, fixed incident reference; paired geometry 95% intervals',fontsize=15)
fig.savefig(R/'blind_and_grid.png',dpi=160);fig.savefig(R/'blind_and_grid.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained');mats=[]
for s in SEEDS:
 mat=np.full((4,6),np.nan)
 for i in range(4):
  for j in range(6):
   region=f'cell_a{i}_n{j}';aa=[r for r in bs if r['seed']==s and r['method']=='local_tangent' and r['metric']=='phase_wmae_deg' and r['region']==region and r['group']=='all' and r['gate']==.05];bb=[r for r in bs if r['seed']==s and r['method']=='arc_only' and r['metric']=='phase_wmae_deg' and r['region']==region and r['group']=='all' and r['gate']==.05]
   if aa and bb:mat[i,j]=aa[0]['mean']-bb[0]['mean']
 mats.append(mat)
lim=max(.5,max(float(np.nanmax(abs(v))) for v in mats))
for ax,s,mat in zip(axs,SEEDS,mats):
 im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
 for i in range(4):
  for j in range(6):
   if np.isfinite(mat[i,j]):ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=8)
 ax.set(title=f'seed{s}',xticks=range(6),xticklabels=['[-3,-1)','[-1,-.5)','[-.5,0)','[0,.5)','[.5,1)','[1,3)'],yticks=range(4),yticklabels=['0-10','10-20','20-35','35-90'],xlabel='Normal coordinate eta=2n/w',ylabel='Local tangent angle (deg)');ax.tick_params(axis='x',labelrotation=35)
fig.colorbar(im,ax=list(axs),label='LT - ArcOnly phase WMAE (deg)');fig.suptitle('72 fresh geometries | spatial phase advantage at training resolution20')
fig.savefig(R/'spatial_phase.png',dpi=160);plt.close(fig)
lines=['# 全新 S-bend 局部相位盲测与网格精度验证','']
blindpass=dec['checks']['blind_both_CIs_below_zero_all_seeds'];ready=dec['application_readiness_gate_passed']
lines+=['基础网格上的独立复现成立：两项主相位指标在三个种子均改善，六个几何配对95%区间均低于零。' if blindpass else '基础网格上未达到预先规定的三个种子一致相位优势门槛。','', '数值精度与应用准备门槛'+('通过。' if ready else '**未通过**。因此本轮停止扩展双向真场和相干应用；基础网格上的独立复现不能替代连续物理解精度认证。'),'']
lines+=['## 实验范围与预先冻结','', '72个新S-bend，按最大局部方向角分为低/中/高三档，每档24个；每档含八种宽度各3个、左右激励各12个。三个训练种子42/2026/2027，四方法best权重，零训练、零重新校准。新几何位于原参数支持内；按6000个几何提案的倾角三分位数划分，界限为'+', '.join(f'{v:.5f}°' for v in cfg['max_angle_stratum_edges_deg'])+'。这是一项按角度分层的前瞻盲测，不是旧评估分布的独立同分布抽样。','', '抽样排除355个历史唯一几何及同宽中心线RMS距离<0.035μm的近重复；基础场导出后检查材料网格、左右/上下反射及180度反演无重复。18个网格检查对象、3个条件resolution60对象及20个条件反向对象均在新仿真前锁定。精确材料网格排重并不证明任意光学响应均独立。','', '完整复场、物理坐标和12组预测均保存。原256点中心线、重叠旋转Block几何、材料、频率、源和参考面保持历史定义；同样使用DFT衰减1e-6。','', '## 新72几何的主要配对结果','', 'Δ=LocalTangent−ArcOnly，负数为改善。逐几何计算后等权平均；95%区间来自10000次几何配对bootstrap，条件于固定已训练模型，不把216条记录当作216个独立训练重复。','', '| 指标 | 种子 | ArcOnly | LocalTangent | Δ | 配对95%区间 | 改善几何数 |','|---|---:|---:|---:|---:|---|---:|']
for met,region,eta,name in [('phase_wmae_deg','bend_core',0,'弯曲芯区相位WMAE/°'),('normal_relative_phase_wmae_deg','normal_pair_all',.5,'法向点对相对相位WMAE/°'),('complex_rel_l2','bend_core',0,'弯曲芯区复场RelL2'),('coherence_complex_rel_l2','normal_pair_all',.5,'横向复交叉项RelL2')]:
 for s in SEEDS:
  v=comp(bp,s,met,region,eta=eta);lines.append(f"| {name} | {s} | {v['baseline_mean']:.5f} | {v['LT_mean']:.5f} | {v['delta']:+.5f} | [{v['ci95_low']:+.5f}, {v['ci95_high']:+.5f}] | {v['improved']}/{v['n']} |")
lines+=['', '相位WMAE以真场强度加权，真幅度门限为弯曲芯区最大幅度的5%；保留1%/10%敏感性。绝对相位为Arg(Epred·Etrue*)，没有拟合对齐。复场RelL2不采用该相位门限。法向点对位于中心线两侧±w/4，交叉项C=E+·E-*；它评价横向相干性，不是两个端口相干输入实验。','', '## 四方法与倾角分层','', '| 种子 | 方法 | 弯曲相位WMAE/° | 法向相位WMAE/° | 复场RelL2 |','|---|---|---:|---:|---:|']
for s in SEEDS:
 for m in METHODS:
  a=summ(bs,s,m,'phase_wmae_deg','bend_core')['mean'];b=summ(bs,s,m,'normal_relative_phase_wmae_deg','normal_pair_all',eta=.5)['mean'];c=summ(bs,s,m,'complex_rel_l2','bend_core')['mean'];lines.append(f'| {s} | {labels[m]} | {a:.4f} | {b:.4f} | {c:.5f} |')
lines+=['','| 种子 | 最大倾角档 | 弯曲相位Δ/° | 法向相位Δ/° |','|---|---|---:|---:|']
for s in SEEDS:
 for group in ['low','middle','high']:
  a=comp(bp,s,'phase_wmae_deg','bend_core',group=group);b=comp(bp,s,'normal_relative_phase_wmae_deg','normal_pair_all',group=group,eta=.5);lines.append(f"| {s} | {group} | {a['delta']:+.5f} | {b['delta']:+.5f} |")
lines+=['', '局部斜率与法向位置二维统计保存在blind_summary.csv及spatial_phase.png。最大倾角分层和像素局部方向角分箱是两个不同量；组间几何也不同，相关趋势不能单独证明因果。','', '主指标在两个激励方向均改善；完整配对区间见blind_paired.csv。1%/5%/10%门限下主芯区结果相同，但72个几何在10%门限下也没有像素被剔除；该项门限在主区域不构成额外弱场压力测试。法向相位收益在高最大倾角档更明显，弯曲绝对相位收益则不随最大倾角单调增大。Path v2的细小局部优势完整保留，不能把全场无稳定优势泛化为所有局部指标均无收益。','', '## 18几何网格检查','', '统一在resolution20的物理坐标评价场误差；高分辨率复场做复数双线性插值，不插值包裹相位。法向点对直接从各原始网格插值到相同物理点，避免细真场先下采样再插值。模型及预测网格始终固定，不重训。','', '原始源相位约定与固定入射参考规范化结果并列报告。后者用每个分辨率在同一输入监视面中心的入射Ez（独立E/H模态投影×单位功率模态）除真场，用resolution20的相同参考除预测。它不依赖弯曲真场拟合，不保证消除网格相关传播常数差异。','', '| 网格变化 | 指标 | 平均 | 最大 |','|---|---|---:|---:|']
for lo,hi in [(20,30),(30,40)]:
 for region,conv,met,name in [('bend_core','raw','phase_wmae_deg','原始弯曲相位WMAE/°'),('bend_core','incident_normalized','phase_wmae_deg','入射参考规范化弯曲相位WMAE/°'),('normal_pair_all','raw','normal_relative_phase_wmae_deg','法向相对相位WMAE/°'),('bend_core','incident_normalized','complex_rel_l2','入射参考规范化复场RelL2')]:
  vals=[r[met] for r in tc if r['low_resolution']==lo and r['high_resolution']==hi and r['region']==region and r['convention']==conv];lines.append(f'| {lo}→{hi} | {name} | {np.mean(vals):.5f} | {max(vals):.5f} |')
lines+=['', '| 种子 | 真值resolution | 约定 | 弯曲相位Δ/° | 法向相位Δ/° |','|---|---:|---|---:|---:|']
for s in SEEDS:
 for res in [20,30,40]:
  for conv in ['raw','incident_normalized']:
   a=comp(gp,s,'phase_wmae_deg','bend_core',res=res,conv=conv);b=comp(gp,s,'normal_relative_phase_wmae_deg','normal_pair_all',res=res,conv=conv,eta=.5);lines.append(f"| {s} | {res} | {conv} | {a['delta']:+.5f} | {b['delta']:+.5f} |")
lines+=['','18例全体中，LocalTangent相对ArcOnly的两项相位均值差在三个种子、20/30/40真值网格及两种参考约定下均为负。固定入射参考下，30/40的法向相位改善在三个种子的配对区间也均低于零；对应40网格均值下降约0.447/0.522/0.841°。但30/40的弯曲绝对相位改善区间均跨零，不能宣称两个指标在细网格上具有同样强的证据。真场30→40的法向相位变化约1.70°、规范化弯曲相位变化约8.81°，尚不满足精度门槛。数值参考未收敛与相对改善仍存在是可以同时成立的两个结论。','', '### 四方法排序对真值网格的敏感性','', '固定入射参考下，按18例几何平均误差从小到大排列；完整原始约定及复场指标保存在grid_summary.csv。','', '| 种子 | 真值resolution | 指标 | 排序（从优到差） |','|---|---:|---|---|']
for s in SEEDS:
 for res in [20,30,40]:
  for met,region,eta,name in [('phase_wmae_deg','bend_core',0,'弯曲相位'),('normal_relative_phase_wmae_deg','normal_pair_all',.5,'法向相对相位')]:
   vals=[(m,summ(gs,s,m,met,region,res=res,conv='incident_normalized',eta=eta)['mean']) for m in METHODS];vals.sort(key=lambda v:v[1]);rank=' < '.join(labels[m]+' ('+f'{v:.3f}'+')' for m,v in vals);lines.append(f'| {s} | {res} | {name}/° | {rank} |')
lines+=['','### 插值敏感性复核','', '在18例resolution40上，双线性与三次复数插值引起的法向相位指标平均差约0.00411°。弯曲芯区绝对相位的LT−Arc平均差值变化不超过0.000054°（raw与固定入射参考均检查），见field_interpolation_sensitivity.json。插值方式不足以解释度量级网格变化。','', '## 预先规定的研究决策','', '模型用resolution20数值标签训练。细网格排序改变混合了训练标签偏差与表示误差，不能单凭此证明LocalTangent机制本身失效；但足以阻止把当前小幅收益直接宣称为连续物理解或实际相干控制的可靠优势。保持原离散Block几何的网格检查也不检验理想平滑曲线的几何离散误差。','', '门槛用于本轮是否投入下游仿真，不是期刊要求，也不是严格连续解误差界。','', '```json',json.dumps(dec['checks'],ensure_ascii=False,indent=2),'```','', '附加resolution60：'+('已按预选清单执行3个，未继续扩样。' if refined['executed'] else '未执行，原因：'+refined['reason'])]
if refined['executed']:
 rt=read(R/'refined_truth_changes.json');lines+=['','| 40→60指标（仅预选3例） | 平均 | 最大 |','|---|---:|---:|']
 for reg,conv,met,name in [('bend_core','incident_normalized','phase_wmae_deg','入射参考规范化弯曲相位WMAE/°'),('normal_pair_all','raw','normal_relative_phase_wmae_deg','法向相对相位WMAE/°')]:
  vals=[r[met] for r in rt if r['low_resolution']==40 and r['region']==reg and r['convention']==conv];lines.append(f'| {name} | {np.mean(vals):.5f} | {max(vals):.5f} |')
 lines+=['','3例40→60检查是诊断性细化，每倾角档一例，均为w=0.4μm且左端激励；对象在结果前已锁定，不替代18例全体或跨宽度收敛认证。']
lines+=['','反向激励阶段：'+('已生成'+str(reverse['truth_pairs'])+'个同几何双向配对真场；没有开展相干控制。' if reverse['executed'] else '未执行；'+reverse['reason']+'。'),'','研究决策：'+('可据本轮证据另行设计双端相干应用，不能预先宣称设计优势。' if ready else '保留LocalTangent在新几何、训练网格上的局部波前独立复现结论；关于细网格物理精度与应用价值的结论须收窄。优先处理FDTD参考精度，暂停双向应用扩展。'),'','## 计算与核验','']
base=read(R/'base_costs.json');fine=read(R/'fine_costs.json');end=read(R/'analysis_complete.json');pres=read(R/'preservation_check.json');ind=read(R/'independent_verification.json');feature=read(R/'feature_replay.json')
lines +=[f"实际完成72个resolution20基础仿真、36个resolution30/40细网格仿真，条件resolution60 {len(refined.get('runs',[]))}例；12组冻结模型推理，无训练。分析完成累计{end['elapsed_seconds']/60:.1f}分钟，四小时为上限，不为用满预算继续扩展。基础仿真冷进程均值{np.mean([r['seconds'] for r in base]):.2f}秒，细网格冷进程均值{np.mean([r['seconds'] for r in fine]):.2f}秒；完整耗时分别保存在base_costs/fine_costs/inference_costs。",'',f"{pres['protected_inputs']}个受保护源文件、清单、权重及校准文件哈希未变；12份新预测冻结哈希未变。左右源输入及三载波重放原缓存逐值相等；ArcOnly以标量弧长积分另查方向符号。36次标量相位核验；{ind['normal_pair_hand_bilinear_checks']}次手写双线性/复相位点对核验，最大相位差{ind['max_phase_difference_deg']:.3g}°、复交叉项RelL2差{ind['max_cross_term_relL2_difference']:.3g}。独立点对核验初次使用complex64预测，已修正为与冻结主评价相同的complex128后重放；未改变原始数组或指标定义。",'', '12个best权重亦与历史训练summary的哈希逐一一致；三个初始化不同，同种子四方法初始化与批次一致，见training_seed_verification.json。细网格源坐标和手写插值核验见fine_coordinate_verification.json；固定入射参考的公共相位不变性见normalization_verification.json。','', '文件入口：PROTOCOL.md、config.json、geometry_manifest.json、freeze.json、prediction_freeze.json、decision.json；逐几何原表blind_per_geometry.csv、grid_per_geometry.csv、grid_truth_changes.csv；全部12份预测位于predictions/，新全场位于data/及fine/。','', '![Blind and grid summary](blind_and_grid.png)','', '![Local slope and normal position](spatial_phase.png)']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(R/'report_complete.json',dict(unix=time.time(),report_sha256=sha(R/'REPORT.md'),figures=['blind_and_grid.png','spatial_phase.png'],automatic_extension=False));print('REPORT_COMPLETE',flush=True)






