"""Scientific diagnostics report; data-driven prose, no continuum certification."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import R,read,save,sha
cfg=read(R/'config.json');beta=read(R/'beta.json');decomp=read(R/'decomposition.json');attr=read(R/'attribution_tables.json');sens=read(R/'sensitivity.json');mat=read(R/'material_summary.json');mc=read(R/'material_controls.json');co=read(R/'courant_sensitivity.json');cond=read(R/'conditional80_decision.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
for ax,w in zip(axs[0],cfg['widths_um']):
 rr=sorted([r for r in beta if r['kind']=='centered' and r['width_um']==w and r['Courant']==.5],key=lambda r:r['resolution']);xx=[r['resolution'] for r in rr]
 for key,label,col in [('beta_FDTD','FDTD field slope','#207e86'),('beta_MPB','MPB default','#c58a2b'),('beta_MPB_high','MPB 4x resolution','#8b61a5')]:ax.plot(xx,[r[key] for r in rr],'-o',label=label,color=col)
 ax.axhline(rr[0]['beta_analytic_slab'],color='black',ls='--',label='Analytic slab TE0');ax.set(title=f'Straight guide width {w:g} um',xlabel='Resolution (pixels/um)',ylabel='Propagation constant (rad/um)',xticks=xx);ax.grid(alpha=.2)
 if w==.4:ax.legend(fontsize=8)
for ax,(lo,hi) in zip(axs[1],[(20,30),(30,40),(40,60)]):
 rr=[r for r in decomp if r['low_resolution']==lo and r['high_resolution']==hi];xx=np.arange(len(rr));ax.bar(xx-.18,[r['bend_phase_change_wmae_deg'] for r in rr],.36,label='Measured grid phase change',color='#c58a2b');ax.bar(xx+.18,[r['residual_after_delta_beta_wmae_deg'] for r in rr],.36,label='After measured straight delta-beta L',color='#207e86');ax.set(title=f'{lo} to {hi}: propagation diagnostic',xlabel='Fixed diagnostic S-bend index',ylabel='Bend-core phase WMAE (deg)',xticks=xx,xticklabels=[str(i+1) for i in xx]);ax.grid(axis='y',alpha=.2)
 if lo==20:ax.legend(fontsize=8)
fig.suptitle('FDTD phase-error decomposition | fixed original Block geometry\nPropagation subtraction is diagnostic only; no neural predictions corrected',fontsize=15);fig.savefig(R/'phase_error_decomposition.png',dpi=160);fig.savefig(R/'phase_error_decomposition.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
for ax,metric,title in zip(axs,['bend_phase_wmae_deg','normal_phase_wmae_deg','physical_t_phase_difference_deg'],['Bend-core phase WMAE','Normal-pair phase WMAE','Physical transmission phase change']):
 types=['control_plus100','control_strict','pml2','noavg','reordered'];xx=np.arange(len(types))
 for off,gid,color in [(-.22,cfg['control_ids'][0],'#207e86'),(0,cfg['control_ids'][1],'#c58a2b'),(.22,cfg['control_ids'][2],'#8b61a5')]:
  rr=[next(r for r in sens if r['geometry_id']==gid and r['variant']==v) for v in types];ax.bar(xx+off,[max(abs(r[metric]),1e-7) for r in rr],.21,color=color,label=gid[-6:])
 ax.set_yscale('log');ax.set(title=title,xlabel='Setting relative to baseline res40',ylabel='Absolute change (deg)',xticks=xx,xticklabels=['+100 time','tol1e-9','PML2','No smoothing','Reverse Blocks']);ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
axs[0].legend(title='Control geometry suffix',fontsize=8);fig.suptitle('Numerical setting sensitivity | sources, reference planes and material union fixed');fig.savefig(R/'setting_sensitivity.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for ax,gid in zip(axs,cfg['control_ids']):
 for res,col in [(20,'#b54b47'),(30,'#c58a2b'),(40,'#207e86'),(60,'#8b61a5')]:
  with np.load(R/'materials'/f'original_{gid}_r{res}/samples.npz') as z:f=(z['epszz']-z['eps_clad'])/(z['eps_core']-z['eps_clad']);eta=z['eta'];v=f[len(f)//2]
  ax.plot(eta,v,label=f'res{res}',color=col)
 ax.set(title=gid[-6:],xlabel='Normal coordinate eta=2n/w',ylabel='Normalized Ez material fraction');ax.grid(alpha=.2)
axs[0].legend();fig.suptitle('Actual epsilon_zz normal cut at bend midpoint | unchanged Block geometry');fig.savefig(R/'material_profiles.png',dpi=160);plt.close(fig)
lines=['# FDTD 相位误差来源分解','', '本轮只研究FDTD参考：复用9个S-bend的既有细网格结果，补齐resolution60，计算直波导传播常数与材料、终止、PML敏感性；没有训练、推理、增加网络或校正既有预测。四小时为硬上限。','', '本实验不把几个耦合数值效应强行拆成可独立相加的因果百分比。ΔβL扣除与同宽直参考仅作解释性诊断，不能作为原网络精度提高的证据。','', '## 传播常数：MPB与FDTD交叉检查','', '| 宽度/μm | resolution | FDTD β | 中心线相位斜率β | MPB β | MPB高分辨率β | 解析平板TE0 β |','|---:|---:|---:|---:|---:|---:|---:|']
for w in cfg['widths_um']:
 for r in sorted([r for r in beta if r['kind']=='centered' and r['width_um']==w and r['Courant']==.5],key=lambda r:r['resolution']):lines.append(f"| {w:g} | {r['resolution']} | {r['beta_FDTD']:.8f} | {r['beta_centerline']:.8f} | {r['beta_MPB']:.8f} | {r['beta_MPB_high']:.8f} | {r['beta_analytic_slab']:.8f} |")
lines+=['', 'β单位rad/μm，MPB波矢单位cycles/μm，已乘2π。FDTD β由导模E/H投影的纵向相位斜率提取；中心线斜率为独立场采样对照。固定窗口[-9,9]μm，其他[-7,7]/[-9,-1]/[1,9]窗口完整保留；MPB高分辨率显式设eig_resolution=4×resolution，不等同于FDTD离散β。解析平板TE0仅适用于相同连续、对称、均匀平板模型，是辅助连续参照。','', f"全部直波导的主模态斜率拟合残差最大{max(r['modal_slope_residual_rms_deg'] for r in beta):.5g}°；不同拟合窗口β跨度最大{max(r['window_beta_span'] for r in beta):.5g}rad/μm。反向投影比包含FDTD与MPB离散不正交底噪，不能直接视作真实反射幅度。原始拟合结果见beta.json。",'', '## ΔβL可以解释多少局部相位变化','', '每个S-bend使用同宽、同输入端横向位置及同激励方向的直波导β，传播长度取从固定输入参考面到物理采样点的原中心线有向弧长。跨网格相位差与ΔβL逐点按圆周相位比较，没有拟合相位偏置或比例系数。','', '| 网格变化 | 几何数 | 原弯曲相位变化均值/° | ΔβL后残余均值/° | 原法向相位变化/° | ΔβL后法向残余/° |','|---|---:|---:|---:|---:|---:|']
for lo,hi in [(20,30),(30,40),(40,60)]:
 rr=[r for r in decomp if r['low_resolution']==lo and r['high_resolution']==hi];mean=lambda key:np.mean([r[key] for r in rr]);lines.append(f"| {lo}→{hi} | {len(rr)} | {mean('bend_phase_change_wmae_deg'):.5f} | {mean('residual_after_delta_beta_wmae_deg'):.5f} | {mean('normal_phase_change_wmae_deg'):.5f} | {mean('normal_residual_after_delta_beta_wmae_deg'):.5f} |")
lines+=['', '该残余把弯曲传播、局部波前、材料离散、参考提取等效应混合在一起，不能直接命名为纯曲率误差；残余减少也不是独立方差解释率。横向法向点对自然消去公共相位，因此它是否被ΔβL解释需单独检查。逐几何结果见decomposition.csv。','', '## 同宽直参考抵消共同传播相位','', '| 网格变化 | S-bend端口相位变化绝对均值/° | 直波导端口相位变化绝对均值/° | S-bend/直参考附加相位变化绝对均值/° | 再扣βΔL残余绝对均值/° |','|---|---:|---:|---:|---:|']
for lo,hi in [(20,30),(30,40),(40,60)]:
 rr=[r for r in attr if r['low_resolution']==lo and r['high_resolution']==hi];mean=lambda k:np.mean([abs(r[k]) for r in rr]);lines.append(f"| {lo}→{hi} | {mean('Sbend_port_phase_change_deg'):.5f} | {mean('straight_port_phase_change_deg'):.5f} | {mean('bend_extra_phase_change_deg'):.5f} | {mean('bend_extra_after_betaDeltaL_deg'):.5f} |")
lines+=['','这里的相位变化均按圆周Arg定义，单位°；不能在跨越π时当作未经包裹的总累积相位。原生模态t与独立E/H物理Ez幅度约定的对照都保存在port_reference.csv。同一width在居中/横移网格下β差异见attribution_tables.csv，避免把几何平移在离散格点上的差异掩盖。','', '## 材料界面与Block数值表示','', '原始连续几何始终是256点中心线生成的重叠旋转Block。同材料对象顺序对照不改变连续并集；未替换Prism或平滑曲线。实际Ez材料取fields.get_inveps(Ez,Z,physical_point)的逆，另存trace-average epsilon和原始Block解析并集占据。坐标是固定的129条法向切线，eta∈[-1.8,1.8]145点，导出值本身包含Yee网格插值；采样到的几何宽度是诊断积分，不是精确边界定位。','', '| 变体 | resolution | 控制几何数 | 归一化εzz差RMS均值 | 最大点差 | 连续并集是否相同 |','|---|---:|---:|---:|---:|---|']
for typ in ['reordered','noavg']:
 for res in [20,40]:
  rr=[r for r in mc if r['variant']==typ and r['resolution']==res];lines.append(f"| {typ} | {res} | {len(rr)} | {np.mean([r['epsilonzz_difference_normalized_RMS'] for r in rr]):.6g} | {max(r['epsilonzz_difference_normalized_max'] for r in rr):.6g} | 是 |")
lines+=['', 'εzz差按εcore−εclad归一化。平滑开关有差异说明该设置影响实际数值介质；不自动证明默认平滑正确或更接近连续解。若同连续并集、仅Block顺序就出现可见差异，可将其视作当前重叠界面数值表示的敏感性证据，但仍需结合实际场差才能判断相位影响。完整分辨率依赖、局部有效宽度、标量trace与εzz差见material_summary.csv及material_controls.csv。','', '## 提取、终止、PML与时间步长对照','', '| 变体（相对res40基线） | 控制数 | 弯曲相位WMAE平均/最大° | 法向相位WMAE平均/最大° | 端口相位变化绝对最大° |','|---|---:|---|---|---:|']
for typ in ['cold_baseline_replay','control_plus100','control_strict','pml2','noavg','reordered']:
 rr=[r for r in sens if r['variant']==typ];lines.append(f"| {typ} | {len(rr)} | {np.mean([r['bend_phase_wmae_deg'] for r in rr]):.6g} / {max(r['bend_phase_wmae_deg'] for r in rr):.6g} | {np.mean([r['normal_phase_wmae_deg'] for r in rr]):.6g} / {max(r['normal_phase_wmae_deg'] for r in rr):.6g} | {max(abs(r['physical_t_phase_difference_deg']) for r in rr):.6g} |")
lines+=['', 'cold_baseline_replay重放先前独立运行；+100与tol1e-9是在同一次仿真中继续累积DFT，没有重置。最初严格续算的minimum参数被发现按绝对时间解释，因此保留初始结果，并额外对原定三个控制独立运行tol1e-9、绝对minimum400/maximum800；表内control_strict使用这些校验结果。实际时间均保存，达到上限不等于已收敛。PML1→2时cell26×10→28×12，保留几何、源和参考面，使非PML边界保持x±12/y±4。noavg为关闭eps_averaging；reordered只反转同材料曲线Block顺序。','', '| 宽度/μm | β(Courant0.5) | β(Courant0.25) | Δβ | Δβ×23μm/° |','|---:|---:|---:|---:|---:|']
for r in co:lines.append(f"| {r['width_um']:g} | {r['beta_Courant05']:.8f} | {r['beta_Courant025']:.8f} | {r['delta_beta']:+.8f} | {r['phase23_change_deg']:+.5f} |")
lines+=['', 'Courant对照可检测时间离散敏感性，但不能单独把FDTD空间差与时间差完全正交分解。MPB、FDTD和解析平板的离散差、材料界面与截断范围也会耦合。','', '## 有限resolution80补查','', '执行状态：'+('按预先限定的三个控制结构执行。' if cond['executed'] else '未执行；'+cond['reason'])]
if cond['executed']:
 lines+=['','| 控制几何 | 宽度 | 60→80弯曲相位WMAE/° | 法向相位WMAE/° |','|---|---:|---:|---:|']
 for r in decomp:
  if r['high_resolution']==80:lines.append(f"| {r['geometry_id']} | {r['width_um']:g} | {r['bend_phase_change_wmae_deg']:.5f} | {r['normal_phase_change_wmae_deg']:.5f} |")
 lines+=['','没有生成straight80，因此本表不做60→80的ΔβL解释。三个控制分别为窄/中/宽及低/中/高角，覆盖两方向，但不是完整因子设计或全9例的80网格认证。']
lines+=['','## 可据此得出的结论与限制','', '请结合上表分别判断：①直波导传播常数差可以解释的累积相位；②扣除后仍存的局部法向相位/弯曲附加相位；③界面表示对εzz与实际场的敏感性；④终止、PML和时间步长的量级。没有任何一步重新校准神经网络或用新真场拟合预测相位。若数值参考仍变化明显，应继续限定原论文结论为相对于明确分辨率数值标签的结果。','', '九个结构为诊断子集，已在上一轮分析过，不是新的独立盲测。宽度与激励方向在当前九例中有关联（窄左、中宽右），因此不能声称已分离方向和宽度的因果影响。Block并集不变也不意味着理想连续平滑波导被认证。','', '## 核验与文件','', '主要入口：config.json、PROTOCOL.md、freeze.json、jobs_freeze.json；完整真场runs/，材料采样materials/，beta.csv、decomposition.csv、attribution_tables.csv、port_reference.csv、material_summary.csv、sensitivity.csv。历史输入哈希检查见verification.json，直波导标量回归见independent_beta_verification.json，逐点标量圆周残余核验见independent_decomposition_verification.json。辅助geometry导入冲突在材料任务开始前已修复并保留初始脚本；分析边界检查只放宽浮点舍入容差。PML扩域导出包含额外边界像素，按实际坐标与维度核验后重试，失败记录保存在failures/；原预算未重置。严格DFT时间语义另作明确时长重查。详情见IMPLEMENTATION_REPAIR.json。','', '参考依据：[Meep模态分解](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/)说明MPB与FDTD离散存在差异；[子像素平滑](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)说明有效介电张量及角点局限；[PML说明](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/)要求检查厚度收敛；[Python接口](https://meep.readthedocs.io/en/latest/Python_User_Interface/)定义DFT停止条件。这些文档是方法依据，不是本实验的精度证书。','', '![Decomposition](phase_error_decomposition.png)','', '![Settings](setting_sensitivity.png)','', '![Materials](material_profiles.png)']
costs=[read(p) for p in (R/'costs').glob('*.json')];budget=read(R/'budget.json');elapsed=(__import__('time').time()-budget['start_unix'])/60
lines+=['','### 实际计算范围与预算','',f'成功保存{len(costs)}个独立FDTD运行成本记录；其中原定69任务、3个明确时长DFT核验、3个条件resolution80。原定三个控制另保存+100与严格续算快照。完成48组材料初始化采样；MPB横向窗口检查仅初始化材料、不运行时域传播。复用9例20/30/40和3例旧60全场，未重算全部72几何。报告生成累计{elapsed:.1f}分钟，原四小时截止未重置。PML导出检查失败的单次记录保留在failures/，未计为成功。','']
primary=[r for r in decomp if r['low_resolution']==30 and r['high_resolution']==40]
raw=np.mean([r['bend_phase_change_wmae_deg'] for r in primary]);res=np.mean([r['residual_after_delta_beta_wmae_deg'] for r in primary]);normal=np.mean([r['normal_phase_change_wmae_deg'] for r in primary]);reorder=[r for r in sens if r['variant']=='reordered'];stop=[r for r in sens if r['variant']=='control_strict'];pml=[r for r in sens if r['variant']=='pml2']
lines[2:2]=[f'主要发现：大部分弯曲绝对相位的网格变化可由直波导传播常数变化解释。9例30→40相位变化均值{raw:.3f}°，以实测ΔβL诊断后残余{res:.3f}°；法向相对相位仍变化{normal:.3f}°，该局部形状差异不能被公共传播相位扣除解释。','',f'在3个控制上，严格DFT时长重查的弯曲相位变化最大{max(r["bend_phase_wmae_deg"] for r in stop):.3g}°，PML加厚最大{max(r["bend_phase_wmae_deg"] for r in pml):.3g}°。相同连续Block并集仅改顺序，法向相位仍变化{min(r["normal_phase_wmae_deg"] for r in reorder):.3f}°–{max(r["normal_phase_wmae_deg"] for r in reorder):.3f}°，实际εzz也改变。当前证据更支持传播离散误差与界面表示敏感性，而非DFT停止或PML是主要来源。','']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(R/'report_complete.json',dict(unix=__import__('time').time(),report_sha256=sha(R/'REPORT.md'),figures=['phase_error_decomposition.png','setting_sensitivity.png','material_profiles.png']));print('REPORT_COMPLETE',flush=True)






