import numpy as np
from support import ROOT,OUT,V1,OLD,read,sha,save,now

def link(name,label=None):return f'[{label or name}](<D:/game/research_sbend_path_v2/{name}>)'
def oldlink(name,label):return f'[{label}](<D:/game/{name}>)'
def main():
    decision=read(OUT/'decision.json');summary=read(OUT/'phase_summary.json');g=read(OUT/'gradient_checks.json')
    p=read(OUT/'continuity_probes.json');inv=read(OUT/'geometry_invariants.json');dev=read(OUT/'core_deviation.json')
    refine=read(OUT/'refinement_summary.json');runtime=read(OUT/'runtime.json');records=read(OLD/'sbend_index.json')
    data=read(OUT/'phase_diagnostics.json')
    def find(axis='propagation',band='core',zone='all',baseline='path',group='all',frame='curvilinear',gate='amplitude_1pct'):
        return next(r for r in summary if all(r[k]==v for k,v in dict(axis=axis,band=band,zone=zone,baseline=baseline,group=group,frame=frame,gate=gate).items()))
    def metric(gid,method,axis):
        r=next(r for r in data if r['geometry_id']==gid and r['method']==method and r['axis']==axis
            and r['frame']=='curvilinear' and r['zone']=='all' and r['band']=='core' and r['gate']=='amplitude_1pct')
        return r['weighted_rms']
    pr=find();nr=find(axis='normal');h=find(axis='normal',band='halo');strong=find(baseline='local_tangent')
    worst=max((r for r in g if r['method']=='path' and r['band']=='all'),key=lambda r:r['max_abs_coordinate_gradient'])
    lines=['# S-bend Path v2有限开发与全域连续性检查报告','',
        f'完成UTC：{now()}。seed2026；唯一候选，16个已看S-bend development样本；未新仿真、训练/推理、生成正式几何或评价新的保留集。','',
        '**结论：唯一v2候选消除了v1的最近投影坐标跳变，在16例上保留了芯内相位补偿，值得以修复二维扩展的独立版本加入正式对照。其优势是连续性，尚无神经网络精度证据；halo法向残余比v1恶化，不能描述为所有区域更优。**','',
        '## 1. 阅读与历史保护','',
        '已阅读全文'+oldlink('research_sbend_path/S-bend路径适配与物理诊断报告.md','上一轮报告')+'、'+oldlink('research_sbend_path/PROTOCOL.md','开发协议')+'、'+oldlink('research_sbend_path/FORMAL_PROTOCOL.md','正式协议')+'及path_geometry/diagnose/验证/补充分析源码，复用16例固定坐标、物理源和保存Ez。旧Linear、LocalTangent、Path v1载波及统计定义均原样保留。本目录独立新增，不覆盖旧冻结哈希。','',
        '诊断前写入'+link('PROTOCOL.md')+'并冻结'+link('candidate_freeze.json')+'；只构造一个v2。没有参数搜索、看结果后修改核宽/过渡宽/门控、重新标定beta或删除不利区域。额外网格细化只改变检查采样，不改变算法sigma。','',
        '## 2. 为什么不能只把v1乘平滑权重','',
        'v1以最近折线段的弧坐标为二维扩展，在远包层不同曲线位置之间切换，甚至折线顶点周围也可能存在很小的多重投影。仅用到中心线的距离混合，若非零权重覆盖多值区域，仍可能留下跳变；这不能用原网格看起来平滑来认证。','',
        '唯一候选同时做固定尺度正核弧长平均和远场切换。前者只使用几何求积节点，绝不是由真值场拟合载波；它近似保留芯内v1弧长，而非承诺逐点等同。','',
        '设节点(c_j,s_j)沿官方256点中心折线均匀弧长求积，梯形正权w_j、间距≤0.0125μm，两端直线延长0.5μm。核宽sigma=0.05μm（一个原网格），Q为：','',
        'Q(r)=Σw_j s_j exp(−|r−c_j|²/(2sigma²))/Σw_j exp(−|r−c_j|²/(2sigma²))。令z=y−yc(x)、r0=w/2+0.05μm、r1=r0+0.5μm，B=1−smoothstep5((|z|−r0)/(r1−r0))，A=smoothstep5((x−xb_start)/0.05)·smoothstep5((xb_end−x)/0.05)，则：','',
        'D_v2=σdir·{s(x)+A[B(Q−s(x))+(1−B)t_y(x)z]−s_source}；carrier=exp(i beta D_v2)。','',
        'smoothstep5在[0,1]外截为0/1，区间内为6u^5−15u^4+10u^3。σdir=左源+1/右源−1，真实源x=±11.9μm；s_source用真实source_center_um投影得到。A在弯曲端各一个网格内起落，在直引线为0，确保整个真实源面为零。远包层是几何门控后的LocalTangent延拓；弯曲内部A=1时就是旧LocalTangent，直引线上严格列弧长。旧LocalTangent未修改。','',
        '源参考、传播方向、几何平移/镜像/180°旋转关系及单位模保留。beta与旧方法完全相同，来自冻结Y训练标定；不读取16例MPB波矢/相位斜率/S来构造或校准候选。接口见'+link('path_v2.py')+'，未接收Ez/H/S或geometry_id。','',
        '## 3. 连续性：结构论证与实际检查','',
        '有限正高斯核和的分母严格为正，故数学Q连续且光滑；指数减去行最大值只作等价数值稳定。列坐标s(x)、yc(x)与插值切向t_y(x)连续，A/B为连续切换。因此物理域上D_v2及单位复载波全域C0连续，未保留最近点分支选择。实际仅在A·B>0处计算Q；边界系数及前两阶导数为零，舍去带外Q不引入跳跃。','',
        '这不是全域C1/C2或eikonal解：官方折线和分段线性切向仍有导数折点。连续不代表各处梯度都小；以下最大梯度是16例数值证据，不是未来整个参数空间的统一上界证明。','',
        '| 载波 | 全16例最大单轴距离梯度 | 最大未取模相位梯度 rad/μm | 梯度>10的网格边总数 |','|---|---:|---:|---:|']
    for method,label in [('linear','Linear'),('local_tangent','LocalTangent'),('path','Path v1'),('path_v2','Path v2')]:
        a=[r for r in g if r['method']==method and r['band']=='all']
        lines.append(f'| {label} | {max(r["max_abs_coordinate_gradient"] for r in a):.6f} | {max(r["max_abs_unwrapped_phase_gradient_rad_um"] for r in a):.6f} | {sum(r["count_gradient_gt_10"] for r in a)} |')
    lines+=['',f'原v1最坏例仍是{worst["geometry_id"]}，x={worst["x_um"]:.3f}、y={worst["y_um"]:.3f}μm。每例先在旧最坏边按几何弧坐标变化二分定位，再固定位置用epsilon=1e-2..1e-8μm检查，不按v2效果重新选点。','',
        '| 探针半间距 μm | v1旧切换处最大距离差 μm | v2同处最大距离差 μm | v2同处最大复载波差 |','|---|---:|---:|---:|']
    for eps in [1e-2,1e-4,1e-6,1e-8]:
        a=[r for r in p if r['kind']=='old_worst_cut' and abs(r['epsilon_um']/eps-1)<.01]
        lines.append(f'| {eps:.0e} | {max(r["max_abs_distance_difference_um"] for r in a if r["method"]=="path"):.9g} | {max(r["max_abs_distance_difference_um"] for r in a if r["method"]=="path_v2"):.9g} | {max(r["max_complex_carrier_difference"] for r in a if r["method"]=="path_v2"):.9g} |')
    lines+=['','v1在最小间距仍有有限跳变；v2差值随间距缩小而线性下降。该检查同时保留未取模距离/相位和复载波差，避免把2π取模掩盖当作修复。','',
        '| v2检查位置 | epsilon=1e-8时最大距离差 μm | 最大有限差分梯度 |','|---|---:|---:|']
    for kind in ['r0','r1','bend_gate','source_plane']:
        a=[r for r in p if r['method']=='path_v2' and r['kind']==kind and r['epsilon_um']<1.01e-8]
        lines.append(f'| {kind} | {max(r["max_abs_distance_difference_um"] for r in a):.9g} | {max(r["apparent_max_gradient"] for r in a):.6f} |')
    lines+=['',f'全16例保持sigma=0.05μm、检查网格由0.05细化到0.025μm，最大梯度分别为{refine["grid_max_gradient"]["0.05"]:.6f}、{refine["grid_max_gradient"]["0.025"]:.6f}；v2自身最陡边epsilon=1e-8处最大梯度{refine["peak_epsilon_1e8_max_gradient"]:.6f}，无发散/非有限。见'+link('grid_refinement.csv')+'、'+link('continuity_probes.csv')+'、'+link('v2_peak_refinement.csv')+'。','',
        '### 3.1 几何不变量与源面','',
        f'16/16通过全部预定不变量。最大180°旋转/输入交换误差{max(r["rotation_equivariance_um"] for r in inv):.3g}μm，最大平移误差{max(r["translation_equivariance_um"] for r in inv):.3g}μm，y镜像误差{max(r["y_reflection_equivariance_um"] for r in inv):.3g}μm。源面横向采样最大参考误差{max(r["source_plane_zero_um"] for r in inv):.3g}μm，方向反号误差{max(r["direction_antisymmetry_um"] for r in inv):.3g}μm，单位模误差{max(r["unit_modulus_error"] for r in inv):.3g}。','',
        f'解析直线极限最大误差{decision["analytic_straight_max_error_um"]:.3g}μm。保存复64载波存在正常浮点舍入，不改变算法exp(i beta D)的单位模定义。详细逐例记录：'+link('geometry_invariants.json')+'。','',
        '![连续性与芯内相位](D:/game/research_sbend_path_v2/continuity_and_core.png)','',
        '![同一几何梯度图](D:/game/research_sbend_path_v2/gradient_maps.png)','',
        '## 4. 芯内相位补偿是否保留','',
        '完全复用旧profiles的固定s/n坐标、真值Ez及1%幅度门控，权重仍为相邻幅度乘积；不为v2移动ROI或门槛。每几何先算加权RMS，再几何等权平均。原v1/Linear/LocalTangent所有重复计算的加权RMS与历史表最大差为0，历史文件未重写。单位rad/μm。','',
        '| 位置/方向 | Linear | LocalTangent | Path v1 | Path v2 | v2相对v1改善 | v2胜/负v1 |','|---|---:|---:|---:|---:|---:|---|']
    for frame,axis,zone,band in [('curvilinear','propagation','all','core'),('curvilinear','normal','all','core'),
        ('curvilinear','propagation','bend','core'),('curvilinear','normal','bend','core'),
        ('curvilinear','propagation','all','halo'),('curvilinear','normal','all','halo'),
        ('curvilinear','normal','bend','halo'),('cartesian','x','all','whole'),('cartesian','y','all','whole')]:
        a=find(axis,band,zone,'path',frame=frame);b=find(axis,band,zone,'linear',frame=frame);c=find(axis,band,zone,'local_tangent',frame=frame)
        lines.append(f'| {frame}/{zone}/{band}/{axis} | {b["baseline_mean"]:.6f} | {c["baseline_mean"]:.6f} | {a["baseline_mean"]:.6f} | {a["v2_mean"]:.6f} | {a["relative_reduction"]*100:.3f}% | {a["improved"]}/{a["worsened"]} |')
    lines+=['',f'芯内相对v1最大未取模相位偏差{decision["max_core_phase_difference_vs_v1_rad"]:.6f}rad（预定门槛0.1rad），最坏单例芯内沿程/法向RMS恶化{100*decision["worst_per_geometry_core_rms_relative_degradation"]:.4f}%（门槛2%）。两个均值亦未恶化。故保留芯内补偿的门槛通过。相对Linear，v2沿程降低{100*find(baseline="linear")["relative_reduction"]:.2f}%、法向降低{100*find(axis="normal",baseline="linear")["relative_reduction"]:.2f}%，16例均改善。','',
        '### 4.1 逐几何配对与不利结果','',
        '| 几何尾号 | 输入 | v1沿程 | v2沿程 | v1法向 | v2法向 | 芯内最大相位差 rad |','|---|---|---:|---:|---:|---:|---:|']
    for rec in records:
        gid=rec['geometry_id'];dd=next(r for r in dev if r['geometry_id']==gid)
        lines.append(f'| {gid[-12:]} | {rec["input_port"]} | {metric(gid,"path","propagation"):.6f} | {metric(gid,"path_v2","propagation"):.6f} | {metric(gid,"path","normal"):.6f} | {metric(gid,"path_v2","normal"):.6f} | {dd["max_abs_core_phase_difference_vs_v1_rad"]:.6f} |')
    lines+=['',f'不利结果：芯内沿程有{pr["worsened"]}/16例v2略差，芯内法向有{nr["worsened"]}/16例略差。halo法向均值恶化{-100*h["relative_reduction"]:.3f}%，{h["worsened"]}/16例均较v1差；弯曲halo法向恶化{-100*find(axis="normal",band="halo",zone="bend")["relative_reduction"]:.3f}%。此为过渡改变波前的实际代价，连续性不能保证残余相位各区都改善。本轮不调参数去消除此负结果。','',
        f'与强基线LocalTangent，芯内沿程均值只差约{100*strong["relative_reduction"]:.4f}%（v2略低），法向低约{100*find(axis="normal",baseline="local_tangent")["relative_reduction"]:.3f}%；不能据此宣传显著或有实质预测优势。v2主要价值仍是有连续性依据的近弧长二维扩展。','',
        '| 输入方向 | v1芯内沿程 | v2芯内沿程 | v1芯内法向 | v2芯内法向 |','|---|---:|---:|---:|---:|']
    for group in ['input1','input2']:
        a=find(group=group);b=find(axis='normal',group=group)
        lines.append(f'| {group} | {a["baseline_mean"]:.6f} | {a["v2_mean"]:.6f} | {b["baseline_mean"]:.6f} | {b["v2_mean"]:.6f} |')
    lines+=['','正负法向芯边、halo、弯曲/直引线、无门控RMS/MAE/p95和笛卡尔x/y结果全部保留在'+link('phase_diagnostics.csv')+'与'+link('phase_summary.json')+'。这些是已看开发集的描述，无独立测试/种子稳定性声明；没有按场/S过滤样本。','',
        '## 5. 对路线A的判断与正式载波','',
        '1. **全域不连续是否解决？是，针对v1最近分支选择造成的C0跳变。** v2构造有连续性依据，16例远场/切换边界/源面与网格细化通过；仍继承折线的导数折点，不能称全域光滑物理波前。','',
        '2. **是否保留芯内补偿？是，按预定门槛通过。** 最大相位偏差约0.0143rad，最坏单例RMS恶化约0.24%；不是严格逐点保留，也不掩盖halo损失。','',
        '3. **是否值得加入？是，作为有明确几何修复价值的唯一v2。** 尚无网络训练或器件功能改善证据；不能凭微小相位均值优势替代LocalTangent。','',
        '4. **正式比较：Direct、Linear、LocalTangent、Path v1、Path v2共五组。** v2对LocalTangent为主要强对照，v2对v1为实现消融；全部保留而不依据未来test挑选“Path版本”。五组共享全部12实通道几何输入、相同模型/初始化/批次/2400步和训练标定。v2包含近芯核平均与源面门控，v2/v1差异不能归为纯远场单因素效应。','',
        '5. **能否冻结并进入64/16/100？算法和本增补规则可以锁定，本轮算法开发到此结束。** 启动正式执行前仍须完成12通道打包、共享先验与无训练前向兼容性核对，再冻结可运行源码/版本和180几何划分清单。不能将尚未实现的训练适配器称为已全部冻结。这里只给出就绪边界，本轮不生成正式几何。','',
        '独立增补见'+link('FORMAL_PROTOCOL_ADDENDUM.md')+'；旧冻结协议保留。原16开发几何全部排除，beta只从新64训练场标定一次，100测试封存；五组预测与评价实现冻结后才一次评价。正式新几何若无场几何QC超过预定限额，暂停报告，不补挑/调v2或自动开新候选。','',
        '## 6. 历史Y-Branch证据与预算','',
        '既有'+oldlink('research_m4/summary_table.csv','M4冻结汇总')+'：seed2026的Linear RelL2=0.581503、Path=0.495988；'+oldlink('research_m4/结果解读与论文表述.md','M4解释')+'保留端口差异、单种子与M1/M2负面证据。E2历史资料'+oldlink('research_e2/README.md','索引')+'同样原样保留。这些只说明既有Y-Branch输出表示有历史收益，不验证S-bend v2，不将已看E2/M4集合重新称为下一方法独立测试。','',
        f'本轮主诊断用时{decision["wall_seconds"]:.2f}秒；v2原160×480距离图构造每例{min(r["candidate_grid_seconds"] for r in runtime):.3f}–{max(r["candidate_grid_seconds"] for r in runtime):.3f}秒，可预先缓存。细化检查不增加候选、不改sigma。','',
        '正式数据仍64+16+100，共180不同新几何：已测FDTD外推约47.4分钟、含QC预留1–2小时、物理数据1.5–2GB；五组训练历史预算外推约50–55分钟，含验证/输出预留1–2小时，总轮预留2–4小时，模型/预测留6–12GB。新12通道资源须先做无训练检查；这些均是未来估计，本轮未执行。','',
        '## 7. 产物、复算与完整性','',
        '- '+link('path_v2.py')+'：唯一候选；'+link('candidate_freeze.json')+'：诊断前参数与哈希。','',
        '- '+link('decision.json')+'：预定门槛结论；'+link('core_deviation.csv')+'、'+link('gradient_checks.csv')+'、'+link('continuity_probes.csv')+'、'+link('geometry_invariants.json')+'：逐几何证据。','',
        '- '+link('phase_diagnostics.csv')+'、'+link('phase_summary.json')+'、'+link('grid_refinement.csv')+'、'+link('maps','maps/')+'、'+link('profiles','profiles/')+'：含全部不利结果和固定候选载波。','',
        '- '+link('verification.json')+'、'+link('outputs_manifest.json')+'：历史哈希、原v1冻结文件、新候选哈希及输出完整性核查。','',
        '使用已有picflow-meep Python -B运行analyze.py、refine_geometry.py、plots.py、make_report.py可复算本目录输出；不重运行历史实验入口。候选参数和path_v2.py已冻结，不应覆写改参。']
    (OUT/'Path_v2连续性与物理诊断报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Chinese report written.',flush=True)
if __name__=='__main__':main()
