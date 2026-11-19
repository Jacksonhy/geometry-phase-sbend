import csv
import numpy as np
from common import ROOT,OUT,OLD,read,save,sha,now

def link(name,label=None):return f'[{label or name}](<D:/game/research_sbend_path/{name}>)'
def oldlink(rel,label):return f'[{label}](<D:/game/{rel}>)'
def f(x):return f'{x:.6g}'
def main():
    geo=read(OUT/'geometry_checks.json');ver=read(OUT/'geometry_verification.json');cal=read(OUT/'calibration_summary.json');cf=read(OUT/'calibration_field_crosscheck.json')
    summary=read(OUT/'phase_summary.json');local=read(OUT/'local_failure_cases.json');far=read(OUT/'far_field_extension.json');physical=read(OUT/'physical_tangent_diagnostics.json')
    center=read(OUT/'centerline_residuals.json');records=read(OLD/'sbend_index.json')
    data=list(csv.DictReader((OUT/'phase_diagnostics.csv').open(encoding='utf-8-sig')))
    def find(frame='curvilinear',axis='propagation',zone='all',band='core',gate='amplitude_1pct',group='all',baseline='linear',metric='weighted_rms'):
        return next(r for r in summary if all(r[k]==v for k,v in locals_copy(frame,axis,zone,band,gate,group,baseline,metric).items()))
    def metric(gid,method,axis='propagation',zone='all',band='core'):
        r=next(r for r in data if r['geometry_id']==gid and r['method']==method and r['frame']=='curvilinear' and r['axis']==axis and r['zone']==zone and r['band']==band and r['gate']=='amplitude_1pct')
        return float(r['weighted_rms'])
    pr=find();nr=find(axis='normal');br=find(zone='bend');bn=find(axis='normal',zone='bend');strong=find(baseline='local_tangent');strong_n=find(axis='normal',baseline='local_tangent')
    worst=max((r for r in far if r['method']=='path'),key=lambda r:r['max_abs_coordinate_gradient'])
    loc=[r for r in local if r['baseline']=='linear' and r['zone']=='core_bend' and r['axis']=='propagation']
    # No posthoc method changes: derive all tables from stored, fixed comparison rows.
    lines=['# S-bend单中心线路径适配与物理诊断报告','',f'完成时间UTC：{now()}。seed2026；只分析既有16个development样本。新增目录`D:\\game\\research_sbend_path`。',
        '', '**结论：近波导芯的单中心线弧长与左右源参考已正确适配；Path相对Linear确实减少残余传播/横向相位。但与强基线LocalTangent几乎持平，沿程指标略差；全域最近投影还存在远包层非光滑问题。** 本轮未生成正式180个几何，未运行FDTD、神经网络训练/推理或新的冻结测试。',
        '', '## 1. 阅读、输入与研究范围','',
        '已读取研究交接、Meep闭环报告/生成代码、原Y-Branch geometry_features、E3/M4冻结标定和16例完整S-bend样本。旧Y函数含上下支路、中线裁限及换支逻辑，不能用于本器件。本次独立实现，不改旧函数。交接中的后续建议作为背景，本次执行范围以用户明确任务为准。',
        '', '16例来自'+oldlink('research_meep_validation/sbend_index.json','原开发索引')+'，λ=1.55μm，dx=dy=0.05μm，160×480，左右激励各8。物理坐标来自保存DFT体素中心，源来自source_center_um（x=±11.9μm）；不以src_mask质心或旧±12μm标签替代。保存的Ez/MPB/S均只作诊断，不进入路径构造。',
        '', '诊断前定义见'+link('PROTOCOL.md')+'，完整算法见'+link('path_geometry.py')+'。协议先固定方法、ROI、门控、β参照与判据；本轮没有β/平滑/阈值搜索，也没有依据不利结果修改算法。补充物理步长和局部失败分析明确保留为补充，不替换原定主结果。',
        '', '## 2. 几何算法、源参考与数据隔离','',
        '用原样AST提取的官方euler_sbend_points(offset,R,n_pts=256)，按官方居中规则得到实际中心折线，左右接直引线到x=±12。它与官方生成几何的中心轨迹一致；没有把请求offset当实际端点位移，也没有按场的亮度/相位找路径。w仅用于材料/ROI及后续β，不影响中心线积分。',
        '', '设折线按+x排列，节点累计弧长s，源中心投影的弧长s_src，方向σ=+1（左源）或−1（右源）。对任意二维点r，先作最近线段的精确投影，再取：',
        '', '```text\nd_Path(r) = σ [s_nearest(r) − s_src]\nd_Linear(r) = σ (x − x_src)\ncarrier(r) = exp(i β d(r))\n```',
        '', '单位统一为μm与rad/μm。Path在源之前允许负距离，避免把上游强行截为0而引入新边界。源平面在直引线上，横向同一真实源面上的路径参考一致。所有carrier为单位模，改变表示不会改变同一预测包络误差到物理场误差的范数。',
        '', '主接口`geometry_features(params,x_um,y_um,source_center_um,input_port)`不接收Ez/H/S、geometry_id或查询相位；`carriers_from_features(features,beta)`只接收几何和预定β。β函数只接收eps_max、dx与训练系数。诊断脚本先构造载波，再读取保存MPB波矢与相位斜率作比较；本轮没有任何新β拟合。',
        '', '### 2.1 几何正确性实际核验','',
        f'16/16通过：弧长节点一致性、单调性/弦长下界、真实源零参考、同源方向反号、180°旋转与输入交换等变、芯内法向投影、直线及倾斜直线解析测试。中心线采样在eps芯内覆盖率最小{min(r["centerline_eps_core_coverage"] for r in ver["records"]):.6f}；栅格芯中心与实际折线最大偏差{max(r["raster_centerline_max_um"] for r in ver["records"]):.6f}μm（约0.65像素）。最近投影近芯正确不意味着整个包层光学相位准确。',
        '', f'弯曲区相对其x跨度的额外弧长为{min(r["excess_bend_length_um"] for r in geo):.6f}–{max(r["excess_bend_length_um"] for r in geo):.6f}μm；所有载波模长双精度误差最大{max(r["max_carrier_modulus_error"] for r in geo):.3g}。详细检查在'+link('geometry_verification.json')+'与'+link('geometry_checks.csv')+'。',
        '', '## 3. Y-Branch传播常数能否直接迁移','',
        '本轮所有载波统一使用历史64个Y训练样本冻结的k=a·eps_max+b，a=0.05577425214043362、b=0.12500072719840435，k单位rad/pixel，β=k/0.05。不因16例诊断重新拟合。源文件'+oldlink('research_e3_2400/manifest.json','E3冻结manifest')+'及其SHA已留档。',
        '', '| 对照参照 | β平均绝对相对差 | 最大相对差 | 至输出监测面的最大累积相位差 |','|---|---|---|---|',
        f'| 保存的MPB端口波矢 | {100*cal["relative_MAE_vs_MPB"]:.4f}% | {100*cal["relative_max_vs_MPB"]:.4f}% | {cal["max_abs_accumulated_phase_bias_rad"]:.6f} rad |',
        f'| 同次FDTD下游直引线相位斜率 | {100*cf["FDTD_lead_relative_MAE"]:.4f}% | {100*cf["FDTD_lead_relative_max"]:.4f}% | {cf["FDTD_lead_accumulated_phase_max"]:.6f} rad |',
        '', '预定MPB诊断门槛为平均≤1%、最大≤2%、全传播累积最大≤π/4；按该门槛**未通过**。但同样数值门槛用于FDTD空间相位斜率则通过。MPB模式波矢与离散FDTD的空间相位斜率不是相同参照，不能仅因MPB失败就宣称旧标定对FDTD失效。实际观察：旧标定在本开发宽度/网格/波长范围内足以做定性相位比较，仍有约0.18%平均斜率差。',
        '', '**正式S-bend任务仍须只用新64训练场标定一次。** 沿每例下游直引线，离曲线端点0.75μm至x=±10.5μm，在中心±0.25w的5条横线复场平均后unwrap、OLS求传播方向斜率；64例等权拟合β=a·eps_max+b。所有方法共用系数。验证/测试场、每查询MPB波矢及S均不用于重标。拟合失败则在正式测试前停下，不能事后换ROI/拟合阶数。见'+link('FORMAL_PROTOCOL.md')+'。',
        '', '此处OLS直引线斜率只用于评价β，本轮没有把它写入任何载波；'+link('calibration_transfer.csv')+'保留逐几何两种参照，'+link('calibration_summary.json')+'保留MPB判据失败，'+link('calibration_field_crosscheck.json')+'单独记录FDTD结果，不覆盖负面判据。',
        '', '## 4. 实际竞争的相位基线','',
        '除Linear外，预先实现LocalTangent：设yc(x)为同一折线中心、s(x)为沿线弧长、t_y(x)为归一化局部切向的y分量，构造：',
        '', '```text\nd_LocalTangent(x,y) = σ [s(x) + t_y(x)(y−yc(x)) − s_src]\n```',
        '', '这是最近弧坐标在中心线附近的一阶几何展开，已包含几何弧长与横向传播方向修正，使用与Path相同β、源和几何信息。切向由固定中心差分在折线节点得到，不调平滑参数。它是本轮设计/实现的强几何相位对照，不冒充某篇论文的原算法或全波求解器。',
        '', '补充源平面复核：Path、Linear和ColumnArc在真实源面横向范围内严格为零；LocalTangent在源中心也为零，但节点切向插值使左源面出现最大约2.45e-6μm的横向距离偏差（载波相位约2.3e-5rad以内）。这属于强基线的近似边界，保留原冻结实现，不为消除该误差重算或调参。最终检查分别验证严格源面参考和该近似项，未将四者都当作严格平面常数。',
        '', '另外保存ColumnArc=σ[s(x)−s_src]，整列共享弧长，用来区分中心线补偿与二维扩展。ColumnArc不是正式第五个模型；正式四组预定Direct、Linear、LocalTangent、Path，强主比较为Path对LocalTangent。',
        '', '## 5. 残余相位诊断定义','',
        '先按几何建立s/n曲线网格，中心线步长0.05μm、法向n∈[-1,1]μm/步长0.025。Ez只在此时双线性取样。相邻残余相位梯度为angle(E1·conj(E0)·exp(−iΔφ_carrier))/Δ坐标。主ROI为芯内|n|≤w/2、物理x∈[-11,11]；弯曲/直引线分开。四载波使用完全相同采样、幅度门控和权重。',
        '', '主表报告每例幅度乘积加权RMS，再对16几何等权平均，单位rad/μm；门控为相邻两点均≥该几何中心线最大幅度的1%。同时保留无门控RMS/MAE/p95、全域笛卡尔x/y、法向正负侧/halo，未删除低幅结果。沿程s导数是曲线坐标导数；另用相邻采样点的真实欧氏距离计算物理切向梯度，单列补充。',
        '', '相位的π分支和低幅零点会影响局部相位；未以相位unwrap拟合或校正载波。中心线总变化使用unwrap，图中的起点常数仅用于显示，与模型评价的全局相位对齐无关。',
        '', '### 5.1 总体与弯曲区（幅度加权RMS）','',
        '| 位置/方向 | Linear | ColumnArc | LocalTangent | Path | Path对Linear降低 | Path胜/负 |','|---|---|---|---|---|---|---|']
    for frame,axis,zone,band in [('curvilinear','propagation','all','core'),('curvilinear','propagation','bend','core'),('curvilinear','normal','all','core'),('curvilinear','normal','bend','core'),('curvilinear','propagation','all','halo'),('curvilinear','normal','all','halo'),('cartesian','x','all','core'),('cartesian','y','all','core'),('cartesian','x','all','whole'),('cartesian','y','all','whole')]:
        r=find(frame,axis,zone,band);ct=find(frame,axis,zone,band,baseline='column_arc');lt=find(frame,axis,zone,band,baseline='local_tangent')
        label=f'{frame}/{zone}/{band}/{axis}'
        lines.append(f'| {label} | {f(r["baseline_mean"])} | {f(ct["baseline_mean"])} | {f(lt["baseline_mean"])} | {f(r["path_mean"])} | {100*r["relative_reduction"]:.2f}% | {r["improved"]}/{r["worsened"]} |')
    physmean={m:np.mean([r['physical_tangent_weighted_rms'] for r in physical if r['method']==m and r['zone']=='all']) for m in ('linear','column_arc','local_tangent','path')}
    lines+=['', f'真实空间步长的物理切向补充：Linear={f(physmean["linear"])}、ColumnArc={f(physmean["column_arc"])}、LocalTangent={f(physmean["local_tangent"])}、Path={f(physmean["path"])} rad/μm，与主s坐标结论一致。',
        '', f'中心线净残余相位变化的绝对值均值：Linear={np.mean([abs(r["unwrapped_net_residual_rad"]) for r in center if r["method"]=="linear"]):.6f}rad，Path={np.mean([abs(r["unwrapped_net_residual_rad"]) for r in center if r["method"]=="path"]):.6f}rad；Path范围{min(r["unwrapped_net_residual_rad"] for r in center if r["method"]=="path"):.6f}–{max(r["unwrapped_net_residual_rad"] for r in center if r["method"]=="path"):.6f}rad。LocalTangent在中心线与Path相同，不能由中心线结果证明精确二维投影更优。',
        '', '### 5.2 逐几何、方向与弯曲程度','',
        '| 几何尾号 | 输入 | R_min μm | 芯内沿程Linear | LocalTangent | Path | 芯内法向Linear | Path |','|---|---|---|---|---|---|---|']
    for rec in records:
        gid=rec['geometry_id'];lines.append(f'| {gid[-12:]} | {rec["input_port"]} | {f(rec["params"]["R_min_um"])} | {f(metric(gid,"linear"))} | {f(metric(gid,"local_tangent"))} | {f(metric(gid,"path"))} | {f(metric(gid,"linear","normal"))} | {f(metric(gid,"path","normal"))} |')
    lines+=['', '| 分组 | N | 沿程Linear | 沿程Path | Path对Linear降低 |','|---|---|---|---|---|']
    for group in ['input1','input2','smaller_R','larger_R']:
        r=find(group=group);lines.append(f'| {group} | {r["geometries"]} | {f(r["baseline_mean"])} | {f(r["path_mean"])} | {100*r["relative_reduction"]:.2f}% |')
    lines+=['', 'R分组按16例R_min中位数固定，仅描述开发集；实际弯曲还受offset/R比值影响，不将R分组当控制变量的因果效应。未观察到某输入侧或某几何的芯内总体Path劣于Linear，但这不等同不存在局部失败。',
        '', '![标定与整体相位](D:/game/research_sbend_path/calibration_and_phase.png)',
        '', '![全部16例中心线残余](D:/game/research_sbend_path/all16_centerline_residuals.png)',
        '', '## 6. 不利结果与适用边界','',
        '### 6.1 Path没有在强对照上取得一致优势','',
        f'芯内沿程LocalTangent={strong["baseline_mean"]:.6f}、Path={strong["path_mean"]:.6f}：Path平均约差{-100*strong["relative_reduction"]:.3f}%，{strong["worsened"]}/16例更差；芯内法向Path优约{100*strong_n["relative_reduction"]:.2f}%。弯曲区沿程14/16例也是Path较差，halo沿程16/16例Path较差。这些差异很小，不作开发样本显著性声明，但足以证明强基线具有实际竞争力。',
        '', 'ColumnArc沿程也接近Path，法向却未消去局部倾斜相位。这说明“二维路径如何扩展”是主要待检因素之一；不能将Linear对Path的差异直接归为精确投影的独特优势。',
        '', '### 6.2 局部相位有恶化，尤其横向边缘','',
        f'即使总体下降，弯曲芯内沿程局部绝对梯度Path比Linear更大的幅度权重占比均值为{100*np.mean([r["path_worse_weight_fraction"] for r in loc]):.2f}%，逐例{100*min(r["path_worse_weight_fraction"] for r in loc):.2f}%–{100*max(r["path_worse_weight_fraction"] for r in loc):.2f}%。最大单位置恶化达{max(r["max_local_increase_rad_um"] for r in loc):.6f}rad/μm。',
        '', '正负法向侧、真实曲率内/外侧都保留；Path残余在芯边缘/halo大于中心，弯曲辐射、干涉、模式变化和旧β误差均可能贡献，不能仅据图分解各原因。物理内侧定义n·κ>0、外侧n·κ<0（|κ|>0.02/μm）；逐例坐标和误差记录在'+link('local_failure_cases.csv')+'。',
        '', '| 区域 | 沿程Linear | 沿程Path | 法向Linear | 法向Path |','|---|---|---|---|']
    for zone in ['inner_half','outer_half']:
        vals=[]
        for axis in ['propagation','normal']:
            a=[r for r in local if r['baseline']=='linear' and r['zone']==zone and r['axis']==axis]
            vals.extend([np.mean([r['baseline_weighted_rms'] for r in a]),np.mean([r['path_weighted_rms'] for r in a])])
        lines.append(f'| {zone} | '+' | '.join(f(x) for x in vals)+' |')
    lines+=['', '![局部负例比例](D:/game/research_sbend_path/local_adverse_regions.png)',
        '', '### 6.3 全域最近投影并不光滑：尚未解决的实现风险','',
        f'最差例`{worst["geometry_id"]}`，位置x={worst["x_um"]:.3f}、y={worst["y_um"]:.3f}μm，离中心线约{worst["distance_to_centerline_um"]:.3f}μm：相邻0.05μm网格的投影弧坐标跳变{worst["coordinate_jump_um"]:.4f}μm，未取模载波相位差{worst["unwrapped_carrier_phase_jump_rad"]:.4f}rad；坐标梯度约{worst["max_abs_coordinate_gradient"]:.2f}。此处在远包层，最近曲线投影分支切换，不是波导真的跳跃。',
        '', f'全16例芯内Path坐标梯度最大{max(r["gradient_max_core"] for r in far if r["method"]=="path"):.4f}，而全域最大{max(r["max_abs_coordinate_gradient"] for r in far if r["method"]=="path"):.2f}；LocalTangent全域最大{max(r["max_abs_coordinate_gradient"] for r in far if r["method"]=="local_tangent"):.4f}。最近投影是明确可复算的几何坐标，在芯附近正确，但不能称全域光滑波前或eikonal解。单位模保证范数关系，不保证对神经网络好学；远场跳变可能增添高频困难。',
        '', '**本轮没有增加裁切、平滑、混合半径或调β来改善该结果。** v1冻结版本保留风险，正式协议不能宣称全域平滑。如果下一轮决定换平滑扩展，必须在正式几何/测试评价前形成v2开发协议，并保留本次负结果，不能在看过正式测试后选择。',
        '', '![远场投影负例](D:/game/research_sbend_path/far_field_failure.png)',
        '', '## 7. 下一轮64/16/100协议与预算','',
        '完整冻结候选协议为'+link('FORMAL_PROTOCOL.md')+'，机器清单为'+link('frozen_protocol.json')+'，16例ID/整数参数/eps及180°旋转指纹排除表为'+link('development_exclusions.json')+'。这里只冻结规则，未实例化180个正式参数、ID或场。',
        '', '数据方案：180个不同几何，8档宽度均覆盖；训练每宽8例、验证每宽2例，测试前四宽13例/后四宽12例；总体64/16/100，左右输入分别32/32、8/8、50/50。offset与R沿官方范围0.025μm格点采样。PCG64(2026)固定顺序，只按结构/几何可行性和重复拒绝，不看S/相位表现。原16例无论换输入侧或换ID一律排除，训练/验证/测试还按参数与eps指纹去重。',
        '', '四组Direct/Linear/LocalTangent/Path，同一11实特征与Fourier主干、同一初始化/批次、2400步，不增加lr搜索；主比较Path对LocalTangent，完整报告Linear。每50步在16验证上选best，保留固定last；全部模型和评价代码冻结后才一次性评价100测试，不按test选权重、β、表示或预算。先核验11通道打包与共同输入，不能直接调用固定80几何的旧训练入口。',
        '', '**需要的材料均可由本机后续生成，无需现在下载。** 未来正式样本保留同次E/H与alpha/S以供端口代理误差底核查；只用训练场求β/尺度，测试封存。本轮16例仅开发，不迁入64训练或16验证。',
        '', '| 项目 | 后续预算估计 |','|---|---|',
        '| 180例FDTD | 已测均值外推约47.4分钟，最快/最慢范围36.6–61.3分钟；连同QC预留1–2小时，串行 |',
        '| 存储 | 完整物理审计约1.5–2GB，紧凑包约105MB；模型/预测另留5–10GB |',
        '| 4组×2400步 | 历史单模型10–11分钟，训练约40–44分钟；含验证/重放/预测预留1–2小时 |',
        '| 资源 | FDTD旧实测峰RSS约169MiB，仍设4GiB/900秒单例限额；训练旧约6.4GiB已分配显存，新11通道先核资源 |',
        '| 整轮 | 预留2–4小时；这不是本轮已执行内容或完成保证 |',
        '', '在正式test结果前必须冻结：数据/软件版本、180几何及分组、16开发排除指纹、曲线与二维扩展/投影tie规则、真实源参考、β与尺度训练标定公式/ROI、全部载波和共同输入、主干与训练预算/检查点选择、端口代理及诊断门控、配对区间/负例报告规则。若要改变远场扩展，需先发布新版本，不能偷偷改当前冻结哈希。',
        '', '## 8. 可支持的论文结论与不能支持的结论','',
        '有实际证据：在16个已看S-bend开发几何上，几何路径表示在左右激励、不同R与横向区域减少了部分残余传播相位，中心线/源/弧长构造无Y分叉错误。支持“第二器件家族存在可检验的几何相位补偿机制”。',
        '', '不能推出：Path降低神经网络预测误差、优于所有合理相位、降低内在流形维数、严格Maxwell/S参数精度、全域平滑或正式未见几何泛化。本轮强对照反而提示精确投影不是唯一有效构造。今后的正式实验应允许LocalTangent持平或优于Path，保留远场风险，而不是只以Path赢Linear作结。',
        '', '## 9. 产物与复算','',
        '- '+link('path_geometry.py')+'：独立几何算法；'+link('maps','maps/')+'：16例距离/载波/坐标图；'+link('profiles','profiles/')+'：固定s/n采样与诊断相位。',
        '- '+link('phase_diagnostics.csv')+'：4352行固定区域/方法指标；'+link('phase_summary.json')+'：逐几何配对及完整分组；'+link('centerline_residuals.csv')+'。',
        '- '+link('geometry_verification.json')+'、'+link('calibration_transfer.csv')+'、'+link('far_field_extension.csv')+'、'+link('physical_tangent_diagnostics.csv')+'。',
        '- '+link('PROTOCOL.md')+'、'+link('FORMAL_PROTOCOL.md')+'、'+link('frozen_protocol.json')+'：开发定义、未来规则与冻结哈希。',
        '- '+link('verification.json')+'与'+link('outputs_manifest.json')+'：最终核查/新产物哈希；历史源码、模型、旧诊断/物理数据未修改。',
        '', 'CPU复算使用已有picflow-meep/bin/python -B依次运行diagnose.py、verify_geometry.py、local_failure_analysis.py、supplement.py、plots.py。它们不训练、不运行FDTD；输出只在本目录。freeze.py拒绝覆盖现有冻结清单。', '']
    (OUT/'S-bend路径适配与物理诊断报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Chinese adaptation report written.')
def locals_copy(frame,axis,zone,band,gate,group,baseline,metric):
    return dict(frame=frame,axis=axis,zone=zone,band=band,gate=gate,group=group,baseline=baseline,metric=metric)
if __name__=='__main__':main()
