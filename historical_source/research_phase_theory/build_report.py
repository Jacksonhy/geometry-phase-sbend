"""Assemble a Chinese, self-contained report from fixed offline results."""
from summarize_and_plot import *

REPORT='几何相位分解_理论频谱与曲率模态可行性报告.md'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])+'\n'
def localpath():return str(OUT).replace('/mnt/d/','D:/')
def link(name):return f'[{name}]({localpath()}/{name})'
def fig(name):return f'![{name}]({localpath()}/figures/{name}.png)\n'

REFERENCES=[
 dict(id='R1',authors='Mordehai Heiblum; Jay H. Harris',title='Analysis of Curved Optical Waveguides by Conformal Transformation',year=1975,venue='IEEE Journal of Quantum Electronics 11(2), 75–83',doi='10.1109/JQE.1975.1068563',url='https://doi.org/10.1109/JQE.1975.1068563',fulltext='https://courses.ece.ucsb.edu/ECE227/227C_S10Coldren/curved_wgds.pdf',verified='作者机构书目与原文；弯曲标量波导的共形变换与损耗分析'),
 dict(id='R2',authors='Yuetsu Takuma; Mitsunobu Miyagi; Shojiro Kawakami',title='Bent asymmetric dielectric slab waveguides: a detailed analysis',year=1981,venue='Applied Optics 20(13), 2291–2298',doi='10.1364/AO.20.002291',url='https://doi.org/10.1364/AO.20.002291',verified='出版社书目与摘要；未取得受限全文，未引用其具体级数系数'),
 dict(id='R3',authors='Steven G. Johnson; Peter Bienstman; M. A. Skorobogatiy; Mihai Ibanescu; Elefterios Lidorikis; J. D. Joannopoulos',title='Adiabatic theorem and continuous coupled-mode theory for efficient taper transitions in photonic crystals',year=2002,venue='Physical Review E 66, 066608',doi='10.1103/PhysRevE.66.066608',url='https://doi.org/10.1103/PhysRevE.66.066608',fulltext='https://math.mit.edu/~stevenj/papers/JohnsonBi02.pdf',verified='作者全文；局域模态/谱隙/导模条件，不能无条件推广到辐射连续谱'),
 dict(id='R4',authors='Allan W. Snyder',title='Coupled-Mode Theory for Optical Fibers',year=1972,venue='Journal of the Optical Society of America 62(11), 1267–1277',doi='10.1364/JOSA.62.001267',url='https://doi.org/10.1364/JOSA.62.001267',verified='出版社书目与摘要；未取得受限全文，只使用其确立耦合模框架的范围性信息'),
 dict(id='R5',authors='Dzmitry M. Shyroki',title='Exact Equivalent Straight Waveguide Model for Bent and Twisted Waveguides',year=2008,venue='IEEE Transactions on Microwave Theory and Techniques 56(2), 414–419',doi='10.1109/TMTT.2007.914637',url='https://doi.org/10.1109/TMTT.2007.914637',fulltext='https://backend.orbit.dtu.dk/ws/files/3801367/Shyroki.pdf',verified='DTU机构原文；严格变换涉及材料张量；均匀曲率/扭率下模式分离，不据此声称变曲率S-bend只有单一模态'),
 dict(id='R6',authors='Zongyi Li; Nikola Kovachki; Kamyar Azizzadenesheli; Burigede Liu; Kaushik Bhattacharya; Andrew Stuart; Anima Anandkumar',title='Fourier Neural Operator for Parametric Partial Differential Equations',year=2021,venue='ICLR 2021',arxiv='2010.08895',url='https://arxiv.org/abs/2010.08895',verified='作者arXiv记录与本地网络源码；Fourier核层不是单一输出截断器'),
 dict(id='R7',authors='Jiaqi Gu; Zhengqi Gao; Chenghao Feng; Hanqing Zhu; Ray T. Chen; Duane S. Boning; David Z. Pan',title='NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation',year=2022,venue='NeurIPS 2022',arxiv='2209.10098',url='https://arxiv.org/abs/2209.10098',verified='作者arXiv摘要、已有本地材料与实际源码；compact wave prior、cross-shaped Fourier模块'),
 dict(id='R8',authors='Wei Cai; Xiaoguang Li; Lizuo Liu',title='A Phase Shift Deep Neural Network for High Frequency Approximation and Wave Problems',year=2020,venue='SIAM Journal on Scientific Computing 42(5), A3285–A3312',doi='10.1137/19M1310050',arxiv='1909.11759',url='https://epubs.siam.org/doi/abs/10.1137/19M1310050',preprint='https://arxiv.org/abs/1909.11759',verified='出版社书目/摘要与作者预印本；phase shift高频学习已有先例，不等同当前几何输出载波'),
 dict(id='R9',authors='Tak Shing Au Yeung; Ka Chun Cheung; Eric T. Chung; Shubin Fu; Jianliang Qian',title='Learning rays via deep neural network in a ray-based IPDG method for high-frequency Helmholtz equations in inhomogeneous media',year=2022,venue='Journal of Computational Physics 465, 111380',doi='10.1016/j.jcp.2022.111380',arxiv='2106.10523',url='https://doi.org/10.1016/j.jcp.2022.111380',fulltext='https://users.math.msu.edu/users/jqian/papers/YeungCheungChungFuQianJCP2022.pdf',verified='作者全文；学习射线后增强数值基，使用低频场探测，信息访问条件不同于几何载波'),
]

def main():
 summary=readnums('spectral_summary.csv');pairs=readnums('spectral_paired.csv');cr=readnums('curvature_per_geometry.csv')
 cmain=[r for r in cr if r['normal_width_fraction']==0 and r['window_um']==.3 and r['gate']==.01]
 modal=readnums('closed_scalar_modal_summary.csv');phase=readnums('closed_scalar_phase_scale.csv')
 hv=read(OUT/'historical_verification.json');checks=readnums('carrier_checks.csv');ver=read(OUT/'spectrum_verification.json')
 old={r['method']:r for r in read(FORM/'test_summary.json') if r['checkpoint']=='best' and r['group']=='all'}
 def s(sub,wn,m,cy=40):return next(r for r in summary if (r['subset'],r['window'],r['method'],r['ny_count'])==(sub,wn,m,cy))
 def p(wn,base):return next(r for r in pairs if (r['subset'],r['window'],r['baseline'],r['ny_count'],r['group'])==('train64',wn,base,40,'all'))
 text='# 几何相位分解：理论、Fourier频谱与曲率模态可行性研究\n\n'
 text+='本轮为已完成主实验后的独立机制研究；全部新增计算seed2026。读取16个已看development和64个正式训练几何；正式100测试只读取冻结方法统计摘要，没有读取其场或预测，没有训练、推理、FDTD、新正式载波或超参数搜索。研究文档中的旧建议没有被当作额外执行授权。\n\n'
 text+='## 1. 执行摘要\n\n'
 text+='**理论成立，统一的“去高频解释”不成立，曲率模态升级目前证据不足（C）。**\n\n'
 text+='同载波L²等距、局部残余梯度恒等式、光滑条件下解调Helmholtz、周期H¹ Fourier尾界、参数导数和Frenet度量均有严格推导，但都不能证明网络泛化提升。最近投影不连续和v2仅C⁰的限制必须进入论述。\n\n'
 text+='80例、五表示、六种窗、四个固定矩形频带实际计算完成。64训练全域40/70参照尾能量：Linear 0.09782%，LocalTangent 0.10111%，v1 0.10419%，v2 0.10108%。v2比Linear的尾能量**高约3.33%**，尽管冻结正式模型RelL²低54.53%。统一边界窗后Direct尾能量最小。低频窄带、芯/弯曲区和v1远场跳变修复存在实证支持，但不足以把神经网络收益归因于“进入保留频带”。LocalTangent与v2的谱几乎重合。\n\n'
 text+='16开发中心线的弯曲区相对下游直引线平均场斜率增加0.01571 rad/μm，累计相位增加0.16782 rad；局部κ²单项平均R²仅0.01474，13正/3负拟合系数。独立封闭标量模态算例给出约0.044–0.189 rad的几何累计修正尺度，但最窄波导有显著外边界/分支敏感性，不能认证开放泄漏模态。**不能用已有场斜率或封闭实本征值直接替代真实弯曲β。**\n\n'
 text+='现有证据足以开始组织Optics Express方向的诚实论文，定位应是“几何输出表示的实证收益、强LocalTangent对照及其适用边界”，而非“v2显著优于强基线”或“新曲率全波理论”。目前不足以宣称已经具备更高档次算法贡献，不能保证录用。下一步只建议一个有限的开放边界恒曲率导模独立基准，先验证物理修正是否可辨识，不进入新网络训练。\n\n'
 text+='本轮读取输入'+str(hv['file_count'])+'个文件SHA-256前后完全一致。40次独立FFT重构核验最大误差'+f"{ver['max_reconstruction_formula_error']:.2e}"+'；对实际训练complex64目标的敏感性最大尾能量差'+f"{ver['max_complex64_vs128_tail_difference']:.2e}"+'。\n\n'
 text+=(OUT/'严格理论推导.md').read_text(encoding='utf-8')+'\n\n'
 text+='## 3. Fourier频谱诊断\n\n### 3.1 数据与统一条件\n\n'
 text+='诊断规则在计算前写入'+link('analysis_config.json')+'与'+link('diagnostic_freeze.json')+'。开发16与训练64分开汇总，均不是本轮机制的独立确认测试。没有加载正式测试场，也没有针对其结果改频带/ROI。\n\n'
 text+=table(['项目','实际采用'],[
  ['场','原始complex128 Ez；160×480；所有方法同一场、坐标与源规范'],
  ['幅值尺度','全部用正式训练RMS 3.822906536435209；没有方法专属幅值缩放'],
  ['冻结β','β=1.1047227347705053·eps_max+2.5607120653989606 rad/μm；读取calibration.json，不重拟合'],
  ['载波','正式features.geometry直接调用v1/v2；exp(iβd)按正式流程保存为complex64'],
  ['源','真实source_center_um，x=±11.9μm；沿用DFT激励相位，不用真值整体相位对齐'],
  ['网格','dx=dy=.05μm，Lx=24、Ly=8μm；FFT正交归一norm=ortho'],
  ['能量统计','先逐几何归一化谱、后等权平均；未用大振幅几何支配总体'],
  ['频带（y/x计数）','10/18、20/35、40/70、60/105；保留严格|m|<计数'],
  ['径向截止','4、8、12、16、24、40 rad/μm；96等宽径向bins，附CDF与尾部'],
  ['训练载波核对','64/64几何、五载波与冻结cache逐元素完全一致'],
 ])+'\n'
 text+='代码依据：[正式公共接口](D:/game/research_sbend_formal/features.py:6)、[v1最近投影](D:/game/research_sbend_path/path_geometry.py:32)、[v1距离](D:/game/research_sbend_path/path_geometry.py:63)、[v2正核/混合](D:/game/research_sbend_path_v2/path_v2.py:25)、[训练β标定](D:/game/research_sbend_formal/calibration.py:20)。实际Fourier层见[逐轴rFFT实现](D:/game/references/NeurOLight-source/NeurOLight-main/core/models/layers/neurolight_conv2d.py:60)。\n\n'
 text+=f"complex64载波最大单位模浮点误差{max(r['carrier_modulus_error'] for r in checks):.3e}；解调总能量相对差最大{max(r['demod_energy_relative_error'] for r in checks):.3e}。归一化功率谱分母只涉及此舍入量级差异，不是为某方法另行调尺度。独立核验8个宽度各一个训练几何，真实逆FFT重构与sqrt(尾能量)一致；使用实际训练complex64场也不改变结论。\n\n"
 text+='### 3.2 窗函数与区域定义\n\n'
 text+='full=完整原矩形；tukey=两个轴α=.2的Tukey窗相乘，边缘相位/源附近能量被抑制。其余区域沿用正式几何ROI：芯距中心线≤w/2；bend另限官方弯曲x区间；halo为w/2到w/2+.5μm；far为其外。先将0/1区域用σ=.10μm（2像素）高斯滤波，再乘同一Tukey窗；窗作用于场幅值，能量权重为窗平方。未用裸硬mask的谱作结论。平滑窗混合邻近区域并引入共同窗谱，结果是“加权场的谱”，不是局域模态或唯一空间谱。\n\n'
 text+=table(['窗','训练平均原场能量保留比例'],[[wn,f"{s('train64',wn,'path_v2')['window_energy_fraction']*100:.4f}%"] for wn in CONFIG['windows']])+'\n'
 text+='软窗不是互斥能量分割，不能把这几行简单相加。远场软窗只保留约0.4895%的原始能量，其相对尾部可以很大；应同时看此能量份额，不能让弱场相对指标主导器件功能判断。硬区域/软区域以及下游端口功率也不是同一概念。\n\n'
 text+='### 3.3 全部固定频带：保留有利与不利排序\n\n'
 text+=table(['表示']+[f'{cy}/{cx}尾能量%' for cy,cx in CONFIG['rectangles_y_x_counts']],
   [[l]+[f"{s('train64','full',m,cy)['tail_fraction']*100:.6f}" for cy,cx in CONFIG['rectangles_y_x_counts']] for m,l in zip(METHODS,LABELS)])+'\n'
 text+='窄带10/18下，v2相对Linear尾能量约减49.1%，支持把主要传播振荡移向低频；但20/35下v2略差，40/70下更差，60/105几乎无差。Raw在40/70内本已保留99.8777%能量，Direct的大预测误差不能简单解释成主要真值能量在网络参照频带外。不能只展示10/18。\n\n'
 text+=table(['方法','训练full尾能量%','平均重构RelL²','训练Tukey尾能量%','平均重构RelL²','开发full尾能量%'],
   [[l,f"{s('train64','full',m)['tail_fraction']*100:.6f}",f"{s('train64','full',m)['reconstruction_rel_l2']:.6f}",f"{s('train64','tukey',m)['tail_fraction']*100:.6f}",f"{s('train64','tukey',m)['reconstruction_rel_l2']:.6f}",f"{s('development16','full',m)['tail_fraction']*100:.6f}"] for m,l in zip(METHODS,LABELS)])+'\n'
 text+='所有重构误差先逐几何sqrt再平均，不是总体平均尾能量再开方。加窗后误差是相对加窗场，不能作为原全域物理场误差。未加窗周期拼接能量比约.00207，加窗后边界为零。full→Tukey使Direct尾部降约98.7%，说明此频带Raw高频很受裁剪边界影响；加窗后Raw反而明显优于解调场，符合相位调制可能扩展部分离散谱的事实。尚未作唯一的辐射/反射物理归因。\n\n'
 text+=fig('cutoff_and_window_effects')+'\n'+fig('mean_2d_power')+'\n'
 text+='### 3.4 区域、方向、v1跳变与v2修复\n\n'
 text+=table(['40/70尾能量%']+LABELS,
   [[wn]+[f"{s('train64',wn,m)['tail_fraction']*100:.6f}" for m in METHODS] for wn in ['core_soft','bend_soft','halo_soft','far_soft']])+'\n'
 text+='v2对Linear在芯、弯曲芯、halo的配对改善分别64/64、64/64、63/64；远场则0/64。这支持有限区域收益，否定全域一致变简单。v2对LocalTangent在halo尾能量还略增约0.37%，平均差虽小但方向不利，不能省略。\n\n'
 text+=table(['训练配对，正差支持v2','对照−v2尾能量差','95%几何bootstrap区间','改善几何'],
   [[wn+' / '+base,f"{p(wn,base)['mean_tail_difference']:.7g}",f"[{p(wn,base)['ci_lower']:.7g}, {p(wn,base)['ci_upper']:.7g}]",f"{int(p(wn,base)['improved'])}/64"] for wn,base in [('full','linear'),('full','local_tangent'),('full','path_v1'),('tukey','path_v1'),('core_soft','linear'),('far_soft','path_v1')]])+'\n'
 text+='v1远场软窗尾部2.1833%→v2的1.2700%，降低约41.8%；统一Tukey全域v1 .020798%→v2 .016792%，降低约19.3%。这与修复跳变一致，是局部机制证据，但没有分离核平均/过渡与其他相位延拓效应的因果贡献。v1在10/18最窄带甚至略优于v2，某些几何40/70也更好；连续性并不保证所有频带最优。v2与LocalTangent全域尾能量相差仅2.84×10⁻⁷，不能据此预测可靠学习优势。\n\n'
 text+='方向上，full40/70 v2相对Linear左源仅13/32、右源11/32尾能量改善，两方向平均都恶化；对v1左24/32、右27/32改善。逐轴尾部也已保存：full训练Linear的x/y尾为0.000962491/0.000016126，v2为0.000995088/0.000016283；此频带主要由x尾主导。v1更大的y尾1.9922×10⁻⁵降到v2的1.6283×10⁻⁵，与去除部分横向跳变相容。\n\n'
 text+=fig('directional_marginals')+'\n'+fig('radial_tail_curves')+'\n'+fig('per_geometry_distributions')+'\n'
 text+='预先规定保留相对Linear及v1各自最有利/最不利几何，全部来自训练集：\n\n'
 text+=table(['对照','极值类型','几何','对照尾能量','v2尾能量'],[[r['baseline'],r['label'],r['geometry_id'],f"{r['baseline_tail']:.7f}",f"{r['v2_tail']:.7f}"] for r in read(OUT/'examples.json')])+'\n'
 text+='这些是说明分布极端情况的例子，不能当作新增确认集。没有据其改变任何窗、公式或参数。\n\n'+fig('positive_and_negative_examples')+'\n'
 text+='### 3.5 与冻结学习结果及Y-Branch负面机制的关系\n\n'
 text+=table(['方法','训练full40/70尾能量%','历史best测试RelL²'],[[l,f"{s('train64','full',m)['tail_fraction']*100:.6f}",f"{old[m]['rel_l2']:.6f}"] for m,l in zip(METHODS,LABELS)])+'\n'
 text+='仅比较五个方法均值，full10/18频谱与误差Spearman排序相关0.9，20/35为0.1，40/70为0.3；统一Tukey与其他区域的全部排序见机器可读JSON。n=5的方法级相关没有足够独立自由度，不能作显著性或因果证据；三个几何载波n=3更不能推断稳定规律。训练场谱与测试误差也不是同批几何的逐样本因果关系。\n\n'
 text+='**明确反例：** Linear在full40/70比v2谱尾更小，学习却差很多；统一Tukey下Direct谱尾更小，学习最差。v2比v1远场谱更简单，但正式远场RelL²为1.000711，高于v1的.997092，原冻结统计65/100更差；v2-v1总体主差区间跨零，last也不支持稳定优势。改善表示指标不保证改善学习。\n\n'
 text+='另一个看似矛盾的实测：训练全域内部差分J Direct83.1068、Linear16.2030、LocalTangent10.8187、v1 10.8334、v2 10.8134 μm⁻²，几何表示能降低**总梯度二阶矩**，却没有降低某个特定高频带尾部。两者测量不同，Fourier上界并未给出逐频带排序。\n\n'
 text+='Y-Branch既有[M1/M2/M3机制报告](D:/game/research_mechanism/M1_M2_M3机制分析报告.md)中，训练导波J Linear10.3161→Path17.8985；旧E2非seam二维J上升76.25%，r8 POD oracle误差.492985→.534896，尽管原网络Path更好。本轮没有重读旧测试场，旧数字仅作历史机制证据。两家族结果共同否定“处处二维更平滑/全局更低秩”的通用叙事。\n\n'
 text+='**对“多少学习收益可由频谱支持”的回答：** 可支持低频聚集、芯/弯曲区局部表示改善，以及v1跳变带来的部分远场尾部；不能把54.53%网络误差改善中某个百分比归因于它们，没有定义可识别的因果分解。对于v2对LocalTangent仅0.19%的best增益，频谱差异太小、检查点/方向不利结果仍在，不能形成强机制结论。跨几何相位对齐、固定2400步的优化行为、隐层特征耦合是合理候选解释，尚非已验证因果机制。\n\n'
 text+='## 4. 曲率相关物理诊断\n\n### 4.1 传统物理与候选积分的适用条件\n\n'
 text+=r'''弯曲传播常数不是新概念。[Heiblum与Harris（1975）](https://doi.org/10.1109/JQE.1975.1068563)用共形变换处理弯曲波导及损耗；[Takuma等（1981）](https://doi.org/10.1364/AO.20.002291)的出版社摘要明确涉及均匀曲率slab场与传播常数的曲率级数，但本轮未取得其受限全文，不能借摘要直接移植其系数。[Shyroki（2008）](https://doi.org/10.1109/TMTT.2007.914637)给出更完整的等效张量建模，提示简单修改一个标量折射率不是任意情况下的严格矢量理论。

直波导β₀(w,λ)可由同材料与极化的slab解析色散、数值横向本征问题或直波导MPB获得。当前数据是**二维有效折射率模型**，core index由官方width/λ表转换，不是真实3D硅截面的完整矢量β。宽度改变时eps_core也改变，因此宽度和材料在本样本中并非独立物理因素。

局域弯曲β_eff需定义中心线参考半径、极化、模式分支及开放辐射条件。均匀圆弯可用柱坐标出射Hankel匹配、柱PML横向本征求解或严格等效材料张量建模；若传播按exp(iβs−iωt)定义，Imβ>0对应衰减。直引线MPB输出β不等于弯曲模态β。把当前直波导MPB模式直接加κ标签不是曲率本征计算。

候选

$$
\phi_{mode}(s)=\int_{s_{src}}^s\operatorname{Re}\beta_{eff}(w(s'),\kappa(s'),\lambda)\,ds'
$$

仅在主导局域模式可辨识、几何变化相对模态拍长足够慢、耦合和损耗可控时有意义。[Johnson等（2002）](https://doi.org/10.1103/PhysRevE.66.066608)的连续耦合模与绝热理论要求全程传播/导引及相应谱分离；不是任意紧弯、辐射连续谱的无条件结论。传统模态耦合已见[Snyder（1972）](https://doi.org/10.1364/JOSA.62.001267)。本征模相位规范变化还会给局域基系数/连接项，积分β并不总是场的完整总相位。开放模态的虚部和辐射振幅变化若留在包络中，则实相位载波仍单位模；若一起纳入载波，L²等距消失。

在横向镜像对称材料/边界、固定中心线参考下，n↦−n将κ↔−κ，若对应同一简单模态分支，则β(κ)=β(−κ)。**只有再加解析性**才得β=β₀+c₂κ²+O(κ⁴)。不对称背景、改变参考半径、简并/模式交叉、泄漏阈值都可能破坏这一简化；偶对称不自动保证实解析或c₂为正。3D应力双折射也没有包含在本二维模型中。

受限封闭标量问题可说明系数如何独立算出。恒κ时从Frenet标量方程令E=ψ(n)exp(iβs)，得到

$$
(h\psi')'+k_0^2\varepsilon h\psi=\beta^2\psi/h,\quad h=1-\kappa n.
$$

有限区间Dirichlet下是实自伴广义本征问题。这里算子A_op=∂n(h∂n)+k₀²εh，B=1/h、C=B^(-1/2)A_op B^(-1/2)，展开C=C₀+κC₁+κ²C₂+…，若λ₀=β₀²简单隔离且{u_j}为归一基，则

$$
\lambda_2=\langle u_0,C_2u_0\rangle+
\sum_{j\ne0}\frac{|\langle u_j,C_1u_0\rangle|^2}{\lambda_0-\lambda_j},\qquad
c_2=\lambda_2/(2\beta_0).
$$

该公式是对指定解析封闭本征族的标准扰动论，不是对开放弯曲模态的普遍认证。后者要处理出射、复谱及可能非解析的辐射隧穿效应。

即便沿程β更准确，二维延拓仍需处理横向模态形变、反射及多波前。将某一积分沿中心线再扩展到整个包层并非物理唯一解；不能以改β自动解决v1/v2原有二维波前局限。Eikonal/WKB处理的高频射线极限，与横向约束导模本征值不同，不能用k₀√eps_core代替所有导波相位。

'''
 text+='### 4.2 同次FDTD场的观测方法与精度边界\n\n'
 text+='使用16开发样本保存的完整Ez/Hx/Hy及直端口mode_Ez/H、alpha±、S_native和result.json元数据。中心线沿左→右s采样ds=.05μm，传播方向左+、右−；选n/w=−.25、0、+.25，复数双线性插值。场相位unwrap后用振幅平方权重的局部线性拟合估计方向修正后的dθ/ds。几何切向角先用固定σ=.10μm平滑，再差分得离散κ、κ′；不拟合任何载波。\n\n'
 text+='主诊断预定窗口.30μm、幅值>各横线最大值1%，弯曲两端各排.30μm；直引线基准用下游距弯曲>.75μm且|x|<10.5μm的斜率中位数。另保留.15/.60名义窗口和5%门控；.15名义窗因奇数采样及round实际端到端跨度为.10μm，.30/.60实际为.30/.60，已在curvature_sensitivity.csv明确列出，未看结果后改窗。两种幅值门控在这些芯内横线上保留相同点，因此结果相同，不能声称检验了近零点稳健性。\n\n'
 text+='n≠0时这里估计的是对**中心线s参数**的相位斜率，不是除以(1−κn)后的物理切向梯度；横向对照用于检查同一波前的变形，不能把不同n的值当成三个独立β_eff。有限窗口、插值和FDTD网格可引入伪波动，尤其直接点态相位导数。\n\n'
 text+=table(['主诊断16几何（逐几何等权）','结果'],[
  ['直引线场β范围','7.49676–9.44618 rad/μm'],
  ['弯曲平均β−下游直引线β','0.015709 rad/μm；95%几何bootstrap [0.010401,0.020726]'],
  ['弯曲局部偏差RMS','均值0.129947 rad/μm；范围0.054940–0.195545'],
  ['相对冻结β的局部残余RMS','均值0.129934 rad/μm'],
  ['累计弯曲场相位增加','均值0.167821 rad；范围−0.069218到0.293913；CI [0.111424,0.217327]'],
  ['相对冻结β累计弯曲残余','均值0.080560 rad；范围−0.233983到0.372425'],
  ['局部corr(κ,δβ) / corr(κ²,δβ)','几何均值−0.00790 / 0.07863'],
  ['局部corr(方向修正κ′,δβ)','均值−0.00780'],
  ['描述性δβ~常数+c₂κ²','c₂均值.59570μm；范围−.63917到1.76328；13正3负'],
  ['描述性拟合R²','均值.01474，中位数.00272，最大.11732'],
  ['直引线场β−MPB β','均值.10051 rad/μm，比观测弯曲平均偏差更大'],
  ['直引线场β−冻结β RMS','.01865 rad/μm；与待辨识平均曲率信号相当'],
 ])+'\n'
 text+='bootstrap以16个几何为单位，5000次PCG64(2026)，并非把大量s点当独立样本；区间条件于当前开发集和诊断规则，不是独立模态物理认证。每几何拟合系数只供描述相关性，**从未写入载波、预测或正式β**。三例负c₂是sbend_929112034289、sbend_061b0b035103、sbend_e5675321b865，即便这三例的弯曲平均相位增加仍为正；均值正不等于局部κ²规律正确。sbend_131384a21641弯曲平均偏差为负，也保留。\n\n'
 text+='几何累计∫κ²ds与累计相位增加的相关为.723（左8例.782、右8例.897）；对均值偏差按宽度组中心化后与1/R²的相关.862。**这是比局部拟合更积极的关联线索**，但部分宽度仅一个样本、其他宽度/材料/偏移/方向混杂，不能把它当作已分离出的c₂或统计因果。左右分别为不同几何，不是同一几何双向/镜像配对，不能声称由此验证互易或方向等变物理。\n\n'
 text+='窗口敏感性也不能省略：中心线局部偏差RMS随实际.10/.30/.60μm为.28491/.12995/.05049 rad/μm，κ²相关均值.03598/.07863/.18885，累计相位均值.18334/.16782/.20166 rad；横向±w/4在.30窗的偏差RMS约.2117/.2115，高于中心，累计差分别.08181/.35121rad。平滑改变局部估计与端部影响，没有单一“真实β曲线”被本诊断识别出来。\n\n'
 text+='完整E/H的额外观测：用时间平均Poynting Sx=−Re(Ez·conj(Hy))/2、Sy=Re(Ez·conj(Hx))/2，在同一中心线采样。沿传播方向的流在所检查点未反向，流向相对切线的振幅加权RMS平均2.03°（范围.95°–4.57°），说明主能量大致沿几何引导，但不证明单一局域模态。已有Ez端口代理对原生Meep非输入端误差均值.000859、最大.001694，仍是直端口代理；E/H插值重叠也有其数值误差，未以S标签校准任何相位。完整逐例数据见existing_modal_checks.csv和EH_local_flux_diagnostic.csv。\n\n'
 text+=fig('curvature_evidence_and_limits')+'\n'+fig('curvature_local_profiles')+'\n'
 text+='### 4.3 独立封闭标量模态可行性计算（本轮实际完成）\n\n'
 text+='为了不把FDTD相位斜率冒充本征模β，另做了**288个离线一维封闭标量本征算例**：8个实际宽度/有效材料，κ=0,.05,.10,.20,.25,−.20μm⁻¹，横向半宽1.5/2.0/2.5μm，步长.01/.005μm。使用上述广义本征方程、Dirichlet边界、守恒面通量差分和芯边界单元体积分数。由解析直波导基模作参照，在最高12个候选中取最大横向重叠，保存选择分支而非假定最大本征值总是芯模。没有运行Meep/MPB或新的FDTD。此计算不产生正式二维相位载波。\n\n'
 text+=table(['w μm','Δβ(κ=.20) rad/μm','c₂(κ=.05) μm','κ=.20外盒Δβ范围','最小直模重叠'],
   [[f"{r['width_um']:.3f}",f"{r['delta_beta_k02']:.6f}",f"{r['c2_k005']:.4f}",f"{r['box_delta_beta_range']:.2e}",f"{r['min_overlap']:.4f}"] for r in modal])+'\n'
 text+='表中Δβ/c₂/重叠主值用细网格、最大盒；外盒敏感性保留全部三个盒。±.20镜像β误差≤1.8×10⁻¹⁵，直波导与解析slab相对误差约0.8–1.5×10⁻⁵；但这些只是封闭问题的正确性/收敛检查，不认证开放模态。κ=.20网格绝对β差约.00012–.00054rad/μm。宽.4μm的外盒Δβ变化达.00935rad/μm，约占该盒主值的26.5%，最小直模重叠.666；宽.425在更强曲率也有重叠降至.544。外侧盒模/反射与芯模可能混合，不能挑一个盒结果当真值。其余较宽样本在本盒检查内更稳定，也仍缺出射损耗和开放边界认证。\n\n'
 text+=f"小κ=.05的封闭c₂约.496–.789μm；把这些**独立计算的尺度**乘开发几何∫κ²ds，得到累计相位尺度{min(r['closed_model_phase_scale_rad'] for r in phase):.5f}–{max(r['closed_model_phase_scale_rad'] for r in phase):.5f}rad，平均{np.mean([r['closed_model_phase_scale_rad'] for r in phase]):.5f}rad。它是量级估计，非新载波，也没有用16个真值场拟合该c₂；部分几何峰值κ超出.25表格，不能把小曲率展开自动视为可靠外推。\n\n"
 text+='若误差恰为单一模式的均匀纯相位偏差δ，RelL²=2|sin(δ/2)|，.1rad约为10%；所以本量级并非先验可忽略。但真实场有横向变化/反射，网络误差也非纯相位，不能据此声称新载波可回收10%或解释当前.153的预测误差。\n\n'
 text+='官方几何实际是256点Euler折线的旋转Block并集，短块长度为段长1.05倍（[polyline_to_blocks](D:/game/PIC-Flow-新研究资料/PIC-Flow-main/PIC-Flow-main/FDTD/sbend/sbend.py:112)）；还有.05μm网格、有效材料和亚像素界面处理。当前理想恒曲率slab与其并不严格同构。直场β、直端口MPB、连续slabβ之间的.05–.13rad/μm量级差异来源可能包括有限差分色散、栅格/材料采样、有限监测域和局部估计，**本轮没有把它们唯一归因于某项**。应先解决独立物理参照，不能用同量级的数值差异论证新曲率算法正确。\n\n'
 text+='### 4.4 可独立计算的信息、泄漏风险与复杂度\n\n'
 text+='几何中心线、w、κ、波长、有效材料表、源位置，以及独立模式求解器的β/模态形状，原则上均无需查询真实场；独立模式表可在部署前固定。相反，从每个查询真值Ez解包相位再拟合β_eff、用真实S校正积分常数、在正式测试误差上挑c₂都属信息泄漏，本轮没有做。\n\n'
 text+='一旦有可信局域模态表，沿程积分成本O(Ns)，但曲率表生成需处理开放复谱/模式跟踪/收敛；再叠加二维延拓成本并不能解决多模式和辐射。现有LocalTangent/v2在训练测试表现几乎相当，引入一套模态求解器必须以明确物理/学习收益抵偿，不能只靠相位图更好看。本轮一维288例计算为秒级受限可行性算例，不能外推为完整矢量开放模态工具的成本。\n\n'
 text+='### 4.5 文献新颖性审计\n\n'
 text+='曲率β、局域模态、绝热与耦合模都已有长期传统，不能本身作为新算法首创。[PhaseDNN（Cai等，2020）](https://epubs.siam.org/doi/abs/10.1137/19M1310050)已用相位移降低高频学习负担；[NeurOLight（Gu等，2022）](https://arxiv.org/abs/2209.10098)已有紧凑wave prior与光子器件算子；[Au Yeung等（2022）](https://doi.org/10.1016/j.jcp.2022.111380)已把学习射线方向用于振荡数值基，其需低频场探测、与本任务几何信息条件不同。三者都不能被描述成与本方法完全同一构造，但说明“物理相位+学习”的广泛想法不是空白。\n\n'
 text+='本轮检索覆盖curved-waveguide modes、curvature propagation expansion、phase-shift learning、photonic neural operator、ray-based neural Helmholtz等关键词及这些原始来源；没有完成所有数据库、专利和2026全部预印本的穷尽式检索。**几何条件的模态/曲率输出相位组合是否具有具体实现新颖性，仍未确认；未搜索到完全相同式子不能写成首次。**\n\n'
 text+='## 5. 对当前论文的影响\n\n'
 text+=table(['适合位置','建议放入的证据','应避免的表述'],[
  ['正文方法','同载波L²等距、残余梯度式及一个简短条件说明；几何/源信息与训练β访问范围','证明了泛化/优化稳定；声称这些标准恒等式为新数学定理'],
  ['正文结果','两器件家族冻结模型结果；S-bend五组和LocalTangent强基线；低带/40/70/加窗反例组合图','只画有利频带；把v2微小优势当作可靠算法突破'],
  ['正文讨论','频谱无法充分解释54.53%改善，v2连续性与学习价值分开','高频全部消除；神经算子就是线性截断器'],
  ['补充材料','完整证明、弱形式/Frenet边界、全窗全频带逐几何CSV、POD负例、反例与哈希','去掉不利区域或不显著对照'],
  ['补充/未来工作','曲率观测、封闭标量可行性和外盒失败案例','已开发更优曲率载波、已认证弯曲Maxwell传播常数'],
  ['器件功能','非输入端代理和同次Meep闭环的误差边界','网络已预测完整严格S、反射回波损耗已认证'],
 ])+'\n'
 text+='**Optics Express判断：可以进入论文写作，但不是本轮就证明“稳录”。** 当前有Y-Branch与新S-bend、冻结协议、共同输入/训练预算、端口代理物理闭环、强基线和诚实负面机制，足以构成一篇范围明确的光场代理方法研究。相位变换本身与传统物理并不新，最强几何基线几乎追平Path，训练仅seed2026且2400步，因此说服力应来自具体表示/跨几何应用、复现实证和适用边界；目前不能宣称普遍优于LocalTangent、跨种子稳健或充分收敛后仍有优势。\n\n'
 text+='该判断是对本地证据的研究评估，不是编辑承诺。[Optics Express官方范围](https://opg.optica.org/content/journal/about/item/oe/about)强调光学与光子学的科学/技术创新；本报告不以影响因子或期刊档次替代创新审计。现有材料可完成一份OE方向初稿，是否达到录用要求还取决于完整文献对照、论证及评审。没有证据支持为了投稿级别立即投入复杂新相位载波。\n\n'
 text+='## 6. 对下一代算法的判断\n\n'
 text+='**选择C：证据不足。置信度：对“目前不能认证曲率模态收益”的判断中高；对“曲率最终能否改善网络”的判断低。** 理论上存在独立计算路径，封闭模型和几何累计相关提示约0.1rad量级值得一次有限物理排查；但局部R²低、反例和方向/窗口差异、直参照偏差及开放边界缺失，尚不足以选择A“值得开发完整算法”，也不足以断言B“物理效应可忽略”。停止把现有16场拟合成新载波。\n\n'
 text+='**唯一建议的下一项实验：独立开放边界恒曲率导模基准。** 固定两个宽度（.40、.50μm）与本数据有效材料，波长1.55μm，考察直波导和κ=±.10、±.20μm⁻¹，共10个结构条件；采用柱坐标出射Hankel匹配或经过验证的柱PML模式求解，β以中心线弧长为参考，进行至少两级网格/外边界收敛与镜像分支跟踪，同时保存Reβ、Imβ及模态。直基准必须与该求解器解析slab一致，而非强行匹配当前FDTD场；分开报告FDTD离散色散。不要用正式测试场拟合、不要更改当前载波，也不训练网络。\n\n'
 text+='该单项基准的目的，是先区分真曲率β、泄漏损耗和封闭盒伪效应。建议在实施前固定接受规则：ΔReβ在域/网格变化下相对差≤10%且绝对差≤.002rad/μm，镜像/模式身份一致；用其独立几何积分只估量相位尺度，不向查询真值拟合参数。预计10条件×2分辨率，串行CPU约20–60分钟求解及半日至一天实现/核验；这是待实测的预算，不是已验证运行时间。本轮未自动启动该实验。若过不了物理收敛，停止曲率算法扩展并完成当前论文。\n\n'
 text+='## 8. 文献与复现材料\n\n### 8.1 核查文献\n\n'
 for r in REFERENCES:
  text+=f"- **{r['id']}** {r['authors']}. [{r['title']}]({r['url']}). {r['venue']}, {r['year']}. "+(f"DOI: {r['doi']}. " if r.get('doi') else f"arXiv:{r['arxiv']}. ")+r['verified']+'。'
  if r.get('fulltext'):text+=f" [作者/机构全文]({r['fulltext']})。"
  text+='\n'
 text+='\nR2/R4只有出版社摘要，未索要付费全文；现有推导不依赖无法核查的具体公式。R7的arXiv HTML路径返回404，已采用作者arXiv记录和本地原文/源码，不需要用户补文件。未更改网络设置，没有长时间重试。\n\n'
 text+='### 8.2 文件与执行\n\n'
 text+=table(['材料','作用'],[
  [link('analyze_phase.py'),'80例冻结载波FFT、理论数值核验、16例曲率分析；只读历史'],
  [link('modal_feasibility.py'),'288个封闭标量模式与相位量级算例；非正式载波'],
  [link('summarize_and_plot.py'),'独立逆FFT核验、E/H流向、敏感性汇总、8套PNG/PDF图'],
  [link('analysis_config.json')+' / '+link('diagnostic_freeze.json'),'诊断前规则、脚本/配置哈希与允许样本清单'],
  [link('spectral_per_geometry.csv')+' / '+link('spectral_summary.csv'),'9600条逐几何/逐方法/逐窗/逐频带指标和总体摘要'],
  [link('spectral_paired.csv'),'所有窗/频带/左右方向配对及几何bootstrap'],
  ['mean_spectra/*.npz / '+link('mean_radial_spectra.csv'),'60份平均二维谱、x/y边际、径向功率与CDF'],
  [link('frequency_axes.npz')+' / '+link('examples.json'),'物理波数轴、固定极值实例索引（正负均存）'],
  [link('ordering_vs_saved_predictions.json'),'频谱与冻结预测均值排序；不读取测试场'],
  [link('curvature_per_geometry.csv')+' / '+link('curvature_profiles.csv'),'288条方向/横向/窗口/门控指标与中心线局部曲线'],
  [link('curvature_sensitivity.csv')+' / '+link('curvature_geometry_bootstrap.csv'),'实际窗口跨度、门控敏感性、几何统计边界'],
  [link('existing_modal_checks.csv')+' / '+link('EH_local_flux_diagnostic.csv'),'开发样本完整E/H与直端口标签的实际核查'],
  [link('closed_scalar_modal_grid.csv')+' / '+link('closed_scalar_modal_summary.csv'),'封闭模态完整288例、镜像/网格/外盒检查与失败'],
  [link('closed_scalar_phase_scale.csv')+' / '+link('modal_scope.json'),'独立封闭模型尺度与明确的非认证范围'],
  [link('input_hashes_before.json')+' / '+link('historical_verification.json'),'311个输入前后哈希核对；正式代码冻结校验通过'],
  [link('outputs_manifest.json')+' / '+link('references.json'),'新增复现材料SHA-256及机器可读文献'],
 ])+'\n'
 text+='在WSL可复现：\n\n```bash\n/home/cudauser/.conda/envs/picflow-meep/bin/python -B /mnt/d/game/research_phase_theory/analyze_phase.py\n/home/cudauser/.conda/envs/picflow-meep/bin/python -B /mnt/d/game/research_phase_theory/modal_feasibility.py\n/home/cudauser/.conda/envs/picflow-meep/bin/python -B /mnt/d/game/research_phase_theory/summarize_and_plot.py\n/home/cudauser/.conda/envs/picflow-meep/bin/python -B /mnt/d/game/research_phase_theory/build_report.py\n```\n\n'
 text+='上述为各阶段入口；analysis已完成时会拒绝覆盖。完整复现使用replay.py --out指定新的、尚不存在的结果目录，自动复制源码和理论材料并顺序运行；不删除或改写本次冻结证据，也不改历史研究目录。环境Python3.11/NumPy2.4.6/SciPy1.17.1；未安装依赖。图中文字用英文轴标以避免字体缺字，中文解释在本报告。图同时保存PDF以供论文排版。初次作图发生相对examples.json路径错误，已只修正新作图脚本的输出目录路径，没有改变任何数值规则，修复记在postprocess_complete.json。报告拼装的字符串换行错误也已在本目录修复，数值分析脚本和冻结配置未变。\n\n'
 text+='对旧结果的保护核查覆盖本轮实际读取的80个sample及相关顶层代码/报告/统计（共311文件），不声称重新哈希全部历史权重和预测；本轮根本未访问权重或运行模型。完整正式test的旧一次评价地位保持不变；本轮所有机制解释均明确事后分析。\n'
 write('references.json',REFERENCES)
 (OUT/REPORT).write_text(text,encoding='utf-8')
 print(f'Wrote {REPORT}: {len(text)} characters',flush=True)

if __name__=='__main__':main()
