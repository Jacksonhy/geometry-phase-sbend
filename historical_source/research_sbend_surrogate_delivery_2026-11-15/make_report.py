"""Assemble user-facing delivery; all figures/tables derived from saved results."""
import csv,math,time
from pathlib import Path
import numpy as np
from predict import R,BASE,METHODS,SEEDS,read,save,digest
LABELS=dict(linear='Linear',arc_only='ArcOnly',local_tangent='LocalTangent',path_v2='Path v2')
def rows(name):
 with (R/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fnum(x,n=4):return f'{float(x):.{n}f}'
def write_report():
 c=read(R/'cost_analysis.json');summary=rows('main_table_three_seed_summary.csv');per=rows('main_table_per_seed.csv');pairs=rows('paired_differences.csv');rates=rows('illustrative_tolerance_rates.csv');lt=[r for r in c['online'] if r['method']=='local_tangent'];get=lambda name,m:next(float(r['mean_of_seed_means']) for r in summary if r['endpoint']==name and r['method']==m)
 main='|指标（72几何，三种子均值）|Linear|ArcOnly|LocalTangent|Path v2|\n|---|---:|---:|---:|---:|\n'
 for name,label in [('whole_rel_l2','全域复场 RelL2'),('bend_rel_l2','弯曲芯区复场 RelL2'),('bend_phase_deg','弯曲芯区相位 WMAE /°'),('normal_phase_deg','法向相对相位 WMAE /°'),('cross_rel_l2','横向复交叉项 RelL2'),('bend_amplitude_rel_l2','弯曲芯区幅度 RelL2')]:main+='|'+label+'|'+'|'.join(fnum(get(name,m)) for m in METHODS)+'|\n'
 seedtable='|种子|全域复场 Δ|95%几何配对区间|弯曲相位 Δ/°|法向相位 Δ/°|\n|---|---:|---|---:|---:|\n'
 for s in SEEDS:
  ps={r['endpoint']:r for r in pairs if int(r['seed'])==s and r['baseline']=='arc_only'};q=ps['whole_rel_l2'];seedtable+=f"|{s}|{float(q['delta']):+.5f}|[{float(q['ci95_low']):+.5f}, {float(q['ci95_high']):+.5f}]|{float(ps['bend_phase_deg']['delta']):+.3f}|{float(ps['normal_phase_deg']['delta']):+.3f}|\n"
 onlinetable='|使用方式|完整查询秒/几何|相对存档 FDTD 耗时比|重建离线成本后的回本查询数|\n|---|---:|---:|---:|\n'
 modes=['cold_single','resident_single','batch72_b2'];names=['独立进程冷启动单例','GPU模型常驻单例','72例批量，batch_size=2']
 for mode,label in zip(modes,names):
  q=next(r for r in lt if r['mode']==mode);b=next(r for r in c['break_even'] if r['method']=='local_tangent' and r['mode']==mode and r['scenario']=='historical_subtotal_plus_current_overhead_replay');count=str(b['minimum_queries']) if 'batch' not in mode else f"{b['minimum_queries']}（整批按216例）";onlinetable+=f"|{label}|{q['query_seconds']:.3f}|{q['online_speed_ratio']:.2f}×|{count}|\n"
 ratetable='|示例全域复场容差|Linear|ArcOnly|LocalTangent|Path v2|\n|---|---:|---:|---:|---:|\n'
 for t in [.1,.15,.2]:
  vals=[np.mean([float(r['fraction']) for r in rates if r['method']==m and r['criterion']=='whole_rel_l2' and float(r['threshold'])==t]) for m in METHODS];ratetable+=f'|RelL2 ≤ {t:.2f}|'+'|'.join(f'{100*v:.1f}%' for v in vals)+'|\n'
 qlt=next(q for q in lt if q['mode']=='batch72_b2');offline=qlt['reconstructed_offline_seconds'];train=offline-c['shared_offline_measured_subtotal_seconds']-c['reconstructed_missing_overhead_seconds'];gain=100*(1-get('whole_rel_l2','local_tangent')/get('whole_rel_l2','arc_only'))
 text=f'''# S-bend复光场代理模型：独立预测、成本与72几何证据

本轮三项交付已完成：仅凭几何和单端口激励即可预测；完整工作流成本及回本情景可追溯；72个前瞻盲测几何的全场、局部波前、尾部误差和场图已整合。没有新训练或新FDTD时域传播，没有改动冻结权重、校准和历史预测。

当前最稳妥定位：**具有明确离散参考与参数范围的复光场代理预测，以及受控输出表示研究**。在本机与既有FDTD工作流比较，确有端到端查询耗时优势；这尚不足以称为在相同物理精度下普遍替代FDTD。

## 1. 无需真场文件的预测入口

`predict.py`只接收几何JSON、方法和种子。参数为宽度0.400–0.575 μm、offset2.000–5.500 μm、R_min3.000–7.000 μm，沿既有0.025 μm参数格点，并检查原构造器的几何容纳条件。固定1.55 μm、二维有效折射率材料、原Gaussian EigenModeSource，单端口1或2，原resolution20输出网格160×480、dx=dy=0.05 μm，物理区域24×8 μm。

流程：几何/源 → Meep材料初始化（仿真时间始终为0）→ eps、源掩膜、实际坐标 → 原12个实特征与载波 → 冻结神经算子 → 原始约定的复Ez。保留同一256点旋转Block几何，没有改成解析二值掩膜。Meep材料初始化是依赖与成本；它不是时域求解。

部署脚本禁止`Simulation.run/step`，并由文件访问审计拒绝`sample.npz`和历史data/cache/predictions路径。核验/绘图脚本另行读取历史真值。**无需查询几何的真场文件，不等于无需原训练权重、训练集校准、原几何/网络代码和现有Python环境。** 这些依赖都列在`protected_inputs.json`和读取日志中。

72个新生成输入的eps、源掩膜、坐标、源位置、公共网络输入和四载波与历史缓存逐值一致。四方法×三种子共12份重放预测与原72几何预测全部逐值一致，最大差为0。独立标量检查另核验3几何×4方法的全域RelL2；非法端口/超范围参数拒绝、真场路径拦截均通过。

输出NPZ含`prediction[N,160,480]`、`x_um/y_um`、几何ID、端口、方法、种子、波长和源约定；配套输入NPZ保留每几何实际源中心和beta，JSON含成本/依赖记录。场幅度沿用原仿真数值归一化，未标定为实验V/m；相位相对于固定源和DFT约定，未使用真值拟合对齐。没有承诺宽带、任意源、多端口相干激励或新拓扑。

运行方法见`README.md`。最终CLI示例为`final_cli_example.npz`，已真实运行。最终脚本另增加预生成输入的几何/端口/校准身份核对，防止拿错输入；计时版保留为`predict_benchmarked.py`，见`FINAL_INPUT_CONTRACT_NOTE.md`。数值路径不变，最终身份核对的额外时间单列在示例JSON中，未混入早先计时。

## 2. 72几何、三个训练种子的统一结果

{main}

每个种子先对同一72个几何等权平均，再对三个种子均值汇总。三种子SD、各自中位数/P90/最大值和逐几何值分别在`main_table_three_seed_summary.csv`、`main_table_per_seed.csv`、`endpoints_per_geometry.csv`。72几何为原支持内倾角分层抽样，各档24例；不是旧测试分布的独立同分布随机样本。5%幅度相位门限与eta=0.5法向点对沿用原协议，复场RelL2不采用相位门限。

LocalTangent相对ArcOnly的全域复场平均误差下降{gain:.2f}%。下表Δ=LocalTangent−ArcOnly，负数为改善：

{seedtable}

完整六指标几何配对区间见`paired_differences.csv`，条件于固定模型，不能将三种子×72当作216个独立训练重复。bootstrap按geometry_id排序重放原规则，10000次。逐倾角、宽度和激励方向统计在`stratified_metrics.csv`；这些是描述性分层，不是分离几何效应的因果实验。

Path v2与LocalTangent接近，部分均值略好，未因此选出“最佳模型”。四方法参数均为1,593,122；全部共同输入含四类几何距离信息，区别是输出相位参考，所有构造成本都计入。结果支持局部相位与复场改善，不支持所有指标或所有几何都改善。

![72几何主指标](blind_main_metrics.png)

## 3. 完整查询成本

设备：Intel Core Ultra7 356H，RTX5070 Laptop GPU（约12GB），WSL Ubuntu24.04；Meep1.35.1；PyTorch {read(R/'benchmark_environment.json')['torch']}、CUDA {read(R/'benchmark_environment.json')['cuda']}。FDTD为CPU2线程设置，网络torch4线程；GPU计时同步，warmup另存，未测能耗/货币成本。计时固定seed2026，不能据几百毫秒差异判断某种表示速度更优。

{onlinetable}

基线是原72个resolution20冷进程FDTD成本，均值{c['fdtd_baseline']['mean']:.3f}秒，中位数{c['fdtd_baseline']['median']:.3f}秒，范围{c['fdtd_baseline']['minimum']:.3f}–{c['fdtd_baseline']['maximum']:.3f}秒。冷启动代理在固定首例重复3次；常驻代理用每倾角档首个左右激励共6例；batch1/2各重复3次72几何推理导出，几何准备72例只计时一次，其成本在每个批流程中都加入，没有当作免费缓存。

批量数值包含材料初始化、共同特征和全部载波构造、输入压缩/读取、CPU/GPU传输、网络和载波恢复、结果压缩导出；几何worker启动开销由12个冷启动工作流计时差估计并加入。会话模型加载和预热另计入回本公式。冷启动包含完整进程与模型加载；常驻单例仍每次启动材料worker，网络驻留GPU。

72例每几何准备的平均分解：材料初始化{c['preprocessing']['material_init_seconds']['mean']:.3f}秒，共同几何特征{c['preprocessing']['feature_seconds']['mean']:.3f}秒，输入导出{c['preprocessing']['export_seconds']['mean']:.3f}秒。当前主要瓶颈在几何预处理；只报GPU前向会明显夸大完整流程加速。

**比较范围：**存档FDTD还导出Hx/Hy、原生端口系数和审计信息，代理主要输出Ez；未补做删减这些输出后的优化FDTD基线。因此表中耗时比是已测完整工作流的比较，不能当成最快Ez-only求解器的严格对照，也不能直接与其他硬件或文献的加速倍数比较。

## 4. 离线投入与回本

部署一个seed2026 LocalTangent模型的离线账本：

- 原64训练+16验证几何的标签生成：{c['training_validation_label_seconds']:.3f}秒。
- 原80例共同几何特征计算：{c['training_validation_feature_compute_seconds']:.3f}秒。
- 单模型2400步训练，含训练循环内验证/检查点：{train:.3f}秒。
- 以上历史有记录小计：{qlt['offline_measured_subtotal_seconds']:.3f}秒。
- 缺失历史独立计时的校准、数据装载、模型优化器准备、缓存读写：本轮重放估计{c['reconstructed_missing_overhead_seconds']:.3f}秒。校准仅重算核验原系数、结果丢弃，未更新部署校准；系数差约7.1e-15、scale差约4.4e-16。
- 由历史实测小计加当前开销重放得到的工作流重建成本：**{offline:.3f}秒，约{offline/60:.2f}分钟**。这不是对历史总耗时的精确追溯。

回本定义为墙钟时间：`C_sur(N)=C_offline+C_session+N*t_query`，`C_fdtd(N)=N*t_fdtd`，按正分母求最小整数N。72批量按连续吞吐估计约215例，真正以72为整批时为216例；常驻单例约244次，冷启动单例约376次。既有训练作为沉没成本的增量使用可以更快获益，但不能把这种情景写成从零开始的整体加速。

`offline_cost_ledger.csv`逐项保留来源；`training_costs.csv`列12个模型训练成本；`break_even_scenarios.csv`同时保留历史小计下界、额外60/300秒假设和本轮重建情景。训练80与评价/验证研究用标签分别计算；三种子、四方法、旧100与新72测试和网格诊断属于研究成本，不是部署一个冻结模型所需的训练预算。已定位研究费用小计见`cost_analysis.json`，不冒充整个项目总成本。

## 5. 速度必须与误差一起阅读

下列门限是本轮预先声明的示例报告场景，并非用户给定应用验收标准，也不是独立的前瞻门限验证：

{ratetable}

表内为各种子72几何通过率的均值。全部查询都在速度分母中，没有只保留成功样本。另有弯曲复场RelL2≤0.15且相位WMAE≤5°的联合统计，见`illustrative_tolerance_rates.csv`。例如全域RelL2≤0.10时LocalTangent通过率仍很低；这说明当前模型不适合直接宣称高精度逐例替代。

这些达标情况依赖事后真值。现有代理没有部署时的可靠误差判别器；不能假设自动知道哪些样本需要回退FDTD，也不能据此声称具有保证的回退加速或误差认证。代价曲线只适用于允许该经验误差水平的探索性使用。

![成本与精度](cost_and_accuracy.png)

## 6. 场图与失败案例

固定seed2026，各倾角档选择最接近几何倾角中位数的结构，另取LT弯曲复场最差例与LT−ArcOnly最大退化例；后两者恰好为同一几何，按预先规则保留重复行，实际为4个独立结构。选择记录见`gallery_selection.json`。所有四方法都显示，没有逐图选择最有利种子或真值相位对齐。

实部图每行共享对称颜色范围，保持原数值幅度；相位误差图以真幅度5%门限屏蔽低场，并标出±60°显示裁剪；复场误差按真场峰值归一化，显示范围0–0.5，完整误差不裁剪地计入统计。truth列在相位/复误差图中分别展示原相位/归一化幅度，其色标范围与预测误差列不同，图标题已注明。

![复光场实部](field_gallery_real_field.png)

![局部相位误差](field_gallery_phase_error.png)

![复场误差](field_gallery_complex_error.png)

## 7. 论文主张与停止边界

可支持：同一参数化家族、固定求解和训练预算下，LocalTangent相对ArcOnly在独立新几何上稳定降低复场与局部相位误差；无需新增原始输入或网络参数；独立几何预测入口可运行；本机指定工作流在重复查询中具有可量化的计算时间收益。

仍需限定：模型以resolution20标签训练。此前细网格检查发现30→40法向相位变化约1.70°；后续误差来源分解中，9例30→40弯曲相位变化8.79°，ΔbetaL诊断后残余1.43°，法向变化仍2.14°。三控制60→80法向变化0.106–0.498°。因此当前结果是相对于明确离散参考的代理误差，不能把约4°局部MAE解释为相对于连续Maxwell解的认证精度。原数值精度诊断报告保持不变。

建议论文以“受控输出表示改善参数化光场代理预测”为主线，将可运行代理与完整成本分析作为实际计算价值证据。当前不需要新增逆向设计、双端口相干控制或新器件。若以后目标转为严格物理相位精度，再优先处理界面表示和数值参考收敛。

## 文件入口与核验

`README.md`提供命令；`predict.py`为预测入口；`main_table_per_seed.csv`、`main_table_three_seed_summary.csv`、`paired_differences.csv`为主表；`online_cost_summary.csv`、`offline_cost_ledger.csv`、`break_even_scenarios.csv`为成本表；全部图另有PDF。

`inputs_verification.json`、`prediction_replay_verification.json`、`table_reconstruction_verification.json`、`independent_delivery_verification.json`、`preservation_check.json`为核验记录。保护577个历史文件；严格文件哈希检查在最终交付再次执行。训练/时域FDTD计数均为0。绘图标题索引和跨环境浮点一致性检查曾修复，未改变原数据或科学指标；首次计时重放失败日志保留。
'''
 (R/'REPORT.md').write_text(text,encoding='utf-8')
 readme='''# 几何独立S-bend预测交付

入口：REPORT.md。运行依赖现有D:/game代码、权重、训练校准及WSL环境；无需任何查询真场或历史预测文件。本目录的发布ZIP仅含脚本、表图和审计，不含权重和大型数组。

## 单个几何

在PowerShell运行以下单行命令。输出路径必须尚不存在，目录自动创建；原路径和权重不会覆写。

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python -B /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/predict.py --geometry-json /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/example_geometry.json --method local_tangent --seed 2026 --output /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/my_prediction.npz
```

JSON示例（原始几何参数单位μm，R_min为原构造器参数；端口1左激励、2右激励）：

```json
{"geometry_id":"my_query","params":{"wg_width_um":0.4,"lateral_offset_um":3.5,"R_min_um":5.85},"input_port":1}
```

支持linear / arc_only / local_tangent / path_v2与种子42 /2026 /2027。固定1.55μm、resolution20、160×480输出。允许范围与几何可行性见REPORT；超范围拒绝。不能通过修改JSON请求新频率或材料。

## 批量几何

JSON可以是对象数组或含records数组的对象。对blind_geometries.json执行同一命令可一次预测72例。batch-size默认2，也支持1。

先在Meep Python用--prepare-only --output目录生成输入，再用GPU Python指定--prepared-inputs目录，可多方法共享同一次预处理。首次预处理必须计入总查询成本；再次使用共享缓存需明确说明缓存收益。例：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.conda/envs/picflow-meep/bin/python -B /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/predict.py --geometry-json /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/blind_geometries.json --prepare-only --output /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/my_inputs
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python -B /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/predict.py --geometry-json /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/blind_geometries.json --prepared-inputs /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/my_inputs --method local_tangent --seed 2026 --output /mnt/d/game/research_sbend_surrogate_delivery_2026-11-15/my_batch.npz
```

已运行的独立示例：final_cli_example.npz及其JSON（另保留cold_outputs/local_tangent_0.npz）；全部12份重放：replays/。NPZ的prediction维度[N,160,480]，行对应y_um、列对应x_um；geometry_id/input_port保持顺序。场为原源DFT约定的复Ez数值幅度，不是实验标定的V/m。

函数load_net + forward_records可以在现有Python程序保留GPU模型；几何prepare需使用Meep环境。预测入口有真场路径访问拦截；核验和绘图脚本则刻意读取历史参考。

## 复核脚本

integrate_results.py整合保存结果并绘图；analyze_costs.py重建成本表；audit.py verify检查保护文件；verify_delivery.py核验标量误差、输入范围和访问拦截。run_delivery.py是已完成实验的记录脚本，存在输出时会拒绝覆盖，不要直接重跑生产。

图中代表样本规则、计时边界、校准时间重放及失败记录详见PROTOCOL.md、COST_MEASUREMENT_ADDENDUM.md、REPORT.md。历史代码/模型依赖与哈希见protected_inputs.json，部署实际读取列表在每次输出JSON中。
'''
 (R/'README.md').write_text(readme,encoding='utf-8');save(R/'report_complete.json',dict(unix=time.time(),report_sha256=digest(R/'REPORT.md'),readme_sha256=digest(R/'README.md'),training_runs=0,FDTD_runs=0))
if __name__=='__main__':write_report()

