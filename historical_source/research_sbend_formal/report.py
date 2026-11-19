import csv
import numpy as np
from common import *
LABEL={'direct':'Direct','linear':'Linear','local_tangent':'LocalTangent','path_v1':'Path v1','path_v2':'Path v2'}
def link(name):return '['+name+'](D:/game/research_sbend_formal/'+name+')'
def main():
    rows=read(OUT/'test_summary.json');pairs=read(OUT/'paired_comparisons.json');train=read(OUT/'training_complete.json')['results']
    verification=read(OUT/'verification.json');cal=read(OUT/'calibration.json');manifest=read(OUT/'geometry_manifest.json')
    lookup={(r['checkpoint'],r['group'],r['method']):r for r in rows}
    table=['| 方法 | best测试RelL2 | last测试RelL2 | best验证 | best步 | 训练分钟 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        t=next(a for a in train if a['method']==m)
        table.append('| '+LABEL[m]+' | '+f"{lookup['best','all',m]['rel_l2']:.6f} | {lookup['last','all',m]['rel_l2']:.6f} | {t['best_val']:.6f} | {t['best_step']} | {t['train_wall_seconds']/60:.2f} |")
    primary=[r for r in pairs if r['checkpoint']=='best' and r['group']=='all' and r['metric']=='rel_l2']
    pt=['| Path v2 对照 | 对照−v2均差（正值支持v2） | 几何bootstrap95%区间 | 相对均值改善 | 改善几何 |','|---|---:|---|---:|---:|']
    for r in primary:
        ci=r['bootstrap_geometry_ci95'];pt.append('| '+LABEL[r['baseline']]+f" | {r['mean_baseline_minus_v2']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {100*r['relative_mean_error_reduction']:.2f}% | {r['improved_geometries']}/100 |")
    bt=next(r for r in primary if r['baseline']=='local_tangent');b1=next(r for r in primary if r['baseline']=='path_v1');bl=next(r for r in primary if r['baseline']=='linear')
    def verdict(r):
        lo,hi=r['bootstrap_geometry_ci95']
        return '配对区间支持v2误差更低' if lo>0 else ('配对区间支持v2误差更高' if hi<0 else '配对区间跨零，未确立误差差异')
    dirs=['| 方法 | 左入射best RelL2 | 右入射best RelL2 | 芯 | 弯曲芯 | halo | 远场 |','|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        a=lookup['best','all',m];dirs.append('| '+LABEL[m]+f" | {lookup['best','input1',m]['rel_l2']:.6f} | {lookup['best','input2',m]['rel_l2']:.6f} | {a['core_rel_l2']:.6f} | {a['bend_core_rel_l2']:.6f} | {a['halo_rel_l2']:.6f} | {a['far_rel_l2']:.6f} |")
    ports=['| 方法 | 非输入端复系数误差 | 幅值误差 | 平方幅值误差 |','|---|---:|---:|---:|']
    for m in METHODS:
        a=lookup['best','all',m];ports.append('| '+LABEL[m]+f" | {a['noninput_complex_error']:.6f} | {a['noninput_amplitude_error']:.6f} | {a['noninput_squared_amplitude_error']:.6f} |")
    with (OUT/'port_coefficients.csv').open(encoding='utf-8-sig') as f:portrows=list(csv.DictReader(f))
    proxyrows=[r for r in portrows if r['method']=='truth_proxy' and r['is_reflection']=='False']
    reflection=[r for r in portrows if r['method']=='truth_proxy' and r['is_reflection']=='True']
    trutherr={k:float(np.mean([float(r[k]) for r in proxyrows])) for k in ['complex_error','amplitude_error','squared_amplitude_error']}
    reflecterr=float(np.mean([float(r['complex_error']) for r in reflection]))
    costs=[read(p) for p in sorted((OUT/'logs').glob('fdtd_*_resource.json'))]
    cache=read(OUT/'cache_complete.json')['records'];pf=read(OUT/'preflight.json')
    text=f"""# S-bend正式五组实验报告

完成UTC：{now()}。唯一训练seed2026。180个新几何按64训练/16验证/100测试，16个已看开发几何及所检查等价版本排除。Path v3作为负面开发证据保留，不参加正式比较；本轮未修改v1/v2/v3算法或搜索超参数。

## 1. 执行、冻结与数据隔离

运行前已读取研究交接、v1/v2/v3报告及正式协议。既有文档的未来建议与本次用户明确授权区分；本次用户授权第一阶段通过即执行后续阶段。

{link('PROTOCOL.md')}、{link('config.json')}、{link('preflight.json')}、{link('runtime_checks.json')}、{link('code_freeze.json')}记录可运行实现与SHA-256。12实特征统一打包为7复通道（14实槽，epsilon虚部和最后虚部为零），原Fourier入口重建纯实介电常数，附加4实exp先验。所有组均使用同一12特征，包括全部几何距离；只改变输出载波。参数元素{pf['model_checks'][0]['parameters']}，实标量自由度{pf['model_checks'][0]['real_scalar_degrees']}。初始化、批次、先验哈希及前向/反向有限性检查一致，运行前无优化器更新。

额外运行级检查发现原开发代理使用仿真float64 eps，导出eps为float32；直接以导出eps复算差3.23e-9。恢复原始输入后精确复现，记录在runtime_check_initial_failure.json；未改变代理算法或以S校准预测。正式预测/真值代理始终共同使用同一导出eps。

几何清单{link('geometry_manifest.json')}及{link('geometry_freeze.json')}在首个FDTD前冻结：PCG64(2026)，8个宽度格点分层、每几何仅一个激励侧，train32/32、val8/8、test50/50。参数元组、中心线+宽度等价指纹、最终float32 eps及轴反射/180度旋转哈希排重。拒绝候选{len(manifest['rejections'])}个，完整保存。该规则不证明设计空间远距离或外部实验室盲测。

物理配置为Meep1.35.1、MPB1.12.0、PIC-Flow ZIP commit fb752f74fdb3c1cee4db8e6fd6a0bb6cac616f34，26×10μm/PML1μm、内域24×8μm/160×480、dx=dy=.05μm、波长1.55μm。源使用真实±11.9μm位置，mask使用已验证物理体素中心规则。保存复Ez/Hx/Hy、eps、真实源、端口、alpha±、原生复S及版本。当前仿真不称历史HF数据逐位复现。

测试在训练前只经预定自动结构QC/几何载波QC，不查看相位诊断或器件性能排序。beta只用64训练场标定一次，a,b={cal['coefficients_rad_um']} rad/μm，固定线性拟合相对RMSE={100*cal['relative_rmse']:.4f}%，共同训练RMS尺度={cal['scale']:.8g}。验证/测试标签未标定载波或归一化。

## 2. 全部五组结果

每组相同Fourier12层×64、modes40/70、2400步、batch2、AdamW lr.002、wd1e-4、clip5、FP32/TF32；余弦至.0001，dropout.1，seed2026，同初始化/批次。每50步和步1以16验证平均逐几何物理复场RelL2选best，保留last。五组权重和评价代码冻结后一次评价100测试，best/last均在同一次事件内保存。

{chr(10).join(table)}

RelL2逐几何计算后等权平均，无全局真值相位对齐；没有删不利几何。全部1000条见{link('test_per_geometry.csv')}，十组预测见predictions目录，权重见runs目录。

## 3. 预定主比较与统计边界

{chr(10).join(pt)}

相同几何配对，PCG64(2026)共10000次bootstrap；正差表示v2更好。区间只度量这一训练种子下测试几何的抽样波动，不能当训练随机性、跨种子稳定性或严格等效检验。完整best/last、方向、宽度、R和区域对照见{link('paired_comparisons.json')}。

## 4. 方向、空间区域与不利结果

{chr(10).join(dirs)}

区域以几何最近距离定义：芯<=w/2；弯曲芯另限官方弯曲x区间；halo为w/2到w/2+.5；其余远场。区域相对误差以该区真值能量归一化，远场低能量可使相对误差更大，不能直接将各区数值视为对全域误差的贡献。

所有逐几何负差和参数可由完整CSV与冻结manifest核对；图中同时保留误差改善和恶化。没有按正式测试结果改变载波/主对照/门槛，也不以某个方向或区域的有利子组替代总体主结论。

## 5. 非输入端代理及自身误差

{chr(10).join(ports)}

上表预测对同一真实Ez提取的代理，不是预测对严格Meep S。真值Ez代理对Meep非输入端的复/幅值/平方幅值MAE分别为{trutherr['complex_error']:.6g}/{trutherr['amplitude_error']:.6g}/{trutherr['squared_amplitude_error']:.6g}。反射复系数MAE={reflecterr:.6g}，仅探索，不能据此认证回波损耗。逐端口复数、反射标记与两种参照全部见{link('port_coefficients.csv')}。

沿用冻结标量模式+法向差分代理，端口方向和入射归一化不变，不使用真实S调整预测；网络场可能不满足Maxwell，代理相对真实器件功能仍有已知局限。

## 6. 计算成本与核验

串行FDTD180例，累计{sum(r['wall_seconds'] for r in costs)/60:.2f}分钟，单例{min(r['wall_seconds'] for r in costs):.2f}–{max(r['wall_seconds'] for r in costs):.2f}秒，监测峰RSS {max(r['peak_rss_MiB'] for r in costs):.1f}MiB。v2几何构造均值{np.mean([r['v2_seconds'] for r in cache]):.4f}秒/例，最大单轴距离梯度{max(r['v2_max_axis_gradient'] for r in cache):.6f}；所有正式几何通过冻结<=5门槛。

v1及Linear/LocalTangent共同几何预处理均值{np.mean([r['v1_and_baselines_seconds'] for r in cache]):.4f}秒/例；这不是单独某一种载波的独立计时。共同12输入包含全部载波几何，因此实际缓存准备共同承担该成本。训练成本在主表，GPU模型前向及场恢复成本单独保存{link('inference_costs.json')}，不混同FDTD总时间或端到端在线时间。

核验{link('verification.json')}：{verification['history_files_hashed']}个历史文件SHA无变化，{verification['shards_stat_checked']}个分片大小/mtime无变化（不冒称分片全哈希）；180新样本、10权重、10预测哈希通过；保存预测复算{verification['metric_checks']}项，最大差{verification['metric_max_difference']:.3g}，无新推理或再训练。

## 7. 对路线A的五项回答

1. **相对Linear是否改善？** Path v2：均值改善{100*bl['relative_mean_error_reduction']:.2f}%，{bl['improved_geometries']}/100几何改善；{verdict(bl)}。v1/LocalTangent/Direct和固定last同样完整报告。
2. **LocalTangent是否已达到Path相当表现？** v2对LocalTangent均差{bt['mean_baseline_minus_v2']:.6f}；{verdict(bt)}。没有预设等效界，区间跨零不能写成证明等效；单种子也不能认证训练稳定性。
3. **v2连续性修复是否有学习收益？** v2对v1均差{b1['mean_baseline_minus_v2']:.6f}；{verdict(b1)}。几何连续性修复与网络精度收益是不同结论；v2包含近芯核平均、过渡和源锚定，不能把差异严格归因于远场连续性一个因素。
4. **是否有恶化？** v2相对LocalTangent恶化{bt['worsened_geometries']}/100，相对v1恶化{b1['worsened_geometries']}/100，相对Linear恶化{bl['worsened_geometries']}/100。方向/区域及全部负例均保存，未据此回调算法。
5. **原创性和投稿水平？** 本轮增加第二器件家族和强几何对照的独立保留集证据，有助于界定几何表示的适用范围。数值收益不自动证明方法新颖性、复杂度降低、Maxwell一致性或可替代FDTD；投稿水平仍需要相关工作定位、方法论和物理误差论证，不能由单种子100几何主表直接认证。若LocalTangent持平/更优，应收缩Path独有贡献，保留该负面结论，不再以v3/v4或测试后调参补救。

后续若写论文应围绕冻结结果及限制展开；本轮不再算法搜索、不加训练种子、不增加模型或重用这100例作新方法独立盲测。
"""
    (OUT/'S-bend正式五组实验报告.md').write_text(text,encoding='utf-8')
    print('Chinese formal report written',flush=True)
if __name__=='__main__':main()
