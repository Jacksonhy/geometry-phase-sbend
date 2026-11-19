"""Produce scientific plots and a candid paired full-resolution experiment report."""
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    summary=read(ROOT/'summary.json');search=read(ROOT/'search_results.json')
    final=read(ROOT/'final_results.json');verification=read(ROOT/'verification.json')
    diagnostics=read(ROOT/'validation_diagnostics.json')
    manifest=read(ROOT/'manifest.json');comparison=verification['comparison']
    spectrum=read(ROOT/'spectral_diagnostic.json')['results']
    variants=['direct','path'];colors=['#437ca0','#238573'];labels=['Direct complex field','Path demodulation']
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    for i,v in enumerate(variants):
        rs=sorted([r for r in search if r['variant']==v],key=lambda r:r['lr'])
        axes[0].plot([r['lr'] for r in rs],[r['best_val'] for r in rs],'-o',label=labels[i],color=colors[i])
        vals=[next(r['best_val'] for r in final if r['variant']==v and r['seed']==s) for s in [2026,2027,2028]]
        axes[1].bar(i,summary[v]['val_mean'],yerr=summary[v]['val_seed_sd'],capsize=6,color=colors[i],alpha=.65)
        axes[1].scatter(i+np.linspace(-.08,.08,3),vals,color=colors[i],edgecolors='black',s=30,zorder=4)
        axes[1].text(i,summary[v]['val_mean']+summary[v]['val_seed_sd']+.025,f"{summary[v]['val_mean']:.4f}",ha='center')
    axes[0].set_xscale('log');axes[0].set_xticks([.0005,.001,.002],['0.0005','0.001','0.002']);axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set(xlabel='Initial learning rate',ylabel='Best validation RelL2',title='Equal LR search, seed 2026');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].set_xticks([0,1],['Direct','Path']);axes[1].set(ylabel='Validation RelL2',title='Selected LR: mean ± seed SD')
    axes[1].set_ylim(0,max(summary[v]['val_mean']+summary[v]['val_seed_sd'] for v in variants)+.12);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Official Fourier backbone, full 160×480 resolution — exploratory validation only')
    fig.tight_layout(rect=(0,.05,1,.93));fig.text(.5,.015,'Both arms use identical inputs and official wave prior; 64 train / 8 validation. No independent test.',ha='center',fontsize=9)
    fig.savefig(figdir/'full_resolution_results.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,v in enumerate(variants):
        hs=[read(ROOT/'runs'/f"{v}_lr{summary[v]['lr']:g}_seed{s}"/'history.json') for s in [2026,2027,2028]]
        values=np.array([[r['val_rel_l2'] for r in h] for h in hs]);steps=[r['step'] for r in hs[0]]
        ax.plot(steps,values.mean(0),label=labels[i],color=colors[i])
        ax.fill_between(steps,values.mean(0)-values.std(0,ddof=1),values.mean(0)+values.std(0,ddof=1),color=colors[i],alpha=.15)
    ax.set(xlabel='Optimizer updates',ylabel='Validation RelL2',title='Selected-LR validation curves (mean ± seed SD)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'validation_curves.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5))
    for seed in [2026,2027,2028]:
        p={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in variants}
        ax.scatter(p['direct']['val_per_sample'],p['path']['val_per_sample'],label=f'Seed {seed}',s=35,alpha=.7)
    limit=max(max(r['val_per_sample']) for r in final)*1.05
    ax.plot([0,limit],[0,limit],'--',color='gray');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Direct: per-geometry validation RelL2',ylabel='Path: per-geometry validation RelL2',title='Paired validation errors (below diagonal favors Path)')
    ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'paired_validation.png',dpi=180);plt.close(fig)

    # Predetermined validation indices, not selected for visual advantage.
    with np.load(ROOT/'validation_predictions.npz',allow_pickle=False) as z:
        truth=z['truth'][:,0];eps=z['eps'];geoms=z['geometry_id']
        preds={v:z['prediction'][list(z['run']).index(f"{v}_lr{summary[v]['lr']:g}_seed2026"),:,0] for v in variants}
        fig,axes=plt.subplots(3,5,figsize=(15,5.8),squeeze=False)
        for row,idx in enumerate([0,3,7]):
            scale=max(float(abs(truth[idx]).max()),1e-8)
            fields=[truth[idx].real,preds['direct'][idx].real,preds['path'][idx].real,
                    abs(preds['direct'][idx]-truth[idx]),abs(preds['path'][idx]-truth[idx])]
            for col,field in enumerate(fields):
                ax=axes[row,col]
                ax.imshow(field/scale,origin='lower',cmap='RdBu_r' if col<3 else 'magma',
                          vmin=-1 if col<3 else 0,vmax=1,aspect='equal')
                ax.contour(eps[idx],levels=[(eps[idx].min()+eps[idx].max())/2],colors='gray',linewidths=.35,alpha=.7)
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(['Truth Re(E)','Direct Re(E)','Path Re(E)','Direct |error|','Path |error|'][col])
                if col==0:ax.set_ylabel(str(geoms[idx]).replace('ybranch_',''),fontsize=8)
        fig.suptitle('Validation IDs at fixed positions 0, 3, 7; seed 2026; each row scaled by truth max |E|')
        fig.tight_layout(rect=(0,.04,1,.93));fig.text(.5,.015,'Signed field colors: −1 to 1; error colors: 0 to 1 (larger errors saturate). Geometry contours shown.',ha='center',fontsize=9)
        fig.savefig(figdir/'validation_fields.png',dpi=180);plt.close(fig)

    table='\n'.join(f"| {labels[i]} | {summary[v]['lr']:g} | {summary[v]['val_mean']:.4f} ± {summary[v]['val_seed_sd']:.4f} | {summary[v]['train_mean']:.4f} | {summary[v]['mean_train_seconds']/60:.2f} |" for i,v in enumerate(variants))
    seeds=[]
    for seed in [2026,2027,2028]:
        p={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in variants}
        seeds.append(f"| {seed} | {p['direct']['best_val']:.6f} ({p['direct']['best_step']}) | {p['path']['best_val']:.6f} ({p['path']['best_step']}) | {p['direct']['best_val']-p['path']['best_val']:.6f} |")
    lrtable='\n'.join(f"| {lr:g} | "+' | '.join(f"{next(r['best_val'] for r in search if r['variant']==v and r['lr']==lr):.6f}" for v in variants)+' |' for lr in manifest['lrs'])
    diagnostic_rows=[]
    spectrum_rows=[]
    for i,v in enumerate(variants):
        ds=[diagnostics[f"{v}_lr{summary[v]['lr']:g}_seed{s}"]['overall'] for s in [2026,2027,2028]]
        diagnostic_rows.append(f"| {labels[i]} | "+' | '.join(f"{np.mean([d[k] for d in ds]):.4f}" for k in ['amplitude_rel_l2','intensity_weighted_phase_mae','intensity_ratio'])+' |')
        spectrum_rows.append(f"| {labels[i]} | "+' | '.join(f"{spectrum[v][axis]['val']['mean_fraction']:.4%}" for axis in ['row_axis_modes40','column_axis_modes70'])+' |')
    reduction=comparison['relative_mean_error_reduction'];benefit=reduction>0
    verdict='支持此设置下 Path 的验证收益' if benefit else '没有支持此设置下 Path 的平均验证收益'
    allseed=all(v>0 for v in comparison['paired_seed_improvements'])
    total_seconds=sum(read(p)['train_wall_seconds'] for p in (ROOT/'runs').glob('*/summary.json'))
    peak=max(read(p)['peak_allocated_MiB'] for p in (ROOT/'runs').glob('*/summary.json'))
    report=f'''# 第五轮：官方 Fourier 主干与完整分辨率的 Path 验证

日期：2026-11-11。全部基于本地材料与本机 GPU，没有网络下载。

## 结论

本轮{verdict}。Direct 平均验证 RelL2 为 {summary['direct']['val_mean']:.4f}，Path 为 {summary['path']['val_mean']:.4f}；按两组均值计算的相对误差下降为 {reduction:.2%}。三个种子的配对差依次为 {', '.join(f'{v:.6f}' for v in comparison['paired_seed_improvements'])}，{'方向全部支持 Path' if allseed else '需按下表区分各个种子'}。

双方均使用官方 exp wave prior、相同的完整分辨率输入和同一 Fourier 网络，唯一实验干预是输出表示及确定性的相位恢复。因此本轮已经排除了上一轮 U-Net 输出降采样/插值这一因素。但当前仍是 **8 个已有验证几何上、有限预算下的探索性结果**，不等于完整复现原论文，也不是独立测试结论。

## 配对实验设置

- 数据：原64训练+8验证，160×480。历史45个测试几何不推理。波数标定与尺度归一化只从64训练场得到。
- 骨干：GitHub commit `08cad10a78a01e4a522f8b4d287db0b363054e33`，12层、64通道、modes40/70、head256、原卷积stem、FFN与depthwise卷积、GELU；aug_path=False、dropout0.1、drop_path0。原类体和Fourier算子不改写；依赖使用第四轮兼容加载器。
- 输入：双方同样获得 ε、二值src_mask、x/y、源x/y、方向、训练标定k、path/W、linear/W。通过原复数接口无损打包为6个复通道，展开为12实通道后拼接官方4通道prior。stem共16实输入；增加的输入通道是公开的数据适配，不是原论文原生源条件。
- 物理量：λ1.55μm、dx=dy0.05μm。网络的输入、prior、输出、监督全部160×480；没有降采样或输出插值。保持相同物理域下的modes40/70；未额外搜索频率截断。
- Direct直接输出复场；Path输出复包络再乘由几何与训练标定k构造的单位复载波。双方都在恢复后的物理场计算相对复数MSE，以平均逐样本RelL2选权重，不做标签相位对齐。
- 优化：FP32（允许TF32）、物理batch2、1200更新，无梯度累积。AdamW wd1e-4，余弦衰减至初始5%，裁剪5，无额外增强。有限显存决定batch2；这与第四轮batch8不同，不能把两轮绝对误差直接解释为单纯换架构的收益。
- 搜索：每组同样的lr[0.0005,0.001,0.002]，seed2026；各选最低验证值的lr后扩展seed2027/2028，共10次训练。每50步验证，含第1步。同种子共享初始化与批次序列，已逐哈希核验。三个种子中含用于选lr的2026。

预设规则见 [PROTOCOL.md](D:/game/research_v5/PROTOCOL.md)；训练前冻结记录见 [manifest.json](D:/game/research_v5/manifest.json)。每方法相同更新数、搜索次数和样本曝光（1200×2/64=37.5次平均曝光），没有根据结果追加候选。

## 实测结果

| 方法 | 选定lr | 验证RelL2，均值±种子SD | 对应权重训练RelL2均值 | 平均训练分钟 |
| --- | ---: | ---: | ---: | ---: |
{table}

种子标准差描述训练波动，不是新几何上的置信区间。训练指标从各自最佳验证权重重新计算。

| seed | Direct（最佳步数） | Path（最佳步数） | Direct − Path |
| --- | ---: | ---: | ---: |
{chr(10).join(seeds)}

seed2026学习率搜索，每格为该运行的最佳验证RelL2：

| lr | Direct | Path |
| --- | ---: | ---: |
{lrtable}

按三个种子平均每个几何的误差后，{comparison['positive_geometries_after_seed_average']}/8个几何改善，{comparison['negative_geometries_after_seed_average']}/8个变差。逐几何表见 [validation_samples.csv](D:/game/research_v5/validation_samples.csv)。这些8个样本也用于选模型，不能把改善比例当作无偏泛化统计。

额外物理场诊断（各样本指标先平均，再对种子平均；未作相位对齐）：

| 方法 | 幅度RelL2，越低越好 | 真值能量加权相位MAE，rad，越低越好 | 预测/真值总强度比，理想为1 |
| --- | ---: | ---: | ---: |
{chr(10).join(diagnostic_rows)}

强度比是该标量场的幅度平方和之比，不等同于端口Poynting功率或S参数守恒；phase指标对真值低能量像素权重较小。明细见 validation_diagnostics.json。

![完整分辨率配对结果](D:/game/research_v5/figures/full_resolution_results.png)

![验证学习曲线](D:/game/research_v5/figures/validation_curves.png)

![逐几何配对误差](D:/game/research_v5/figures/paired_validation.png)

![预先固定位置的验证场](D:/game/research_v5/figures/validation_fields.png)

场图固定展示验证序列位置0、3、7与seed2026，没有挑选Path最有利案例。每行以真值最大幅度归一化，误差颜色超过1会饱和；复场原始幅度预测见 validation_predictions.npz。

## 资源与执行核验

两组均1,593,058个参数元素（复数计一个），对应2,944,738个实标量自由度。训练前batch2 FP32峰值约6498MiB，热身后单步约0.224秒。BF16试运行报Unsupported dtype BFloat16，尚未进入正式搜索；双方统一采用FP32，未更改原FFT算子。

正式运行最大已分配显存{peak:.1f}MiB。10次训练循环合计{total_seconds/60:.2f}分钟；循环时间包含周期验证和保存，排除特征准备、最终重算训练指标、最终核验和报告。只比较本机实测耗时，不当作部署吞吐基准。

[interface_verification.json](D:/game/research_v5/interface_verification.json) 验证了完整分辨率特征打包、与第四轮相同的基础特征/标定/载波、单位载波和物理损失等价。官方forward会从FP32归一化ε还原ε，与直接从原ε计算prior的最大差约1.53×10⁻⁵，来自高列坐标的相位舍入，两组完全共享同一路径。

[verification.json](D:/game/research_v5/verification.json) 核验10次运行组合与权重哈希、全部25次验证记录、最优验证检查点、学习率选择、三种子聚合、初始化/批次配对、10次验证重放，以及之前41个冻结权重未改。验证重放最大绝对误差{verification['max_validation_replay_absolute_difference']:.3g}。

## 解释边界与下一步

本轮回答的是：在固定骨干、信息和有限优化预算下，几何Path相位分解能否改善学习。它没有穷尽直接场表示的优化；1200更新与batch2可能尚未充分收敛，且BatchNorm的行为与batch有关。训练曲线和对应训练误差需一同解读，不能只按单个最优验证数评价模型潜力。

完整空间分辨率排除了输出插值这一瓶颈，但Fourier层仍截断为40/70个模式。Direct与包络的频谱内容不同，因此收益可能部分体现为有限频率预算下的表示效率。这也是合理的受控收益，但不能外推为所有模式数下均有相同改善；若扩展模式数，应给双方同样的新预算并单独报告。

补充的事后目标频谱诊断如下。对复数目标做各轴FFT，保留对称±(modes−1)频率，计算覆盖能量比例；没有加窗：

| 验证目标表示 | 行轴40个模式对应频率内的能量比例 | 列轴70个模式对应频率内的能量比例 |
| --- | ---: | ---: |
{chr(10).join(spectrum_rows)}

两种表示的绝大多数目标频谱能量已在该范围内，且本诊断中Direct的覆盖率更高。因此当前不能把收益简单归因于直接场高频超出模式范围。该诊断只观察目标，不是隐藏特征频谱、模型表达能力或误差下界；非线性stem与FFN也能产生额外频率。明细见 spectral_diagnostic.json，没有据此改变配置。

原生NeurOLight的已知入射复场、同几何多端口叠加增强、normalized cMAE(+TV)、非零drop-path都没有在本轮复现。因此应称为“NeurOLight Fourier骨干的PIC-Flow适配”，不能称为原论文完整复现。双方都获得训练标定k和路径输入，结论是输出相位分解在这些共享信息之外的作用。

独立的新几何测试仍是下一步必需的确认环节。此前45个测试样本已用于研究观察，不能重新命名为保留集。现阶段不据8个验证样本作统计显著性或架构普适性声明。

如果继续强化结论，应先固定当前代码、权重及有限搜索规则，使用独立几何确认；如要扩展训练长度或原生损失，则双方按预先固定的新预算同时开展，并把新增验证探索与最终独立测试区分。
'''
    (ROOT/'完整分辨率验证报告.md').write_text(report,encoding='utf-8')
    (ROOT/'README.md').write_text('''# 第五轮完整分辨率实验

先读 [完整分辨率验证报告](D:/game/research_v5/完整分辨率验证报告.md)。10次训练已完成并冻结，结果为验证集探索，没有独立测试推理。

复核（读取原64训练+8验证；GPU执行10次验证重放）：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_v5/verify_results.py
```

重新生成报告和图：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_v5/make_report.py
```

fourier_full.py会保护已完成运行并核验manifest，不静默覆盖未完成运行。新增实验请使用新目录及新协议。本轮依赖第四轮兼容加载器、第三轮geometry_features、原80样本NPZ和本地官方源码。结果ZIP包含本轮权重与日志、支持源码和原作者MIT许可；不包含数据、之前41个权重、原始论文或完整官方工程，需保留现有工作目录相对结构。
''',encoding='utf-8')
    (ROOT/'STATUS.md').write_text('''# 已完成

- 160×480完整分辨率、官方12×64 Fourier主干；Direct/Path同输入和wave prior。
- 每组3次lr搜索，选定lr后3种子，共10次1200步训练。
- 10次验证重放和41个历史权重保护核验通过。
- 无测试推理；独立新几何确认仍待后续。
''',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'picflow_v5_full_resolution_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip']
        paths += [ROOT.parent/'research_v4'/p for p in ['official_architecture.py','official_bridge.py','stage_a.py']]
        paths += [ROOT.parent/'research_v3'/p for p in ['experiment.py','evaluate_fresh.py']]
        source=ROOT.parent/'references/NeurOLight-source/NeurOLight-main'
        paths += [source/p for p in ['LICENSE','core/models/pde_base.py','core/models/neurolight_cnn.py','core/models/layers/neurolight_conv2d.py','core/models/layers/fno_conv2d.py']]
        for p in sorted(paths):z.write(p,p.relative_to(ROOT.parent))
    print('Report and 4 figures saved. ZIP MiB:',(ROOT/'picflow_v5_full_resolution_results.zip').stat().st_size/1024**2)


if __name__=='__main__':main()
