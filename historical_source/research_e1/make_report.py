"""Static scientific figures and E1 report from frozen validation logs."""
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent;V5=BASE/'research_v5'
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    summary=read(ROOT/'summary.json');final=read(ROOT/'final_results.json');audit=read(ROOT/'verification.json')
    diagnostic=read(ROOT/'validation_diagnostics.json')
    variants=['direct','linear','path'];labels=['Direct','Linear','Path'];colors=['#84909e','#437da1','#238573']
    searches=read(V5/'search_results.json')+read(ROOT/'search_results.json')
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for i,v in enumerate(variants):
        rs=sorted([r for r in searches if r['variant']==v],key=lambda r:r['lr'])
        axes[0].plot([r['lr'] for r in rs],[r['best_val'] for r in rs],'-o',label=labels[i],color=colors[i])
        values=[next(r['best_val'] for r in final if r['variant']==v and r['seed']==seed) for seed in [2026,2027,2028]]
        axes[1].bar(i,summary[v]['val_mean'],yerr=summary[v]['val_seed_sd'],capsize=5,color=colors[i],alpha=.65)
        axes[1].scatter(i+np.linspace(-.08,.08,3),values,color=colors[i],edgecolors='black',s=25,zorder=4)
        axes[1].text(i,summary[v]['val_mean']+summary[v]['val_seed_sd']+.035,f"{summary[v]['val_mean']:.4f}",ha='center')
    axes[0].set_xscale('log');axes[0].set_xticks([.0005,.001,.002],['0.0005','0.001','0.002']);axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set(xlabel='Initial learning rate',ylabel='Best validation RelL2',title='Equal LR search, seed 2026');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].set_xticks(range(3),labels);axes[1].set(ylim=(0,1.05),ylabel='Validation RelL2',title='Selected LR: mean ± seed SD');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('E1: full-resolution Fourier ablation — all arms use identical inputs and wave prior')
    fig.tight_layout(rect=(0,.05,1,.93));fig.text(.5,.015,'64 train / 8 validation, 1200 updates. Exploratory validation only; no independent test.',ha='center',fontsize=9)
    fig.savefig(figdir/'e1_results.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,v in enumerate(variants):
        base=ROOT if v=='linear' else V5
        histories=[read(base/'runs'/f"{v}_lr{summary[v]['lr']:g}_seed{seed}"/'history.json') for seed in [2026,2027,2028]]
        values=np.array([[r['val_rel_l2'] for r in h] for h in histories]);steps=[r['step'] for r in histories[0]]
        ax.plot(steps,values.mean(0),label=labels[i],color=colors[i]);ax.fill_between(steps,values.mean(0)-values.std(0,ddof=1),values.mean(0)+values.std(0,ddof=1),color=colors[i],alpha=.15)
    ax.set(xlabel='Optimizer updates',ylabel='Validation RelL2',title='Selected-rate curves (mean ± seed SD)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'validation_curves.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5))
    for seed in [2026,2027,2028]:
        p={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in ['linear','path']}
        ax.scatter(p['linear']['val_per_sample'],p['path']['val_per_sample'],s=30,alpha=.75,label=str(seed))
    limit=max(max(r['val_per_sample']) for r in final if r['variant']!='direct')*1.06
    ax.plot([0,limit],[0,limit],'--',color='gray');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Linear per-geometry validation RelL2',ylabel='Path per-geometry validation RelL2',title='Primary comparison: below diagonal favors Path');ax.set_aspect('equal');ax.legend(title='Seed');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'linear_path_paired.png',dpi=180);plt.close(fig)

    with np.load(ROOT/'validation_predictions.npz',allow_pickle=False) as z:
        truth=z['truth'][:,0];eps=z['eps'];geoms=z['geometry_id'];names=list(z['run'])
        pred={v:z['prediction'][names.index(f"{v}_lr{summary[v]['lr']:g}_seed2026"),:,0] for v in variants}
        fig,axes=plt.subplots(3,6,figsize=(17,5.5))
        for row,idx in enumerate([0,3,7]):
            scale=max(float(abs(truth[idx]).max()),1e-8)
            fields=[truth[idx].real]+[pred[v][idx].real for v in variants]+[abs(pred[v][idx]-truth[idx]) for v in ['linear','path']]
            for col,f in enumerate(fields):
                ax=axes[row,col];ax.imshow(f/scale,origin='lower',cmap='RdBu_r' if col<4 else 'magma',vmin=-1 if col<4 else 0,vmax=1)
                ax.contour(eps[idx],levels=[(eps[idx].min()+eps[idx].max())/2],colors='gray',linewidths=.3,alpha=.7)
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(['Truth Re(E)','Direct Re(E)','Linear Re(E)','Path Re(E)','Linear |error|','Path |error|'][col])
                if col==0:ax.set_ylabel(str(geoms[idx]).replace('ybranch_',''),fontsize=8)
        fig.suptitle('Fixed validation positions 0, 3, 7; seed 2026; each row scaled by truth max |E|')
        fig.tight_layout(rect=(0,.04,1,.93));fig.text(.5,.015,'Signed fields: −1 to 1; error: 0 to 1 (larger errors saturate). Geometry contours shown.',ha='center',fontsize=9)
        fig.savefig(figdir/'validation_fields.png',dpi=180);plt.close(fig)

    comp=audit['comparisons']['linear_minus_path'];gain=comp['relative_error_reduction']
    verdict='支持本设置下弧长路径相对直线载波的额外验证收益' if gain>0 else '没有支持本设置下弧长路径相对直线载波的平均验证收益'
    table='\n'.join(f"| {labels[i]} | {summary[v]['lr']:g} | {summary[v]['val_mean']:.4f} ± {summary[v]['val_seed_sd']:.4f} | {summary[v]['train_mean']:.4f} |" for i,v in enumerate(variants))
    seedrows=[]
    for seed in [2026,2027,2028]:
        p={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in variants}
        seedrows.append(f"| {seed} | "+' | '.join(f"{p[v]['best_val']:.6f}" for v in variants)+f" | {p['linear']['best_val']-p['path']['best_val']:.6f} |")
    lrrows='\n'.join(f"| {lr:g} | "+' | '.join(f"{next(r['best_val'] for r in searches if r['variant']==v and r['lr']==lr):.6f}" for v in variants)+' |' for lr in [.0005,.001,.002])
    metricrows=[]
    for i,v in enumerate(variants):
        ds=[diagnostic[f"{v}_lr{summary[v]['lr']:g}_seed{seed}"]['overall'] for seed in [2026,2027,2028]]
        metricrows.append(f"| {labels[i]} | "+' | '.join(f"{np.mean([d[k] for d in ds]):.4f}" for k in ['amplitude_rel_l2','intensity_weighted_phase_mae','intensity_ratio'])+' |')
    matched='\n'.join(f"| {r['lr']:g} | {r['linear']:.6f} | {r['path']:.6f} | {r['linear_minus_path']:.6f} |" for r in audit['matched_lr_seed2026_linear_vs_path'])
    seconds=sum(read(p)['train_wall_seconds'] for p in (ROOT/'runs').glob('*/summary.json'))
    peak=max(read(p)['peak_allocated_MiB'] for p in (ROOT/'runs').glob('*/summary.json'))
    report=f'''# E1：Fourier完整表示消融

日期：2026-11-12。全部本地计算。新增5次Linear训练，复用且核验第五轮10个Direct/Path候选。

## 核心结论

本轮{verdict}。Linear平均验证RelL2为{summary['linear']['val_mean']:.4f}，Path为{summary['path']['val_mean']:.4f}，按两组均值计算的相对误差下降为 **{gain:.2%}**。

三个种子的Linear−Path差依次为 {', '.join(f'{v:.6f}' for v in comp['paired_seed_differences'])}。按种子平均后，8个验证几何中{comp['positive_geometries']}个改善、{comp['negative_geometries']}个变差。此处只描述反复用于开发和选择的验证集，不是独立测试或显著性证明。

改善分布不均匀：两个几何贡献了较大的正差，三个几何略变差。因此支持的是当前样本上的平均额外收益，不能写成每个几何均获益。逐几何差值全部保存在verification.json和validation_samples.csv。

此前Path对Direct的45.62%差值包含一般解调与路径选择两部分，不能全部归为弧长的独立贡献。本轮以 **Path对Linear** 为主比较；Direct对Linear用于观察一般解调收益。由于方法和最佳检查点均按验证选择，两个收益不能机械地分解为因果贡献比例。

## 实验设置与公平性

官方NeurOLight Fourier骨干的PIC-Flow适配：12层×64通道、modes40/70、head256、GELU、原卷积stem与FFN，aug_path=False、dropout0.1、drop_path0。原始GitHub commit为08cad10a78a01e4a522f8b4d287db0b363054e33。所有组均带官方exp wave prior。

双方基础输入相同：ε、二值源掩膜、x/y、源x/y、方向、训练标定k、path/W、linear/W。Linear仍接收路径输入；对比只干预输出表示。所有输入、prior、输出和监督均160×480，不做降采样或插值。λ1.55μm，dx=dy0.05μm。

Direct输出E；Linear输出A再乘exp(i·k·直线距离)；Path输出A再乘exp(i·k·几何弧长)。k与尺度仅由64训练场拟合，推理输入接口不接收目标场。所有损失在恢复后的物理复场计算，不做标签相位对齐。

训练仍为64训练+8验证，FP32允许TF32、batch2、1200更新、AdamW wd1e-4、梯度裁剪5、余弦到初始5%、无增强。每50步验证，含第1步，按平均逐样本RelL2选权重。各组相同lr候选0.0005/0.001/0.002、seed2026搜索，选定lr再重复2027/2028。Linear新增3搜索+2重复，共5次；Direct/Path各已有5次，合计15候选、9个主结果权重。

通过包装调用原第五轮训练函数，原代码和历史权重不改动。相同种子的模型初始状态、完整批次序列、归一化和标定都按哈希/数值核验一致。不同组选择的lr如有差异，下方同时提供同lr配对结果。

## 主结果

| 方法 | 选定lr | 验证RelL2，均值±种子SD | 最佳验证权重对应训练RelL2均值 |
| --- | ---: | ---: | ---: |
{table}

| seed | Direct | Linear | Path | Linear−Path |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(seedrows)}

seed2026的完整学习率搜索：

| lr | Direct | Linear | Path |
| --- | ---: | ---: | ---: |
{lrrows}

Linear的0.001与0.002在搜索种子上非常接近，差约0.000109；按预设最小验证值规则选择0.002，不能把这个微小排序当作稳健的超参数优劣结论。没有因为近似平局而追加搜索。

同学习率、同seed2026的主比较：

| lr | Linear | Path | Linear−Path |
| --- | ---: | ---: | ---: |
{matched}

正差值表示Path更低。不同学习率可选择各自的最佳验证步数，不是强行配对同一步数。完整记录与逐几何表保留全部结果。

![E1汇总](D:/game/research_e1/figures/e1_results.png)

![验证曲线](D:/game/research_e1/figures/validation_curves.png)

![逐几何Linear/Path配对](D:/game/research_e1/figures/linear_path_paired.png)

## 幅度、相位与场图

| 方法 | 幅度RelL2 | 真值能量加权相位MAE，rad | 预测/真值总强度比，理想为1 |
| --- | ---: | ---: | ---: |
{chr(10).join(metricrows)}

指标对各样本先平均，再对三个种子平均；不做标签相位对齐。总强度比是该标量场幅度平方和之比，不等同于端口功率/S参数守恒。

本次Path的幅度与相位误差均低于Linear，但Linear的平均总强度比更接近1。因此也不能概括为所有物理诊断指标都改善。

![固定位置验证场](D:/game/research_e1/figures/validation_fields.png)

预先固定展示验证序列位置0、3、7、seed2026，没有挑选最有利案例。每行以真值最大幅度归一化；误差色标超过1会饱和。原始复场预测保存在validation_predictions.npz，涵盖9个主结果、8个验证几何。

## 核验与资源

[verification.json](D:/game/research_e1/verification.json) 核验15次验证重放、每运行25个验证记录、最优检查点、学习率选择和三种子聚合；相同种子初始化与批次序列一致。最大验证重放绝对差为{audit['max_validation_replay_absolute_difference']:.3g}。历史51个冻结权重保持不变（第三轮21、第四轮20、第五轮10）。没有运行测试推理。

Linear新增5次训练循环合计{seconds/60:.2f}分钟，峰值已分配显存{peak:.1f}MiB。时间包含周期验证和保存，排除特征准备、最终训练指标重算及报告，不能当作部署吞吐比较。每模型参数元素数1,593,058（复数计一个），实标量自由度2,944,738。

## 适用范围与后续

这是同一Fourier配置、共享物理/几何信息、64训练与有限1200步预算下的输出表示消融。不能外推到所有数据量、器件、波长、模式数或充分优化后的方法。三个种子中2026参与lr选择；验证集仅8个几何且已反复使用，因此标准差不是新几何置信区间。

原生NeurOLight的已知入射复场、多端口叠加增强、cMAE+TV与非零drop-path尚未复现，不应称为原论文完整复现。Path是近似几何距离和训练标定波数，不是严格Maxwell光程解。

E3训练充分性和E2独立测试都未在本轮启动。下一步应按既定方案检查训练预算，再在所有模型、指标和数据规则冻结后，用未参与开发的新几何确认。

代码入口：[run_linear.py](D:/game/research_e1/run_linear.py)。预设协议：[PROTOCOL.md](D:/game/research_e1/PROTOCOL.md)。来源清单：[manifest.json](D:/game/research_e1/manifest.json)。逐样本：[validation_samples.csv](D:/game/research_e1/validation_samples.csv)。
'''
    (ROOT/'E1完整消融报告.md').write_text(report,encoding='utf-8')
    (ROOT/'README.md').write_text('''# E1完成

先读 [E1完整消融报告](D:/game/research_e1/E1完整消融报告.md)。新增5次Linear训练并与冻结Direct/Path组成完整三表示对照。仅验证集探索，未启动E2/E3。

在PowerShell重算核验：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_e1/verify_results.py
```

生成报告和图：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_e1/make_report.py
```

复核依赖现有research_v5原运行与10个权重、research_v4/research_v3历史冻结清单及权重、原80样本数据与官方源码。ZIP为现有工作目录的补充包，包含新增5个权重和日志、主结果预测、报告图表、所需支持代码与原作者MIT许可；不分发原始数据、PDF或历史51个权重。保留原相对目录关系。
''',encoding='utf-8')
    (ROOT/'STATUS.md').write_text('E1新增5次Linear训练、完整三表示比较、15次验证重放已完成。E2/E3未启动。\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'picflow_e1_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip']
        paths += [V5/'fourier_full.py']
        paths += [BASE/'research_v4'/p for p in ['official_architecture.py','official_bridge.py','stage_a.py']]
        paths += [BASE/'research_v3'/p for p in ['experiment.py','evaluate_fresh.py']]
        upstream=BASE/'references/NeurOLight-source/NeurOLight-main'
        paths += [upstream/p for p in ['LICENSE','core/models/pde_base.py','core/models/neurolight_cnn.py','core/models/layers/neurolight_conv2d.py','core/models/layers/fno_conv2d.py']]
        for p in sorted(paths):z.write(p,p.relative_to(BASE))
    print('Report, four figures, result ZIP created. MiB:',(ROOT/'picflow_e1_results.zip').stat().st_size/1024**2)


if __name__=='__main__':main()
