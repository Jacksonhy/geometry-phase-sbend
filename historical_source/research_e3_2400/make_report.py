"""Build E3 figures and a factual report from verified, frozen results."""
import csv
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
DISPLAY = 'D:/game/research_e3_2400'
VARIANTS = ['direct', 'linear', 'path']
LABELS = dict(direct='Direct', linear='Linear', path='Path')
COLORS = dict(direct='#7b8794', linear='#397ca5', path='#18836f')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def oldrun(variant):
    folder = 'research_e1' if variant == 'linear' else 'research_v5'
    return BASE/folder/'runs'/f'{variant}_lr0.002_seed2026'


def main():
    audit = read(ROOT/'verification.json')
    assert audit['status'] == 'passed' and audit['seed'] == 2026
    assert not audit['test_evaluated']
    comp = audit['comparisons']
    pair = audit['paired_comparisons']
    runs = {r['variant']: r for r in read(ROOT/'results.json')}
    diagnostic = read(ROOT/'validation_diagnostics.json')
    frozen = read(ROOT/'frozen_weights.json')
    for name, digest in frozen.items():
        assert sha(ROOT/name) == digest
    figdir = ROOT/'figures'
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    displayed = [comp[v][k] for v in VARIANTS for k in ['best1200', 'best2400', 'terminal1200_logged', 'terminal2400']]
    ybounds = (max(0, min(displayed)-.08), max(displayed)+.08)
    for i, v in enumerate(VARIANTS):
        c = comp[v]
        for ax, keys in zip(axes, [('best1200', 'best2400'), ('terminal1200_logged', 'terminal2400')]):
            vals = [c[k] for k in keys]
            ax.plot([1200, 2400], vals, '-o', color=COLORS[v], label=LABELS[v])
            for x, y in zip([1200, 2400], vals):
                ax.annotate(f'{y:.4f}', (x, y), xytext=(0, -16 if v == 'path' else 8),
                            textcoords='offset points', ha='center', fontsize=9, color=COLORS[v])
        for ax in axes:
            ax.set_xticks([1200, 2400])
            ax.set(xlim=(1030, 2570), ylim=ybounds, xlabel='Total optimizer updates', ylabel='Validation RelL2')
            ax.grid(alpha=.2)
    axes[0].set_title('Validation-selected checkpoints (25 vs 49 choices)')
    axes[1].set_title('Fixed terminal update (1200 from original logs)')
    axes[0].legend()
    fig.suptitle('E3: training budget, seed 2026 only — lower is better')
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.text(.5, .015, 'Separate cosine horizons; 64 training / 8 reused validation geometries. No independent test.', ha='center', fontsize=9)
    fig.savefig(figdir/'budget_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        for budget, linestyle, alpha in [(1200, '--', .6), (2400, '-', 1)]:
            folder = oldrun(v) if budget == 1200 else ROOT/'runs'/f'{v}_steps2400_seed2026'
            hist = read(folder/'history.json')
            steps = np.array([h['step'] for h in hist])
            values = np.array([h['val_rel_l2'] for h in hist])
            ax.plot(steps, values, linestyle, color=COLORS[v], alpha=alpha, label=f'{budget}-step schedule')
            idx = int(values.argmin())
            ax.scatter(steps[idx], values[idx], s=45, color=COLORS[v], marker='o' if budget == 2400 else 's', zorder=5)
        ax.axvline(1200, color='gray', linewidth=.7, alpha=.4)
        ax.set(title=LABELS[v], xlabel='Optimizer updates', xlim=(0, 2450))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation RelL2')
    fig.suptitle('Raw validation curves; markers indicate selected checkpoints; seed 2026')
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.text(.5, .015, 'Separate cosine horizons change the learning-rate trajectory. No smoothing or seed error bands.', ha='center', fontsize=9)
    fig.savefig(figdir/'validation_curves.png', dpi=180)
    plt.close(fig)

    with (ROOT/'validation_samples.csv').open(encoding='utf-8', newline='') as f:
        samples = list(csv.DictReader(f))
    geom_ids = [r['geometry_id'] for r in samples if r['variant'] == 'linear' and r['checkpoint'] == '1200_best']

    def values(v, tag):
        lookup = {r['geometry_id']: float(r['val_rel_l2']) for r in samples if r['variant'] == v and r['checkpoint'] == tag}
        return np.array([lookup[g] for g in geom_ids])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True, sharey=True)
    limit = max(float(values(v, f'{s}_best').max()) for v in ['linear', 'path'] for s in [1200, 2400])*1.08
    for ax, s in zip(axes, [1200, 2400]):
        a, b = values('linear', f'{s}_best'), values('path', f'{s}_best')
        p = pair[f'{s}_linear_minus_path']
        ax.scatter(a, b, color=COLORS['path'], s=35)
        for i, (x, y) in enumerate(zip(a, b)):
            ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.plot([0, limit], [0, limit], '--', color='gray', linewidth=1)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='Linear validation RelL2',
               title=f'{s} updates: Path improves {p["improved_geometries"]}/8 geometries')
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Path validation RelL2')
    fig.suptitle('Per-geometry paired comparison; below diagonal favors Path')
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.text(.5, .015, 'Labels are fixed validation positions 0–7; each model uses its own validation-selected checkpoint.', ha='center', fontsize=9)
    fig.savefig(figdir/'linear_path_paired.png', dpi=180)
    plt.close(fig)

    with np.load(ROOT/'validation_predictions.npz', allow_pickle=False) as z:
        truth = z['truth'][:, 0]
        names = list(z['run'])
        eps = z['eps']
        pred = {v: z['prediction'][names.index(v+'_2400_best'), :, 0] for v in VARIANTS}
        fig, axes = plt.subplots(3, 6, figsize=(17, 5.5))
        for row, idx in enumerate([0, 3, 7]):
            scale = max(float(abs(truth[idx]).max()), 1e-8)
            fields = [truth[idx].real]+[pred[v][idx].real for v in VARIANTS]+[abs(pred[v][idx]-truth[idx]) for v in ['linear', 'path']]
            for col, field in enumerate(fields):
                ax = axes[row, col]
                ax.imshow(field/scale, origin='lower', cmap='RdBu_r' if col < 4 else 'magma', vmin=-1 if col < 4 else 0, vmax=1)
                ax.contour(eps[idx], levels=[(eps[idx].min()+eps[idx].max())/2], colors='gray', linewidths=.3, alpha=.7)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(['Truth Re(E)', 'Direct Re(E)', 'Linear Re(E)', 'Path Re(E)', 'Linear |error|', 'Path |error|'][col])
                if col == 0:
                    ax.set_ylabel(str(z['geometry_id'][idx]).replace('ybranch_', ''), fontsize=8)
        fig.suptitle('2400-budget best checkpoints; fixed validation positions 0, 3, 7; seed 2026')
        fig.tight_layout(rect=(0, .04, 1, .93))
        fig.text(.5, .015, 'Each row scaled by truth max |E|. Signed fields: −1 to 1; error: 0 to 1 (larger errors saturate).', ha='center', fontsize=9)
        fig.savefig(figdir/'validation_fields.png', dpi=180)
        plt.close(fig)

    bestrows = '\n'.join(f"| {LABELS[v]} | {comp[v]['best1200']:.6f} | {comp[v]['best_step1200']} | {comp[v]['best2400']:.6f} | {comp[v]['best_step2400']} | {comp[v]['relative_best_error_reduction']:+.2%} |" for v in VARIANTS)
    lastrows = '\n'.join(f"| {LABELS[v]} | {comp[v]['terminal1200_logged']:.6f} | {comp[v]['terminal2400']:.6f} | {comp[v]['relative_terminal_error_reduction']:+.2%} |" for v in VARIANTS)
    trainrows = '\n'.join(f"| {LABELS[v]} | {comp[v]['train_at_best1200']:.6f} | {comp[v]['train_at_best2400']:.6f} | {comp[v]['train_terminal2400']:.6f} | {comp[v]['terminal2400']:.6f} |" for v in VARIANTS)
    metricrows = []
    for s in [1200, 2400]:
        for v in VARIANTS:
            d = diagnostic[f'{v}_{s}_best']
            metricrows.append(f"| {s} | {LABELS[v]} | {d['amplitude_rel_l2']:.4f} | {d['intensity_weighted_phase_mae']:.4f} | {d['intensity_ratio']:.4f} |")
    geo = {s: {v: values(v, f'{s}_best') for v in ['linear', 'path']} for s in [1200, 2400]}
    georows = '\n'.join(f"| {i} | {g} | {geo[1200]['linear'][i]:.6f} | {geo[1200]['path'][i]:.6f} | {geo[2400]['linear'][i]:.6f} | {geo[2400]['path'][i]:.6f} | {geo[2400]['linear'][i]-geo[2400]['path'][i]:+.6f} |" for i, g in enumerate(geom_ids))
    p12, p24 = [pair[f'{s}_linear_minus_path'] for s in [1200, 2400]]
    terminal_gain = 1-runs['path']['final_val']/runs['linear']['final_val']
    direct_path_gain = pair['2400_direct_minus_path']['relative_error_reduction']
    verdict = '仍低于' if p24['relative_error_reduction'] > 0 else '未低于'
    total_seconds = sum(r['train_wall_seconds'] for r in runs.values())
    resource_rows = '\n'.join(f"| {LABELS[v]} | {runs[v]['train_wall_seconds']/60:.2f} | {runs[v]['peak_allocated_MiB']:.1f} | {runs[v]['resume_count']} |" for v in VARIANTS)
    report = f'''# E3：2400步训练预算比较

日期：2026-11-12。新增Direct / Linear / Path各一次2400步训练，所有新增实验只使用seed2026。

## 主要结果

在本次2400步预算下，Path平均验证RelL2{verdict}Linear，Linear→Path相对误差下降为 **{p24['relative_error_reduction']:.2%}**；1200步同种子对应值为 **{p12['relative_error_reduction']:.2%}**。2400步Path相对Direct下降{direct_path_gain:.2%}。

增加到2400步后，Linear和Path均小幅改善，Direct的最优验证误差基本持平，Path对Linear的额外收益继续存在。这支持当前配置与预算范围内的表示收益；不能据此断言Direct在所有优化方案下都无法追近。

| 方法 | 1200步最优验证RelL2 | 选中步数 | 2400步最优验证RelL2 | 选中步数 | 增预算相对误差下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
{bestrows}

相对下降定义为1−新误差/旧误差，负号表示变差。上表1200与2400均为seed2026，未混用历史三种子均值。每个预算下按各方法自己的最低验证均值选检查点，RelL2为逐样本复场相对L2再平均。

![预算比较]({DISPLAY}/figures/budget_comparison.png)

## 固定末步与训练误差

2400步有49次验证选择机会，1200步有25次；单纯比较最优值会同时包含训练预算与检查点选择机会的变化，因此补报固定末步：

| 方法 | 1200末步验证，原日志 | 2400末步验证，已重放 | 增预算相对误差下降 |
| --- | ---: | ---: | ---: |
{lastrows}

固定2400末步时，Path相对Linear的验证误差下降为{terminal_gain:.2%}。1200末步权重历史上未保留，这一列只能引用原日志；没有重训、插值或把它标成末步权重重放。

| 方法 | 1200最优权重训练RelL2 | 2400最优权重训练RelL2 | 2400末步训练RelL2 | 2400末步验证RelL2 |
| --- | ---: | ---: | ---: | ---: |
{trainrows}

Direct的2400末步训练RelL2为{runs['direct']['final_train']:.6f}，验证为{runs['direct']['final_val']:.6f}；其最优验证值相对1200预算变化{comp['direct']['relative_best_error_reduction']:+.2%}。判断是否追近要结合验证差距，不能只看更低的训练损失。

训练RelL2在eval模式下对全部64训练样本计算，不是训练日志中两个样本的带dropout批次损失。训练误差下降但验证误差未相应改善时，增加更新数不足以说明泛化已解决；2400步也不能被称为理论上充分收敛或排除了所有优化问题。

![完整验证曲线]({DISPLAY}/figures/validation_curves.png)

## Path对Linear的逐几何结果

1200步{p12['improved_geometries']}/8个验证几何改善，2400步{p24['improved_geometries']}/8个改善、{p24['worsened_geometries']}/8个变差。所有几何均列出；平均收益不代表每个器件受益。

| 固定位置 | 几何ID | Linear1200 | Path1200 | Linear2400 | Path2400 | 2400 Linear−Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{georows}

![逐几何比较]({DISPLAY}/figures/linear_path_paired.png)

## 幅度与相位诊断

下表均为各预算最优权重，不做目标场相位对齐。

| 预算 | 方法 | 幅度RelL2 | 真值能量加权相位MAE，rad | 预测/真值总强度比 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(metricrows)}

总强度比的理想值是1；该指标为标量场幅度平方和之比，不等同于端口功率或S参数守恒。不能根据复场主指标推断每个物理诊断指标都会改善。

![固定验证场图]({DISPLAY}/figures/validation_fields.png)

固定展示验证序列位置0、3、7，均使用2400预算最优权重，没有按效果挑选案例。每行用真值最大幅度归一化，误差色标超过1会饱和。validation_predictions.npz保存9组权重的全部8个验证预测与真值。

## 设置与比较的边界

沿用官方NeurOLight Fourier主干PIC-Flow适配，官方源码commit为08cad10a78a01e4a522f8b4d287db0b363054e33：12层×64，modes40/70、head256、原卷积stem与FFN、dropout0.1、drop_path0、aug_path=False，均有官方exp wave prior。输入、prior、输出、监督均为160×480完整分辨率。每模型参数元素数{runs['direct']['parameters']:,}（复数计一个），实标量自由度{runs['direct']['real_scalar_degrees']:,}。

64训练+原8验证；三组共享全部10基础输入、训练标定k、尺度、相同初始化及2400批次序列，前1200批次与原结果一致。Linear也接收几何路径输入；对比只改变输出的固定载波表示。Direct直接预测E，Linear预测包络后乘直线载波，Path乘几何弧长载波；损失均在恢复后的物理复场计算。

本轮固定lr0.002，无新增搜索；FP32允许TF32，batch2，AdamW wd1e-4，梯度裁剪5，无增强。三组从原相同初始化重新训练，使用总长2400的余弦调度，末lr0.0001。1200基线使用总长1200的余弦调度，所以两组从开头的学习率轨迹就不同；本实验是两个总预算方案的比较，不能声称是同一训练轨迹直接延长。

只使用一个种子，不报告种子标准差或跨种子稳定性。原8验证几何已反复用于开发与选模，这些结果不能代替E2独立测试，也不构成统计显著性证明。它们只能回答本配置增加到2400步后的表现；不能排除其他学习率、调度、训练量和架构配置下Direct追近的可能。

这是原Fourier算子的适配实验，原生NeurOLight已知入射复场、多端口叠加增强、原论文损失等未完整复现。几何路径是近似载波，不是严格Maxwell光程解。

## 核验、时间与文件

核验通过：9次验证权重重放（3个历史1200最优、3个新增2400最优、3个2400末步）；最大均值绝对重放差为{audit['max_replay_absolute_difference']:.3g}。校验初始化、完整批次序列、前1200批次前缀、49行日志、余弦学习率、最优权重选择和6个新权重哈希。2400末步另校验全部AdamW状态步数、调度器、随机数状态，并重放训练指标。历史{audit['old_frozen_weights_preserved']}个冻结权重保持不变。未运行新测试推理、4800步训练或网络请求。

| 方法 | 训练循环分钟 | 峰值已分配显存，MiB | 恢复次数 |
| --- | ---: | ---: | ---: |
{resource_rows}

三组训练循环合计{total_seconds/60:.2f}分钟，包含周期验证和保存，不含特征准备、最终指标重算、复核和报告。该时间不作为部署吞吐比较。

[预设协议]({DISPLAY}/PROTOCOL.md) · [核验结果]({DISPLAY}/verification.json) · [来源清单]({DISPLAY}/manifest.json) · [逐样本表]({DISPLAY}/validation_samples.csv) · [训练代码]({DISPLAY}/train.py)
'''
    (ROOT/'E3_2400步报告.md').write_text(report, encoding='utf-8')
    (ROOT/'README.md').write_text(f'''# E3 2400步已完成

先读[报告]({DISPLAY}/E3_2400步报告.md)。Direct / Linear / Path各训练2400步，只用seed2026。保留最优和完整末步状态；未启动4800步或独立测试。

PowerShell复核（依赖当前工作目录中的数据、官方源码和历史冻结运行）：

```powershell
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_e3_2400/verify_results.py
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_e3_2400/make_report.py
```

训练入口为同目录train.py；重复启动会校验清单并复用已完成运行。中断时使用last.pt中的完整AdamW、调度器与随机数状态恢复。

ZIP为现有项目的补充包，包含本轮6个权重（3最优+3完整末步）、日志、报告、预测、支持代码和原作者MIT许可。保留原相对目录关系。ZIP不包含原数据、历史56个权重或全部历史清单；完整重放核验需要现有项目。运行环境和哈希在runtime_and_sources.json中。
''', encoding='utf-8')
    (ROOT/'STATUS.md').write_text('E3的2400步阶段完成：3次seed2026训练、9次验证重放、历史56权重保全。4800步与E2未启动。\n', encoding='utf-8')
    runtime = dict(date='2026-11-12', python=sys.version, platform=platform.platform(), torch=torch.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), numpy=np.__version__, matplotlib=matplotlib.__version__,
                   source_hashes={p.name: sha(p) for p in [ROOT/'train.py', ROOT/'verify_results.py', Path(__file__)]})
    write(ROOT/'runtime_and_sources.json', runtime)
    paths = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip', '.tmp'] and p.name not in ['artifacts.json', 'package_verification.json']]
    paths += [BASE/'research_e1/run_linear.py', BASE/'research_v5/fourier_full.py']
    paths += [BASE/'research_v4'/p for p in ['official_architecture.py', 'official_bridge.py', 'stage_a.py']]
    paths += [BASE/'research_v3'/p for p in ['experiment.py', 'evaluate_fresh.py']]
    upstream = BASE/'references/NeurOLight-source/NeurOLight-main'
    paths += [upstream/p for p in ['LICENSE', 'core/models/pde_base.py', 'core/models/neurolight_cnn.py', 'core/models/layers/neurolight_conv2d.py', 'core/models/layers/fno_conv2d.py']]
    paths = sorted(set(paths))
    write(ROOT/'artifacts.json', {p.relative_to(BASE).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})
    paths += [ROOT/'artifacts.json']
    archive = ROOT/'picflow_e3_2400_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(BASE))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        entries = len(z.infolist())
        assert len([n for n in z.namelist() if n.endswith('.pt')]) == 6
    package = dict(status='passed', entries=entries, bytes=archive.stat().st_size, sha256=sha(archive), crc_verified=True, new_checkpoints=6)
    write(ROOT/'package_verification.json', package)
    print(json.dumps(dict(report=str(ROOT/'E3_2400步报告.md'), training_minutes=total_seconds/60, package=package), ensure_ascii=False))


if __name__ == '__main__':
    main()
