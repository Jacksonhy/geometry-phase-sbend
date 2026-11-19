"""Generate publication-style static plots and an evidence-linked research report."""
import csv
import hashlib
import json
import platform
import zipfile
from pathlib import Path
import numpy as np
import scipy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, geometry_features, write_json

def read(path): return json.loads(path.read_text(encoding='utf-8'))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
main=read(ROOT/'fresh_evaluation/results.json')
agg=read(ROOT/'fresh_evaluation/summary.json')
sup=read(ROOT/'exposure_matched/results.json')
supagg=read(ROOT/'exposure_matched/summary.json')
baseline=read(ROOT/'fresh_evaluation/baselines.json')
phase=read(ROOT/'fresh_evaluation/phase_diagnostics.json')
old=read(ROOT/'results.json')
colors={'raw':'#87929c','linear':'#d88128','path':'#16897f'}

def save(fig,name,vector=True):
    fig.savefig(figdir/(name+'.png'),bbox_inches='tight',facecolor='white')
    if vector: fig.savefig(figdir/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def comparison():
    keys=['raw_n32','linear_n32','path_n32','linear_n64','path_n64']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.15,1]})
    a=axes[0]
    vals=[agg['groups'][k]['metrics']['rel_l2']['mean'] for k in keys]
    sd=[agg['groups'][k]['metrics']['rel_l2']['seed_sd'] for k in keys]
    bars=a.bar(np.arange(5),vals,yerr=sd,color=[colors[k.split('_')[0]] for k in keys],capsize=4,width=.68)
    for b,m,s in zip(bars,vals,sd): a.text(b.get_x()+b.get_width()/2,m+s+.025,f'{m:.3f}',ha='center',fontsize=10)
    a.axhline(1,color='#596773',linestyle='--',linewidth=1,label='Zero-field baseline')
    a.set_xticks(np.arange(5),['Raw\n32','Linear\n32','Path\n32','Linear\n64','Path\n64'])
    a.set_ylim(0,1.13); a.set_ylabel('Complex-field relative L2 (lower is better)')
    a.set_xlabel('Representation / training sample count; 600 updates each')
    a.set_title('Frozen models: 37 new test geometries')
    a.legend(frameon=False,loc='upper right',fontsize=9)
    a.text(.01,-.25,'Bars: mean of 3 seeds; error bars: seed SD',transform=a.transAxes,fontsize=9,color='#4a5660')
    a=axes[1]
    rs={k:sorted([r for r in main if r['mode']==k and r['n']==32],key=lambda r:r['seed']) for k in ['linear','path']}
    x=np.mean([[s['rel_l2'] for s in r['samples']] for r in rs['linear']],axis=0)
    y=np.mean([[s['rel_l2'] for s in r['samples']] for r in rs['path']],axis=0)
    ports=np.array([s['port'] for s in rs['linear'][0]['samples']])
    for p,c in [(1,'#3563a1'),(2,'#b85050'),(3,'#8b66a8')]:
        a.scatter(x[ports==p],y[ports==p],s=36,alpha=.8,label=f'Port {p} (n={(ports==p).sum()})',color=c)
    bounds=[.2,1.05]
    a.plot(bounds,bounds,color='#687580',linestyle='--',linewidth=1)
    a.set_xlim(bounds); a.set_ylim(bounds); a.set_aspect('equal',adjustable='box')
    a.set_xlabel('Linear carrier RelL2 (seed average)'); a.set_ylabel('Path carrier RelL2 (seed average)')
    a.set_title('30 improved; 7 worsened (32 training samples)')
    a.legend(frameon=False,fontsize=9)
    fig.tight_layout(w_pad=2.5)
    save(fig,'fresh_comparison')


def budget_and_ports():
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    a=axes[0]
    for mode in ['linear','path']:
        objects=[agg['groups'][f'{mode}_n32']['metrics']['rel_l2'],agg['groups'][f'{mode}_n64']['metrics']['rel_l2'],supagg[mode]['metrics']['rel_l2']]
        a.errorbar(np.arange(3),[v['mean'] for v in objects],yerr=[v['seed_sd'] for v in objects],
                   marker='o',capsize=4,color=colors[mode],label=mode.title())
    a.set_xticks(np.arange(3),['32 samples\n600 updates\n150 exposures','64 samples\n600 updates\n75 exposures','64 samples\n1,200 updates*\n150 exposures'])
    a.set_ylabel('Fresh-test mean relative L2'); a.set_ylim(.43,.72)
    a.set_title('Budget control (exposures are expectations)'); a.legend(frameon=False)
    a.text(.01,-.27,'*Supplemental experiment; twice the update budget.',transform=a.transAxes,fontsize=9,color='#4a5660')
    a=axes[1]
    x=np.arange(3)
    for offset,mode in [(-.18,'linear'),(.18,'path')]:
        values=[agg['groups'][f'{mode}_n32']['port_rel_l2'][str(p)] for p in [1,2,3]]
        a.bar(x+offset,values,width=.33,color=colors[mode],label=mode.title())
    a.set_xticks(x,['Port 1\nn=18','Port 2\nn=12','Port 3\nn=7'])
    a.set_ylim(0,.85); a.set_ylabel('Mean relative L2'); a.set_title('32 samples / 600 updates: all ports improve')
    a.legend(frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig,'budget_and_ports')


def curves():
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    for mode in ['raw','linear','path']:
        histories=[read(ROOT/'runs'/f'{mode}_n32_seed{s}'/'history.json') for s in [2026,2027,2028]]
        steps=[r['step'] for r in histories[0]]
        for a,key in zip(axes,['train_rel_l2','val_rel_l2']):
            y=np.array([[r[key] for r in h] for h in histories])
            a.plot(steps,y.mean(axis=0),color=colors[mode],label=mode.title())
            a.fill_between(steps,y.mean(axis=0)-y.std(axis=0,ddof=1),y.mean(axis=0)+y.std(axis=0,ddof=1),color=colors[mode],alpha=.13)
            a.set_xlabel('Optimizer updates'); a.set_ylabel('Complex-field relative L2')
    axes[0].set_title('32 training samples: training curves')
    axes[1].set_title('Original 8 validation samples: selection curves')
    for a in axes: a.legend(frameon=False); a.set_ylim(.4,1.13)
    fig.tight_layout()
    save(fig,'training_curves')


def fields():
    with np.load(ROOT/'fresh_test.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    truth=d['Ez_real']+1j*d['Ez_imag']
    fixed={mode:np.load(ROOT/'fresh_evaluation'/f'{mode}_n32_seed2026_predictions.npz')['prediction'] for mode in ['linear','path']}
    error=np.sqrt((abs(fixed['path']-truth)**2).sum((1,2))/(abs(truth)**2).sum((1,2)))
    order=np.argsort(error)
    delta=np.array(agg['comparisons']['32']['per_geometry_difference'])
    candidates=[('Best path case',int(order[0])),('Median path case',int(order[len(order)//2])),('Worst path case',int(order[-1])),('Largest path regression',int(np.argmin(delta)))]
    fig,axes=plt.subplots(4,5,figsize=(16,6.5),layout='constrained')
    for row,(label,i) in enumerate(candidates):
        eps=d['eps'][i]; src=d['src_mask'][i]
        vmax=float(np.max(abs(truth[i].real)))
        arrays=[eps,truth[i].real,fixed['linear'][i].real,fixed['path'][i].real,abs(fixed['path'][i]-truth[i])]
        titles=['Permittivity / source','FDTD Re(Ez)','Linear Re(Ez)','Path Re(Ez)','Path |complex error|']
        for col,(a,img,title) in enumerate(zip(axes[row],arrays,titles)):
            if col==0:
                im=a.imshow(img,cmap='gray',origin='lower',aspect='equal'); yy,xx=np.where(src>0); a.scatter(xx,yy,s=.6,color='#eb4a46')
            elif col==4:
                im=a.imshow(img,cmap='magma',origin='lower',aspect='equal',vmin=0,vmax=float(abs(truth[i]).max()))
            else:
                im=a.imshow(img,cmap='RdBu_r',origin='lower',aspect='equal',vmin=-vmax,vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if row==0: a.set_title(title,fontsize=10)
            if col in [1,4]: fig.colorbar(im,ax=a,fraction=.033,pad=.01)
        axes[row,0].set_ylabel(f'{label}\nindex={i}, port={int(d["input_port"][i])}\nPath RelL2={error[i]:.3f}',fontsize=9)
    fig.suptitle('New test fields; fixed seed=2026, 32 training samples, 600 updates\nShared real-field color range within each row; all four cases are shown without phase alignment.',fontsize=12)
    save(fig,'field_cases',vector=False)
    write_json(ROOT/'fresh_evaluation/visualized_cases.json',[dict(reason=label,index=i,geometry_id=str(d['geometry_id'][i]),port=int(d['input_port'][i]),fixed_seed_path_rel_l2=float(error[i])) for label,i in candidates])


def phase_and_regions():
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    a=axes[0]; x=np.arange(2)
    for offset,mode in [(-.22,'raw'),(0,'linear'),(.22,'path')]:
        vals=[phase['train32']['summary'][r][mode] for r in ['trunk','bend']]
        a.bar(x+offset,vals,width=.21,color=colors[mode],label=mode.title())
    a.set_xticks(x,['Straight trunk','Curved branches']); a.set_ylabel('Residual phase-increment RMS (rad / pixel)')
    a.set_title('Training-label diagnostic: carrier smoothness'); a.legend(frameon=False)
    a=axes[1]; x=np.arange(4)
    keys=['core','trunk','source_side','far_side']
    for offset,mode in [(-.18,'linear'),(.18,'path')]:
        a.bar(x+offset,[agg['groups'][f'{mode}_n32']['regions'][r] for r in keys],width=.33,color=colors[mode],label=mode.title())
    a.set_xticks(x,['Core','Trunk','Near-source\nhalf-domain','Far-source\nhalf-domain'])
    a.set_ylabel('Region-normalized relative L2'); a.set_ylim(0,.9)
    a.set_title('New-test diagnosis: far-side errors remain'); a.legend(frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig,'phase_and_regions')


def outputs():
    rows=[]
    for label,rs in [('frozen_main',main),('supplemental_budget',sup)]:
        for r in rs:
            for s in r['samples']:
                rows.append(dict(experiment=label,mode=r['mode'],train_n=r['n'],seed=r['seed'],steps=600 if label=='frozen_main' else 1200,**s))
    with (ROOT/'fresh_test_per_sample.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
             matplotlib=matplotlib.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
             platform=platform.platform(),cpu_threads=4,tf32_enabled=True,checkpoint_parameters=123330)
    write_json(ROOT/'environment.json',env)
    write_json(ROOT/'source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})


def report():
    groups=agg['groups']
    lines=[]
    for mode,n in [('raw',32),('linear',32),('path',32),('linear',64),('path',64)]:
        g=groups[f'{mode}_n{n}']; m=g['metrics']['rel_l2']
        train=np.mean([r['splits']['train']['rel_l2'] for r in old if r['mode']==mode and r['n']==n])
        val=np.mean([r['splits']['val']['rel_l2'] for r in old if r['mode']==mode and r['n']==n])
        oldtest=np.mean([r['splits']['test']['rel_l2'] for r in old if r['mode']==mode and r['n']==n])
        lines.append(f'| {mode} | {n} | 600 | {train:.4f} | {val:.4f} | {oldtest:.4f} | {m["mean"]:.4f} ± {m["seed_sd"]:.4f} | {g["port_macro_rel_l2"]:.4f} |')
    table='\n'.join(lines)
    complines=[]
    for n,c in agg['comparisons'].items():
        lo,hi=c['bootstrap_geometry_ci95']
        complines.append(f'- {n} 样本：linear−path = {c["linear_minus_path_mean"]:.4f}，相对降低 {100*c["relative_error_reduction"]:.2f}%；按几何成对 bootstrap 的条件95%区间 [{lo:.4f}, {hi:.4f}]；{c["positive_geometries"]}/37 改善，{c["negative_geometries"]}/37 退步。')
    porttable='\n'.join(f'| {p} | {count} | {groups["linear_n32"]["port_rel_l2"][str(p)]:.4f} | {groups["path_n32"]["port_rel_l2"][str(p)]:.4f} |' for p,count in [(1,18),(2,12),(3,7)])
    suptable='\n'.join(f'| {mode} | 64 | 1200 | {g["old_splits"]["val"]:.4f} | {g["old_splits"]["test"]:.4f} | {g["metrics"]["rel_l2"]["mean"]:.4f} ± {g["metrics"]["rel_l2"]["seed_sd"]:.4f} | {g["port_macro_rel_l2"]:.4f} |' for mode,g in supagg.items())
    basetable='\n'.join(f'| {name} | {baseline["n32"][name]["overall"]["rel_l2"]:.4f} | {baseline["n64"][name]["overall"]["rel_l2"]:.4f} |' for name in baseline['n32'])
    physical='\n'.join(f'| {mode} | {groups[f"{mode}_n32"]["metrics"]["amplitude_rel_l2"]["mean"]:.4f} | {groups[f"{mode}_n32"]["metrics"]["intensity_weighted_phase_mae"]["mean"]:.4f} | {groups[f"{mode}_n32"]["metrics"]["intensity_ratio"]["mean"]:.4f} |' for mode in ['raw','linear','path'])
    report_text=f'''# PIC-Flow 第三轮研究报告：弧长相位表示的受控验证

日期：2026-11-11。所有结果来自本机 WSL / RTX 5070 Laptop GPU 对真实 FDTD 数据的执行。

## 结论

在**下载新标签前冻结的15个模型**上，沿几何中心线累计弧长的相位包络表示，优于相同网络、输入和训练预算的直线载波表示。32个训练样本、600次更新，在37个新测试几何上，平均复场RelL2从 **0.6156降至0.5482，降低10.95%**，三个种子方向一致。按种子平均后30个几何改善、7个退步。端口等权平均也从0.6450降至0.5744，收益不是单靠端口比例造成的。

补充64样本、1200次更新对照中，path达到 **0.5089 ± 0.0242**；同预算linear为0.5706 ± 0.0081。此组是额外预算诊断，与主冻结验证分开报告。模型仍有明显远源误差，不能称为高精度或实用FDTD替代器。

## 1. 数据和实验可信度

- 原始数据：现有80个Y-Branch样本，官方train/val/test=64/8/8。前32训练样本作为嵌套子集，不按结果筛选。
- 新测试：按官方索引元信息事先选择5个shard中的全部37个原始Y-Branch test记录。端口1/2/3分别18/12/7；与旧80样本无几何ID重叠、无完全相同eps图，新37张eps图也互不相同。哈希无重复不是“设计空间距离足够远”的证明。
- 仓库固定版本：`d368c45a5f5040c193fd79c5027669fab0453736`。用户在浏览器下载，随后本地离线校验、提取；文件SHA256和选择清单见[fresh_manifest.json](fresh_manifest.json)。本地哈希用于追踪文件，未声称完成服务器端哈希认证。
- 15个主模型的SHA256全部与下载前清单一致。独立推理入口只接受eps、src_mask和训练拟合常数，回放15组旧测试预测通过相对差<1e-5验证。新标签只进入指标和后验诊断。
- 原8样本测试已被前几轮使用，因此只作历史探索性对照；新37样本在本轮首次用于冻结模型验证。它们是测试样本密集shard中的便利样本，不是完整官方测试集的随机抽样。
- 新增6个预算对照模型保存于独立目录；它们不覆盖主模型，不用新测试标签选检查点或调参。三种子的标准差衡量训练随机性，不等于37个几何的不确定性。

数据背景参考：[官方数据集](https://huggingface.co/datasets/RizzoLab/PIC-Flow-Dataset)、[原论文](https://arxiv.org/abs/2605.06929)。本报告是独立的小规模监督学习实验，不是原Flow Matching方法的性能复现。

## 2. 本轮模型和公平对照

三种表示共享一个123,330参数的U-Net，通道8/16/32/64，GroupNorm+SiLU。输入从160×480平均降采样到80×240，输出复包络双通道，再双线性恢复160×480。10个输入通道包括eps、src_mask、x/y、全局源坐标、传播方向、预测k、弧长距离和直线距离。**raw/linear也得到相同弧长输入**，因此主消融改变的是输出载波表示，不是额外几何信息量。

- raw：直接预测复场。
- linear：E = A·exp(i k·d_linear)，距离以源位置为参考、按入射方向取符号。
- path：E = A·exp(i k·d_path)，从eps阈值图提取上/下分支中心线、平滑并累计弧长；右侧注入时对另一个分支使用经分叉点的路径。

弧长载波只是一种几何近似，未严格求解光程、反射或多模耦合。k只用对应训练子集直干区域的真实相位斜率拟合到eps最大值。32训练样本拟合为k≈0.0557596·eps_core+0.1251541 rad/网格，训练拟合RMSE≈0.0009312；这不是普适Maxwell公式。归一化也只来自训练子集。整个评价不做标签相位对齐。

AdamW，lr=0.002，weight_decay=1e-4，余弦衰减至初始5%，batch=8，每50步在原8个验证样本上选平均逐样本RelL2最低的检查点。种子2026/2027/2028；同种子各方法具有相同初始网络和批抽样序列。损失为逐样本归一化复场平方误差。完整配置见[原协议](PROTOCOL.md)、[新测试协议](FRESH_TEST_PROTOCOL.md)。

## 3. 主冻结结果

表内训练/验证/旧测试为三个种子的平均；新测试列为种子均值±种子标准差。RelL2为每样本复场L2误差除以真实场L2范数，再对样本平均，越低越好。端口宏平均对三个端口等权。

| 表示 | 训练N | 更新 | 训练 | 验证8 | 旧测试8 | 新测试37 | 新测试端口宏平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

![冻结测试结果和逐几何比较](figures/fresh_comparison.png)

{chr(10).join(complines)}

bootstrap先将每个几何的成对差对三个种子平均，再以几何为单位重采样10,000次。该区间描述所选几何的条件重采样波动，不代表完整设计空间的置信范围，也未校正研究方向选择。共享训练集和三个种子并不等于111个独立测试样本。

按端口分层，32样本600步：

| 端口 | 新测试数 | linear RelL2 | path RelL2 |
|---|---:|---:|---:|
{porttable}

## 4. 训练预算与数据量

32样本600步时，每个样本预期被访问150次；64样本600步仅75次。因此固定更新比较不能同时保证单样本曝光量相同。

补充64样本1200步，使预期曝光量恢复150次；三种子、架构、损失和检查点选择保持一致。所有6模型完成后先记录哈希再评价，详见[补充协议](EXPOSURE_PROTOCOL.md)。

| 表示 | 训练N | 更新 | 验证8 | 旧测试8 | 新测试37 | 端口宏平均 |
|---|---:|---:|---:|---:|---:|---:|
{suptable}

path的64样本新测试均值从600步0.5543降至1200步0.5089，约降低8.20%。这支持“原预算下尚未充分训练”的解释，但更长训练同时改变计算量和学习率时间表，不能称为增加数据的纯因果效应。32/64的传播常数校准也来自不同标签子集，因此数据量比较包含校准变化。

![预算与端口对照](figures/budget_and_ports.png)

## 5. 简单基线

以下全部只用相应训练子集。均值和最近邻限制在相同激励端口；最近邻按归一化eps图与src_mask拼接后的均方距离选取，不用真实测试场选邻居。linear/path版本在去载波包络空间计算均值或借用近邻包络，再乘新几何载波。

| 基线 | N=32新测试RelL2 | N=64新测试RelL2 |
|---|---:|---:|
{basetable}

path均值基线已优于raw均值，支持载波表示本身有作用；神经网络path仍明显优于这些简单基线。本次最近邻定义只是一个明确定义的参照，不代表所有几何匹配或插值算法的上限。零场的相位未定义，不报告其相位误差。

## 6. 改善发生在哪里，仍缺什么

32样本600步，新测试平均：

| 表示 | 幅度RelL2 | 真实强度加权圆周相位MAE（rad） | 积分场强比 |
|---|---:|---:|---:|
{physical}

相位MAE定义为sum(|E_true|²·|arg(E_pred·conj(E_true))|)/sum(|E_true|²)。积分场强比=sum|E_pred|²/sum|E_true|²；**它不是传输功率、效率或S参数**，本轮也未验证物理功率守恒。

相同32样本模型，path在靠源半域RelL2≈0.313，在远源半域≈0.696；linear分别≈0.362和0.775。波导核心域path≈0.421、linear≈0.505。左右半域只是定位误差的图像区域，不是精确传播长度。全域主指标也包含边界；interior诊断只去掉10像素边框，未将其冒称官方PML裁剪。

真实训练场的输入支路中心线诊断中，直干段raw/linear/path平均相位增量RMS约0.436/0.0190/0.0190 rad/像素；弯曲段约0.439/0.0764/0.0681。统计先在每条中心线按相邻场幅度加权，再对路径段平均。两种载波都去除了主要振荡，弧长仅进一步降低一部分弯曲段残余；这与测试改善一致，但不是唯一机制的证明。

**负面诊断也需保留：**80×240包络先平均降采样再插值回原网格的训练重建误差，linear≈0.0450、path≈0.0688，path并未在这个压缩检查中占优。因此不能简单声称它让整个场包络处处更平滑或压缩下界更低。更合理的解释是对传播几何的表示更适合当前网络学习，仍需进一步验证。

![相位与区域诊断](figures/phase_and_regions.png)

固定种子2026展示最佳、中位、最差path预测和相对linear最大退步案例；没有按新测试结果选择最佳种子。实部图同一样本共享色标，误差单独用场幅度标尺，保留失败情况。

![新测试场与失败案例](figures/field_cases.png)

![训练与验证曲线](figures/training_curves.png)

raw部分运行在继续降低训练误差的同时验证误差上升，因此本轮不能沿用“直接模型完全无法拟合多样本”的绝对说法。准确说法是：在本次训练预算、输入与验证选择规则下，直接复场表示泛化很差，而显式载波改善了学习。

## 7. 下一阶段建议与边界

当前足以支持一个有物理解释的本科研究结论：在固定波长的特定Y-Branch分布和小样本预算下，几何弧长载波改善轻量U-Net的复场预测。没有证据支持任意器件、3D、宽带或高精度求解的推广，也不能仅凭本轮声称方法学首创。

后续优先处理右端口注入和远源区域：在训练/验证数据上检查沿弯曲路径的传播常数变化、相位误差累计及反向分量，再决定是否加入双向载波或分段校准。不要依据当前37个样本的表现挑超参后继续把它称为全新测试集。

扩大到200/500/1000训练样本之前，应先冻结采样、预算和评价协议，另留不参与开发的新测试几何，并保留端口分层。当前本地材料已足够复核本轮；继续扩大训练集需要另行准备数据，不在本轮自动下载。

## 8. 复现与交付

Windows工作目录：`D:\\game`。WSL解释器：`/home/cudauser/.venvs/picflow-gpu/bin/python`。

```powershell
Set-Location D:\\game
python .\\research_v3\\prepare_fresh.py --offline
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_v3/evaluate_fresh.py
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_v3/evaluate_exposure.py
wsl -d Ubuntu-24.04 -- /home/cudauser/.venvs/picflow-gpu/bin/python /mnt/d/game/research_v3/verify_results.py
```

不要直接再次运行run_suite.py覆盖主模型；fresh_manifest已冻结权重。需要重新训练时先复制项目到新目录，或给train_one传独立output_root。完整运行环境见[environment.json](environment.json)，源代码哈希见[source_hashes.json](source_hashes.json)。

交付：21个模型权重、主/补充协议、逐步训练曲线、逐样本CSV、原始指标JSON、PNG/SVG科学图、重放与数学恒等式检查结果。新增代码还修正了“无分叉直波导、从右侧注入”这一几何辅助函数边界情形；实际Y-Branch都有分叉，主15模型的旧预测重放不受影响。

- [逐新测试样本CSV](fresh_test_per_sample.csv)
- [主冻结验证摘要](fresh_evaluation/summary.json)
- [补充预算结果](exposure_matched/summary.json)
- [检查结果](verification.json)
- [结果包](picflow_v3_results.zip)：包括代码、权重、报告与图表；为控制体积，不重复打包原始数据或全部预测数组。所有预测仍保留在本机fresh_evaluation和exposure_matched/runs目录。
'''
    (ROOT/'研究报告.md').write_text(report_text,encoding='utf-8')


def package():
    path=ROOT/'picflow_v3_results.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts: continue
            if p.suffix.lower() not in ['.py','.md','.json','.csv','.pt','.png','.svg']: continue
            z.write(p,p.relative_to(ROOT))
    print('Report and figures saved. Package bytes:',path.stat().st_size)


if __name__=='__main__':
    comparison(); budget_and_ports(); curves(); fields(); phase_and_regions(); outputs(); report(); package()
