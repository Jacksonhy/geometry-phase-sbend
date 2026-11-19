"""Independent arithmetic and preservation checks; no neural inference or training."""
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;OLD=B/'research_e3_2400'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024**2),b''):h.update(c)
    return h.hexdigest()
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
table=csvrows(P/'test_per_geometry.csv');assert len(table)==1200
keys=['dataset','method','checkpoint','geometry_id']
assert len({tuple(r[k] for k in keys) for r in table})==1200
metrics=['rel_l2','amplitude_rel_l2','intensity_ratio','intensity_weighted_phase_mae']
maximum=0.;count=0;inherited=0
for ds in ['E2','M4']:
    hist=B/('research_e2' if ds=='E2' else 'research_m4')
    historical={}
    for r in csvrows(hist/'test_per_sample.csv'):
        if ds=='E2' and r['model'] in ['linear_2400','path_2400']:
            historical[(r['variant'],'best',r['geometry_id'])]=r
        elif ds=='M4' and r['variant'] in ['linear','path']:
            historical[(r['variant'],r['checkpoint'],r['geometry_id'])]=r
    with np.load(hist/'test_materials.npz',allow_pickle=False) as z:
        gids=z['geometry_id'];ports=z['input_port']
        truth=z['truth'] if ds=='E2' else z['Ez_real']+1j*z['Ez_imag']
    for method in ['linear','path','branch_tangent']:
        for ck in ['best','last']:
            new=method=='branch_tangent' or (ds=='E2' and ck=='last')
            path=P/'predictions'/f'{ds}_{method}_{ck}.npz' if new else hist/'predictions'/(f'{method}_2400.npz' if ds=='E2' else f'{method}_{ck}.npz')
            with np.load(path,allow_pickle=False) as z:
                pred=z['prediction'];ids=z['geometry_id'];pp=z['input_port']
            assert ids.tolist()==gids.tolist() and np.array_equal(pp,ports)
            assert pred.shape==truth.shape==(100,160,480) and np.isfinite(pred).all()
            en=(abs(truth)**2).sum(axis=(1,2))
            values=dict(rel_l2=np.sqrt((abs(pred-truth)**2).sum(axis=(1,2))/en),
                amplitude_rel_l2=np.sqrt(((abs(pred)-abs(truth))**2).sum(axis=(1,2))/en),
                intensity_ratio=(abs(pred)**2).sum(axis=(1,2))/en,
                intensity_weighted_phase_mae=(abs(np.angle(pred*np.conj(truth)))*abs(truth)**2).sum(axis=(1,2))/en)
            selected={r['geometry_id']:r for r in table if r['dataset']==ds and r['method']==method and r['checkpoint']==ck}
            for i,g in enumerate(gids):
                r=selected[str(g)];assert int(r['input_port'])==int(ports[i])
                for k in metrics:
                    error=abs(float(values[k][i])-float(r[k]));maximum=max(maximum,error);assert error<3e-7;count+=1
                if not new:
                    hr=historical[(method,ck,str(g))]
                    assert all(float(hr[k])==float(r[k]) for k in metrics);inherited+=1
summary=read(P/'test_summary.json');pairs=read(P/'paired_comparisons.json')
for s in summary:
    if s['group']=='port_macro':
        q=[r for r in summary if r['dataset']==s['dataset'] and r['checkpoint']==s['checkpoint'] and r['method']==s['method'] and r['group'].startswith('port') and r['group']!='port_macro']
        assert len(q)==3
        for k in metrics:assert abs(np.mean([r[k] for r in q])-s[k])<1e-12
    else:
        q=[r for r in table if r['dataset']==s['dataset'] and r['checkpoint']==s['checkpoint'] and r['method']==s['method'] and (s['group']=='all' or int(r['input_port'])==int(s['group'][-1]))]
        assert len(q)==s['n']
        for k in metrics:assert abs(np.mean([float(r[k]) for r in q])-s[k])<1e-12
for r in pairs:
    if r['metric']!='rel_l2':continue
    by=[]
    for method in [r['baseline'],r['candidate']]:
        by.append(sorted([q for q in table if q['dataset']==r['dataset'] and q['checkpoint']==r['checkpoint'] and q['method']==method and (r['group']=='all' or int(q['input_port'])==int(r['group'][-1]))],key=lambda q:q['geometry_id']))
    a,b=by;assert [q['geometry_id'] for q in a]==[q['geometry_id'] for q in b]
    d=np.array([float(x['rel_l2'])-float(y['rel_l2']) for x,y in zip(a,b)])
    draw=np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))
    assert abs(d.mean()-r['mean_baseline_minus_candidate'])<1e-12
    assert np.max(abs(np.quantile(d[draw].mean(1),[.025,.975])-r['geometry_bootstrap_ci95']))<1e-12
    assert int((d>0).sum())==r['positive_differences']
root=P/'runs/branch_tangent_steps2400_seed2026';s=read(root/'summary.json');h=read(root/'history.json')
assert [r['step'] for r in h]==[1]+list(range(50,2401,50))
assert s['best_step']==min(h,key=lambda r:r['val_rel_l2'])['step']
assert s['best_val']==min(r['val_rel_l2'] for r in h)
assert read(P/'training_complete.json')['best_validation_replay_max_error']==0
for ck in ['best','last']:assert sha(root/(ck+'.pt'))==s[ck+'_sha256']
f=read(P/'freeze.json')
for n,digest in f['protected_hashes'].items():assert sha(B/n)==digest
for n,digest in f['source_hashes'].items():assert sha(P/n)==digest
done=read(P/'evaluation_complete.json')
for n,digest in done['prediction_hashes'].items():assert sha(P/'predictions'/n)==digest
result=dict(passed=True,unique_rows=len(table),independently_recomputed_field_metrics=count,max_absolute_metric_difference=maximum,
    inherited_rows_exact=inherited,all_primary_pair_bootstraps_verified=True,checkpoint_selection_verified=True,
    protected_files_unchanged=len(f['protected_hashes']),no_new_inference_or_training=True,
    metric_table_sha256=sha(P/'test_per_geometry.csv'),time_unix=time.time())
(P/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
def sr(ds,m,ck='best',g='all'):return next(r for r in summary if r['dataset']==ds and r['method']==m and r['checkpoint']==ck and r['group']==g)
def pr(ds,ck,g):return next(r for r in pairs if r['dataset']==ds and r['checkpoint']==ck and r['group']==g and r['baseline']=='path' and r['candidate']=='branch_tangent' and r['metric']=='rel_l2')
lines=['# Y-branch BranchTangent：seed2026 初步结论','',
'日期：2026-11-13。完成一个新的 BranchTangent 2400 步训练，与同种子的历史 Linear、Branch Path 对照。',
'E2、M4 各有100个历史评估几何，复用同一组训练模型；两组结果不代表两个独立训练种子。','',
'## 主结果','',
'复场 RelL2 越低越好；best 只由原8个验证几何选择，last 为第2400步。','',
'| 数据集 | 方法 | best平均RelL2 | last平均RelL2 |','|---|---|---:|---:|']
for ds in ['E2','M4']:
    for m in ['linear','path','branch_tangent']:
        lines.append(f"| {ds} | {m} | {sr(ds,m)['rel_l2']:.9f} | {sr(ds,m,'last')['rel_l2']:.9f} |")
lines+=['','## Branch Path 与 BranchTangent 配对差异','',
'正差值表示 BranchTangent 误差更低。区间是固定模型下的几何配对自助区间，不能说明跨种子训练稳定性。','',
'| 数据集 | checkpoint | 组 | Path减BT | 相对误差下降 | 几何95%区间 | 改善几何数 |','|---|---|---|---:|---:|---|---:|']
for ds in ['E2','M4']:
    for ck in ['best','last']:
        for g in ['all','port1','port2','port3']:
            r=pr(ds,ck,g);ci=r['geometry_bootstrap_ci95']
            lines.append(f"| {ds} | {ck} | {g} | {r['mean_baseline_minus_candidate']:.9f} | {100*r['relative_reduction']:.2f}% | [{ci[0]:.9f}, {ci[1]:.9f}] | {r['positive_differences']}/{r['n']} |")
lines+=['','## best各端口均值','',
'| 数据集 | 方法 | port1 | port2 | port3 | 三端口等权平均 |','|---|---|---:|---:|---:|---:|']
for ds in ['E2','M4']:
    for m in ['linear','path','branch_tangent']:
        values=[sr(ds,m,g=g)['rel_l2'] for g in ['port1','port2','port3','port_macro']]
        lines.append(f"| {ds} | {m} | "+' | '.join(f'{v:.9f}' for v in values)+' |')
overall=[pr(ds,'best','all') for ds in ['E2','M4']]
if all(r['mean_baseline_minus_candidate']>0 for r in overall):
    conclusion='本种子下，BranchTangent 在 E2/M4 的总体平均误差均低于 Branch Path。是否各端口和 best/last 都支持这一趋势，应分别读取表格；跨种子稳定性尚未检验。'
elif all(r['mean_baseline_minus_candidate']<0 for r in overall):
    conclusion='本种子下，这个局部切线构造没有改善 Y-branch 总体复场预测：E2/M4 的 BranchTangent 平均误差均高于 Branch Path。S-bend 上的切线收益不能直接推广至此分叉构造。'
else:
    conclusion='本种子下，BranchTangent 在 E2/M4 的总体变化方向不一致，暂不支持一致优势。应保留数据集与端口差异。'
lines+=['','## 初步解释与边界','',conclusion,'',
'- 现有 Branch Path 已含中心线累计弧长与源相关分叉路由，相当于这里的分叉版 ArcOnly。',
'- 新方法只在原路径上加入源锚定的一阶横向切线项。原中心线、硬分支分配、10实输入打包为6复通道、2944738实自由度网络均保持一致。',
'- 右侧激励时，入射支路与主干方向为负，另一个出射支路方向为正；合成三端口路由及源锚定检查已通过。',
'- Path 和新载波都可能在硬分支边界产生相位跳变；训练/验证相邻载波差可接近2。本轮没有根据测试结果平滑或调参，因此结果只检验这一冻结构造，不能证明所有切线参考都有效或无效。',
'- 相同64训练/8验证、初始化、batch顺序和2400步训练协议。四个历史best/last验证逐样本回放误差为0，新模型best回放误差为0。',
'- E2 best与M4 best/last历史指标原样复用。E2历史last未保存预测，本轮在独立目录生成其预测以检查checkpoint敏感性。',
'- 未新增FDTD、输入通道、真值相位对齐、测试调参或额外种子。测试集曾被使用，本轮属于回顾性固定基准比较。',
'- intensity_ratio理想值是1，并非越低越好；全场指标不能替代原生端口功率、S参数或器件筛选收益。',
'- 自助区间未进行多重比较校正，不作为训练稳定性或正式等效性结论。',
f"- 训练循环耗时{s['train_wall_seconds']:.2f}秒，best位于第{s['best_step']}步，峰值已分配显存{s['peak_allocated_MiB']:.2f}MiB。",
f"- 独立重算{count}项保存场指标，最大绝对差{maximum:.3g}；{inherited}条历史指标完全一致，{len(f['protected_hashes'])}个受保护文件哈希未变。",
'','完整逐几何结果见 test_per_geometry.csv，配对统计见 paired_comparisons.json，冻结定义见 PROTOCOL.md。']
(P/'初步结论_seed2026.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
readme=['# Y-branch BranchTangent seed2026 pilot','',
'Completed one new 2400-step seed2026 model. Historical endpoints reused after exact validation replay.',
'Read 初步结论_seed2026.md and REPORT.md for results; comparison.png/.pdf for figures.',
'PROTOCOL.md and freeze.json record pretraining definitions and hashes.',
'preflight.json and geometry_checks_trainval.json record routing, source and replay checks.',
'test_per_geometry.csv, test_summary.json and paired_comparisons.json retain all results.',
'independent_verification.json checks saved predictions, metrics, selection and preservation.',
'runs/branch_tangent_steps2400_seed2026 contains best/last restorable checkpoints.',
'predictions contains new BT best/last E2/M4 and E2 Linear/Path last complex fields.',
'The training summary test_evaluated=false records training-stage state;',
'evaluation_complete.json records the subsequent completed evaluation.',
'No additional seeds, FDTD, input changes or test-based tuning.']
(P/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
