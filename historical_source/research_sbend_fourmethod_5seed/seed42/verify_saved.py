"""Independent saved-array arithmetic audit; does not train or infer."""
from pathlib import Path
import json,csv,hashlib,math,itertools
import numpy as np
P=Path(__file__).resolve().parent;PREV=P.parent;O=PREV.parent/'research_sbend_formal'
METHODS=['linear','arc_only','local_tangent','path_v2']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024**2),b''):h.update(c)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
table=rows(P/'test_per_geometry.csv');assert len(table)==800
assert len({(r['method'],r['checkpoint'],r['geometry_id']) for r in table})==800
assert all(math.isfinite(float(r[k])) for r in table for k in r if k not in ['method','checkpoint','geometry_id'])
maximum=0.;fields=0;proxy_count=0;coeffs=rows(P/'port_coefficients.csv')
assert len(coeffs)==3200
# Independent port-extraction function, read exactly the historical frozen function.
import ast
from scipy.linalg import eigh_tridiagonal
src=O.parent/'research_official_audit/validate_sparams.py'
tree=ast.parse(src.read_text(encoding='utf-8-sig'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fd_directional_proxy')
env={'np':np,'eigh_tridiagonal':eigh_tridiagonal}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),env)
proxy=env['fd_directional_proxy']
predhash=read(P/'evaluation_complete.json')['prediction_hashes']
# Verify field metrics from saved arrays and secondary complex proxy arithmetic.
for m in METHODS:
    for ck in ['best','last']:
        path=P/'predictions'/f'{m}_{ck}.npz';assert sha(path)==predhash[path.name]
        selected={r['geometry_id']:r for r in table if r['method']==m and r['checkpoint']==ck}
        co={(r['geometry_id'],r['reference'],int(r['output_port'])):r for r in coeffs if r['method']==m and r['checkpoint']==ck}
        with np.load(path,allow_pickle=False) as z:
            assert len(set(z['geometry_id']))==100
            predictions=z['prediction'];ports=z['input_port']
            for i,g in enumerate(z['geometry_id']):
                g=str(g);r=selected[g];a=predictions[i].astype(np.complex128);assert np.isfinite(a).all()
                with np.load(O/'data'/g/'sample.npz',allow_pickle=False) as tr,np.load(O/'cache'/(g+'.npz'),allow_pickle=False) as cache:
                    b=tr['Ez'].astype(np.complex128);assert a.shape==b.shape==(160,480)
                    port=int(tr['input_port']);assert port==int(ports[i])==int(r['input_port'])
                    error=abs(a-b)**2;power=abs(b)**2
                    masks={'rel_l2':np.ones(b.shape,dtype=bool)}
                    masks.update({k[7:]+'_rel_l2':cache[k] for k in cache.files if k.startswith('region_')})
                    for k,mask in masks.items():
                        v=float(np.sqrt(error[mask].sum()/power[mask].sum()))
                        d=abs(v-float(r[k]));maximum=max(maximum,d);assert d<1e-12;fields+=1
                    args=(tr['eps'],tr['port_masks'],tr['port_ids'],port,1.55,.05)
                    estimate=proxy(a,*args);target=proxy(b,*args)
                    o=1 if port==1 else 0
                    vals=[abs(estimate[o]-target[o]),abs(abs(estimate[o])-abs(target[o])),abs(abs(estimate[o])**2-abs(target[o])**2)]
                    for k,v in zip(['complex_error','amplitude_error','squared_amplitude_error'],vals):
                        d=abs(float(v)-float(r['noninput_'+k]));maximum=max(maximum,d);assert d<1e-12;proxy_count+=1
                    for ref,t in [('truth_proxy',target),('Meep_native',tr['S_native'])]:
                        for j in range(2):
                            q=co[(g,ref,j+1)]
                            es=complex(float(q['estimate_real']),float(q['estimate_imag']))
                            ts=complex(float(q['target_real']),float(q['target_imag']))
                            assert abs(es-estimate[j])<1e-12 and abs(ts-t[j])<1e-12
summ=read(P/'test_summary.json');paired=read(P/'paired_comparisons.json')
for r in summ:
    q=[q for q in table if q['method']==r['method'] and q['checkpoint']==r['checkpoint'] and (r['group']=='all' or int(q['input_port'])==int(r['group'][-1]))]
    assert len(q)==r['n']
    for k in ['rel_l2','core_rel_l2','bend_core_rel_l2','halo_rel_l2','far_rel_l2','noninput_complex_error','noninput_amplitude_error','noninput_squared_amplitude_error']:
        assert abs(np.mean([float(a[k]) for a in q])-r[k])<1e-12
    assert abs(np.median([float(a['rel_l2']) for a in q])-r['rel_l2_median'])<1e-12
    assert abs(np.quantile([float(a['rel_l2']) for a in q],.9)-r['rel_l2_p90'])<1e-12
for r in paired:
    if r['metric']!='rel_l2':continue
    q=[]
    for m in [r['baseline'],r['candidate']]:
        q.append(sorted([a for a in table if a['method']==m and a['checkpoint']==r['checkpoint'] and (r['group']=='all' or int(a['input_port'])==int(r['group'][-1]))],key=lambda a:a['geometry_id']))
    a,b=q;assert [x['geometry_id'] for x in a]==[x['geometry_id'] for x in b]
    d=np.array([float(x['rel_l2'])-float(y['rel_l2']) for x,y in zip(a,b)])
    draws=np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))
    assert abs(d.mean()-r['mean_baseline_minus_candidate'])<1e-12
    assert np.max(abs(np.quantile(d[draws].mean(1),[.025,.975])-r['geometry_bootstrap_ci95']))<1e-12
    assert (d>0).sum()==r['improved']
initial=set();batches=set()
for m in METHODS:
    root=P/'runs'/m;s=read(root/'summary.json');h=read(root/'history.json')
    assert s['seed']==42 and s['steps']==2400 and s['val_replay_error']<1e-7
    assert [r['step'] for r in h]==[1]+list(range(50,2401,50))
    best=min(h,key=lambda r:r['val_rel_l2'])
    assert s['best_step']==best['step'] and s['best_val']==best['val_rel_l2']
    initial.add(s['initial_state_sha256']);batches.add(s['batches_sha256'])
    for ck in ['best','last']:assert sha(root/(ck+'.pt'))==s[ck+'_sha256']
assert len(initial)==len(batches)==1
pre=read(P/'preflight.json')['model_checks'][0]
assert next(iter(initial))==pre['initial_state_sha256'] and next(iter(batches))==pre['batches_sha256']
old=read(PREV/'test_summary.json');seed27=read(PREV/'seed2027/test_summary.json');aggregate=read(P/'three_seed_statistics.json')
assert aggregate['active_seeds']==[42,2026,2027] and aggregate['withdrawn_seed']==2025
for r in aggregate['methods']:
    v=[next(x['rel_l2'] for x in ss if x['method']==r['method'] and x['checkpoint']==r['checkpoint'] and x['group']=='all') for ss in [summ,old,seed27]]
    assert v==r['seed_mean_rel_l2'] and abs(np.mean(v)-r['mean'])<1e-12
    assert abs(np.std(v,ddof=1)-r['sample_sd'])<1e-12
for r in aggregate['pairs']:
    a=next(x for x in aggregate['methods'] if x['method']==r['baseline'] and x['checkpoint']==r['checkpoint'])
    b=next(x for x in aggregate['methods'] if x['method']==r['candidate'] and x['checkpoint']==r['checkpoint'])
    delta=np.array(a['seed_mean_rel_l2'])-np.array(b['seed_mean_rel_l2'])
    assert np.array_equal(delta,r['paired_seed_differences']) and abs(delta.mean()-r['mean'])<1e-12
    assert abs(delta.std(ddof=1)-r['sample_sd'])<1e-12
f=read(P/'freeze.json')
for n,h in f['protected_hashes'].items():assert sha(PREV.parent/n)==h
for n,h in f['source_hashes'].items():assert sha(P/n)==h
result=dict(passed=True,seed=42,unique_rows=800,independently_recomputed_field_region_metrics=fields,
    independently_recomputed_noninput_proxy_metrics=proxy_count,port_coefficient_rows_verified=len(coeffs),
    max_absolute_metric_difference=maximum,paired_geometry_bootstraps_verified=True,
    four_validation_selections_and_weight_hashes_verified=True,three_seed_statistics_verified=True,
    protected_files_unchanged=len(f['protected_hashes']),new_training_or_inference=False)
(P/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
def sr(m,ck='best',group='all',ss=summ):return next(r for r in ss if r['method']==m and r['checkpoint']==ck and r['group']==group)
def pair(a,b,ck='best'):return next(r for r in paired if r['baseline']==a and r['candidate']==b and r['checkpoint']==ck and r['group']=='all' and r['metric']=='rel_l2')
lines=['# S-bend 三种子汇总：42、2026、2027','',
'本轮完成seed42四方法全新训练。当前汇总仅含42、2026、2027，seed2025按用户要求取消，不纳入任何均值、配对或图表。',
'种子变更发生在此前2026/2027结果和2025部分验证日志已被观察之后；seed42协议在本轮优化前冻结。这不是最初预注册的三种子选择。',
'seed2025未进入测试评估。其残留文件只作取消记录，seed42没有复用其权重或优化器状态。',
'全部方法保持原64训练/16验证/100测试几何、训练专用校准、输入、架构和2400步预算。best由验证集选择，last为第2400步。',
'100个测试几何是已使用固定基准，没有测试真值相位对齐或测试调参。','',
'## 三种子主要结果','',
'复场RelL2越低越好；均值及样本标准差在三个训练种子的100几何平均误差上计算，标准差使用ddof=1。','',
'| 方法 | seed42 | seed2026 | seed2027 | 三种子均值 | 种子间样本标准差 |','|---|---:|---:|---:|---:|---:|']
for r in aggregate['methods']:
    if r['checkpoint']=='best':
        v=r['seed_mean_rel_l2']
        lines.append(f"| {r['method']} | {v[0]:.9f} | {v[1]:.9f} | {v[2]:.9f} | {r['mean']:.9f} | {r['sample_sd']:.9f} |")
contrasts=[('linear','arc_only'),('arc_only','local_tangent'),('linear','local_tangent'),('local_tangent','path_v2')]
lines+=['','## 相对误差下降','',
'逐种子下降以对应baseline为分母；总体均值下降以baseline三种子均值为分母，不混用。','',
'| 对照 | seed42 | seed2026 | seed2027 | 总体均值下降 |','|---|---:|---:|---:|---:|']
for a,b in contrasts:
    r=next(r for r in aggregate['pairs'] if r['checkpoint']=='best' and r['baseline']==a and r['candidate']==b)
    v=r['relative_reductions_by_seed']
    lines.append(f"| {a} → {b} | {100*v[0]:.3f}% | {100*v[1]:.3f}% | {100*v[2]:.3f}% | {100*r['relative_reduction_of_grand_means']:.3f}% |")
lines+=['','## 训练种子内配对差值','',
'正差值表示candidate平均误差更低；样本标准差反映三个训练种子之间的变化。','',
'| 对照 | Δ42 | Δ2026 | Δ2027 | 差值均值 | 差值样本标准差 |','|---|---:|---:|---:|---:|---:|']
for a,b in contrasts:
    r=next(r for r in aggregate['pairs'] if r['checkpoint']=='best' and r['baseline']==a and r['candidate']==b)
    v=r['paired_seed_differences']
    lines.append(f"| {a} → {b} | {v[0]:.9f} | {v[1]:.9f} | {v[2]:.9f} | {r['mean']:.9f} | {r['sample_sd']:.9f} |")
lines+=['','## seed42的checkpoint和方向','',
'| 方法 | best测试RelL2 | last测试RelL2 | best input1 | best input2 | best验证RelL2 | best步数 |','|---|---:|---:|---:|---:|---:|---:|']
for m in METHODS:
    s=read(P/'runs'/m/'summary.json')
    lines.append(f"| {m} | {sr(m)['rel_l2']:.9f} | {sr(m,'last')['rel_l2']:.9f} | {sr(m,group='input1')['rel_l2']:.9f} | {sr(m,group='input2')['rel_l2']:.9f} | {s['best_val']:.9f} | {s['best_step']} |")
lines+=['','## seed42的几何配对结果','',
'以下探索性95%几何自助区间以固定训练模型为条件；不是训练种子置信区间，未进行多重比较校正。','',
'| 对照 | 平均差值 | 几何95%区间 | 改善几何数 |','|---|---:|---|---:|']
for a,b in contrasts:
    r=pair(a,b);ci=r['geometry_bootstrap_ci95']
    lines.append(f"| {a} → {b} | {r['mean_baseline_minus_candidate']:.9f} | [{ci[0]:.9f}, {ci[1]:.9f}] | {r['improved']}/100 |")
lines+=['','## seed42区域及端口代理指标','',
'| 方法 | bend_core RelL2 | 非输入端口复系数误差 | 幅度误差 | 平方幅度误差 |','|---|---:|---:|---:|---:|']
for m in METHODS:
    r=sr(m);lines.append(f"| {m} | {r['bend_core_rel_l2']:.9f} | {r['noninput_complex_error']:.9f} | {r['noninput_amplitude_error']:.9f} | {r['noninput_squared_amplitude_error']:.9f} |")
lines+=['','## 边界和核验','',
'- 六组方法配对、最佳和最终checkpoint、两个激励方向及不利结果均保留。',
'- 不把3×100个观测当作300个独立实验，也不声称原五种子计划完成、正式等效或普遍稳健性。',
'- 三个种子共享几何划分，当前证据只检验固定数据与训练协议下的训练随机性，不能推出数据效率、新划分或跨拓扑迁移结论。',
'- 端口幅度或平方幅度代理误差不一定与复场误差同向；代理不是认证原生S参数、端口功率或设计筛选收益。',
'- seed2026三个方法经检查复用，非新增训练重复；seed2027与42四方法均全新训练。训练日志loss为当前batch相对MSE。',
'- 原有2025取消记录保留，不参与active统计；seed2028—2030未启动。没有新增FDTD、输入、测试相位对齐或超参数搜索。',
f"- 独立重算{fields}项场/区域指标、{proxy_count}项非输入端口代理指标，最大绝对差{maximum:.3g}；{len(coeffs)}条端口系数记录核验通过。",
f"- 四模型验证选择、权重哈希、配对自助区间及三种子汇总通过，{len(f['protected_hashes'])}个历史受保护文件哈希未变。",
'','完整机器数据：three_seed_statistics.json、test_per_geometry.csv、paired_comparisons.json；冻结定义：PROTOCOL.md及THREE_SEED_AMENDMENT.md。']
(P/'三种子结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'README.md').write_text('# S-bend seed42 and active three-seed study\n\nRead 三种子结果汇总.md and REPORT.md.\n'
    'Active seeds42,2026,2027; cancelled seed2025 excluded completely.\n'
    'All four seed42 models trained fresh; three_seed_statistics.json contains seed-level means, SD and pairs.\n'
    'PROTOCOL.md,THREE_SEED_AMENDMENT.md and freeze.json document sequential seed selection.\n'
    'independent_verification.json audits saved results and protected historical files.\n',encoding='utf-8')
