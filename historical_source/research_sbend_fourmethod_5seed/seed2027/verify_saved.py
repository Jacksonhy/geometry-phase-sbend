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
    assert s['seed']==2027 and s['steps']==2400 and s['val_replay_error']<1e-7
    assert [r['step'] for r in h]==[1]+list(range(50,2401,50))
    best=min(h,key=lambda r:r['val_rel_l2'])
    assert s['best_step']==best['step'] and s['best_val']==best['val_rel_l2']
    initial.add(s['initial_state_sha256']);batches.add(s['batches_sha256'])
    for ck in ['best','last']:assert sha(root/(ck+'.pt'))==s[ck+'_sha256']
assert len(initial)==len(batches)==1
pre=read(P/'preflight.json')['model_checks'][0]
assert next(iter(initial))==pre['initial_state_sha256'] and next(iter(batches))==pre['batches_sha256']
old=read(PREV/'test_summary.json');aggregate=read(P/'two_seed_statistics.json')
for r in aggregate['methods']:
    v=[next(x['rel_l2'] for x in ss if x['method']==r['method'] and x['checkpoint']==r['checkpoint'] and x['group']=='all') for ss in [old,summ]]
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
result=dict(passed=True,seed=2027,unique_rows=800,independently_recomputed_field_region_metrics=fields,
    independently_recomputed_noninput_proxy_metrics=proxy_count,port_coefficient_rows_verified=len(coeffs),
    max_absolute_metric_difference=maximum,paired_geometry_bootstraps_verified=True,
    four_validation_selections_and_weight_hashes_verified=True,two_seed_statistics_verified=True,
    protected_files_unchanged=len(f['protected_hashes']),new_training_or_inference=False)
(P/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
def sr(m,ck='best',group='all',ss=summ):return next(r for r in ss if r['method']==m and r['checkpoint']==ck and r['group']==group)
def pair(a,b,ck='best'):return next(r for r in paired if r['baseline']==a and r['candidate']==b and r['checkpoint']==ck and r['group']=='all' and r['metric']=='rel_l2')
lines=['# S-bend seed2027 四方法结果','',
'本轮四方法均重新独立训练，使用同一 seed2027、初始化和batch顺序，每方法2400步。原64/16/100几何划分、输入、校准和模型保持不变。',
'best由16个验证几何选择，last为第2400步。100个测试几何是已使用的固定基准；本轮没有测试调参或真值相位对齐。','',
'## 主要结果','',
'| 方法 | seed2026 best RelL2 | seed2027 best RelL2 | seed2027 last RelL2 |','|---|---:|---:|---:|']
for m in METHODS:
    lines.append(f"| {m} | {sr(m,ss=old)['rel_l2']:.9f} | {sr(m)['rel_l2']:.9f} | {sr(m,'last')['rel_l2']:.9f} |")
lines+=['','## seed2027 分层收益','',
'| 对照 | 平均误差降低量 | 相对误差降低 | 几何95%配对区间 | 改善几何数 |','|---|---:|---:|---|---:|']
for a,b in [('linear','arc_only'),('arc_only','local_tangent'),('linear','local_tangent'),('local_tangent','path_v2')]:
    r=pair(a,b);ci=r['geometry_bootstrap_ci95']
    lines.append(f"| {a} → {b} | {r['mean_baseline_minus_candidate']:.9f} | {100*r['relative_error_reduction']:.3f}% | [{ci[0]:.9f}, {ci[1]:.9f}] | {r['improved']}/100 |")
v={m:sr(m)['rel_l2'] for m in METHODS}
lines+=['',f"ArcOnly覆盖本种子Linear到v2平均误差改善的{100*(v['linear']-v['arc_only'])/(v['linear']-v['path_v2']):.3f}%。这一描述性比例不等于机制贡献率或方差解释率。",
'','## 两种子统计','',
'以下只描述训练随机性；没有把2×100个观测当作200个独立实验。两种子尚不足以完成原五种子稳健性验证。','',
'| 方法 | 两种子均值 | 种子间样本标准差 |','|---|---:|---:|']
for r in aggregate['methods']:
    if r['checkpoint']=='best':lines.append(f"| {r['method']} | {r['mean']:.9f} | {r['sample_sd']:.9f} |")
lines+=['','## 激励方向：seed2027 best','',
'| 方法 | input1（50几何） | input2（50几何） |','|---|---:|---:|']
for m in METHODS:lines.append(f"| {m} | {sr(m,group='input1')['rel_l2']:.9f} | {sr(m,group='input2')['rel_l2']:.9f} |")
lines+=['','## 边界及核验','',
'- 所有六组配对、best/last、区域误差及端口代理指标均完整保存，未排除不利结果。',
'- 配对几何区间以训练好的模型为条件，没有进行多重比较校正，也不代表训练种子置信区间或正式等效检验。',
'- 端口代理的复数、幅度、平方幅度误差未必与全场误差同方向；不能替代原生S参数或器件设计收益。',
'- 训练日志loss为当前batch相对MSE，不是全训练集RelL2。训练完成后另保存best模型的全训练集RelL2。',
f"- 独立重算{fields}项场/区域指标及{proxy_count}项非输入端口代理指标，最大绝对差{maximum:.3g}，核验{len(coeffs)}条端口系数记录。",
f"- 四方法验证选择、权重哈希、主要配对自助区间及两种子统计均通过；{len(f['protected_hashes'])}个受保护历史文件哈希未变。",
'- 本轮仅执行seed2027；seed2028—2030未启动。没有新增FDTD或改变输入。',
'','完整数值见REPORT.md、test_per_geometry.csv和two_seed_statistics.json，训练前定义见PROTOCOL.md。']
(P/'结论_seed2027.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'README.md').write_text('# S-bend seed2027 completed\n\nRead 结论_seed2027.md and REPORT.md.\n'
    'PROTOCOL.md + freeze.json record the pretraining definition.\n'
    'All four runs, best/last predictions and 800 per-geometry outcomes saved.\n'
    'two_seed_statistics.json reports individual seed means and paired differences.\n'
    'independent_verification.json audits saved results without new inference.\n',encoding='utf-8')
