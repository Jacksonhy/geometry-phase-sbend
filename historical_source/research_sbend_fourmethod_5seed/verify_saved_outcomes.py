"""Post-evaluation arithmetic QA only: no optimization, inference or model selection."""
from pathlib import Path
import json,csv,hashlib,math
import numpy as np
P=Path(__file__).resolve().parent
O=P.parent/'research_sbend_formal'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
table=rows(P/'test_per_geometry.csv')
assert len(table)==800 and len({(r['method'],r['checkpoint'],r['geometry_id']) for r in table})==800
assert all(math.isfinite(float(r[k])) for r in table for k in r if k not in ['method','checkpoint','geometry_id'])
old={(r['method'],r['checkpoint'],r['geometry_id']):r for r in rows(O/'test_per_geometry.csv')}
for r in table:
    if r['method']!='arc_only':
        ref=old[(r['method'],r['checkpoint'],r['geometry_id'])]
        for k in r:
            if k in ['method','checkpoint','geometry_id']:assert r[k]==ref[k]
            else:assert float(r[k])==float(ref[k])
mx=0.;count=0
for ck in ['best','last']:
    with np.load(P/'predictions'/f'arc_only_{ck}.npz',allow_pickle=False) as pred:
        assert len(set(pred['geometry_id']))==100
        byid={r['geometry_id']:r for r in table if r['method']=='arc_only' and r['checkpoint']==ck}
        for i,gid in enumerate(pred['geometry_id']):
            r=byid[str(gid)]
            with np.load(O/'data'/str(gid)/'sample.npz',allow_pickle=False) as true, np.load(O/'cache'/(str(gid)+'.npz'),allow_pickle=False) as cache:
                a=pred['prediction'][i].astype(np.complex128);b=true['Ez'].astype(np.complex128)
                assert int(pred['input_port'][i])==int(true['input_port'])==int(r['input_port'])
                q=np.square(np.abs(a-b));den=np.square(np.abs(b))
                masks={'rel_l2':np.ones(b.shape,dtype=bool)}
                masks.update({k[7:]+'_rel_l2':cache[k] for k in cache.files if k.startswith('region_')})
                for key,mask in masks.items():
                    v=float(np.sqrt(q[mask].sum()/den[mask].sum()))
                    error=abs(v-float(r[key]));mx=max(mx,error);assert error<1e-12;count+=1
summ=read(P/'test_summary.json');paired=read(P/'paired_comparisons.json')
for r in summ:
    a=[q for q in table if q['method']==r['method'] and q['checkpoint']==r['checkpoint'] and
       (r['group']=='all' or int(q['input_port'])==int(r['group'][-1]))]
    assert len(a)==r['n']
    for k in ['rel_l2','core_rel_l2','bend_core_rel_l2','halo_rel_l2','far_rel_l2',
              'noninput_complex_error','noninput_amplitude_error','noninput_squared_amplitude_error']:
        assert abs(np.mean([float(q[k]) for q in a])-r[k])<1e-12
# Independently recompute the scientific primary pairs and geometry bootstrap.
for r in paired:
    if r['metric']!='rel_l2':continue
    a=sorted([q for q in table if q['method']==r['baseline'] and q['checkpoint']==r['checkpoint'] and (r['group']=='all' or int(q['input_port'])==int(r['group'][-1]))],key=lambda q:q['geometry_id'])
    b=sorted([q for q in table if q['method']==r['candidate'] and q['checkpoint']==r['checkpoint'] and (r['group']=='all' or int(q['input_port'])==int(r['group'][-1]))],key=lambda q:q['geometry_id'])
    assert [q['geometry_id'] for q in a]==[q['geometry_id'] for q in b]
    d=np.array([float(x['rel_l2'])-float(y['rel_l2']) for x,y in zip(a,b)])
    draw=np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))
    assert np.max(abs(np.quantile(d[draw].mean(1),[.025,.975])-r['geometry_bootstrap_ci95']))<1e-12
    assert abs(d.mean()-r['mean_baseline_minus_candidate'])<1e-12 and int((d>0).sum())==r['improved']
hist=read(P/'runs/arc_only/history.json')
assert [r['step'] for r in hist]==[1]+list(range(50,2401,50))
s=read(P/'runs/arc_only/summary.json')
assert s['best_step']==min(hist,key=lambda r:r['val_rel_l2'])['step']
assert s['best_val']==min(r['val_rel_l2'] for r in hist)
assert s['val_replay_error']==0
for ck in ['best','last']:assert sha(P/'runs/arc_only'/(ck+'.pt'))==s[ck+'_sha256']
add=read(P/'statistics_addendum_freeze.json')
assert sha(P/add['file'])==add['sha256'] and not add['pilot_test_evaluation_started']
freeze=read(P/'freeze.json')
for n,h in freeze['protected_hashes'].items():
    assert sha(P.parent/n)==h
for n,h in freeze['new_source_hashes'].items():assert sha(P/n)==h
out=dict(passed=True,unique_method_checkpoint_geometry_rows=800,inherited_metric_rows_verified=600,
         independently_recomputed_arc_field_region_metrics=count,max_absolute_metric_difference=mx,
         all_primary_pair_bootstraps_recomputed=True,validation_selection_verified=True,
         historical_protected_files_unchanged=len(freeze['protected_hashes']),
         statistics_addendum_unchanged=True,no_new_inference_or_training=True,
         reported_summary_sha256=sha(P/'test_summary.json'),reported_table_sha256=sha(P/'test_per_geometry.csv'))
(P/'independent_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out),flush=True)
# Plain Chinese report, including adverse and metric-dependent effects.
def sr(m,ck='best',g='all'):return next(r for r in summ if r['method']==m and r['checkpoint']==ck and r['group']==g)
def pr(a,b,metric='rel_l2',ck='best',g='all'):
    return next(r for r in paired if r['baseline']==a and r['candidate']==b and r['metric']==metric and r['checkpoint']==ck and r['group']==g)
lines=['# S-bend 四方法 seed2026：初步结论','',
'日期：2026-11-13（Asia/Shanghai）。五种子协议已冻结，本轮只完成 seed2026。',
'本轮新增 ArcOnly 训练；Linear、LocalTangent、Path v2 复用经哈希和六次验证回放核验的历史 seed2026 结果，不计为新的独立训练重复。','',
'## 核心判断','',
'在当前固定数据划分、网络和2400步预算下，中心线累计弧长贡献了主要预测收益；局部横向切线修正进一步改善复场预测；v2相对LocalTangent的增益仍很小。',
'ArcOnly 相对 Linear 平均复场误差下降51.99%，覆盖 Linear→v2 平均误差改善量的95.35%。LocalTangent 相对 ArcOnly 再降低5.10%。',
'95.35%是本轮误差差值的描述性比例，不是因果贡献率、普遍规律或跨种子结论。','',
'## 主结果','',
'| 方法 | best测试RelL2 | last测试RelL2 | best验证RelL2 | best步数 |','|---|---:|---:|---:|---:|']
for m in ['linear','arc_only','local_tangent','path_v2']:
    root=P if m=='arc_only' else O;sx=read(root/'runs'/m/'summary.json')
    lines.append(f"| {m} | {sr(m)['rel_l2']:.9f} | {sr(m,'last')['rel_l2']:.9f} | {sx['best_val']:.9f} | {sx['best_step']} |")
lines+=['','## 配对比较与方向','',
'所有区间是给定模型条件下的几何bootstrap区间；不是训练随机性的区间。未做多重比较校正，不据此宣布等效。',
'| best对照 | 平均误差降低量 | 几何95%区间 | 改善几何 |','|---|---:|---|---:|']
for a,b in [('linear','arc_only'),('arc_only','local_tangent'),('local_tangent','path_v2')]:
    r=pr(a,b);ci=r['geometry_bootstrap_ci95']
    lines.append(f"| {a}→{b} | {r['mean_baseline_minus_candidate']:.9f} | [{ci[0]:.9f}, {ci[1]:.9f}] | {r['improved']}/100 |")
lines+=['','| 方法 | 左侧激励 mean RelL2（50例） | 右侧激励 mean RelL2（50例） |','|---|---:|---:|']
for m in ['linear','arc_only','local_tangent','path_v2']:
    lines.append(f"| {m} | {sr(m,g='input1')['rel_l2']:.9f} | {sr(m,g='input2')['rel_l2']:.9f} |")
lines+=['',
'LocalTangent相对ArcOnly的全场均值改善在两激励方向均存在；best改善75/100，last改善79/100。并非所有几何都改善。',
'在bend_core区域，ArcOnly均值为0.122438，LocalTangent为0.107852；切线修正的额外效果在弯曲芯区更明显。',
'v2相对LocalTangent的best总平均差仅0.000295；左激励组v2均值还略差，因此不能强调v2一致占优。','',
'## 端口代理：排序并不与复场误差完全一致','',
'以下指标比较预测代理与真场代理，不是已认证的原生S参数误差或工程功率精度。','',
'| 方法 | 非输入端复系数代理MAE | 幅值代理MAE | 平方幅值代理MAE |','|---|---:|---:|---:|']
for m in ['linear','arc_only','local_tangent','path_v2']:
    q=sr(m);lines.append(f"| {m} | {q['noninput_complex_error']:.9f} | {q['noninput_amplitude_error']:.9f} | {q['noninput_squared_amplitude_error']:.9f} |")
lines+=['',
'ArcOnly的总体幅值、平方幅值代理误差反而低于LocalTangent/v2；全场复误差的额外改善不能自动转述为更好的所有端口功能指标。',
'各指标配对区间及方向详见paired_comparisons.json。','',
'## 执行与核验','',
'- 训练前冻结协议、实验源、历史保护清单；五种子统计补充协议在pilot测试前冻结。',
'- 原64/16/100划分、共同输入、训练专用beta与幅值尺度不变；ArcOnly只改变输出载波。',
'- ArcOnly使用历史训练循环；初始参数、batch序列和原模型一致；2400步训练约666.23秒，峰值已分配显存6489.95MiB。',
'- best由验证集选在2300步；best验证回放误差为0；保留best和last。',
'- 不使用测试真值相位对齐，不用测试S校准，不新增FDTD，不调整超参数。',
'- 历史629个保护文件哈希保持不变；独立复算200个预测场的全场及四区域指标，共1000项，误差小于1e-12。',
'- 三方法600行历史指标逐项一致；四方法800行完整且唯一；主要配对bootstrap、验证选择和权重哈希独立核验通过。',
'- 同一已使用测试基准的新增消融，不能称为全新盲测；仅一个种子，未证明跨种子稳定性、数据效率或最终收敛精度。','',
'## 后续建议','',
'保持协议，下一阶段运行2027–2030以确认ArcOnly主收益与切线增量是否稳定。暂不调v2或扩增新载波。',
'当前实现入口仅支持2026 pilot；后续种子应在独立适配器中参数化种子，并于训练前另行冻结执行源，不能修改历史源或将本轮当成五种子完成。',
'如果重复后成立，论文主线可写为：累计传播相位承担主要收益，局部横向修正提供额外收益，复杂全局延拓的精度收益有限。',
'暂不把上述结果扩展为数据效率、跨拓扑迁移或器件筛选收益。']
(P/'初步结论_seed2026.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Reused source summaries retain their historical false test_evaluated flag.
# Stage completion is authoritative in this directory's evaluation_complete.json.
readme=['# Four-reference five-seed study — seed2026 pilot complete','',
'Read 初步结论_seed2026.md for interpretation; REPORT.md for machine-generated numerical detail.',
'PROTOCOL.md + MULTISEED_STATISTICS_ADDENDUM.md define the frozen plan.',
'preflight.json, freeze.json, statistics_addendum_freeze.json record checks/freeze.',
'independent_verification.json verifies the saved outcomes; preservation_check.json protects history.',
'test_summary.json, test_per_geometry.csv, paired_comparisons.json contain all endpoints.',
'seed2026_comparison.png/.pdf and training_curves.csv contain convergence and results.',
'runs/arc_only/ contains best.pt, last.pt, history.json and summary.json.',
'predictions/ contains unaligned physical complex ArcOnly predictions best/last.',
'The historical-compatible run summary has test_evaluated=false from the training stage;',
'evaluation_complete.json records that evaluation subsequently finished.',
'Only seed2026 executed. Seed2027–2030 are not running.']
(P/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
