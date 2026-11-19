"""Reuse frozen metric functions with an explicit new geometry adapter."""
import sys,json,time,csv
import numpy as np
from experiment import *
sys.modules['experiment']=sys.modules.get('experiment')
E=module('frozen_blind_evaluator',BASE/'research_sbend_blind_local_phase_2026-11-15/evaluate_blind.py')
# The evaluator imports this experiment module, so R, metrics, config and IDs are new.
def report():
 E.verify();summary=read(R/'blind_summary.json');paired=read(R/'blind_paired.json');grid=read(R/'grid_paired.json');truth=read(R/'grid_truth_changes.json')
 primary=[a for a in paired if a['baseline']=='arc_only' and a['group']=='all' and a['gate']==.05 and ((a['region']=='bend_core' and a['metric']=='phase_wmae_deg') or (a['region']=='normal_pair_all' and a['eta']==.5 and a['metric']=='normal_relative_phase_wmae_deg'))]
 save(R/'transfer_primary.json',primary)
 report_rows=[]
 for m in METHODS:
  row={'method':m}
  for reg,metric,key in [('whole','complex_rel_l2','whole_RelL2'),('bend_core','complex_rel_l2','bend_RelL2'),('bend_core','phase_wmae_deg','bend_phase_deg'),('normal_pair_all','normal_relative_phase_wmae_deg','normal_phase_deg'),('normal_pair_all','coherence_complex_rel_l2','cross_RelL2')]:
   a=[v for v in summary if v['method']==m and v['group']=='all' and v['region']==reg and v['metric']==metric and v['gate']==.05 and (reg!='normal_pair_all' or v['eta']==.5)];assert len(a)==3;row[key]=float(np.mean([v['mean'] for v in a]))
  report_rows.append(row)
 csvsave(R/'main_table.csv',report_rows)
 decision=dict(experiment='frozen Euler-model transfer, no new training',primary=primary,all_phase_means_negative=all(v['delta']<0 for v in primary),all_phase_CIs_below_zero=all(v['ci95_high']<0 for v in primary),mesh_absolute_phase_certified=False,automatic_training=False,no_continuum_claim=True)
 save(R/'decision.json',decision)
 lines=['# 正弦中心线冻结模型迁移实验','', '36个新正弦几何，三个已训练种子，四种表示；未重新训练、未拟合测试相位。','', '|方法|全场RelL2|弯曲RelL2|弯曲相位°|法向相对相位°|交叉项RelL2|','|---|---:|---:|---:|---:|---:|']
 for v in report_rows:lines.append('|'+v['method']+'|'+ '|'.join(f'{v[k]:.6f}' for k in list(v)[1:])+'|')
 lines+=['','## LocalTangent − ArcOnly，逐种子配对','', '|种子|指标|均值差|几何配对95%区间|改善几何数|','|---|---|---:|---|---:|']
 for v in primary:lines.append(f"|{v['seed']}|{v['metric']}|{v['delta']:.6f}|[{v['ci95_low']:.6f}, {v['ci95_high']:.6f}]|{v['improved']}/{v['n']}|")
 lines+=['','## 网格参考变化','', '|网格|指标|约定|均值|','|---|---|---|---:|']
 for lo in [20,30]:
  for reg,key,conv in [('bend_core','phase_wmae_deg','raw'),('bend_core','phase_wmae_deg','incident_normalized'),('normal_pair_all','normal_relative_phase_wmae_deg','raw')]:
   a=[v[key] for v in truth if v['low_resolution']==lo and v['region']==reg and v['convention']==conv];lines.append(f'|{lo}→{30 if lo==20 else 40}|{reg}|{conv}|{np.mean(a):.6f}°|')
 lines+=['','这是中心线家族迁移，不是正弦家族重新训练的表示复现实验。结果不能证明跨拓扑迁移；网格细化比较不是连续解认证。','输入端口、实际端点偏移、弯曲区轴向跨度与选定Euler几何匹配；弧长、最大倾角及曲率并未强制相同。','选择基于几何，不基于旧场或旧误差；旧72数据为已分析过的对照，不能作为新的盲测。','独立配对单位是几何，三个训练种子不合并成108次独立训练。','全部真场、预测、逐几何指标和冻结记录保存在本目录；旧模型与论文不修改。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(report_rows));print(json.dumps(primary))
if __name__=='__main__':
 if sys.argv[1]=='blind':E.blind()
 elif sys.argv[1]=='grid':E.grid()
 elif sys.argv[1]=='report':report()
