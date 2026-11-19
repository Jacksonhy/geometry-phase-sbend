"""Descriptive disclosure from saved frozen test metrics, no new inference."""
import csv
from common import *
def main():
    with (OUT/'test_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    records={r['geometry_id']:r for r in read(OUT/'geometry_manifest.json')['records']}
    by={(r['method'],r['checkpoint'],r['geometry_id']):r for r in rows};out=[];tables=[]
    for checkpoint in ['best','last']:
        for baseline in ['linear','local_tangent','path_v1']:
            rr=[]
            for gid in sorted(g for m,c,g in by if m=='path_v2' and c==checkpoint):
                a=float(by[baseline,checkpoint,gid]['rel_l2']);b=float(by['path_v2',checkpoint,gid]['rel_l2']);r=records[gid]
                q={'geometry_id':gid,'checkpoint':checkpoint,'baseline':baseline,'input_port':r['input_port'],**r['params'],
                    'baseline_rel_l2':a,'v2_rel_l2':b,'baseline_minus_v2':a-b,'v2_relative_increase':b/a-1}
                rr.append(q);out.append(q)
            if checkpoint=='best':
                tables.append('### best：相对'+baseline+'的最差5例（全量已保存）')
                tables.extend(['','| 几何ID | 输入 | w/offset/R μm | 对照RelL2 | v2 RelL2 | v2相对恶化 |',
                    '|---|---:|---|---:|---:|---:|'])
                for r in sorted(rr,key=lambda a:a['baseline_minus_v2'])[:5]:
                    tables.append(f"| {r['geometry_id']} | {r['input_port']} | {r['wg_width_um']:.3f}/{r['lateral_offset_um']:.3f}/{r['R_min_um']:.3f} | {r['baseline_rel_l2']:.6f} | {r['v2_rel_l2']:.6f} | {100*r['v2_relative_increase']:.2f}% |")
                tables.append('')
    with (OUT/'paired_per_geometry_details.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    pairs=read(OUT/'paired_comparisons.json');selected=[]
    for p in pairs:
        if p['baseline'] in ['local_tangent','path_v1'] and (
            (p['metric']=='rel_l2' and p['group'] in ['all','input1','input2']) or
            (p['checkpoint']=='best' and p['group']=='all' and p['metric'] in ['far_rel_l2','noninput_complex_error','noninput_amplitude_error','noninput_squared_amplitude_error'])):
            selected.append(p)
    write('adverse_and_secondary_details.json',{'time_utc':now(),'posthoc_descriptive_only':True,'changed_methods_or_metrics':False,
         'selected_precomputed_comparisons':selected,'all_per_geometry_differences':out})
    text="""# 正式结果补充：效应大小、负例和解释边界

本文件在唯一正式测试完成后，从已保存指标整理；没有新增模型评价、重新选择检查点或调整算法。下述最差例只是透明披露，完整600个差值均保存于paired_per_geometry_details.csv。主比较和总体结论不按这些例子更换。

- **LocalTangent已达到同一性能水平。** best均值0.153715，v2为0.153420，差0.000295（相对0.192%），条件几何区间虽不跨零，却远小于对Linear的效应；不能把这个小差说成单种子训练之外的稳健优势，也没有预设等效界可做严格等效认证。
- **v2对v1的场精度优势未确立。** best改善0.152%，仅46/100几何改善，区间跨零；固定last反而v2为0.152286、v1为0.152092，v2恶化0.127%，59/100几何较差，区间同样跨零。保留两种checkpoint，不能据test选择更有利者作为新主结果。
- **方向并非一致改善。** best左入射v2相对LocalTangent恶化0.010%（25胜25负，差的区间跨零）；右入射改善0.388%。v2对v1右入射均值恶化0.117%，区间跨零；不是所有方向都获益。
- **远场仍差，修复连续性没有自动修复辐射场预测。** 三种几何载波的far RelL2都约1；v2的1.000711比v1的0.997092更差，均差v1−v2=-0.003619，几何区间[-0.005047,-0.002227]，65/100较差。此处按远场自身能量归一化，不能等同对总误差的贡献。
- **端口收益有限且指标有分歧。** v2对v1非输入端复代理误差从0.148367降到0.142270（改善4.11%，62/100改善，区间支持）；幅值和平方幅值误差均值却略差，相关区间跨零。v2相对LocalTangent的复/幅值/平方幅值代理差区间全部跨零，没有额外端口收益的明确证据。
- **对路线A的主张应收缩。** 新家族支持几何相位参考显著优于Linear；LocalTangent与弧长Path接近，故证据支持一般几何对齐，更难支持“全域最近弧长延拓有独有精度优势”。v2有连续性修复价值，但尚无清楚的主场学习收益。不能据此认证高水平投稿或实用FDTD替代，也不能否认已有Y-Branch冻结结果；应把器件依赖与强基线结果并列。

"""
    text+='\n'.join(tables)
    (OUT/'正式结果负例与效应大小补充.md').write_text(text,encoding='utf-8')
    print('All paired geometry details and adverse results saved',flush=True)
if __name__=='__main__':main()
