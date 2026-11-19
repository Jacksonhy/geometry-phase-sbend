"""Aggregate frozen coefficient rows by physical output port; no calibration."""
import csv
from common import *
def main():
    with (OUT/'port_coefficients.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={}
    for r in rows:
        key=tuple(r[k] for k in ['method','checkpoint','reference','input_port','output_port','is_reflection'])
        groups.setdefault(key,[]).append(r)
    out=[]
    for key,rr in groups.items():
        out.append({**dict(zip(['method','checkpoint','reference','input_port','output_port','is_reflection'],key)),
            'geometries':len(rr),**{k:float(np.mean([float(a[k]) for a in rr])) for k in ['complex_error','amplitude_error','squared_amplitude_error']}})
    with (OUT/'port_summary_by_direction.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    table=['| 方法 | 输入→输出 | 复误差MAE | 幅值MAE | 平方幅值MAE |','|---|---|---:|---:|---:|']
    for r in out:
        if r['is_reflection']=='False' and ((r['checkpoint']=='best' and r['reference']=='truth_proxy') or r['method']=='truth_proxy'):
            table.append(f"| {r['method']} | {r['input_port']}→{r['output_port']} | {r['complex_error']:.6f} | {r['amplitude_error']:.6f} | {r['squared_amplitude_error']:.6f} |")
    text='# 逐端口方向代理指标\n\n模型best相对同次真实Ez代理；truth_proxy行相对Meep_native。每方向50个几何；反射另保存在CSV，仅探索，不认证回波损耗。\n\n'+'\n'.join(table)+'\n'
    (OUT/'逐端口代理指标.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
