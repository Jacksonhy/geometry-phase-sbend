"""Verify report numbers, fixed source bytes, historical hashes and no fake FDTD."""
import ast, csv, hashlib, json, re
from pathlib import Path
import numpy as np
from audit_materials import ROOT, NEW, OUT, SRC, save, sha, historic

def main():
    before=json.loads((OUT/'historical_before.json').read_text())
    after=historic()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    extra=[p for p in after if p not in before]
    assert not changed,changed
    inventory=json.loads((OUT/'official_file_hashes.json').read_text())
    srcchanged=[p for p,r in inventory.items() if not (NEW/p).is_file() or sha(NEW/p)!=r['sha256']]
    assert not srcchanged,srcchanged
    mats=json.loads((OUT/'materials_summary.json').read_text())
    assert mats['index']['byte_identical'] and mats['audited_historical_samples']==80
    rows=list(csv.DictReader((OUT/'sparam_truth_comparison.csv').open(encoding='utf-8-sig')))
    summary=json.loads((OUT/'sparam_validation_summary.json').read_text())
    diffs={}
    for method in summary['methods']:
        for subset in ['all','reflection','noninput']:
            rr=[r for r in rows if r['method']==method and (subset=='all' or (r['is_reflection']=='True')==(subset=='reflection'))]
            t=np.array([complex(float(r['official_real']),float(r['official_imag'])) for r in rr])
            p=np.array([complex(float(r['estimate_real']),float(r['estimate_imag'])) for r in rr])
            recompute=float(np.sqrt(np.sum((p.real-t.real)**2+(p.imag-t.imag)**2)/len(rr)))
            delta=abs(recompute-summary['methods'][method][subset]['complex_rmse'])
            assert delta<1e-12
            diffs[method+'/'+subset]=delta
    candidates=json.loads((OUT/'sbend_geometry_candidates_seed2026.json').read_text())['records']
    assert len(candidates)==16 and len({r['geometry_id'] for r in candidates})==16
    assert all(r['status']=='geometry_only_NOT_FDTD' for r in candidates)
    assert not list(OUT.glob('*.npz'))
    report=(OUT/'官方数据与S参数审计报告.md').read_text()
    assert all(f'第{x}部分' in report for x in ['一','二','三','四','五','六'])
    broken=[]
    for target in re.findall(r'\]\((D:/[^)]+)\)',report):
        parts=target.rsplit(':',1)
        lineno=int(parts[1]) if len(parts)==2 and parts[1].isdigit() else None
        path=parts[0] if lineno else target
        local=Path('/mnt/d')/path[3:]
        if not local.exists():broken.append(target)
        elif lineno and lineno>len(local.read_text(encoding='utf-8-sig').splitlines()):broken.append(target)
    assert not broken,broken
    for p in OUT.glob('*.py'):ast.parse(p.read_text())
    idmap={r['geometry_id']:r for r in map(json.loads,(NEW/'geometries.jsonl').read_text().splitlines())}
    index=json.loads((NEW/'index.json').read_text())
    mismatches=[r['geometry_id'] for r in index if any(r[k]!=idmap[r['geometry_id']][k] for k in ['device','split'])]
    assert not mismatches
    save('verification.json',{'status':'passed','historical_files_checked':len(before),'historical_changed_or_missing':changed,
        'new_historical_paths':extra,'downloaded_files_checked':len(inventory),'downloaded_files_changed':srcchanged,
        'csv_rows':len(rows),'independent_complex_rmse_max_abs_difference':max(diffs.values()),
        'index_geometries_device_split_mismatches':len(mismatches),'report_local_links_valid':True,
        'python_syntax_passed':True,'sbend_geometry_candidates':16,'sbend_fdtd_samples':0,
        'network_training_runs':0,'report_sha256':sha(OUT/'官方数据与S参数审计报告.md'),
        'scripts_sha256':{p.name:sha(p) for p in OUT.glob('*.py')}})
    print((OUT/'verification.json').read_text())

if __name__=='__main__':main()
