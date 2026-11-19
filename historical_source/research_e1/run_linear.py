"""E1 extension reusing frozen v5 trainer; never alters historical code or weights."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_v5'))
import fourier_full as ff

V5=ff.ROOT
ORIGINAL_FORWARD=ff.forward


def forward(model,x,c,variant,precision):
    # Original trainer's carrier-multiplication branch, with linear c supplied.
    return ORIGINAL_FORWARD(model,x,c,'path' if variant=='linear' else variant,precision)


def prepare():
    previous=ff.ROOT
    ff.ROOT=V5
    try:data=ff.prepare()
    finally:ff.ROOT=previous
    calibration=data['meta']['calibration']
    k=np.polyval(calibration,data['eps'].max((1,2)))
    carriers=[]
    for i,(eps,src) in enumerate(zip(data['eps'],data['src'])):
        f=ff.geometry_features(eps,src)
        carriers.append(np.exp(1j*k[i]*f['linear']).astype(np.complex64))
    data['path_c']=data['c']
    data['c']=torch.from_numpy(np.stack(carriers)[:,None])
    assert float((data['c'].abs()-1).abs().max())<2e-7
    return data


def bind():
    ff.ROOT=ROOT
    ff.forward=forward


def check_existing():
    manifest=ff.read(V5/'manifest.json')
    assert ff.sha(ROOT.parent/'picflow_ybranch_smoke_80.npz')==manifest['data_sha256']
    for name,digest in manifest['sources'].items():assert ff.sha(ROOT.parent/name)==digest,name
    frozen=ff.read(V5/'frozen_weights.json');assert len(frozen)==10
    for name,digest in frozen.items():assert ff.sha(V5/'runs'/name/'best.pt')==digest,name
    return manifest,frozen


def main():
    ff.configure();old,frozen=check_existing();data=prepare()
    for key,value in data['meta'].items():assert old[key]==value
    settings=ff.read(V5/'preflight.json')
    manifest=dict(date='2026-11-12',purpose='E1 Linear extension; validation only',
        reused_v5_weights=frozen,reused_v5_manifest_sha256=ff.sha(V5/'manifest.json'),
        inherited_sources=old['sources'],settings=settings,
        data_sha256=old['data_sha256'],lrs=[.0005,.001,.002],search_seed=2026,repeat_seeds=[2027,2028],
        steps=1200,primary_comparison='path vs linear',
        sources={str(p.relative_to(ROOT.parent)):ff.sha(p) for p in [Path(__file__),ROOT/'PROTOCOL.md']},**data['meta'])
    dst=ROOT/'manifest.json'
    if dst.exists():assert ff.read(dst)==manifest
    else:ff.write(dst,manifest)
    bind();search=[]
    for lr in manifest['lrs']:
        r=ff.train_one(data,'linear',lr,2026,settings)
        baseline=ff.read(V5/'runs'/f'direct_lr{lr:g}_seed2026'/'summary.json')
        assert r['initial_state_sha256']==baseline['initial_state_sha256']
        assert r['batches_sha256']==baseline['batches_sha256']
        search.append(r);ff.write(ROOT/'search_results.json',search)
    selected=min(search,key=lambda r:r['best_val']);ff.write(ROOT/'selected_lr.json',selected)
    repeats=[]
    for seed in manifest['repeat_seeds']:
        r=ff.train_one(data,'linear',selected['lr'],seed,settings)
        baseline=ff.read(V5/'runs'/f'direct_lr0.002_seed{seed}'/'summary.json')
        assert r['initial_state_sha256']==baseline['initial_state_sha256']
        assert r['batches_sha256']==baseline['batches_sha256']
        repeats.append(r);ff.write(ROOT/'repeat_results.json',repeats)
    final=ff.read(V5/'final_results.json')+[selected]+repeats
    ff.write(ROOT/'final_results.json',final)
    summary={v:dict(lr=next(r['lr'] for r in final if r['variant']==v),
        val_mean=float(np.mean([r['best_val'] for r in final if r['variant']==v])),
        val_seed_sd=float(np.std([r['best_val'] for r in final if r['variant']==v],ddof=1)),
        train_mean=float(np.mean([r['best_train'] for r in final if r['variant']==v])),
        mean_train_seconds=float(np.mean([r['train_wall_seconds'] for r in final if r['variant']==v]))) for v in ['direct','linear','path']}
    ff.write(ROOT/'summary.json',summary)
    ff.write(ROOT/'frozen_weights.json',{p.parent.name:ff.sha(p) for p in (ROOT/'runs').glob('*/best.pt')})
    print(summary,flush=True)


if __name__=='__main__':main()
