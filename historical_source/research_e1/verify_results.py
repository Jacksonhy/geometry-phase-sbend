"""Replay all 15 E1 candidates on validation only; preserve historical artifacts."""
import csv
import numpy as np
import torch
from run_linear import ROOT,V5,ff,prepare,bind,check_existing
from experiment import metric_arrays


@torch.no_grad()
def main():
    ff.configure();old,oldfrozen=check_existing();data=prepare();bind()
    manifest=ff.read(ROOT/'manifest.json')
    for name,digest in manifest['sources'].items():assert ff.sha(ROOT.parent/name)==digest
    frozen=ff.read(ROOT/'frozen_weights.json');assert len(frozen)==5
    search=ff.read(ROOT/'search_results.json');assert len(search)==3
    selected=ff.read(ROOT/'selected_lr.json');assert selected==min(search,key=lambda r:r['best_val'])
    final=ff.read(ROOT/'final_results.json');assert len(final)==9
    summary=ff.read(ROOT/'summary.json');settings=manifest['settings'];batch=settings['selected_batch'];precision=settings['selected_precision']
    expected={('linear',lr,2026) for lr in manifest['lrs']}|{('linear',selected['lr'],s) for s in [2027,2028]}
    observed=set();rows=[];checks=[];initials={};batches={};predictions=[];names=[];diagnostics={}
    items=[(V5,name,digest) for name,digest in oldfrozen.items()]+[(ROOT,name,digest) for name,digest in frozen.items()]
    for base,name,digest in items:
        run=base/'runs'/name;assert ff.sha(run/'best.pt')==digest
        r=ff.read(run/'summary.json');hist=ff.read(run/'history.json')
        assert len(hist)==25 and [h['step'] for h in hist]==[1]+list(range(50,1201,50))
        assert all(np.isfinite([h['loss'],h['val_rel_l2'],h['seconds']]).all() for h in hist)
        best=min(hist,key=lambda h:h['val_rel_l2'])
        assert best['step']==r['best_step'] and best['val_rel_l2']==r['best_val']
        assert r['batch']==2 and r['precision']=='fp32' and r['steps']==1200 and not r['test_evaluated']
        assert r['parameters']==1593058 and r['real_scalar_degrees']==2944738
        initials.setdefault(r['seed'],set()).add(r['initial_state_sha256'])
        batches.setdefault(r['seed'],set()).add(r['batches_sha256'])
        if base==ROOT:observed.add((r['variant'],r['lr'],r['seed']))
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert ck['variant']==r['variant'] and ck['step']==r['best_step']
        for k,value in data['meta'].items():assert ck[k]==value
        model=ff.model_new(r['seed']).eval();model.load_state_dict(ck['state'])
        carrier=data['c'] if r['variant']=='linear' else data['path_c']
        current={**data,'c':carrier}
        rel=ff.score(model,current,data['va'],r['variant'],precision,batch)
        delta=abs(float(rel.mean())-r['best_val'])
        assert delta<1e-5 and np.allclose(rel,r['val_per_sample'],atol=1e-5)
        checks.append(dict(run=name,absolute_replay_difference=delta))
        for gid,value in zip(data['meta']['val_geometry_ids'],rel):
            rows.append(dict(run=name,variant=r['variant'],seed=r['seed'],lr=r['lr'],geometry_id=gid,val_rel_l2=float(value)))
        if r in final:
            parts=[]
            for j in range(0,len(data['va']),batch):
                ids=data['va'][j:j+batch]
                parts.append(ff.forward(model,data['x'][ids].cuda(),carrier[ids].cuda(),r['variant'],precision).cpu().numpy())
            pred=np.concatenate(parts)*data['meta']['scale'];predictions.append(pred);names.append(name)
            truth=data['y'][data['va']].numpy()*data['meta']['scale']
            metrics=metric_arrays(pred[:,0],truth[:,0])
            diagnostics[name]=dict(overall={k:float(v.mean()) for k,v in metrics.items()})
        del model,ck
    assert observed==expected
    assert all(len(s)==1 for s in initials.values()) and all(len(s)==1 for s in batches.values())
    for v in ['direct','linear','path']:
        rs=[r for r in final if r['variant']==v];assert len(rs)==3
        assert np.isclose(np.mean([r['best_val'] for r in rs]),summary[v]['val_mean'])
        assert np.isclose(np.std([r['best_val'] for r in rs],ddof=1),summary[v]['val_seed_sd'])
    comparisons={}
    for a,b in [('linear','path'),('direct','linear'),('direct','path')]:
        diff=[]
        for seed in [2026,2027,2028]:
            p={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in [a,b]}
            diff.append(np.array(p[a]['val_per_sample'])-np.array(p[b]['val_per_sample']))
        diff=np.array(diff);mean=diff.mean(0)
        comparisons[f'{a}_minus_{b}']=dict(mean_difference=float(mean.mean()),
            relative_error_reduction=float(1-summary[b]['val_mean']/summary[a]['val_mean']),
            paired_seed_differences=diff.mean(1).tolist(),positive_geometries=int((mean>0).sum()),negative_geometries=int((mean<0).sum()),
            per_geometry_difference=dict(zip(data['meta']['val_geometry_ids'],mean.tolist())))
    matching=[]
    for r in search:
        p=ff.read(V5/'runs'/f"path_lr{r['lr']:g}_seed2026"/'summary.json')
        matching.append(dict(lr=r['lr'],linear=r['best_val'],path=p['best_val'],linear_minus_path=r['best_val']-p['best_val']))
    with (ROOT/'validation_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(ROOT/'validation_predictions.npz',run=np.array(names),prediction=np.stack(predictions),truth=truth,
        geometry_id=np.array(data['meta']['val_geometry_ids']),eps=data['eps'][data['va']])
    ff.write(ROOT/'validation_diagnostics.json',diagnostics)
    preserved=10
    for folder,mapfile,sub in [('research_v4','stage_a/frozen_weights.json','stage_a/runs'),('research_v3','exposure_matched/frozen_weights.json','exposure_matched/runs')]:
        for name,digest in ff.read(ROOT.parent/folder/mapfile).items():
            assert ff.sha(ROOT.parent/folder/sub/name/'best.pt')==digest;preserved+=1
    for name,digest in ff.read(ROOT.parent/'research_v3/fresh_manifest.json')['weights'].items():
        assert ff.sha(ROOT.parent/'research_v3/runs'/name/'best.pt')==digest;preserved+=1
    result=dict(status='passed',validation_replays=checks,run_count=15,new_run_count=5,previous_weights_preserved=preserved,
        max_validation_replay_absolute_difference=max(r['absolute_replay_difference'] for r in checks),
        identical_initialization_and_batches_within_seed=True,comparisons=comparisons,
        matched_lr_seed2026_linear_vs_path=matching,test_evaluated=False,
        interpretation='Exploratory selection on the existing 8 validation geometries; no independent test or significance claim.')
    ff.write(ROOT/'verification.json',result)
    print(result)


if __name__=='__main__':main()
