"""Audit frozen full-resolution runs and replay only train/validation artifacts."""
import csv
import hashlib
import json
import numpy as np
import torch
from fourier_full import ROOT,CONFIG,configure,prepare,model_new,score,forward,read,write,sha
from experiment import metric_arrays


@torch.no_grad()
def main():
    configure();data=prepare();manifest=read(ROOT/'manifest.json')
    assert manifest['data_sha256']==sha(ROOT.parent/'picflow_ybranch_smoke_80.npz')
    for name,digest in manifest['sources'].items():assert sha(ROOT.parent/name)==digest,name
    assert manifest['resolution']==[160,480] and manifest['no_test_evaluated']
    assert len(set(manifest['train_geometry_ids']))==64
    assert len(set(manifest['val_geometry_ids']))==8
    assert set(manifest['train_geometry_ids']).isdisjoint(manifest['val_geometry_ids'])
    for key,value in data['meta'].items(): assert value==manifest[key]
    assert data['y'].shape[-2:]==(160,480)
    frozen=read(ROOT/'frozen_weights.json');assert len(frozen)==10
    search=read(ROOT/'search_results.json');assert len(search)==6
    selected=read(ROOT/'selected_lrs.json');final=read(ROOT/'final_results.json');assert len(final)==6
    summary=read(ROOT/'summary.json')
    settings=manifest['settings'];batch=settings['selected_batch'];precision=settings['selected_precision']
    expected={(v,lr,2026) for v in ['direct','path'] for lr in manifest['lrs']}
    expected|={(v,selected[v]['lr'],s) for v in ['direct','path'] for s in [2027,2028]}
    observed=set();checks=[];rows=[];initials={};batches={}
    predictions=[];prediction_names=[]
    for name,digest in frozen.items():
        run=ROOT/'runs'/name;assert sha(run/'best.pt')==digest
        r=read(run/'summary.json');hist=read(run/'history.json')
        assert len(hist)==25 and [v['step'] for v in hist]==[1]+list(range(50,1201,50))
        assert all(np.isfinite([v['loss'],v['val_rel_l2'],v['seconds']]).all() for v in hist)
        best=min(hist,key=lambda v:v['val_rel_l2'])
        assert best['step']==r['best_step'] and best['val_rel_l2']==r['best_val']
        assert r['steps']==1200 and r['batch']==batch and r['precision']==precision
        assert r['parameters']==1593058 and r['real_scalar_degrees']==2944738
        assert not r['test_evaluated'] and r['best_sha256']==digest
        observed.add((r['variant'],r['lr'],r['seed']))
        initials.setdefault(r['seed'],set()).add(r['initial_state_sha256'])
        batches.setdefault(r['seed'],set()).add(r['batches_sha256'])
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert ck['step']==r['best_step'] and ck['variant']==r['variant']
        assert ck['initial_state_sha256']==r['initial_state_sha256']
        assert ck['batches_sha256']==r['batches_sha256']
        for k in data['meta']:assert ck[k]==data['meta'][k]
        model=model_new(r['seed']).eval();model.load_state_dict(ck['state'])
        rel=score(model,data,data['va'],r['variant'],precision,batch)
        error=abs(float(rel.mean())-r['best_val'])
        assert error<1e-5,(name,error)
        assert np.allclose(rel,r['val_per_sample'],atol=1e-5)
        checks.append(dict(run=name,max_absolute_replay_difference=error))
        for geom,value in zip(manifest['val_geometry_ids'],rel):
            rows.append(dict(run=name,variant=r['variant'],seed=r['seed'],lr=r['lr'],geometry_id=geom,val_rel_l2=float(value)))
        if r in final:
            chunks=[]
            for j in range(0,len(data['va']),batch):
                ids=data['va'][j:j+batch]
                chunks.append(forward(model,data['x'][ids].cuda(),data['c'][ids].cuda(),r['variant'],precision).cpu().numpy())
            predictions.append(np.concatenate(chunks)*data['meta']['scale']);prediction_names.append(name)
        del model,ck
    assert observed==expected
    assert all(len(v)==1 for v in initials.values()),'Initial weights differ within paired seed'
    assert all(len(v)==1 for v in batches.values()),'Batch schedule differs within paired seed'
    for v in ['direct','path']:
        assert selected[v]==min([r for r in search if r['variant']==v],key=lambda r:r['best_val'])
        values=[r['best_val'] for r in final if r['variant']==v]
        assert np.isclose(np.mean(values),summary[v]['val_mean'])
        assert np.isclose(np.std(values,ddof=1),summary[v]['val_seed_sd'])
    # Store original-amplitude predictions only for the 8 original validation IDs.
    np.savez_compressed(ROOT/'validation_predictions.npz',run=np.array(prediction_names),
        prediction=np.stack(predictions),truth=data['y'][data['va']].numpy()*data['meta']['scale'],
        eps=data['eps'][data['va']],src_mask=data['src'][data['va']],geometry_id=np.array(manifest['val_geometry_ids']))
    diagnostics={}
    truth=data['y'][data['va']].numpy()[:,0]*data['meta']['scale']
    for name,pred in zip(prediction_names,predictions):
        metrics=metric_arrays(pred[:,0],truth)
        diagnostics[name]=dict(overall={k:float(v.mean()) for k,v in metrics.items()},
            per_geometry={g:{k:float(v[i]) for k,v in metrics.items()} for i,g in enumerate(manifest['val_geometry_ids'])})
    write(ROOT/'validation_diagnostics.json',diagnostics)
    with (ROOT/'validation_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    diffs=[]
    for seed in [2026,2027,2028]:
        pair={v:next(r for r in final if r['variant']==v and r['seed']==seed) for v in ['direct','path']}
        diffs.append(np.array(pair['direct']['val_per_sample'])-np.array(pair['path']['val_per_sample']))
    diffs=np.array(diffs);by_geom=diffs.mean(0)
    comparison=dict(direct_minus_path_mean=float(by_geom.mean()),
        relative_mean_error_reduction=float(1-summary['path']['val_mean']/summary['direct']['val_mean']),
        paired_seed_improvements=diffs.mean(1).tolist(),
        positive_geometries_after_seed_average=int((by_geom>0).sum()),
        negative_geometries_after_seed_average=int((by_geom<0).sum()),
        per_geometry_mean_difference=dict(zip(manifest['val_geometry_ids'],by_geom.tolist())),
        interpretation='Exploratory validation only; same 8 geometries used for checkpoint/LR selection. No significance claim.')
    # Preservation checks involve hashes only, never test inference.
    preserved=0
    for directory,mapfile,sub in [('research_v4','stage_a/frozen_weights.json','stage_a/runs'),
                                  ('research_v3','exposure_matched/frozen_weights.json','exposure_matched/runs')]:
        root=ROOT.parent/directory
        for name,digest in read(root/mapfile).items():assert sha(root/sub/name/'best.pt')==digest;preserved+=1
    for name,digest in read(ROOT.parent/'research_v3/fresh_manifest.json')['weights'].items():
        assert sha(ROOT.parent/'research_v3/runs'/name/'best.pt')==digest;preserved+=1
    result=dict(status='passed',runs_verified=10,validation_replays=checks,
        max_validation_replay_absolute_difference=max(c['max_absolute_replay_difference'] for c in checks),
        paired_initialization_and_batch_schedule_verified=True,previous_frozen_weights_preserved=preserved,
        no_downsampling_or_output_interpolation=True,test_evaluated=False,comparison=comparison)
    write(ROOT/'verification.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
