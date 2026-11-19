"""Verify new best/terminal weights and replay historical seed-2026 baselines."""
import csv
import hashlib
import math
import numpy as np
import torch
from train import ROOT,ff,e1,setup,prepare,carrier,score,check_old,prior_summary,STEPS,SEED,LR,BATCH,PRECISION
from experiment import metric_arrays


@torch.no_grad()
def main():
    setup();check_old();data=prepare();manifest=ff.read(ROOT/'manifest.json')
    for group in ['sources','inherited_sources']:
        for name,digest in manifest[group].items():assert ff.sha(ROOT.parent/name)==digest
    assert ff.sha(ROOT.parent/'picflow_ybranch_smoke_80.npz')==manifest['data_sha256']
    frozen=ff.read(ROOT/'frozen_weights.json');assert len(frozen)==6
    for name,digest in frozen.items():assert ff.sha(ROOT/name)==digest
    results=ff.read(ROOT/'results.json');assert len(results)==3
    checks=[];rows=[];curves=[];names=[];predictions=[];diagnostics={};comparisons={}
    initials=set();batchhashes=set();rng=np.random.default_rng(SEED)
    batches=np.stack([rng.choice(data['tr'],BATCH,replace=False) for _ in range(STEPS)])
    for r in results:
        v=r['variant'];old=prior_summary(v);run=ROOT/'runs'/f'{v}_steps2400_seed2026'
        hist=ff.read(run/'history.json')
        assert len(hist)==49 and [h['step'] for h in hist]==[1]+list(range(50,2401,50))
        assert all(np.isfinite([h['loss'],h['val_rel_l2'],h['seconds']]).all() for h in hist)
        assert r['best_step']==min(hist,key=lambda h:h['val_rel_l2'])['step']
        assert r['best_val']==min(h['val_rel_l2'] for h in hist)
        assert np.isclose(r['final_val'],hist[-1]['val_rel_l2'])
        assert r['steps']==2400 and r['seed']==2026 and r['lr']==.002 and not r['test_evaluated']
        assert r['initial_state_sha256']==old['initial_state_sha256']
        assert r['first1200_batches_sha256']==old['batches_sha256']
        assert r['batches_sha256']==hashlib.sha256(batches.tobytes()).hexdigest()
        initials.add(r['initial_state_sha256']);batchhashes.add(r['batches_sha256'])
        for h in hist:
            used=LR*.05+(LR-LR*.05)*(1+math.cos(math.pi*(h['step']-1)/STEPS))/2
            nxt=LR*.05+(LR-LR*.05)*(1+math.cos(math.pi*h['step']/STEPS))/2
            assert np.isclose(h['used_lr'],used,rtol=1e-9,atol=1e-12)
            assert np.isclose(h['next_lr'],nxt,rtol=1e-9,atol=1e-12)
            curves.append(dict(variant=v,steps_budget=2400,**h))
        base=ROOT.parent/('research_e1' if v=='linear' else 'research_v5')
        oldrun=base/'runs'/f'{v}_lr0.002_seed2026';oldhist=ff.read(oldrun/'history.json')
        assert oldhist[-1]['step']==1200
        comparisons[v]=dict(best1200=old['best_val'],best2400=r['best_val'],
            best_step1200=old['best_step'],best_step2400=r['best_step'],
            relative_best_error_reduction=float(1-r['best_val']/old['best_val']),
            terminal1200_logged=oldhist[-1]['val_rel_l2'],terminal2400=r['final_val'],
            relative_terminal_error_reduction=float(1-r['final_val']/oldhist[-1]['val_rel_l2']),
            train_at_best1200=old['best_train'],train_at_best2400=r['best_train'],train_terminal2400=r['final_train'])
        candidates=[('1200_best',oldrun/'best.pt',old['best_val']),('2400_best',run/'best.pt',r['best_val']),('2400_terminal',run/'last.pt',r['final_val'])]
        for tag,path,expected in candidates:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            for key,value in data['meta'].items():assert ck[key]==value
            assert ck['seed']==SEED and ck['variant']==v and ck['initial_state_sha256']==old['initial_state_sha256']
            if tag=='2400_terminal':
                assert ck['step']==2400 and np.array_equal(ck['batches'],batches)
                assert ck['history']==hist and ck['best_step']==r['best_step']
                assert ck['scheduler']['last_epoch']==2400
                assert np.isclose(ck['optimizer']['param_groups'][0]['lr'],.0001)
                assert set(ck['rng'])=={'python','numpy','torch','cuda'}
                assert all(float(state['step'])==2400 for state in ck['optimizer']['state'].values())
            model=ff.model_new(SEED).eval();model.load_state_dict(ck['state'])
            values=score(model,data,data['va'],v);delta=abs(float(values.mean())-expected)
            assert delta<1e-5,(v,tag,delta)
            checks.append(dict(variant=v,checkpoint=tag,absolute_replay_difference=delta))
            for gid,value in zip(data['meta']['val_geometry_ids'],values):
                rows.append(dict(variant=v,checkpoint=tag,geometry_id=gid,val_rel_l2=float(value)))
            chunks=[];c=carrier(data,v)
            for j in range(0,len(data['va']),BATCH):
                ids=data['va'][j:j+BATCH]
                chunks.append(ff.forward(model,data['x'][ids].cuda(),c[ids].cuda(),v,PRECISION).cpu().numpy())
            pred=np.concatenate(chunks)*data['meta']['scale'];truth=data['y'][data['va']].numpy()*data['meta']['scale']
            label=v+'_'+tag;names.append(label);predictions.append(pred)
            metrics=metric_arrays(pred[:,0],truth[:,0]);diagnostics[label]={k:float(value.mean()) for k,value in metrics.items()}
            if tag.startswith('2400'):
                train_values=score(model,data,data['tr'],v)
                reference=r['best_train'] if tag=='2400_best' else r['final_train']
                assert abs(float(train_values.mean())-reference)<1e-5
            del model,ck
    assert len(initials)==len(batchhashes)==1
    np.savez_compressed(ROOT/'validation_predictions.npz',run=np.array(names),prediction=np.stack(predictions),truth=truth,
        geometry_id=np.array(data['meta']['val_geometry_ids']),eps=data['eps'][data['va']])
    for path,records in [(ROOT/'validation_samples.csv',rows),(ROOT/'curves2400.csv',curves)]:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    paired={}
    for steps in [1200,2400]:
        for a,b in [('direct','linear'),('linear','path'),('direct','path')]:
            va=np.array([row['val_rel_l2'] for row in rows if row['variant']==a and row['checkpoint']==f'{steps}_best'])
            vb=np.array([row['val_rel_l2'] for row in rows if row['variant']==b and row['checkpoint']==f'{steps}_best'])
            diff=va-vb
            paired[f'{steps}_{a}_minus_{b}']=dict(mean_difference=float(diff.mean()),relative_error_reduction=float(1-vb.mean()/va.mean()),
                improved_geometries=int((diff>0).sum()),worsened_geometries=int((diff<0).sum()),per_geometry_difference=diff.tolist())
    preserved=0
    for folder,mapfile,sub in [('research_e1','frozen_weights.json','runs'),('research_v5','frozen_weights.json','runs'),
        ('research_v4','stage_a/frozen_weights.json','stage_a/runs'),('research_v3','exposure_matched/frozen_weights.json','exposure_matched/runs')]:
        for name,digest in ff.read(ROOT.parent/folder/mapfile).items():assert ff.sha(ROOT.parent/folder/sub/name/'best.pt')==digest;preserved+=1
    for name,digest in ff.read(ROOT.parent/'research_v3/fresh_manifest.json')['weights'].items():
        assert ff.sha(ROOT.parent/'research_v3/runs'/name/'best.pt')==digest;preserved+=1
    ff.write(ROOT/'validation_diagnostics.json',diagnostics)
    report=dict(status='passed',replays=checks,max_replay_absolute_difference=max(c['absolute_replay_difference'] for c in checks),
        old_frozen_weights_preserved=preserved,all_batch_sequences_and_initialization_match=True,cosine_schedule_verified=True,
        complete_terminal_optimizer_scheduler_rng_present=True,comparisons=comparisons,paired_comparisons=paired,
        seed=2026,test_evaluated=False,only_2400_new_training=True)
    ff.write(ROOT/'verification.json',report);print(report)


if __name__=='__main__':main()
