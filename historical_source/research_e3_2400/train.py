"""Fixed 2400-update budget experiment, seed 2026; no test access."""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_e1'))
import run_linear as e1
ff=e1.ff
STEPS=2400;SEED=2026;LR=.002;BATCH=2;PRECISION='fp32'


def prepare():return e1.prepare()
def carrier(data,variant):return data['c'] if variant=='linear' else data['path_c']


def score(model,data,ids,variant):
    current={**data,'c':carrier(data,variant)}
    return ff.score(model,current,ids,variant,PRECISION,BATCH)


def setup():
    ff.configure();e1.bind()


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def prior_summary(variant):
    base=ROOT.parent/('research_e1' if variant=='linear' else 'research_v5')
    return ff.read(base/'runs'/f'{variant}_lr0.002_seed2026'/'summary.json')


def check_old():
    old,frozen=e1.check_existing()
    em=ff.read(e1.ROOT/'manifest.json')
    for name,digest in em['sources'].items():assert ff.sha(ROOT.parent/name)==digest
    for name,digest in ff.read(e1.ROOT/'frozen_weights.json').items():
        assert ff.sha(e1.ROOT/'runs'/name/'best.pt')==digest
    return old


def train_one(data,variant):
    run=ROOT/'runs'/f'{variant}_steps2400_seed2026';run.mkdir(parents=True,exist_ok=True)
    if (run/'summary.json').exists():
        r=ff.read(run/'summary.json')
        assert ff.sha(run/'best.pt')==r['best_sha256'] and ff.sha(run/'last.pt')==r['last_sha256']
        print('Reusing completed '+run.name,flush=True);return r
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    random.seed(SEED);model=ff.model_new(SEED);initial=ff.state_hash(model)
    rng=np.random.default_rng(SEED);batches=np.stack([rng.choice(data['tr'],BATCH,replace=False) for _ in range(STEPS)])
    batch_hash=hashlib.sha256(batches.tobytes()).hexdigest();prefix=hashlib.sha256(batches[:1200].tobytes()).hexdigest()
    old=prior_summary(variant)
    assert initial==old['initial_state_sha256'] and prefix==old['batches_sha256']
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=LR*.05)
    history=[];best=float('inf');best_step=0;best_values=[];first=1;elapsed=0;resumes=0;peak_before=0
    if (run/'last.pt').exists():
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert ck['variant']==variant and ck['steps']==STEPS and ck['seed']==SEED
        assert ck['initial_state_sha256']==initial and np.array_equal(ck['batches'],batches)
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best_val'];best_step=ck['best_step'];best_values=ck['best_values']
        first=ck['step']+1;elapsed=ck['elapsed_seconds'];peak_before=ck['peak_allocated_MiB'];resumes=ck['resume_count']+1
        restore_rng(ck['rng']);del ck
        print('Restored full state at step '+str(first-1),flush=True)
    elif any(run.iterdir()):raise RuntimeError('Incomplete run without resumable state: '+str(run))
    start=time.perf_counter();c=carrier(data,variant)
    for step in range(first,STEPS+1):
        ids=batches[step-1];model.train();opt.zero_grad(set_to_none=True)
        used_lr=opt.param_groups[0]['lr']
        pred=ff.forward(model,data['x'][ids].cuda(),c[ids].cuda(),variant,PRECISION)
        loss=ff.rel_mse(pred,data['y'][ids].cuda()).mean()
        assert torch.isfinite(loss),'Nonfinite loss'
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5)
        assert torch.isfinite(grad),'Nonfinite gradient'
        opt.step();sch.step()
        if step==1 or step%50==0:
            values=score(model,data,data['va'],variant);val=float(values.mean())
            row=dict(step=step,loss=float(loss.detach()),val_rel_l2=val,used_lr=used_lr,next_lr=sch.get_last_lr()[0],seconds=elapsed+time.perf_counter()-start)
            history.append(row)
            if val<best:
                best=val;best_step=step;best_values=values.tolist()
                save(run/'best.pt',dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},variant=variant,
                    lr=LR,seed=SEED,step=step,steps=STEPS,precision=PRECISION,batch=BATCH,config=ff.CONFIG,
                    initial_state_sha256=initial,batches_sha256=batch_hash,**data['meta']))
            ff.write(run/'history.json',history)
            print(json.dumps(dict(run=run.name,**row,best_val=best)),flush=True)
        if step%200==0 or step==STEPS:
            save(run/'last.pt',dict(state=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=capture_rng(),
                variant=variant,lr=LR,seed=SEED,step=step,steps=STEPS,precision=PRECISION,batch=BATCH,config=ff.CONFIG,
                initial_state_sha256=initial,batches_sha256=batch_hash,first1200_batches_sha256=prefix,batches=batches,
                history=history,best_val=best,best_step=best_step,best_values=best_values,
                elapsed_seconds=elapsed+time.perf_counter()-start,resume_count=resumes,
                peak_allocated_MiB=max(peak_before,torch.cuda.max_memory_allocated()/1024**2),**data['meta']))
    duration=elapsed+time.perf_counter()-start
    final_val=score(model,data,data['va'],variant);final_train=score(model,data,data['tr'],variant)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
    best_train=score(model,data,data['tr'],variant)
    result=dict(variant=variant,seed=SEED,steps=STEPS,lr=LR,batch=BATCH,precision=PRECISION,n=64,
        best_val=best,best_step=best_step,best_val_per_sample=best_values,best_train=float(best_train.mean()),
        final_val=float(final_val.mean()),final_val_per_sample=final_val.tolist(),final_train=float(final_train.mean()),
        train_wall_seconds=duration,peak_allocated_MiB=max(peak_before,torch.cuda.max_memory_allocated()/1024**2),
        initial_state_sha256=initial,batches_sha256=batch_hash,first1200_batches_sha256=prefix,resume_count=resumes,
        parameters=sum(p.numel() for p in model.parameters()),real_scalar_degrees=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
        best_sha256=ff.sha(run/'best.pt'),last_sha256=ff.sha(run/'last.pt'),test_evaluated=False)
    ff.write(run/'summary.json',result);print(json.dumps(result),flush=True)
    del model,opt,sch,ck
    return result


def main():
    setup();old=check_old();data=prepare()
    for key,value in data['meta'].items():assert old[key]==value
    sources=[Path(__file__),ROOT/'PROTOCOL.md',e1.ROOT/'run_linear.py',e1.V5/'fourier_full.py']
    manifest=dict(date='2026-11-12',variants=['direct','linear','path'],seed=SEED,lr=LR,steps=STEPS,batch=BATCH,precision=PRECISION,
        scheduler='independent cosine horizon 2400; eta_min=0.0001',checkpoint_selection='original 8 validation only',
        old_seed2026_results={v:prior_summary(v) for v in ['direct','linear','path']},
        sources={str(p.relative_to(ROOT.parent)):ff.sha(p) for p in sources},inherited_sources=old['sources'],
        data_sha256=old['data_sha256'],**data['meta'])
    target=ROOT/'manifest.json'
    if target.exists():assert ff.read(target)==manifest,'Frozen configuration changed'
    else:ff.write(target,manifest)
    results=[]
    for variant in manifest['variants']:
        results.append(train_one(data,variant));ff.write(ROOT/'results.json',results)
    ff.write(ROOT/'frozen_weights.json',{p.relative_to(ROOT).as_posix():ff.sha(p) for p in (ROOT/'runs').glob('*/*.pt')})
    print('All 3 fixed 2400-step runs complete.',flush=True)


if __name__=='__main__':main()
