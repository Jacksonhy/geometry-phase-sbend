"""Full-resolution paired representation study using unchanged official model bodies."""
import argparse
import copy
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_v4'))
sys.path.insert(0,str(ROOT.parent/'research_v3'))
from experiment import geometry_features
from official_architecture import load_original_model_class

MODEL_CLASS=load_original_model_class()
CONFIG=dict(in_channels=12,out_channels=2,dim=64,kernel_list=[64]*12,
            kernel_size_list=[1]*12,padding_list=[1]*12,hidden_list=[256],
            mode_list=[(40,70)]*12,act_func='GELU',dropout_rate=.1,drop_path_rate=0,
            eps_min=2.0851359,eps_max=2.0851359+4.2,pos_encoding='exp',conv_stem=True,
            aug_path=False,ffn=True,ffn_dwconv=True,with_cp=False)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


def model_new(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    return MODEL_CLASS(**CONFIG,device=torch.device('cuda')).cuda()


def state_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer_inputs(eps,src,calibration):
    """No target field is accepted by this geometry/source-only interface."""
    h,w=eps.shape[-2:];gy,gx=np.mgrid[:h,:w]
    k=np.polyval(calibration,eps.max((1,2)))
    features=[];carriers=[]
    for i,(a,b) in enumerate(zip(eps,src)):
        f=geometry_features(a,b)
        features.append(np.stack([(a-2.0851359)/4.2,b,gx/(w-1)*2-1,gy/(h-1)*2-1,
            np.full((h,w),f['sx']/(w-1)*2-1),np.full((h,w),f['sy']/(h-1)*2-1),
            np.full((h,w),f['direction']),np.full((h,w),k[i]),f['path']/w,f['linear']/w]).astype(np.float32))
        carriers.append(np.exp(1j*k[i]*f['path']).astype(np.complex64))
    a=np.stack(features)
    # Preserve epsilon as purely real at complex channel 0. Other pairs merely
    # pack real-valued covariates for the official complex-input interface.
    z=np.zeros((len(eps),6,h,w),dtype=np.complex64)
    z[:,0]=a[:,0]
    for j in range(1,10):
        c=1+(j-1)//2
        if j%2: z[:,c].real=a[:,j]
        else: z[:,c].imag=a[:,j]
    assert np.all(z[:,0].imag==0)
    return torch.from_numpy(z),torch.from_numpy(np.stack(carriers)[:,None])


def prepare():
    with np.load(ROOT.parent/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z:
        keep=np.flatnonzero(z['split']!='test')
        d={k:z[k][keep] for k in ['eps','src_mask','Ez_real','Ez_imag','split','geometry_id','input_port']}
    tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='val')
    assert len(tr)==64 and len(va)==8
    field=d['Ez_real']+1j*d['Ez_imag']
    slopes=[abs(np.polyfit(np.arange(120),np.unwrap(np.angle(field[i,78:82,40:160].mean(0))),1)[0]) for i in tr]
    calibration=np.polyfit(d['eps'].max((1,2))[tr],slopes,1)
    scale=float(np.sqrt(np.mean(abs(field[tr])**2)))
    x,c=infer_inputs(d['eps'],d['src_mask'],calibration)
    meta=dict(calibration=calibration.tolist(),scale=scale,train_geometry_ids=d['geometry_id'][tr].tolist(),
              val_geometry_ids=d['geometry_id'][va].tolist(),no_test_evaluated=True,resolution=[160,480])
    return dict(x=x,y=torch.from_numpy((field[:,None]/scale).astype(np.complex64)),c=c,
                tr=tr,va=va,meta=meta,eps=d['eps'],src=d['src_mask'])


def autocast(precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def forward(model,x,c,variant,precision):
    batch=len(x)
    wave=torch.full((batch,1),1.55,device=x.device)
    grid=torch.full((batch,2),.05,device=x.device)
    with autocast(precision):
        out=model(x,wave,grid)
    out=out.to(torch.complex64)
    assert out.shape==x[:,0:1].shape
    return out*c if variant=='path' else out


def rel_mse(pred,target):
    return (pred-target).abs().square().sum((1,2,3))/target.abs().square().sum((1,2,3))


def preflight():
    if (ROOT/'preflight.json').exists():
        raise RuntimeError('Preflight already frozen; use its recorded settings.')
    data=prepare(); records=[]
    for precision in ['fp32','bf16']:
        for batch in [2]:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=None; opt=None
            try:
                model=model_new(615);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
                x=data['x'][:batch].cuda();y=data['y'][:batch].cuda();c=data['c'][:batch].cuda()
                durations=[]
                for step in range(4):
                    model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    pred=forward(model,x,c,'path',precision)
                    loss=rel_mse(pred,y).mean();assert torch.isfinite(loss)
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.cuda.synchronize()
                    durations.append(time.perf_counter()-start)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                records.append(dict(precision=precision,batch=batch,status='passed',times=durations,
                    warm_step_seconds=float(np.median(durations[1:])),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    parameters=sum(p.numel() for p in model.parameters()),
                    real_scalar_degrees=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())))
            except (RuntimeError,AssertionError) as e:
                records.append(dict(precision=precision,batch=batch,status='failed',error=str(e)))
            print(json.dumps(records[-1]),flush=True)
            del model,opt
            for name in ['x','y','c','pred','loss']:
                if name in locals():
                    # Retained local tensors are released below by rebinding.
                    pass
            x=y=c=pred=loss=None
            torch.cuda.empty_cache()
    good=[r for r in records if r['status']=='passed']
    if not good: raise RuntimeError('No executable configuration; inspect console before adapting.')
    fp=next((r for r in good if r['precision']=='fp32'),None)
    bf=next((r for r in good if r['precision']=='bf16'),None)
    chosen=bf if bf and (not fp or bf['warm_step_seconds']<fp['warm_step_seconds']*.85) else fp or bf
    result=dict(config=CONFIG,measurements=records,selected_precision=chosen['precision'],selected_batch=chosen['batch'],
        decision='bf16 only if supported and at least 15% faster; otherwise FP32',test_evaluated=False,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
    write(ROOT/'preflight.json',result)


@torch.no_grad()
def score(model,data,ids,variant,precision,batch):
    model.eval();values=[]
    for j in range(0,len(ids),batch):
        ind=ids[j:j+batch]
        pred=forward(model,data['x'][ind].cuda(),data['c'][ind].cuda(),variant,precision)
        values.append(rel_mse(pred,data['y'][ind].cuda()).sqrt().cpu())
    return torch.cat(values).numpy()


def train_one(data,variant,lr,seed,settings):
    run=ROOT/'runs'/f'{variant}_lr{lr:g}_seed{seed}'
    if (run/'summary.json').exists():
        result=read(run/'summary.json')
        assert sha(run/'best.pt')==result['best_sha256']
        print('Reusing completed:',run.name,flush=True);return result
    if run.exists(): raise RuntimeError('Incomplete run exists; do not silently overwrite: '+str(run))
    run.mkdir(parents=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=model_new(seed);initial_hash=state_hash(model)
    batch=settings['selected_batch'];precision=settings['selected_precision'];steps=1200
    rng=np.random.default_rng(seed);batches=np.stack([rng.choice(data['tr'],batch,replace=False) for _ in range(steps)])
    batch_hash=hashlib.sha256(batches.tobytes()).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=lr*.05)
    # Inputs are staged per batch: no test labels and no large persistent GPU dataset.
    history=[];best=float('inf');start=time.perf_counter()
    for step,ids in enumerate(batches,1):
        model.train();opt.zero_grad(set_to_none=True)
        pred=forward(model,data['x'][ids].cuda(),data['c'][ids].cuda(),variant,precision)
        loss=rel_mse(pred,data['y'][ids].cuda()).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss: '+run.name)
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient: '+run.name)
        opt.step();sch.step()
        if step==1 or step%50==0:
            values=score(model,data,data['va'],variant,precision,batch);val=float(values.mean())
            row=dict(step=step,loss=float(loss.detach()),val_rel_l2=val,seconds=time.perf_counter()-start)
            history.append(row)
            if val<best:
                best=val;best_step=step;best_values=values.tolist()
                torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},variant=variant,
                    lr=lr,seed=seed,step=step,steps=steps,precision=precision,batch=batch,config=CONFIG,
                    initial_state_sha256=initial_hash,batches_sha256=batch_hash,**data['meta']),run/'best.pt')
            write(run/'history.json',history)
            print(json.dumps(dict(run=run.name,**row,best_val=best)),flush=True)
    duration=time.perf_counter()-start
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
    train_values=score(model,data,data['tr'],variant,precision,batch)
    result=dict(variant=variant,lr=lr,seed=seed,n=64,steps=steps,batch=batch,precision=precision,
        best_val=best,best_step=best_step,val_per_sample=best_values,best_train=float(train_values.mean()),
        train_wall_seconds=duration,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
        parameters=sum(p.numel() for p in model.parameters()),
        real_scalar_degrees=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
        initial_state_sha256=initial_hash,batches_sha256=batch_hash,best_sha256=sha(run/'best.pt'),test_evaluated=False)
    write(run/'summary.json',result);print(json.dumps(result),flush=True)
    del model,opt,sch,pred,loss,ck
    return result


def train():
    settings=read(ROOT/'preflight.json');data=prepare()
    sources=[Path(__file__),ROOT/'PROTOCOL.md',ROOT.parent/'research_v4/official_architecture.py',
             ROOT.parent/'research_v4/official_bridge.py',ROOT.parent/'research_v3/experiment.py']
    upstream=ROOT.parent/'references/NeurOLight-source/NeurOLight-main'
    sources += [upstream/p for p in ['core/models/pde_base.py','core/models/neurolight_cnn.py','core/models/layers/neurolight_conv2d.py','core/models/layers/fno_conv2d.py']]
    manifest=dict(config=CONFIG,settings=settings,variants=['direct','path'],lrs=[.0005,.001,.002],search_seed=2026,
        repeat_seeds=[2027,2028],steps=1200,checkpoint_selection='original 8 validation samples only',
        data_sha256=sha(ROOT.parent/'picflow_ybranch_smoke_80.npz'),
        sources={str(p.relative_to(ROOT.parent)):sha(p) for p in sources},**data['meta'])
    path=ROOT/'manifest.json'
    if path.exists(): assert read(path)==manifest,'Frozen manifest changed'
    else: write(path,manifest)
    search=[];selected={};repeats=[]
    for lr in manifest['lrs']:
        for variant in manifest['variants']:
            search.append(train_one(data,variant,lr,2026,settings));write(ROOT/'search_results.json',search)
    for variant in manifest['variants']:
        selected[variant]=min([r for r in search if r['variant']==variant],key=lambda r:r['best_val'])
    write(ROOT/'selected_lrs.json',selected)
    for seed in manifest['repeat_seeds']:
        for variant in manifest['variants']:
            repeats.append(train_one(data,variant,selected[variant]['lr'],seed,settings))
            write(ROOT/'repeat_results.json',repeats)
    final=[selected[v] for v in manifest['variants']]+repeats;write(ROOT/'final_results.json',final)
    summary={v:dict(lr=selected[v]['lr'],val_mean=float(np.mean([r['best_val'] for r in final if r['variant']==v])),
        val_seed_sd=float(np.std([r['best_val'] for r in final if r['variant']==v],ddof=1)),
        train_mean=float(np.mean([r['best_train'] for r in final if r['variant']==v])),
        mean_train_seconds=float(np.mean([r['train_wall_seconds'] for r in final if r['variant']==v]))) for v in manifest['variants']}
    write(ROOT/'summary.json',summary)
    write(ROOT/'frozen_weights.json',{p.parent.name:sha(p) for p in (ROOT/'runs').glob('*/best.pt')})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train']);args=parser.parse_args()
    configure()
    preflight() if args.action=='preflight' else train()
