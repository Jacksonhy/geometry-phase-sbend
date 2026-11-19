"""Matched representation experiments on real PIC-Flow data. See PROTOCOL.md."""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'picflow_ybranch_smoke_80.npz'


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def geometry_features(eps, src):
    h,w=eps.shape
    core=eps>(eps.min()+eps.max())/2
    yy,xx=np.where(src>0)
    sx,sy=float(xx.mean()),float(yy.mean())
    direction=1 if sx<w/2 else -1
    half=np.median(core[:,20:150].sum(axis=0))/2
    upper=np.empty(w); lower=np.empty(w)
    for x in range(w):
        ys=np.where(core[:,x])[0]
        if not len(ys):
            raise ValueError('No waveguide at column '+str(x))
        upper[x]=min((h-1)/2,ys.min()+half-0.5)
        lower[x]=max((h-1)/2,ys.max()-half+0.5)
    upper=gaussian_filter1d(upper,3)
    lower=gaussian_filter1d(lower,3)
    arc=[]
    for center in [upper,lower]:
        arc.append(np.r_[0,np.cumsum(np.sqrt(1+np.diff(center)**2))])
    arc=np.array(arc)
    split_columns=np.flatnonzero(lower-upper>1.5)
    j=int(split_columns[0]) if len(split_columns) else w
    gridy,gridx=np.mgrid[:h,:w]
    branch=(gridy>(upper+lower)[None,:]/2).astype(int)
    s=arc[branch,gridx]
    if direction==1:
        distance=s-np.interp(sx,np.arange(w),arc[0])
    else:
        source_branch=0 if sy<h/2 else 1
        source_s=np.interp(sx,np.arange(w),arc[source_branch])
        distance=source_s-s
        other=(branch!=source_branch)&(gridx>j)
        if j<w:
            distance[other]=source_s+s[other]-2*arc[source_branch,j]
    linear=direction*(gridx-sx)
    return dict(core=core,upper=upper,lower=lower,sx=sx,sy=sy,direction=direction,
                path=distance.astype(np.float32),linear=linear.astype(np.float32),junction=j)


def load_data(n):
    with np.load(DATA,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    train=np.where(d['split']=='train')[0][:n]
    val=np.where(d['split']=='val')[0]
    test=np.where(d['split']=='test')[0]
    assert len(set(d['geometry_id']))==80
    assert all(np.isfinite(d[k]).all() for k in ['eps','src_mask','Ez_real','Ez_imag'])
    e=d['Ez_real']+1j*d['Ez_imag']
    features=[geometry_features(a,b) for a,b in zip(d['eps'],d['src_mask'])]
    # Fit phase slope in the same straight trunk ROI for every training sample.
    slopes=[]
    for i in train:
        line=e[i,78:82,40:160].mean(axis=0)
        slopes.append(abs(np.polyfit(np.arange(120),np.unwrap(np.angle(line)),1)[0]))
    core_eps=d['eps'].max(axis=(1,2))
    coeff=np.polyfit(core_eps[train],slopes,1)
    k=np.polyval(coeff,core_eps)
    scale=float(np.sqrt(np.mean(abs(e[train])**2)))
    inputs=[]
    h,w=e.shape[-2:]
    gridy,gridx=np.mgrid[:h,:w]
    for i,f in enumerate(features):
        inputs.append(np.stack([
            (d['eps'][i]-2.0851359)/4.2,d['src_mask'][i],
            gridx/(w-1)*2-1,gridy/(h-1)*2-1,
            np.full((h,w),f['sx']/(w-1)*2-1),
            np.full((h,w),f['sy']/(h-1)*2-1),
            np.full((h,w),f['direction']),np.full((h,w),k[i]),
            f['path']/w,f['linear']/w,
        ]).astype(np.float32))
    x=F.interpolate(torch.tensor(np.stack(inputs)),scale_factor=0.5,mode='area')
    carriers={mode:np.stack([np.exp(1j*k[i]*f[mode]) for i,f in enumerate(features)]).astype(np.complex64)
              for mode in ['linear','path']}
    carriers['raw']=np.ones_like(e)
    audit=dict(data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
               shapes={key:list(d[key].shape) for key in ['eps','src_mask','Ez_real','Ez_imag']},
               splits=dict(Counter(d['split'].tolist())),
               ports={s:dict(Counter(map(str,d['input_port'][d['split']==s].tolist()))) for s in ['train','val','test']},
               train_indices=train.tolist(),val_indices=val.tolist(),test_indices=test.tolist(),
               calibration_coefficients=coeff.tolist(),train_slope_rmse=float(np.sqrt(np.mean((np.polyval(coeff,core_eps[train])-slopes)**2))),
               field_scale=scale,source_phase_train=np.angle(np.array([(e[i]*d['src_mask'][i]).sum() for i in train])).tolist())
    # Diagnose target representation at fixed output resolution without training.
    compression={}
    for mode,c in carriers.items():
        envelope=e/c
        t=torch.tensor(np.stack([envelope.real,envelope.imag],axis=1))
        recon=F.interpolate(F.interpolate(t,scale_factor=.5,mode='area'),size=(h,w),mode='bilinear',align_corners=False).numpy()
        er=(recon[:,0]+1j*recon[:,1])*c
        rel=np.sqrt((abs(er-e)**2).sum(axis=(1,2))/(abs(e)**2).sum(axis=(1,2)))
        compression[mode]={s:float(rel[inds].mean()) for s,inds in [('train',train),('val',val)]}
    audit['compression_reconstruction_rel_l2']=compression
    return d,e,x,carriers,train,val,test,scale,audit


class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU())
    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=Block(10,8); self.b=Block(8,16); self.c=Block(16,32); self.d=Block(32,64)
        self.u=Block(96,32); self.v=Block(48,16); self.w=Block(24,8)
        self.out=nn.Conv2d(8,2,1)
    def forward(self,x):
        a=self.a(x); b=self.b(F.avg_pool2d(a,2)); c=self.c(F.avg_pool2d(b,2)); d=self.d(F.avg_pool2d(c,2))
        def up(a,b): return torch.cat([F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1)
        return self.out(self.w(up(self.v(up(self.u(up(d,c)),b)),a)))


def metric_arrays(p,t):
    energy=(abs(t)**2).sum(axis=(1,2))
    phase=abs(np.angle(p*np.conj(t)))
    return dict(rel_l2=np.sqrt((abs(p-t)**2).sum(axis=(1,2))/energy),
                amplitude_rel_l2=np.sqrt(((abs(p)-abs(t))**2).sum(axis=(1,2))/energy),
                intensity_ratio=(abs(p)**2).sum(axis=(1,2))/energy,
                intensity_weighted_phase_mae=(phase*abs(t)**2).sum(axis=(1,2))/energy)


def train_one(mode,n,seed,steps,batch,lr,output_root=None):
    out=(Path(output_root) if output_root is not None else ROOT/'runs')/f'{mode}_n{n}_seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    d,e,x,carriers,tr,va,te,scale,audit=load_data(n)
    write_json(out/'audit.json',audit)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    model=UNet().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=lr*.05)
    x=x.to(device)
    target=e/carriers[mode]/scale
    y=torch.tensor(np.stack([target.real,target.imag],1),device=device)
    den=y.square().sum(dim=(1,2,3)).clamp_min(1e-12)
    gen=np.random.default_rng(seed)
    history=[]; best=math.inf; best_step=0; best_state=None
    start=time.perf_counter()
    def forward(ids):
        return F.interpolate(model(x[ids]),size=y.shape[-2:],mode='bilinear',align_corners=False)
    @torch.no_grad()
    def score(ids):
        model.eval()
        return float(torch.cat([((forward(ids[j:j+batch])-y[ids[j:j+batch]]).square().sum((1,2,3))/den[ids[j:j+batch]]).sqrt() for j in range(0,len(ids),batch)]).mean().cpu())
    for step in range(1,steps+1):
        ids=gen.choice(tr,batch,replace=False)
        model.train(); opt.zero_grad(set_to_none=True)
        pred=forward(ids)
        loss=((pred-y[ids]).square().sum((1,2,3))/den[ids]).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        opt.step(); sch.step()
        if step==1 or step%50==0 or step==steps:
            v=score(va); t=score(tr)
            row=dict(step=step,loss=float(loss.detach().cpu()),train_rel_l2=t,val_rel_l2=v,seconds=time.perf_counter()-start)
            history.append(row); write_json(out/'history.json',history)
            print(mode,n,seed,json.dumps(row),flush=True)
            if v<best:
                best=v; best_step=step; best_state=copy.deepcopy(model.state_dict())
                torch.save(dict(state=best_state,mode=mode,n=n,seed=seed,step=step,scale=scale,
                                calibration=audit['calibration_coefficients'],architecture='UNet-10-8-16-32-64'),out/'best.pt')
    summary=dict(mode=mode,n=n,seed=seed,steps=steps,batch=batch,lr=lr,best_step=best_step,best_val=best,
                 seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
                 torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                 python=platform.python_version(),data_sha256=audit['data_sha256'])
    write_json(out/'summary.json',summary)
    return summary


@torch.no_grad()
def evaluate():
    torch.set_num_threads(4)
    all_results=[]
    for run in sorted((ROOT/'runs').glob('*')):
        if not (run/'summary.json').exists(): continue
        summary=json.loads((run/'summary.json').read_text())
        n=summary['n']; mode=summary['mode']
        d,e,x,carriers,tr,va,te,scale,audit=load_data(n)
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model=UNet().cuda().eval(); model.load_state_dict(ck['state'])
        pred=[]
        for j in range(0,len(x),8):
            v=F.interpolate(model(x[j:j+8].cuda()),size=e.shape[-2:],mode='bilinear',align_corners=False).cpu().numpy()
            pred.append((v[:,0]+1j*v[:,1])*scale*carriers[mode][j:j+8])
        pred=np.concatenate(pred)
        metrics=metric_arrays(pred,e)
        result=dict(**summary,splits={})
        for name,inds in [('train',tr),('val',va),('test',te)]:
            result['splits'][name]={k:float(v[inds].mean()) for k,v in metrics.items()}
        result['test_samples']=[dict(index=int(i),geometry_id=str(d['geometry_id'][i]),port=int(d['input_port'][i]),
                                     **{k:float(v[i]) for k,v in metrics.items()}) for i in te]
        write_json(run/'evaluation.json',result)
        np.savez_compressed(run/'test_predictions.npz',prediction=pred[te],indices=te)
        all_results.append(result)
        print(run.name,result['splits'],flush=True)
    write_json(ROOT/'results.json',all_results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['raw','linear','path'],default='linear')
    p.add_argument('--n',type=int,default=32)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    if a.evaluate: evaluate()
    else: train_one(a.mode,a.n,a.seed,a.steps,a.batch,a.lr)
