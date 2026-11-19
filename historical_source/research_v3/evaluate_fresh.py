"""Evaluate pre-frozen checkpoints; inference inputs never include field labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import ROOT, UNet, geometry_features, metric_arrays, write_json


def inference_inputs(eps, src, calibration, mode):
    """Only geometry/source and train-fitted constants are accepted."""
    h,w=eps.shape[-2:]
    gy,gx=np.mgrid[:h,:w]
    k=np.polyval(calibration,eps.max(axis=(1,2)))
    inputs=[]; carriers=[]; features=[]
    for i,(a,b) in enumerate(zip(eps,src)):
        f=geometry_features(a,b)
        features.append(f)
        inputs.append(np.stack([
            (a-2.0851359)/4.2,b,gx/(w-1)*2-1,gy/(h-1)*2-1,
            np.full((h,w),f['sx']/(w-1)*2-1),np.full((h,w),f['sy']/(h-1)*2-1),
            np.full((h,w),f['direction']),np.full((h,w),k[i]),f['path']/w,f['linear']/w,
        ]).astype(np.float32))
        carriers.append(np.ones((h,w),dtype=np.complex64) if mode=='raw'
                        else np.exp(1j*k[i]*f[mode]).astype(np.complex64))
    return F.interpolate(torch.tensor(np.stack(inputs)),scale_factor=.5,mode='area'),np.stack(carriers),features


@torch.no_grad()
def predict(checkpoint, eps, src):
    x,c,features=inference_inputs(eps,src,checkpoint['calibration'],checkpoint['mode'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=UNet().to(device).eval()
    model.load_state_dict(checkpoint['state'])
    chunks=[]
    for start in range(0,len(x),8):
        y=F.interpolate(model(x[start:start+8].to(device)),size=eps.shape[-2:],mode='bilinear',align_corners=False).cpu().numpy()
        chunks.append((y[:,0]+1j*y[:,1])*checkpoint['scale']*c[start:start+8])
    return np.concatenate(chunks),features


def regions(features):
    masks={k:[] for k in ['trunk','split_and_branches','source_side','far_side','core','interior']}
    for f in features:
        h,w=f['core'].shape
        gy,gx=np.mgrid[:h,:w]
        masks['trunk'].append(gx<f['junction'])
        masks['split_and_branches'].append(gx>=f['junction'])
        masks['source_side'].append(gx<w/2 if f['direction']==1 else gx>=w/2)
        masks['far_side'].append(gx>=w/2 if f['direction']==1 else gx<w/2)
        masks['core'].append(f['core'])
        masks['interior'].append((gx>=10)&(gx<w-10)&(gy>=10)&(gy<h-10))
    return {k:np.stack(v) for k,v in masks.items()}


def summarize(pred,target,ports,features):
    metrics=metric_arrays(pred,target)
    overall={k:float(v.mean()) for k,v in metrics.items()}
    by_port={str(p):{k:float(v[ports==p].mean()) for k,v in metrics.items()} for p in np.unique(ports)}
    macro={k:float(np.mean([v[k] for v in by_port.values()])) for k in metrics}
    region={}
    for name,m in regions(features).items():
        den=(abs(target)**2*m).sum(axis=(1,2))
        rel=np.sqrt((abs(pred-target)**2*m).sum(axis=(1,2))/np.maximum(den,1e-12))
        region[name]=dict(mean_rel_l2=float(rel.mean()),per_sample=rel.tolist(),
                          mean_truth_energy_fraction=float(np.mean(den/(abs(target)**2).sum(axis=(1,2)))))
    return metrics,dict(overall=overall,by_port=by_port,port_macro=macro,regions=region)


def compare_groups(results,boot=10000):
    groups={}
    for result in results:
        key=f"{result['mode']}_n{result['n']}"
        groups.setdefault(key,[]).append(result)
    aggregate={}
    for key,rs in groups.items():
        aggregate[key]=dict(seeds=[r['seed'] for r in rs],metrics={})
        for m in rs[0]['overall']:
            vals=np.array([r['overall'][m] for r in rs])
            aggregate[key]['metrics'][m]=dict(mean=float(vals.mean()),seed_sd=float(vals.std(ddof=1)))
        aggregate[key]['port_rel_l2']={p:float(np.mean([r['by_port'][p]['rel_l2'] for r in rs])) for p in rs[0]['by_port']}
        aggregate[key]['port_macro_rel_l2']=float(np.mean([r['port_macro']['rel_l2'] for r in rs]))
        aggregate[key]['regions']={p:float(np.mean([r['regions'][p]['mean_rel_l2'] for r in rs])) for p in rs[0]['regions']}
    comparisons={}
    for n in [32,64]:
        a=sorted(groups[f'linear_n{n}'],key=lambda r:r['seed'])
        b=sorted(groups[f'path_n{n}'],key=lambda r:r['seed'])
        # Average paired differences across seeds before bootstrapping geometries.
        diff=np.array([[u['rel_l2']-v['rel_l2'] for u,v in zip(ra['samples'],rb['samples'])] for ra,rb in zip(a,b)])
        mean_per_geometry=diff.mean(axis=0)
        rng=np.random.default_rng(74219+n)
        sampled=mean_per_geometry[rng.integers(0,len(mean_per_geometry),(boot,len(mean_per_geometry)))].mean(axis=1)
        comparisons[str(n)]=dict(linear_minus_path_mean=float(mean_per_geometry.mean()),
            relative_error_reduction=float(mean_per_geometry.mean()/aggregate[f'linear_n{n}']['metrics']['rel_l2']['mean']),
            bootstrap_geometry_ci95=np.quantile(sampled,[.025,.975]).tolist(),
            positive_geometries=int((mean_per_geometry>0).sum()),negative_geometries=int((mean_per_geometry<0).sum()),
            seed_mean_improvements=diff.mean(axis=1).tolist(),per_geometry_difference=mean_per_geometry.tolist(),
            interpretation='Conditional resampling interval for the selected 37 geometries, averaged over three seeds; not population or selection-adjusted inference.')
    return dict(groups=aggregate,comparisons=comparisons)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    manifest=json.loads((ROOT/'fresh_manifest.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'fresh_test.npz',allow_pickle=False) as z: fresh={k:z[k] for k in z.files}
    with np.load(ROOT.parent/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z: old={k:z[k] for k in z.files}
    assert len(fresh['eps'])==37 and len(set(fresh['geometry_id']))==37
    assert set(fresh['geometry_id']).isdisjoint(old['geometry_id'])
    assert len({hashlib.sha256(v.tobytes()).hexdigest() for v in fresh['eps']})==37
    expected={v['geometry_id'] for v in manifest['selection']}
    assert set(fresh['geometry_id'])==expected
    old_hashes={hashlib.sha256(v.tobytes()).hexdigest() for v in old['eps']}
    assert all(hashlib.sha256(v.tobytes()).hexdigest() not in old_hashes for v in fresh['eps'])
    out=ROOT/'fresh_evaluation'; out.mkdir(exist_ok=True)
    results=[]; replay=[]
    target=fresh['Ez_real']+1j*fresh['Ez_imag']
    for name,sha in manifest['weights'].items():
        run=ROOT/'runs'/name
        assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==sha, 'Frozen checkpoint changed: '+name
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        # Replay old predictions to verify the standalone inference path exactly matches training preprocessing.
        ids=np.where(old['split']=='test')[0]
        old_pred,_=predict(ck,old['eps'][ids],old['src_mask'][ids])
        with np.load(run/'test_predictions.npz',allow_pickle=False) as z: saved=z['prediction']
        replay_rel=float(np.linalg.norm(old_pred-saved)/max(np.linalg.norm(saved),1e-12))
        assert replay_rel<1e-5, (name,replay_rel)
        replay.append(dict(run=name,prediction_replay_relative_difference=replay_rel,sha256=sha))
        pred,features=predict(ck,fresh['eps'],fresh['src_mask'])
        assert np.isfinite(pred).all()
        metrics,summary=summarize(pred,target,fresh['input_port'],features)
        result=dict(run=name,mode=ck['mode'],n=ck['n'],seed=ck['seed'],best_step=ck['step'],**summary)
        result['samples']=[dict(index=i,geometry_id=str(fresh['geometry_id'][i]),port=int(fresh['input_port'][i]),
                               **{k:float(v[i]) for k,v in metrics.items()}) for i in range(len(pred))]
        write_json(out/(name+'.json'),result)
        np.savez_compressed(out/(name+'_predictions.npz'),prediction=pred,geometry_id=fresh['geometry_id'])
        results.append(result)
        print(name,'RelL2',result['overall']['rel_l2'],'macro',result['port_macro']['rel_l2'],flush=True)
    write_json(out/'results.json',results)
    write_json(out/'inference_verification.json',dict(replay=replay,fresh_sha256=hashlib.sha256((ROOT/'fresh_test.npz').read_bytes()).hexdigest(),
               no_label_inputs=True,all_frozen_weights_verified=True))
    aggregate=compare_groups(results)
    write_json(out/'summary.json',aggregate)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__': main()
