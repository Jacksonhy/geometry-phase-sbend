"""Supplemental budget control; report separately from the 15 frozen models."""
import hashlib
import json
import numpy as np
import torch
from experiment import ROOT, write_json
from evaluate_fresh import predict, summarize

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
out=ROOT/'exposure_matched'
frozen=json.loads((out/'frozen_weights.json').read_text())
assert len(frozen)==6
with np.load(ROOT/'fresh_test.npz',allow_pickle=False) as z: fresh={k:z[k] for k in z.files}
with np.load(ROOT.parent/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z: old={k:z[k] for k in z.files}
fresh_truth=fresh['Ez_real']+1j*fresh['Ez_imag']
results=[]
for name,sha in frozen.items():
    run=out/'runs'/name
    assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==sha
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    pred,features=predict(ck,fresh['eps'],fresh['src_mask'])
    metrics,summary=summarize(pred,fresh_truth,fresh['input_port'],features)
    result=dict(run=name,mode=ck['mode'],n=ck['n'],seed=ck['seed'],steps=1200,best_step=ck['step'],**summary)
    result['samples']=[dict(index=i,geometry_id=str(fresh['geometry_id'][i]),port=int(fresh['input_port'][i]),
                           **{k:float(v[i]) for k,v in metrics.items()}) for i in range(len(pred))]
    np.savez_compressed(run/'fresh_predictions.npz',prediction=pred,geometry_id=fresh['geometry_id'])
    ids=np.r_[np.where(old['split']=='train')[0],np.where(old['split']=='val')[0],np.where(old['split']=='test')[0]]
    old_pred,_=predict(ck,old['eps'][ids],old['src_mask'][ids])
    old_target=old['Ez_real'][ids]+1j*old['Ez_imag'][ids]
    from experiment import metric_arrays
    m=metric_arrays(old_pred,old_target)
    result['old_splits']={split:{k:float(v[old['split'][ids]==split].mean()) for k,v in m.items()} for split in ['train','val','test']}
    write_json(run/'evaluation.json',result)
    results.append(result)
    print(name,'fresh',result['overall']['rel_l2'],'old',result['old_splits']['test']['rel_l2'],flush=True)
summary={}
for mode in ['linear','path']:
    rs=[r for r in results if r['mode']==mode]
    summary[mode]=dict(metrics={k:dict(mean=float(np.mean([r['overall'][k] for r in rs])),
                                     seed_sd=float(np.std([r['overall'][k] for r in rs],ddof=1))) for k in rs[0]['overall']},
                       port_macro_rel_l2=float(np.mean([r['port_macro']['rel_l2'] for r in rs])),
                       ports={p:float(np.mean([r['by_port'][p]['rel_l2'] for r in rs])) for p in rs[0]['by_port']},
                       old_splits={s:float(np.mean([r['old_splits'][s]['rel_l2'] for r in rs])) for s in ['train','val','test']})
write_json(out/'results.json',results)
write_json(out/'summary.json',summary)
print(json.dumps(summary,indent=2),flush=True)
