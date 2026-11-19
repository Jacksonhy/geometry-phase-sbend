"""Predeclared doubled-update control. Original frozen models remain untouched."""
import hashlib
import time
from experiment import ROOT, train_one, write_json

out=ROOT/'exposure_matched'
out.mkdir(exist_ok=True)
jobs=[(mode,64,seed) for seed in [2026,2027,2028] for mode in ['linear','path']]
write_json(out/'manifest.json',dict(jobs=jobs,steps=1200,batch=8,lr=.002,start_unix=time.time(),
                                   expected_exposures_per_sample=150,checkpoint_selection='original validation set only'))
for mode,n,seed in jobs:
    train_one(mode,n,seed,1200,8,.002,output_root=out/'runs')
write_json(out/'frozen_weights.json',{p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'runs').glob('*/best.pt')})
print('All six supplemental models frozen.',flush=True)
