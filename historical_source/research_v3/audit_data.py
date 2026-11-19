"""Inspect the provided real data before defining phase representations."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
z = np.load(ROOT.parent / 'picflow_ybranch_smoke_80.npz', allow_pickle=False)
print('keys', z.files)
print('splits', Counter(z['split'].tolist()), 'ports', Counter(z['input_port'].tolist()))
for k in ['eps', 'src_mask', 'Ez_real', 'Ez_imag']:
    a=z[k]
    print(k, a.shape, a.dtype, float(a.min()), float(a.max()), 'finite', bool(np.isfinite(a).all()))
for i in [0,1,2,3,4,32,63,64,65,72]:
    eps=z['eps'][i]; src=z['src_mask'][i]
    e=z['Ez_real'][i]+1j*z['Ez_imag'][i]
    yy,xx=np.where(src>0)
    print('sample',i,str(z['geometry_id'][i]),str(z['split'][i]),'port',int(z['input_port'][i]),
          'eps min/max', [float(eps.min()),float(eps.max())],'src extent', [int(xx.min()),int(xx.max()),int(yy.min()),int(yy.max())],
          'rms',float(np.sqrt(np.mean(abs(e)**2))))
    core=eps>(eps.min()+eps.max())/2
    for x in [30,50,100,180,240,300,400,440]:
        y=np.where(core[:,x])[0]
        cuts=np.split(y,np.where(np.diff(y)>1)[0]+1)
        print(' x',x,[(float(c.mean()),len(c)) for c in cuts if len(c)])
    y=int(np.median(np.where(core[:,50])[0]))
    for x1,x2 in [(35,100),(100,160),(320,400)]:
        sl=e[y,x1:x2]; phase=np.unwrap(np.angle(sl))
        print(' slope',y,x1,x2,float(np.polyfit(np.arange(len(sl)),phase,1)[0]))
with np.load(ROOT.parent/'picflow_source/shards'/str(z['source_shard'][0]),allow_pickle=False) as sh:
    pref=f"s{int(z['source_slot'][0])}/"
    print('raw keys', [k for k in sh.files if k.startswith(pref)])
    for k in sh.files:
        if k.startswith(pref) and sh[k].size<30:
            print(k,sh[k])
