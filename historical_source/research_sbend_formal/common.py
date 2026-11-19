"""Formal S-bend experiment: independent outputs, read-only historical inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import json,hashlib,datetime,sys
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/d/game');OUT=Path(__file__).resolve().parent
V1=ROOT/'research_sbend_path';V2=ROOT/'research_sbend_path_v2';DEV=ROOT/'research_meep_validation'
SRC=next(p.parent for p in (ROOT/'PIC-Flow-新研究资料').rglob('README.md') if (p.parent/'FDTD/unified_sweep.py').exists())
for p in [V1,V2]:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import path_geometry as v1
import path_v2 as v2
METHODS=['direct','linear','local_tangent','path_v1','path_v2']
SEED=2026
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,obj):
    p=Path(p);p=p if p.is_absolute() else OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def params_key(p):return tuple(int(round(p[k]/.025)) for k in ['wg_width_um','lateral_offset_um','R_min_um'])
def curve_fingerprint(poly):
    p=np.asarray(poly,dtype=float)
    # Translation and axis reflections / 180-degree input swaps are equivalent.
    values=[]
    for sx in [-1,1]:
        for sy in [-1,1]:
            q=p*np.array([sx,sy]);q=q[np.argsort(q[:,0])];q-=q[0]
            values.append(ahash(np.round(q,9)))
    return min(values)
def eps_fingerprints(ep):
    ep=np.asarray(ep,dtype=np.float32)
    return {ahash(a) for a in [ep,ep[::-1],ep[:,::-1],ep[::-1,::-1]]}
def verify_freeze():
    f=read(OUT/'code_freeze.json')
    for name,digest in f['sources'].items():assert sha(ROOT/name)==digest,'Frozen source changed: '+name
    return f
