import json,hashlib,datetime
from pathlib import Path
ROOT=Path('/mnt/d/game');OUT=ROOT/'research_sbend_path';OLD=ROOT/'research_meep_validation'
SRC=next(p.parent for p in (ROOT/'PIC-Flow-新研究资料').rglob('README.md') if (p.parent/'FDTD/unified_sweep.py').exists())
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def historic():
    paths=[]
    for folder in ROOT.glob('research_*'):
        if folder==OUT:continue
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend(p for p in (ROOT/'PIC-Flow-新研究资料').rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend([ROOT/'研究交接_给下一位Agent.md',ROOT/'picflow_source/shards/index.json',ROOT/'picflow_ybranch_smoke_80.npz'])
    return {str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))}
if __name__=='__main__':
    assert not (OUT/'historical_before.json').exists()
    save('historical_before.json',historic())
    save('shards_before.json',{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'picflow_source/shards').glob('*.npz')})
    save('started.json',{'time_utc':now(),'seed':2026,'scope':'16 prior development fields; no FDTD, official geometry generation, training, inference or new test'})
    print('Historical protection snapshot complete.')
