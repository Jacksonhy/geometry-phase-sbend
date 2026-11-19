import json,csv,hashlib,datetime
from pathlib import Path
ROOT=Path('/mnt/d/game');OUT=ROOT/'research_sbend_path_v2';V1=ROOT/'research_sbend_path';OLD=ROOT/'research_meep_validation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protected_paths():
    paths=[]
    for d in ROOT.glob('research_*'):
        if d==OUT:continue
        paths.extend(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend(p for p in (ROOT/'PIC-Flow-新研究资料').rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend([ROOT/'研究交接_给下一位Agent.md',ROOT/'picflow_source/shards/index.json',ROOT/'picflow_ybranch_smoke_80.npz'])
    return sorted(set(paths))
if __name__=='__main__':
    assert not (OUT/'historical_before.json').exists()
    save('candidate_freeze.json',{'time_utc':now(),'seed':2026,'candidate_count':1,
        'sources':{n:sha(OUT/n) for n in ['PROTOCOL.md','path_v2.py']},
        'sigma_um':.05,'quadrature_max_step_um':.0125,'r0':'width/2+0.05um','transition_width_um':.5,'bend_end_gate_um':.05,
        'fdtd_runs':0,'formal_geometries':0,'network_runs':0,'new_test_evaluations':0})
    save('historical_before.json',{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in protected_paths()})
    save('shards_before.json',{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'picflow_source/shards').glob('*.npz')})
    print('Candidate fixed and historical snapshot saved.',flush=True)
