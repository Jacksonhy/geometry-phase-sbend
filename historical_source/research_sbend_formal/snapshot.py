from common import ROOT,OUT,write,sha,now
def main():
    assert not (OUT/'historical_before.json').exists()
    paths=[]
    for folder in list(ROOT.glob('research_*'))+[ROOT/'references',ROOT/'PIC-Flow-新研究资料']:
        if folder==OUT:continue
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.extend([ROOT/'研究交接_给下一位Agent.md',ROOT/'picflow_source/shards/index.json',ROOT/'picflow_ybranch_smoke_80.npz'])
    write('historical_before.json',{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))})
    write('shards_before.json',{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'picflow_source/shards').glob('*.npz')})
    print('Historical snapshot completed '+now(),flush=True)
if __name__=='__main__':main()
