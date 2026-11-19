"""Fetch a metadata-selected fresh test set from official shards, no training."""
import argparse
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--offline',action='store_true',help='Use downloaded files only; never access the network.')
args=parser.parse_args()

root=Path(__file__).resolve().parent
source=root.parent/'picflow_source/shards'
entries=json.loads((source/'index.json').read_text())
with np.load(root.parent/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z:
    used=set(z['geometry_id'].tolist())
    old_shards=set(z['source_shard'].tolist())
    old_eps={hashlib.sha256(e.tobytes()).hexdigest() for e in z['eps']}
counts=Counter(v['shard'] for v in entries if v['device']=='ybranch' and v['split']=='test'
               and v.get('augment','orig')=='orig' and v['shard'] not in old_shards)
chosen=sorted(counts,key=lambda s:(-counts[s],s))[:5]
samples=[v for v in entries if v['shard'] in chosen and v['device']=='ybranch'
         and v['split']=='test' and v.get('augment','orig')=='orig' and v['geometry_id'] not in used]
revision=(root.parent/'picflow_source/.cache/huggingface/download/shards/index.json.metadata').read_text().splitlines()[0].strip()
manifest=dict(revision=revision,shards=chosen,selection=samples,weights={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'runs').glob('*/best.pt')},created_unix=time.time())
manifest_path=root/'fresh_manifest.json'
if manifest_path.exists():
    frozen=json.loads(manifest_path.read_text(encoding='utf-8'))
    for key in ['revision','shards','selection','weights']:
        if frozen[key]!=manifest[key]:
            raise RuntimeError('Frozen manifest mismatch: '+key+'. Preserve the frozen models and selection.')
    manifest=frozen
else:
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if args.offline:
    missing=[s for s in chosen if not (source/s).is_file()]
    if missing:
        print('Missing files in '+str(source)+':',flush=True)
        for s in missing: print('  '+s,flush=True)
        print('Download all five files, then run this command again. No network access attempted.',flush=True)
        raise SystemExit(2)
arrays={k:[] for k in ['eps','src_mask','Ez_real','Ez_imag']}
meta={k:[] for k in ['geometry_id','input_port','source_shard','source_slot']}
for shard in chosen:
    dst=source/shard
    if not dst.exists():
        url=f'https://huggingface.co/datasets/RizzoLab/PIC-Flow-Dataset/resolve/{revision}/shards/{shard}'
        print('Downloading',shard,flush=True)
        req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'PICFlow-local-research'})
        temp=dst.with_suffix('.npz.part')
        with urllib.request.urlopen(req,timeout=60) as response, temp.open('wb') as f:
            while block:=response.read(1024*1024): f.write(block)
        temp.replace(dst)
    manifest.setdefault('shard_sha256',{})[shard]=hashlib.sha256(dst.read_bytes()).hexdigest()
    with np.load(dst,allow_pickle=False) as z:
        for v in samples:
            if v['shard']!=shard: continue
            prefix=f"s{int(v['slot'])}/"
            assert str(z[prefix+'geometry_id'].item())==v['geometry_id'], 'Wrong file or dataset revision: '+shard
            assert str(z[prefix+'device'].item())=='ybranch'
            assert str(z[prefix+'split'].item())=='test'
            assert int(z[prefix+'input_port'].item()) in [1,2,3]
            for k in arrays:
                a=np.asarray(z[prefix+k],dtype=np.float32)
                assert a.shape==(160,480) and np.isfinite(a).all()
                arrays[k].append(a)
            meta['geometry_id'].append(v['geometry_id'])
            meta['input_port'].append(int(z[prefix+'input_port'].item()))
            meta['source_shard'].append(shard); meta['source_slot'].append(int(v['slot']))
    print('Read',shard,'collected',len(meta['geometry_id']),flush=True)
hashes=[hashlib.sha256(e.tobytes()).hexdigest() for e in arrays['eps']]
assert len(set(meta['geometry_id']))==len(samples)
overlap=[i for i,h in enumerate(hashes) if h in old_eps]
manifest['eps_overlap_with_old']=overlap
manifest['eps_unique_count']=len(set(hashes))
manifest['ports']=dict(Counter(map(str,meta['input_port'])))
(root/'fresh_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
np.savez_compressed(root/'fresh_test.npz',**{k:np.stack(v) for k,v in arrays.items()},**{k:np.array(v) for k,v in meta.items()})
print('Fresh set saved',len(samples),'ports',manifest['ports'],'eps overlap',overlap,flush=True)
print('SUCCESS: '+str(root/'fresh_test.npz'),flush=True)
