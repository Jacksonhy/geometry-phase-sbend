"""Inspect unused local samples by metadata only; no field-label evaluation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
entries=json.loads((root.parent/'picflow_source/shards/index.json').read_text())
with np.load(root.parent/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z:
    used=set(z['geometry_id'].tolist())
available={p.name for p in (root.parent/'picflow_source/shards').glob('*.npz')}
pool=[v for v in entries if v['shard'] in available and v['device']=='ybranch' and v['geometry_id'] not in used]
print('Unused local Y-branch records',len(pool))
print('By split',Counter(v['split'] for v in pool))
for v in pool:
    if v['split']=='test': print(v)
(root/'unused_local_pool.json').write_text(json.dumps(pool,indent=2),encoding='utf-8')
