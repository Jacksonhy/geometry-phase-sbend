"""Run the predeclared suite, then evaluate all frozen checkpoints together."""
import json
import time
from pathlib import Path
from experiment import train_one, evaluate, write_json

root=Path(__file__).resolve().parent
jobs=[(mode,n,seed) for n in [32,64] for seed in [2026,2027,2028]
      for mode in (['raw','linear','path'] if n==32 else ['linear','path'])]
write_json(root/'suite_manifest.json',dict(jobs=jobs,steps=600,batch=8,lr=.002,
    test_evaluation='only after all planned training jobs complete',start_unix=time.time()))
for mode,n,seed in jobs:
    train_one(mode,n,seed,600,8,.002)
evaluate()
