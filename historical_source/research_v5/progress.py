"""Read local logs without touching training or using the GPU."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
completed=[];active=[]
for p in sorted((root/'runs').glob('*')):
    if not p.is_dir():continue
    s=p/'summary.json'
    if s.exists():
        r=json.loads(s.read_text(encoding='utf-8'))
        completed.append((p.name,r['best_val'],r['best_step'],r['train_wall_seconds']))
    elif (p/'history.json').exists():
        rows=json.loads((p/'history.json').read_text(encoding='utf-8'))
        active.append((p.name,rows[-1]['step'],min(r['val_rel_l2'] for r in rows)))
print(f'Completed: {len(completed)}/10')
for name,best,step,seconds in completed:print(f'  {name}: best validation={best:.6f}, step={step}, minutes={seconds/60:.2f}')
for name,step,best in active:print(f'Active: {name}, last logged update={step}/1200, best validation={best:.6f}')
if (root/'summary.json').exists():print((root/'summary.json').read_text(encoding='utf-8'))
