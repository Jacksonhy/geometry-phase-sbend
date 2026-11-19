"""Export-only repair: heterogeneous field and normal-pair summary columns."""
import csv,json,time,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent;p=R/'analyze.py';s=importlib.util.spec_from_file_location('local_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def load(name):
 with (R/name).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k,v in list(r.items()):
   if k in ['method','geometry_id','region']:continue
   r[k]=int(v) if k in ['seed','input_port','pixels','phase_pixels','pairs'] else float(v)
 return rows
def csvunion(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
partial=R/'summary_per_seed.csv'
if partial.exists():partial.rename(R/'summary_per_seed_partial_failed.csv')
a.csvsave=csvunion
a.save(R/'SUMMARY_EXPORT_REPAIR.json',dict(time_unix=time.time(),error='heterogeneous field and normal-pair CSV columns',repair='union of columns, no metric or statistical definition changes',original_script_unchanged=True,source_tables=['per_geometry_fields.csv','per_geometry_normal_pairs.csv'],script_sha256=a.sha(Path(__file__))))
a.summarize(load('per_geometry_fields.csv'),load('per_geometry_normal_pairs.csv'));a.verify();a.save(R/'complete.json',dict(completed=True,wall_seconds=time.time()-a.read(R/'freeze.json')['time_unix'],summary_export_repair='SUMMARY_EXPORT_REPAIR.json'));print('LOCAL_PHASE_COMPLETE',flush=True)
