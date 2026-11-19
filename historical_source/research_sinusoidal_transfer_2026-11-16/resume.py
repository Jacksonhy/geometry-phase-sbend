"""Documented startup compatibility fix; original freeze remains unchanged."""
import inspect,sys,os,time,traceback
import numpy as np
import experiment as E
from experiment import R,read,save,sha,remaining

def field_free_guard():
 opened=[]
 def audit(event,args):
  if event!='open':return
  try:p=os.fspath(args[0])
  except TypeError:return
  if not isinstance(p,str) or '/mnt/d/game/' not in p.replace('\\','/'):return
  q=p.replace('\\','/');mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  if q.endswith('/sample.npz') or '/data/' in q or '/cache/' in q or ('/predictions/' in q and not (writing and q.startswith(str(R/'predictions')+'/'))):raise RuntimeError('Field-free access guard blocked '+q)
  opened.append(('WRITE ' if writing else 'READ ')+q)
 sys.addaudithook(audit);return opened

def prepare():
 # Already prepared inputs are preserved. Exact eps/grid/source agreement is
 # independently checked against the later FDTD for every geometry.
 source=inspect.getsource(E.prepare).replace("t=time.perf_counter();dev,fw=device(r,20);mp=fw.mp;mp.verbosity(0)","t=time.perf_counter()\n  dest=R/'inputs'/(r['geometry_id']+'.npz')\n  if dest.exists():\n   with np.load(dest) as z:assert z['x'].shape==(7,160,480) and np.isfinite(z['x']).all()\n   cost.append(dict(geometry_id=r['geometry_id'],seconds=None,reused_prepared_input=True));continue\n  dev,fw=device(r,20);mp=fw.mp;mp.verbosity(0)")
 exec(compile(source,str(R/'resume.py')+':prepare','exec'),E.__dict__);E.prepare()

def infer():
 d=E.deployment();d.field_free_guard=field_free_guard;E.deployment=lambda:d;E.infer()

def run():
 assert not (R/'prediction_freeze.json').exists()
 for p,h in read(R/'freeze.json')['hashes'].items():assert sha(p)==h,p
 save(R/f'startup_amendment_{int(time.time())}.json',dict(unix=time.time(),before_new_predictions=True,before_new_FDTD=True,reason='Windows patch command used default GBK decoding and failed; launch proceeded before intended output-path fix. Preparation interrupted at26 inputs; resumed without deleting outputs.',changes='resume remaining field-free inputs; permit write-only opens for this experiment prediction outputs while blocking all truth and historical prediction reads; no numerical algorithm or selection changes',runner_sha256=sha(R/'resume.py'),deadline_unchanged=read(R/'budget.json')['deadline_unix']))
 for path in (R/'inputs').glob('*.npz'):
  try:
   with np.load(path) as z:assert z['x'].shape==(7,160,480) and np.isfinite(z['x']).all()
  except (EOFError,ValueError,OSError,AssertionError):
   dest=path.with_suffix('.interrupted');assert not dest.exists();path.rename(dest)
 stamp=str(int(time.time()));E.launch('resume.py',['prepare'],'prepare_resume_'+stamp+'.log',limit=2400);E.launch('resume.py',['infer'],'inference.log',E.GPU,2400);rr=read(R/'geometry_manifest.json')['records'];cost=[]
 for i,r in enumerate(rr):
  sec=E.launch('experiment.py',['simworker',i,20],f'base_{i:02d}.log');cost.append(dict(geometry_id=r['geometry_id'],resolution=20,seconds=sec));save(R/'progress.json',dict(stage='base_FDTD',completed=i+1,total=36,remaining_seconds=remaining()));print('BASE',i+1,36,flush=True)
 E.launch('analyze.py',['blind'],'analysis_base.log',limit=1800)
 for gid in read(R/'config.json')['fine_geometry_ids']:
  i=next(i for i,r in enumerate(rr) if r['geometry_id']==gid)
  for res in [30,40]:
   sec=E.launch('experiment.py',['simworker',i,res],f'fine_{i:02d}_{res}.log');cost.append(dict(geometry_id=gid,resolution=res,seconds=sec));save(R/'progress.json',dict(stage='fine_FDTD',completed=len(cost)-36,total=12,remaining_seconds=remaining()));print('GRID',len(cost)-36,12,flush=True)
 save(R/'simulation_costs.json',cost);E.launch('analyze.py',['grid'],'analysis_grid.log',limit=1800);E.launch('analyze.py',['report'],'report.log',limit=1800);save(R/'complete.json',dict(unix=time.time(),elapsed_seconds=14400-remaining(),remaining_seconds=remaining(),base=36,fine=12,training_runs=0));print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='run':run()
  elif sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='infer':infer()
 except Exception:
  traceback.print_exc()
  if sys.argv[1]=='run':save(R/'stopped.json',dict(unix=time.time(),reason=traceback.format_exc(),automatic_extension=False))
  raise
