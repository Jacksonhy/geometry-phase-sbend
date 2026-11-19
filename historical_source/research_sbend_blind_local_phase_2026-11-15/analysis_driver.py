"""Bounded follow-through for analysis and conditional simulations."""
import os,time,sys,subprocess,signal,traceback
from experiment import R,read,save,remaining,launch,SEEDS,METHODS,MEEP

def evaluation(action,log,limit=3600):
 assert remaining()>90
 with (R/'logs'/log).open('x') as f:
  p=subprocess.Popen([MEEP,'-B','-u',str(R/'evaluate_blind.py'),action],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,remaining()-60))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('Analysis budget timeout '+action)
 if code:raise RuntimeError('Analysis failed '+log)
 print('ANALYSIS_DONE',action,flush=True)
def awaitfile(p):
 while not p.exists():
  assert remaining()>90,'Budget expired waiting for simulations'
  if (R/'stopped.json').exists():raise RuntimeError('Simulation stopped; inspect stopped.json')
  time.sleep(15)
try:
 awaitfile(R/'prediction_freeze.json');evaluation('blind','blind_analysis.log');awaitfile(R/'simulations_complete.json');evaluation('grid','grid_analysis.log')
 cfg=read(R/'config.json');records=read(R/'geometry_manifest.json')['records'];d=read(R/'decision.json');conditional=[]
 if not d['application_readiness_gate_passed'] and remaining()>1200:
  for gid in cfg['conditional_res60_ids']:
   i=next(j for j,r in enumerate(records) if r['geometry_id']==gid);wall=launch(['fineworker',i,60],f'refine_{i}_60.log');conditional.append(dict(geometry_id=gid,resolution=60,seconds=wall))
  evaluation('refined','refined_analysis.log')
  save(R/'conditional_refinement.json',dict(executed=True,reason='frozen grid readiness gate not passed; three preselected geometries only',runs=conditional))
 else:save(R/'conditional_refinement.json',dict(executed=False,reason='grid gate passed or insufficient remaining budget'))
 if d['reverse20_eligible'] and remaining()>1200:
  rr=[]
  for gid in cfg['conditional_reverse_geometry_ids']:
   i=next(j for j,r in enumerate(records) if r['geometry_id']==gid);wall=launch(['fineworker',i,20,'reverse'],f'reverse_{i}_20.log');rr.append(dict(geometry_id=gid,seconds=wall))
  save(R/'reverse_stage.json',dict(executed=True,truth_pairs=len(rr),runs=rr,no_coherent_application=True))
 else:save(R/'reverse_stage.json',dict(executed=False,reason='pre-registered blind/grid application-readiness gate failed' if not d['reverse20_eligible'] else 'less than20min budget remaining',no_coherent_application=True))
 evaluation('verify','preservation.log',600);save(R/'analysis_complete.json',dict(unix=time.time(),elapsed_seconds=time.time()-read(R/'budget.json')['start_unix'],remaining_budget_seconds=remaining(),automatic_extension=False,report_pending=True));print('ANALYSIS_COMPLETE',flush=True)
except Exception:
 save(R/'analysis_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));traceback.print_exc();raise
