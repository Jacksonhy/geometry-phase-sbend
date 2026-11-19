"""Follow finite simulations through diagnostics and conditional res80."""
import time,sys,subprocess,signal,traceback
from experiment import R,PY,remaining,read,save

def awaitfile(p):
 while not p.exists():
  if (R/'stopped.json').exists():raise RuntimeError('FDTD pipeline stopped')
  assert remaining()>120,'Deadline reached waiting';time.sleep(20)
def command(script,args,log,limit):
 assert remaining()>120
 with (R/'logs'/log).open('x') as f:
  p=subprocess.Popen([PY,'-B','-u',str(R/script),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,remaining()-90))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('Deadline/task timeout '+log)
 if code:raise RuntimeError('Failed '+log)
 print('FOLLOW_DONE',log,flush=True)
try:
 awaitfile(R/'runs/centered_w0.575_r60/result.json');command('analyze.py',['beta'],'beta_initial.log',600);awaitfile(R/'primary_complete.json');command('analyze.py',['all'],'analysis_primary.log',1800)
 cfg=read(R/'config.json');rr=read(R/'decomposition.json');controls=[r for r in rr if r['geometry_id'] in cfg['conditional80_ids'] and r['low_resolution']==40];assert len(controls)==3;trigger=any(r['bend_phase_change_wmae_deg']>.5 or r['normal_phase_change_wmae_deg']>.25 for r in controls)
 if trigger and remaining()>=2400:
  save(R/'conditional80_decision.json',dict(executed=True,trigger_controls=controls,preselected_ids=cfg['conditional80_ids'],remaining_budget_at_decision=remaining()))
  for gid in cfg['conditional80_ids']:command('conditional80.py',[gid],f'res80_{gid}.log',1800)
 else:save(R/'conditional80_decision.json',dict(executed=False,trigger_controls=controls,reason='thresholds not exceeded or less than40min remain'))
 command('analyze.py',['final'],'analysis_final.log',1800);save(R/'analysis_complete.json',dict(unix=time.time(),elapsed_seconds=time.time()-read(R/'budget.json')['start_unix'],deadline_respected=remaining()>0,no_extension=True,report_pending=True));print('ANALYSIS_COMPLETE',flush=True)
except Exception:
 save(R/'analysis_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));traceback.print_exc();raise
