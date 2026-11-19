import time,sys,subprocess,signal,traceback,os
import experiment as e
R=e.R
def cmd(name,args,limit=1200):
 assert e.remaining()>120
 log=R/'logs'/(name+str(int(time.time()))+'.log')
 with log.open('x') as f:
  p=subprocess.Popen([e.PY,'-B','-u',str(R/name),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(limit,e.remaining()-90))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('hard deadline '+name)
 if code:raise RuntimeError('failed '+str(log))
 print('FINISHED',name,args,flush=True)
try:
 f=R/'failures';f.mkdir(exist_ok=True)
 for n in ['stopped.json','analysis_stopped.json','report_stopped.json']:
  p=R/n
  if p.exists():p.rename(f/(str(int(time.time()))+'_'+n))
 repair=e.read(R/'IMPLEMENTATION_REPAIR.json');repair['PML_shape_check_repair']='Interior DFT includes up to2 boundary pixels; use actual metadata coordinates. No physical or budget change.';repair['corrected_sha256']=e.sha(R/'experiment.py');e.save(R/'IMPLEMENTATION_REPAIR.json',repair)
 jobs=e.read(R/'jobs.json')
 for i,j in enumerate(jobs):
  d=R/'runs'/j['id']
  if (d/'result.json').exists():continue
  if d.exists():d.rename(f/(str(int(time.time()))+'_'+j['id']))
  e.launch('worker',i,j['id']);e.save(R/'progress.json',dict(stage='FDTD_resumed',completed=i+1,total=len(jobs),last=j['id'],remaining_budget_seconds=e.remaining()));print('RESUMED',i+1,len(jobs),flush=True)
 c=e.read(R/'config.json');mj=[]
 for r in c['diagnostic_records']:
  for res in c['resolutions']:mj.append(dict(id=f'original_{r["geometry_id"]}_r{res}',kind='sbend',geometry_id=r['geometry_id'],resolution=res))
 for gid in c['control_ids']:
  for res in [20,40]:mj +=[dict(id=f'noavg_{gid}_r{res}',kind='sbend',geometry_id=gid,resolution=res,eps_averaging=False),dict(id=f'reordered_{gid}_r{res}',kind='sbend',geometry_id=gid,resolution=res,reverse_order=True)]
 e.save(R/'material_jobs.json',mj)
 for i,j in enumerate(mj):
  if (R/'materials'/j['id']/'result.json').exists():continue
  e.launch('material',i,'material_'+j['id']);e.save(R/'progress.json',dict(stage='material_initialization',completed=i+1,total=len(mj),remaining_budget_seconds=e.remaining()))
  if (i+1)%6==0:print('MATERIALS',i+1,len(mj),flush=True)
 e.save(R/'primary_complete.json',dict(unix=time.time(),fdtd_jobs=len(jobs),material_jobs=len(mj),remaining_budget_seconds=e.remaining()))
 cmd('analyze.py',['all'],1500);rr=e.read(R/'decomposition.json');controls=[r for r in rr if r['geometry_id'] in c['conditional80_ids'] and r['low_resolution']==40];trigger=any(r['bend_phase_change_wmae_deg']>.5 or r['normal_phase_change_wmae_deg']>.25 for r in controls);execute=trigger and e.remaining()>=2400
 e.save(R/'conditional80_decision.json',dict(executed=execute,trigger_controls=controls,remaining_budget_at_decision=e.remaining(),reason='threshold not exceeded or less than40min remain' if not execute else 'frozen threshold exceeded'))
 if execute:
  for gid in c['conditional80_ids']:cmd('conditional80.py',[gid],1800)
 cmd('analyze.py',['final'],1200);e.save(R/'analysis_complete.json',dict(unix=time.time(),elapsed_seconds=time.time()-e.read(R/'budget.json')['start_unix'],deadline_respected=e.remaining()>0,no_extension=True))
 for n in ['attribution.py','verify_beta.py','make_report.py']:cmd(n,[],300)
 e.save(R/'resumed_complete.json',dict(unix=time.time(),deadline_respected=e.remaining()>0,elapsed_seconds=time.time()-e.read(R/'budget.json')['start_unix'],no_extension=True));print('ALL_COMPLETE',flush=True)
except Exception:
 e.save(R/'resume_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));traceback.print_exc();raise

