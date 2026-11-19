"""Complete report after finite analysis driver finishes, within same deadline."""
import time,subprocess,sys,traceback
from experiment import R,MEEP,remaining,save
try:
 while not (R/'analysis_complete.json').exists():
  if (R/'analysis_stopped.json').exists():raise RuntimeError('Analysis stopped; report not generated')
  assert remaining()>90,'Hard budget reached before report';time.sleep(15)
 for name in ['verify_normalization.py','make_report.py']:
  assert remaining()>60
  with (R/'logs'/(name+'.log')).open('x') as f:
   p=subprocess.run([MEEP,'-B','-u',str(R/name)],stdout=f,stderr=subprocess.STDOUT,timeout=min(120,remaining()-30))
  if p.returncode:raise RuntimeError('Failed '+name)
  print('DONE',name,flush=True)
except Exception:
 save(R/'report_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
