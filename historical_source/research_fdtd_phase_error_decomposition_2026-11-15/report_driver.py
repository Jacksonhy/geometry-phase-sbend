"""Finish bounded attribution and report after analysis, same four-hour deadline."""
import time,subprocess,traceback
from experiment import R,PY,remaining,save
try:
 while not (R/'analysis_complete.json').exists():
  if (R/'analysis_stopped.json').exists() or (R/'stopped.json').exists():raise RuntimeError('Pipeline stopped; inspect saved diagnostic')
  assert remaining()>120;time.sleep(20)
 for name in ['attribution.py','verify_beta.py','make_report.py']:
  assert remaining()>90
  with (R/'logs'/(name+'.log')).open('x') as f:p=subprocess.run([PY,'-B','-u',str(R/name)],stdout=f,stderr=subprocess.STDOUT,timeout=min(180,remaining()-60))
  if p.returncode:raise RuntimeError('Failed '+name)
  print('REPORT_STAGE',name,flush=True)
except Exception:
 save(R/'report_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
