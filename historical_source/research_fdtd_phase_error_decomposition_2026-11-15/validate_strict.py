import time,traceback
import experiment as e
R=e.R
try:
 while not (R/'costs/courant025_w0.575_r40.json').exists():
  assert e.remaining()>600;time.sleep(15)
 for gid in e.read(R/'config.json')['control_ids']:
  j=dict(id=f'strictvalidated_{gid}_r40',kind='sbend',geometry_id=gid,resolution=40);t=time.perf_counter();state=e.build(j);sim=state[3];sim.run(until_after_sources=state[0].mp.stop_when_dft_decayed(tol=1e-9,minimum_run_time=400,maximum_run_time=800));e.snapshot(j,R/'runs'/j['id'],state);sim.reset_meep();e.save(R/'costs'/(j['id']+'.json'),dict(job=j,wall_seconds=time.perf_counter()-t,DFT_tol=1e-9,absolute_minimum_time=400,absolute_maximum_time=800));print('STRICT_VALIDATED',gid,flush=True)
 e.save(R/'strict_validated_complete.json',dict(unix=time.time(),controls=3,definition='independent tol1e-9 runs absolute minimum400 maximum800; check actual times',no_budget_reset=True))
except Exception:
 e.save(R/'strict_validation_stopped.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise
