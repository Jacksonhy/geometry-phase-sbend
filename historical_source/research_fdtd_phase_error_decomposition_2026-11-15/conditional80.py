"""Only the frozen three conditional res80 structures; no expansion."""
import sys,time
from experiment import *
gid=sys.argv[1];assert gid in read(R/'config.json')['conditional80_ids'];job=dict(id='res80_'+gid,kind='sbend',geometry_id=gid,resolution=80);t=time.perf_counter();state=build(job);state[3].run(until_after_sources=state[0].mp.stop_when_dft_decayed(tol=1e-6));snapshot(job,R/'runs'/job['id'],state);state[3].reset_meep();save(R/'costs'/(job['id']+'.json'),dict(job=job,wall_seconds=time.perf_counter()-t));print('RES80_DONE',gid,flush=True)
