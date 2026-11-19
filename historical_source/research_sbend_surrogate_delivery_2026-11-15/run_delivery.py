import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,subprocess
from predict import R,MEEP,METHODS,SEEDS,save
GPU=sys.executable
start=time.time()
def call(args,name):
 t=time.perf_counter()
 with (R/'logs'/(name+'.log')).open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,round(time.perf_counter()-t,2),flush=True)
call([GPU,'-B',str(R/'audit.py'),'freeze'],'freeze')
call([GPU,'-B',str(R/'audit.py'),'inputs','smoke_inputs','1'],'check_smoke')
call([MEEP,'-B',str(R/'predict.py'),'--geometry-json',str(R/'blind_geometries.json'),'--prepare-only','--output',str(R/'inputs')],'prepare72')
call([GPU,'-B',str(R/'audit.py'),'inputs'],'check72')
for s in SEEDS:
 for m in METHODS:call([GPU,'-B',str(R/'predict.py'),'--geometry-json',str(R/'blind_geometries.json'),'--prepared-inputs',str(R/'inputs'),'--method',m,'--seed',str(s),'--output',str(R/'replays'/f'seed{s}_{m}.npz')],f'replay_{s}_{m}')
call([GPU,'-B',str(R/'audit.py'),'predictions'],'check_predictions')
call([GPU,'-B',str(R/'benchmark.py')],'benchmarks')
call([GPU,'-B',str(R/'audit.py'),'verify'],'verify')
save(R/'computation_complete.json',dict(passed=True,elapsed_seconds=time.time()-start,unix=time.time(),training_runs=0,FDTD_runs=0))
