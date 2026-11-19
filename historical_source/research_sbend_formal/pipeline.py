import subprocess,sys,traceback
from common import *
GPU='/home/cudauser/.venvs/picflow-gpu/bin/python'
MEEP='/home/cudauser/.conda/envs/picflow-meep/bin/python'
def main():
    verify_freeze()
    for script,python in [('sampling.py',MEEP),('generate_data.py',MEEP),('prepare_training.py',MEEP),('train.py',GPU),('evaluate.py',GPU)]:
        write('pipeline_status.json',{'time_utc':now(),'stage':script,'status':'running'})
        subprocess.run([python,'-B','-u',str(OUT/script)],check=True)
    write('pipeline_status.json',{'time_utc':now(),'status':'complete'})
if __name__=='__main__':
    try:main()
    except Exception as e:
        write('pipeline_failure.json',{'time_utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'paused':True});raise
