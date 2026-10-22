from online_common import *
import os,subprocess,time,traceback
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONPATH':str(C)}
def run(name,extra=None):
    write(R/'pipeline_progress.json',dict(stage=name,time=time.time()))
    with (R/(name+'.log')).open('ab') as log:
        subprocess.run([PY,'-B','-u',str(C/(name+'.py'))],cwd=C,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    run('prepare_prefix')
    env.update(PYTHONPATH=str(C/'guard')+':'+str(C),CVPR_ONLINE_GUARD=str(R/'input_guard.json'))
    run('map_online')
    run('export_prefix_ply')
    run('test_guard',{'CVPR_GUARD_TEST':'1'})
    write(R/'pipeline_progress.json',dict(stage='mapping_complete_selection_not_started',time=time.time()))
except Exception:
    write(R/'pipeline_failure.json',dict(error=traceback.format_exc(),time=time.time()))
    raise
