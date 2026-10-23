"""Run the actual downstream pipelines independently, preserving each candidate set."""
from pathlib import Path
import json
import os
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)

def main():
    records=[]
    for mode in ['reference','checkpoint']:
        code=HERE/mode;root=OUT/mode
        env={**os.environ,'CVPR_CHECKPOINT':'200','CVPR_TRACK_GROUP':'A',
             'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
        env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
        for stage,script in [('selection_tracking','selection_run.py'),('repair','repair_continue.py'),('commit','commit_continue.py')]:
            markers={'selection_tracking':root/'selection_tracking_status.json',
                     'repair':root/'repair/repair_complete.json','commit':root/'repair/commit_complete.json'}
            marker=markers[stage]
            if marker.exists() and json.loads(marker.read_text()).get('status')=='PASS':continue
            if stage!='selection_tracking':
                env.update(CVPR_ONLINE_GUARD=str(root/'input_guard.json'),
                           PYTHONPATH=str(code/'selection_guard')+':'+str(code))
            start=time.perf_counter()
            with (root/(stage+'_execution.log')).open('ab',buffering=0) as log:
                proc=subprocess.Popen([PY,'-B','-u',str(code/script)],cwd=code,env=env,
                                      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(OUT/'closed_loop_status.json',dict(status='RUNNING',mode=mode,stage=stage,pid=proc.pid,completed=records))
                ret=proc.wait()
            row=dict(mode=mode,stage=stage,returncode=ret,seconds=time.perf_counter()-start)
            records.append(row)
            write(root/(stage+'_execution_timing.json'),row)
            assert ret==0,(mode,stage,ret)
            assert json.loads(marker.read_text())['status']=='PASS'
    write(OUT/'closed_loop_status.json',dict(status='COMMITTED_BOTH',completed=records))

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'closed_loop_status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
