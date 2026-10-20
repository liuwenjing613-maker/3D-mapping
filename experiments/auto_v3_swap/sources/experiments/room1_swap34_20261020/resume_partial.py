from controller import C,R,save
import json
import os
import subprocess
import sys
import time
import traceback

try:
    assert json.loads((R/'e1_repair/run_complete.json').read_text())['status']=='PASS'
    env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
    save('status.json',dict(status='RUNNING',stage='partial',attempt=2,unix=time.time()))
    with (R/'partial.controller.log').open('x') as stream:
        p=subprocess.Popen([sys.executable,'-B','-u',str(C/'partial/controller.py')],cwd=C/'partial',
                            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        assert p.wait()==0,'partial failed'
    save('complete.json',dict(status='PASS',unix=time.time(),scope='room1',partial_attempt=2))
    save('status.json',dict(status='PASS',stage='complete',unix=time.time()))
except Exception:
    save('failure.json',dict(status='FAIL',traceback=traceback.format_exc()))
    raise
