from online_common import *
import os,subprocess,time,traceback
from run_selection import ENV
def run(name,script):
    write(R/'pipeline_progress.json',dict(stage=name,time=time.time()))
    with (R/(name+'.log')).open('ab') as log:
        subprocess.run([PY,'-B','-u',str(C/script)],cwd=C,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    while True:
        if (R/'pipeline_failure.json').exists():raise RuntimeError((R/'pipeline_failure.json').read_text())
        p=read(R/'pipeline_progress.json') if (R/'pipeline_progress.json').exists() else {}
        if p.get('stage')=='mapping_complete_selection_not_started':break
        time.sleep(5)
    run('selection_driver','run_selection.py')
    run('visuals','prepare_visuals.py')
    run('finalize','finalize_delivery.py')
    write(R/'pipeline_progress.json',dict(stage='complete',time=time.time(),result=read(R/'selection_complete.json')['result']))
except Exception:
    write(R/'pipeline_failure.json',dict(error=traceback.format_exc(),time=time.time()))
    raise
