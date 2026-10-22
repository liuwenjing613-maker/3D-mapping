from online_common import *
from run_selection import ENV
import subprocess,time,traceback
try:
 for stage,args in [('selection_driver',[str(C/'run_selection.py'),'--start','v2']),('finalize',[str(C/'finalize_delivery.py')])]:
  write(R/'pipeline_progress.json',dict(stage=stage,time=time.time()))
  with (R/(stage+'_approved.log')).open('ab') as f:
   subprocess.run([PY,'-B','-u',*args],cwd=C,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(R/'pipeline_progress.json',dict(stage='complete',result=read(R/'selection_complete.json')['result'],time=time.time()))
except Exception:
 write(R/'pipeline_failure.json',dict(error=traceback.format_exc(),time=time.time()))
 raise
