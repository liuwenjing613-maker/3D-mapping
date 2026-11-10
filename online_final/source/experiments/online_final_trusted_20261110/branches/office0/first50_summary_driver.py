from common_continue import *
from selection_runtime import run
import time,traceback
frontend=CANDIDATE_ROOT/'frontend';out=frontend/'run_summary';out.mkdir(parents=True,exist_ok=True)
try:
 while True:
  states=[]
  for name in ['sam3','cropformer']:
   f=frontend/('run_'+name)/'driver_status.json';states.append(read(f)['status'] if f.exists() else 'RUNNING')
  if 'FAILED' in states:raise RuntimeError('candidate model failed: '+str(states))
  if states==['PASS','PASS']:break
  dump(out/'driver_status.json',dict(status='WAITING_MODELS',models=states,unix=time.time()));time.sleep(30)
 dump(out/'driver_status.json',dict(status='RUNNING',stage='combine_proposals',unix=time.time()))
 row=run('summary',CONT/'selection_code/50/B/frontend/summarize_frontend.py',out=out)
 result=read(frontend/'pipeline_complete.json');assert result['status']=='PASS'
 dump(out/'driver_status.json',dict(status='PASS',process=row,tracking_or_repair=False))
except Exception:
 dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
