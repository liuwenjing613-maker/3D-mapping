from common_continue import *
from selection_runtime import run
import traceback
out=CANDIDATE_ROOT/'semantic_runtime';out.mkdir(exist_ok=True);sc=CONT/'selection_code/50/B'
try:
 assert read(CANDIDATE_ROOT/'versions_runtime/driver_status.json')['status']=='PASS'
 for stage,script,gpu in [('classify',sc/'semantic_route/classify.py',True),('cabinet',sc/'semantic_route/refine_cabinet.py',True),('window_policy',CONT/'window_source_policy.py',False),('route',sc/'semantic_route/route.py',False)]:
  dump(out/'driver_status.json',dict(status='RUNNING',stage=stage,unix=time.time(),not_fixedH=True));run(stage,script,gpu=gpu,out=out)
 dump(out/'driver_status.json',dict(status='PASS',stage='original_semantic_route',not_fixedH=True,not_tracking_or_repair=True))
except Exception:dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc(),not_fixedH=True));raise
