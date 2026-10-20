from run_v1 import run,check,save,R
import json,time,traceback,sys
try:
 stages=['grouping_r2','grouping_r3','v1']
 start=sys.argv[1] if len(sys.argv)>1 else 'grouping_r2'
 for stage in stages[stages.index(start):]:check(run(stage,'compute.py'))
 audit=json.loads((R/'v1/audit.json').read_text());assert audit['status']=='PASS'
 save('v1_complete',dict(status='PASS',stats=audit['stats'],time=time.time(),resumed_from='grouping_r2'))
 save('v1_status',dict(stage='complete',time=time.time()))
except Exception:
 save('v1_failure',dict(error=traceback.format_exc(),time=time.time()));raise
