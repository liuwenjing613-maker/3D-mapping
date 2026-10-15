from pathlib import Path
import json,subprocess,time
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
for name in ['tracking_queue_failure.json','tracking_queue_status.json','pipeline_status.json','pipeline_failure.json']:
 p=R/name
 if p.exists():
  d=json.loads(p.read_text())
  if name=='tracking_queue_status.json':
   d.pop('completed',None);d['pending_count']=len(d.pop('pending',[]))
   for a in d['active']:
    pp=R/'cases'/a['case_uid']/'tracking/status.json'
    if pp.exists():
     x=json.loads(pp.read_text());a.update(frames=x.get('completed_frames'),seconds=x.get('elapsed_seconds'))
    else:
     log=R/'cases'/a['case_uid']/'tracking.log';a['log_tail']=log.read_text(errors='replace')[-500:] if log.exists() else None
  print(name,json.dumps(d),flush=True)
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
