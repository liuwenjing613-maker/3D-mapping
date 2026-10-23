from pathlib import Path
import json,os
O=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023/window_only_repair400')
for branch in ['A','B']:
 root=O/'400'/branch
 for name in ['status.json','tracking_cache_plan.json','tracking/tracking_queue_status.json','tracking/tracking_queue_complete.json','tracking/progress.json','repair/progress.json','repair/commit_progress.json']:
  p=root/name
  if p.exists():
   d=json.loads(p.read_text());keep=['status','stage','seconds','frames','total','processed_frames','total_frames','completed_cases','total_cases','matching_cases','inferred_cases','reused_cases','traceback']
   print(branch,name,json.dumps({k:v for k,v in d.items() if k in keep}))
 for p in (root/'tracking/cases').glob('*/tracking/status.json'):
  if p.with_name('complete.json').exists():continue
  d=json.loads(p.read_text());print(branch,'ACTIVE_SAM',json.dumps({k:v for k,v in d.items() if k in ['case_uid','completed_frames','total_frames','elapsed_seconds']}))
for name in ['post_status.json','post_complete.json']:
 p=O/name
 if p.exists():
  d=json.loads(p.read_text());print(name,json.dumps({k:v for k,v in d.items() if k!='PLY_files'}))
