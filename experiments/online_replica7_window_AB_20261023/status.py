from pathlib import Path
import json
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
read=lambda p:json.loads(Path(p).read_text())
if (O/'batch_status.json').exists():print('BATCH',json.dumps(read(O/'batch_status.json')))
if (O/'post_status.json').exists():print('POST',json.dumps(read(O/'post_status.json')))
if (O/'recovery_status.json').exists():print('RECOVERY',json.dumps(read(O/'recovery_status.json')))
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 for b in ['A','B']:
  p=O/scene/f'status_{b}.json'
  if not p.exists():continue
  d=read(p);print(scene,b,json.dumps(d))
  if d.get('status')=='RUNNING':
   root=O/scene/str(d['checkpoint'])/b
   for rel in ['mapping/mapping_progress.json','candidates/selection_progress.json','tracking/tracking_queue_status.json','tracking/progress.json','repair/progress.json','repair/commit_progress.json']:
    q=root/rel
    if q.exists():
     x=read(q);keys=['status','stage','processed_frames','frames','total','completed_cases','total_cases','retained','seconds']
     print(' ',rel,json.dumps({k:v for k,v in x.items() if k in keys}))
