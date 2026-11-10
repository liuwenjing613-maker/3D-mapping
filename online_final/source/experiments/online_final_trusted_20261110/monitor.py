from pathlib import Path
import json
R=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110')
out={}
def rd(p):return json.loads(p.read_text())
for s in ['room0','room1','room2','office0','office1','office2','office3','office4']:
 b=R/'online'/s;p=b/'optimization_status.json';d=rd(p) if p.exists() else {};n=d.get('checkpoint',400);root=b/str(n)/'B'
 row={'status':d.get('status'),'checkpoint':n,'stage':d.get('stage'),'completed_checkpoints':[cp for cp in [50,100,200,300,400] if (b/str(cp)/'B/repair/commit_complete.json').exists() and rd(b/str(cp)/'B/repair/commit_complete.json')['status']=='PASS']}
 if d.get('status')=='FAILED':row['error']=d.get('traceback')
 p=root/'candidate_chain_status.json'
 if p.exists():
  q=rd(p);row['chain']={k:q[k] for k in ['status','stage','traceback'] if k in q}
 for name,p in [('CF',root/'candidates/frontend/cropformer/progress.json'),('SAM3',root/'candidates/frontend/sam2_1/progress.json')]:
  if p.exists():
   q=rd(p);row[name]={k:q[k] for k in ['status','done','total','traceback'] if k in q}
 p=root/'tracking_sam3/driver_status.json'
 if p.exists():
  q=rd(p);row['tracking']={k:q[k] for k in ['status','completed','input_seeds','pending','traceback'] if k in q}
 p=root/'repair/progress.json'
 if p.exists():
  q=rd(p);row['repair']={k:q[k] for k in ['stage','frames','total'] if k in q}
 p=root/'repair/commit_progress.json'
 if p.exists():
  q=rd(p);row['commit']={k:q[k] for k in ['status','stage','processed_frames'] if k in q}
 p=R/'evaluation'/s/'status.json'
 if p.exists():row['evaluation']=rd(p)
 errors=[]
 for pattern in ['candidates/*/driver_status.json','candidates/frontend/*/driver_status.json']:
  for p in root.glob(pattern):
   q=rd(p)
   if q.get('status')=='FAILED':errors.append({'path':str(p),'error':q.get('traceback')})
 if errors:row['child_errors']=errors
 out[s]=row
for scene in ['room0','office4']:
 p=R/'baseline_terminal_control/control_replay'/scene/'complete.json'
 if p.exists():out['baseline_control/'+scene]={'status':rd(p)['status'],'exact':rd(p)['control_exact']}
for name in ['status.json','failure.json','comparison.json']:
 p=R/'comparison'/name
 if p.exists():
  q=rd(p);out['comparison/'+name]={k:q[k] for k in ['status','completed','total','results','traceback'] if k in q}
print(json.dumps(out))
