from pathlib import Path
import json,time
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
def read(n):
 p=R/n
 return json.loads(p.read_text()) if p.exists() else None
s=read('tracking_queue_status.json');cases={r['case_uid']:r for r in read('seed_diagnostics.json')['cases']}
out={'completed_tasks':s['completed_cases'],'total_tasks':s['total_cases'],'completed_seed_masks':sum(len(cases[x['case_uid']]['objects']) for x in s['completed']),
 'active_tasks':len(s['active']),'active_seed_masks':sum(len(cases[x['case_uid']]['objects']) for x in s['active']),'queue_heartbeat_age':time.time()-s['updated_unix']}
for name in ['pipeline_status.json','postprocess_status.json','review_assets_status.json']:
 d=read(name)
 if d:out[name]={k:v for k,v in d.items() if k not in ('completed','stages')}
out['failures']={p.name:json.loads(p.read_text()) for p in R.glob('*failure.json')}
for name in ['validated/full_track/status.json','original_repair/progress.json','modified_repair/progress.json','e1/progress.json']:
 d=read(name)
 if d:out[name]=d
masks=sorted((R/'validated/full_track/accepted_masks').glob('f*.png'))
if masks:out['gate_entered_mapping_frames']=len(masks);out['gate_current_raw_frame']=int(masks[-1].stem[1:])
stage=read('pipeline_status.json')
if stage and stage.get('pid'):
 proc=Path('/proc')/str(stage['pid'])
 if proc.exists():
  out['stage_process']={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (proc/'status').read_text().splitlines() if line.startswith(('State:','VmRSS:','Threads:'))}
for name in ['raw_repair','strict_repair','history_core','expand','E1','evaluation_r2','review']:
 p=R/(name+'.log')
 if p.exists() and name not in ('identity',):out[name+'_tail']=p.read_text(errors='replace')[-1000:]
print(json.dumps(out,ensure_ascii=False))
