from pathlib import Path
from collections import Counter
import json,hashlib,time
R=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021')
O=R/'all_edge_20261021';W=O/'review'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(R/'mapping_complete.json');causal=read(R/'causality_audit.json')
assert m['status']==causal['status']=='PASS'
assert m['frame_ids']==list(range(0,396,5)) and m['frame_count']==80 and m['last_frame_id']==395
assert causal['independent_raw_support_vote_recount'] and causal['forbidden_read_attempts_in_computation']==0
d=read(W/'data.json');a=read(O/'boundary_audit.json');g=read(O/'v1/grouping.json');s=d['summary'];rows=d['rows']
assert d['prefix']==80 and s['status']=='PASS' and s['cutoff']==395
assert len(rows)==a['stats']['input_masks']
assert len([r for r in rows if r['status']=='kept'])==s['retained']
assert a['stats']['total_edge_removed']==a['stats']['full_frame_removed']+a['stats']['crop_removed']-a['stats']['both_edge_removed']
for p,h in g['source_sha256'].items():assert sha(p)==h,p
for r in rows:
 assert r['frame'] in m['frame_ids']
 assert (r['status']=='edge')==(any(r['edge_pixels'].values()) or bool(r['crop_edges']))
 for key in ('full','zoom','rgb','mask'):assert (W/r[key]).is_file(),r[key]
 if r['status']=='kept':assert not any(r['edge_pixels'].values()) and not r['crop_edges']
for r in g['seeds']:assert not r['touches_any_edge'] and not r['artificial_crop_edges']
events=0;blocked=0;probes=0
allowed={x['source']:x['frame'] for x in read(R/'inputs_manifest.json')['files']}
for p in (R/'io_audit').glob('*.jsonl'):
 for line in p.read_text().splitlines():
  e=json.loads(line)
  if e.get('probe'):probes+=bool(e['blocked']);continue
  events+=1;blocked+=bool(e['blocked'])
  if e['path'] in allowed:assert allowed[e['path']]<=e['cutoff']<=395
assert blocked==0
out=dict(status='PASS',prefix=80,cutoff=395,source_masks_unchanged=True,all_active_and_auxiliary_masks_edge_free=True,
 counts=dict(Counter(r['status'] for r in rows)),initial=a['stats'],retained=s['retained'],assets_verified=True,recorded_IO_events=events,forbidden_read_attempts=blocked,intentional_fence_probes_blocked=probes,
 GT_used=False,future_frames_used=False,repair_applied=False,data_sha256=sha(W/'data.json'),at=time.time())
(W/'qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
