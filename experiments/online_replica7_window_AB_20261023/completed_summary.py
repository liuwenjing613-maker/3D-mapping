"""Compact read-only results for already audited and evaluated branches."""
from pathlib import Path
import json
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
read=lambda p:json.loads(Path(p).read_text())
rows=[]
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 for group in ['A','B']:
  marker=O/scene/f'post_{group}_complete.json'
  if not marker.exists():continue
  assert read(marker)['status']=='PASS'
  root=O/scene/'evaluation_r2'/group;e=read(root/'evaluation_complete.json')
  assert e['status']=='PASS' and e['identical_geometry_verified'] and e['profile_revision']==2 and e['frozen'] is False
  before=read(root/'before/metrics.json')['summary'];after=read(root/'after/metrics.json')['summary']
  audit=read(O/scene/'verification'/f'{group}_causal_audit.json');assert audit['status']=='PASS'
  pair=read(root/'pair/paired_repair_metrics.json')
  exports=list((O/'deliverables'/scene).glob(f'{scene}_{group}_*_export.json'));assert len(exports)==10
  rows.append(dict(scene=scene,branch=group,AP50_before=before['CA_AP50_uniform'],AP50_after=after['CA_AP50_uniform'],
   F1_before=before['CA_PRF1_0_5']['F1'],F1_after=after['CA_PRF1_0_5']['F1'],GT=e['scene_evaluable_GT'],
   corrected=pair['corrected_instances'],degraded=pair['degraded_instances'],unchanged=pair['unchanged_instances'],
   source_files_verified=audit['source_files_verified'],checkpoint_audits=len(audit['checkpoints']),PLY_exports=len(exports)))
print(json.dumps(dict(completed=len(rows),total=14,rows=rows)))
