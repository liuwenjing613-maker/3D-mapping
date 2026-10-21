from pathlib import Path
import json
R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
for d in sorted(R.glob('*/*')):
 if not d.is_dir():continue
 row=dict(group=d.parent.name,scene=d.name)
 for n in ['progress.json','mapping_failure.json','mapping_complete.json','evaluation_complete.json','evaluation_failure.json']:
  p=d/n
  if p.exists():
   v=json.loads(p.read_text());row[n]={k:v[k] for k in ['status','stage','frames','variant','seconds','traceback'] if k in v}
 if len(row)>2:print(json.dumps(row))
