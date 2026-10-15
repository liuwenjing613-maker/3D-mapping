from pathlib import Path
import json,numpy as np
from PIL import Image
R=Path('/data/chenkejun/CVPR/results/e1_roi_top10_20261014')
d=json.loads((R/'input_manifest.json').read_text());rows=[]
for c in d['cases']:
 for v in c['views']:
  a=np.asarray(Image.open(R/'source'/v['source_alarm']));k={'U':0,'T':1,'CONFLICT':2}[v['trigger_group']]
  target=a[:,:,k]>0;occluded=target&((a[:,:,k+1:]>0).any(2) if k<2 else False)
  if occluded.any():rows.append(dict(ROI=v['source_entry'],view=v['source_view'],pixels=int(target.sum()),occluded=int(occluded.sum())))
print(json.dumps({'views_with_collisions':rows,'total':len(rows)}))
print('RENDER', (R/'render_annotation.log').read_text()[-600:])
