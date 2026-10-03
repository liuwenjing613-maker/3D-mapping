from pathlib import Path
import json
import numpy as np
from PIL import Image
P=Path(__file__).parent/'inference_arrays'
m=json.loads((P/'input_manifest.json').read_text())
rows=[]
for j in m['jobs']:
 if j.get('strategy')!='witness':continue
 b=np.asarray(Image.open(P/'inputs'/f'{j["key"]}_baseline.png'));a=np.asarray(Image.open(P/'inference'/f'{j["key"]}.png'));u,v=map(int,j['alarm_projection_anchor_uv']);before=int(b[v,u]);after=int(a[v,u])
 assert before==j['source_mask_local_id']
 rows.append({'key':j['key'],'source_instance_ID':j['source_instance_id'],'before_F':before,'after_L':after,'before_mask_crop_area':int((b==before).sum()),'after_mask_crop_area':int((a==after).sum()),'source_point_pixel_uv':[u,v]})
(P.parent/'witness_pixel_diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
