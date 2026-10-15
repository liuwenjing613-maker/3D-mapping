from pathlib import Path
import json,numpy as np
base=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain')
for p in [base/'final/instance_surface.npz',base/'surface_evidence.npz']:
 with np.load(p) as z:
  print(p.name,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
  for k in ['state','instance_id']:
   if k in z:
    ids,n=np.unique(z[k],return_counts=True); print(k,'unique',len(ids),'first',list(zip(ids[:12].tolist(),n[:12].tolist())))
c=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json').read_text())
print('config',c)
m=json.loads(Path('/data/chenkejun/CVPR/results/e1_roi_top10_20261014/input_manifest.json').read_text())
print('manifest keys',list(m),'crop jobs',sum(j['kind']=='crop' for j in m['jobs']),'frames',sorted(set(j['frame'] for j in m['jobs'] if j['kind']=='crop')))
