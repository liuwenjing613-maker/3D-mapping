from pathlib import Path
import json,numpy as np
M=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
protocol=json.loads((M/'unified_eval/configs/current_protocol.json').read_text())
for p in [Path(protocol['canonical_GT']['office2']['file']),O/'office2/evaluation_r2/B/after/overlap_matrix.npz',O/'office2/evaluation_r2/B/fixed_correspondence.npz']:
 with np.load(p,allow_pickle=False) as z:
  print('SCHEMA',p,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
  if 'metadata_json' in z:
   meta=json.loads(str(z['metadata_json']));print('META_KEYS',list(meta))
   for k,v in meta.items():
    if k!='observation_metadata':print(k,str(v)[:800])
for p in [O/'office2/400/B/repair/seed_id_map.json',O/'office2/400/B/repair/seed_diagnostics.json']:
 d=json.loads(p.read_text());print('JSON',p,str(d)[:2800])
