from pathlib import Path
import os,sys,json
import numpy as np
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110')
F=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101')
sys.path.insert(0,str(F/'policies/online'))
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src')
from scene_completion.pipeline import manifest_from_commit,finalize
s=sys.argv[1];root=R/'online'/s/'400/B';out=R/'published/online_B'/s;out.mkdir(parents=True,exist_ok=True)
for n in [50,100,200,300,400]:
 d=json.loads((R/'online'/s/str(n)/'B/repair/commit_complete.json').read_text());assert d['status']=='PASS' and d['processed_frames']==n
manifest=out/'scene_complete.json'
if not manifest.exists():manifest.write_text(json.dumps(manifest_from_commit(root,scene=s,workflow='online-v2'),indent=2))
receipt=finalize(manifest,out/'final_B');assert receipt['status']=='PASS' and not receipt['C_enabled'] and not receipt['online_state_updated']
np.testing.assert_array_equal(np.load(receipt['final_surface'])['xyz_m'],np.load(R/'attributes'/s/'400/normals.npz')['xyz_m'])
print('FINAL_B_PASS',s)
