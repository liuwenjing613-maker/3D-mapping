from pathlib import Path
import json,sys,hashlib
import numpy as np
R=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110')
F=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
s=sys.argv[1];n=int(sys.argv[2]);root=R/'online'/s/str(n)/'B';m=root/'mapping'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dig(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
d=rd(m/'mapping_complete.json');assert d['status']=='PASS' and d['frame_ids']==list(range(0,n*5,5))
a=np.load(m/'baseline/final/instance_surface.npz');b=np.load(F/'attributes'/s/str(n)/'normals.npz');rgb=np.load(F/'attributes'/s/str(n)/'rgb.npz')
np.testing.assert_array_equal(a['xyz_m'],b['xyz_m']);np.testing.assert_array_equal(a['rgb'],rgb['rgb'])
manifest=rd(F/'attributes'/s/str(n)/'complete.json');assert dig(a['xyz_m'])==manifest['geometry_sha256']
prior=rd(m/'prior_commit.json');old={50:0,100:50,200:100,300:200,400:300}[n];assert prior['processed_frames']==old
if n>50:
 cp=R/'online'/s/str(old)/'B/repair/commit_complete.json'
 assert prior['prior_commit_sha256']==sha(cp)
out=dict(status='PASS',scene=s,checkpoint=n,previous_checkpoint=old,geometry_xyz_sha256=dig(a['xyz_m']),points=len(a['xyz_m']),same_PPT_full_depth_xyz_and_rgb=True,actual_prior_repair_inherited=n>50,GT_used=False)
(root/'geometry_parity_audit.json').write_text(json.dumps(out,indent=2))
print(out)
