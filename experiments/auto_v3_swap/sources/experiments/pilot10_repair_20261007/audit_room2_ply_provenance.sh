set -euo pipefail
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
import json, hashlib, numpy as np
root=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
out=root/'tracking_review'
c=next(c for c in json.loads((out/'ply_models/manifest.json').read_text())['cases'] if c['scene']=='room2')
p=out/c['full']['path']; raw=p.read_bytes(); offset=raw.index(b'end_header\n')+len(b'end_header\n')
dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('baseline_id','<i4'),('seed_id','<i4'),('instance_id','<i4'),('seed_track_id','<i4')])
a=np.frombuffer(raw,dt,offset=offset)
checks=[]
for (path,expected),label in zip(c['source_maps'].items(),['baseline_id','seed_id','instance_id']):
    path=Path(path); digest=hashlib.sha256(path.read_bytes()).hexdigest(); assert digest==expected
    with np.load(path) as z:
        np.testing.assert_array_equal(np.column_stack([a[k] for k in ['x','y','z']]),z['xyz_m'])
        np.testing.assert_array_equal(a[label],z['instance_id'])
        np.testing.assert_array_equal(np.column_stack([a[k] for k in ['red','green','blue']]),np.rint(np.clip(z['rgb'],0,1)*255).astype(np.uint8))
        checks.append({'source':str(path),'sha256':digest,'npz_keys':z.files,'geometry_exact':True,'labels_exact':True,'RGB_exact_to_uchar_rounding':True})
record={'status':'PASS','points':len(a),'sources':checks,'server_PLY_hash_matches':hashlib.sha256(raw).hexdigest()==c['full']['sha256']}
(root/'tracking_review/room2_ply_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
find /data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/final/room2/P1-A1 -maxdepth 3 -type f | head -n 35
