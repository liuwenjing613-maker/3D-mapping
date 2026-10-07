set -euo pipefail
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
import json, numpy as np
root=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/final/room2/P1-A1')
types={'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4','int32':'<i4','uint':'<u4','short':'<i2','ushort':'<u2'}
with np.load(root/'diffusion/holes_geodesic/instance_surface.npz') as z:
    xyz=z['xyz_m']; ids=z['instance_id']
records=[]
for name in ['room2_repaired_full.ply','room2_repaired_assigned_only.ply']:
    path=root/'diffusion'/name
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline().decode('ascii').strip(); lines.append(line)
            if line=='end_header':break
        props=[]; n=None; vertex=False
        for line in lines:
            parts=line.split()
            if parts[:2]==['element','vertex']:n=int(parts[2]);vertex=True
            elif parts and parts[0]=='element':vertex=False
            elif vertex and parts[0]=='property':props.append((parts[2],types[parts[1]]))
        a=np.fromfile(f,np.dtype(props),count=n)
    rgb=np.column_stack([a[k] for k in ['red','green','blue']]); gray=(rgb[:,0]==rgb[:,1])&(rgb[:,1]==rgb[:,2])
    record={'path':str(path),'header':lines,'points':n,'gray_rgb_points':int(gray.sum()),'gray_colors':np.unique(rgb[gray],axis=0).tolist()[:20]}
    if n==len(xyz):
        record['geometry_exact_to_frozen_baseline']=bool(np.array_equal(np.column_stack([a[k] for k in ['x','y','z']]),xyz))
        record['gray_where_baseline_unpublished']=int(np.sum(gray&(ids<0)))
        record['gray_where_baseline_published']=int(np.sum(gray&(ids>0)))
        if 'instance_id' in a.dtype.names:record['labels_exact_to_frozen_baseline']=bool(np.array_equal(a['instance_id'],ids))
    records.append(record)
print(json.dumps(records))
for name in ['summary.json','diffusion/materialization_report.json']:
    x=json.loads((root/name).read_text())
    print(json.dumps({'path':str(root/name),'report':x}))
PY
