set -euo pipefail
PYTHONDONTWRITEBYTECODE=1 /data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
from plyfile import PlyData,PlyElement
root=Path('/data/chenkejun/CVPR/results/p1a1_history_core_reassociation_20261009')
out=root/'exports_20261010'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
freeze=json.loads((root/'predictions_freeze.json').read_text())
assert freeze['status']=='TWO_SCENE_PREDICTIONS_FROZEN_BEFORE_NEW_R2_SCORING'
out.mkdir(exist_ok=True)
receipt=out/'ply_export_complete.json'
if receipt.exists():
    old=json.loads(receipt.read_text())
    for item in old['files']:assert sha(item['path'])==item['sha256']
    print(json.dumps(old));raise SystemExit(0)
files=[]
for scene in ('room0','room2'):
    source=root/scene/'final/instance_surface.npz'
    evidence=root/scene/'native/surface_evidence.npz'
    assert sha(source)==freeze['output_sha256'][str(source)]
    assert sha(evidence)==freeze['output_sha256'][str(evidence)]
    with np.load(source,allow_pickle=False) as z:
        xyz=z['xyz_m'].copy();rgb=z['rgb'].copy();ids=z['instance_id'].copy()
    with np.load(evidence,allow_pickle=False) as z:state=z['state'].copy()
    n=len(xyz);col=(matplotlib.colormaps['hsv']((ids.astype(np.int64)*.61803398875)%1)[:,:3]*255).round().astype(np.uint8)
    col[ids<=0]=153
    coord='f4' if xyz.dtype.itemsize<=4 else 'f8'
    dtype=[('x',coord),('y',coord),('z',coord),('red','u1'),('green','u1'),('blue','u1'),
           ('instance_id','i4'),('native_state','u1'),('surface_index','i4'),
           ('original_red','u1'),('original_green','u1'),('original_blue','u1')]
    vertices=np.empty(n,dtype=dtype)
    for j,k in enumerate(('x','y','z')):vertices[k]=xyz[:,j]
    for j,k in enumerate(('red','green','blue')):vertices[k]=col[:,j];vertices['original_'+k]=rgb[:,j]
    vertices['instance_id']=ids;vertices['native_state']=state;vertices['surface_index']=np.arange(n,dtype=np.int32)
    target=out/(scene+'_history_reassociation_instances.ply');assert not target.exists()
    PlyData([PlyElement.describe(vertices,'vertex')],text=False,byte_order='<',comments=[
        'Full fixed TSDF surface; final strict one-vote plus abstention and manual-core history reassociation',
        'Positive instance_id identifies instance; nonpositive unassigned points are gray',
        'native_state: 0 UNOBSERVED, 1 TENTATIVE, 2 CONFIRMED, 3 CONFLICT; final labels include postprocessing',
        'original_red/green/blue retain original surface RGB; colors match case report',
        'Evaluation: object_observed_repair revision 2 development']).write(target)
    check=PlyData.read(target)['vertex'].data
    assert len(check)==n
    for field in vertices.dtype.names:np.testing.assert_array_equal(check[field],vertices[field])
    assert sha(source)==freeze['output_sha256'][str(source)]
    files.append({'scene':scene,'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
        'vertices':n,'gray_points':int(np.sum(ids<=0)),'source':str(source),'source_sha256':sha(source),
        'all_exported_fields_roundtrip_exact':True})
result={'status':'PASS','format':'binary_little_endian PLY','files':files}
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
