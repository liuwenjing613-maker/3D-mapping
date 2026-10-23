"""Export full-resolution instance-colored PLY, preserving every point and ID."""
from pathlib import Path
import colorsys, hashlib, json, sys
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
scene,group,count,stage=sys.argv[1:5]
assert scene in ['room1','room2','office0','office1','office2','office3','office4']
n=int(count);assert n in [50,100,200,300,400] and stage in ['before','after']
assert group in ('A', 'B')
root=OUT/scene/str(n)/group
commit = json.loads((root / 'repair/commit_complete.json').read_text())
assert commit['status'] == 'PASS' and commit['processed_frames'] == n
source=Path(commit['final_map']) if stage=='after' else root/'mapping/baseline/final/instance_surface.npz'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
if stage=='after':assert source_hash==commit['final_map_sha256']
with np.load(source, allow_pickle=False) as data:
    xyz, labels = data['xyz_m'].copy(), data['instance_id'].copy()
assert xyz.shape == (len(labels), 3) and np.all(np.isfinite(xyz))
assert np.array_equal(labels, labels.astype(np.int32))
record = np.empty(len(labels), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('instance_id', '<i4')])
for i, name in enumerate(['x', 'y', 'z']):
    record[name] = xyz[:, i]
colors = np.full((len(labels), 3), 150, dtype=np.uint8)
ids = np.unique(labels[labels > 0])
for pid in ids:
    rgb = colorsys.hsv_to_rgb((int(pid) * 0.6180339887498949) % 1.0, 0.67, 0.92)
    colors[labels == pid] = np.rint(np.asarray(rgb) * 255).astype(np.uint8)
for i, name in enumerate(['red', 'green', 'blue']):
    record[name] = colors[:, i]
record['instance_id'] = labels
dest=OUT/'deliverables'/scene
dest.mkdir(parents=True,exist_ok=True)
path = dest / f'{scene}_{group}_{n}_{stage}.ply'
header = ('ply\nformat binary_little_endian 1.0\n'
          'comment Full resolution; colors visualize persistent IDs; gray means unassigned\n'
          f'element vertex {len(labels)}\nproperty double x\nproperty double y\nproperty double z\n'
          'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n').encode()
with path.open('wb') as f:
    f.write(header)
    record.tofile(f)
with path.open('rb') as f:
    assert f.read(len(header)) == header
    check = np.fromfile(f, dtype=record.dtype)
assert np.array_equal(check, record)
assert all(np.array_equal(check[name], xyz[:, i]) for i, name in enumerate(['x', 'y', 'z']))
assert np.array_equal(check['instance_id'], labels)
result = {'status': 'PASS', 'scene':scene,'branch':group,'checkpoint':n,'stage':stage, 'points': len(labels), 'positive_instances': len(ids),
          'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'source': str(source), 'source_sha256':source_hash,
          'all_points_and_instance_ids_roundtrip_exact': True, 'visualization_only': True}
(dest / f'{scene}_{group}_{n}_{stage}_export.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
