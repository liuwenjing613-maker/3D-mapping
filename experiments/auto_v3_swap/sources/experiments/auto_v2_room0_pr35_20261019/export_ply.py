"""Export the completed room0 auto-v2 E1 map using the established two-PLY format."""
from pathlib import Path
import colorsys
import hashlib
import json
import numpy as np
from plyfile import PlyData, PlyElement

R = Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
F = R / 'e1_repair/e1/E1_expanded_history'
O = R / 'ply_black_yellow_red_20261019'
O.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [F / 'room0/final/instance_surface.npz', F / 'room0/native/surface_evidence.npz']
freeze = json.loads((F / 'predictions_freeze.json').read_text())
source_hashes = {str(p): sha(p) for p in paths}
for p in paths:
    assert source_hashes[str(p)] == freeze['output_sha256'][str(p)], p
z = np.load(paths[0])
e = np.load(paths[1])
xyz, ids, state = z['xyz_m'], z['instance_id'], e['state']
np.testing.assert_array_equal(xyz, e['xyz_m'])
np.testing.assert_array_equal(z['rgb'], e['rgb'])
assert np.isfinite(xyz).all()
unassigned = ids <= 0
assert np.isin(state[unassigned], [0, 1, 3]).all()
counts = {str(s): int(((state == s) & unassigned).sum()) for s in [0, 1, 3]}
evaluation=json.loads((R/'e1_repair/evaluation_r2/comparison.json').read_text())
row=next(r for r in evaluation['rows'] if r['variant']=='after' and r['stage']=='final')
assert counts == {str(s):row['gray_states'][n] for s,n in [(0,'U'),(1,'T'),(3,'C')]}, counts
assert sum(counts.values())==row['gray_total']
files, exported_indices = [], []
for kind, mask in [('instances', ~unassigned), ('three_states_black_yellow_red', unassigned)]:
    ix = np.flatnonzero(mask).astype(np.int32)
    rgb = np.empty((len(ix), 3), np.uint8)
    if kind == 'instances':
        for pid in np.unique(ids[ix]):
            rgb[ids[ix] == pid] = np.asarray(colorsys.hsv_to_rgb((int(pid) * .61803398875) % 1, .67, .94)) * 255
    else:
        for st, col in [(0, [0, 0, 0]), (1, [255, 255, 0]), (3, [255, 0, 0])]:
            rgb[state[ix] == st] = col
    a = np.empty(len(ix), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('instance_id', '<i4'), ('state', 'u1'), ('surface_index', '<i4')])
    for j, n in enumerate(['x', 'y', 'z']):
        a[n] = xyz[ix, j]
    for j, n in enumerate(['red', 'green', 'blue']):
        a[n] = rgb[:, j]
    a['instance_id'], a['state'], a['surface_index'] = ids[ix], state[ix], ix
    p = O / ('room0_auto35_E1_' + kind + '.ply')
    PlyData([PlyElement.describe(a, 'vertex')], text=False, byte_order='<').write(p)
    back = PlyData.read(p)['vertex'].data
    for n in a.dtype.names:
        np.testing.assert_array_equal(a[n], back[n])
    np.testing.assert_array_equal(np.column_stack([back[n] for n in ['x', 'y', 'z']]), xyz[ix])
    np.testing.assert_array_equal(back['instance_id'], ids[ix])
    np.testing.assert_array_equal(back['state'], state[ix])
    exported_indices.append(back['surface_index'].copy())
    files.append({'file': p.name, 'sha256': sha(p), 'points': len(ix), 'bytes': p.stat().st_size})
assert sum(x['points'] for x in files) == len(xyz)
np.testing.assert_array_equal(np.sort(np.concatenate(exported_indices)), np.arange(len(xyz)))
for p in paths:
    assert sha(p) == source_hashes[str(p)], p
manifest = {'status': 'PASS', 'source_run': str(R), 'source_sha256': source_hashes,
    'total_points': len(xyz), 'bounds': [xyz.min(0).tolist(), xyz.max(0).tolist()],
    'files': files, 'state_counts': counts,
    'state_colors': {'0': [0, 0, 0], '1': [255, 255, 0], '3': [255, 0, 0]},
    'partition_disjoint_and_complete': True, 'all_points_exported_without_downsampling': True,
    'xyz_ids_and_surface_indices_exact': True, 'source_files_unchanged': True,
    'record_stride': 24}
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
