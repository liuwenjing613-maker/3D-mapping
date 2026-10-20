"""Export exact five-version PLYs and fixed-view projections, without GT rendering."""
from pathlib import Path
import json
import sys
import tarfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'strict_local_repair_20261007'))
import run_repairs as r
import audit_and_render as render
import fixed_surface_repair as f

ROOT = r.ROOT / 'objectwise_repair_20261007'
OUT = ROOT / 'review'
NAMES = ['baseline', 'gated_strict', 'ungated_strict', 'sam_objectwise', 'original_objectwise']
DTYPE = np.dtype([(name, '<f4') for name in ('x', 'y', 'z')] +
                 [(name, 'u1') for name in ('red', 'green', 'blue')] +
                 [(name + '_id', '<i4') for name in NAMES] + [('seed_track_id', '<i4')])


def bounds(xyz):
    return [xyz.min(0).tolist(), xyz.max(0).tolist()]


def ply(path, xyz, rgb, labels, owners, indices):
    data = np.empty(len(indices), DTYPE)
    for axis, name in enumerate(('x', 'y', 'z')):
        data[name] = xyz[indices, axis]
    colors = np.rint(np.clip(rgb[indices], 0, 1) * 255).astype(np.uint8)
    for axis, name in enumerate(('red', 'green', 'blue')):
        data[name] = colors[:, axis]
    for name in NAMES:
        data[name + '_id'] = labels[name][indices]
    data['seed_track_id'] = owners[indices]
    lines = ['ply', 'format binary_little_endian 1.0', 'comment Immutable TSDF; five exact label versions',
             'element vertex %d' % len(indices), 'property float x', 'property float y', 'property float z',
             'property uchar red', 'property uchar green', 'property uchar blue']
    lines.extend('property int ' + name + '_id' for name in NAMES)
    lines.extend(['property int seed_track_id', 'end_header', ''])
    header = '\n'.join(lines).encode('ascii')
    with path.open('wb') as stream:
        stream.write(header); data.tofile(stream)
    restored = np.memmap(path, mode='r', dtype=DTYPE, offset=len(header), shape=(len(indices),))
    np.testing.assert_array_equal(np.c_[restored['x'], restored['y'], restored['z']], xyz[indices])
    for name in NAMES:
        np.testing.assert_array_equal(restored[name + '_id'], labels[name][indices])
    return {'path': 'ply_models/' + path.name, 'sha256': r.sha(path), 'points': len(indices), 'bytes': path.stat().st_size}


def main():
    OUT.mkdir(exist_ok=True)
    for name in ('assets', 'ply_models', 'reports'):
        (OUT / name).mkdir(exist_ok=True)
    cases = {name: ROOT / folder for name, folder in [('control', 'room2_original_control'),
                                                    ('sam', 'room2_sam_seed'), ('original', 'room2_original_seed')]}
    reports = {}
    for name, work in cases.items():
        complete = json.loads((work / 'complete.json').read_text())
        evaluation = json.loads((work / 'evaluation_summary.json').read_text())
        assert complete['status'] == evaluation['status'] == 'PASS'
        assert complete['repair']['strict_commit']['final_outside_changes'] == 0
        for path, expected in complete['repair']['output_sha256'].items():
            assert r.sha(path) == expected
        reports[name] = {'complete': complete, 'evaluation': evaluation}
        for filename in ['complete.json', 'source_freeze.json', 'evaluation_summary.json', 'evaluation_freeze.json',
                         'frozen_association.json', 'frame_decisions.json', 'tracking_validation.json']:
            if (work / filename).exists():
                f.atomic_json(OUT / 'reports' / (name + '_' + filename), json.loads((work / filename).read_text()))
    strict = r.ROOT / 'strict_local_repair_20261007/room2_mask14_mask24'
    sources = {'baseline': r.BASE / 'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz',
               'gated_strict': strict / 'gated_strict/final/instance_surface.npz',
               'ungated_strict': strict / 'ungated_strict/final/instance_surface.npz',
               'sam_objectwise': cases['sam'] / 'final/instance_surface.npz',
               'original_objectwise': cases['original'] / 'final/instance_surface.npz'}
    baseline = f.load_arrays(sources['baseline'])
    xyz, rgb = baseline['xyz_m'], baseline['rgb']
    labels, hashes = {}, {}
    for name, path in sources.items():
        value = f.load_arrays(path)
        f.require_same_geometry(baseline, value)
        labels[name] = value['instance_id']; hashes[str(path)] = r.sha(path)
    control = f.load_arrays(cases['control'] / 'final/instance_surface.npz')
    np.testing.assert_array_equal(control['instance_id'], labels['baseline'])
    support = f.load_arrays(cases['sam'] / 'seed_projected_support.npz')
    owners = np.zeros(len(xyz), np.int32)
    points = np.unique(support['surface_point_index'])
    for oid in (4, 7):
        owners[support['surface_point_index'][support['track_id'] == oid]] = oid
    seed_bounds = np.array(bounds(xyz[points]))
    changed = np.any(np.stack([labels[name] != labels['baseline'] for name in NAMES]), axis=0)
    indices = np.flatnonzero(np.all((xyz >= seed_bounds[0] - .55) & (xyz <= seed_bounds[1] + .55), axis=1) | changed)
    source = r.ReplicaFrameSource(r.INPUT / 'room2/configs/raw.json')
    seed_frame = source.load_frame(1065)
    model = {'scene': 'room2', 'case_uid': 'room2-14047feb0df3aaa5', 'state_names': NAMES,
             'record_stride_bytes': DTYPE.itemsize, 'seed_bounds': seed_bounds.tolist(),
             'seed_camera_to_world': seed_frame.camera_to_world.tolist(), 'native_bounds': bounds(xyz),
             'local_bounds': bounds(xyz[indices]), 'source_hashes': hashes, 'preview_downsampling': False}
    model['local'] = ply(OUT / 'ply_models/objectwise_local.ply', xyz, rgb, labels, owners, indices)
    model['full'] = ply(OUT / 'ply_models/objectwise_full.ply', xyz, rgb, labels, owners, np.arange(len(xyz)))
    case = next(row for row in json.loads((HERE.parent / 'review_manifest.json').read_text())['cases'] if row['case_uid'] == model['case_uid'])
    palette = render.palette(max(int(values.max()) for values in labels.values()))
    palette[52] = [40, 186, 245]; palette[355] = [64, 234, 120]
    views = []
    for view in case['views']:
        fid, box = view['frame'], tuple(view['box'])
        frame = source.load_frame(fid)
        pixels, surface = render.visible_pixels(xyz, frame)
        assets = {}
        for name, values in labels.items():
            asset = 'assets/f%06d_%s.jpg' % (fid, name)
            Image.fromarray(render.label_overlay(frame.rgb, pixels, surface, values, palette)).crop(box).save(OUT / asset, quality=94)
            assets[name] = asset
        views.append({'frame': fid, 'box': box, 'assets': assets,
                      'same_geometry_depth_visibility_for_every_state': True})
    timeline = {name: json.loads((work / 'frame_decisions.json').read_text())['frames'] for name, work in cases.items() if name != 'control'}
    old_summary = json.loads((strict / 'complete.json').read_text())
    data = {'status': 'PASS', 'reports': reports, 'previous_strict_summary': old_summary,
            'model': model, 'views': views, 'timeline': timeline, 'rendering_uses_GT': False,
            'all_maps_geometry_and_PLY_label_roundtrips_exact': True, 'control_final_equals_baseline': True}
    f.atomic_json(OUT / 'comparison_data.json', data)
    for path, expected in hashes.items():
        assert r.sha(path) == expected
    archive = ROOT / 'review_bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(OUT, arcname='objectwise_repair_20261007')
    record = {'status': 'PASS', 'path': str(archive), 'sha256': r.sha(archive), 'bytes': archive.stat().st_size,
              'local_points': len(indices), 'full_points': len(xyz), 'record_stride_bytes': DTYPE.itemsize}
    f.atomic_json(ROOT / 'review_bundle.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
