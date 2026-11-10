#!/usr/bin/env python3
"""Repair immutable P0 ownership offline; never load GT or alter observation state."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from revisable_instance_map.offline_surface_assignment import AssignmentSettings, assign_surface, REJECTION_NAMES
    from revisable_instance_map.local_surface_decoder import DecoderSettings, fill_small_holes
except ImportError:
    from offline_surface_assignment import AssignmentSettings, assign_surface, REJECTION_NAMES
    from local_surface_decoder import DecoderSettings, fill_small_holes


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def write_ply(path, xyz, normals, colors, labels, state, route, keep=None):
    if keep is None:
        keep = np.arange(len(xyz))
    elif np.asarray(keep).dtype == bool:
        keep = np.flatnonzero(keep)
    vertex = np.empty(len(keep), dtype=[(k, '<f4') for k in ('x','y','z','nx','ny','nz')] +
                      [(k, 'u1') for k in ('red','green','blue')] +
                      [('instance_id','<i4'),('original_state','u1'),('assignment_route','u1')])
    for c, name in enumerate(('x','y','z')):
        vertex[name] = xyz[keep, c]
    for c, name in enumerate(('nx','ny','nz')):
        vertex[name] = normals[keep, c]
    for c, name in enumerate(('red','green','blue')):
        vertex[name] = np.rint(np.clip(colors[keep,c],0,1) * 255).astype(np.uint8)
    vertex['instance_id'], vertex['original_state'], vertex['assignment_route'] = labels[keep], state[keep], route[keep]
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False, byte_order='<', comments=[
        'P0 offline ownership; original_state: U=0 T=1 CONFIRMED=2 CONFLICT=3',
        'assignment_route: retained=0 original_confirmed=1 enclosed_hole=2 bounded_geodesic=3',
        'Original evidence and point order are unchanged; inferred identity is not new CONFIRMED evidence',
    ]).write(str(path))
    check = PlyData.read(str(path))['vertex'].data
    recovered = np.column_stack([check[k] for k in ('x','y','z')])
    if not np.array_equal(recovered, xyz[keep].astype(np.float32)) or not np.array_equal(check['instance_id'], labels[keep]):
        raise AssertionError('PLY coordinate or identity verification failed')
    return {'path': str(path), 'vertices': int(len(keep)), 'sha256': sha256(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source, out = args.source_dir, args.output_dir
    required = [source / f for f in ('surface_evidence.npz', 'surface_instance_frame_votes.npz',
                'instance_surface.npz', 'materialization_report.json', 'frame_region_support_manifest.jsonl')] + [args.geometry]
    hashes = {str(f): sha256(f) for f in required}
    parent = json.loads((source / 'materialization_report.json').read_text())
    frames = [json.loads(line)['frame_id'] for line in (source / 'frame_region_support_manifest.jsonl').read_text().splitlines()]
    if frames != list(range(0,2000,5)) or parent['frame_count'] != 400 or parent['ground_truth_used']:
        raise ValueError('P0 frame protocol or source lineage mismatch')
    if parent['tsdf_surface_sha256'] != hashes[str(args.geometry)]:
        raise ValueError('Source geometry hash differs from P0 report')
    with np.load(source / 'surface_evidence.npz', allow_pickle=False) as d:
        evidence = {k: d[k] for k in d.files}
    with np.load(source / 'surface_instance_frame_votes.npz', allow_pickle=False) as d:
        pair = {k: d[k] for k in d.files}
    with np.load(source / 'instance_surface.npz', allow_pickle=False) as d:
        xyz, rgb, original = d['xyz_m'].copy(), d['rgb'].copy(), d['instance_id'].copy()
    if not np.array_equal(xyz, evidence['xyz_m']) or not np.array_equal(rgb, evidence['rgb']):
        raise ValueError('Source map and evidence point order or color differ')
    geometry = o3d.io.read_point_cloud(str(args.geometry))
    if len(geometry.points) != len(xyz) or not geometry.has_normals():
        raise ValueError('Missing source normals or inconsistent geometry count')
    error = float(np.max(np.abs(np.asarray(geometry.points) - xyz)))
    if error > 1e-5:
        raise ValueError('Source geometry point order mismatch')
    normals = np.asarray(geometry.normals).astype(np.float32)
    settings = AssignmentSettings()
    def progress(key, value):
        print(json.dumps({'scene': parent['scene_id'], 'phase': key, 'value': value}), flush=True)
    variants, detail, stats = assign_surface(xyz, normals, rgb, evidence, pair, original, settings, progress)
    # Controls use the exact same frozen map, without the old confirmed-label
    # decoder, so all compared variants leave originally published IDs fixed.
    tree = cKDTree(xyz)
    assigned = np.flatnonzero(original > 0)
    target = detail['surface_point_index']
    nn = original.copy()
    if len(assigned) and len(target):
        distances, indices = cKDTree(xyz[assigned]).query(xyz[target], workers=8)
        take = distances <= stats['path_budget_m']
        nn[target[take]] = original[assigned[indices[take]]]
    local_fill, _ = fill_small_holes(xyz, normals, rgb, evidence, original, DecoderSettings(), tree)
    variants = {'baseline': original, 'distance_nn': nn, 'existing_local_fill': local_fill, **variants}
    maps = {}
    for name, labels in variants.items():
        if not np.array_equal(labels[original > 0], original[original > 0]):
            raise AssertionError('Control or repair changed original labels')
        destination = out / name
        destination.mkdir()
        map_file = destination / 'instance_surface.npz'
        if name == 'baseline':
            map_file.symlink_to((source / 'instance_surface.npz').resolve())
        else:
            np.savez_compressed(map_file, xyz_m=xyz, rgb=rgb, instance_id=labels)
        maps[name] = {'map_path': str(map_file), 'sha256': sha256(map_file),
                      'labeled_surface_points': int((labels > 0).sum()),
                      'newly_assigned_points': int(((original < 0) & (labels > 0)).sum()),
                      'original_published_labels_unchanged': True}
    final = variants['holes_geodesic']
    route = np.zeros(len(xyz), np.uint8)
    route[original > 0] = 1
    route[target] = detail['assignment_route']
    np.savez_compressed(out / 'assignment_diagnostics.npz', **detail)
    np.savez_compressed(out / 'point_provenance.npz', original_state=evidence['state'],
                        original_instance_id=original, final_instance_id=final, assignment_route=route)
    np.savez_compressed(out / 'repaired_points.npz', xyz_m=xyz[target[route[target] > 0]],
                        surface_point_index=target[route[target] > 0], original_state=evidence['state'][target[route[target] > 0]],
                        instance_id=final[target[route[target] > 0]])
    palette = np.random.default_rng(7).uniform(.2, 1., size=(int(original.max(initial=0)) + 1, 3))
    colors = np.full((len(xyz),3), .35, np.float32)
    positive = final > 0
    colors[positive] = palette[final[positive]]
    ply = {}
    ply['full'] = write_ply(out / f"{parent['scene_id']}_repaired_full.ply", xyz, normals, colors, final, evidence['state'], route)
    ply['assigned_only'] = write_ply(out / f"{parent['scene_id']}_repaired_assigned_only.ply", xyz, normals, colors, final, evidence['state'], route, positive)
    overlay = np.full((len(xyz),3), .25, np.float32)
    overlay[route == 2] = [0.1, 1., .2]
    overlay[route == 3] = [1., .70, .05]
    ply['repair_routes'] = write_ply(out / f"{parent['scene_id']}_repair_routes.ply", xyz, normals, overlay, final, evidence['state'], route)
    if parent['scene_id'] == 'room0':
        before = np.full((len(xyz),3), .35, np.float32)
        before[original > 0] = palette[original[original > 0]]
        ply['baseline'] = write_ply(out / 'room0_before.ply', xyz, normals, before, original, evidence['state'], np.where(original > 0,1,0).astype(np.uint8))
    if hashes != {str(f): sha256(f) for f in required}:
        raise AssertionError('Original input was modified')
    module_path = Path(sys.modules[assign_surface.__module__].__file__)
    setting_hash = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True, separators=(',',':')).encode()).hexdigest()
    report = {
        'status': 'PASS', 'purpose': 'offline_three_state_surface_assignment', 'scene_id': parent['scene_id'],
        'frame_count': 400, 'frame_list_sha256': hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest(),
        'tsdf_surface_points': len(xyz), 'ground_truth_used': False,
        'raw_evidence_and_source_maps_unchanged': True, 'geometry_and_point_order_unchanged': True,
        'input_geometry_coordinate_max_error_m': error, 'source_files_sha256': hashes,
        'settings_sha256': setting_hash, 'code_sha256': {str(module_path): sha256(module_path), str(Path(__file__)): sha256(Path(__file__))},
        'statistics': stats, 'variants': maps, 'ply': ply, 'rejection_codes': REJECTION_NAMES,
        'route_codes': {0:'retained',1:'original_confirmed',2:'enclosed_surface_hole',3:'bounded_surface_propagation'},
        'seconds': time.perf_counter() - started,
        'peak_process_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    dump(out / 'materialization_report.json', report)
    print(json.dumps({'scene':parent['scene_id'],'status':'PASS','by_state':stats['by_state'],
                      'variants':{k:v['newly_assigned_points'] for k,v in maps.items()},
                      'seconds':report['seconds']}),flush=True)


if __name__ == '__main__':
    main()
