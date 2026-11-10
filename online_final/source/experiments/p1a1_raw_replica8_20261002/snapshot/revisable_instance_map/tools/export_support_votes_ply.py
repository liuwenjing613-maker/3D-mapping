#!/usr/bin/env python3
"""Export 1 cm vote evidence as colored PLY with owner and vote properties.

The owner colors are copied from a selected instance-surface PLY, so support and
final labeling can be compared directly. No GT is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--votes', type=Path, required=True)
    parser.add_argument('--instance-surface', type=Path, required=True)
    parser.add_argument('--instance-colored-ply', type=Path, required=True)
    parser.add_argument('--output-ply', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.votes, allow_pickle=False) as d:
        voxels = d['voxel_coordinates']
        owners = d['owner_instance_id']
        winning = d['winning_votes']
        total = d['total_votes']
        ties = d['tie_count']
        size = float(d['voxel_size_m'][0])
    n = len(voxels)
    if voxels.shape != (n, 3) or any(len(x) != n for x in (owners, winning, total, ties)):
        raise ValueError('Vote arrays differ in length')
    if size <= 0 or not np.all((winning > 0) & (total >= winning) & (ties >= 1)):
        raise ValueError('Invalid vote values')
    with np.load(args.instance_surface, allow_pickle=False) as d:
        surface_xyz, surface_ids = d['xyz_m'], d['instance_id']
    surface = o3d.io.read_point_cloud(str(args.instance_colored_ply))
    if len(surface.points) != len(surface_xyz) or len(surface.colors) != len(surface_xyz):
        raise ValueError('Instance colored PLY and instance surface differ')
    if not np.allclose(np.asarray(surface.points), surface_xyz, atol=1e-7, rtol=0):
        raise ValueError('Instance PLY points differ from the label array')
    old_colors = np.asarray(surface.colors)
    positive = surface_ids > 0
    IDs, first = np.unique(surface_ids[positive], return_index=True)
    first_point = np.flatnonzero(positive)[first]
    max_id = max(int(owners.max()), int(surface_ids.max()))
    # Existing baseline palette is the deterministic fallback for IDs absent
    # from the final surface but still present in the vote evidence.
    palette = np.random.default_rng(7).uniform(0.2, 1.0, size=(max_id + 1, 3))
    palette[IDs] = old_colors[first_point]
    if np.max(np.abs(old_colors[positive] - palette[surface_ids[positive]])) > 1e-7:
        raise ValueError('Instance PLY does not have a constant color per ID')
    unlabeled = np.flatnonzero(~positive)
    gray = old_colors[unlabeled[0]] if len(unlabeled) else np.array([0.35] * 3)
    rgb = np.empty((n, 3), np.float64)
    rgb[:] = gray
    selected = owners > 0
    rgb[selected] = palette[owners[selected]]
    rgb8 = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    xyz = (voxels.astype(np.float64) + 0.5) * size

    record = np.empty(n, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('instance_id', '<i4'), ('winning_votes', '<i4'),
        ('total_votes', '<i4'), ('tie_count', '<i4'),
    ]))
    record['x'], record['y'], record['z'] = xyz.T
    record['red'], record['green'], record['blue'] = rgb8.T
    record['instance_id'] = owners
    record['winning_votes'] = winning
    record['total_votes'] = total
    record['tie_count'] = ties
    header = f'''ply
format binary_little_endian 1.0
comment instance_id -1 means an unresolved ownership tie
comment positions are 1 cm support voxel centers in world coordinates
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int instance_id
property int winning_votes
property int total_votes
property int tie_count
end_header
'''
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_ply.with_suffix('.ply.tmp')
    with temp.open('wb') as stream:
        stream.write(header.encode('ascii'))
        record.tofile(stream)
    os.replace(temp, args.output_ply)
    check = o3d.io.read_point_cloud(str(args.output_ply))
    if len(check.points) != n or len(check.colors) != n:
        raise ValueError('PLY readback failed')
    if np.max(np.abs(np.asarray(check.points) - xyz)) > 2e-6:
        raise ValueError('PLY point coordinates changed beyond float32 precision')
    if not np.array_equal(np.rint(np.asarray(check.colors) * 255).astype(np.uint8), rgb8):
        raise ValueError('PLY point colors differ from assigned palette')
    support_only = np.setdiff1d(np.unique(owners[selected]), IDs)
    report = {
        'status': 'PASS', 'votes': str(args.votes), 'votes_sha256': sha256(args.votes),
        'instance_surface': str(args.instance_surface),
        'instance_colored_ply': str(args.instance_colored_ply),
        'instance_colored_ply_sha256': sha256(args.instance_colored_ply),
        'output_ply': str(args.output_ply), 'output_ply_sha256': sha256(args.output_ply),
        'voxel_size_m': size, 'support_points': n, 'positive_owner_points': int(selected.sum()),
        'tie_points': int((owners < 0).sum()),
        'support_only_instance_count': len(support_only),
        'support_only_point_count': int(np.isin(owners, support_only).sum()),
        'additional_properties': ['instance_id', 'winning_votes', 'total_votes', 'tie_count'],
        'ground_truth_used': False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'output_ply', 'support_points',
          'positive_owner_points', 'tie_points', 'support_only_instance_count')}), flush=True)


if __name__ == '__main__':
    main()
