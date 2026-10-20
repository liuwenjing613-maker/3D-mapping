#!/usr/bin/env python3
"""Color a new instance surface with colors inherited from an older map.

Both surfaces must use the same ordered TSDF points. A one-to-one maximum-IoU
matching transfers old colors; remaining IDs get spatially contrasting colors.
No GT or semantic labels are used.
"""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path
import os

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_surface(path):
    with np.load(path, allow_pickle=False) as d:
        return d['xyz_m'], d['instance_id']


def make_color_candidates():
    # Quantize now so the palette optimized here equals colors stored in PLY.
    colors = []
    for sat, val in ((0.78, 0.98), (0.92, 0.78), (0.60, 0.85)):
        for i in range(360):
            rgb = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, sat, val)
            colors.append(np.round(np.asarray(rgb) * 255) / 255)
    return np.asarray(colors, np.float64)


def lab(rgb):
    arr = np.asarray(rgb, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-surface', type=Path, required=True)
    ap.add_argument('--reference-surface', type=Path, required=True)
    ap.add_argument('--reference-colored-ply', type=Path, required=True)
    ap.add_argument('--output-ply', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--minimum-iou', type=float, default=0.12)
    ap.add_argument('--minimum-target-overlap', type=float, default=0.30)
    args = ap.parse_args()
    if not (0 < args.minimum_iou <= 1 and 0 < args.minimum_target_overlap <= 1):
        raise ValueError('Invalid matching thresholds')
    xyz, target_id = load_surface(args.target_surface)
    ref_xyz, reference_id = load_surface(args.reference_surface)
    if not np.array_equal(xyz, ref_xyz):
        raise ValueError('Surfaces do not contain the same ordered TSDF points')
    ref_cloud = o3d.io.read_point_cloud(str(args.reference_colored_ply))
    ref_ply_xyz = np.asarray(ref_cloud.points)
    reference_colors = np.asarray(ref_cloud.colors)
    if len(ref_ply_xyz) != len(xyz) or not np.allclose(ref_ply_xyz, xyz, atol=1e-7, rtol=0):
        raise ValueError('Reference PLY coordinates do not match the reference labels')
    if reference_colors.shape != xyz.shape:
        raise ValueError('Reference PLY has no RGB colors')

    target_ids = np.unique(target_id[target_id > 0])
    reference_ids, first = np.unique(reference_id[reference_id > 0], return_index=True)
    positive_ref_indices = np.flatnonzero(reference_id > 0)
    ref_color = np.zeros((int(reference_id.max()) + 1, 3), np.float64)
    ref_color[reference_ids] = reference_colors[positive_ref_indices[first]]
    if np.max(np.abs(reference_colors[reference_id > 0] - ref_color[reference_id[reference_id > 0]])) > 1e-7:
        raise ValueError('Reference PLY does not use a constant color per instance')
    gray_indices = np.flatnonzero(reference_id <= 0)
    gray = reference_colors[gray_indices[0]] if len(gray_indices) else np.round(np.array([0.35] * 3) * 255) / 255

    n_target = int(target_id.max()) + 1
    n_ref = int(reference_id.max()) + 1
    target_size = np.bincount(target_id[target_id > 0], minlength=n_target)
    ref_size = np.bincount(reference_id[reference_id > 0], minlength=n_ref)
    both = (target_id > 0) & (reference_id > 0)
    pair_keys, overlap_count = np.unique(
        target_id[both].astype(np.int64) * n_ref + reference_id[both].astype(np.int64),
        return_counts=True,
    )
    pair_target, pair_ref = np.divmod(pair_keys, n_ref)
    iou = overlap_count / (target_size[pair_target] + ref_size[pair_ref] - overlap_count)
    target_overlap = overlap_count / target_size[pair_target]
    eligible = (iou >= args.minimum_iou) & (target_overlap >= args.minimum_target_overlap)
    scores = np.zeros((n_target, n_ref), dtype=np.float32)
    scores[pair_target[eligible], pair_ref[eligible]] = iou[eligible]
    rows, cols = linear_sum_assignment(-scores[np.ix_(target_ids, reference_ids)])
    matches = []
    target_colors = np.zeros((n_target, 3), dtype=np.float64)
    assigned = np.zeros(n_target, dtype=bool)
    for row, col in zip(rows, cols):
        tid, rid = int(target_ids[row]), int(reference_ids[col])
        score = float(scores[tid, rid])
        if score <= 0:
            continue
        target_colors[tid] = ref_color[rid]
        assigned[tid] = True
        index = np.searchsorted(pair_keys, np.int64(tid) * n_ref + rid)
        matches.append({'target_instance_id': tid, 'reference_instance_id': rid,
                        'iou': score, 'shared_points': int(overlap_count[index]),
                        'target_points': int(target_size[tid]), 'reference_points': int(ref_size[rid])})

    # Place unmatched colors far from nearby already-colored object centroids.
    xyz64 = xyz.astype(np.float64, copy=False)
    centroids = np.column_stack([
        np.bincount(target_id[target_id > 0], weights=xyz64[target_id > 0, axis], minlength=n_target)
        / np.maximum(target_size, 1) for axis in range(3)
    ])
    tree = cKDTree(centroids[target_ids])
    candidates = make_color_candidates()
    candidate_lab = lab(candidates)
    used = np.zeros(len(candidates), dtype=bool)
    unmatched = sorted((int(tid) for tid in target_ids if not assigned[tid]),
                       key=lambda tid: (-target_size[tid], tid))
    for tid in unmatched:
        k = min(17, len(target_ids))
        _, neighbor_positions = tree.query(centroids[tid], k=k)
        neighbors = np.atleast_1d(target_ids[neighbor_positions])
        neighbors = [int(other) for other in neighbors if other != tid and assigned[int(other)]][:12]
        if neighbors:
            other_lab = lab(target_colors[neighbors])
            color_distance = np.linalg.norm(candidate_lab[:, None, :] - other_lab[None, :, :], axis=2)
            scores_color = np.min(color_distance, axis=1)
        else:
            scores_color = np.full(len(candidates), 1.0)
        scores_color[used] = -1
        choice = int(np.argmax(scores_color))
        target_colors[tid] = candidates[choice]
        used[choice] = True
        assigned[tid] = True

    colored = np.empty((len(xyz), 3), dtype=np.float64)
    colored[:] = gray
    labeled = target_id > 0
    colored[labeled] = target_colors[target_id[labeled]]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colored)
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_ply.with_name(args.output_ply.stem + '.tmp.ply')
    if not o3d.io.write_point_cloud(str(temp), cloud, write_ascii=False):
        raise OSError(f'Could not write {temp}')
    os.replace(temp, args.output_ply)
    check = o3d.io.read_point_cloud(str(args.output_ply))
    if len(check.points) != len(xyz) or len(check.colors) != len(xyz):
        raise ValueError('Written PLY point/color count differs from target')
    if np.max(np.abs(np.asarray(check.colors) - colored)) > 1 / 255 + 1e-8:
        raise ValueError('Written PLY colors differ beyond 8-bit quantization')

    matched_ids = np.asarray([row['target_instance_id'] for row in matches], dtype=np.int32)
    matched_point_count = int(target_size[matched_ids].sum()) if len(matched_ids) else 0
    report = {
        'status': 'PASS', 'target_surface': str(args.target_surface),
        'target_surface_sha256': sha256(args.target_surface),
        'reference_surface': str(args.reference_surface),
        'reference_surface_sha256': sha256(args.reference_surface),
        'reference_colored_ply': str(args.reference_colored_ply),
        'reference_colored_ply_sha256': sha256(args.reference_colored_ply),
        'output_ply': str(args.output_ply), 'output_ply_sha256': sha256(args.output_ply),
        'point_count': len(xyz), 'labeled_point_count': int(labeled.sum()),
        'target_instance_count': len(target_ids), 'reference_instance_count': len(reference_ids),
        'matched_instance_count': len(matches), 'matched_target_point_count': matched_point_count,
        'matched_target_point_fraction': matched_point_count / int(labeled.sum()),
        'unmatched_instance_count': len(unmatched),
        'minimum_iou': args.minimum_iou,
        'minimum_target_overlap': args.minimum_target_overlap,
        'matching': 'one-to-one maximum IoU on identical TSDF points',
        'unmatched_colors': 'deterministic spatial contrast HSV palette',
        'ground_truth_used': False, 'matches': matches,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'output_ply', 'point_count',
          'target_instance_count', 'matched_instance_count', 'matched_target_point_fraction')}), flush=True)


if __name__ == '__main__':
    main()
