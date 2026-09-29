#!/usr/bin/env python3
"""P0: attribute historical RGB-D masks directly to the shared TSDF surface.

Association, front-end masks, and TSDF geometry are immutable inputs. No GT is
read. Per-frame region support is saved before applying persistent instance IDs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.observations import RawInstanceObservation
from revisable_instance_map.surface_evidence import (
    STATE_NAMES, UNOBSERVED, TENTATIVE, CONFIRMED, CONFLICT,
    project_frame_regions, frame_instance_keys, reduce_surface_votes,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_colored_ply(path, xyz, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f'Could not write {path}')


def load_observations(path, selected_ids):
    by_frame = defaultdict(list)
    observation_ids = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['frame_id'] not in selected_ids:
            continue
        obs = RawInstanceObservation(**row)
        if obs.observation_id in observation_ids:
            raise ValueError('Duplicate observation ID')
        observation_ids.add(obs.observation_id)
        by_frame[obs.frame_id].append(obs)
    return by_frame, observation_ids


def load_assignments(path, observation_ids):
    assigned = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        obs_id = row['observation_id']
        if obs_id in observation_ids:
            if obs_id in assigned or row['instance_id'] <= 0:
                raise ValueError('Duplicate or invalid association')
            assigned[obs_id] = int(row['instance_id'])
    if set(assigned) != observation_ids:
        raise ValueError('Associations do not cover selected observations exactly')
    return assigned


def validate_frame_and_lookup(frame, frame_observations, assigned, mask_digest):
    max_local = int(frame.mask_local.max())
    lookup = np.full(max_local + 1, -1, dtype=np.int32)
    counts = np.bincount(frame.mask_local.ravel(), minlength=max_local + 1)
    projectable = np.isfinite(frame.depth_m) & (frame.depth_m > 0) & (frame.depth_m < 10.0)
    projectable_counts = np.bincount(frame.mask_local[projectable].ravel(), minlength=max_local + 1)
    seen = set()
    for obs in frame_observations:
        local_id = obs.mask_local_id
        if local_id <= 0 or local_id > max_local or local_id in seen:
            raise ValueError(f'Invalid local ID in frame {frame.frame_id}')
        if (obs.source_mask_sha256 != mask_digest or
            obs.pixel_count != int(counts[local_id]) or
            obs.projectable_pixel_count != int(projectable_counts[local_id])):
            raise ValueError(f'Observation source differs in frame {frame.frame_id}')
        lookup[local_id] = assigned[obs.observation_id]
        seen.add(local_id)
    if seen != set(np.flatnonzero(counts[1:]) + 1):
        raise ValueError(f'Unaccounted mask pixels in frame {frame.frame_id}')
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--associations', type=Path, required=True)
    ap.add_argument('--tsdf-surface', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--frame-count', type=int, default=400)
    ap.add_argument('--pixel-stride', type=int, default=2)
    ap.add_argument('--max-surface-distance-m', type=float, default=0.015)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk-frames', type=int, default=25)
    ap.add_argument('--min-confirmed-votes', type=int, default=2)
    ap.add_argument('--min-confirmed-ratio', type=float, default=0.67)
    args = ap.parse_args()
    if args.frame_count < 1 or args.pixel_stride < 1 or args.chunk_frames < 1:
        raise ValueError('Invalid frame, stride, or chunk count')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ReplicaFrameSource(args.config)
    frame_ids = source.frame_ids[:args.frame_count]
    if len(frame_ids) != args.frame_count:
        raise ValueError('Frame prefix exceeds fixed protocol')
    observations, observation_ids = load_observations(args.observations, set(frame_ids))
    assigned = load_assignments(args.associations, observation_ids)
    instance_base = max(assigned.values()) + 1

    surface = o3d.io.read_point_cloud(str(args.tsdf_surface))
    xyz = np.asarray(surface.points, dtype=np.float32)
    rgb = np.asarray(surface.colors, dtype=np.float32)
    if len(xyz) == 0 or rgb.shape != xyz.shape:
        raise ValueError('TSDF surface is empty or lacks RGB')
    tree = cKDTree(xyz.astype(np.float64))
    frame_support_dir = args.output_dir / 'frame_region_support'
    frame_support_dir.mkdir(exist_ok=True)
    frame_manifest_path = args.output_dir / 'frame_region_support_manifest.jsonl'
    frame_manifest_temp = frame_manifest_path.with_suffix('.jsonl.tmp')
    chunk_paths, chunk_arrays = [], []
    stats = Counter()
    started = time.perf_counter()
    with frame_manifest_temp.open('w', encoding='utf-8') as manifest:
        for frame_index, frame_id in enumerate(frame_ids, start=1):
            frame = source.load_frame(frame_id)
            mask_path = source.mask_root / source.config['source']['mask_pattern'].format(frame=frame_id)
            digest = sha256_file(mask_path)
            lookup = validate_frame_and_lookup(frame, observations[frame_id], assigned, digest)
            points, local_ids, projection_stats = project_frame_regions(
                frame, tree, args.pixel_stride, args.max_surface_distance_m, args.workers)
            support_path = frame_support_dir / f'f{frame_id:06d}.npz'
            np.savez_compressed(
                support_path, frame_id=np.asarray([frame_id], np.int32),
                surface_point_index=points, mask_local_id=local_ids,
                source_mask_sha256=np.asarray([digest]),
                pixel_stride=np.asarray([args.pixel_stride], np.int32),
                max_surface_distance_m=np.asarray([args.max_surface_distance_m], np.float64),
            )
            keys = frame_instance_keys(points, local_ids, lookup, instance_base)
            chunk_arrays.append(keys)
            point_ids = keys // instance_base
            stats['same_frame_multi_id_points'] += int(np.count_nonzero(point_ids[1:] == point_ids[:-1]))
            stats['eligible_pixels'] += projection_stats['eligible_pixels']
            stats['matched_pixels'] += projection_stats['matched_pixels']
            stats['frame_region_surface_pairs'] += len(points)
            stats['frame_instance_surface_votes'] += len(keys)
            record = dict(frame_id=frame_id, observation_count=len(observations[frame_id]),
                          frame_instance_votes=len(keys), support_file=str(support_path),
                          support_sha256=sha256_file(support_path), **projection_stats)
            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
            if frame_index % args.chunk_frames == 0 or frame_index == len(frame_ids):
                all_keys = np.concatenate(chunk_arrays)
                keys_chunk, counts_chunk = np.unique(all_keys, return_counts=True)
                chunk_path = args.output_dir / f'surface_vote_chunk_{len(chunk_paths):02d}.npz'
                np.savez_compressed(chunk_path, keys=keys_chunk,
                                    counts=counts_chunk.astype(np.int32))
                chunk_paths.append(chunk_path)
                chunk_arrays.clear()
                progress = {'frames_processed': frame_index, 'last_frame_id': frame_id,
                            'matched_pixels': stats['matched_pixels'],
                            'frame_instance_surface_votes': stats['frame_instance_surface_votes'],
                            'elapsed_seconds': time.perf_counter() - started}
                write_json(args.output_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    frame_manifest_temp.replace(frame_manifest_path)

    all_keys, all_counts = [], []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as data:
            all_keys.append(data['keys'])
            all_counts.append(data['counts'])
    keys = np.concatenate(all_keys)
    counts = np.concatenate(all_counts)
    order = np.argsort(keys, kind='mergesort')
    keys, counts = keys[order], counts[order]
    unique_keys, starts = np.unique(keys, return_index=True)
    vote_counts = np.add.reduceat(counts, starts).astype(np.int32)
    evidence = reduce_surface_votes(unique_keys, vote_counts, len(xyz), instance_base,
                                    args.min_confirmed_votes, args.min_confirmed_ratio)
    evidence_path = args.output_dir / 'surface_evidence.npz'
    np.savez_compressed(evidence_path, xyz_m=xyz, rgb=rgb, **evidence)
    pair_path = args.output_dir / 'surface_instance_frame_votes.npz'
    np.savez_compressed(pair_path,
                        surface_point_index=(unique_keys // instance_base).astype(np.int32),
                        instance_id=(unique_keys % instance_base).astype(np.int32),
                        frame_votes=vote_counts)

    states = evidence['state']
    top1 = evidence['top1_instance_id']
    confirmed_labels = np.where(states == CONFIRMED, top1, -1).astype(np.int32)
    tentative_labels = np.where((states == CONFIRMED) | (states == TENTATIVE),
                                top1, -1).astype(np.int32)
    main_path = args.output_dir / 'instance_surface.npz'
    tentative_path = args.output_dir / 'instance_surface_tentative_included.npz'
    np.savez_compressed(main_path, xyz_m=xyz, rgb=rgb, instance_id=confirmed_labels)
    np.savez_compressed(tentative_path, xyz_m=xyz, rgb=rgb, instance_id=tentative_labels)
    palette = np.random.default_rng(7).uniform(0.2, 1.0, size=(instance_base, 3))
    map_colors = np.full((len(xyz), 3), 0.35, np.float32)
    labeled = confirmed_labels > 0
    map_colors[labeled] = palette[confirmed_labels[labeled]]
    map_ply = args.output_dir / 'instance_surface_colored.ply'
    write_colored_ply(map_ply, xyz, map_colors)
    state_colors = np.full((len(xyz), 3), 0.35, np.float32)
    confirmed = states == CONFIRMED
    state_colors[confirmed] = palette[top1[confirmed]]
    state_colors[states == TENTATIVE] = (1.0, 1.0, 1.0)
    state_colors[states == CONFLICT] = (1.0, 0.0, 0.0)
    state_ply = args.output_dir / 'surface_evidence_states.ply'
    write_colored_ply(state_ply, xyz, state_colors)

    state_counts = {name: int(np.count_nonzero(states == index))
                    for index, name in enumerate(STATE_NAMES)}
    report = {
        'status': 'PASS', 'purpose': 'surface_centric_multiview_instance_evidence_P0',
        'ground_truth_used': False, 'scene_id': source.config['scene'],
        'frame_count': len(frame_ids),
        'observation_count': len(observation_ids), 'tsdf_surface_points': len(xyz),
        'config_sha256': sha256_file(args.config),
        'observation_catalog_sha256': sha256_file(args.observations),
        'association_decisions_sha256': sha256_file(args.associations),
        'tsdf_surface_sha256': sha256_file(args.tsdf_surface),
        'pixel_stride': args.pixel_stride,
        'max_surface_distance_m': args.max_surface_distance_m,
        'min_confirmed_votes': args.min_confirmed_votes,
        'min_confirmed_ratio': args.min_confirmed_ratio,
        'state_counts': state_counts,
        'labeled_surface_points': int(np.count_nonzero(labeled)),
        'labeled_surface_fraction': float(np.mean(labeled)),
        'tentative_included_surface_points': int(np.count_nonzero(tentative_labels > 0)),
        'confirmed_surface_points': state_counts['confirmed'],
        'confirmed_surface_fraction': state_counts['confirmed'] / len(xyz),
        'surface_instance_pairs': len(unique_keys),
        'frame_region_support_manifest_sha256': sha256_file(frame_manifest_path),
        'surface_evidence_sha256': sha256_file(evidence_path),
        'instance_surface_sha256': sha256_file(main_path),
        'same_frame_multi_id_points': stats['same_frame_multi_id_points'],
        'peak_process_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'total_seconds': time.perf_counter() - started,
        'files': {'surface_evidence': str(evidence_path), 'instance_surface': str(main_path),
                  'tentative_included_surface': str(tentative_path),
                  'colored_ply': str(map_ply), 'state_ply': str(state_ply),
                  'surface_instance_frame_votes': str(pair_path),
                  'frame_region_support_manifest': str(frame_manifest_path)},
        'sample_stats': dict(stats),
    }
    write_json(args.output_dir / 'materialization_report.json', report)
    print(json.dumps({'status': 'PASS', 'states': state_counts,
                      'labeled_fraction': report['labeled_surface_fraction'],
                      'seconds': report['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
