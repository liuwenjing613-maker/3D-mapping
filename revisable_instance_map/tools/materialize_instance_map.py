#!/usr/bin/env python3
"""Materialize a shared-TSDF surface with rebuildable instance ownership."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource  # noqa: E402
from revisable_instance_map.observations import RawInstanceObservation, source_pixel_indices  # noqa: E402


KEY_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("id", "<i4")])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sample_voxels(frame, observation, voxel_size_m, max_pixels):
    pixels = source_pixel_indices(frame, observation)
    count = min(len(pixels), max_pixels)
    pixels = pixels[np.linspace(0, len(pixels) - 1, num=count, dtype=np.int64)]
    rows, cols = np.divmod(pixels, frame.camera.width)
    depth = frame.depth_m.ravel()[pixels].astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.int32), 0
    rows, cols, depth = rows[valid], cols[valid], depth[valid]
    camera_xyz = np.column_stack((
        (cols - frame.camera.cx) * depth / frame.camera.fx,
        (rows - frame.camera.cy) * depth / frame.camera.fy,
        depth,
    ))
    world_xyz = camera_xyz @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
    voxels = np.unique(np.floor(world_xyz / voxel_size_m).astype(np.int32), axis=0)
    return voxels, len(depth)


def pack_pairs(voxels, instance_id):
    keys = np.empty(len(voxels), dtype=KEY_DTYPE)
    keys["x"], keys["y"], keys["z"] = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    keys["id"] = instance_id
    return keys


def merge_vote_chunks(paths):
    all_keys, all_counts = [], []
    for path in paths:
        with np.load(path) as data:
            all_keys.append(data["keys"])
            all_counts.append(data["counts"])
    keys = np.concatenate(all_keys)
    counts = np.concatenate(all_counts)
    order = np.argsort(keys, kind="mergesort")
    keys, counts = keys[order], counts[order]
    unique_keys, starts = np.unique(keys, return_index=True)
    votes = np.add.reduceat(counts, starts).astype(np.int32)
    return unique_keys, votes


def choose_voxel_owners(keys, votes):
    if not len(keys):
        raise ValueError("No support voxels were produced")
    change = (
        (keys["x"][1:] != keys["x"][:-1])
        | (keys["y"][1:] != keys["y"][:-1])
        | (keys["z"][1:] != keys["z"][:-1])
    )
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], len(keys)]
    group_sizes = ends - starts
    max_votes = np.maximum.reduceat(votes, starts)
    total_votes = np.add.reduceat(votes, starts)
    top = votes == np.repeat(max_votes, group_sizes)
    top_count = np.add.reduceat(top.astype(np.int32), starts)
    first_top = np.minimum.reduceat(np.where(top, np.arange(len(keys)), len(keys)), starts)
    owners = keys["id"][first_top].astype(np.int32)
    owners[top_count > 1] = -1
    coordinates = np.column_stack((keys["x"][starts], keys["y"][starts], keys["z"][starts]))
    return coordinates, owners, max_votes, total_votes, top_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--tsdf-surface", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels-per-observation", type=int, default=1024)
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--label-distance-m", type=float, default=0.03)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--frame-count", type=int, default=400)
    args = parser.parse_args()
    if args.max_pixels_per_observation < 1 or args.voxel_size_m <= 0 or args.label_distance_m <= 0:
        raise ValueError("Invalid sampling or geometry parameters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ReplicaFrameSource(args.config)
    selected_frame_ids = source.frame_ids[:args.frame_count]
    if len(selected_frame_ids) != args.frame_count or args.frame_count < 1:
        raise ValueError("Invalid prefix frame count")
    observations = defaultdict(list)
    for line in args.observations.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        observations[int(row["frame_id"])].append(RawInstanceObservation(**row))
    assignment = {}
    for line in args.associations.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["observation_id"] in assignment:
            raise ValueError("Duplicate association decision")
        assignment[row["observation_id"]] = int(row["instance_id"])
    if len(assignment) != sum(map(len, observations.values())):
        raise ValueError("Assignment and observation counts differ")

    started = time.perf_counter()
    chunk_pairs = []
    chunk_paths = []
    sampled_pixels = 0
    observations_processed = 0
    for frame_index, frame_id in enumerate(selected_frame_ids, start=1):
        frame = source.load_frame(frame_id)
        mask_path = source.mask_root / source.config["source"]["mask_pattern"].format(frame=frame_id)
        digest = sha256_file(mask_path)
        for observation in observations[frame_id]:
            if observation.source_mask_sha256 != digest:
                raise ValueError(f"Mask checksum mismatch in frame {frame_id}")
            instance_id = assignment[observation.observation_id]
            voxels, count = sample_voxels(
                frame, observation, args.voxel_size_m, args.max_pixels_per_observation
            )
            sampled_pixels += count
            observations_processed += 1
            if len(voxels):
                chunk_pairs.append(pack_pairs(voxels, instance_id))
        if frame_index % args.chunk_frames == 0 or frame_index == len(selected_frame_ids):
            keys, counts = np.unique(np.concatenate(chunk_pairs), return_counts=True)
            chunk_path = args.output_dir / f"vote_chunk_{len(chunk_paths):02d}.npz"
            np.savez_compressed(chunk_path, keys=keys, counts=counts.astype(np.int32))
            chunk_paths.append(chunk_path)
            chunk_pairs.clear()
            progress = {
                "frames_processed": frame_index,
                "observations_processed": observations_processed,
                "chunks": len(chunk_paths),
                "sampled_pixels": sampled_pixels,
                "elapsed_seconds": time.perf_counter() - started,
            }
            write_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)

    keys, votes = merge_vote_chunks(chunk_paths)
    coordinates, owners, winning_votes, total_votes, top_count = choose_voxel_owners(keys, votes)
    vote_path = args.output_dir / "surface_support_votes_1cm.npz"
    np.savez_compressed(
        vote_path, voxel_coordinates=coordinates, owner_instance_id=owners,
        winning_votes=winning_votes, total_votes=total_votes, tie_count=top_count,
        voxel_size_m=np.array([args.voxel_size_m], dtype=np.float32),
    )
    print(f"Aggregated {len(coordinates)} spatial voxels", flush=True)

    surface = o3d.io.read_point_cloud(str(args.tsdf_surface))
    xyz = np.asarray(surface.points, dtype=np.float32)
    rgb = np.asarray(surface.colors, dtype=np.float32)
    if len(xyz) == 0 or len(xyz) != len(rgb):
        raise ValueError("TSDF surface is empty or lacks colors")
    centers = (coordinates.astype(np.float64) + 0.5) * args.voxel_size_m
    distances, nearest = cKDTree(centers).query(xyz, k=1, workers=-1)
    close = distances < args.label_distance_m
    surface_labels = np.where(close, owners[nearest], -1).astype(np.int32)
    result_path = args.output_dir / "instance_surface.npz"
    np.savez_compressed(result_path, xyz_m=xyz, rgb=rgb, instance_id=surface_labels)
    del surface

    largest_id = int(max(assignment.values()))
    palette = np.random.default_rng(7).uniform(0.2, 1.0, size=(largest_id + 1, 3))
    visual_colors = np.full((len(surface_labels), 3), 0.35)
    labeled = surface_labels > 0
    visual_colors[labeled] = palette[surface_labels[labeled]]
    visual = o3d.geometry.PointCloud()
    visual.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    visual.colors = o3d.utility.Vector3dVector(visual_colors)
    visual_path = args.output_dir / "instance_surface_colored.ply"
    if not o3d.io.write_point_cloud(str(visual_path), visual):
        raise RuntimeError("Could not save visualized instance surface")

    report = {
        "status": "PASS",
        "purpose": "rebuildable_instance_attribution_on_shared_tsdf_surface",
        "config_sha256": sha256_file(args.config),
        "observation_catalog_sha256": sha256_file(args.observations),
        "association_decisions_sha256": sha256_file(args.associations),
        "tsdf_surface_sha256": sha256_file(args.tsdf_surface),
        "frame_count": len(selected_frame_ids),
        "observation_count": observations_processed,
        "sampled_valid_pixels": sampled_pixels,
        "max_pixels_per_observation": args.max_pixels_per_observation,
        "support_voxel_size_m": args.voxel_size_m,
        "label_distance_m": args.label_distance_m,
        "support_instance_voxel_pairs": len(keys),
        "spatial_support_voxels": len(coordinates),
        "tied_owner_voxels": int(np.count_nonzero(owners < 0)),
        "tied_owner_voxel_fraction": float(np.mean(owners < 0)),
        "tsdf_surface_points": len(xyz),
        "labeled_surface_points": int(np.count_nonzero(labeled)),
        "labeled_surface_fraction": float(np.mean(labeled)),
        "nearest_support_distance_median_m": float(np.median(distances)),
        "nearest_support_distance_p90_m": float(np.percentile(distances, 90)),
        "surface_points_within_2cm": float(np.mean(distances < 0.02)),
        "surface_points_within_3cm": float(np.mean(distances < 0.03)),
        "surface_points_within_5cm": float(np.mean(distances < 0.05)),
        "distinct_surface_instance_ids": int(len(np.unique(surface_labels[labeled]))),
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "files": {
            "support_votes": str(vote_path),
            "instance_surface": str(result_path),
            "colored_ply": str(visual_path),
        },
    }
    write_json(args.output_dir / "materialization_report.json", report)
    print(f"PASS: {report['labeled_surface_points']}/{len(xyz)} TSDF points labeled in {report['total_seconds']:.1f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

