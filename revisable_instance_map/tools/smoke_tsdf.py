#!/usr/bin/env python3
"""Build a small shared TSDF from an empty map and inspect geometry only."""

import argparse
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
from revisable_instance_map.geometry_tsdf import SparseTSDFGeometry  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--reference", type=Path, help="Geometry-only QA; not passed to mapper")
    args = parser.parse_args()
    if args.frame_count < 1:
        raise ValueError("frame-count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ReplicaFrameSource(args.config)
    frame_ids = source.frame_ids[: args.frame_count]
    geometry = SparseTSDFGeometry(block_count=20000)
    started = time.perf_counter()
    integration = []
    for frame_id in frame_ids:
        integration.append(geometry.integrate(source.load_frame(frame_id)))

    point_cloud = geometry.extract_point_cloud(weight_threshold=1.0)
    positions = point_cloud.point.positions.numpy()
    colors = point_cloud.point.colors.numpy()
    export_cloud = point_cloud.to_legacy()
    export_colors = np.asarray(export_cloud.colors)
    export_colors[:] = np.clip(export_colors, 0.0, 1.0)
    o3d.io.write_point_cloud(str(args.output_dir / "surface.ply"), export_cloud)
    geometry.save(args.output_dir / "tsdf_grid.npz")

    first = source.load_frame(frame_ids[0])
    rendered = np.squeeze(geometry.raycast_depth(first, weight_threshold=1.0))
    if rendered.shape != first.depth_m.shape:
        raise ValueError(f"Raycast shape {rendered.shape} differs from depth shape")
    np.save(args.output_dir / "raycast_depth_frame0_m.npy", rendered)
    valid_render = np.isfinite(rendered) & (rendered > 0) & (rendered < geometry.depth_max_m)
    valid_observed = np.isfinite(first.depth_m) & (first.depth_m > 0)
    common = valid_render & valid_observed
    depth_errors = np.abs(rendered[common] - first.depth_m[common])

    report = {
        "status": "PENDING",
        "purpose": "small_empty_map_shared_geometry_only",
        "config_sha256": sha256_file(args.config),
        "open3d_version": o3d.__version__,
        "device": "CPU:0",
        "voxel_size_m": geometry.voxel_size_m,
        "block_resolution": 8,
        "block_count_capacity": 20000,
        "depth_max_m": geometry.depth_max_m,
        "frame_ids": list(frame_ids),
        "integration": integration,
        "active_blocks": int(geometry.grid.hashmap().size()),
        "surface_points": int(len(positions)),
        "surface_color_min": colors.min(axis=0).tolist() if len(colors) else None,
        "surface_color_max": colors.max(axis=0).tolist() if len(colors) else None,
        "surface_color_out_of_range_fraction": float(np.mean(np.any((colors < 0) | (colors > 1), axis=1))) if len(colors) else None,
        "surface_bounds_min_m": positions.min(axis=0).tolist() if len(positions) else None,
        "surface_bounds_max_m": positions.max(axis=0).tolist() if len(positions) else None,
        "raycast_valid_fraction_of_image": float(valid_render.mean()),
        "raycast_common_pixels": int(common.sum()),
        "raycast_abs_depth_error_median_m": float(np.median(depth_errors)) if len(depth_errors) else None,
        "raycast_abs_depth_error_p90_m": float(np.percentile(depth_errors, 90)) if len(depth_errors) else None,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "reference_geometry_for_qa_only": str(args.reference) if args.reference else None,
    }
    if args.reference and len(positions):
        with np.load(args.reference) as stored:
            reference = stored["xyz"]
        sample = positions[:: max(1, len(positions) // 50000)]
        distances, _ = cKDTree(reference).query(sample, workers=-1)
        report["reference_mesh_distance_median_m"] = float(np.median(distances))
        report["reference_mesh_distance_p90_m"] = float(np.percentile(distances, 90))
        report["reference_mesh_fraction_within_5cm"] = float(np.mean(distances < 0.05))
        report["reference_sha256"] = sha256_file(args.reference)

    report["status"] = (
        "PASS"
        if report["surface_points"] > 0
        and report["raycast_common_pixels"] > 0
        and report["raycast_abs_depth_error_median_m"] is not None
        and report["raycast_abs_depth_error_median_m"] < 0.05
        else "FAIL"
    )
    output = args.output_dir / "geometry_smoke.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(frame_ids)} frames, {report['surface_points']} surface points; {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

