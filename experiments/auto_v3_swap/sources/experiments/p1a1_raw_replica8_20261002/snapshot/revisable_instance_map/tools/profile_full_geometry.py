#!/usr/bin/env python3
"""Profile a full 400-frame shared TSDF built from an empty map."""

import argparse
import gc
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


def peak_rss_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Geometry QA only; never passed to mapper")
    parser.add_argument("--block-count", type=int, default=100000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    if args.block_count < 1 or args.checkpoint_every < 1:
        raise ValueError("block-count and checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ReplicaFrameSource(args.config)
    frame_ids = source.frame_ids
    if len(frame_ids) != 400:
        raise ValueError(f"Expected frozen 400 frames, found {len(frame_ids)}")
    geometry = SparseTSDFGeometry(block_count=args.block_count)
    started = time.perf_counter()
    integration = []
    checkpoints = []
    progress_path = args.output_dir / "progress.json"
    for index, frame_id in enumerate(frame_ids, start=1):
        frame = source.load_frame(frame_id)
        stats = geometry.integrate(frame)
        integration.append(stats)
        del frame
        if index % args.checkpoint_every == 0 or index == len(frame_ids):
            gc.collect()
            checkpoint = {
                "frames_integrated": index,
                "last_frame_id": frame_id,
                "active_blocks": stats["active_blocks"],
                "capacity_fraction": stats["active_blocks"] / args.block_count,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_mb": peak_rss_mb(),
            }
            checkpoints.append(checkpoint)
            write_json(progress_path, {"status": "INTEGRATING", "checkpoints": checkpoints})
            print(json.dumps(checkpoint), flush=True)

    cloud = geometry.extract_point_cloud(weight_threshold=1.0)
    positions = cloud.point.positions.numpy()
    colors = cloud.point.colors.numpy()
    print(f"Extracted {len(positions)} surface points", flush=True)
    export_cloud = cloud.to_legacy()
    export_colors = np.asarray(export_cloud.colors)
    export_colors[:] = np.clip(export_colors, 0.0, 1.0)
    cloud_path = args.output_dir / "surface.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), export_cloud):
        raise RuntimeError("Point cloud export failed")
    del export_cloud
    gc.collect()
    grid_path = args.output_dir / "tsdf_grid.npz"
    geometry.save(grid_path)
    print(f"Saved grid: {grid_path.stat().st_size / (1024**3):.2f} GiB", flush=True)

    raycasts = []
    for frame_id in (frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]):
        frame = source.load_frame(frame_id)
        rendered = np.squeeze(geometry.raycast_depth(frame, weight_threshold=1.0))
        if rendered.shape != frame.depth_m.shape:
            raise ValueError(f"Raycast shape mismatch for frame {frame_id}")
        valid_render = np.isfinite(rendered) & (rendered > 0) & (rendered < geometry.depth_max_m)
        valid_observed = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        common = valid_render & valid_observed
        errors = np.abs(rendered[common] - frame.depth_m[common])
        raycasts.append({
            "frame_id": frame_id,
            "valid_render_fraction_of_image": float(valid_render.mean()),
            "valid_observed_fraction_of_image": float(valid_observed.mean()),
            "common_pixels": int(common.sum()),
            "abs_depth_error_median_m": float(np.median(errors)) if len(errors) else None,
            "abs_depth_error_p90_m": float(np.percentile(errors, 90)) if len(errors) else None,
        })
        print(json.dumps(raycasts[-1]), flush=True)
        del frame, rendered, errors
        gc.collect()

    reference_qa = None
    if args.reference and len(positions):
        with np.load(args.reference) as stored:
            reference = stored["xyz"]
        sample = positions[::max(1, len(positions) // 50000)]
        distances, _ = cKDTree(reference).query(sample, workers=-1)
        reference_qa = {
            "reference_path": str(args.reference),
            "reference_sha256": sha256_file(args.reference),
            "surface_sample_count": int(len(sample)),
            "nearest_distance_median_m": float(np.median(distances)),
            "nearest_distance_p90_m": float(np.percentile(distances, 90)),
            "fraction_within_5cm": float(np.mean(distances < 0.05)),
        }
        print(json.dumps(reference_qa), flush=True)

    active_blocks = int(geometry.grid.hashmap().size())
    report = {
        "status": "PENDING",
        "purpose": "full_400_frame_empty_map_shared_geometry_capacity_profile",
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "open3d_version": o3d.__version__,
        "device": "CPU:0",
        "voxel_size_m": geometry.voxel_size_m,
        "block_resolution": 8,
        "block_count_capacity": args.block_count,
        "depth_max_m": geometry.depth_max_m,
        "frame_ids": list(frame_ids),
        "integration": integration,
        "checkpoints": checkpoints,
        "active_blocks": active_blocks,
        "capacity_fraction": active_blocks / args.block_count,
        "surface_points": int(len(positions)),
        "surface_bounds_min_m": positions.min(axis=0).tolist() if len(positions) else None,
        "surface_bounds_max_m": positions.max(axis=0).tolist() if len(positions) else None,
        "surface_color_out_of_range_fraction": float(np.mean(np.any((colors < 0) | (colors > 1), axis=1))) if len(colors) else None,
        "raycasts": raycasts,
        "reference_geometry_for_qa_only": reference_qa,
        "peak_process_rss_mb": peak_rss_mb(),
        "total_seconds": time.perf_counter() - started,
        "files": {
            "surface_ply": {"path": str(cloud_path), "bytes": cloud_path.stat().st_size},
            "tsdf_grid": {"path": str(grid_path), "bytes": grid_path.stat().st_size},
        },
    }
    report["status"] = (
        "PASS"
        if len(integration) == 400
        and report["surface_points"] > 0
        and report["capacity_fraction"] < 0.9
        and all(item["common_pixels"] > 0 and item["abs_depth_error_median_m"] < 0.05 for item in raycasts)
        and (reference_qa is None or reference_qa["nearest_distance_median_m"] < 0.05)
        else "FAIL"
    )
    report_path = args.output_dir / "geometry_full400.json"
    write_json(report_path, report)
    write_json(progress_path, {"status": report["status"], "checkpoints": checkpoints})
    print(f"{report['status']}: {len(integration)} frames, {active_blocks} blocks, {len(positions)} points, {report['total_seconds']:.1f} seconds", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
