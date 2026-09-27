#!/usr/bin/env python3
"""Save prefix-only shared TSDF snapshots for online diagnostics."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource  # noqa: E402
from revisable_instance_map.geometry_tsdf import SparseTSDFGeometry  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 200])
    args = parser.parse_args()
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("Checkpoints must be positive")
    source = ReplicaFrameSource(args.config)
    if checkpoints[-1] > len(source.frame_ids):
        raise ValueError("Checkpoint beyond input protocol")
    args.output_root.mkdir(parents=True, exist_ok=True)
    geometry = SparseTSDFGeometry(block_count=100000)
    started = time.perf_counter()
    summaries = []
    for index, frame_id in enumerate(source.frame_ids[:checkpoints[-1]], start=1):
        geometry.integrate(source.load_frame(frame_id))
        if index not in checkpoints:
            continue
        target = args.output_root / f"prefix_{index:03d}"
        target.mkdir(parents=True, exist_ok=True)
        surface = geometry.extract_point_cloud(weight_threshold=1.0)
        legacy = surface.to_legacy()
        colors = np.asarray(legacy.colors)
        colors[:] = np.clip(colors, 0.0, 1.0)
        surface_path = target / "surface.ply"
        if not o3d.io.write_point_cloud(str(surface_path), legacy):
            raise RuntimeError("Surface export failed")
        grid_path = target / "tsdf_grid.npz"
        geometry.save(grid_path)
        summary = {
            "frame_count": index,
            "last_frame_id": frame_id,
            "active_blocks": int(geometry.grid.hashmap().size()),
            "surface_points": int(len(surface.point.positions)),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "surface_ply": str(surface_path),
            "tsdf_grid": str(grid_path),
        }
        summaries.append(summary)
        write_json(target / "snapshot_report.json", summary)
        print(json.dumps(summary), flush=True)
    write_json(args.output_root / "geometry_snapshots.json", {
        "status": "PASS",
        "purpose": "prefix_only_shared_tsdf_snapshots",
        "summaries": summaries,
    })


if __name__ == "__main__":
    main()

