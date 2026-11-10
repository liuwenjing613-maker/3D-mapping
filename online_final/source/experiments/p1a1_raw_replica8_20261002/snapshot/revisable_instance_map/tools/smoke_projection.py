#!/usr/bin/env python3
"""Read three real frames and check RGB-D projection before map fusion."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map import ReplicaFrameSource, unproject_frame  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ply_write(path, points, colors):
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for index, key in enumerate(("x", "y", "z")):
        vertex[key] = points[:, index]
    for index, key in enumerate(("red", "green", "blue")):
        vertex[key] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1000, 1995])
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--reference", type=Path, help="Optional geometry-only QA reference; never used by the mapper")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ReplicaFrameSource(args.config)
    reference_tree = None
    if args.reference:
        from scipy.spatial import cKDTree

        with np.load(args.reference) as stored:
            reference_tree = cKDTree(stored["xyz"])

    rows = []
    for frame_id in args.frames:
        frame = source.load_frame(frame_id)
        projected = unproject_frame(frame, args.pixel_stride)
        valid_full = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
        if len(projected.world_xyz_m) == 0:
            raise RuntimeError(f"Frame {frame_id} has no valid projected points")

        pose_inverse = np.linalg.inv(frame.camera_to_world)
        back = projected.world_xyz_m.astype(np.float64) @ pose_inverse[:3, :3].T + pose_inverse[:3, 3]
        u_back = frame.camera.fx * back[:, 0] / back[:, 2] + frame.camera.cx
        v_back = frame.camera.fy * back[:, 1] / back[:, 2] + frame.camera.cy
        error_px = np.hypot(u_back - projected.pixel_uv[:, 0], v_back - projected.pixel_uv[:, 1])
        depth_error_m = np.abs(back[:, 2] - projected.camera_xyz_m[:, 2])

        row = {
            "frame_id": frame_id,
            "valid_depth_pixels": int(len(valid_full)),
            "valid_depth_fraction": float(len(valid_full) / frame.depth_m.size),
            "depth_m_percentiles_5_50_95": np.percentile(valid_full, [5, 50, 95]).tolist(),
            "sampled_projected_points": int(len(projected.world_xyz_m)),
            "sampled_points_inside_mask": int(np.count_nonzero(projected.mask_local)),
            "world_bounds_min_m": projected.world_xyz_m.min(axis=0).tolist(),
            "world_bounds_max_m": projected.world_xyz_m.max(axis=0).tolist(),
            "max_reprojection_error_px": float(error_px.max()),
            "max_recovered_depth_error_m": float(depth_error_m.max()),
        }
        if reference_tree is not None:
            distances, _ = reference_tree.query(projected.world_xyz_m, workers=-1)
            row["reference_mesh_distance_median_m"] = float(np.median(distances))
            row["reference_mesh_distance_p90_m"] = float(np.percentile(distances, 90))
            row["reference_mesh_fraction_within_5cm"] = float(np.mean(distances < 0.05))
        ply_write(args.output_dir / f"frame{frame_id:06d}_world_stride{args.pixel_stride}.ply", projected.world_xyz_m, projected.rgb)
        rows.append(row)

    status = "PASS" if all(row["max_reprojection_error_px"] < 0.001 for row in rows) else "FAIL"
    report = {
        "status": status,
        "purpose": "input_projection_smoke_only_no_instance_association_no_fusion",
        "config_sha256": sha256_file(args.config),
        "pixel_stride": args.pixel_stride,
        "reference_geometry_for_qa_only": str(args.reference) if args.reference else None,
        "reference_sha256": sha256_file(args.reference) if args.reference else None,
        "frames": rows,
    }
    output = args.output_dir / "projection_smoke.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{status}: {len(rows)} frames; {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
