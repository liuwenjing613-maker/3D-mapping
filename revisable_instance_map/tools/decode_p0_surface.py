#!/usr/bin/env python3
"""Materialize two separately evaluated derivatives of immutable P0 evidence."""
import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.local_surface_decoder import (
    DecoderSettings, decode_confirmed, fill_small_holes)

ROOT = Path("/home/chenkejun/CVPR/revisable_instance_map")
DATA = Path("/data/chenkejun/CVPR/revisable_instance_map")
SOURCE = DATA / "surface_p0_room0_stride2_15mm_final"
GEOMETRY = DATA / "geometry_full400_1cm_capacity100k/surface.ply"

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_ply(path, xyz, labels):
    colors = np.full((len(xyz), 3), 0.35, np.float32)
    positive = labels > 0
    if np.any(positive):
        palette = np.random.default_rng(7).uniform(0.2, 1.0, size=(int(labels.max()) + 1, 3))
        colors[positive] = palette[labels[positive]]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("2", "3"), required=True)
    parser.add_argument("--stage2-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "3" and args.stage2_dir is None:
        parser.error("--stage2-dir is required for stage 3")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    settings = DecoderSettings()
    evidence_path = SOURCE / "surface_evidence.npz"
    with np.load(evidence_path, allow_pickle=False) as data:
        evidence = {name: data[name] for name in data.files}
    xyz = evidence["xyz_m"]
    rgb = evidence["rgb"]
    base = np.where(evidence["state"] == 2, evidence["top1_instance_id"], -1).astype(np.int32)
    cloud = o3d.io.read_point_cloud(str(GEOMETRY))
    geometry_xyz = np.asarray(cloud.points)
    if len(geometry_xyz) != len(xyz) or not cloud.has_normals():
        raise ValueError("TSDF normals unavailable or surface count changed")
    xyz_error = float(np.max(np.abs(geometry_xyz - xyz)))
    if xyz_error > 1e-5:
        raise ValueError(f"TSDF surface order mismatch: {xyz_error}")
    normals = np.asarray(cloud.normals, dtype=np.float32)
    norm_length = np.linalg.norm(normals, axis=1)
    if np.any(norm_length < 0.5):
        raise ValueError("Invalid TSDF surface normals")
    normals = normals / norm_length[:, None]
    tree = cKDTree(xyz)
    origin = np.zeros(len(xyz), np.uint8)
    origin[base > 0] = 1  # direct confirmed P0
    if args.stage == "2":
        pair_path = SOURCE / "surface_instance_frame_votes.npz"
        with np.load(pair_path, allow_pickle=False) as data:
            pair = {name: data[name] for name in data.files}
        labels, detail = decode_confirmed(xyz, normals, rgb, evidence, pair, settings, tree)
        changed = detail.pop("changed_indices")
        np.savez_compressed(args.output_dir / "decode_changes.npz",
                            surface_point_index=changed,
                            old_instance_id=detail.pop("old_instance_id"),
                            new_instance_id=detail.pop("new_instance_id"),
                            old_votes=detail.pop("old_votes"),
                            new_direct_votes=detail.pop("new_direct_votes"),
                            neighbor_fraction=detail.pop("neighbor_fraction"),
                            support_neighbor_indices=detail.pop("support_neighbor_indices"))
        origin[changed] = 2  # local correction with direct competing vote
        diagnostic = detail
        parent_hash = sha256_file(SOURCE / "instance_surface.npz")
    else:
        source_map = args.stage2_dir / "instance_surface.npz"
        with np.load(source_map, allow_pickle=False) as data:
            parent_xyz = data["xyz_m"]
            parent_labels = data["instance_id"]
        if not np.array_equal(parent_xyz, xyz):
            raise ValueError("Stage 2 surface changed")
        with np.load(args.stage2_dir / "point_provenance.npz", allow_pickle=False) as data:
            origin = data["label_origin"].copy()
        labels, detail = fill_small_holes(xyz, normals, rgb, evidence, parent_labels, settings, tree)
        new_unknown = detail.pop("filled_unknown")
        new_tentative = detail.pop("promoted_tentative")
        support_neighbors = detail.pop("support_neighbor_indices")
        np.savez_compressed(args.output_dir / "decode_changes.npz",
                            surface_point_index=np.r_[new_unknown, new_tentative],
                            old_instance_id=parent_labels[np.r_[new_unknown, new_tentative]],
                            new_instance_id=labels[np.r_[new_unknown, new_tentative]],
                            source_state=evidence["state"][np.r_[new_unknown, new_tentative]],
                            support_neighbor_indices=support_neighbors)
        origin[new_unknown] = 3  # geometry-constrained unobserved hole
        origin[new_tentative] = 4  # confirmed by neighboring support
        diagnostic = {"candidate_count": detail["candidate_count"],
                      "filled_unknown": int(len(new_unknown)),
                      "promoted_tentative": int(len(new_tentative)),
                      "conflict_points_untouched": detail["conflict_points_untouched"]}
        parent_hash = sha256_file(source_map)
    if np.any((evidence["state"] == 3) & (labels > 0)):
        raise AssertionError("Conflict evidence was published")
    np.savez_compressed(args.output_dir / "instance_surface.npz",
                        xyz_m=xyz, rgb=rgb, instance_id=labels)
    np.savez_compressed(args.output_dir / "point_provenance.npz",
                        original_state=evidence["state"], original_instance_id=base,
                        final_instance_id=labels, label_origin=origin)
    save_ply(args.output_dir / "instance_surface_colored.ply", xyz, labels)
    parent = json.loads((SOURCE / "materialization_report.json").read_text())
    report = {
        "status": "PASS", "purpose": "p0_local_surface_decoder", "stage": int(args.stage),
        "ground_truth_used": False, "frame_count": parent["frame_count"],
        "tsdf_surface_points": len(xyz), "settings": asdict(settings),
        "source_p0_evidence_sha256": sha256_file(evidence_path),
        "source_p0_pair_votes_sha256": sha256_file(SOURCE / "surface_instance_frame_votes.npz"),
        "parent_map_sha256": parent_hash,
        "geometry_sha256": sha256_file(GEOMETRY),
        "input_xyz_max_error_m": xyz_error, "diagnostics": diagnostic,
        "labeled_surface_points": int(np.sum(labels > 0)),
        "label_origin_counts": {str(k): int(v) for k, v in zip(*np.unique(origin, return_counts=True))},
        "instance_surface_sha256": sha256_file(args.output_dir / "instance_surface.npz"),
        "ply_sha256": sha256_file(args.output_dir / "instance_surface_colored.ply"),
        "seconds": time.perf_counter() - start,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    (args.output_dir / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": args.stage, "diagnostics": diagnostic,
                      "output_dir": str(args.output_dir), "seconds": report["seconds"]}))

if __name__ == "__main__":
    main()
