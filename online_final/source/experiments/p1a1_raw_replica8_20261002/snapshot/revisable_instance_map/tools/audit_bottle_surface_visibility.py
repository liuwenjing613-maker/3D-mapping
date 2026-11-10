#!/usr/bin/env python3
"""Diagnostic GT bottle-to-TSDF visibility; no GT enters mapping or decoding."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path("/home/chenkejun/CVPR/revisable_instance_map")
DATA = Path("/data/chenkejun/CVPR/revisable_instance_map")
sys.path.insert(0, str(ROOT / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource

with np.load(DATA / "surface_p0_room0_stride2_15mm_final/surface_evidence.npz") as d:
    xyz, state = d["xyz_m"], d["state"]
with np.load("/data/chenkejun/CVPR/evaluation_results/debug_room0/gt.npz") as d:
    gt_xyz = d["xyz_ref"][d["instance_id"] == 46000]
distance, nearest = cKDTree(xyz).query(gt_xyz)
point_idx = np.unique(nearest[distance <= 0.01])
points = xyz[point_idx]
used = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5_ovimap_parent_regrouped.json")
raw = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5.json")
ref = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5_ovimap_refined.json")
n = len(points)
ever = np.zeros((n, 5), bool)
samples = np.zeros((n, 5), np.int32)
for frame_id in used.frame_ids:
    frame = used.load_frame(frame_id)
    cam = (points - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
    c = frame.camera
    z = np.maximum(cam[:, 2], 1e-6)
    u = np.rint(c.fx * cam[:, 0] / z + c.cx).astype(int)
    v = np.rint(c.fy * cam[:, 1] / z + c.cy).astype(int)
    inside = (cam[:, 2] > 0) & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
    q = np.flatnonzero(inside)
    if not len(q):
        continue
    depth = frame.depth_m[v[q], u[q]]
    visible = np.isfinite(depth) & (depth > 0) & (np.abs(depth - cam[q, 2]) <= .02)
    q = q[visible]
    if not len(q):
        continue
    with Image.open(raw.mask_root / raw.config["source"]["mask_pattern"].format(frame=frame_id)) as im:
        raw_mask = np.asarray(im)
    with Image.open(ref.mask_root / ref.config["source"]["mask_pattern"].format(frame=frame_id)) as im:
        ref_mask = np.asarray(im)
    flags = np.stack((np.ones(len(q), bool),
                      raw_mask[v[q], u[q]] > 0,
                      ref_mask[v[q], u[q]] > 0,
                      frame.mask_local[v[q], u[q]] > 0,
                      (frame.mask_local[v[q], u[q]] > 0) & (u[q] % 2 == 0) & (v[q] % 2 == 0)), axis=1)
    ever[q] |= flags
    samples[q] += flags
names = ["depth_visible", "raw_mask", "refined_mask", "used_mask", "used_at_even_pixel"]
report = {"bottle_gt_vertices": len(gt_xyz), "near_tsdf_points_1cm": n, "groups": {}}
for name, selection in [("unobserved", state[point_idx] == 0),
                        ("tentative", state[point_idx] == 1),
                        ("confirmed", state[point_idx] == 2),
                        ("conflict", state[point_idx] == 3)]:
    report["groups"][name] = {
        "surface_points": int(np.sum(selection)),
        "ever": {key: int(np.sum(ever[selection, i])) for i, key in enumerate(names)},
        "projected_samples": {key: int(np.sum(samples[selection, i])) for i, key in enumerate(names)}
    }
out = DATA / "p0_decoder_room0_audit/bottle_surface_visibility.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
