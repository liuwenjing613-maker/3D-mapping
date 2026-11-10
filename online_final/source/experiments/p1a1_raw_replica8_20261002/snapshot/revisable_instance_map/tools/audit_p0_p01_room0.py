#!/usr/bin/env python3
"""Read-only room0 regression and bottle lineage audit. GT is diagnostic only."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource

DATA = Path("/data/chenkejun/CVPR/revisable_instance_map")
ROOT = Path("/home/chenkejun/CVPR/revisable_instance_map")
GT_PATH = Path("/data/chenkejun/CVPR/evaluation_results/debug_room0/gt.npz")
CLASS_PATH = Path("/home/chenkejun/beauty/ovimap_aligned_eval_20260908/reference/room0/manifest.json")
EVALS = {
    "legacy": DATA / "evaluation_ovimap_parent_regrouped_multi_vf_dedup",
    "p0": DATA / "evaluation_surface_p0_room0_stride2_15mm_final",
    "p01": DATA / "evaluation_surface_p01_room0_depth15mm",
}
MAPS = {
    "p0": DATA / "surface_p0_room0_stride2_15mm_final",
    "p01": DATA / "surface_p01_room0_depth15mm",
}

def load_overlap(path):
    with np.load(path / "metrics/overlap_matrix.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def matching(o):
    m = o["iou"]
    if not m.size:
        return {}
    eligible = m > 0.5
    rows, cols = linear_sum_assignment(-(eligible * (min(m.shape) + 1 + m)))
    return {int(c): int(r) for r, c in zip(rows, cols) if eligible[r, c]}

def object_rows(overlaps, classes):
    reference = overlaps["p0"]
    if any(not np.array_equal(reference["gt_ids"], o["gt_ids"]) for o in overlaps.values()):
        raise ValueError("GT IDs differ between evaluation runs")
    matches = {name: matching(o) for name, o in overlaps.items()}
    rows = []
    for col, gt_id in enumerate(reference["gt_ids"]):
        gt_id = int(gt_id)
        row = {"gt_id": gt_id, "class": classes[gt_id // 1000] if gt_id // 1000 < len(classes) else "unknown",
               "gt_vertices": int(reference["gt_size"][col])}
        for name, o in overlaps.items():
            best = int(np.argmax(o["iou"][:, col])) if len(o["pred_uids"]) else None
            row[name + "_matched"] = int(col in matches[name])
            row[name + "_best_uid"] = str(o["pred_uids"][best]) if best is not None else ""
            for metric in ("iou", "precision", "recall"):
                row[name + "_" + metric] = float(o[metric][best, col]) if best is not None else 0.0
            row[name + "_intersection"] = int(o["intersection"][best, col]) if best is not None else 0
        row["p01_minus_p0_iou"] = row["p01_iou"] - row["p0_iou"]
        rows.append(row)
    return rows

def label_transitions():
    with np.load(MAPS["p0"] / "instance_surface.npz") as a, np.load(MAPS["p01"] / "instance_surface.npz") as b:
        old, new = a["instance_id"], b["instance_id"]
        if not np.array_equal(a["xyz_m"], b["xyz_m"]):
            raise ValueError("P0 and P0.1 surfaces are not point-aligned")
        return {
            "same_labeled": int(np.sum((old > 0) & (new == old))),
            "changed_id": int(np.sum((old > 0) & (new > 0) & (new != old))),
            "revoked_to_unknown": int(np.sum((old > 0) & (new <= 0))),
            "newly_labeled": int(np.sum((old <= 0) & (new > 0))),
            "both_unknown": int(np.sum((old <= 0) & (new <= 0))),
        }

def state_near_bottle(gt_xyz):
    out = {}
    for name, map_path in MAPS.items():
        with np.load(map_path / "surface_evidence.npz") as d:
            xyz = d["xyz_m"]
            state = d["state"]
            top = d["top1_instance_id"]
            votes = d["top1_votes"]
        distance, indices = cKDTree(xyz).query(gt_xyz, k=1, workers=-1)
        near = distance <= 0.03
        ids = np.unique(indices[near])
        out[name] = {
            "gt_vertices_within_3cm": int(np.sum(near)),
            "unique_surface_points": int(len(ids)),
            "distance_median_m": float(np.median(distance)),
            "state_counts": {str(k): int(v) for k, v in zip(*np.unique(state[ids], return_counts=True))},
            "top1_ids": Counter(map(int, top[ids])).most_common(10),
            "top1_vote_counts": Counter(map(int, votes[ids])).most_common(10),
        }
    return out

def bottle_lineage(gt_xyz, out_dir):
    raw = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5.json")
    ref = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5_ovimap_refined.json")
    used = ReplicaFrameSource(ROOT / "configs/replica_room0_stride5_ovimap_parent_regrouped.json")
    assoc = {}
    with (DATA / "association_ovimap_parent_regrouped_multi/associations.jsonl").open() as f:
        for line in f:
            a = json.loads(line)
            assoc[(a["frame_id"], a["mask_local_id"])] = a
    c = used.camera
    per_frame = []
    aggregate = Counter()
    persistent = Counter()
    for frame_id in used.frame_ids:
        frame = used.load_frame(frame_id)
        pose = frame.camera_to_world
        cam = (gt_xyz - pose[:3, 3]) @ pose[:3, :3]
        front = cam[:, 2] > 0
        u = np.rint(c.fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + c.cx).astype(int)
        v = np.rint(c.fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + c.cy).astype(int)
        valid = front & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
        if not np.any(valid):
            continue
        u, v, z = u[valid], v[valid], cam[valid, 2]
        d = frame.depth_m[v, u]
        depth_valid = np.isfinite(d) & (d > 0) & (np.abs(d - z) <= 0.02)
        if not np.any(depth_valid):
            continue
        u, v = u[depth_valid], v[depth_valid]
        with Image.open(raw.mask_root / raw.config["source"]["mask_pattern"].format(frame=frame_id)) as im:
            raw_id = np.asarray(im)[v, u]
        with Image.open(ref.mask_root / ref.config["source"]["mask_pattern"].format(frame=frame_id)) as im:
            ref_id = np.asarray(im)[v, u]
        used_id = frame.mask_local[v, u]
        raw_count = int(np.sum(raw_id > 0))
        ref_count = int(np.sum(ref_id > 0))
        used_count = int(np.sum(used_id > 0))
        aggregate.update({"visible_gt_samples": len(u), "raw_mask_samples": raw_count,
                          "refined_mask_samples": ref_count, "used_mask_samples": used_count})
        ids, counts = np.unique(used_id[used_id > 0], return_counts=True)
        mapped = []
        for mask_id, n in zip(ids, counts):
            a = assoc.get((frame_id, int(mask_id)))
            pid = int(a["instance_id"]) if a else -1
            mapped.append({"local_id": int(mask_id), "samples": int(n), "persistent_id": pid,
                           "decision": a["decision"] if a else "NO_ASSOCIATION"})
            if pid > 0:
                persistent[pid] += int(n)
                aggregate["associated_samples"] += int(n)
        per_frame.append({"frame_id": frame_id, "visible_gt_samples": len(u), "raw": raw_count,
                          "refined": ref_count, "used": used_count, "associations": mapped})
    with (out_dir / "bottle_frame_lineage.jsonl").open("w") as f:
        for item in per_frame:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"totals": dict(aggregate), "frame_count_with_depth_visible_gt": len(per_frame),
            "persistent_ids_by_gt_projected_samples": persistent.most_common(15),
            "first_frames_with_used_mask": [x for x in per_frame if x["used"] > 0][:12]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA / "p0_decoder_room0_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(GT_PATH) as d:
        labels = d["instance_id"]
        xyz = d["xyz_ref"]
    classes = json.loads(CLASS_PATH.read_text())["semantic_classes"]
    overlaps = {name: load_overlap(path) for name, path in EVALS.items()}
    rows = object_rows(overlaps, classes)
    with (args.output_dir / "gt_object_regression.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    bottle_ids = sorted(int(x) for x in np.unique(labels) if 46000 <= x < 47000)
    bottles = {}
    for gt_id in bottle_ids:
        points = xyz[labels == gt_id]
        bottles[str(gt_id)] = {"gt_points": int(len(points)), "surface": state_near_bottle(points),
                               "lineage": bottle_lineage(points, args.output_dir)}
    report = {
        "scope": "read_only_diagnostic_GT_never_used_in_mapping",
        "label_transitions": label_transitions(),
        "matching_regression": {
            name: {"matched_gt": int(sum(r[name + "_matched"] for r in rows)),
                   "bottle_rows": [r for r in rows if r["class"] == "bottle"]}
            for name in EVALS
        },
        "p0_to_p01_lost_gt": [r for r in rows if r["p0_matched"] and not r["p01_matched"]],
        "p0_to_p01_gained_gt": [r for r in rows if not r["p0_matched"] and r["p01_matched"]],
        "largest_iou_drops": sorted(rows, key=lambda r: r["p01_minus_p0_iou"])[:15],
        "bottles": bottles,
    }
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "transitions": report["label_transitions"],
                      "lost_gt": len(report["p0_to_p01_lost_gt"]),
                      "gained_gt": len(report["p0_to_p01_gained_gt"]),
                      "bottles": bottles}, ensure_ascii=False))

if __name__ == "__main__":
    main()
