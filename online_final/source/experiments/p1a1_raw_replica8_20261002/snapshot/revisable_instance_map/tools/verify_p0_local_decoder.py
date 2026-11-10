#!/usr/bin/env python3
"""Verify reversibility, provenance, PLY integrity, and object-level regressions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DATA = Path("/data/chenkejun/CVPR/revisable_instance_map")
SOURCE = DATA / "surface_p0_room0_stride2_15mm_final"
STAGES = [DATA / "surface_p0_local_decode_stage2_trace",
          DATA / "surface_p0_local_decode_stage3_trace"]
AUDIT = DATA / "p0_decoder_room0_audit"
EVALS = ["surface_p0_room0_provenance_fixed",
         "p0_local_decode_stage2_trace",
         "p0_local_decode_stage3_trace"]

def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            d.update(block)
    return d.hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def matched(o):
    m = o["iou"]
    good = m > .5
    rr, cc = linear_sum_assignment(-(good * (min(m.shape) + 1 + m)))
    return {int(c): int(r) for r, c in zip(rr, cc) if good[r, c]}

def main():
    with np.load(SOURCE / "instance_surface.npz") as d:
        xyz, base = d["xyz_m"], d["instance_id"]
    raw_report = json.loads((SOURCE / "materialization_report.json").read_text())
    assert sha(SOURCE / "surface_evidence.npz") == raw_report["surface_evidence_sha256"]
    assert sha(SOURCE / "instance_surface.npz") == raw_report["instance_surface_sha256"]
    evidence = load(SOURCE / "surface_evidence.npz")
    labels = [base]
    stage_reports = []
    for stage_index, directory in enumerate(STAGES, start=2):
        data = load(directory / "instance_surface.npz")
        report = json.loads((directory / "materialization_report.json").read_text())
        provenance = load(directory / "point_provenance.npz")
        changes = load(directory / "decode_changes.npz")
        cloud = o3d.io.read_point_cloud(str(directory / "instance_surface_colored.ply"))
        assert report["source_p0_evidence_sha256"] == sha(SOURCE / "surface_evidence.npz")
        assert report["instance_surface_sha256"] == sha(directory / "instance_surface.npz")
        assert report["ply_sha256"] == sha(directory / "instance_surface_colored.ply")
        assert np.array_equal(data["xyz_m"], xyz)
        assert len(cloud.points) == len(xyz)
        assert np.max(np.abs(np.asarray(cloud.points) - xyz)) < 1e-5
        assert np.array_equal(provenance["original_instance_id"], base)
        assert np.array_equal(provenance["final_instance_id"], data["instance_id"])
        idx = changes["surface_point_index"]
        assert np.array_equal(changes["old_instance_id"], labels[-1][idx])
        assert np.array_equal(changes["new_instance_id"], data["instance_id"][idx])
        diff = np.flatnonzero(labels[-1] != data["instance_id"])
        assert np.array_equal(np.sort(diff), np.sort(idx))
        assert np.all(data["instance_id"][evidence["state"] == 3] < 0)
        support = changes["support_neighbor_indices"]
        assert len(support) == len(idx)
        assert np.all(np.sum(support >= 0, axis=1) >= (1 if stage_index == 2 else 8))
        for row in range(len(idx)):
            n = support[row, support[row] >= 0]
            assert np.all(labels[-1][n] == changes["new_instance_id"][row])
        if stage_index == 2:
            assert np.all(evidence["state"][idx] == 2)
            assert np.all(changes["new_direct_votes"] >= 1)
            assert np.all(provenance["label_origin"][idx] == 2)
        else:
            assert np.all(np.isin(evidence["state"][idx], [0, 1]))
            assert np.all(labels[-1][idx] < 0)
            assert np.all(np.isin(provenance["label_origin"][idx], [3, 4]))
        labels.append(data["instance_id"])
        stage_reports.append({"stage": stage_index, "changed_surface_points": len(idx),
                              "labeled_surface_points": int(np.sum(labels[-1] > 0)),
                              "ply_points": len(cloud.points), "traceable_changes": len(support)})
    overlaps = {}
    metrics = {}
    for name in EVALS:
        edir = DATA / ("evaluation_" + name)
        overlaps[name] = load(edir / "metrics/overlap_matrix.npz")
        metrics[name] = json.loads((edir / "metrics/metrics.json").read_text())
    base_o = overlaps[EVALS[0]]
    assert all(np.array_equal(base_o["gt_ids"], overlaps[n]["gt_ids"]) for n in EVALS)
    matches = {n: matched(overlaps[n]) for n in EVALS}
    rows = []
    for col, gt_id in enumerate(base_o["gt_ids"]):
        row = {"gt_id": int(gt_id), "gt_vertices": int(base_o["gt_size"][col])}
        for n in EVALS:
            o = overlaps[n]
            best = int(np.argmax(o["iou"][:, col]))
            row[n + "_matched"] = int(col in matches[n])
            row[n + "_best_id"] = str(o["pred_uids"][best])
            row[n + "_best_iou"] = float(o["iou"][best, col])
            row[n + "_best_recall"] = float(o["recall"][best, col])
        rows.append(row)
    AUDIT.mkdir(parents=True, exist_ok=True)
    with (AUDIT / "decoder_object_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with np.load("/data/chenkejun/CVPR/evaluation_results/debug_room0/gt.npz") as d:
        bottle_gt = d["xyz_ref"][d["instance_id"] == 46000]
    dist, nearest = cKDTree(xyz).query(bottle_gt, workers=-1)
    bottle_points = np.unique(nearest[dist <= .03])
    report = {
        "status": "PASS", "source_evidence_immutable": True,
        "stage_verification": stage_reports,
        "metrics": {n: {"AP50": metrics[n]["CA_AP50_uniform"],
                        "PQ": metrics[n]["CA_PQ"]["PQ"],
                        "TP": metrics[n]["CA_PQ"]["TP"],
                        "FP": metrics[n]["CA_PQ"]["FP"],
                        "FN": metrics[n]["CA_PQ"]["FN"]} for n in EVALS},
        "gt_objects_best_iou_decline_over_0_005": {
            n: [{"gt_id": r["gt_id"], "delta": r[n + "_best_iou"] - r[EVALS[0] + "_best_iou"]}
                for r in rows if r[n + "_best_iou"] - r[EVALS[0] + "_best_iou"] < -.005]
            for n in EVALS[1:]},
        "gt_objects_newly_unmatched": {
            n: [r["gt_id"] for r in rows if r[EVALS[0] + "_matched"] and not r[n + "_matched"]]
            for n in EVALS[1:]},
        "bottle_46000": {
            "gt_vertices": len(bottle_gt),
            "near_surface_points": len(bottle_points),
            "native_labeled_near_bottle": {name: int(np.sum(label[bottle_points] > 0))
                                            for name, label in zip(EVALS, labels)},
            "gt_best_iou": {name: next(r[name + "_best_iou"] for r in rows if r["gt_id"] == 46000)
                            for name in EVALS},
        },
    }
    (AUDIT / "decoder_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))

if __name__ == "__main__":
    main()
