#!/usr/bin/env python3
"""Trace representative Replica P0 errors and build a static diagnostic dashboard."""
import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


SCENES = ("room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from unified_eval.io import load_gt, load_prediction  # noqa: E402

DATA_ROOT = Path("/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main")
REFERENCE_ROOT = Path("/home/chenkejun/beauty/ovimap_aligned_eval_20260908/reference")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def one_to_one_matches(iou):
    if not iou.size:
        return []
    eligible = iou > 0.5
    bonus = min(iou.shape) + 1
    rows, cols = linear_sum_assignment(-(eligible * (bonus + iou)))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if eligible[r, c]]


def prediction_vertices(prediction):
    return {instance.instance_uid: np.asarray(instance.vertex_indices, np.int32)
            for instance in prediction.instances}


def make_error_rows(scene, overlap):
    iou = overlap["iou"]
    matches = one_to_one_matches(iou)
    matched_rows = {row for row, _ in matches}
    matched_cols = {col for _, col in matches}
    rows = []
    for col, gt_id in enumerate(overlap["gt_ids"]):
        if col in matched_cols:
            continue
        best = int(np.argmax(iou[:, col])) if iou.shape[0] else None
        significant = np.flatnonzero((overlap["intersection"][:, col] >= 50) &
                                     (overlap["recall"][:, col] >= 0.10))
        rows.append({
            "scene": scene, "error_type": "FN", "gt_id": int(gt_id),
            "pred_uid": str(overlap["pred_uids"][best]) if best is not None else "",
            "gt_size": int(overlap["gt_size"][col]),
            "pred_size": int(overlap["pred_size"][best]) if best is not None else 0,
            "iou": float(iou[best, col]) if best is not None else 0.0,
            "precision": float(overlap["precision"][best, col]) if best is not None else 0.0,
            "recall": float(overlap["recall"][best, col]) if best is not None else 0.0,
            "significant_predictions": [str(overlap["pred_uids"][x]) for x in significant],
            "significant_count": int(len(significant)),
            "severity": float(overlap["gt_size"][col] * (1.0 - (iou[best, col] if best is not None else 0))),
        })
    for row, uid in enumerate(overlap["pred_uids"]):
        if row in matched_rows or overlap["pred_void_fraction"][row] > 0.5:
            continue
        best = int(np.argmax(iou[row])) if iou.shape[1] else None
        significant = np.flatnonzero((overlap["intersection"][row] >= 50) &
                                     (overlap["recall"][row] >= 0.10))
        rows.append({
            "scene": scene, "error_type": "FP",
            "gt_id": int(overlap["gt_ids"][best]) if best is not None else -1,
            "pred_uid": str(uid),
            "gt_size": int(overlap["gt_size"][best]) if best is not None else 0,
            "pred_size": int(overlap["pred_size"][row]),
            "iou": float(iou[row, best]) if best is not None else 0.0,
            "precision": float(overlap["precision"][row, best]) if best is not None else 0.0,
            "recall": float(overlap["recall"][row, best]) if best is not None else 0.0,
            "void_fraction": float(overlap["pred_void_fraction"][row]),
            "significant_gt_ids": [int(overlap["gt_ids"][x]) for x in significant],
            "significant_count": int(len(significant)),
            "severity": float(overlap["pred_size"][row] * (1.0 - (iou[row, best] if best is not None else 0))),
        })
    return rows


def select_cases(rows):
    selected = []
    fn = sorted((row for row in rows if row["error_type"] == "FN"),
                key=lambda row: row["severity"], reverse=True)
    fp = sorted((row for row in rows if row["error_type"] == "FP"),
                key=lambda row: row["severity"], reverse=True)
    selected.extend(fn[:7])
    selected.extend(fp[:4])
    split = next((row for row in fn if row["significant_count"] >= 2 and row not in selected), None)
    merge = next((row for row in fp if row["significant_count"] >= 2 and row not in selected), None)
    if split:
        selected.append(split)
    if merge:
        selected.append(merge)
    return selected


def load_lineage(scene_dir):
    observations = {}
    with (scene_dir / "observations/observations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            observations[row["observation_id"]] = row
    by_instance = defaultdict(list)
    association_lookup = {}
    with (scene_dir / "association/associations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            obs = observations[row["observation_id"]]
            merged = {**row, "pixel_count": obs["pixel_count"],
                      "projectable_pixel_count": obs["projectable_pixel_count"]}
            if "bbox_xyxy" in obs:
                merged["bbox_xyxy"] = obs["bbox_xyxy"]
            by_instance[int(row["instance_id"])].append(merged)
            association_lookup[(int(row["frame_id"]), int(row["mask_local_id"]))] = int(row["instance_id"])
    return by_instance, association_lookup


def project_points(points, pose, camera, depth):
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    index = np.flatnonzero(np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0))
    if not len(index):
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32)
    projected = cam[index]
    u_float = np.rint(camera[0] * projected[:, 0] / projected[:, 2] + camera[2])
    v_float = np.rint(camera[1] * projected[:, 1] / projected[:, 2] + camera[3])
    inside = (np.isfinite(u_float) & np.isfinite(v_float) &
              (u_float >= 0) & (u_float < depth.shape[1]) &
              (v_float >= 0) & (v_float < depth.shape[0]))
    index = index[inside]
    u = u_float[inside].astype(np.int32)
    v = v_float[inside].astype(np.int32)
    if not len(index):
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32)
    observed = depth[v, u]
    visible = np.isfinite(observed) & (observed > 0) & (np.abs(observed - cam[index, 2]) <= 0.02)
    return index[visible], u[visible], v[visible]


def trace_gt_visibility_batch(scene_dir, cases, gt, association_lookup):
    config = read_json(scene_dir / "configs/p0_parent_regrouped.json")
    raw_root = Path(read_json(scene_dir / "configs/raw.json")["source"]["mask_root"])
    final_root = Path(config["source"]["mask_root"])
    scene_root = Path(config["source"]["scene_root"])
    poses = np.loadtxt(scene_root / config["source"]["trajectory"], dtype=np.float64).reshape(-1, 4, 4)
    camera = (config["camera"]["fx"], config["camera"]["fy"],
              config["camera"]["cx"], config["camera"]["cy"])
    traces = {}
    for case in cases:
        gt_points = gt.xyz_ref[gt.instance_id == case["gt_id"]]
        if len(gt_points) > 2500:
            choose = np.linspace(0, len(gt_points) - 1, 2500, dtype=np.int32)
            gt_points = gt_points[choose]
        traces[case["gt_id"]] = {
            "points": gt_points, "totals": Counter(), "persistent": Counter(),
            "frames": [],
        }
    for frame_id in range(0, 2000, 5):
        depth = cv2.imread(str(scene_root / f"results/depth{frame_id:06d}.png"),
                           cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / config["camera"]["depth_png_units_per_meter"]
        raw = cv2.imread(str(raw_root / f"frame{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        final = cv2.imread(str(final_root / f"frame{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        for trace in traces.values():
            _, u, v = project_points(trace["points"], poses[frame_id], camera, depth_m)
            if not len(u):
                continue
            raw_ids = raw[v, u]
            final_ids = final[v, u]
            raw_count = int(np.sum(raw_ids > 0))
            final_count = int(np.sum(final_ids > 0))
            trace["totals"].update({"visible": len(u), "raw": raw_count,
                                    "final": final_count})
            ids, counts = np.unique(final_ids[final_ids > 0], return_counts=True)
            mapped = []
            for local_id, count in zip(ids, counts):
                instance_id = association_lookup.get((frame_id, int(local_id)), -1)
                if instance_id > 0:
                    trace["persistent"][instance_id] += int(count)
                mapped.append({"local_id": int(local_id), "samples": int(count),
                               "instance_id": int(instance_id)})
            trace["frames"].append({"frame_id": frame_id, "visible": int(len(u)),
                                    "raw": raw_count, "final": final_count,
                                    "mapped": mapped, "u": u, "v": v})
    result = {}
    for gt_id, trace in traces.items():
        trace["frames"].sort(key=lambda row: (row["visible"], row["final"]), reverse=True)
        totals = trace["totals"]
        result[gt_id] = {
            "visible_samples": int(totals["visible"]),
            "raw_coverage": totals["raw"] / totals["visible"] if totals["visible"] else 0.0,
            "final_coverage": totals["final"] / totals["visible"] if totals["visible"] else 0.0,
            "persistent_ids": trace["persistent"].most_common(8),
            "frames": trace["frames"], "config": config, "poses": poses,
            "camera": camera, "points": trace["points"],
        }
    return result


def surface_trace(evidence, surface_tree, points):
    distance, index = surface_tree.query(points, workers=-1)
    near = distance <= 0.03
    index = np.unique(index[near])
    state_count = {str(key): int(value) for key, value in
                   zip(*np.unique(evidence["state"][index], return_counts=True))}
    top_ids = Counter(map(int, evidence["top1_instance_id"][index])).most_common(8)
    return {"geometry_coverage": float(np.mean(near)), "surface_points": int(len(index)),
            "state_counts": state_count, "top_instance_ids": top_ids}


def summarize_gt_support(labels):
    labels = labels[labels > 0]
    if not len(labels):
        return {"samples": 0, "purity": 0.0, "mixed": False, "top_gt_ids": []}
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    ranked = [(int(ids[i]), int(counts[i])) for i in order[:6]]
    total = int(np.sum(counts))
    second = ranked[1][1] if len(ranked) > 1 else 0
    return {
        "samples": total,
        "purity": ranked[0][1] / total,
        "mixed": second >= max(20, int(np.ceil(0.15 * total))),
        "top_gt_ids": ranked,
    }


def mask_gt_support(mask, local_id, depth, pose, camera, gt_tree, gt_labels,
                    max_pixels=4000):
    support = ((mask == local_id) & np.isfinite(depth) & (depth > 0))
    v, u = np.nonzero(support)
    if len(u) > max_pixels:
        choose = np.linspace(0, len(u) - 1, max_pixels, dtype=np.int32)
        u, v = u[choose], v[choose]
    if not len(u):
        return summarize_gt_support(np.empty(0, np.int64))
    z = depth[v, u]
    camera_xyz = np.column_stack(((u - camera[2]) * z / camera[0],
                                  (v - camera[3]) * z / camera[1], z))
    world_xyz = camera_xyz @ pose[:3, :3].T + pose[:3, 3]
    distance, index = gt_tree.query(world_xyz, workers=-1)
    near = np.isfinite(distance) & (distance <= 0.03)
    return summarize_gt_support(gt_labels[index[near]])


def choose_lineage_observations(observations, limit=16):
    if len(observations) <= limit:
        return sorted(observations, key=lambda row: int(row["frame_id"]))
    by_size = sorted(observations, key=lambda row: row["projectable_pixel_count"],
                     reverse=True)[:limit // 2]
    by_time = sorted(observations, key=lambda row: int(row["frame_id"]))
    temporal = [by_time[i] for i in np.linspace(
        0, len(by_time) - 1, limit - len(by_size), dtype=np.int32)]
    selected = {(int(row["frame_id"]), int(row["mask_local_id"])): row
                for row in by_size + temporal}
    return sorted(selected.values(), key=lambda row: int(row["frame_id"]))


def trace_prediction_lineage(scene_dir, instance_ids, gt, lineage):
    config = read_json(scene_dir / "configs/p0_parent_regrouped.json")
    raw_root = Path(read_json(scene_dir / "configs/raw.json")["source"]["mask_root"])
    final_root = Path(config["source"]["mask_root"])
    scene_root = Path(config["source"]["scene_root"])
    poses = np.loadtxt(scene_root / config["source"]["trajectory"],
                       dtype=np.float64).reshape(-1, 4, 4)
    camera = (config["camera"]["fx"], config["camera"]["fy"],
              config["camera"]["cx"], config["camera"]["cy"])
    valid_gt = ((gt.instance_id > 0) & np.isfinite(gt.xyz_ref).all(axis=1))
    gt_tree = cKDTree(gt.xyz_ref[valid_gt])
    gt_labels = gt.instance_id[valid_gt]
    frame_cache = {}

    def load_frame(frame_id):
        if frame_id not in frame_cache:
            depth = cv2.imread(str(scene_root / f"results/depth{frame_id:06d}.png"),
                               cv2.IMREAD_UNCHANGED)
            raw = cv2.imread(str(raw_root / f"frame{frame_id:06d}.png"),
                             cv2.IMREAD_UNCHANGED)
            final = cv2.imread(str(final_root / f"frame{frame_id:06d}.png"),
                               cv2.IMREAD_UNCHANGED)
            if depth is None or raw is None or final is None:
                raise FileNotFoundError(f"Missing frame data for {frame_id}")
            frame_cache[frame_id] = (
                depth.astype(np.float32) /
                config["camera"]["depth_png_units_per_meter"], raw, final)
        return frame_cache[frame_id]

    traces = {}
    for instance_id in sorted(set(instance_ids)):
        observations = choose_lineage_observations(lineage.get(instance_id, []))
        records = []
        dominant_observations = Counter()
        dominant_samples = Counter()
        raw_mixed = 0
        final_mixed = 0
        for row in observations:
            frame_id = int(row["frame_id"])
            local_id = int(row["mask_local_id"])
            depth, raw, final = load_frame(frame_id)
            raw_support = mask_gt_support(raw, local_id, depth, poses[frame_id],
                                          camera, gt_tree, gt_labels)
            final_support = mask_gt_support(final, local_id, depth, poses[frame_id],
                                            camera, gt_tree, gt_labels)
            raw_mixed += int(raw_support["mixed"])
            final_mixed += int(final_support["mixed"])
            if final_support["top_gt_ids"]:
                dominant = final_support["top_gt_ids"][0]
                dominant_observations[dominant[0]] += 1
                dominant_samples[dominant[0]] += dominant[1]
            records.append({
                "frame_id": frame_id, "mask_local_id": local_id,
                "decision": row["decision"], "raw_support": raw_support,
                "final_support": final_support,
            })
        stable_ids = [int(gt_id) for gt_id, count in dominant_observations.items()
                      if count >= 2 and dominant_samples[gt_id] >= 100]
        traces[instance_id] = {
            "total_observations": len(lineage.get(instance_id, [])),
            "sampled_observations": len(records),
            "raw_mixed_observations": raw_mixed,
            "final_mixed_observations": final_mixed,
            "stable_dominant_gt_ids": sorted(stable_ids),
            "dominant_observation_counts": dominant_observations.most_common(),
            "dominant_sample_counts": dominant_samples.most_common(),
            "observations": records,
        }
    return traces


def refine_merge_cause(trace):
    sampled = trace.get("sampled_observations", 0)
    mixed = trace.get("raw_mixed_observations", 0)
    if mixed >= max(2, int(np.ceil(0.20 * sampled))):
        return "cropformer_mask_merge", "多个单帧原始 CropFormer mask 已同时覆盖不同 GT 实例"
    if len(trace.get("stable_dominant_gt_ids", [])) >= 2:
        return "association_merge", "单帧 mask 多数较纯，但不同 GT 的观测被累计到同一 persistent ID"
    return "surface_attribution_merge", "抽样观测未显示稳定前端混合或跨对象关联；优先检查表面归属及未抽样长尾观测"


def classify_fn(case, visibility, surface):
    if surface["geometry_coverage"] < 0.50:
        return "geometry_gap", "GT 表面很少能在最终 TSDF 中找到 3 cm 内对应点"
    if visibility["visible_samples"] == 0:
        return "visibility_gap", "GT 对象在固定 400 帧中没有通过深度可见性检查"
    if visibility["raw_coverage"] < 0.30:
        return "cropformer_miss", "可见 GT 投影大部分落在原始 CropFormer 背景"
    if visibility["final_coverage"] < 0.60 * visibility["raw_coverage"]:
        return "depth_refinement_erasure", "深度细化删除了大部分原始前景支持"
    if case["precision"] < 0.50 and case["recall"] >= 0.50:
        return "instance_merge", "最佳预测覆盖对象但同时包含大量其他表面"
    ids = visibility["persistent_ids"]
    if len(ids) >= 2 and ids[1][1] >= 0.20 * ids[0][1]:
        return "association_fragmentation", "同一 GT 的逐帧支持被分配给多个 persistent ID"
    states = surface["state_counts"]
    uncertain = sum(states.get(str(code), 0) for code in (0, 1, 3))
    if surface["surface_points"] and uncertain / surface["surface_points"] > 0.45:
        return "surface_evidence_gap", "附近 TSDF 点多数未确认或存在 ID 冲突"
    return "borderline_overlap", "证据链存在，但最终 IoU 未超过 0.5"


def classify_fp(case):
    if case.get("significant_count", 0) >= 2 and case["precision"] < 0.75:
        return "instance_merge", "同一预测显著覆盖多个 GT 对象"
    if case["precision"] >= 0.75 and case["recall"] < 0.50:
        return "duplicate_fragment", "预测较纯但只覆盖 GT 的局部，通常是重复出生或关联断裂"
    if case["iou"] >= 0.30 and case["recall"] >= 0.30:
        return "duplicate_fragment", "预测与一个 GT 有明显重叠，但未成为一对一匹配"
    if case["precision"] < 0.30 or case.get("void_fraction", 0) > 0.30:
        return "background_leakage", "预测中大部分表面不属于任何对应 GT 对象"
    return "spurious_or_small_fragment", "弱关联产生了未匹配 persistent 实例"


def downsample(index, limit=5000):
    if len(index) <= limit:
        return index
    return index[np.linspace(0, len(index) - 1, limit, dtype=np.int32)]


def save_3d_plot(path, xyz, gt_index, pred_index, title):
    gt_set = np.zeros(len(xyz), bool)
    pred_set = np.zeros(len(xyz), bool)
    gt_set[gt_index] = True
    pred_set[pred_index] = True
    overlap = np.flatnonzero(gt_set & pred_set)
    gt_only = np.flatnonzero(gt_set & ~pred_set)
    pred_only = np.flatnonzero(pred_set & ~gt_set)
    fig = plt.figure(figsize=(9.2, 4.1), facecolor="#0b1220")
    for position, azimuth in enumerate((35, 125), start=1):
        ax = fig.add_subplot(1, 2, position, projection="3d", facecolor="#0b1220")
        for indices, color, label in ((gt_only, "#f6c85f", "GT only"),
                                      (pred_only, "#ef5da8", "Prediction only"),
                                      (overlap, "#27d3c2", "Overlap")):
            sample = downsample(indices)
            if len(sample):
                ax.scatter(xyz[sample, 0], xyz[sample, 1], xyz[sample, 2],
                           s=2.0, c=color, alpha=0.85, label=label)
        ax.view_init(elev=25, azim=azimuth)
        ax.set_axis_off()
        if position == 1:
            ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="white")
    fig.suptitle(title, color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def crop_and_save_overlay(path, rgb_path, mask_path, mask_ids, u=None, v=None, caption=""):
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    support = np.isin(mask, list(mask_ids)) if mask_ids else np.zeros(mask.shape, bool)
    canvas = image.copy()
    canvas[support] = (0.55 * canvas[support] + 0.45 * np.array([60, 220, 90])).astype(np.uint8)
    contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (60, 255, 120), 2)
    if u is not None and len(u):
        for x, y in zip(u[::max(1, len(u) // 250)], v[::max(1, len(v) // 250)]):
            cv2.circle(canvas, (int(x), int(y)), 2, (0, 220, 255), -1)
    ys, xs = np.nonzero(support)
    if u is not None and len(u):
        xs = np.r_[xs, u]
        ys = np.r_[ys, v]
    if len(xs):
        x0, x1 = max(0, int(xs.min()) - 90), min(canvas.shape[1], int(xs.max()) + 91)
        y0, y1 = max(0, int(ys.min()) - 70), min(canvas.shape[0], int(ys.max()) + 71)
        if x1 - x0 < 280:
            extra = (280 - (x1 - x0)) // 2
            x0, x1 = max(0, x0 - extra), min(canvas.shape[1], x1 + extra)
        if y1 - y0 < 220:
            extra = (220 - (y1 - y0)) // 2
            y0, y1 = max(0, y0 - extra), min(canvas.shape[0], y1 + extra)
        canvas = canvas[y0:y1, x0:x1]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (9, 18, 34), -1)
    cv2.putText(canvas, caption[:92], (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 242, 255), 1, cv2.LINE_AA)
    scale = min(1.0, 760 / canvas.shape[1])
    if scale < 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(path)


def render_case(scene, scene_dir, case, gt, pred_vertices, classes, lineage,
                visibility_by_gt, prediction_traces, evidence, surface_tree,
                assets):
    gt_id = case["gt_id"]
    pred_uid = case["pred_uid"]
    gt_index = np.flatnonzero(gt.instance_id == gt_id) if gt_id >= 0 else np.empty(0, np.int32)
    pred_index = pred_vertices.get(pred_uid, np.empty(0, np.int32))
    from review_error_dashboard import audit_case_label
    audit_case_label(case, gt, classes)
    slug = f"{scene}_{case['error_type'].lower()}_g{gt_id}_p{pred_uid or 'none'}"
    plot_path = assets / f"{slug}_3d.png"
    save_3d_plot(plot_path, gt.xyz_ref, gt_index, pred_index,
                 f"{scene} · {case['error_type']} · GT {gt_id} · Pred {pred_uid or 'none'}")
    case["plot_3d"] = f"assets/{plot_path.name}"
    frame_images = []
    instance_id = int(pred_uid) if pred_uid.isdigit() else -1
    prediction_trace = prediction_traces.get(instance_id)
    if case["error_type"] == "FN" and len(gt_index):
        visibility = visibility_by_gt[gt_id]
        surface = surface_trace(evidence, surface_tree, gt.xyz_ref[gt_index])
        cause, explanation = classify_fn(case, visibility, surface)
        if cause == "instance_merge" and prediction_trace:
            cause, explanation = refine_merge_cause(prediction_trace)
        case["root_cause"] = cause
        case["root_cause_explanation"] = explanation
        case["trace"] = {
            "visible_samples": visibility["visible_samples"],
            "raw_coverage": visibility["raw_coverage"],
            "final_coverage": visibility["final_coverage"],
            "persistent_ids": visibility["persistent_ids"],
            "surface": surface,
        }
        if prediction_trace:
            case["trace"]["prediction_lineage"] = prediction_trace
        config = visibility["config"]
        scene_root = Path(config["source"]["scene_root"])
        mask_root = Path(config["source"]["mask_root"])
        for rank, frame in enumerate(visibility["frames"][:2]):
            frame_id = frame["frame_id"]
            ids = [entry["local_id"] for entry in frame["mapped"]]
            output = assets / f"{slug}_f{frame_id:06d}.jpg"
            crop_and_save_overlay(
                output, scene_root / f"results/frame{frame_id:06d}.jpg",
                mask_root / f"frame{frame_id:06d}.png", ids, frame["u"], frame["v"],
                f"frame {frame_id} | visible {frame['visible']} | raw {frame['raw']} | final {frame['final']}")
            frame_images.append(f"assets/{output.name}")
    else:
        cause, explanation = classify_fp(case)
        if cause == "instance_merge" and prediction_trace:
            cause, explanation = refine_merge_cause(prediction_trace)
        case["root_cause"] = cause
        case["root_cause_explanation"] = explanation
        observations = sorted(lineage.get(instance_id, []),
                              key=lambda row: row["projectable_pixel_count"], reverse=True)
        config = read_json(scene_dir / "configs/p0_parent_regrouped.json")
        scene_root = Path(config["source"]["scene_root"])
        mask_root = Path(config["source"]["mask_root"])
        case["trace"] = {
            "observation_count": len(observations),
            "birth_decision": min(observations, key=lambda row: int(row["frame_id"]))["decision"] if observations else "none",
            "top_frames": [int(row["frame_id"]) for row in observations[:8]],
        }
        if prediction_trace:
            case["trace"]["prediction_lineage"] = prediction_trace
        for row in observations[:2]:
            frame_id = int(row["frame_id"])
            output = assets / f"{slug}_f{frame_id:06d}.jpg"
            crop_and_save_overlay(
                output, scene_root / f"results/frame{frame_id:06d}.jpg",
                mask_root / f"frame{frame_id:06d}.png", [int(row["mask_local_id"])],
                caption=f"frame {frame_id} | mask {row['mask_local_id']} | {row['decision']} | pixels {row['pixel_count']}")
            frame_images.append(f"assets/{output.name}")
    case["frame_images"] = frame_images
    return case


def dashboard_html(payload):
    from review_error_dashboard import dashboard_html as render_review
    return render_review(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root
    if read_json(run_root / "run_manifest.json").get("status") != "PASS":
        raise RuntimeError("Eight-scene run has not completed successfully")
    output = args.output_dir or (run_root / "error_dashboard")
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    all_rows = []
    detailed = []
    summaries = []
    for scene in SCENES:
        scene_dir = run_root / scene
        summary = read_json(scene_dir / "scene_summary.json")
        summaries.append(summary)
        overlap = load_npz(scene_dir / "evaluation/metrics/overlap_matrix.npz")
        rows = make_error_rows(scene, overlap)
        all_rows.extend(rows)
        gt = load_gt(scene_dir / "ground_truth/gt.npz")
        prediction = load_prediction(scene_dir / "evaluation/adapter/canonical_prediction.npz")
        vertices = prediction_vertices(prediction)
        classes = read_json(REFERENCE_ROOT / scene / "manifest.json")["semantic_classes"]
        lineage, association_lookup = load_lineage(scene_dir)
        selected = select_cases(rows)
        selected_fn = [case for case in selected if case["error_type"] == "FN"]
        visibility_by_gt = trace_gt_visibility_batch(
            scene_dir, selected_fn, gt, association_lookup)
        prediction_ids = [int(case["pred_uid"]) for case in selected
                          if case["pred_uid"].isdigit()]
        prediction_traces = trace_prediction_lineage(
            scene_dir, prediction_ids, gt, lineage)
        evidence = load_npz(scene_dir / "surface_p0/surface_evidence.npz")
        surface_tree = cKDTree(evidence["xyz_m"])
        for case in selected:
            detailed.append(render_case(scene, scene_dir, dict(case), gt, vertices,
                                        classes, lineage, visibility_by_gt,
                                        prediction_traces, evidence, surface_tree,
                                        assets))
    fieldnames = sorted({key for row in all_rows for key in row})
    with (output / "all_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})
    cause_labels = {
        "geometry_gap": "TSDF 几何缺口", "visibility_gap": "缺少有效深度可见性",
        "cropformer_miss": "CropFormer 漏检", "depth_refinement_erasure": "深度细化误删",
        "association_fragmentation": "跨帧关联碎片化", "surface_evidence_gap": "表面证据未确认",
        "instance_merge": "实例合并", "cropformer_mask_merge": "CropFormer 单帧混合",
        "association_merge": "跨帧关联合并", "surface_attribution_merge": "表面归属合并",
        "borderline_overlap": "IoU 临界失败",
        "duplicate_fragment": "重复或碎片预测", "background_leakage": "背景泄漏",
        "spurious_or_small_fragment": "游离小碎片",
    }
    aggregate = {key: float(np.mean([row[key] for row in summaries]))
                 for key in ("AP", "AP25", "AP50", "PQ")}
    payload = {"scenes": summaries, "aggregate": aggregate, "cases": detailed,
               "cause_labels": cause_labels,
               "diagnostic_scope": "representative_deep_traces_plus_all_error_csv"}
    write_json(output / "error_cases.json", payload)
    (output / "index.html").write_text(dashboard_html(payload), encoding="utf-8")
    root_counts = Counter(case["root_cause"] for case in detailed)
    write_json(output / "analysis_summary.json", {
        "status": "PASS", "scenes": len(summaries), "all_errors": len(all_rows),
        "deep_traced_cases": len(detailed), "root_cause_counts": root_counts,
        "aggregate": aggregate, "dashboard": str(output / "index.html"),
        "warning": "Root-cause labels are post-hoc diagnostics and never mapping inputs.",
    })
    from review_error_dashboard import rebuild
    rebuild(run_root, output)
    print(json.dumps({"status": "PASS", "output": str(output),
                      "cases": len(detailed), "root_causes": root_counts},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
