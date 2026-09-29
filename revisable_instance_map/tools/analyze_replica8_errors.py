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
                      "projectable_pixel_count": obs["projectable_pixel_count"],
                      "bbox_xyxy": obs["bbox_xyxy"]}
            by_instance[int(row["instance_id"])].append(merged)
            association_lookup[(int(row["frame_id"]), int(row["mask_local_id"]))] = int(row["instance_id"])
    return by_instance, association_lookup


def project_points(points, pose, camera, depth):
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    z = np.maximum(cam[:, 2], 1e-6)
    u = np.rint(camera[0] * cam[:, 0] / z + camera[2]).astype(np.int32)
    v = np.rint(camera[1] * cam[:, 1] / z + camera[3]).astype(np.int32)
    inside = ((cam[:, 2] > 0) & (u >= 0) & (u < depth.shape[1]) &
              (v >= 0) & (v < depth.shape[0]))
    index = np.flatnonzero(inside)
    if not len(index):
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32)
    observed = depth[v[index], u[index]]
    visible = np.isfinite(observed) & (observed > 0) & (np.abs(observed - cam[index, 2]) <= 0.02)
    index = index[visible]
    return index, u[index], v[index]


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


def classify_fn(case, visibility, surface):
    if surface["geometry_coverage"] < 0.50:
        return "geometry_gap", "GT 表面很少能在最终 TSDF 中找到 3 cm 内对应点"
    if visibility["visible_samples"] == 0:
        return "visibility_gap", "GT 对象在固定 400 帧中没有通过深度可见性检查"
    if visibility["raw_coverage"] < 0.30:
        return "cropformer_miss", "可见 GT 投影大部分落在原始 CropFormer 背景"
    if visibility["final_coverage"] < 0.60 * visibility["raw_coverage"]:
        return "depth_refinement_erasure", "深度细化删除了大部分原始前景支持"
    ids = visibility["persistent_ids"]
    if len(ids) >= 2 and ids[1][1] >= 0.20 * ids[0][1]:
        return "association_fragmentation", "同一 GT 的逐帧支持被分配给多个 persistent ID"
    states = surface["state_counts"]
    uncertain = sum(states.get(str(code), 0) for code in (0, 1, 3))
    if surface["surface_points"] and uncertain / surface["surface_points"] > 0.45:
        return "surface_evidence_gap", "附近 TSDF 点多数未确认或存在 ID 冲突"
    if case["precision"] < 0.50 and case["recall"] >= 0.50:
        return "instance_merge", "最佳预测覆盖对象但同时包含大量其他表面"
    return "borderline_overlap", "证据链存在，但最终 IoU 未超过 0.5"


def classify_fp(case):
    if case.get("significant_count", 0) >= 2:
        return "instance_merge", "同一预测显著覆盖多个 GT 对象"
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
                visibility_by_gt, evidence, surface_tree, assets):
    gt_id = case["gt_id"]
    pred_uid = case["pred_uid"]
    gt_index = np.flatnonzero(gt.instance_id == gt_id) if gt_id >= 0 else np.empty(0, np.int32)
    pred_index = pred_vertices.get(pred_uid, np.empty(0, np.int32))
    class_index = gt_id // 1000 if gt_id >= 0 else -1
    case["class_name"] = classes[class_index] if 0 <= class_index < len(classes) else "unknown"
    slug = f"{scene}_{case['error_type'].lower()}_g{gt_id}_p{pred_uid or 'none'}"
    plot_path = assets / f"{slug}_3d.png"
    save_3d_plot(plot_path, gt.xyz_ref, gt_index, pred_index,
                 f"{scene} · {case['error_type']} · GT {gt_id} · Pred {pred_uid or 'none'}")
    case["plot_3d"] = f"assets/{plot_path.name}"
    frame_images = []
    if case["error_type"] == "FN" and len(gt_index):
        visibility = visibility_by_gt[gt_id]
        surface = surface_trace(evidence, surface_tree, gt.xyz_ref[gt_index])
        cause, explanation = classify_fn(case, visibility, surface)
        case["root_cause"] = cause
        case["root_cause_explanation"] = explanation
        case["trace"] = {
            "visible_samples": visibility["visible_samples"],
            "raw_coverage": visibility["raw_coverage"],
            "final_coverage": visibility["final_coverage"],
            "persistent_ids": visibility["persistent_ids"],
            "surface": surface,
        }
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
        case["root_cause"] = cause
        case["root_cause_explanation"] = explanation
        instance_id = int(pred_uid) if pred_uid.isdigit() else -1
        observations = sorted(lineage.get(instance_id, []),
                              key=lambda row: row["projectable_pixel_count"], reverse=True)
        config = read_json(scene_dir / "configs/p0_parent_regrouped.json")
        scene_root = Path(config["source"]["scene_root"])
        mask_root = Path(config["source"]["mask_root"])
        case["trace"] = {
            "observation_count": len(observations),
            "birth_decision": observations[0]["decision"] if observations else "none",
            "top_frames": [int(row["frame_id"]) for row in observations[:8]],
        }
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
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replica P0 错误追溯</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b1220'/%3E%3Cpath d='M7 22V10h7c5 0 8 2 8 6s-3 6-8 6H7zm5-4h2c2 0 3-.6 3-2s-1-2-3-2h-2v4z' fill='%2327d3c2'/%3E%3C/svg%3E">
<style>
:root{{--bg:#07101e;--panel:#0e1a2c;--panel2:#13233a;--text:#e9f0fb;--muted:#91a4bd;--line:#243650;--cyan:#27d3c2;--pink:#ef5da8;--yellow:#f6c85f;--red:#ff6b6b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,"Microsoft YaHei",sans-serif}}header{{position:sticky;top:0;z-index:5;background:rgba(7,16,30,.94);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);padding:18px 24px}}h1{{font-size:1.28rem;margin:0 0 4px}}.sub{{color:var(--muted);font-size:.9rem}}main{{max-width:1500px;margin:auto;padding:20px 24px 60px}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin-bottom:18px}}.metric,.controls,.case{{background:var(--panel);border:1px solid var(--line);border-radius:12px}}.metric{{padding:13px 15px}}.metric b{{font-size:1.35rem;display:block;color:var(--cyan)}}.metric span{{color:var(--muted);font-size:.82rem}}.controls{{display:flex;gap:12px;flex-wrap:wrap;padding:12px;margin-bottom:18px}}select,input{{background:#091525;color:var(--text);border:1px solid #314966;border-radius:8px;padding:9px 11px;font-size:.9rem}}input{{min-width:240px;flex:1}}.scene-table{{overflow:auto;margin-bottom:20px;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-size:.78rem;text-transform:uppercase}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px}}.case{{overflow:hidden}}.case-head{{display:flex;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid var(--line)}}.title{{font-weight:700}}.badge{{display:inline-block;padding:3px 8px;border-radius:999px;background:#203652;color:var(--cyan);font-size:.75rem;margin-right:5px}}.badge.fn{{color:var(--yellow)}}.badge.fp{{color:var(--pink)}}.case-body{{padding:14px 16px}}.cause{{border-left:3px solid var(--cyan);padding-left:10px;margin-bottom:12px}}.cause b{{display:block}}.cause span{{color:var(--muted);font-size:.86rem}}.numbers{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}}.numbers div{{background:var(--panel2);padding:8px;border-radius:8px;font-size:.78rem;color:var(--muted)}}.numbers strong{{display:block;color:var(--text);font-size:.95rem}}.visuals{{display:grid;grid-template-columns:1.25fr 1fr;gap:8px}}.visuals img{{width:100%;height:190px;object-fit:cover;background:#06101d;border-radius:8px;border:1px solid var(--line)}}.frames{{display:grid;grid-template-rows:1fr 1fr;gap:8px}}.frames img{{height:91px}}details{{margin-top:10px;color:var(--muted);font-size:.82rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#081421;padding:9px;border-radius:7px;max-height:190px;overflow:auto}}.empty{{padding:40px;text-align:center;color:var(--muted)}}@media(max-width:760px){{header,main{{padding-left:12px;padding-right:12px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.visuals{{grid-template-columns:1fr}}.visuals img{{height:auto}}.frames{{grid-template-columns:1fr 1fr;grid-template-rows:none}}}}
</style></head><body><header><h1>Replica P0 无修复地图 · 错误追溯</h1><div class="sub">统一 Replica-CA-v1 协议 · GT 仅用于建图完成后的诊断 · 黄色点为 GT 可见投影，绿色区域为系统 mask</div></header><main>
<section class="metrics" id="metrics"></section><section class="controls"><select id="scene"><option value="">全部场景</option></select><select id="type"><option value="">全部错误</option><option>FN</option><option>FP</option></select><select id="cause"><option value="">全部根因</option></select><input id="search" placeholder="搜索 GT ID、预测 ID 或类别"></section><section class="scene-table"><table><thead><tr><th>场景</th><th>AP</th><th>AP50</th><th>PQ</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody id="sceneRows"></tbody></table></section><section class="grid" id="cases"></section></main>
<script>const DATA={data};const $=s=>document.querySelector(s);const fmt=x=>Number(x).toFixed(3);const scenes=[...new Set(DATA.cases.map(x=>x.scene))];const causes=[...new Set(DATA.cases.map(x=>x.root_cause))].sort();scenes.forEach(x=>$('#scene').insertAdjacentHTML('beforeend',`<option>${{x}}</option>`));causes.forEach(x=>$('#cause').insertAdjacentHTML('beforeend',`<option value="${{x}}">${{DATA.cause_labels[x]||x}}</option>`));$('#sceneRows').innerHTML=DATA.scenes.map(x=>`<tr><td>${{x.scene_id}}</td><td>${{fmt(x.AP)}}</td><td>${{fmt(x.AP50)}}</td><td>${{fmt(x.PQ)}}</td><td>${{x.TP}}</td><td>${{x.FP}}</td><td>${{x.FN}}</td></tr>`).join('');const total=k=>DATA.scenes.reduce((a,x)=>a+(x[k]||0),0);$('#metrics').innerHTML=`<div class="metric"><b>${{fmt(DATA.aggregate.AP50)}}</b><span>八场景宏平均 AP50</span></div><div class="metric"><b>${{fmt(DATA.aggregate.PQ)}}</b><span>八场景宏平均 PQ</span></div><div class="metric"><b>${{total('TP')}}</b><span>匹配 TP</span></div><div class="metric"><b>${{total('FP')}}</b><span>未匹配 FP</span></div><div class="metric"><b>${{total('FN')}}</b><span>未匹配 FN</span></div>`;
function render(){{const scene=$('#scene').value,type=$('#type').value,cause=$('#cause').value,q=$('#search').value.toLowerCase();const rows=DATA.cases.filter(x=>(!scene||x.scene===scene)&&(!type||x.error_type===type)&&(!cause||x.root_cause===cause)&&(!q||`${{x.gt_id}} ${{x.pred_uid}} ${{x.class_name}}`.toLowerCase().includes(q)));$('#cases').innerHTML=rows.length?rows.map(x=>`<article class="case"><div class="case-head"><div><span class="badge ${{x.error_type.toLowerCase()}}">${{x.error_type}}</span><span class="badge">${{x.scene}}</span><span class="badge">${{x.class_name}}</span></div><div class="title">GT ${{x.gt_id}} · Pred ${{x.pred_uid||'—'}}</div></div><div class="case-body"><div class="cause"><b>${{DATA.cause_labels[x.root_cause]||x.root_cause}}</b><span>${{x.root_cause_explanation}}</span></div><div class="numbers"><div><strong>${{fmt(x.iou)}}</strong>IoU</div><div><strong>${{fmt(x.precision)}}</strong>Precision</div><div><strong>${{fmt(x.recall)}}</strong>Recall</div><div><strong>${{x.gt_size}} / ${{x.pred_size}}</strong>GT / Pred 点数</div></div><div class="visuals"><img loading="lazy" src="${{x.plot_3d}}"><div class="frames">${{x.frame_images.map(p=>`<img loading="lazy" src="${{p}}">`).join('')}}</div></div><details><summary>查看追溯证据</summary><pre>${{JSON.stringify(x.trace,null,2)}}</pre></details></div></article>`).join(''):'<div class="empty">当前筛选条件下没有案例</div>'}}['scene','type','cause','search'].forEach(id=>$('#'+id).addEventListener(id==='search'?'input':'change',render));render();</script></body></html>"""


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
        evidence = load_npz(scene_dir / "surface_p0/surface_evidence.npz")
        surface_tree = cKDTree(evidence["xyz_m"])
        for case in selected:
            detailed.append(render_case(scene, scene_dir, dict(case), gt, vertices,
                                        classes, lineage, visibility_by_gt,
                                        evidence, surface_tree, assets))
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
        "instance_merge": "实例合并", "borderline_overlap": "IoU 临界失败",
        "duplicate_fragment": "重复或碎片预测", "background_leakage": "背景泄漏",
        "spurious_or_small_fragment": "游离小碎片",
    }
    aggregate = {key: float(np.mean([row[key] for row in summaries]))
                 for key in ("AP", "AP50", "PQ")}
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
    print(json.dumps({"status": "PASS", "output": str(output),
                      "cases": len(detailed), "root_causes": root_counts},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
