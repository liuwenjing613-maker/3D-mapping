from __future__ import annotations

from .metrics import (average_precision, build_overlap, diagnostics,
                      instance_precision_recall_f1, panoptic_quality,
                      significant_structure_diagnostics)
from .schema import CanonicalGT, CanonicalPrediction, Protocol


AP_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]


def evaluate_scenes(scenes: list[tuple[CanonicalGT, CanonicalPrediction]], protocol: Protocol,
                    observed_masks: list | None = None) -> tuple[dict, list[dict], list]:
    if not scenes:
        raise ValueError("At least one scene is required")
    if observed_masks is not None and len(observed_masks) != len(scenes):
        raise ValueError("Observed mask count must match scene count")
    overlaps = [build_overlap(gt, pred, protocol, None if observed_masks is None else observed_masks[i])
                for i, (gt, pred) in enumerate(scenes)]
    predictions = [p for _, p in scenes]
    thresholds = [0.25] + AP_THRESHOLDS
    pooled_ap = {f"{t:.2f}": average_precision(overlaps, predictions, protocol.confidence_mode, t,
        protocol.ignore_unmatched_pred_void_fraction_gt) for t in thresholds}
    per_scene = []
    for (gt, pred), overlap in zip(scenes, overlaps):
        ap = {f"{t:.2f}": average_precision([overlap], [pred], protocol.confidence_mode, t,
            protocol.ignore_unmatched_pred_void_fraction_gt) for t in thresholds}
        row = {
            "scene_id": gt.scene_id,
            "CA_AP_uniform": _mean_defined([ap[f"{t:.2f}"]["ap"] for t in AP_THRESHOLDS]),
            "CA_AP50_uniform": ap["0.50"]["ap"], "CA_AP25_uniform": ap["0.25"]["ap"],
            "AP_by_threshold": ap,
            "CA_PQ": panoptic_quality(overlap, pred.is_partition,
                protocol.ignore_unmatched_pred_void_fraction_gt),
            "diagnostics": diagnostics(overlap, protocol),
        }
        if protocol.is_v2:
            row["CA_PRF1_0_5"] = instance_precision_recall_f1(
                overlap, protocol.ignore_unmatched_pred_void_fraction_gt)
            row["structure"] = significant_structure_diagnostics(overlap, protocol)
        per_scene.append(row)
    pq_rows = [row["CA_PQ"] for row in per_scene]
    if all(row["status"] != "N/A: overlapping masks or non-partition prediction" for row in pq_rows):
        tp = sum(row["TP"] for row in pq_rows)
        fp = sum(row["FP"] for row in pq_rows)
        fn = sum(row["FN"] for row in pq_rows)
        sum_iou = sum(row["SQ"] * row["TP"] for row in pq_rows if row["SQ"] is not None)
        denominator = tp + 0.5 * fp + 0.5 * fn
        sq = sum_iou / tp if tp else 0.0 if denominator else None
        rq = tp / denominator if denominator else None
        pooled_pq = {"PQ": sq * rq if rq is not None else None, "SQ": sq, "RQ": rq,
            "TP": tp, "FP": fp, "FN": fn,
            "ignored_prediction_count": sum(row.get("ignored_prediction_count", 0) for row in pq_rows),
            "status": "ok" if denominator else "undefined: no instances"}
    else:
        pooled_pq = {"PQ": None, "SQ": None, "RQ": None, "TP": None, "FP": None, "FN": None,
            "status": "N/A: at least one scene is non-partition"}
    summary = {
        "protocol": protocol.name, "dataset": protocol.dataset,
        "confidence_mode": protocol.confidence_mode,
        "scene_count": len(scenes),
        "CA_AP_uniform": _mean_defined([pooled_ap[f"{t:.2f}"]["ap"] for t in AP_THRESHOLDS]),
        "CA_AP50_uniform": pooled_ap["0.50"]["ap"], "CA_AP25_uniform": pooled_ap["0.25"]["ap"],
        "AP_by_threshold": pooled_ap,
        "CA_PQ": pooled_pq,
        "aggregation": "pooled predictions/GT across scenes; AP across 10 IoU thresholds",
        "per_scene": per_scene,
    }
    if protocol.is_v2:
        prf = [row["CA_PRF1_0_5"] for row in per_scene]
        tp, fp, fn = (sum(row[key] for row in prf) for key in ("TP", "FP", "FN"))
        gt_count = tp + fn
        summary["CA_PRF1_0_5"] = {
            "P": tp / (tp + fp) if gt_count and tp + fp else 0.0 if gt_count else None,
            "R": tp / gt_count if gt_count else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if gt_count else None,
            "TP": tp, "FP": fp, "FN": fn,
            "ignored_prediction_count": sum(row["ignored_prediction_count"] for row in prf),
            "status": "ok" if gt_count else "undefined: no valid GT instances",
        }
        structures = [row["structure"] for row in per_scene]
        gt_total = sum(row["diagnostic_gt_count"] for row in structures)
        pred_total = sum(row["diagnostic_prediction_count"] for row in structures)
        split_total = sum(row["split_gt_count"] for row in structures)
        merge_total = sum(row["merge_prediction_count"] for row in structures)
        duplicate_total = sum(row["duplicate_prediction_count"] for row in structures)
        summary["structure"] = {
            "split_gt_count": split_total, "split_gt_rate": split_total / gt_total if gt_total else None,
            "merge_prediction_count": merge_total,
            "merge_prediction_rate": merge_total / pred_total if pred_total else None,
            "duplicate_prediction_count": duplicate_total,
            "duplicate_prediction_rate": duplicate_total / pred_total if pred_total else None,
            "diagnostic_gt_count": gt_total, "diagnostic_prediction_count": pred_total,
        }
        pq_values = [row["CA_PQ"]["PQ"] for row in per_scene]
        summary["macro_per_scene"] = {
            "CA_AP_uniform": _mean_defined([row["CA_AP_uniform"] for row in per_scene]),
            "CA_AP50_uniform": _mean_defined([row["CA_AP50_uniform"] for row in per_scene]),
            "CA_AP25_uniform": _mean_defined([row["CA_AP25_uniform"] for row in per_scene]),
            "CA_F1_0_5": _mean_defined([row["CA_PRF1_0_5"]["F1"] for row in per_scene]),
            "CA_PQ": sum(pq_values) / len(pq_values) if all(x is not None for x in pq_values) else None,
            "split_gt_rate": _mean_defined([row["structure"]["split_gt_rate"] for row in per_scene]),
            "merge_prediction_rate": _mean_defined([row["structure"]["merge_prediction_rate"] for row in per_scene]),
            "duplicate_prediction_rate": _mean_defined([row["structure"]["duplicate_prediction_rate"] for row in per_scene]),
        }
        summary["aggregation"] = "top-level pooled predictions/GT; macro_per_scene is unweighted scene mean"
    return summary, per_scene, overlaps


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
