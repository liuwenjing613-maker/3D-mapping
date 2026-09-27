from __future__ import annotations

from .metrics import average_precision, build_overlap, diagnostics, panoptic_quality
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
        per_scene.append({
            "scene_id": gt.scene_id,
            "CA_AP_uniform": _mean_defined([ap[f"{t:.2f}"]["ap"] for t in AP_THRESHOLDS]),
            "CA_AP50_uniform": ap["0.50"]["ap"], "CA_AP25_uniform": ap["0.25"]["ap"],
            "AP_by_threshold": ap,
            "CA_PQ": panoptic_quality(overlap, pred.is_partition,
                protocol.ignore_unmatched_pred_void_fraction_gt),
            "diagnostics": diagnostics(overlap),
        })
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
    return summary, per_scene, overlaps


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
