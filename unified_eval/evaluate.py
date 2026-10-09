from __future__ import annotations

from .metrics import (average_precision, build_overlap, diagnostics,
                      instance_precision_recall_f1, panoptic_quality,
                      significant_structure_diagnostics)
from .schema import CanonicalGT, CanonicalPrediction, Protocol


AP_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]


def evaluate_scenes(scenes: list[tuple[CanonicalGT, CanonicalPrediction]], protocol: Protocol,
                    observed_masks: list | None = None,
                    diagnostic_predictions: list[CanonicalPrediction] | None = None) -> tuple[dict, list[dict], list]:
    if not scenes:
        raise ValueError("At least one scene is required")
    if observed_masks is not None and len(observed_masks) != len(scenes):
        raise ValueError("Observed mask count must match scene count")
    if diagnostic_predictions is not None and len(diagnostic_predictions) != len(scenes):
        raise ValueError("Diagnostic prediction count must match scene count")
    if protocol.is_v3 and diagnostic_predictions is not None:
        for (_, pred), diag in zip(scenes, diagnostic_predictions):
            if not pred.is_partition or pred.metadata.get("role") == "structure_diagnostics_only":
                raise ValueError("v3 main prediction must be a competitive partition")
            if diag.metadata.get("role") != "structure_diagnostics_only":
                raise ValueError("v3 diagnostic prediction must carry the diagnostic role")
            if [x.instance_uid for x in pred.instances] != [x.instance_uid for x in diag.instances]:
                raise ValueError("Diagnostic prediction must retain the same native instances and order")
    elif protocol.is_v3 and any(not pred.is_partition or
             pred.metadata.get("role") == "structure_diagnostics_only" for _, pred in scenes):
        raise ValueError("v3 main prediction must be a competitive partition")
    overlaps = [build_overlap(gt, pred, protocol, None if observed_masks is None else observed_masks[i])
                for i, (gt, pred) in enumerate(scenes)]
    diagnostic_overlaps = ([build_overlap(gt, diag, protocol,
        None if observed_masks is None else observed_masks[i])
        for i, ((gt, _), diag) in enumerate(zip(scenes, diagnostic_predictions))]
        if protocol.is_v3 and diagnostic_predictions is not None else None)
    predictions = [p for _, p in scenes]
    thresholds = [0.25] + AP_THRESHOLDS
    pooled_ap = {f"{t:.2f}": average_precision(overlaps, predictions, protocol.confidence_mode, t,
        protocol.ignore_unmatched_pred_void_fraction_gt) for t in thresholds}
    per_scene = []
    for scene_index, ((gt, pred), overlap) in enumerate(zip(scenes, overlaps)):
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
        if protocol.retains_predictions:
            row["CA_PRF1_0_5"] = instance_precision_recall_f1(
                overlap, protocol.ignore_unmatched_pred_void_fraction_gt)
            row["structure"] = (significant_structure_diagnostics(
                diagnostic_overlaps[scene_index] if protocol.is_v3 else overlap, protocol)
                if not protocol.is_v3 or diagnostic_overlaps is not None else None)
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
    if protocol.retains_predictions:
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
        structures = [row["structure"] for row in per_scene if row["structure"] is not None]
        if protocol.is_v3 and not structures:
            summary["structure"] = None
            summary["structure_status"] = "N/A: pairwise native geometry support not provided"
        else:
            summary["structure_status"] = "ok"
        gt_total = sum(row["diagnostic_gt_count"] for row in structures)
        pred_total = sum(row["diagnostic_prediction_count"] for row in structures)
        split_total = sum(row["split_gt_count"] for row in structures)
        merge_total = sum(row["merge_prediction_count"] for row in structures)
        duplicate_total = sum(row["duplicate_prediction_count"] for row in structures)
        structure_summary = {
            "split_gt_count": split_total, "split_gt_rate": split_total / gt_total if gt_total else None,
            "merge_prediction_count": merge_total,
            "merge_prediction_rate": merge_total / pred_total if pred_total else None,
            "duplicate_prediction_count": duplicate_total,
            "duplicate_prediction_rate": duplicate_total / pred_total if pred_total else None,
            "diagnostic_gt_count": gt_total, "diagnostic_prediction_count": pred_total,
        }
        if structures:
            summary["structure"] = structure_summary
        pq_values = [row["CA_PQ"]["PQ"] for row in per_scene]
        summary["macro_per_scene"] = {
            "CA_AP_uniform": _mean_defined([row["CA_AP_uniform"] for row in per_scene]),
            "CA_AP50_uniform": _mean_defined([row["CA_AP50_uniform"] for row in per_scene]),
            "CA_AP25_uniform": _mean_defined([row["CA_AP25_uniform"] for row in per_scene]),
            "CA_F1_0_5": _mean_defined([row["CA_PRF1_0_5"]["F1"] for row in per_scene]),
            "CA_PQ": sum(pq_values) / len(pq_values) if all(x is not None for x in pq_values) else None,
            "split_gt_rate": _mean_defined([row["structure"]["split_gt_rate"] for row in per_scene if row["structure"] is not None]),
            "merge_prediction_rate": _mean_defined([row["structure"]["merge_prediction_rate"] for row in per_scene if row["structure"] is not None]),
            "duplicate_prediction_rate": _mean_defined([row["structure"]["duplicate_prediction_rate"] for row in per_scene if row["structure"] is not None]),
        }
        summary["aggregation"] = "top-level pooled predictions/GT; macro_per_scene is unweighted scene mean"
    if protocol.is_object_observed_repair:
        summary["evaluation_profile"] = protocol.evaluation_profile
        summary["status"] = "DEVELOPMENT / NON_OFFICIAL / PROFILE_NOT_FROZEN"
        summary["reference_scope"] = [{"scene_id": gt.scene_id,
            "gt_scope_sha256": gt.metadata["gt_scope_sha256"],
            "gt_observed_support_sha256": gt.metadata["gt_observed_support_sha256"],
            "evaluation_region_sha256": gt.metadata["evaluation_region_sha256"]} for gt, _ in scenes]
        total = sum(int(o.gt_size.sum()) for o in overlaps)
        coverage_keys = ("correct_owner_vertices", "wrong_owner_vertices", "unpredicted_target_vertices")
        counts = {key: sum(row["diagnostics"][key] for row in per_scene) for key in coverage_keys}
        no_geometry = sum(pred.metadata.get("no_geometry_target_vertex_count", 0) for _, pred in scenes)
        if no_geometry > counts["unpredicted_target_vertices"]:
            raise ValueError("NO_GEOMETRY cannot exceed unpredicted target surface")
        summary["owner_surface"] = {**counts, "target_vertices": total,
            "no_geometry_target_vertices": no_geometry,
            "unassigned_with_geometry_target_vertices": counts["unpredicted_target_vertices"] - no_geometry,
            "Correct_owner_Coverage": counts["correct_owner_vertices"] / total if total else None,
            "Wrong_owner_Coverage": counts["wrong_owner_vertices"] / total if total else None,
            "Unassigned_Coverage": (counts["unpredicted_target_vertices"] - no_geometry) / total if total else None,
            "No_geometry_Coverage": no_geometry / total if total else None,
            "owner_alignment": "optimal_one_to_one_maximum_intersection_for_scoring_only"}
        summary["CA_mCov"] = sum(float(o.iou.max(axis=0).sum()) if len(o.pred_uids) else 0.0
            for o in overlaps) / sum(len(o.gt_ids) for o in overlaps) if any(len(o.gt_ids) for o in overlaps) else None
        for key in ("background_only_prediction_count", "unknown_or_unobserved_prediction_count", "unverifiable_prediction_count", "empty_native_prediction_count"):
            summary[key] = sum(row["diagnostics"][key] for row in per_scene)
    return summary, per_scene, overlaps


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
