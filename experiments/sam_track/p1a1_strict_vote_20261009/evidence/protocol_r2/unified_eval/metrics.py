from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import sha256_array, repair_evaluator_code_hashes
from .schema import CanonicalGT, CanonicalPrediction, EvaluationError, Protocol


@dataclass
class Overlap:
    pred_uids: list[str]
    gt_ids: np.ndarray
    intersection: np.ndarray
    iou: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    pred_size: np.ndarray
    pred_void_fraction: np.ndarray
    gt_size: np.ndarray
    active_pred_vertices: list[np.ndarray]
    eval_mask: np.ndarray
    gt_object_mask: np.ndarray
    ignored_small_gt_ids: list[int]
    dropped_prediction_uids: list[str]
    unmatched_ignore_mask: np.ndarray | None = None
    unmatched_policy: list[str] = field(default_factory=list)
    prediction_ignore_fraction: np.ndarray | None = None


def _prediction_ignored(overlap: Overlap, row: int, threshold: float) -> bool:
    if overlap.unmatched_ignore_mask is not None:
        return bool(overlap.unmatched_ignore_mask[row])
    return bool(overlap.pred_void_fraction[row] > threshold)


def _explicit_unmatched_policy(gt: CanonicalGT, instance, target_count: int, background_count: int) -> str:
    meta = instance.metadata
    if target_count:
        return "FP_IF_UNMATCHED_TARGET_SUPPORT"
    if background_count:
        return ("FP_IF_UNMATCHED_OBJECT_ON_BACKGROUND" if meta.get("prediction_type") == "object"
                else "IGNORE_BACKGROUND_ONLY_REPORT")
    if meta.get("native_target_support_count", 0) > 0:
        return "FP_IF_UNMATCHED_NATIVE_TARGET_SUPPORT"
    if meta.get("native_point_count", len(instance.vertex_indices)) == 0:
        return "FP_IF_UNMATCHED_EMPTY_NATIVE_EXPORT"
    if meta.get("native_known_non_target_support_count", 0):
        return ("FP_IF_UNMATCHED_OBJECT_ON_BACKGROUND" if meta.get("prediction_type") == "object"
                else "IGNORE_BACKGROUND_ONLY_REPORT")
    if len(instance.vertex_indices) or meta.get("native_ignore_support_count", 0):
        return "IGNORE_UNKNOWN_OR_UNOBSERVED"
    return "UNVERIFIABLE_OUTSIDE_REFERENCE"


def build_overlap(gt: CanonicalGT, pred: CanonicalPrediction, protocol: Protocol,
                  observed_mask: np.ndarray | None = None) -> Overlap:
    gt.validate()
    pred.validate()
    if gt.scene_id != pred.scene_id or gt.vertex_count != pred.reference_vertex_count:
        raise EvaluationError("GT and prediction must use the same scene and reference mesh")
    if pred.protocol_version != protocol.name:
        raise EvaluationError("Prediction protocol_version does not match evaluation protocol")
    if gt.metadata.get("role") == "raw_gt_qualification_source_only":
        raise EvaluationError("Raw qualification source must be converted to a fixed GT scope before scoring")
    if protocol.is_object_observed_repair:
        if observed_mask is not None:
            raise EvaluationError("Repair observation scope is fixed; build a separate GT scope for a different input prefix")
        if gt.evaluation_region is None or gt.raw_instance_id is None:
            raise EvaluationError("Repair profile requires preserved raw GT and explicit regions")
        if gt.metadata.get("evaluation_profile") != protocol.evaluation_profile or pred.metadata.get("evaluation_profile") != protocol.evaluation_profile:
            raise EvaluationError("GT/prediction evaluation_profile mismatch")
        if protocol.profile_revision >= 2:
            if pred.metadata.get("profile_revision") != protocol.profile_revision or pred.metadata.get(
                    "profile_semantics_sha256") != protocol.profile_semantics_sha256:
                raise EvaluationError("Adapted profile revision/configuration semantics differ from scoring")
            if pred.metadata.get("evaluator_code_sha256") != repair_evaluator_code_hashes():
                raise EvaluationError("Adapted evaluator code hashes differ from the current scoring implementation")
            if not pred.metadata.get("evaluation_only_discrete_gt_oracle") and pred.metadata.get(
                    "duplicate_coordinate_policy") != protocol.duplicate_coordinate_policy:
                raise EvaluationError("Adapted duplicate-coordinate policy differs from scoring")
        elif pred.metadata.get("profile_revision", 1) != 1:
            raise EvaluationError("Revision-2 predictions cannot be scored under the locked revision-1 profile")
        if not pred.metadata.get("evaluation_only_discrete_gt_oracle"):
            if pred.metadata.get("geometry_mapping_method") != "fixed_surface_correspondence" or pred.metadata.get("geometry_mapping_max_distance_m") != protocol.geometry_mapping_max_distance_m:
                raise EvaluationError("Adapted main geometry distance differs from the scoring profile")
            if pred.metadata.get("role") == "structure_diagnostics_only" and pred.metadata.get("diagnostic_mapping_max_distance_m") != protocol.diagnostic_max_distance_m:
                raise EvaluationError("Adapted diagnostic geometry distance differs from the scoring profile")
        for key in ("gt_scope_sha256", "gt_observed_support_sha256", "evaluation_region_sha256", "reference_xyz_sha256"):
            if not gt.metadata.get(key) or gt.metadata[key] != pred.metadata.get(key):
                raise EvaluationError(f"Fixed repair scope/reference mismatch: {key}")
        for key, array in (("gt_observed_support_sha256", gt.observation_count),
                           ("evaluation_region_sha256", gt.evaluation_region),
                           ("reference_xyz_sha256", gt.xyz_ref),
                           ("qualified_instance_sha256", gt.instance_id),
                           ("raw_instance_sha256", gt.raw_instance_id)):
            if array is None or gt.metadata.get(key) != sha256_array(array):
                raise EvaluationError(f"Fixed GT array/hash mismatch: {key}")
    elif gt.evaluation_region is not None or pred.metadata.get("evaluation_profile") == "object_observed_repair":
        raise EvaluationError("Explicit repair regions cannot be scored under the historical profile")
    valid = np.asarray(gt.valid_vertex_mask, dtype=bool) & ~np.asarray(gt.ignore_vertex_mask, dtype=bool)
    if protocol.is_object_observed_repair and not np.array_equal(valid, gt.evaluation_region != 0):
        raise EvaluationError("Valid/ignore masks disagree with fixed evaluation regions")
    if observed_mask is not None:
        if len(observed_mask) != gt.vertex_count:
            raise EvaluationError("Observed mask must match reference vertex count")
        valid &= np.asarray(observed_mask, dtype=bool)
    labels = np.asarray(gt.instance_id, dtype=np.int64)
    candidate_ids, candidate_sizes = np.unique(labels[valid & (labels >= 0)], return_counts=True)
    min_size = protocol.min_valid_instance_vertices
    small_ids = candidate_ids[candidate_sizes < min_size]
    gt_ids = candidate_ids[candidate_sizes >= min_size]
    gt_size = candidate_sizes[candidate_sizes >= min_size].astype(np.int64)
    small_gt_mask = valid & np.isin(labels, small_ids) if protocol.retains_predictions else np.zeros(len(labels), dtype=bool)
    ignored_region = valid & ((labels < 0) | small_gt_mask)
    pred_uids = []
    active_vertices = []
    ignore_fractions = []
    policies = []
    explicit_fractions = []
    dropped = []
    for instance in pred.instances:
        vertices = np.asarray(instance.vertex_indices, dtype=np.int64)
        if protocol.is_object_observed_repair:
            regions = gt.evaluation_region[vertices]
            policies.append(_explicit_unmatched_policy(gt, instance,
                int(np.count_nonzero(regions == 1)), int(np.count_nonzero(regions == 2))))
            explicit_fractions.append(float(np.mean(regions == 0)) if len(regions) else None)
        vertices = vertices[valid[vertices]]
        if protocol.retains_predictions:
            ignore_fraction = float(np.mean(ignored_region[vertices])) if len(vertices) else 0.0
            vertices = vertices[~small_gt_mask[vertices]]
        if not protocol.retains_predictions and len(vertices) < min_size:
            dropped.append(instance.instance_uid)
        else:
            pred_uids.append(instance.instance_uid)
            active_vertices.append(vertices)
            if protocol.retains_predictions:
                ignore_fractions.append(ignore_fraction)
    pred_size = np.array([len(v) for v in active_vertices], dtype=np.int64)
    pred_void_fraction = (np.array(ignore_fractions, dtype=float) if protocol.retains_predictions else
                          np.array([np.mean(labels[v] < 0) for v in active_vertices], dtype=float))
    intersection = np.zeros((len(active_vertices), len(gt_ids)), dtype=np.int64)
    for row, vertices in enumerate(active_vertices):
        gt_labels = labels[vertices]
        positive = np.isin(gt_labels, gt_ids)
        if positive.any():
            columns = np.searchsorted(gt_ids, gt_labels[positive])
            intersection[row] = np.bincount(columns, minlength=len(gt_ids))
    union = pred_size[:, None] + gt_size[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    precision = np.divide(intersection, pred_size[:, None], out=np.zeros_like(intersection, dtype=float), where=pred_size[:, None] > 0)
    recall = np.divide(intersection, gt_size[None, :], out=np.zeros_like(intersection, dtype=float), where=gt_size[None, :] > 0)
    result = Overlap(pred_uids, gt_ids, intersection, iou, precision, recall, pred_size, pred_void_fraction, gt_size,
                   active_vertices, valid, valid & np.isin(labels, gt_ids), small_ids.astype(int).tolist(), dropped)
    if protocol.is_object_observed_repair:
        result.unmatched_policy = policies
        result.unmatched_ignore_mask = np.array([not x.startswith("FP_IF_") for x in policies], dtype=bool)
        result.prediction_ignore_fraction = np.array([np.nan if x is None else x for x in explicit_fractions])
    return result


def instance_precision_recall_f1(overlap: Overlap, void_fraction_threshold: float) -> dict:
    """One-to-one IoU>0.5 counts, independent of the partition/PQ requirement."""
    matches = _matching(overlap.iou, 0.5, strict=True)
    tp = len(matches)
    matched = {r for r, _ in matches}
    ignored = sum(_prediction_ignored(overlap, r, void_fraction_threshold)
                  for r in range(len(overlap.pred_uids)) if r not in matched)
    fp = len(overlap.pred_uids) - tp - ignored
    fn = len(overlap.gt_ids) - tp
    if not len(overlap.gt_ids):
        return {"P": None, "R": None, "F1": None, "TP": tp, "FP": fp, "FN": fn,
                "ignored_prediction_count": int(ignored), "status": "undefined: no valid GT instances"}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"P": precision, "R": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
            "ignored_prediction_count": int(ignored), "status": "ok"}


def significant_structure_diagnostics(overlap: Overlap, protocol: Protocol) -> dict:
    """Classify significant GT-to-prediction edges; duplicate has precedence over split."""
    if not protocol.retains_predictions:
        raise EvaluationError("Significant structure diagnostics require Replica-CA-v2/v3")
    eligible = ((overlap.pred_size > 0) &
                np.array([not _prediction_ignored(overlap, r, protocol.ignore_unmatched_pred_void_fraction_gt)
                          for r in range(len(overlap.pred_uids))], dtype=bool))
    significant = ((overlap.intersection >= protocol.significant_min_intersection_vertices) &
                   (overlap.recall >= protocol.significant_min_gt_fraction) & eligible[:, None])
    matches = _matching(overlap.iou, 0.5, strict=True)
    matched_pred = {row for row, _ in matches}
    matched_gt = {col for _, col in matches}
    duplicates = [row for row in range(len(overlap.pred_uids))
                  if eligible[row] and row not in matched_pred and
                  any(overlap.iou[row, col] > 0.5 for col in matched_gt)]
    duplicate_gt = {col for row in duplicates for col in matched_gt if overlap.iou[row, col] > 0.5}
    split_gt = [col for col in range(len(overlap.gt_ids))
                if np.count_nonzero(significant[:, col]) >= 2 and col not in duplicate_gt]
    merge_pred = [row for row in range(len(overlap.pred_uids))
                  if eligible[row] and np.count_nonzero(significant[row]) >= 2]
    pred_count = int(np.count_nonzero(eligible))
    gt_count = len(overlap.gt_ids)
    return {
        "significant_min_intersection_vertices": protocol.significant_min_intersection_vertices,
        "significant_min_gt_fraction": protocol.significant_min_gt_fraction,
        "significant_predictions_per_gt": np.count_nonzero(significant, axis=0).astype(int).tolist(),
        "significant_gt_per_prediction": np.count_nonzero(significant, axis=1).astype(int).tolist(),
        "split_gt_count": len(split_gt), "split_gt_rate": len(split_gt) / gt_count if gt_count else None,
        "merge_prediction_count": len(merge_pred),
        "merge_prediction_rate": len(merge_pred) / pred_count if pred_count else None,
        "duplicate_prediction_count": len(duplicates),
        "duplicate_prediction_rate": len(duplicates) / pred_count if pred_count else None,
        "diagnostic_gt_count": gt_count, "diagnostic_prediction_count": pred_count,
        "duplicate_gt_count": len(duplicate_gt),
    }


def _matching(matrix: np.ndarray, threshold: float, strict: bool) -> list[tuple[int, int]]:
    """Max cardinality first, then max IoU. Stable under ID permutations."""
    if matrix.size == 0:
        return []
    eligible = matrix > threshold if strict else matrix >= threshold
    # An eligible edge is worth more than every possible IoU tie break combined.
    cardinality_bonus = min(matrix.shape) + 1
    benefit = eligible * (cardinality_bonus + matrix)
    rows, cols = linear_sum_assignment(-benefit)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if eligible[r, c]]


def panoptic_quality(overlap: Overlap, is_partition: bool, void_fraction_threshold: float) -> dict:
    if not is_partition:
        return {"PQ": None, "SQ": None, "RQ": None, "TP": None, "FP": None, "FN": None,
                "status": "N/A: overlapping masks or non-partition prediction"}
    matches = _matching(overlap.iou, 0.5, strict=True)
    tp = len(matches)
    matched_pred = {r for r, _ in matches}
    ignored_pred = int(sum(_prediction_ignored(overlap, r, void_fraction_threshold)
                       for r in range(len(overlap.pred_uids)) if r not in matched_pred))
    fp = len(overlap.pred_uids) - tp - ignored_pred
    fn = len(overlap.gt_ids) - tp
    sq = float(np.mean([overlap.iou[r, c] for r, c in matches])) if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom else None
    return {"PQ": sq * rq if rq is not None else None, "SQ": sq if denom else None,
            "RQ": rq, "TP": tp, "FP": fp, "FN": fn, "ignored_prediction_count": ignored_pred,
            "status": "ok" if denom else "undefined: no instances"}


def _ap_101(precision: np.ndarray, recall: np.ndarray) -> float:
    levels = np.linspace(0.0, 1.0, 101)
    return float(np.mean([precision[recall >= x].max(initial=0.0) for x in levels]))


def average_precision(overlaps: list[Overlap], predictions: list[CanonicalPrediction],
                      confidence_mode: str, threshold: float, void_fraction_threshold: float) -> dict:
    if len(overlaps) != len(predictions):
        raise EvaluationError("Overlap/prediction count differs")
    num_gt = sum(len(o.gt_ids) for o in overlaps)
    if num_gt == 0:
        return {"ap": None, "TP": 0, "FP": sum(sum(not _prediction_ignored(o, r, void_fraction_threshold)
                for r in range(len(o.pred_uids))) for o in overlaps) if any(o.unmatched_ignore_mask is not None for o in overlaps)
                else sum(len(o.pred_uids) for o in overlaps), "FN": 0,
                "status": "undefined: no valid GT instances"}
    # Score ties are evaluated as a group. This avoids an arbitrary UID/scene order
    # deciding uniform-confidence AP. Matching is one-to-one within each scene.
    scored: dict[float, list[tuple[int, int]]] = {}
    for scene_idx, (o, p) in enumerate(zip(overlaps, predictions)):
        instances = {x.instance_uid: x for x in p.instances}
        for row, uid in enumerate(o.pred_uids):
            score = 1.0 if confidence_mode == "uniform" else instances[uid].confidence
            if score is None or not np.isfinite(score):
                raise EvaluationError(f"Native confidence missing for {p.scene_id}/{uid}")
            scored.setdefault(float(score), []).append((scene_idx, row))
    used = [set() for _ in overlaps]
    tp_total = 0
    fp_total = 0
    precision = []
    recall = []
    for score in sorted(scored, reverse=True):
        group = scored[score]
        tp_group = 0
        for scene_idx in sorted({scene for scene, _ in group}):
            rows = [row for scene, row in group if scene == scene_idx]
            available = [col for col in range(len(overlaps[scene_idx].gt_ids)) if col not in used[scene_idx]]
            sub = overlaps[scene_idx].iou[np.ix_(rows, available)]
            matches = _matching(sub, threshold, strict=True)
            for _, col in matches:
                used[scene_idx].add(available[col])
            tp_group += len(matches)
            matched_rows = {rows[r] for r, _ in matches}
            fp_total -= int(sum(_prediction_ignored(overlaps[scene_idx], row, void_fraction_threshold)
                            for row in rows if row not in matched_rows))
        tp_total += tp_group
        fp_total += len(group) - tp_group
        precision.append(tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0)
        recall.append(tp_total / num_gt)
    if not scored:
        precision = [0.0]
        recall = [0.0]
    return {"ap": _ap_101(np.array(precision), np.array(recall)),
            "TP": tp_total, "FP": fp_total, "FN": num_gt - tp_total, "status": "ok"}


def diagnostics(overlap: Overlap, protocol: Protocol | None = None) -> dict:
    best_iou = overlap.iou.max(axis=0) if overlap.iou.shape[0] else np.zeros(len(overlap.gt_ids))
    gt_recall = overlap.recall.max(axis=0) if overlap.recall.shape[0] else np.zeros(len(overlap.gt_ids))
    purity = overlap.precision.max(axis=1) if overlap.precision.shape[1] else np.zeros(len(overlap.pred_uids))
    if protocol is not None and protocol.retains_predictions:
        eligible = ((overlap.pred_size > 0) &
                    np.array([not _prediction_ignored(overlap, r, protocol.ignore_unmatched_pred_void_fraction_gt)
                              for r in range(len(overlap.pred_uids))], dtype=bool))
        purity = purity[eligible]
    gt_vertices = int(overlap.gt_size.sum())
    covered = np.zeros(len(overlap.eval_mask), dtype=bool)
    for vertices in overlap.active_pred_vertices:
        covered[vertices] = True
    matches = _matching(overlap.iou, 0.5, strict=True)
    # This counts coverage of object GT surface, irrespective of which instance claimed it.
    result = {
        "gt_instance_count": len(overlap.gt_ids), "prediction_instance_count": len(overlap.pred_uids),
        "mean_best_gt_iou": float(best_iou.mean()) if len(best_iou) else None,
        "median_best_gt_iou": float(np.median(best_iou)) if len(best_iou) else None,
        "gt_surface_coverage": float(np.count_nonzero(covered & overlap.gt_object_mask) / gt_vertices) if gt_vertices else None,
        "macro_best_gt_recall": float(gt_recall.mean()) if len(gt_recall) else None,
        "macro_prediction_purity": float(purity.mean()) if len(purity) else None,
        "unmatched_gt_count_iou_gt_0_5": len(overlap.gt_ids) - len(matches),
        "unmatched_prediction_count_iou_gt_0_5": len(overlap.pred_uids) - len(matches),
        "best_gt_iou_histogram": {
            "[0,0.25)": int(np.sum(best_iou < 0.25)),
            "[0.25,0.50)": int(np.sum((best_iou >= 0.25) & (best_iou < 0.5))),
            "[0.50,0.75)": int(np.sum((best_iou >= 0.5) & (best_iou < 0.75))),
            "[0.75,1]": int(np.sum(best_iou >= 0.75)),
        },
        "overlapping_predictions_per_gt": np.count_nonzero(overlap.intersection, axis=0).astype(int).tolist(),
        "overlapping_gt_per_prediction": np.count_nonzero(overlap.intersection, axis=1).astype(int).tolist(),
        "ignored_small_gt_ids": overlap.ignored_small_gt_ids,
        "dropped_prediction_uids": overlap.dropped_prediction_uids,
        "total_valid_gt_vertices": gt_vertices,
    }
    if overlap.unmatched_ignore_mask is not None:
        result.update(owner_coverage(overlap))
        policies = overlap.unmatched_policy
        matched = {row for row, _ in matches}
        result["background_only_prediction_count"] = policies.count("IGNORE_BACKGROUND_ONLY_REPORT")
        result["unknown_or_unobserved_prediction_count"] = policies.count("IGNORE_UNKNOWN_OR_UNOBSERVED")
        result["unverifiable_prediction_count"] = policies.count("UNVERIFIABLE_OUTSIDE_REFERENCE")
        result["empty_native_prediction_count"] = policies.count("FP_IF_UNMATCHED_EMPTY_NATIVE_EXPORT")
        result["prediction_scope_audit"] = [{"instance_uid": uid,
            "unmatched_policy": policies[row], "matched_iou_gt_0_5": row in matched,
            "ignored_reference_fraction": float(overlap.prediction_ignore_fraction[row])
                if np.isfinite(overlap.prediction_ignore_fraction[row]) else None}
            for row, uid in enumerate(overlap.pred_uids)]
    return result


def owner_coverage(overlap: Overlap) -> dict:
    """Maximum-intersection one-to-one alignment for scoring; deleting labels cannot increase its optimum."""
    matrix = overlap.intersection
    pairs = []
    if matrix.size:
        rows, cols = linear_sum_assignment(-matrix)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if matrix[r, c] > 0]
    correct = int(sum(matrix[r, c] for r, c in pairs))
    assigned = int(matrix.sum())
    total = int(overlap.gt_size.sum())
    if assigned > total:
        raise EvaluationError("Owner coverage requires a partition prediction")
    return {"correct_owner_vertices": correct, "wrong_owner_vertices": assigned - correct,
        "unpredicted_target_vertices": total - assigned,
        "Correct_owner_Coverage": correct / total if total else None,
        "Wrong_owner_Coverage": (assigned - correct) / total if total else None,
        "Unpredicted_Coverage": (total - assigned) / total if total else None,
        "owner_alignment": "optimal_one_to_one_maximum_intersection_for_scoring_only",
        "scoring_owner_alignment": [{"instance_uid": overlap.pred_uids[r], "raw_gt_id": int(overlap.gt_ids[c])} for r, c in pairs]}
