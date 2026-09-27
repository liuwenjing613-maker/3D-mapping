from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def build_overlap(gt: CanonicalGT, pred: CanonicalPrediction, protocol: Protocol,
                  observed_mask: np.ndarray | None = None) -> Overlap:
    gt.validate()
    pred.validate()
    if gt.scene_id != pred.scene_id or gt.vertex_count != pred.reference_vertex_count:
        raise EvaluationError("GT and prediction must use the same scene and reference mesh")
    if pred.protocol_version != protocol.name:
        raise EvaluationError("Prediction protocol_version does not match evaluation protocol")
    valid = np.asarray(gt.valid_vertex_mask, dtype=bool) & ~np.asarray(gt.ignore_vertex_mask, dtype=bool)
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
    pred_uids = []
    active_vertices = []
    dropped = []
    for instance in pred.instances:
        vertices = np.asarray(instance.vertex_indices, dtype=np.int64)
        vertices = vertices[valid[vertices]]
        if len(vertices) < min_size:
            dropped.append(instance.instance_uid)
        else:
            pred_uids.append(instance.instance_uid)
            active_vertices.append(vertices)
    pred_size = np.array([len(v) for v in active_vertices], dtype=np.int64)
    pred_void_fraction = np.array([np.mean(labels[v] < 0) for v in active_vertices], dtype=float)
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
    return Overlap(pred_uids, gt_ids, intersection, iou, precision, recall, pred_size, pred_void_fraction, gt_size,
                   active_vertices, valid, valid & np.isin(labels, gt_ids), small_ids.astype(int).tolist(), dropped)


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
    ignored_pred = int(sum(overlap.pred_void_fraction[r] > void_fraction_threshold
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
        return {"ap": None, "TP": 0, "FP": sum(len(o.pred_uids) for o in overlaps), "FN": 0,
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
            fp_total -= int(sum(overlaps[scene_idx].pred_void_fraction[row] > void_fraction_threshold
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


def diagnostics(overlap: Overlap) -> dict:
    best_iou = overlap.iou.max(axis=0) if overlap.iou.shape[0] else np.zeros(len(overlap.gt_ids))
    gt_recall = overlap.recall.max(axis=0) if overlap.recall.shape[0] else np.zeros(len(overlap.gt_ids))
    purity = overlap.precision.max(axis=1) if overlap.precision.shape[1] else np.zeros(len(overlap.pred_uids))
    gt_vertices = int(overlap.gt_size.sum())
    covered = np.zeros(len(overlap.eval_mask), dtype=bool)
    for vertices in overlap.active_pred_vertices:
        covered[vertices] = True
    matches = _matching(overlap.iou, 0.5, strict=True)
    # This counts coverage of object GT surface, irrespective of which instance claimed it.
    return {
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
