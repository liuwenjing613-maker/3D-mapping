"""GT-only geometry calibration for Replica-CA-v2.

These diagnostics use no method predictions and must never be reported as
benchmark scores. They deliberately leave the protocol unfrozen.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .evaluate import evaluate_scenes
from .geometry import map_instances_to_reference_v2, map_instances_to_reference_v3
from .schema import CanonicalGT, EvaluationError, Protocol


def eligible_oracle_ids(gt: CanonicalGT, min_vertices: int) -> np.ndarray:
    gt.validate()
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask & (gt.instance_id >= 0)
    ids, sizes = np.unique(gt.instance_id[valid], return_counts=True)
    return ids[sizes >= min_vertices]


def oracle_clouds(gt: CanonicalGT, ids: np.ndarray, *, sigma_m: float = 0.0,
                  keep_fraction: float = 1.0, seed: int = 0) -> list[np.ndarray]:
    """Exact GT vertices, optionally perturbed or deterministically thinned."""
    if not np.isfinite(sigma_m) or sigma_m < 0:
        raise EvaluationError("oracle sigma_m must be finite and nonnegative")
    if not np.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise EvaluationError("oracle keep_fraction must lie in (0,1]")
    rng = np.random.default_rng(seed)
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    clouds = []
    for gt_id in ids:
        xyz = np.asarray(gt.xyz_ref[valid & (gt.instance_id == gt_id)], dtype=np.float64)
        if keep_fraction < 1:
            chosen = rng.choice(len(xyz), size=max(1, int(np.ceil(len(xyz) * keep_fraction))),
                                replace=False)
            xyz = xyz[np.sort(chosen)]
        if sigma_m:
            xyz = xyz + rng.normal(0.0, sigma_m, size=xyz.shape)
        clouds.append(xyz)
    return clouds


def evaluate_oracle(gt: CanonicalGT, protocol: Protocol, ids: np.ndarray,
                    clouds: list[np.ndarray], *, delta_m: float) -> dict:
    """Run the actual mapper and metric engine, with GT only as synthetic input."""
    if not protocol.retains_predictions or len(ids) != len(clouds):
        raise EvaluationError("Oracle requires v2/v3 and one cloud per GT ID")
    effective = replace(protocol, geometry_mapping_max_distance_m=float(delta_m))
    mapper = map_instances_to_reference_v3 if protocol.is_v3 else map_instances_to_reference_v2
    mapped = mapper(clouds, gt.xyz_ref, delta_m,
        scene_id=gt.scene_id, method_name="GT geometry oracle",
        method_commit="DIAGNOSTIC_ONLY", adapter_version="oracle_calibration",
        protocol_version=effective.name)
    summary, _, overlaps = evaluate_scenes([(gt, mapped.prediction)], effective)
    overlap = overlaps[0]
    gt_columns = {int(gt_id): col for col, gt_id in enumerate(overlap.gt_ids)}
    own_iou = []
    foreign_vertices = 0
    countable_labels = np.isin(gt.instance_id, overlap.gt_ids)
    for row, gt_id in enumerate(ids):
        col = gt_columns[int(gt_id)]
        own_iou.append(float(overlap.iou[row, col]))
        vertices = mapped.prediction.instances[row].vertex_indices
        foreign_vertices += int(np.count_nonzero(
            countable_labels[vertices] & (gt.instance_id[vertices] != gt_id)))
    prf = summary["CA_PRF1_0_5"]
    return {
        "scene_id": gt.scene_id, "delta_m": float(delta_m),
        "oracle_instance_count": len(ids),
        "AP50": summary["CA_AP50_uniform"], "F1_0_5": prf["F1"],
        "TP": int(prf["TP"]), "FP": int(prf["FP"]), "FN": int(prf["FN"]),
        "self_iou_min": min(own_iou) if own_iou else None,
        "self_iou_mean": float(np.mean(own_iou)) if own_iou else None,
        "self_iou_le_0p5_gt_ids": [int(ids[i]) for i, value in enumerate(own_iou)
                                    if value <= 0.5],
        "foreign_instance_vertex_claims": foreign_vertices,
        "overlapped_ref_vertices": mapped.statistics.get("overlapped_ref_vertices", 0),
        "PQ": summary["CA_PQ"]["PQ"],
        "PQ_status": summary["CA_PQ"]["status"],
    }


def oracle_identity_pass(row: dict) -> bool:
    """Required AP/F1 gate; PQ is undefined when projected masks overlap."""
    count = row["oracle_instance_count"]
    return (row["TP"] == count and row["FP"] == 0 and row["FN"] == 0
            and row["AP50"] == 1.0 and row["F1_0_5"] == 1.0)
