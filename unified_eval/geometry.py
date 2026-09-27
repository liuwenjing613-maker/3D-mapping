from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .schema import CanonicalInstance, CanonicalPrediction, EvaluationError


@dataclass
class MappingResult:
    prediction: CanonicalPrediction
    statistics: dict
    distances_m: np.ndarray


def map_point_labels_to_reference(
    pred_xyz: np.ndarray,
    pred_instance_ids: np.ndarray,
    ref_xyz: np.ndarray,
    max_distance_m: float,
    *,
    scene_id: str,
    method_name: str,
    method_commit: str,
    adapter_version: str,
    protocol_version: str,
    confidence_by_id: dict[int, float] | None = None,
    metadata: dict | None = None,
) -> MappingResult:
    """Reference vertex -> nearest native point. Negative labels mean unassigned."""
    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    labels = np.asarray(pred_instance_ids)
    if pred_xyz.ndim != 2 or pred_xyz.shape[1] != 3 or ref_xyz.ndim != 2 or ref_xyz.shape[1] != 3:
        raise EvaluationError("pred_xyz and ref_xyz must have shape [N,3]")
    if len(pred_xyz) != len(labels) or not np.issubdtype(labels.dtype, np.integer):
        raise EvaluationError("Point labels must be integer and match pred_xyz length")
    if not np.isfinite(pred_xyz).all() or not np.isfinite(ref_xyz).all():
        raise EvaluationError("Point coordinates must be finite")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("max_distance_m must be finite and positive")
    native_ids = np.unique(labels[labels >= 0])
    eligible = labels >= 0
    search_xyz = pred_xyz[eligible]
    search_labels = labels[eligible]
    if len(search_xyz):
        distances, nearest = cKDTree(search_xyz).query(ref_xyz, k=1, workers=-1)
        assigned = distances < max_distance_m
    else:
        distances = np.full(len(ref_xyz), np.inf)
        nearest = np.zeros(len(ref_xyz), dtype=np.int64)
        assigned = np.zeros(len(ref_xyz), dtype=bool)
    instances = []
    for native_id in native_ids:
        vertices = np.flatnonzero(assigned & (search_labels[nearest] == native_id)) if len(search_labels) else np.array([], dtype=int)
        if len(vertices):
            instances.append(CanonicalInstance(
                instance_uid=str(int(native_id)), vertex_indices=vertices.astype(np.int32),
                confidence=None if confidence_by_id is None else confidence_by_id.get(int(native_id)),
            ))
    prediction = CanonicalPrediction(
        scene_id=scene_id, reference_vertex_count=len(ref_xyz), instances=instances,
        method_name=method_name, method_commit=method_commit,
        adapter_version=adapter_version, protocol_version=protocol_version,
        is_partition=True, metadata=metadata or {},
    )
    prediction.validate()
    finite = distances[np.isfinite(distances)]
    stats = {
        "num_native_points": int(len(pred_xyz)),
        "num_native_assigned_points": int(len(search_xyz)),
        "num_pred_instances_before_filter": int(len(native_ids)),
        "num_pred_instances_after_filter": len(instances),
        "mapped_ref_vertices": int(assigned.sum()),
        "unmapped_ref_vertices": int(len(ref_xyz) - assigned.sum()),
        "mapping_coverage": float(assigned.mean()) if len(ref_xyz) else None,
        "mean_nn_distance_m": float(finite.mean()) if len(finite) else None,
        "median_nn_distance_m": float(np.median(finite)) if len(finite) else None,
        "p95_nn_distance_m": float(np.percentile(finite, 95)) if len(finite) else None,
        "max_distance_m": float(max_distance_m),
        "mapping_direction": "reference_to_native_prediction",
    }
    return MappingResult(prediction, stats, distances)
