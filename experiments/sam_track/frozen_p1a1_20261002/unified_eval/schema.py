from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class EvaluationError(ValueError):
    """Invalid protocol or canonical data."""


@dataclass
class CanonicalGT:
    scene_id: str
    xyz_ref: np.ndarray
    instance_id: np.ndarray
    semantic_id: np.ndarray
    valid_vertex_mask: np.ndarray
    ignore_vertex_mask: np.ndarray
    observation_count: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.instance_id)

    def validate(self) -> None:
        n = self.vertex_count
        if not self.scene_id or self.xyz_ref.shape != (n, 3):
            raise EvaluationError("GT scene_id or xyz_ref shape is invalid")
        if not np.isfinite(self.xyz_ref).all():
            raise EvaluationError("GT xyz_ref contains non-finite coordinates")
        for name in ("semantic_id", "valid_vertex_mask", "ignore_vertex_mask"):
            if len(getattr(self, name)) != n:
                raise EvaluationError(f"GT {name} length differs from reference vertices")
        if self.observation_count is not None and len(self.observation_count) != n:
            raise EvaluationError("GT observation_count length differs from reference vertices")


@dataclass
class CanonicalInstance:
    instance_uid: str
    vertex_indices: np.ndarray
    confidence: float | None = None
    semantic_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalPrediction:
    scene_id: str
    reference_vertex_count: int
    instances: list[CanonicalInstance]
    method_name: str
    method_commit: str
    adapter_version: str
    protocol_version: str
    is_partition: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.scene_id or self.reference_vertex_count < 0:
            raise EvaluationError("Prediction scene_id or reference_vertex_count is invalid")
        if not self.method_name or not self.adapter_version or not self.protocol_version:
            raise EvaluationError("Prediction method, adapter and protocol versions are required")
        seen_uids: set[str] = set()
        ownership = np.zeros(self.reference_vertex_count, dtype=np.uint8) if self.is_partition else None
        for instance in self.instances:
            if not instance.instance_uid or instance.instance_uid in seen_uids:
                raise EvaluationError("Instance UIDs must be nonempty and unique")
            seen_uids.add(instance.instance_uid)
            vertices = np.asarray(instance.vertex_indices)
            if vertices.ndim != 1 or not np.issubdtype(vertices.dtype, np.integer):
                raise EvaluationError(f"{instance.instance_uid}: vertex_indices must be a 1D integer array")
            if len(vertices) and (vertices.min() < 0 or vertices.max() >= self.reference_vertex_count):
                raise EvaluationError(f"{instance.instance_uid}: vertex index outside reference mesh")
            if len(np.unique(vertices)) != len(vertices):
                raise EvaluationError(f"{instance.instance_uid}: duplicate vertex index")
            if instance.confidence is not None and not np.isfinite(instance.confidence):
                raise EvaluationError(f"{instance.instance_uid}: confidence is not finite")
            if ownership is not None:
                if np.any(ownership[vertices]):
                    raise EvaluationError("is_partition=true but prediction masks overlap")
                ownership[vertices] = 1


@dataclass(frozen=True)
class Protocol:
    name: str
    dataset: str
    confidence_mode: str
    min_valid_instance_vertices: int
    geometry_mapping_max_distance_m: float | None
    ignore_unmatched_pred_void_fraction_gt: float
    ap_policy: str = "replica_ca_101"
    pq_iou_strictly_greater_than: float = 0.5
    geometry_mapping_method: str = "nearest_neighbor_reference_to_prediction"
    diagnostic_max_distance_m: float | None = None
    prediction_min_valid_vertices: int | None = None
    significant_min_intersection_vertices: int | None = None
    significant_min_gt_fraction: float | None = None

    @property
    def is_v2(self) -> bool:
        return self.name == "Replica-CA-v2"

    @property
    def is_v3(self) -> bool:
        return self.name == "Replica-CA-v3"

    @property
    def retains_predictions(self) -> bool:
        return self.is_v2 or self.is_v3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Protocol":
        geometry = data.get("geometry_mapping", {})
        instance_filter = data.get("instance_filter", {})
        ignore_policy = data.get("ignore_policy", {})
        if "max_distance_m" not in geometry or geometry["max_distance_m"] is None:
            raise EvaluationError("geometry_mapping.max_distance_m must be explicitly set")
        if "min_valid_instance_vertices" not in instance_filter or instance_filter["min_valid_instance_vertices"] is None:
            raise EvaluationError("instance_filter.min_valid_instance_vertices must be explicitly set")
        if "unmatched_prediction_void_fraction_gt" not in ignore_policy:
            raise EvaluationError("ignore_policy.unmatched_prediction_void_fraction_gt must be explicitly set")
        method = geometry.get("method")
        allowed_methods = {"nearest_neighbor_reference_to_prediction",
                           "independent_nearest_neighbor_per_instance",
                           "competitive_nearest_instance"}
        if method not in allowed_methods:
            raise EvaluationError("Unsupported geometry mapping method")
        is_v2 = data.get("name") == "Replica-CA-v2"
        is_v3 = data.get("name") == "Replica-CA-v3"
        if is_v2 and method != "independent_nearest_neighbor_per_instance":
            raise EvaluationError("Replica-CA-v2 requires independent per-instance projection")
        if is_v3 and method != "competitive_nearest_instance":
            raise EvaluationError("Replica-CA-v3 requires competitive nearest-instance projection")
        if not (is_v2 or is_v3) and method != "nearest_neighbor_reference_to_prediction":
            raise EvaluationError("This geometry projection requires Replica-CA-v2 or v3")
        diag_distance = data.get("diagnostic_mapping", {}).get("max_distance_m") if is_v3 else None
        if is_v3 and data.get("diagnostic_mapping", {}).get("method") != "pairwise_geometry_support":
            raise EvaluationError("Replica-CA-v3 requires pairwise_geometry_support diagnostics")
        if is_v3 and diag_distance is None:
            raise EvaluationError("diagnostic_mapping.max_distance_m must be explicitly set")
        significant = data.get("significant_overlap", {})
        if (is_v2 or is_v3) and (significant.get("min_intersection_vertices") is None or
                      significant.get("min_gt_fraction") is None):
            raise EvaluationError("Replica-CA-v2/v3 significant_overlap thresholds must be explicitly set")
        pred_min = instance_filter.get("min_prediction_vertices") if is_v2 or is_v3 else None
        if (is_v2 or is_v3) and pred_min is None:
            raise EvaluationError("Replica-CA-v2/v3 instance_filter.min_prediction_vertices must be explicit")
        obj = cls(
            name=str(data["name"]), dataset=str(data["dataset"]),
            confidence_mode=str(data["confidence_mode"]),
            min_valid_instance_vertices=int(instance_filter["min_valid_instance_vertices"]),
            geometry_mapping_max_distance_m=float(geometry["max_distance_m"]),
            ignore_unmatched_pred_void_fraction_gt=float(ignore_policy["unmatched_prediction_void_fraction_gt"]),
            ap_policy=str(data.get("ap_policy", "replica_ca_101")),
            geometry_mapping_method=method,
            diagnostic_max_distance_m=float(diag_distance) if is_v3 else None,
            prediction_min_valid_vertices=int(pred_min) if is_v2 or is_v3 else None,
            significant_min_intersection_vertices=int(significant["min_intersection_vertices"]) if is_v2 or is_v3 else None,
            significant_min_gt_fraction=float(significant["min_gt_fraction"]) if is_v2 or is_v3 else None,
        )
        if obj.confidence_mode != "uniform":
            raise EvaluationError("replica_ca_101 requires uniform confidence; use the separate official-native runner for official AP")
        if obj.min_valid_instance_vertices < 1:
            raise EvaluationError("Minimum instance sizes must be positive")
        if not np.isfinite(obj.geometry_mapping_max_distance_m) or obj.geometry_mapping_max_distance_m <= 0:
            raise EvaluationError("Mapping distance must be finite and positive")
        if not 0 <= obj.ignore_unmatched_pred_void_fraction_gt <= 1:
            raise EvaluationError("Void fraction threshold must lie in [0,1]")
        if obj.ap_policy != "replica_ca_101":
            raise EvaluationError("Only replica_ca_101 is implemented; official compatibility is pending")
        if is_v2 or is_v3:
            if obj.prediction_min_valid_vertices != 0:
                raise EvaluationError("Replica-CA-v2/v3 must retain every prediction, including empty and small masks")
            if obj.significant_min_intersection_vertices < 1:
                raise EvaluationError("Significant overlap vertex threshold must be positive")
            if not np.isfinite(obj.significant_min_gt_fraction) or not 0 < obj.significant_min_gt_fraction <= 1:
                raise EvaluationError("Significant overlap GT fraction must lie in (0,1]")
        if is_v3 and (not np.isfinite(obj.diagnostic_max_distance_m) or obj.diagnostic_max_distance_m <= 0):
            raise EvaluationError("Diagnostic mapping distance must be finite and positive")
        return obj
