from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .schema import CanonicalInstance, CanonicalPrediction, EvaluationError
from .io import sha256_array


@dataclass
class MappingResult:
    prediction: CanonicalPrediction
    statistics: dict
    distances_m: np.ndarray
    diagnostic_prediction: CanonicalPrediction | None = None


@dataclass(frozen=True)
class FixedSurfaceCorrespondence:
    nearest_native_index: np.ndarray
    distances_m: np.ndarray
    native_xyz_sha256: str
    reference_xyz_sha256: str
    native_point_count: int
    max_distance_m: float

    @property
    def sha256(self) -> str:
        value = {"native_xyz_sha256": self.native_xyz_sha256,
            "reference_xyz_sha256": self.reference_xyz_sha256,
            "native_point_count": self.native_point_count,
            "max_distance_m": self.max_distance_m,
            "index_sha256": sha256_array(self.nearest_native_index),
            "distances_sha256": sha256_array(self.distances_m)}
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate(self, native_xyz: np.ndarray, ref_xyz: np.ndarray) -> None:
        if sha256_array(native_xyz) != self.native_xyz_sha256 or sha256_array(ref_xyz) != self.reference_xyz_sha256:
            raise EvaluationError("Fixed-surface geometry/reference hash mismatch")
        n = len(ref_xyz)
        if self.native_point_count != len(native_xyz) or self.nearest_native_index.shape != (n,) or self.distances_m.shape != (n,):
            raise EvaluationError("Fixed correspondence dimensions are invalid")
        indices = self.nearest_native_index
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < -1) or np.any(indices >= len(native_xyz)):
            raise EvaluationError("Fixed correspondence index outside native geometry")
        if not np.isfinite(self.max_distance_m) or self.max_distance_m <= 0:
            raise EvaluationError("Correspondence distance must be finite and positive")
        if np.any(np.isnan(self.distances_m)) or np.any(self.distances_m < 0):
            raise EvaluationError("Correspondence distances are invalid")
        if not np.array_equal(indices >= 0, self.distances_m < self.max_distance_m):
            raise EvaluationError("Fixed correspondence violates its strict geometry gate")


def build_fixed_surface_correspondence(native_xyz: np.ndarray, ref_xyz: np.ndarray,
                                       max_distance_m: float) -> FixedSurfaceCorrespondence:
    """One immutable geometry query including every unlabeled/conflict/insufficient point."""
    native, ref = np.asarray(native_xyz), np.asarray(ref_xyz)
    for name, array in (("native_xyz", native), ("ref_xyz", ref)):
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise EvaluationError(f"{name} must be finite [N,3]")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("max_distance_m must be finite and positive")
    nearest = np.full(len(ref), -1, dtype=np.int64)
    distances = np.full(len(ref), np.inf, dtype=np.float64)
    if len(native) and len(ref):
        # Coincident native points have a fixed first source index, independent of labels.
        unique, first = np.unique(native, axis=0, return_index=True)
        distances, query = cKDTree(unique).query(ref, k=1, workers=-1)
        matched = distances < max_distance_m
        nearest[matched] = first[query[matched]]
    nearest.setflags(write=False)
    distances.setflags(write=False)
    result = FixedSurfaceCorrespondence(nearest, distances, sha256_array(native),
        sha256_array(ref), len(native), float(max_distance_m))
    result.validate(native, ref)
    return result


def save_fixed_surface_correspondence(path: str | Path, value: FixedSurfaceCorrespondence) -> None:
    np.savez_compressed(path, nearest_native_index=value.nearest_native_index,
        distances_m=value.distances_m, native_xyz_sha256=value.native_xyz_sha256,
        reference_xyz_sha256=value.reference_xyz_sha256, native_point_count=value.native_point_count,
        max_distance_m=value.max_distance_m, correspondence_sha256=value.sha256)


def load_fixed_surface_correspondence(path: str | Path, native_xyz: np.ndarray,
                                      ref_xyz: np.ndarray) -> FixedSurfaceCorrespondence:
    with np.load(path, allow_pickle=False) as data:
        result = FixedSurfaceCorrespondence(data["nearest_native_index"].copy(), data["distances_m"].copy(),
            str(data["native_xyz_sha256"].item()), str(data["reference_xyz_sha256"].item()),
            int(data["native_point_count"].item()), float(data["max_distance_m"].item()))
        if result.sha256 != str(data["correspondence_sha256"].item()):
            raise EvaluationError("Correspondence cache checksum mismatch")
    result.validate(np.asarray(native_xyz), np.asarray(ref_xyz))
    result.nearest_native_index.setflags(write=False)
    result.distances_m.setflags(write=False)
    return result


def map_fixed_surface_labels(native_xyz: np.ndarray, labels: np.ndarray, ref_xyz: np.ndarray,
                             correspondence: FixedSurfaceCorrespondence, *, scene_id: str,
                             method_name: str, method_commit: str,
                             native_instance_ids: np.ndarray | None = None,
                             diagnostic_distance_m: float | None = None,
                             metadata: dict | None = None) -> MappingResult:
    """Read labels from fixed native indices; never rebuild the tree after a revision."""
    native_xyz, ref_xyz, labels = np.asarray(native_xyz), np.asarray(ref_xyz), np.asarray(labels)
    correspondence.validate(native_xyz, ref_xyz)
    if labels.shape != (len(native_xyz),) or not np.issubdtype(labels.dtype, np.integer):
        raise EvaluationError("Native labels must be an integer vector matching complete geometry")
    positive = np.unique(labels[labels > 0])
    native_ids = positive if native_instance_ids is None else np.asarray(native_instance_ids)
    if native_ids.ndim != 1 or not np.issubdtype(native_ids.dtype, np.integer) or np.any(native_ids <= 0):
        raise EvaluationError("Exported native instance IDs must be positive integers")
    if len(np.unique(native_ids)) != len(native_ids) or not np.isin(positive, native_ids).all():
        raise EvaluationError("Exported inventory must be unique and include every positive native label")
    reference_labels = np.full(len(ref_xyz), -1, dtype=np.int64)
    matched = correspondence.nearest_native_index >= 0
    reference_labels[matched] = labels[correspondence.nearest_native_index[matched]]
    reference_labels[reference_labels <= 0] = -1
    instances = [CanonicalInstance(str(int(native_id)),
        np.flatnonzero(reference_labels == native_id).astype(np.int32),
        metadata={"native_instance_id": int(native_id),
                  "native_point_count": int(np.count_nonzero(labels == native_id))}) for native_id in native_ids]
    meta = {**(metadata or {}), "evaluation_profile": "object_observed_repair",
            "geometry_xyz_sha256": correspondence.native_xyz_sha256,
            "reference_xyz_sha256": correspondence.reference_xyz_sha256,
            "correspondence_sha256": correspondence.sha256,
            "geometry_mapping_method": "fixed_surface_correspondence", "mapping_uses_gt_labels": False,
            "geometry_mapping_max_distance_m": correspondence.max_distance_m,
            "diagnostic_mapping_max_distance_m": diagnostic_distance_m,
            "instance_inventory_source": "explicit_export" if native_instance_ids is not None else "positive_labels_in_export"}
    prediction = CanonicalPrediction(scene_id, len(ref_xyz), instances, method_name,
        method_commit, "fixed_complete_surface_v1", "Replica-CA-v3", True, meta)
    prediction.validate()
    diagnostic = None
    if diagnostic_distance_m is not None:
        clouds = [native_xyz[labels == native_id] for native_id in native_ids]
        diagnostic = pairwise_geometry_support(clouds, ref_xyz, diagnostic_distance_m,
            scene_id=scene_id, method_name=method_name, method_commit=method_commit,
            adapter_version="fixed_complete_surface_v1", protocol_version="Replica-CA-v3",
            metadata={**meta, "role": "structure_diagnostics_only"})
        for instance, native_id, main in zip(diagnostic.instances, native_ids, instances):
            instance.instance_uid = str(int(native_id))
            instance.metadata = dict(main.metadata)
        diagnostic.validate()
    statistics = {"geometry_point_count": len(native_xyz), "positive_native_label_count": int(np.count_nonzero(labels > 0)),
        "nonpositive_native_state_counts": {str(int(x)): int(np.count_nonzero(labels == x)) for x in np.unique(labels[labels <= 0])},
        "no_geometry_ref_vertices": int(np.count_nonzero(~matched)),
        "unassigned_with_geometry_ref_vertices": int(np.count_nonzero(matched & (reference_labels <= 0))),
        "assigned_ref_vertices": int(np.count_nonzero(reference_labels > 0)),
        "exported_prediction_count": len(instances),
        "empty_native_exports": sum(x.metadata["native_point_count"] == 0 for x in instances),
        "empty_projected_predictions": sum(len(x.vertex_indices) == 0 for x in instances),
        "geometry_xyz_sha256": correspondence.native_xyz_sha256,
        "correspondence_sha256": correspondence.sha256}
    return MappingResult(prediction, statistics, correspondence.distances_m, diagnostic)


def native_reference_region_support(native_xyz: np.ndarray, ref_xyz: np.ndarray,
                                    reference_regions: np.ndarray, max_distance_m: float) -> np.ndarray:
    """Geometry-only evidence for empty projections; regions classify FP, never ownership."""
    native = np.asarray(native_xyz)
    ref = np.asarray(ref_xyz)
    regions = np.asarray(reference_regions)
    if regions.shape != (len(ref),) or not np.isin(regions, [0, 1, 2]).all():
        raise EvaluationError("Native support requires explicit reference regions")
    support = np.full(len(native), -1, dtype=np.int8)  # outside all reference support
    for region in (0, 2, 1):  # near TARGET takes precedence for duplicate evidence
        selected = ref[regions == region]
        if len(selected) and len(native):
            distance, _ = cKDTree(selected).query(native, k=1, workers=-1)
            support[distance < max_distance_m] = region
    return support


def attach_reference_support(result: MappingResult, labels: np.ndarray, native_support: np.ndarray,
                             gt, correspondence: FixedSurfaceCorrespondence) -> None:
    """Bind post-hoc support evidence to this fixed GT scope and observation mask."""
    if gt.evaluation_region is None or len(labels) != len(native_support):
        raise EvaluationError("Support dimensions or GT regions missing")
    targets = gt.evaluation_region == 1
    no_geometry = int(np.count_nonzero(targets & (correspondence.nearest_native_index < 0)))
    for prediction in (result.prediction, result.diagnostic_prediction):
        if prediction is None:
            continue
        prediction.metadata.update({"gt_scope_sha256": gt.metadata["gt_scope_sha256"],
            "gt_observed_support_sha256": gt.metadata["gt_observed_support_sha256"],
            "evaluation_region_sha256": gt.metadata["evaluation_region_sha256"],
            "no_geometry_target_vertex_count": no_geometry})
        for instance in prediction.instances:
            selected = labels == instance.metadata["native_instance_id"]
            for region, name in ((1, "target"), (2, "known_non_target"), (0, "ignore"), (-1, "outside")):
                instance.metadata["native_" + name + "_support_count"] = int(np.count_nonzero(selected & (native_support == region)))


def _validated_clouds(point_clouds: list[np.ndarray]) -> list[np.ndarray]:
    clouds = []
    for index, points in enumerate(point_clouds):
        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise EvaluationError(f"Native object {index} points must be finite [N,3]")
        clouds.append(xyz)
    return clouds


def pairwise_geometry_support(point_clouds: list[np.ndarray], ref_xyz: np.ndarray,
                              max_distance_m: float, *, scene_id: str,
                              method_name: str, method_commit: str,
                              adapter_version: str, protocol_version: str,
                              metadata: dict | None = None) -> CanonicalPrediction:
    """Independent native-to-reference support, exclusively for structure diagnostics.

    This may overlap and must never be passed to the main AP/PQ calculation.
    """
    return map_instances_to_reference_v2(point_clouds, ref_xyz, max_distance_m,
        scene_id=scene_id, method_name=method_name, method_commit=method_commit,
        adapter_version=adapter_version + "_pairwise_support",
        protocol_version=protocol_version, metadata=metadata).prediction


def map_instances_to_reference_v3(
    point_clouds: list[np.ndarray], ref_xyz: np.ndarray, max_distance_m: float, *,
    scene_id: str, method_name: str, method_commit: str, adapter_version: str,
    protocol_version: str, confidence_by_index: dict[int, float] | None = None,
    diagnostic_distance_m: float | None = None, metadata: dict | None = None,
) -> MappingResult:
    """One nearest native instance per reference vertex; retain empty instances.

    Exact coordinate ties are resolved by a content-derived cloud ordering, so
    source file, object ID and point order do not alter the metric values.
    """
    ref = np.asarray(ref_xyz, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all():
        raise EvaluationError("ref_xyz must be finite and have shape [N,3]")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("max_distance_m must be finite and positive")
    clouds = _validated_clouds(point_clouds)
    canonical = []
    for index, xyz in enumerate(clouds):
        ordered = np.ascontiguousarray(xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))])
        digest = hashlib.sha256(ordered.tobytes()).hexdigest()
        canonical.append((digest, index, ordered))
    canonical.sort(key=lambda row: row[0])
    nonempty = [(index, xyz) for _, index, xyz in canonical if len(xyz)]
    if nonempty and len(ref):
        all_xyz = np.concatenate([xyz for _, xyz in nonempty])
        all_owner = np.concatenate([np.full(len(xyz), index, dtype=np.int32)
                                    for index, xyz in nonempty])
        # Identical native coordinates are represented once. The first owner is
        # deterministic in the content-derived ordering; no duplicate is lost.
        unique_xyz, first = np.unique(all_xyz, axis=0, return_index=True)
        owners = all_owner[first]
        distances, nearest = cKDTree(unique_xyz).query(ref, k=1, workers=-1)
        winner = owners[nearest]
        assigned = distances < max_distance_m
    else:
        distances = np.full(len(ref), np.inf)
        winner = np.full(len(ref), -1, dtype=np.int32)
        assigned = np.zeros(len(ref), dtype=bool)
    instances = [CanonicalInstance(str(index),
        np.flatnonzero(assigned & (winner == index)).astype(np.int32),
        None if confidence_by_index is None else confidence_by_index.get(index))
        for index in range(len(clouds))]
    prediction = CanonicalPrediction(scene_id, len(ref), instances, method_name,
        method_commit, adapter_version, protocol_version, True, metadata or {})
    prediction.validate()
    diagnostic = None
    if diagnostic_distance_m is not None:
        diagnostic = pairwise_geometry_support(clouds, ref, diagnostic_distance_m,
            scene_id=scene_id, method_name=method_name, method_commit=method_commit,
            adapter_version=adapter_version, protocol_version=protocol_version,
            metadata={"role": "structure_diagnostics_only"})
    finite = distances[np.isfinite(distances)]
    stats = {
        "num_native_points": int(sum(len(x) for x in clouds)),
        "num_pred_instances_before_filter": len(clouds),
        "num_pred_instances_after_filter": len(instances),
        "native_empty_object_count": sum(len(x) == 0 for x in clouds),
        "projected_empty_object_count": sum(len(x.vertex_indices) == 0 for x in instances),
        "mapped_ref_vertices": int(assigned.sum()),
        "unmapped_ref_vertices": int(len(ref) - assigned.sum()),
        "mapping_coverage": float(assigned.mean()) if len(ref) else None,
        "mean_nn_distance_m": float(finite.mean()) if len(finite) else None,
        "median_nn_distance_m": float(np.median(finite)) if len(finite) else None,
        "p95_nn_distance_m": float(np.percentile(finite, 95)) if len(finite) else None,
        "max_distance_m": float(max_distance_m),
        "diagnostic_max_distance_m": diagnostic_distance_m,
        "mapping_direction": "competitive_reference_to_nearest_native_instance",
        "is_partition": True,
    }
    return MappingResult(prediction, stats, distances, diagnostic)


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


def map_instances_to_reference_v2(
    point_clouds: list[np.ndarray],
    ref_xyz: np.ndarray,
    max_distance_m: float,
    *,
    scene_id: str,
    method_name: str,
    method_commit: str,
    adapter_version: str,
    protocol_version: str,
    confidence_by_index: dict[int, float] | None = None,
    metadata: dict | None = None,
) -> MappingResult:
    """Project each native object independently; retain empty objects and overlap."""
    ref = np.asarray(ref_xyz, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all():
        raise EvaluationError("ref_xyz must be finite and have shape [N,3]")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("max_distance_m must be finite and positive")
    validated = []
    for index, points in enumerate(point_clouds):
        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise EvaluationError(f"Native object {index} points must be finite [N,3]")
        validated.append(xyz)
    all_points = np.concatenate(validated) if validated else np.empty((0, 3))
    if len(all_points) and len(ref):
        distances, _ = cKDTree(all_points).query(ref, k=1, workers=-1)
    else:
        distances = np.full(len(ref), np.inf)
    candidate_pool = np.flatnonzero(distances < max_distance_m)
    instances: list[CanonicalInstance] = []
    ownership = np.zeros(len(ref), dtype=np.int32)
    for index, xyz in enumerate(validated):
        vertices = np.empty(0, dtype=np.int32)
        if len(xyz) and len(candidate_pool):
            lower = xyz.min(axis=0) - max_distance_m
            upper = xyz.max(axis=0) + max_distance_m
            candidates = candidate_pool[np.all(
                (ref[candidate_pool] >= lower) & (ref[candidate_pool] <= upper), axis=1)]
            if len(candidates):
                distance, _ = cKDTree(xyz).query(ref[candidates], k=1, workers=-1)
                vertices = candidates[distance < max_distance_m].astype(np.int32)
                ownership[vertices] += 1
        instances.append(CanonicalInstance(str(index), vertices,
            None if confidence_by_index is None else confidence_by_index.get(index)))
    prediction = CanonicalPrediction(scene_id, len(ref), instances, method_name,
        method_commit, adapter_version, protocol_version,
        is_partition=not bool(np.any(ownership > 1)), metadata=metadata or {})
    prediction.validate()
    finite = distances[np.isfinite(distances)]
    stats = {
        "num_native_points": int(len(all_points)),
        "num_native_assigned_points": int(len(all_points)),
        "num_pred_instances_before_filter": len(point_clouds),
        "num_pred_instances_after_filter": len(instances),
        "native_empty_object_count": sum(len(x) == 0 for x in validated),
        "projected_empty_object_count": sum(len(x.vertex_indices) == 0 for x in instances),
        "mapped_ref_vertices": int(np.count_nonzero(ownership)),
        "unmapped_ref_vertices": int(len(ref) - np.count_nonzero(ownership)),
        "overlapped_ref_vertices": int(np.count_nonzero(ownership > 1)),
        "mapping_coverage": float(np.count_nonzero(ownership) / len(ref)) if len(ref) else None,
        "mean_nn_distance_m": float(finite.mean()) if len(finite) else None,
        "median_nn_distance_m": float(np.median(finite)) if len(finite) else None,
        "p95_nn_distance_m": float(np.percentile(finite, 95)) if len(finite) else None,
        "max_distance_m": float(max_distance_m),
        "mapping_direction": "independent_reference_to_each_native_instance",
        "is_partition": prediction.is_partition,
    }
    return MappingResult(prediction, stats, distances)
