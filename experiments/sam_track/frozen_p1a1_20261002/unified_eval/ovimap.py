from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import MappingResult, map_instances_to_reference_v2, map_instances_to_reference_v3
from .io import sha256_file
from .schema import EvaluationError


def adapt_export(export_path: str | Path, ref_xyz: np.ndarray, max_distance_m: float,
                 *, scene_id: str, method_name: str, method_commit: str,
                 protocol_version: str, mapping_method: str = "independent_nearest_neighbor_per_instance",
                 diagnostic_distance_m: float | None = None) -> MappingResult:
    """Map every OVI-MAP exported instance independently onto the shared GT mesh."""
    export_path = Path(export_path)
    with np.load(export_path, allow_pickle=False) as data:
        required = {"xyz", "instance", "native_instance_ids"}
        if not required.issubset(data.files):
            raise EvaluationError(f"OVI-MAP export lacks {sorted(required - set(data.files))}")
        xyz = data["xyz"].copy()
        labels = data["instance"].copy()
        native_ids = data["native_instance_ids"].copy()
        classes = data["classes"].copy() if "classes" in data else None
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise EvaluationError("OVI-MAP xyz must be finite [N,3]")
    if labels.ndim != 1 or len(labels) != len(xyz) or not np.issubdtype(labels.dtype, np.integer):
        raise EvaluationError("OVI-MAP instance labels must be integer and match xyz")
    if native_ids.ndim != 1 or not np.issubdtype(native_ids.dtype, np.integer):
        raise EvaluationError("OVI-MAP native_instance_ids must be a 1D integer array")
    if len(np.unique(native_ids)) != len(native_ids) or np.any(native_ids < 0):
        raise EvaluationError("OVI-MAP native instance IDs must be unique and nonnegative")
    if classes is not None and len(classes) != len(native_ids):
        raise EvaluationError("OVI-MAP classes must align with native_instance_ids")
    assigned = labels >= 0
    # The native exporter writes compact owner indices; native_instance_ids maps
    # each index back to the original OVI-MAP instance ID.
    if np.any(labels[assigned] >= len(native_ids)):
        raise EvaluationError("OVI-MAP export has owner indices outside native_instance_ids")
    # Sort once, then retain views for each owner index, including empty IDs.
    order = np.argsort(labels[assigned], kind="stable")
    sorted_labels = labels[assigned][order]
    sorted_xyz = xyz[assigned][order]
    clouds = []
    for index in range(len(native_ids)):
        left = np.searchsorted(sorted_labels, index, side="left")
        right = np.searchsorted(sorted_labels, index, side="right")
        clouds.append(sorted_xyz[left:right])
    if mapping_method not in ("independent_nearest_neighbor_per_instance", "competitive_nearest_instance"):
        raise EvaluationError("Unsupported OVI-MAP mapping method")
    mapper = (map_instances_to_reference_v3 if mapping_method == "competitive_nearest_instance"
              else map_instances_to_reference_v2)
    kwargs = ({"diagnostic_distance_m": diagnostic_distance_m}
              if mapping_method == "competitive_nearest_instance" else {})
    result = mapper(clouds, ref_xyz, max_distance_m,
        scene_id=scene_id, method_name=method_name, method_commit=method_commit,
        adapter_version="ovimap_export_v3" if kwargs else "ovimap_export_v2",
        protocol_version=protocol_version, **kwargs,
        metadata={"source_export": str(export_path),
                  "source_export_sha256": sha256_file(export_path),
                  "native_object_count": len(native_ids),
                  "source_label_encoding": "compact_index_into_native_instance_ids",
                  "unassigned_source_vertices": int((~assigned).sum())})
    for index, instance in enumerate(result.prediction.instances):
        instance.instance_uid = f"ovi-id:{int(native_ids[index])}"
        instance.metadata["native_instance_id"] = int(native_ids[index])
        if classes is not None:
            instance.semantic_id = int(classes[index]) if classes[index] > 0 else None
    if result.diagnostic_prediction is not None:
        for index, instance in enumerate(result.diagnostic_prediction.instances):
            instance.instance_uid = f"ovi-id:{int(native_ids[index])}"
    result.prediction.validate()
    result.statistics.update({"num_native_objects": len(native_ids),
                              "source_vertex_count": len(xyz),
                              "unassigned_source_vertices": int((~assigned).sum())})
    return result
