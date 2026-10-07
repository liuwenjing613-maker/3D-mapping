from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

from .geometry import (MappingResult, map_instances_to_reference_v2,
                       map_instances_to_reference_v3, map_point_labels_to_reference)
from .io import sha256_file
from .schema import EvaluationError


def adapt_map(map_path: str | Path, ref_xyz: np.ndarray, max_distance_m: float,
              *, scene_id: str, method_name: str, method_commit: str,
              protocol_version: str,
              mapping_method: str = "nearest_neighbor_reference_to_prediction",
              diagnostic_distance_m: float | None = None) -> MappingResult:
    """Adapt a trusted ConceptGraphs pcd_*.pkl.gz map to reference vertex masks.

    Pickle executes code when loaded. Only pass local outputs produced by the
    trusted ConceptGraphs run, never a downloaded/untrusted pickle.
    """
    map_path = Path(map_path)
    with gzip.open(map_path, "rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("objects"), list):
        raise EvaluationError("ConceptGraphs map must contain an objects list")
    points = []
    owners = []
    confidence = {}
    native_uids = {}
    for index, obj in enumerate(raw["objects"]):
        xyz = np.asarray(obj["pcd_np"], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise EvaluationError(f"ConceptGraphs object {index} pcd_np is not [N,3]")
        points.append(xyz)
        owners.append(np.full(len(xyz), index, dtype=np.int64))
        native_uids[index] = str(obj.get("id", index))
        values = obj.get("conf", [])
        if values:
            confidence[index] = float(max(values))
    pred_xyz = np.concatenate(points) if points else np.empty((0, 3))
    labels = np.concatenate(owners) if owners else np.empty(0, dtype=np.int64)
    base_metadata = {"source_map": str(map_path), "source_map_sha256": sha256_file(map_path),
                     "native_object_count": len(raw["objects"]),
                     "native_background_objects_included": True}
    if mapping_method in ("independent_nearest_neighbor_per_instance", "competitive_nearest_instance"):
        mapper = (map_instances_to_reference_v3 if mapping_method == "competitive_nearest_instance"
                  else map_instances_to_reference_v2)
        kwargs = ({"diagnostic_distance_m": diagnostic_distance_m}
                  if mapping_method == "competitive_nearest_instance" else {})
        result = mapper(points, ref_xyz, max_distance_m,
            scene_id=scene_id, method_name=method_name, method_commit=method_commit,
            adapter_version="conceptgraphs_pcd_v3" if kwargs else "conceptgraphs_pcd_v2",
            protocol_version=protocol_version,
            confidence_by_index=confidence,
            metadata={**base_metadata, "native_object_index_is_uid": True}, **kwargs)
        for instance in result.prediction.instances:
            native_index = int(instance.instance_uid)
            instance.instance_uid = f"cg-index:{native_index}"
            instance.metadata.update({"native_object_index": native_index,
                                      "native_object_id": native_uids[native_index]})
        if result.diagnostic_prediction is not None:
            for instance in result.diagnostic_prediction.instances:
                native_index = int(instance.instance_uid)
                instance.instance_uid = f"cg-index:{native_index}"
    elif mapping_method == "nearest_neighbor_reference_to_prediction":
        result = map_point_labels_to_reference(pred_xyz, labels, ref_xyz, max_distance_m,
            scene_id=scene_id, method_name=method_name, method_commit=method_commit,
            adapter_version="conceptgraphs_pcd_v1", protocol_version=protocol_version,
            confidence_by_id=confidence,
            metadata={**base_metadata, "native_object_index_is_uid": True})
        for instance in result.prediction.instances:
            native_index = int(instance.instance_uid)
            instance.instance_uid = native_uids[native_index]
            instance.metadata["native_object_index"] = native_index
    else:
        raise EvaluationError(f"Unsupported mapping method: {mapping_method}")
    result.prediction.validate()
    result.statistics["num_native_objects"] = len(raw["objects"])
    return result
