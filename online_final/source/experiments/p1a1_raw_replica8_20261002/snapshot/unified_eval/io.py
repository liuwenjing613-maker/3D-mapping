from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import CanonicalGT, CanonicalInstance, CanonicalPrediction, EvaluationError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save_gt(path: str | Path, gt: CanonicalGT) -> None:
    gt.validate()
    np.savez_compressed(path, scene_id=gt.scene_id, xyz_ref=gt.xyz_ref,
        instance_id=gt.instance_id, semantic_id=gt.semantic_id,
        valid_vertex_mask=gt.valid_vertex_mask, ignore_vertex_mask=gt.ignore_vertex_mask,
        observation_count=gt.observation_count if gt.observation_count is not None else np.array([], dtype=np.uint16),
        metadata_json=_json(gt.metadata))


def load_gt(path: str | Path) -> CanonicalGT:
    with np.load(path, allow_pickle=False) as data:
        count = data["observation_count"] if "observation_count" in data else np.array([])
        gt = CanonicalGT(str(data["scene_id"].item()), data["xyz_ref"].copy(),
            data["instance_id"].copy(), data["semantic_id"].copy(),
            data["valid_vertex_mask"].copy(), data["ignore_vertex_mask"].copy(),
            count.copy() if len(count) else None,
            json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {})
    gt.validate()
    return gt


def save_prediction(path: str | Path, pred: CanonicalPrediction) -> None:
    pred.validate()
    offsets = np.zeros(len(pred.instances) + 1, dtype=np.int64)
    for i, instance in enumerate(pred.instances):
        offsets[i + 1] = offsets[i] + len(instance.vertex_indices)
    indices = np.concatenate([np.asarray(x.vertex_indices, dtype=np.int32) for x in pred.instances]) if pred.instances else np.array([], dtype=np.int32)
    np.savez_compressed(path, scene_id=pred.scene_id,
        reference_vertex_count=pred.reference_vertex_count, offsets=offsets,
        vertex_indices=indices, instance_uids=np.array([x.instance_uid for x in pred.instances], dtype=str),
        confidence=np.array([np.nan if x.confidence is None else x.confidence for x in pred.instances]),
        semantic_id=np.array([-1 if x.semantic_id is None else x.semantic_id for x in pred.instances], dtype=np.int64),
        instance_metadata_json=np.array([_json(x.metadata) for x in pred.instances], dtype=str),
        method_name=pred.method_name, method_commit=pred.method_commit,
        adapter_version=pred.adapter_version, protocol_version=pred.protocol_version,
        is_partition=pred.is_partition, metadata_json=_json(pred.metadata))


def load_prediction(path: str | Path) -> CanonicalPrediction:
    with np.load(path, allow_pickle=False) as data:
        offsets = data["offsets"]
        indices = data["vertex_indices"]
        uids = data["instance_uids"]
        if len(offsets) != len(uids) + 1 or offsets[0] != 0 or offsets[-1] != len(indices) or np.any(np.diff(offsets) < 0):
            raise EvaluationError("Prediction offsets are invalid")
        scores = data["confidence"]
        semantics = data["semantic_id"]
        metadata = data["instance_metadata_json"]
        if not (len(scores) == len(semantics) == len(metadata) == len(uids)):
            raise EvaluationError("Prediction instance fields have different lengths")
        instances = [CanonicalInstance(str(uids[i]), indices[offsets[i]:offsets[i + 1]].copy(),
            None if np.isnan(scores[i]) else float(scores[i]),
            None if semantics[i] < 0 else int(semantics[i]), json.loads(str(metadata[i])))
            for i in range(len(uids))]
        pred = CanonicalPrediction(str(data["scene_id"].item()), int(data["reference_vertex_count"].item()),
            instances, str(data["method_name"].item()), str(data["method_commit"].item()),
            str(data["adapter_version"].item()), str(data["protocol_version"].item()),
            bool(data["is_partition"].item()), json.loads(str(data["metadata_json"].item())))
    pred.validate()
    return pred
