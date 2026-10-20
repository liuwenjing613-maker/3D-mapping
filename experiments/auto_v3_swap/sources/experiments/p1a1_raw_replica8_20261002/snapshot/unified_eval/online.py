from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .evaluate import evaluate_scenes
from .io import load_prediction, sha256_file
from .schema import CanonicalGT, EvaluationError, Protocol


def observed_vertices_from_depth(depth_m: np.ndarray, intrinsics: np.ndarray,
                                 world_from_camera: np.ndarray, ref_xyz: np.ndarray,
                                 max_distance_m: float) -> np.ndarray:
    """Return reference vertices supported by one RGB-D frame, without future views."""
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    ref = np.asarray(ref_xyz, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or pose.shape != (4, 4) or ref.ndim != 2 or ref.shape[1] != 3:
        raise EvaluationError("Depth, intrinsics, pose or reference shape is invalid")
    if not np.isfinite(k).all() or not np.isfinite(pose).all() or not np.isfinite(ref).all():
        raise EvaluationError("Camera or reference values are not finite")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("Observation mapping distance must be finite and positive")
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(rows) or not len(ref):
        return np.zeros(len(ref), dtype=bool)
    homogeneous = np.stack((cols, rows, np.ones(len(rows))), axis=0)
    camera = np.linalg.solve(k, homogeneous).T * depth[rows, cols, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    distance, nearest = cKDTree(world).query(ref, k=1, workers=-1)
    return distance < max_distance_m


def update_observation_count(count: np.ndarray, observed_this_frame: np.ndarray) -> np.ndarray:
    """Count each reference vertex once per frame; saturate uint16 safely."""
    value = np.asarray(count)
    observed = np.asarray(observed_this_frame, dtype=bool)
    if value.dtype != np.uint16 or value.shape != observed.shape:
        raise EvaluationError("Observation count must be uint16 and match observed mask")
    result = value.copy()
    active = observed & (result < np.iinfo(np.uint16).max)
    result[active] += 1
    return result


def evaluate_online_prefixes(gt: CanonicalGT, protocol: Protocol,
                             online_manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Evaluate committed snapshots against only depth observed through each checkpoint."""
    if not protocol.retains_predictions:
        raise EvaluationError("Online prefix evaluation requires Replica-CA-v2 or v3")
    path = Path(online_manifest_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    observation_distance = (spec.get("observation_max_distance_m") if protocol.is_v3
                            else protocol.geometry_mapping_max_distance_m)
    if not isinstance(observation_distance, (int, float)) or not np.isfinite(observation_distance) or observation_distance <= 0:
        raise EvaluationError("v3 online manifest requires observation_max_distance_m")
    if spec.get("scene_id") != gt.scene_id:
        raise EvaluationError("Online scene differs from GT")
    source_path = Path(spec["source_experiment_manifest"])
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    source = json.loads(source_path.read_text(encoding="utf-8"))
    entries = [e for e in source.get("entries", [])
               if e.get("scene") == gt.scene_id and e.get("method") == spec.get("source_method")]
    if len(entries) != 1:
        raise EvaluationError("Online source manifest must identify one scene/method run")
    source_input = entries[0].get("input", {})
    try:
        start, end, stride = (int(source_input[k]) for k in ("start", "end", "stride"))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Source experiment lacks a valid frame range") from exc
    if start < 0 or end <= start or stride <= 0:
        raise EvaluationError("Source experiment frame range is invalid")
    expected_ids = list(range(start, end, stride))
    if entries[0].get("cost", {}).get("frames") != len(expected_ids):
        raise EvaluationError("Source experiment frame count differs from its range")
    frames = spec.get("frames", [])
    checkpoints = spec.get("checkpoints", [])
    if [frame.get("frame_id") for frame in frames] != expected_ids:
        raise EvaluationError("Online frames must exactly match the source experiment frame list")
    checkpoint_ids = [item.get("frame_id") for item in checkpoints]
    if not checkpoint_ids or checkpoint_ids != sorted(set(checkpoint_ids)) or not set(checkpoint_ids).issubset(expected_ids):
        raise EvaluationError("Checkpoints must be nonempty, unique, ordered source frames")
    source_hash = sha256_file(source_path)
    def resolve(name: str) -> Path:
        candidate = Path(name)
        return candidate if candidate.is_absolute() else path.parent / candidate
    observed = np.zeros(gt.vertex_count, dtype=bool)
    counts = np.zeros(gt.vertex_count, dtype=np.uint16)
    result_rows = []
    file_sources = []
    checkpoint_by_frame = {item["frame_id"]: item for item in checkpoints}
    for frame in frames:
        frame_id = frame["frame_id"]
        paths = {key: resolve(frame[key]) for key in ("depth_m", "intrinsics", "world_from_camera")}
        depth, intrinsics, pose = (np.load(paths[key], allow_pickle=False)
                                   for key in ("depth_m", "intrinsics", "world_from_camera"))
        seen = observed_vertices_from_depth(depth, intrinsics, pose, gt.xyz_ref,
                                            observation_distance)
        observed |= seen
        counts = update_observation_count(counts, seen)
        file_sources.append({"frame_id": frame_id, **{key: {"path": str(value), "sha256": sha256_file(value)}
                                                  for key, value in paths.items()}})
        if frame_id not in checkpoint_by_frame:
            continue
        pred_path = resolve(checkpoint_by_frame[frame_id]["prediction"])
        pred = load_prediction(pred_path)
        diagnostic = None
        if protocol.is_v3:
            if "diagnostic_prediction" not in checkpoint_by_frame[frame_id]:
                raise EvaluationError("v3 checkpoint requires diagnostic_prediction")
            diagnostic = load_prediction(resolve(checkpoint_by_frame[frame_id]["diagnostic_prediction"]))
            if diagnostic.metadata.get("role") != "structure_diagnostics_only":
                raise EvaluationError("v3 checkpoint diagnostic prediction has the wrong role")
            for key in ("source_map_sha256", "source_export_sha256"):
                if key in pred.metadata and diagnostic.metadata.get(key) != pred.metadata[key]:
                    raise EvaluationError("v3 checkpoint diagnostic and main predictions differ in source map")
        if pred.metadata.get("committed_frame_id") != frame_id:
            raise EvaluationError(f"Checkpoint {frame_id} lacks matching committed_frame_id")
        max_input = pred.metadata.get("max_input_frame_id")
        if not isinstance(max_input, int) or max_input > frame_id:
            raise EvaluationError(f"Checkpoint {frame_id} uses or fails to declare its latest input frame")
        declared_source_hash = pred.metadata.get("source_experiment_manifest_sha256")
        if declared_source_hash is not None and declared_source_hash != source_hash:
            raise EvaluationError(f"Checkpoint {frame_id} has a different source experiment manifest")
        _, per_scene, _ = evaluate_scenes([(gt, pred)], protocol, observed_masks=[observed],
            diagnostic_predictions=[diagnostic] if diagnostic is not None else None)
        row = per_scene[0]
        row["frame_id"] = frame_id
        row["input_frame_count"] = len(file_sources)
        row["observed_reference_vertices"] = int(np.count_nonzero(observed))
        row["prediction_file"] = str(pred_path)
        row["prediction_sha256"] = sha256_file(pred_path)
        row["prediction_debug_only"] = bool(pred.metadata.get("debug_only"))
        result_rows.append(row)
    frame_hash = hashlib.sha256(json.dumps(expected_ids, separators=(",", ":")).encode()).hexdigest()
    provenance = {"source_experiment_manifest": str(source_path),
                  "source_experiment_manifest_sha256": source_hash,
                  "frame_list_sha256": frame_hash,
                  "frame_count": len(expected_ids), "frame_files": file_sources,
                  "observation_max_distance_m": observation_distance,
                  "observation_count_max": int(counts.max()) if len(counts) else 0}
    return result_rows, provenance
