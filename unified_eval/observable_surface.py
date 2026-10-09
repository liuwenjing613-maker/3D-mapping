"""Fixed RGB-D support using projection, valid depth and source GT instance visibility."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import sha256_array
from .schema import EvaluationError


@dataclass(frozen=True)
class ObservationFrame:
    frame_id: int
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    gt_instance_image: np.ndarray
    gt_depth_m: np.ndarray | None = None
    source_hashes: dict | None = None


def observed_reference_vertices_from_frame(ref_xyz: np.ndarray, raw_gt_ids: np.ndarray,
        frame: ObservationFrame, depth_tolerance_m: float) -> np.ndarray:
    """A nearby wall pixel cannot observe a socket with a different raw GT image ID."""
    ref, raw = np.asarray(ref_xyz), np.asarray(raw_gt_ids)
    depth, ids = np.asarray(frame.depth_m), np.asarray(frame.gt_instance_image)
    k, pose = np.asarray(frame.intrinsics), np.asarray(frame.world_from_camera)
    if ref.ndim != 2 or ref.shape[1] != 3 or raw.shape != (len(ref),):
        raise EvaluationError("Reference/raw GT dimensions are invalid")
    if depth.ndim != 2 or ids.shape != depth.shape or not np.issubdtype(ids.dtype, np.integer):
        raise EvaluationError("Depth and raw GT instance image must have the same HxW shape")
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise EvaluationError("Camera matrices must be finite 3x3/4x4")
    if not np.isfinite(ref).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-4):
        raise EvaluationError("Reference or rigid camera transform is invalid")
    if k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise EvaluationError("Camera intrinsic calibration is invalid")
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise EvaluationError("Depth observation tolerance must be finite and positive")
    camera = (ref - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    positive = np.flatnonzero(z > 0)
    uvw = camera[positive] @ k.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    pixels = np.floor(uv + .5).astype(np.int64)
    height, width = depth.shape
    in_image = (pixels[:,0] >= 0) & (pixels[:,0] < width) & (pixels[:,1] >= 0) & (pixels[:,1] < height)
    vertices, pixels = positive[in_image], pixels[in_image]
    measured = depth[pixels[:,1], pixels[:,0]]
    visible_id = ids[pixels[:,1], pixels[:,0]]
    supported = np.isfinite(measured) & (measured > 0) & (raw[vertices] > 0)
    supported &= (visible_id == raw[vertices]) & (np.abs(measured - z[vertices]) < depth_tolerance_m)
    if frame.gt_depth_m is not None:
        rendered = np.asarray(frame.gt_depth_m)
        if rendered.shape != depth.shape:
            raise EvaluationError("Rendered GT depth must match input camera dimensions")
        expected = rendered[pixels[:,1], pixels[:,0]]
        supported &= np.isfinite(expected) & (expected > 0) & (np.abs(expected - z[vertices]) < depth_tolerance_m)
    result = np.zeros(len(ref), dtype=bool)
    result[vertices[supported]] = True
    return result


def accumulate_observed_support(ref_xyz: np.ndarray, raw_gt_ids: np.ndarray,
        frames: Iterable[ObservationFrame], depth_tolerance_m: float) -> tuple[np.ndarray, dict]:
    counts = np.zeros(len(ref_xyz), dtype=np.uint32)
    frame_ids, evidence = [], []
    all_mesh_verified = True
    for frame in frames:
        if frame.frame_id in frame_ids:
            raise EvaluationError("Observation frame IDs must be unique")
        counts += observed_reference_vertices_from_frame(ref_xyz, raw_gt_ids, frame, depth_tolerance_m)
        frame_ids.append(int(frame.frame_id))
        all_mesh_verified &= frame.gt_depth_m is not None
        evidence.append({"frame_id": int(frame.frame_id),
            "depth_sha256": sha256_array(frame.depth_m), "gt_id_image_sha256": sha256_array(frame.gt_instance_image),
            "intrinsics_sha256": sha256_array(frame.intrinsics), "pose_sha256": sha256_array(frame.world_from_camera),
            "gt_depth_sha256": sha256_array(frame.gt_depth_m) if frame.gt_depth_m is not None else None,
            "source_hashes": frame.source_hashes})
    if not frame_ids:
        raise EvaluationError("At least one fixed input frame is required")
    metadata = {"method": "gt_vertex_projection_valid_depth_and_raw_gt_instance_visibility",
        "depth_tolerance_m": float(depth_tolerance_m), "frame_ids": frame_ids, "frame_count": len(frame_ids),
        "frame_list_sha256": hashlib.sha256(json.dumps(frame_ids, separators=(",", ":")).encode()).hexdigest(),
        "input_evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "input_evidence": evidence,
        "mesh_visibility_verified": bool(all_mesh_verified),
        "status": "DEPTH_AND_MESH_VERIFIED" if all_mesh_verified else "DEVELOPMENT_CACHE_VISIBILITY_NOT_MESH_VERIFIED",
        "pixel_size_filter": None, "minimum_trusted_observation_count": 1,
        "qualification_uses_predictions": False}
    return counts, metadata


def save_observed_support(path: str | Path, ref_xyz: np.ndarray, raw_gt_ids: np.ndarray,
        counts: np.ndarray, metadata: dict, ambiguity_mask: np.ndarray) -> None:
    np.savez_compressed(path, observed_mask=(counts > 0) & ~ambiguity_mask,
        observation_count=counts, ambiguity_mask=ambiguity_mask,
        reference_xyz_sha256=sha256_array(ref_xyz), raw_instance_sha256=sha256_array(raw_gt_ids),
        metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
        observation_count_sha256=sha256_array(counts))
