"""Prediction-independent object qualification and explicit reference regions."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import sha256_array
from .schema import CanonicalGT, EvaluationError

IGNORE, TARGET, KNOWN_NON_TARGET = 0, 1, 2
PROFILE = "object_observed_repair"


def scope_sha256(scope: dict) -> str:
    return hashlib.sha256(json.dumps(scope, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_gt_scope(gt: CanonicalGT, semantic_classes: list[str], target_classes: list[str], *,
                   review_required_ids: tuple[int, ...] = (),
                   official_category_by_id: dict[int, str] | None = None) -> dict:
    """Keep every raw ID; a review flag never automatically declares a tiny GT invalid."""
    gt.validate()
    if gt.raw_instance_id is None:
        raise EvaluationError("Qualification requires preserved raw instance IDs")
    targets = set(target_classes)
    non_targets = {"wall", "floor", "ceiling"}
    if targets & non_targets:
        raise EvaluationError("Structural background cannot be in the fixed target category list")
    rows = []
    for raw_id in np.unique(gt.raw_instance_id):
        vertices = np.flatnonzero(gt.raw_instance_id == raw_id)
        semantics = np.unique(gt.semantic_id[vertices])
        category = None
        if len(semantics) == 1 and 0 < semantics[0] <= len(semantic_classes):
            category = semantic_classes[int(semantics[0]) - 1]
        official = (official_category_by_id or {}).get(int(raw_id))
        if official is not None:
            if category is not None and category != official:
                raise EvaluationError(f"Official metadata disagrees for GT {raw_id}")
            category = official
        status = ("INVALID" if raw_id <= 0 else "TARGET" if category in targets else
                  "NON_TARGET" if category in non_targets else "UNKNOWN")
        review = int(raw_id) in review_required_ids
        verified = bool(raw_id > 0 and category is not None and not review and len(semantics) == 1)
        reason = ("VOID_RAW_ID" if raw_id <= 0 else "PENDING_GEOMETRY_REVIEW" if review else
                  "UNRESOLVED_OFFICIAL_CATEGORY" if status == "UNKNOWN" else
                  "FIXED_NON_TARGET_CATEGORY" if status == "NON_TARGET" else "TARGET_CANDIDATE")
        rows.append({"scene_id": gt.scene_id, "raw_gt_id": int(raw_id),
            "semantic_class": category, "object_status": status,
            "quality_verified": verified, "observable": None, "reason": reason,
            "review_required": review, "reference_vertex_count": int(len(vertices)),
            "reference_semantic_ids": semantics.astype(int).tolist(),
            "evaluable": False})
    return {"schema_version": 1, "evaluation_profile": PROFILE, "scene_id": gt.scene_id,
            "status": "CANDIDATE_SCOPE_NOT_FROZEN", "target_classes": sorted(targets),
            "non_target_classes": sorted(non_targets), "objects": rows,
            "reference_xyz_sha256": sha256_array(gt.xyz_ref),
            "raw_instance_sha256": sha256_array(gt.raw_instance_id),
            "source_reference_sha256": gt.metadata.get("source_reference_sha256"),
            "qualification_uses_predictions": False}


def apply_observed_scope(gt: CanonicalGT, scope: dict, observation_count: np.ndarray, *,
                         observation_metadata: dict, ambiguity_mask: np.ndarray | None = None) -> tuple[CanonicalGT, dict]:
    """Score the same trusted, observed target/background surface for every method."""
    gt.validate()
    if scope.get("scene_id") != gt.scene_id or scope.get("evaluation_profile") != PROFILE:
        raise EvaluationError("Scope scene/profile mismatch")
    if gt.raw_instance_id is None or scope["raw_instance_sha256"] != sha256_array(gt.raw_instance_id):
        raise EvaluationError("Scope raw GT identity mismatch")
    if scope["reference_xyz_sha256"] != sha256_array(gt.xyz_ref):
        raise EvaluationError("Scope reference geometry mismatch")
    counts = np.asarray(observation_count)
    if counts.shape != (gt.vertex_count,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise EvaluationError("Observation counts must be nonnegative integers on the reference")
    ambiguous = np.zeros(gt.vertex_count, dtype=bool) if ambiguity_mask is None else np.asarray(ambiguity_mask, dtype=bool)
    if ambiguous.shape != counts.shape:
        raise EvaluationError("Ambiguity mask must match the reference")
    observed = (counts > 0) & ~ambiguous & gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    qualified = json.loads(json.dumps(scope))
    raw_ids = set(np.unique(gt.raw_instance_id).astype(int).tolist())
    row_ids = [x["raw_gt_id"] for x in qualified["objects"]]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != raw_ids:
        raise EvaluationError("Scope must preserve exactly every source raw GT ID")
    regions = np.full(gt.vertex_count, IGNORE, dtype=np.uint8)
    labels = np.full(gt.vertex_count, -1, dtype=np.int64)
    for row in qualified["objects"]:
        selected = gt.raw_instance_id == row["raw_gt_id"]
        visible = selected & observed
        row["observable"] = bool(np.any(visible))
        row["observed_vertex_count"] = int(visible.sum())
        row["observation_count_sum"] = int(counts[visible].sum())
        row["evaluable"] = bool(row["object_status"] == "TARGET" and row["quality_verified"] and row["observable"])
        if row["evaluable"]:
            regions[visible] = TARGET
            labels[visible] = row["raw_gt_id"]
            row["reason"] = "TRUSTED_OBSERVED_TARGET"
        elif row["object_status"] == "NON_TARGET" and row["quality_verified"]:
            regions[visible] = KNOWN_NON_TARGET
        elif row["object_status"] == "TARGET" and row["quality_verified"] and not row["observable"]:
            row["reason"] = "NO_TRUSTED_INPUT_OBSERVATION"
    qualified["status"] = "DEVELOPMENT_SCOPE_NOT_FROZEN"
    qualified["observation_metadata"] = observation_metadata
    qualified["observation_count_sha256"] = sha256_array(counts)
    qualified["ambiguity_mask_sha256"] = sha256_array(ambiguous)
    qualified["evaluation_region_sha256"] = sha256_array(regions)
    metadata = {**gt.metadata, "role": "observed_object_evaluation_gt", "evaluation_profile": PROFILE,
        "gt_scope_sha256": scope_sha256(qualified),
        "gt_observed_support_sha256": sha256_array(counts),
        "evaluation_region_sha256": sha256_array(regions),
        "reference_xyz_sha256": sha256_array(gt.xyz_ref),
        "qualified_instance_sha256": sha256_array(labels),
        "raw_instance_sha256": sha256_array(gt.raw_instance_id),
        "observation_metadata": observation_metadata,
        "scope_status": qualified["status"]}
    result = replace(gt, instance_id=labels, valid_vertex_mask=regions != IGNORE,
                     ignore_vertex_mask=regions == IGNORE, observation_count=counts.copy(),
                     evaluation_region=regions, metadata=metadata)
    result.validate()
    return result, qualified


def coincident_label_ambiguity(xyz: np.ndarray, raw_ids: np.ndarray) -> np.ndarray:
    """Exactly coincident coordinates carrying different GT labels are unresolvable by geometry."""
    _, inverse = np.unique(xyz, axis=0, return_inverse=True)
    minimum = np.full(int(inverse.max()) + 1 if len(inverse) else 0, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full_like(minimum, np.iinfo(np.int64).min)
    np.minimum.at(minimum, inverse, raw_ids)
    np.maximum.at(maximum, inverse, raw_ids)
    return (minimum != maximum)[inverse]
