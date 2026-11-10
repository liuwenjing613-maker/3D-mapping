from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import sha256_file
from .schema import CanonicalGT, EvaluationError


SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")


def load_existing_reference(reference_root: str | Path, scene_id: str) -> CanonicalGT:
    """Read the existing eight-scene fullmesh reference; never regenerate labels."""
    if scene_id not in SCENES:
        raise EvaluationError(f"Replica scene {scene_id!r} is outside the existing eight-scene set")
    directory = Path(reference_root) / scene_id
    manifest_path = directory / "manifest.json"
    reference_path = directory / "reference.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["scene"] != scene_id or sha256_file(reference_path) != manifest["reference_sha256"]:
        raise EvaluationError("Existing Replica reference scene/hash mismatch")
    for path_key, hash_key in (("mesh", "mesh_sha256"),
                               ("semantic_label_file", "semantic_label_sha256"),
                               ("instance_label_file", "instance_label_sha256")):
        source = Path(manifest[path_key])
        if not source.is_file() or sha256_file(source) != manifest[hash_key]:
            raise EvaluationError(f"Existing Replica {path_key} source is missing or hash changed")
    with np.load(reference_path, allow_pickle=False) as data:
        xyz = data["xyz"].copy()
        semantic = data["semantic"].copy()
        instance = data["instance"].copy()
    names = manifest["semantic_classes"]
    countable = {names.index(name) + 1 for name in manifest["instance_classes"]}
    # Mirrors the existing evaluator's valid GT class selection. The remaining
    # vertices stay in the evaluation surface and count in predicted mask area.
    foreground = np.isin(semantic, list(countable)) & (instance >= 1000)
    gt_ids = np.where(foreground, instance, -1)
    result = CanonicalGT(scene_id, xyz, gt_ids, semantic,
        np.ones(len(xyz), dtype=bool), np.zeros(len(xyz), dtype=bool),
        metadata={"source_reference": str(reference_path),
                  "source_reference_sha256": manifest["reference_sha256"],
                  "source_manifest": str(manifest_path),
                  "source_manifest_sha256": sha256_file(manifest_path),
                  "source_mesh": manifest["mesh"],
                  "source_mesh_sha256": manifest["mesh_sha256"],
                  "source_instance_labels": manifest["instance_label_file"],
                  "source_instance_labels_sha256": manifest["instance_label_sha256"],
                  "source_semantic_labels": manifest["semantic_label_file"],
                  "source_semantic_labels_sha256": manifest["semantic_label_sha256"],
                  "existing_protocol": manifest["protocol"],
                  "countable_classes": manifest["instance_classes"]})
    result.validate()
    return result
