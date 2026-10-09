"""Reproducible adapters and paired scoring for the explicit V3 repair profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate_scenes
from .geometry import (attach_reference_support, build_fixed_surface_correspondence,
    load_fixed_surface_correspondence, map_fixed_surface_labels, native_reference_region_support,
    save_fixed_surface_correspondence)
from .io import load_gt, load_prediction, save_prediction, sha256_file
from .metrics import build_overlap, owner_coverage
from .schema import EvaluationError, Protocol


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def load_surface_export(path: str | Path, exported_instance_ids: np.ndarray | None = None):
    """Normalize native exports without filtering any physical points or guessing classes."""
    with np.load(path, allow_pickle=False) as data:
        if {"xyz_m", "instance_id"}.issubset(data.files):
            xyz, labels = data["xyz_m"].copy(), data["instance_id"].copy()
            inventory = data["native_instance_ids"].copy() if "native_instance_ids" in data else exported_instance_ids
            if "native_instance_ids" in data and exported_instance_ids is not None and not np.array_equal(inventory, exported_instance_ids):
                raise EvaluationError("Explicit inventory disagrees with the native export")
            return xyz, labels, inventory, None, None, {"source_label_encoding": "positive_native_id_nonpositive_unassigned"}
        required = {"xyz", "instance", "native_instance_ids"}
        if not required.issubset(data.files):
            raise EvaluationError("Surface export requires TSDF or OVI-MAP native fields")
        if exported_instance_ids is not None:
            raise EvaluationError("OVI-MAP already exports an explicit original instance inventory")
        xyz, compact = data["xyz"].copy(), data["instance"].copy()
        original = data["native_instance_ids"].copy()
        classes = data["classes"].copy() if "classes" in data else None
    if compact.shape != (len(xyz),) or not np.issubdtype(compact.dtype, np.integer):
        raise EvaluationError("OVI-MAP compact owners must be an integer vector on all native points")
    if original.ndim != 1 or not np.issubdtype(original.dtype, np.integer) or np.any(original < 0) or len(np.unique(original)) != len(original):
        raise EvaluationError("OVI-MAP original inventory must be unique nonnegative IDs")
    if np.any(compact[compact >= 0] >= len(original)):
        raise EvaluationError("OVI-MAP compact owner index outside its original inventory")
    if classes is not None and (classes.shape != original.shape or not np.issubdtype(classes.dtype, np.integer)):
        raise EvaluationError("Native semantic classes must align with the original OVI-MAP inventory")
    labels = np.where(compact >= 0, compact.astype(np.int64) + 1, -1)
    inventory = np.arange(1, len(original) + 1, dtype=np.int64)
    return xyz, labels, inventory, original, classes, {
        "source_label_encoding": "compact_index_into_original_inventory_normalized_to_positive_index_plus_one",
        "source_original_native_instance_ids": original.astype(int).tolist(),
        "semantic_classes_source": "native_export_only_no_GT_class_completion"}


def adapt_surface(surface_path: str | Path, gt_path: str | Path, config_path: str | Path,
                  correspondence_path: str | Path, output_dir: str | Path, *,
                  method_name: str, method_commit: str, exported_instance_ids: np.ndarray | None = None,
                  prediction_types: dict[int, str] | None = None,
                  source_provenance: dict | None = None) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    protocol = Protocol.from_dict(config)
    if not protocol.is_object_observed_repair:
        raise EvaluationError("This adapter requires object_observed_repair")
    gt = load_gt(gt_path)
    if gt.metadata.get("evaluation_profile") != protocol.evaluation_profile or gt.evaluation_region is None:
        raise EvaluationError("Surface adapter requires the fixed observed-object GT scope")
    xyz, labels, inventory, original_ids, classes, source_meta = load_surface_export(surface_path, exported_instance_ids)
    provenance = source_provenance or {}
    if provenance.get("frame_list_sha256") is not None and provenance["frame_list_sha256"] != gt.metadata.get("observation_metadata", {}).get("frame_list_sha256"):
        raise EvaluationError("Method input frame list differs from the fixed observed GT scope")
    cache_path, out = Path(correspondence_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        correspondence = load_fixed_surface_correspondence(cache_path, xyz, gt.xyz_ref)
        if correspondence.max_distance_m != protocol.geometry_mapping_max_distance_m:
            raise EvaluationError("Fixed cache distance differs from the selected profile")
    else:
        correspondence = build_fixed_surface_correspondence(xyz, gt.xyz_ref, protocol.geometry_mapping_max_distance_m)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_fixed_surface_correspondence(cache_path, correspondence)
    result = map_fixed_surface_labels(xyz, labels, gt.xyz_ref, correspondence,
        scene_id=gt.scene_id, method_name=method_name, method_commit=method_commit,
        native_instance_ids=inventory, diagnostic_distance_m=protocol.diagnostic_max_distance_m,
        metadata={**source_meta, "source_provenance": provenance,
            "method_input_scope_verified": provenance.get("input_scope_verified") is True,
            "source_instance_surface": str(surface_path),
            "source_instance_surface_sha256": sha256_file(surface_path),
            "profile_config_sha256": sha256_file(config_path)})
    support = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region,
                                              protocol.geometry_mapping_max_distance_m)
    attach_reference_support(result, labels, support, gt, correspondence)
    for prediction in (result.prediction, result.diagnostic_prediction):
        for index, instance in enumerate(prediction.instances):
            if original_ids is not None:
                instance.metadata["normalized_positive_instance_id"] = instance.metadata["native_instance_id"]
                instance.metadata["native_instance_id"] = int(original_ids[index])
                instance.instance_uid = f"ovi-id:{int(original_ids[index])}"
                if classes is not None:
                    instance.metadata["native_semantic_class_id"] = int(classes[index])
                    instance.semantic_id = int(classes[index]) if classes[index] > 0 else None
            kind = (prediction_types or {}).get(instance.metadata["native_instance_id"], "unknown")
            if kind not in ("unknown", "object", "structure"):
                raise EvaluationError("Prediction type must originate from a native object/structure/unknown declaration")
            instance.metadata["prediction_type"] = kind
    main_path, diagnostic_path = out / "canonical_prediction.npz", out / "diagnostic_support_prediction.npz"
    save_prediction(main_path, result.prediction)
    save_prediction(diagnostic_path, result.diagnostic_prediction)
    summary, rows, overlaps = evaluate_scenes([(gt, result.prediction)], protocol,
        diagnostic_predictions=[result.diagnostic_prediction])
    write_json(out / "metrics.json", {"summary": summary, "per_scene": rows})
    overlap = overlaps[0]
    np.savez_compressed(out / "overlap_matrix.npz", iou=overlap.iou,
        intersection=overlap.intersection, gt_ids=overlap.gt_ids, pred_uids=np.asarray(overlap.pred_uids),
        pred_size=overlap.pred_size, gt_size=overlap.gt_size,
        unmatched_ignore_mask=overlap.unmatched_ignore_mask,
        unmatched_policy=np.asarray(overlap.unmatched_policy))
    diagnostic_overlap = build_overlap(gt, result.diagnostic_prediction, protocol)
    np.savez_compressed(out / "pairwise_support_matrix.npz", intersection=diagnostic_overlap.intersection,
        recall=diagnostic_overlap.recall, iou=diagnostic_overlap.iou,
        gt_ids=diagnostic_overlap.gt_ids, pred_uids=np.asarray(diagnostic_overlap.pred_uids))
    manifest = {"status": summary["status"], "protocol": protocol.name,
        "evaluation_profile": protocol.evaluation_profile, "profile_config": config,
        "profile_config_sha256": sha256_file(config_path), "gt_file": str(gt_path),
        "gt_file_sha256": sha256_file(gt_path), "source_surface_sha256": sha256_file(surface_path),
        "source_surface": str(surface_path), "correspondence_cache": str(cache_path),
        "correspondence_file_sha256": sha256_file(cache_path),
        "geometry_xyz_sha256": correspondence.native_xyz_sha256,
        "correspondence_sha256": correspondence.sha256,
        "gt_scope_sha256": gt.metadata["gt_scope_sha256"],
        "gt_observed_support_sha256": gt.metadata["gt_observed_support_sha256"],
        "canonical_prediction_sha256": sha256_file(main_path),
        "diagnostic_prediction_sha256": sha256_file(diagnostic_path),
        "adapter_statistics": result.statistics, "method_name": method_name,
        "source_provenance": provenance,
        "method_commit": method_commit,
        "evaluator_code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
            ("schema.py", "io.py", "gt_scope.py", "geometry.py", "metrics.py", "evaluate.py", "repair_profile.py")}}
    write_json(out / "adapter_manifest.json", manifest)
    return {"manifest": manifest, "summary": summary}


def paired_revision_metrics(gt, before, after, protocol: Protocol) -> dict:
    if not protocol.is_object_observed_repair:
        raise EvaluationError("Paired surface revisions require the explicit repair profile")
    for key in ("geometry_xyz_sha256", "correspondence_sha256", "reference_xyz_sha256",
                "gt_scope_sha256", "gt_observed_support_sha256", "evaluation_region_sha256", "profile_config_sha256"):
        if not before.metadata.get(key) or before.metadata[key] != after.metadata.get(key):
            raise EvaluationError(f"Paired revision changes its physical surface/scope/profile: {key}")
    first, second = build_overlap(gt, before, protocol), build_overlap(gt, after, protocol)
    if not np.array_equal(first.gt_ids, second.gt_ids) or not np.array_equal(first.gt_size, second.gt_size):
        raise EvaluationError("Paired revision GT denominator changed")
    best = lambda o: o.iou.max(axis=0) if len(o.pred_uids) else np.zeros(len(o.gt_ids))
    old, new = best(first), best(second)
    delta = new - old
    rows = [{"raw_gt_id": int(raw_id), "before_best_iou": float(a), "after_best_iou": float(b),
             "delta_iou": float(b-a), "change": "IMPROVED" if b-a > 1e-12 else "DEGRADED" if a-b > 1e-12 else "UNCHANGED"}
            for raw_id, a, b in zip(first.gt_ids, old, new)]
    old_coverage, new_coverage = owner_coverage(first), owner_coverage(second)
    return {"evaluation_profile": protocol.evaluation_profile,
        "geometry_xyz_sha256": before.metadata["geometry_xyz_sha256"],
        "correspondence_sha256": before.metadata["correspondence_sha256"],
        "gt_scope_sha256": before.metadata["gt_scope_sha256"],
        "corrected_instances": int(np.count_nonzero(delta > 1e-12)),
        "degraded_instances": int(np.count_nonzero(delta < -1e-12)),
        "unchanged_instances": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        "instance_change_definition": "best-GT IoU improves/degrades on exactly the same observed reference surface",
        "correct_owner_vertices_before": old_coverage["correct_owner_vertices"],
        "correct_owner_vertices_after": new_coverage["correct_owner_vertices"],
        "correct_owner_vertex_gain": new_coverage["correct_owner_vertices"] - old_coverage["correct_owner_vertices"],
        "per_gt": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    adapt = sub.add_parser("adapt")
    for flag in ("surface", "gt", "config", "correspondence", "output-dir"):
        adapt.add_argument("--" + flag, type=Path, required=True)
    adapt.add_argument("--method-name", required=True)
    adapt.add_argument("--method-commit", required=True)
    adapt.add_argument("--exported-instance-list", type=Path)
    adapt.add_argument("--source-provenance", type=Path)
    pair = sub.add_parser("paired")
    for flag in ("before", "after", "gt", "config", "output"):
        pair.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "adapt":
        ids = np.asarray(json.loads(args.exported_instance_list.read_text()), dtype=np.int64) if args.exported_instance_list else None
        value = adapt_surface(args.surface, args.gt, args.config, args.correspondence, args.output_dir,
            method_name=args.method_name, method_commit=args.method_commit, exported_instance_ids=ids,
            source_provenance=json.loads(args.source_provenance.read_text(encoding='utf-8')) if args.source_provenance else None)
        print(json.dumps({"status": value["summary"]["status"], "output_dir": str(args.output_dir)}))
    else:
        gt, before, after = load_gt(args.gt), load_prediction(args.before), load_prediction(args.after)
        protocol = Protocol.from_dict(json.loads(args.config.read_text()))
        value = paired_revision_metrics(gt, before, after, protocol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, value)
        print(json.dumps({k:v for k,v in value.items() if k != "per_gt"}))


if __name__ == "__main__":
    main()
