from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .conceptgraphs import adapt_map
from .current_protocol import (CURRENT_CONFIG, CURRENT_PROTOCOL, require_current_gt,
    require_current_protocol, reject_current_debug_overrides)
from .evaluate import evaluate_scenes
from .io import load_gt, load_prediction, save_gt, save_prediction, sha256_file
from .metrics import build_overlap
from .official_native import run_official_native
from .online import evaluate_online_prefixes
from .ovimap import adapt_export as adapt_ovimap_export
from .replica import load_existing_reference
from .schema import EvaluationError, Protocol


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else
        x.tolist() if isinstance(x, np.ndarray) else _reject_json_type(x)) + "\n", encoding="utf-8")


def _reject_json_type(value):
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def effective_protocol_sha256(raw: dict) -> str:
    blob = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_protocol(path: Path, args: argparse.Namespace) -> tuple[Protocol, dict, bool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    debug = (args.debug_max_distance_m is not None or
             getattr(args, "debug_diagnostic_max_distance_m", None) is not None or
             args.debug_min_valid_instance_vertices is not None or
             args.debug_significant_min_vertices is not None or
             args.debug_significant_min_gt_fraction is not None or
             raw.get("frozen") is not True)
    if args.debug_max_distance_m is not None:
        raw["geometry_mapping"]["max_distance_m"] = args.debug_max_distance_m
    if getattr(args, "debug_diagnostic_max_distance_m", None) is not None:
        raw["diagnostic_mapping"]["max_distance_m"] = args.debug_diagnostic_max_distance_m
    if args.debug_min_valid_instance_vertices is not None:
        raw["instance_filter"]["min_valid_instance_vertices"] = args.debug_min_valid_instance_vertices
    if args.debug_significant_min_vertices is not None:
        raw["significant_overlap"]["min_intersection_vertices"] = args.debug_significant_min_vertices
    if args.debug_significant_min_gt_fraction is not None:
        raw["significant_overlap"]["min_gt_fraction"] = args.debug_significant_min_gt_fraction
    protocol = Protocol.from_dict(raw)
    return protocol, raw, debug


def add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CURRENT_CONFIG,
        help="Default: locked object_observed_repair revision 2")
    parser.add_argument("--historical-protocol", action="store_true",
        help="Explicit historical reproduction; keep its outputs separate from the current baseline")
    parser.add_argument("--debug-max-distance-m", type=float)
    parser.add_argument("--debug-diagnostic-max-distance-m", type=float)
    parser.add_argument("--debug-min-valid-instance-vertices", type=int)
    parser.add_argument("--debug-significant-min-vertices", type=int)
    parser.add_argument("--debug-significant-min-gt-fraction", type=float)


def cmd_export_replica_gt(args: argparse.Namespace) -> None:
    gt = load_existing_reference(args.reference_root, args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gt(args.out, gt)
    print(f"Saved {args.out} ({gt.vertex_count} reference vertices)")


def cmd_adapt_conceptgraphs(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    gt = load_gt(args.gt)
    experiment = json.loads(args.experiment_manifest.read_text(encoding="utf-8"))
    matching = [entry for entry in experiment.get("entries", [])
                if Path(entry.get("map", "")).resolve() == args.map.resolve() and entry.get("scene") == gt.scene_id]
    if len(matching) != 1:
        raise EvaluationError("Experiment manifest must contain exactly one matching scene/map entry")
    source_entry = matching[0]
    source_input = source_entry.get("input", {})
    frame_keys = ("start", "end", "stride")
    if not all(key in source_input for key in frame_keys):
        raise EvaluationError("Experiment manifest lacks start/end/stride")
    start, end, stride = (int(source_input[key]) for key in frame_keys)
    if start < 0 or end <= start or stride <= 0:
        raise EvaluationError("Experiment frame range is invalid")
    frame_list = list(range(start, end, stride))
    if source_entry.get("cost", {}).get("frames") != len(frame_list):
        raise EvaluationError("Frame list length differs from experiment's recorded frame count")
    frame_hash = hashlib.sha256(json.dumps(frame_list, separators=(",", ":")).encode()).hexdigest()
    map_config_path = args.map.parent / "config_params.json"
    map_config = json.loads(map_config_path.read_text(encoding="utf-8")) if map_config_path.exists() else {}
    trajectory_path = Path(map_config["dataset_root"]) / gt.scene_id / "traj.txt" if "dataset_root" in map_config else None
    result = adapt_map(args.map, gt.xyz_ref, protocol.geometry_mapping_max_distance_m,
        scene_id=gt.scene_id, method_name=args.method_name,
        method_commit=args.method_commit, protocol_version=protocol.name,
        mapping_method=protocol.geometry_mapping_method,
        diagnostic_distance_m=protocol.diagnostic_max_distance_m)
    result.prediction.metadata.update({
        "source_experiment_manifest": str(args.experiment_manifest),
        "source_experiment_manifest_sha256": sha256_file(args.experiment_manifest),
        "source_experiment_protocol": experiment.get("protocol"),
        "frame_range_from_source": {key: source_input[key] for key in frame_keys},
        "frame_count_from_source": len(frame_list), "frame_list_sha256": frame_hash,
        "source_pose": source_input.get("pose_source"),
        "source_depth": source_input.get("depth_source"),
        "source_map_config": str(map_config_path) if map_config_path.exists() else None,
        "source_map_config_sha256": sha256_file(map_config_path) if map_config_path.exists() else None,
        "source_dataset_config": map_config.get("dataset_config"),
        "source_trajectory": str(trajectory_path) if trajectory_path and trajectory_path.exists() else None,
        "source_trajectory_sha256": sha256_file(trajectory_path) if trajectory_path and trajectory_path.exists() else None,
    })
    result.prediction.metadata["debug_only"] = debug
    args.out.mkdir(parents=True, exist_ok=True)
    save_prediction(args.out / "canonical_prediction.npz", result.prediction)
    if protocol.is_v3:
        if result.diagnostic_prediction is None:
            raise EvaluationError("Replica-CA-v3 adapter did not produce pairwise support")
        result.diagnostic_prediction.metadata["source_map_sha256"] = result.prediction.metadata["source_map_sha256"]
        save_prediction(args.out / "diagnostic_support_prediction.npz", result.diagnostic_prediction)
    write_json(args.out / "adapter_stats.json", result.statistics)
    write_json(args.out / "adapter_manifest.json", {
        "status": "DEBUG_ONLY / NON_OFFICIAL" if debug else "protocol_configured",
        "evaluator_version": __version__, "config_file": str(args.config),
        "config_file_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "gt_file": str(args.gt), "gt_file_sha256": sha256_file(args.gt),
        "source_map": str(args.map), "source_map_sha256": sha256_file(args.map),
        "source_experiment_manifest": str(args.experiment_manifest),
        "source_experiment_manifest_sha256": sha256_file(args.experiment_manifest),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Saved canonical prediction to {args.out}")


def cmd_adapt_ovimap(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    if not protocol.retains_predictions:
        raise EvaluationError("OVI-MAP adapter requires Replica-CA-v2 or v3")
    gt = load_gt(args.gt)
    source = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    alignment = json.loads(args.input_alignment.read_text(encoding="utf-8"))
    frames = alignment.get("source_frames")
    if (not isinstance(frames, list) or len(frames) != alignment.get("frames") or
            not all(isinstance(x, int) and x >= 0 for x in frames) or
            any(b <= a for a, b in zip(frames, frames[1:]))):
        raise EvaluationError("OVI-MAP input alignment must give an increasing source frame list")
    result = adapt_ovimap_export(args.export_npz, gt.xyz_ref,
        protocol.geometry_mapping_max_distance_m, scene_id=gt.scene_id,
        method_name=args.method_name, method_commit=args.method_commit,
        protocol_version=protocol.name, mapping_method=protocol.geometry_mapping_method,
        diagnostic_distance_m=protocol.diagnostic_max_distance_m)
    if source.get("vertices") != result.statistics["source_vertex_count"] or \
            source.get("native_instance_count") != result.statistics["num_native_objects"]:
        raise EvaluationError("OVI-MAP export manifest count differs from its NPZ")
    result.prediction.metadata.update({
        "source_export_manifest": str(args.export_manifest),
        "source_export_manifest_sha256": sha256_file(args.export_manifest),
        "source_mesh": source.get("mesh"), "source_mesh_sha256": source.get("mesh_sha256"),
        "source_features": source.get("features"),
        "source_features_sha256": source.get("features_sha256"),
        "source_input_alignment": str(args.input_alignment),
        "source_input_alignment_sha256": sha256_file(args.input_alignment),
        "frame_count_from_source": len(frames),
        "frame_list_sha256": hashlib.sha256(json.dumps(frames, separators=(",", ":")).encode()).hexdigest(),
        "debug_only": debug,
    })
    args.out.mkdir(parents=True, exist_ok=True)
    save_prediction(args.out / "canonical_prediction.npz", result.prediction)
    if protocol.is_v3:
        if result.diagnostic_prediction is None:
            raise EvaluationError("Replica-CA-v3 adapter did not produce pairwise support")
        result.diagnostic_prediction.metadata["source_export_sha256"] = result.prediction.metadata["source_export_sha256"]
        save_prediction(args.out / "diagnostic_support_prediction.npz", result.diagnostic_prediction)
    write_json(args.out / "adapter_stats.json", result.statistics)
    write_json(args.out / "adapter_manifest.json", {
        "status": "DEBUG_ONLY / NON_OFFICIAL" if debug else "protocol_configured",
        "evaluator_version": __version__, "config_file": str(args.config),
        "config_file_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "gt_file": str(args.gt), "gt_file_sha256": sha256_file(args.gt),
        "source_export": str(args.export_npz), "source_export_sha256": sha256_file(args.export_npz),
        "source_export_manifest": str(args.export_manifest),
        "source_export_manifest_sha256": sha256_file(args.export_manifest),
        "source_input_alignment": str(args.input_alignment),
        "source_input_alignment_sha256": sha256_file(args.input_alignment),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Saved OVI-MAP canonical prediction to {args.out}")


def cmd_eval_scene(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    gt = load_gt(args.gt)
    pred = load_prediction(args.pred)
    diag_path = (getattr(args, "diagnostic_pred", None) or
        args.pred.parent / "diagnostic_support_prediction.npz") if protocol.is_v3 else None
    if protocol.is_v3 and not diag_path.is_file():
        raise EvaluationError("Replica-CA-v3 requires the matching diagnostic support prediction")
    diag = load_prediction(diag_path) if diag_path is not None else None
    if diag is not None and diag.metadata.get("role") != "structure_diagnostics_only":
        raise EvaluationError("Diagnostic prediction has the wrong role")
    if diag is not None:
        for key in ("source_map_sha256", "source_export_sha256"):
            if key in pred.metadata and diag.metadata.get(key) != pred.metadata[key]:
                raise EvaluationError("Diagnostic and main predictions come from different native maps")
    summary, per_scene, overlaps = evaluate_scenes([(gt, pred)], protocol,
        diagnostic_predictions=[diag] if diag is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = per_scene[0]
    status = "DEBUG_ONLY / NON_OFFICIAL" if debug or pred.metadata.get("debug_only") else "protocol_configured"
    if protocol.is_object_observed_repair and protocol.profile_revision == 2:
        status = summary["status"]
    metrics["status"] = status
    write_json(args.out / "metrics.json", metrics)
    overlap = overlaps[0]
    np.savez_compressed(args.out / "overlap_matrix.npz",
        pred_uids=np.array(overlap.pred_uids, dtype=str), gt_ids=overlap.gt_ids,
        intersection=overlap.intersection, iou=overlap.iou,
        precision=overlap.precision, recall=overlap.recall,
        pred_size=overlap.pred_size, gt_size=overlap.gt_size,
        pred_void_fraction=overlap.pred_void_fraction,
        **({"pred_ignore_fraction": overlap.pred_void_fraction} if protocol.retains_predictions else {}))
    if diag is not None:
        pairwise = build_overlap(gt, diag, protocol)
        np.savez_compressed(args.out / "pairwise_support_matrix.npz",
            pred_uids=np.array(pairwise.pred_uids, dtype=str), gt_ids=pairwise.gt_ids,
            supported_gt_vertices=pairwise.intersection,
            supported_gt_fraction=pairwise.recall,
            independent_iou=pairwise.iou,
            gt_size=pairwise.gt_size, pred_supported_size=pairwise.pred_size,
            diagnostic_max_distance_m=protocol.diagnostic_max_distance_m)
    write_json(args.out / "manifest.json", {
        "status": status, "protocol": protocol.name, "protocol_config": raw,
        "evaluation_profile": protocol.evaluation_profile, "profile_revision": protocol.profile_revision,
        "protocol_config_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "evaluator_version": __version__, "scene_id": gt.scene_id,
        "dataset": protocol.dataset, "method": pred.method_name,
        "method_commit": pred.method_commit, "adapter_version": pred.adapter_version,
        "confidence_mode": protocol.confidence_mode,
        "gt_file": str(args.gt), "gt_sha256": sha256_file(args.gt),
        "prediction_file": str(args.pred), "prediction_sha256": sha256_file(args.pred),
        "diagnostic_prediction_file": str(diag_path) if diag_path else None,
        "diagnostic_prediction_sha256": sha256_file(diag_path) if diag_path else None,
        "reference_source": gt.metadata,
        "prediction_source": pred.metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Formal aggregate files are deliberately absent while parameters are not frozen.
    print(json.dumps({"status": status, "scene": gt.scene_id,
        "evaluation_profile": protocol.evaluation_profile, "profile_revision": protocol.profile_revision,
        "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
        "CA_AP25_uniform": summary["CA_AP25_uniform"], "CA_PQ": metrics["CA_PQ"]["PQ"]}, ensure_ascii=False))


def cmd_eval_batch(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    entries = json.loads(args.scenes.read_text(encoding="utf-8"))["scenes"]
    if not entries:
        raise EvaluationError("Batch scene list is empty")
    scenes = [(load_gt(entry["gt"]), load_prediction(entry["prediction"])) for entry in entries]
    diagnostic_predictions = None
    if protocol.is_v3:
        if any("diagnostic_prediction" not in entry for entry in entries):
            raise EvaluationError("Replica-CA-v3 batch requires diagnostic_prediction for every scene")
        diagnostic_predictions = [load_prediction(entry["diagnostic_prediction"]) for entry in entries]
        if any(p.metadata.get("role") != "structure_diagnostics_only" for p in diagnostic_predictions):
            raise EvaluationError("Batch diagnostic prediction has the wrong role")
        for (_, pred), diag in zip(scenes, diagnostic_predictions):
            for key in ("source_map_sha256", "source_export_sha256"):
                if key in pred.metadata and diag.metadata.get(key) != pred.metadata[key]:
                    raise EvaluationError("Batch diagnostic and main predictions come from different native maps")
    if len({gt.scene_id for gt, _ in scenes}) != len(scenes):
        raise EvaluationError("Batch contains duplicate scene IDs")
    summary, per_scene, _ = evaluate_scenes(scenes, protocol,
        diagnostic_predictions=diagnostic_predictions)
    status = "DEBUG_ONLY / NON_OFFICIAL" if debug or any(p.metadata.get("debug_only") for _, p in scenes) else "protocol_configured"
    if protocol.is_object_observed_repair and protocol.profile_revision == 2:
        status = summary["status"]
    summary["status"] = status
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "manifest.json", {
        "status": status, "protocol_config": raw,
        "evaluation_profile": protocol.evaluation_profile, "profile_revision": protocol.profile_revision,
        "protocol_config_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "evaluator_version": __version__,
        "scene_inputs": [{"scene": gt.scene_id, "gt": entry["gt"],
            "gt_sha256": sha256_file(entry["gt"]), "prediction": entry["prediction"],
            "prediction_sha256": sha256_file(entry["prediction"]),
            "diagnostic_prediction": entry.get("diagnostic_prediction"),
            "diagnostic_prediction_sha256": sha256_file(entry["diagnostic_prediction"])
                if "diagnostic_prediction" in entry else None,
            "reference_source": gt.metadata} for entry, (gt, _) in zip(entries, scenes)],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    fields = ["scene_id", "CA_AP_uniform", "CA_AP50_uniform", "CA_AP25_uniform",
              "CA_PQ", "CA_SQ", "CA_RQ", "TP", "FP", "FN"]
    if protocol.retains_predictions:
        fields += ["CA_P_0_5", "CA_R_0_5", "CA_F1_0_5", "split_gt_rate",
                   "merge_prediction_rate", "duplicate_prediction_rate"]
    fields += ["status"]
    with (args.out / "per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_scene:
            pq = row["CA_PQ"]
            values = {"scene_id": row["scene_id"], "CA_AP_uniform": row["CA_AP_uniform"],
                "CA_AP50_uniform": row["CA_AP50_uniform"], "CA_AP25_uniform": row["CA_AP25_uniform"],
                "CA_PQ": pq["PQ"], "CA_SQ": pq["SQ"], "CA_RQ": pq["RQ"],
                "TP": pq["TP"], "FP": pq["FP"], "FN": pq["FN"], "status": status}
            if protocol.retains_predictions:
                prf, structure = row["CA_PRF1_0_5"], row["structure"]
                values.update({"CA_P_0_5": prf["P"], "CA_R_0_5": prf["R"],
                               "CA_F1_0_5": prf["F1"],
                               "split_gt_rate": structure["split_gt_rate"],
                               "merge_prediction_rate": structure["merge_prediction_rate"],
                               "duplicate_prediction_rate": structure["duplicate_prediction_rate"]})
            writer.writerow(values)
    if status == "protocol_configured":
        with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            aggregate_fields = ["protocol", "scene_count", "CA_AP_uniform", "CA_AP50_uniform",
                                "CA_AP25_uniform", "CA_PQ", "CA_SQ", "CA_RQ"]
            if protocol.retains_predictions:
                aggregate_fields += ["CA_F1_0_5", "split_gt_rate", "merge_prediction_rate",
                                     "duplicate_prediction_rate", "macro_CA_AP50_uniform", "macro_CA_F1_0_5"]
            writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
            writer.writeheader()
            values = {"protocol": summary["protocol"], "scene_count": summary["scene_count"],
                "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
                "CA_AP25_uniform": summary["CA_AP25_uniform"],
                "CA_PQ": summary["CA_PQ"]["PQ"], "CA_SQ": summary["CA_PQ"]["SQ"],
                "CA_RQ": summary["CA_PQ"]["RQ"]}
            if protocol.retains_predictions:
                values.update({"CA_F1_0_5": summary["CA_PRF1_0_5"]["F1"],
                               "split_gt_rate": summary["structure"]["split_gt_rate"],
                               "merge_prediction_rate": summary["structure"]["merge_prediction_rate"],
                               "duplicate_prediction_rate": summary["structure"]["duplicate_prediction_rate"],
                               "macro_CA_AP50_uniform": summary["macro_per_scene"]["CA_AP50_uniform"],
                               "macro_CA_F1_0_5": summary["macro_per_scene"]["CA_F1_0_5"]})
            writer.writerow(values)
    print(json.dumps({"status": status, "scene_count": len(scenes),
        "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
        "CA_AP25_uniform": summary["CA_AP25_uniform"]}))


def cmd_eval_online_prefix(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    gt = load_gt(args.gt)
    rows, provenance = evaluate_online_prefixes(gt, protocol, args.online_manifest)
    status = "DEBUG_ONLY / NON_OFFICIAL" if debug or any(row["prediction_debug_only"] for row in rows) else "protocol_configured"
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "prefix_metrics.json", {"status": status, "scene_id": gt.scene_id,
                                                   "checkpoints": rows})
    write_json(args.out / "manifest.json", {
        "status": status, "protocol": protocol.name, "protocol_config": raw,
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "protocol_config_sha256": sha256_file(args.config),
        "evaluator_version": __version__, "gt_file": str(args.gt),
        "gt_sha256": sha256_file(args.gt),
        "online_manifest": str(args.online_manifest),
        "online_manifest_sha256": sha256_file(args.online_manifest),
        "source": provenance, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    fields = ("frame_id", "input_frame_count", "observed_reference_vertices",
              "CA_AP50_uniform", "CA_P_0_5", "CA_R_0_5", "CA_F1_0_5", "CA_PQ",
              "split_gt_rate", "merge_prediction_rate", "duplicate_prediction_rate", "status")
    with (args.out / "prefix_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            prf, structure = row["CA_PRF1_0_5"], row["structure"]
            writer.writerow({"frame_id": row["frame_id"],
                "input_frame_count": row["input_frame_count"],
                "observed_reference_vertices": row["observed_reference_vertices"],
                "CA_AP50_uniform": row["CA_AP50_uniform"],
                "CA_P_0_5": prf["P"], "CA_R_0_5": prf["R"], "CA_F1_0_5": prf["F1"],
                "CA_PQ": row["CA_PQ"]["PQ"],
                "split_gt_rate": structure["split_gt_rate"],
                "merge_prediction_rate": structure["merge_prediction_rate"],
                "duplicate_prediction_rate": structure["duplicate_prediction_rate"],
                "status": status})
    print(json.dumps({"status": status, "scene": gt.scene_id,
                      "checkpoint_count": len(rows)}, ensure_ascii=False))


def cmd_current_protocol(args: argparse.Namespace) -> None:
    require_current_protocol()
    print(json.dumps({**CURRENT_PROTOCOL, "resolved_config": str(CURRENT_CONFIG)},
        ensure_ascii=False, indent=2, sort_keys=True))


def cmd_adapt_surface(args: argparse.Namespace) -> None:
    from .repair_profile import adapt_surface
    protocol = Protocol.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    provenance = json.loads(args.source_provenance.read_text(encoding="utf-8")) if args.source_provenance else None
    value = adapt_surface(args.surface, args.gt, args.config, args.fixed_correspondence,
        args.out, method_name=args.method_name, method_commit=args.method_commit,
        source_provenance=provenance)
    print(json.dumps({"status": value["summary"]["status"],
        "evaluation_profile": protocol.evaluation_profile, "profile_revision": protocol.profile_revision,
        "config_sha256": sha256_file(args.config), "output_dir": str(args.out)}))


def cmd_eval_repair_pair(args: argparse.Namespace) -> None:
    from .repair_profile import paired_revision_metrics
    protocol, raw, _ = load_protocol(args.config, args)
    gt = load_gt(args.gt)
    value = paired_revision_metrics(gt, load_prediction(args.before), load_prediction(args.after), protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "paired_repair_metrics.json", value)
    write_json(args.out / "manifest.json", {"protocol_config": raw,
        "protocol_config_sha256": sha256_file(args.config), "gt_file": str(args.gt),
        "gt_sha256": sha256_file(args.gt), "before_file": str(args.before),
        "before_sha256": sha256_file(args.before), "after_file": str(args.after),
        "after_sha256": sha256_file(args.after)})
    print(json.dumps({"evaluation_profile": protocol.evaluation_profile,
        "profile_revision": protocol.profile_revision, "output_dir": str(args.out)}))


def validate_current_command(args: argparse.Namespace) -> None:
    if args.command == "adapt-surface" and any(getattr(args, name, None) is not None for name in (
            "debug_max_distance_m", "debug_diagnostic_max_distance_m", "debug_min_valid_instance_vertices",
            "debug_significant_min_vertices", "debug_significant_min_gt_fraction")):
        raise EvaluationError("adapt-surface reads the literal config; parameter experiments require a separate historical config file")
    if args.command == "export-replica-gt" and not args.historical_protocol:
        raise EvaluationError("The current baseline uses fixed canonical 350-GT files; legacy GT export requires --historical-protocol")
    if not hasattr(args, "config") or not require_current_protocol(args.config,
            historical_protocol=args.historical_protocol):
        return
    reject_current_debug_overrides(args)
    if args.command in ("adapt-conceptgraphs", "adapt-ovimap"):
        raise EvaluationError("Current revision 2 uses complete immutable native surfaces; use adapt-surface with --fixed-correspondence")
    if args.command == "eval-online-prefix":
        raise EvaluationError("The fixed 400-frame 350-GT scope is for final-map evaluation; online prefixes need a separately reviewed prefix scope")
    if hasattr(args, "gt"):
        require_current_gt(args.gt)
    if args.command == "eval-batch":
        for entry in json.loads(args.scenes.read_text(encoding="utf-8"))["scenes"]:
            require_current_gt(Path(entry["gt"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified class-agnostic instance evaluator v1/v2/v3")
    commands = parser.add_subparsers(dest="command", required=True)
    current = commands.add_parser("current-protocol", help="Show the locked current development baseline")
    current.set_defaults(func=cmd_current_protocol)
    export = commands.add_parser("export-replica-gt")
    export.add_argument("--historical-protocol", action="store_true")
    export.add_argument("--reference-root", type=Path, required=True)
    export.add_argument("--scene", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=cmd_export_replica_gt)
    adapt = commands.add_parser("adapt-conceptgraphs")
    add_protocol_args(adapt)
    adapt.add_argument("--gt", type=Path, required=True)
    adapt.add_argument("--map", type=Path, required=True)
    adapt.add_argument("--experiment-manifest", type=Path, required=True)
    adapt.add_argument("--method-name", required=True)
    adapt.add_argument("--method-commit", required=True)
    adapt.add_argument("--out", type=Path, required=True)
    adapt.set_defaults(func=cmd_adapt_conceptgraphs)
    ovi = commands.add_parser("adapt-ovimap")
    add_protocol_args(ovi)
    ovi.add_argument("--gt", type=Path, required=True)
    ovi.add_argument("--export-npz", type=Path, required=True)
    ovi.add_argument("--export-manifest", type=Path, required=True)
    ovi.add_argument("--input-alignment", type=Path, required=True)
    ovi.add_argument("--method-name", required=True)
    ovi.add_argument("--method-commit", required=True)
    ovi.add_argument("--out", type=Path, required=True)
    ovi.set_defaults(func=cmd_adapt_ovimap)
    surface = commands.add_parser("adapt-surface", help="Current fixed-surface P1/OVI adapter")
    add_protocol_args(surface)
    surface.add_argument("--gt", type=Path, required=True)
    surface.add_argument("--surface", type=Path, required=True)
    surface.add_argument("--fixed-correspondence", type=Path, required=True)
    surface.add_argument("--source-provenance", type=Path)
    surface.add_argument("--method-name", required=True)
    surface.add_argument("--method-commit", required=True)
    surface.add_argument("--out", type=Path, required=True)
    surface.set_defaults(func=cmd_adapt_surface)
    pair = commands.add_parser("eval-repair-pair", help="Paired label repair on exactly the same TSDF geometry")
    add_protocol_args(pair)
    pair.add_argument("--gt", type=Path, required=True)
    pair.add_argument("--before", type=Path, required=True)
    pair.add_argument("--after", type=Path, required=True)
    pair.add_argument("--out", type=Path, required=True)
    pair.set_defaults(func=cmd_eval_repair_pair)
    scene = commands.add_parser("eval-scene")
    add_protocol_args(scene)
    scene.add_argument("--gt", type=Path, required=True)
    scene.add_argument("--pred", type=Path, required=True)
    scene.add_argument("--diagnostic-pred", type=Path)
    scene.add_argument("--out", type=Path, required=True)
    scene.set_defaults(func=cmd_eval_scene)
    batch = commands.add_parser("eval-batch")
    add_protocol_args(batch)
    batch.add_argument("--scenes", type=Path, required=True, help="JSON object with scenes:[{gt,prediction}]")
    batch.add_argument("--out", type=Path, required=True)
    batch.set_defaults(func=cmd_eval_batch)
    online = commands.add_parser("eval-online-prefix")
    add_protocol_args(online)
    online.add_argument("--gt", type=Path, required=True)
    online.add_argument("--online-manifest", type=Path, required=True)
    online.add_argument("--out", type=Path, required=True)
    online.set_defaults(func=cmd_eval_online_prefix)
    official = commands.add_parser("eval-official-native")
    official.add_argument("--recipe", type=Path, required=True)
    official.add_argument("--out", type=Path, required=True)
    official.set_defaults(func=lambda args: print(json.dumps(run_official_native(args.recipe, args.out))))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_current_command(args)
    except (EvaluationError, OSError) as exc:
        parser.error(str(exc))
    args.func(args)


if __name__ == "__main__":
    main()
