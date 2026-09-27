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
from .evaluate import evaluate_scenes
from .io import load_gt, load_prediction, save_gt, save_prediction, sha256_file
from .official_native import run_official_native
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
             args.debug_min_valid_instance_vertices is not None or
             raw.get("frozen") is not True)
    if args.debug_max_distance_m is not None:
        raw["geometry_mapping"]["max_distance_m"] = args.debug_max_distance_m
    if args.debug_min_valid_instance_vertices is not None:
        raw["instance_filter"]["min_valid_instance_vertices"] = args.debug_min_valid_instance_vertices
    protocol = Protocol.from_dict(raw)
    return protocol, raw, debug


def add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug-max-distance-m", type=float)
    parser.add_argument("--debug-min-valid-instance-vertices", type=int)


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
        method_commit=args.method_commit, protocol_version=protocol.name)
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


def cmd_eval_scene(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    gt = load_gt(args.gt)
    pred = load_prediction(args.pred)
    summary, per_scene, overlaps = evaluate_scenes([(gt, pred)], protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = per_scene[0]
    status = "DEBUG_ONLY / NON_OFFICIAL" if debug or pred.metadata.get("debug_only") else "protocol_configured"
    metrics["status"] = status
    write_json(args.out / "metrics.json", metrics)
    overlap = overlaps[0]
    np.savez_compressed(args.out / "overlap_matrix.npz",
        pred_uids=np.array(overlap.pred_uids, dtype=str), gt_ids=overlap.gt_ids,
        intersection=overlap.intersection, iou=overlap.iou,
        precision=overlap.precision, recall=overlap.recall,
        pred_size=overlap.pred_size, gt_size=overlap.gt_size,
        pred_void_fraction=overlap.pred_void_fraction)
    write_json(args.out / "manifest.json", {
        "status": status, "protocol": protocol.name, "protocol_config": raw,
        "protocol_config_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "evaluator_version": __version__, "scene_id": gt.scene_id,
        "dataset": protocol.dataset, "method": pred.method_name,
        "method_commit": pred.method_commit, "adapter_version": pred.adapter_version,
        "confidence_mode": protocol.confidence_mode,
        "gt_file": str(args.gt), "gt_sha256": sha256_file(args.gt),
        "prediction_file": str(args.pred), "prediction_sha256": sha256_file(args.pred),
        "reference_source": gt.metadata,
        "prediction_source": pred.metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Formal aggregate files are deliberately absent while parameters are not frozen.
    print(json.dumps({"status": status, "scene": gt.scene_id,
        "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
        "CA_AP25_uniform": summary["CA_AP25_uniform"], "CA_PQ": metrics["CA_PQ"]["PQ"]}, ensure_ascii=False))


def cmd_eval_batch(args: argparse.Namespace) -> None:
    protocol, raw, debug = load_protocol(args.config, args)
    entries = json.loads(args.scenes.read_text(encoding="utf-8"))["scenes"]
    if not entries:
        raise EvaluationError("Batch scene list is empty")
    scenes = [(load_gt(entry["gt"]), load_prediction(entry["prediction"])) for entry in entries]
    if len({gt.scene_id for gt, _ in scenes}) != len(scenes):
        raise EvaluationError("Batch contains duplicate scene IDs")
    summary, per_scene, _ = evaluate_scenes(scenes, protocol)
    status = "DEBUG_ONLY / NON_OFFICIAL" if debug or any(p.metadata.get("debug_only") for _, p in scenes) else "protocol_configured"
    summary["status"] = status
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "manifest.json", {
        "status": status, "protocol_config": raw,
        "protocol_config_sha256": sha256_file(args.config),
        "effective_protocol_sha256": effective_protocol_sha256(raw),
        "evaluator_version": __version__,
        "scene_inputs": [{"scene": gt.scene_id, "gt": entry["gt"],
            "gt_sha256": sha256_file(entry["gt"]), "prediction": entry["prediction"],
            "prediction_sha256": sha256_file(entry["prediction"]),
            "reference_source": gt.metadata} for entry, (gt, _) in zip(entries, scenes)],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    fields = ("scene_id", "CA_AP_uniform", "CA_AP50_uniform", "CA_AP25_uniform", "CA_PQ", "CA_SQ", "CA_RQ", "TP", "FP", "FN", "status")
    with (args.out / "per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_scene:
            pq = row["CA_PQ"]
            writer.writerow({"scene_id": row["scene_id"], "CA_AP_uniform": row["CA_AP_uniform"],
                "CA_AP50_uniform": row["CA_AP50_uniform"], "CA_AP25_uniform": row["CA_AP25_uniform"],
                "CA_PQ": pq["PQ"], "CA_SQ": pq["SQ"], "CA_RQ": pq["RQ"],
                "TP": pq["TP"], "FP": pq["FP"], "FN": pq["FN"], "status": status})
    if status == "protocol_configured":
        with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("protocol", "scene_count", "CA_AP_uniform", "CA_AP50_uniform", "CA_AP25_uniform", "CA_PQ", "CA_SQ", "CA_RQ"))
            writer.writeheader()
            writer.writerow({"protocol": summary["protocol"], "scene_count": summary["scene_count"],
                "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
                "CA_AP25_uniform": summary["CA_AP25_uniform"],
                "CA_PQ": summary["CA_PQ"]["PQ"], "CA_SQ": summary["CA_PQ"]["SQ"],
                "CA_RQ": summary["CA_PQ"]["RQ"]})
    print(json.dumps({"status": status, "scene_count": len(scenes),
        "CA_AP_uniform": summary["CA_AP_uniform"], "CA_AP50_uniform": summary["CA_AP50_uniform"],
        "CA_AP25_uniform": summary["CA_AP25_uniform"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified class-agnostic instance evaluator v1")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-replica-gt")
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
    scene = commands.add_parser("eval-scene")
    add_protocol_args(scene)
    scene.add_argument("--gt", type=Path, required=True)
    scene.add_argument("--pred", type=Path, required=True)
    scene.add_argument("--out", type=Path, required=True)
    scene.set_defaults(func=cmd_eval_scene)
    batch = commands.add_parser("eval-batch")
    add_protocol_args(batch)
    batch.add_argument("--scenes", type=Path, required=True, help="JSON object with scenes:[{gt,prediction}]")
    batch.add_argument("--out", type=Path, required=True)
    batch.set_defaults(func=cmd_eval_batch)
    official = commands.add_parser("eval-official-native")
    official.add_argument("--recipe", type=Path, required=True)
    official.add_argument("--out", type=Path, required=True)
    official.set_defaults(func=lambda args: print(json.dumps(run_official_native(args.recipe, args.out))))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
