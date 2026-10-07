#!/usr/bin/env python3
"""Run a causal fixed-mask, no-repair instance association baseline."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.association import OnlineVoxelAssociator  # noqa: E402
from revisable_instance_map.frame_io import ReplicaFrameSource  # noqa: E402
from revisable_instance_map.observations import RawInstanceObservation  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--allow-multiple-observations-per-instance-per-frame", action="store_true")
    parser.add_argument("--improved-association", action="store_true")
    parser.add_argument("--association-mode", choices=("legacy", "binary-ledger", "probabilistic"), default="legacy")
    parser.add_argument("--relative-weight-temperature", type=float, default=0.1)
    parser.add_argument("--null-candidate-score", type=float)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--valid-first-association-sampling", action="store_true")
    sampling.add_argument("--legacy-association-sampling", action="store_true")
    parser.add_argument("--parent-child-lineage", type=Path)
    args = parser.parse_args()
    if args.frame_count < 1 or args.checkpoint_every < 1:
        raise ValueError("Invalid frame or checkpoint count")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ReplicaFrameSource(args.config)
    frame_ids = source.frame_ids[:args.frame_count]
    if len(frame_ids) != args.frame_count:
        raise ValueError("Requested more frames than fixed input protocol")
    all_observations = defaultdict(list)
    for line in args.observations.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        all_observations[int(record["frame_id"])].append(RawInstanceObservation(**record))
    associator = OnlineVoxelAssociator(
        allow_multiple_observations_per_instance_per_frame=args.allow_multiple_observations_per_instance_per_frame,
        improved_association=args.improved_association,
        valid_first_sampling=(False if args.legacy_association_sampling else
                              True if args.valid_first_association_sampling else None),
        association_mode=args.association_mode,
        relative_weight_temperature=args.relative_weight_temperature,
        null_candidate_score=args.null_candidate_score,
    )
    trace_dir = args.output_dir / "candidate_support_traces"
    if associator.identity_evidence is not None:
        trace_dir.mkdir(exist_ok=True)
    lineage = {}
    if args.parent_child_lineage:
        for line in args.parent_child_lineage.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["parent_observation_id"] in lineage:
                raise ValueError("Duplicate parent lineage")
            lineage[row["parent_observation_id"]] = row
        expected = {obs.observation_id for frame_obs in all_observations.values() for obs in frame_obs}
        if set(lineage) != expected:
            raise ValueError("Parent lineage does not cover the observation catalog exactly")
    decision_counts = Counter()
    frame_summaries = []
    started = time.perf_counter()
    decision_path = args.output_dir / "associations.jsonl"
    temp_path = decision_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        for index, frame_id in enumerate(frame_ids, start=1):
            frame = source.load_frame(frame_id)
            observations = tuple(all_observations[frame_id])
            mask_path = source.mask_root / source.config["source"]["mask_pattern"].format(frame=frame_id)
            digest = sha256_file(mask_path)
            if any(item.source_mask_sha256 != digest for item in observations):
                raise ValueError(f"Mask checksum differs from observation catalog at {frame_id}")
            decisions = associator.process_frame(frame, observations)
            if associator.identity_evidence is not None:
                associator.save_candidate_support_trace(trace_dir / f"f{frame_id:06d}.npz", frame_id)
            if len(decisions) != len(observations):
                raise ValueError("An observation was not assigned")
            for decision in decisions:
                if lineage:
                    parent = lineage[decision["observation_id"]]
                    decision["geometric_child_ids"] = [item["refined_local_id"] for item in parent["children"]]
                    decision["residual_pixel_count"] = parent["residual_pixel_count"]
                decision_counts[decision["decision"]] += 1
                output.write(json.dumps(decision, ensure_ascii=False) + "\n")
            frame_summaries.append({
                "frame_id": frame_id,
                "observation_count": len(observations),
                "persistent_instance_count": len(associator.instances),
                "matched_count": sum(item["decision"].startswith("matched") for item in decisions),
                "new_count": sum(not item["decision"].startswith("matched") for item in decisions),
                "elapsed_seconds": time.perf_counter() - started,
            })
            if index % args.checkpoint_every == 0 or index == len(frame_ids):
                progress = {
                    "frames_processed": index,
                    "last_frame_id": frame_id,
                    "decisions": sum(decision_counts.values()),
                    "persistent_instances": len(associator.instances),
                    "matched": decision_counts["matched"] + decision_counts["matched_ambiguous"],
                    "low_margin_matches": decision_counts["matched_ambiguous"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                }
                write_json(args.output_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    temp_path.replace(decision_path)

    instance_path = args.output_dir / "instances.jsonl"
    with instance_path.open("w", encoding="utf-8") as output:
        for instance_id in sorted(associator.instances):
            state = associator.instances[instance_id]
            output.write(json.dumps({
                "instance_id": instance_id,
                "first_frame_id": state.first_frame_id,
                "last_frame_id": state.last_frame_id,
                "observation_count": len(state.observation_ids),
                "support_voxel_count": len(state.voxel_list),
                "observation_ids": state.observation_ids,
            }, ensure_ascii=False) + "\n")

    support_coordinates = []
    support_instance_ids = []
    for instance_id in sorted(associator.instances):
        voxels = associator.instances[instance_id].voxel_list
        support_coordinates.extend(voxels)
        support_instance_ids.extend([instance_id] * len(voxels))
    support_path = args.output_dir / "support_voxels_3cm.npz"
    np.savez_compressed(
        support_path,
        voxel_coordinates=np.asarray(support_coordinates, dtype=np.int32).reshape(-1, 3),
        instance_ids=np.asarray(support_instance_ids, dtype=np.int32),
        voxel_size_m=np.asarray([associator.voxel_size_m], dtype=np.float32),
    )
    ledger_entries = list(associator.observation_support.values())
    ledger_path = args.output_dir / "observation_support_3cm.npz"
    associator.save_checkpoint(ledger_path)
    counts_path = args.output_dir / "identity_voxel_counts_3cm.npz"
    if associator.identity_evidence is not None:
        associator.save_identity_counts(counts_path)
    revision_path = args.output_dir / "identity_revision_events.jsonl"
    revision_path.write_text("", encoding="utf-8")
    report = {
        "status": "PASS",
        "purpose": "causal_fixed_mask_no_repair_" + args.association_mode + "_association",
        "association_mode": args.association_mode,
        "map_version": associator.map_version,
        "next_instance_id": associator.next_instance_id,
        "identity_vote_unit": "one_per_frame_voxel_instance" if associator.identity_evidence is not None else None,
        "relative_weights_are_calibrated_probabilities": False,
        "candidate_support_trace_scope": "all_8_spatially_shortlisted_candidates" if associator.identity_evidence is not None else None,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "observation_catalog_path": str(args.observations),
        "observation_catalog_sha256": sha256_file(args.observations),
        "parent_child_lineage_sha256": sha256_file(args.parent_child_lineage) if args.parent_child_lineage else None,
        "frame_count": len(frame_ids),
        "decision_count": sum(decision_counts.values()),
        "instance_count": len(associator.instances),
        "decision_counts": dict(decision_counts),
        "support_voxel_count": len(support_coordinates),
        "observation_support_entry_count": len(ledger_entries),
        "observation_support_sha256": sha256_file(ledger_path),
        "association_parameters": {
            "voxel_size_m": associator.voxel_size_m,
            "max_points_per_observation": associator.max_points_per_observation,
            "max_visible_support_points": associator.max_visible_support_points,
            "min_geometric_coverage": associator.min_geometric_coverage,
            "min_total_score": associator.min_total_score,
            "min_visible_overlap": associator.min_visible_overlap,
            "ambiguity_margin": associator.ambiguity_margin,
            "depth_tolerance_m": associator.depth_tolerance_m,
            "one_existing_instance_per_frame": not args.allow_multiple_observations_per_instance_per_frame,
            "improved_association": args.improved_association,
            "valid_first_sampling": associator.valid_first_sampling,
            "association_mode": associator.association_mode,
            "relative_weight_temperature": associator.relative_weight_temperature,
            "null_candidate_score": associator.null_candidate_score,
        },
        "frame_summaries": frame_summaries,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "files": {
            "associations_jsonl": str(decision_path),
            "instances_jsonl": str(instance_path),
            "support_voxels_npz": str(support_path),
            "observation_support_npz": str(ledger_path),
            "identity_voxel_counts_npz": str(counts_path) if counts_path.exists() else None,
            "identity_revision_events_jsonl": str(revision_path),
            "candidate_support_trace_directory": str(trace_dir) if trace_dir.exists() else None,
        },
    }
    if report["decision_count"] != sum(len(all_observations[frame_id]) for frame_id in frame_ids):
        raise ValueError("Decision count differs from input observations")
    if report["decision_count"] != sum(len(state.observation_ids) for state in associator.instances.values()):
        raise ValueError("Instance support does not account for all observations")
    write_json(args.output_dir / "association_report.json", report)
    print(f"PASS: {len(frame_ids)} frames, {report['decision_count']} decisions, {report['instance_count']} instances, {report['total_seconds']:.1f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
