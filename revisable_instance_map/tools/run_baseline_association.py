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
    associator = OnlineVoxelAssociator(allow_multiple_observations_per_instance_per_frame=args.allow_multiple_observations_per_instance_per_frame)
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
            if len(decisions) != len(observations):
                raise ValueError("An observation was not assigned")
            for decision in decisions:
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
    report = {
        "status": "PASS",
        "purpose": "causal_fixed_mask_no_repair_baseline_association",
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "observation_catalog_path": str(args.observations),
        "observation_catalog_sha256": sha256_file(args.observations),
        "frame_count": len(frame_ids),
        "decision_count": sum(decision_counts.values()),
        "instance_count": len(associator.instances),
        "decision_counts": dict(decision_counts),
        "support_voxel_count": len(support_coordinates),
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
        },
        "frame_summaries": frame_summaries,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "files": {
            "associations_jsonl": str(decision_path),
            "instances_jsonl": str(instance_path),
            "support_voxels_npz": str(support_path),
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

