#!/usr/bin/env python3
"""Reapply revised identities to immutable P0 per-frame region support.

Uses the same frame_instance_keys/reduce_surface_votes as P0. Cached support
comes from final TSDF geometry and is for offline publication only; the online
associator never reads it. No inferred/diffused label becomes an observed vote.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.surface_evidence import (
    STATE_NAMES, CONFIRMED, TENTATIVE, frame_instance_keys, reduce_surface_votes,
)
from run_baseline_association import sha256_file, write_json


def rebuild_surface(source_dir, observations, associations, checkpoint, config,
                    output_dir, frame_count=400):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    started = time.perf_counter()
    original = json.loads((source_dir / "materialization_report.json").read_text())
    if original["observation_catalog_sha256"] != sha256_file(observations):
        raise ValueError("Cached region support belongs to a different observation catalog")
    if original["config_sha256"] != sha256_file(config):
        raise ValueError("Cached region support belongs to a different input configuration")
    manifest_path = source_dir / "frame_region_support_manifest.jsonl"
    if sha256_file(manifest_path) != original["frame_region_support_manifest_sha256"]:
        raise ValueError("Cached region support manifest was modified")
    evidence_path = source_dir / "surface_evidence.npz"
    if sha256_file(evidence_path) != original["surface_evidence_sha256"]:
        raise ValueError("Cached surface geometry/evidence was modified")
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    selected = manifest[:frame_count]
    if len(selected) != frame_count or [item["frame_id"] for item in selected] != sorted({item["frame_id"] for item in selected}):
        raise ValueError("Invalid cached frame prefix")
    source_config = json.loads(Path(config).read_text())
    selection = source_config["frame_selection"]
    expected_frames = list(range(selection["start"], selection["stop_exclusive"], selection["stride"]))[:frame_count]
    if [item["frame_id"] for item in selected] != expected_frames:
        raise ValueError("Region support frame list differs from the fixed input protocol")
    state = OnlineVoxelAssociator.from_checkpoint(checkpoint)
    rows = [json.loads(line) for line in Path(associations).read_text().splitlines()]
    assigned = {}
    for row in rows:
        key = row["observation_id"]
        if key in assigned:
            raise ValueError("Duplicate association row")
        support = state.observation_support.get(key)
        if support is None or support.instance_id != row["instance_id"] or support.frame_id != row["frame_id"]:
            raise ValueError("Association table and identity checkpoint differ")
        if row.get("assignment_version", 0) != support.assignment_version:
            raise ValueError("Association table and checkpoint assignment versions differ")
        assigned[key] = support.instance_id
    if set(assigned) != set(state.observation_support):
        raise ValueError("Association table does not cover its checkpoint exactly")
    by_frame = defaultdict(list)
    selected_ids = set(expected_frames)
    for line in Path(observations).read_text().splitlines():
        observation = json.loads(line)
        if observation["frame_id"] in selected_ids:
            by_frame[observation["frame_id"]].append(observation)
    selected_observations = {obs["observation_id"] for records in by_frame.values() for obs in records}
    if selected_observations != set(assigned):
        raise ValueError("Checkpoint does not cover precisely the requested frame prefix")
    instance_base = state.next_instance_id
    with np.load(evidence_path, allow_pickle=False) as data:
        xyz, rgb = data["xyz_m"].copy(), data["rgb"].copy()
    keys_by_frame, frame_records = [], []
    for record in selected:
        path = Path(record["support_file"])
        if sha256_file(path) != record["support_sha256"]:
            raise ValueError(f"Cached region support modified: {path}")
        with np.load(path, allow_pickle=False) as data:
            frame_id = int(data["frame_id"][0])
            local_ids = data["mask_local_id"]
            points = data["surface_point_index"]
            digest = str(data["source_mask_sha256"][0])
            if (frame_id != record["frame_id"]
                or int(data["pixel_stride"][0]) != original["pixel_stride"]
                or float(data["max_surface_distance_m"][0]) != original["max_surface_distance_m"]):
                raise ValueError("Cached region projection settings differ")
            if len(points) != len(local_ids) or np.any(points < 0) or np.any(points >= len(xyz)):
                raise ValueError("Invalid cached surface indices")
            lookup = np.full(max([int(local_ids.max(initial=0))] + [o["mask_local_id"] for o in by_frame[frame_id]]) + 1, -1, np.int32)
            for observation in by_frame[frame_id]:
                support = state.observation_support[observation["observation_id"]]
                if observation["source_mask_sha256"] != digest or support.source_mask_sha256 != digest:
                    raise ValueError("Cached support and immutable observation mask hashes differ")
                lookup[observation["mask_local_id"]] = assigned[observation["observation_id"]]
            if np.any(local_ids <= 0) or np.any(lookup[local_ids] <= 0):
                raise ValueError("Unaccounted cached mask region")
            keys = frame_instance_keys(points, local_ids, lookup, instance_base)
        keys_by_frame.append(keys)
        frame_records.append(dict(record, frame_instance_votes=len(keys), map_version=state.map_version))
    all_keys = np.concatenate(keys_by_frame)
    unique_keys, vote_counts = np.unique(all_keys, return_counts=True)
    evidence = reduce_surface_votes(unique_keys, vote_counts.astype(np.int32), len(xyz), instance_base,
                                   original["min_confirmed_votes"], original["min_confirmed_ratio"])
    output_dir.mkdir(parents=True, exist_ok=True)
    publication_manifest = output_dir / "frame_region_support_manifest.jsonl"
    publication_manifest.write_text("".join(json.dumps(row) + "\n" for row in frame_records))
    association_hash = sha256_file(associations)
    version = dict(map_version=np.asarray([state.map_version], np.int64),
                   association_decisions_sha256=np.asarray([association_hash]))
    np.savez_compressed(output_dir / "surface_evidence.npz", xyz_m=xyz, rgb=rgb, **evidence, **version)
    np.savez_compressed(output_dir / "surface_instance_frame_votes.npz",
        surface_point_index=(unique_keys // instance_base).astype(np.int32),
        instance_id=(unique_keys % instance_base).astype(np.int32), frame_votes=vote_counts.astype(np.int32), **version)
    states, top1 = evidence["state"], evidence["top1_instance_id"]
    confirmed = np.where(states == CONFIRMED, top1, -1).astype(np.int32)
    tentative = np.where((states == CONFIRMED) | (states == TENTATIVE), top1, -1).astype(np.int32)
    np.savez_compressed(output_dir / "instance_surface.npz", xyz_m=xyz, rgb=rgb, instance_id=confirmed, **version)
    np.savez_compressed(output_dir / "instance_surface_tentative_included.npz", xyz_m=xyz, rgb=rgb, instance_id=tentative, **version)
    report = {
        "status": "PASS", "purpose": "P0_republication_from_immutable_frame_regions",
        "ground_truth_used": False, "map_version": state.map_version,
        "scene_id": original["scene_id"], "frame_count": frame_count,
        "observation_count": len(assigned), "tsdf_surface_points": len(xyz),
        "config_sha256": sha256_file(config), "observation_catalog_sha256": sha256_file(observations),
        "association_decisions_sha256": association_hash,
        "identity_checkpoint_sha256": sha256_file(checkpoint),
        "raw_region_source_dir": str(source_dir),
        "raw_region_manifest_sha256": original["frame_region_support_manifest_sha256"],
        "geometry_sha256": original["tsdf_surface_sha256"],
        "tsdf_surface_sha256": original["tsdf_surface_sha256"],
        "frame_region_support_manifest_sha256": sha256_file(publication_manifest),
        "final_geometry_used_for_offline_publication_only": True,
        "pixel_stride": original["pixel_stride"], "max_surface_distance_m": original["max_surface_distance_m"],
        "min_confirmed_votes": original["min_confirmed_votes"], "min_confirmed_ratio": original["min_confirmed_ratio"],
        "state_counts": {name: int(np.count_nonzero(states == i)) for i, name in enumerate(STATE_NAMES)},
        "labeled_surface_fraction": float(np.mean(confirmed > 0)),
        "frame_instance_surface_votes": len(all_keys), "frame_records": frame_records,
        "instance_surface_sha256": sha256_file(output_dir / "instance_surface.npz"),
        "surface_evidence_sha256": sha256_file(output_dir / "surface_evidence.npz"),
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "materialization_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "observations", "associations", "identity-checkpoint", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=400)
    args = parser.parse_args()
    report = rebuild_surface(args.source_dir, args.observations, args.associations,
        args.identity_checkpoint, args.config, args.output_dir, args.frame_count)
    print(json.dumps({key: report[key] for key in ("status", "map_version", "state_counts", "total_seconds")}))


if __name__ == "__main__":
    main()
