#!/usr/bin/env python3
"""Fixed final geometry publication; never feeds geometry into online decisions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.assignment_ledger import ACCEPTED
from revisable_instance_map.deferred_association import DeferredAssociator
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.surface_evidence import frame_instance_keys, reduce_surface_votes, CONFIRMED, STATE_NAMES
from revisable_instance_map.pending_publication import protect_publication, REASON_NAMES
from run_baseline_association import sha256_file, write_json


def rebuild_surface(source_dir, observations, associations, checkpoint, config, output_dir, frame_count=400):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    started = time.perf_counter()
    original = json.loads((source_dir / "materialization_report.json").read_text())
    source = ReplicaFrameSource(config)
    frames = source.frame_ids[:frame_count]
    if len(frames) != frame_count:
        raise ValueError("Invalid prefix")
    for path, key in ((observations, "observation_catalog_sha256"), (config, "config_sha256"),
        (source_dir / "frame_region_support_manifest.jsonl", "frame_region_support_manifest_sha256"),
        (source_dir / "surface_evidence.npz", "surface_evidence_sha256")):
        if sha256_file(Path(path)) != original[key]:
            raise ValueError("Immutable P0 source differs: " + key)
    manifest = [json.loads(line) for line in (source_dir / "frame_region_support_manifest.jsonl").read_text().splitlines()][:frame_count]
    if [r["frame_id"] for r in manifest] != list(frames):
        raise ValueError("Surface source prefix differs")
    model = DeferredAssociator.from_checkpoint(checkpoint, source.load_frame)
    rows = [json.loads(line) for line in Path(associations).read_text().splitlines()]
    expected_rows = json.loads(json.dumps(model.assignment_rows()))
    if rows != expected_rows:
        raise ValueError("Association table and checkpoint differ")
    by_frame = defaultdict(list)
    for line in Path(observations).read_text().splitlines():
        row = json.loads(line)
        if row["frame_id"] in frames:
            by_frame[row["frame_id"]].append(row)
    if {r["observation_id"] for rr in by_frame.values() for r in rr} != set(model.ledger.raw_support_store):
        raise ValueError("Source catalog and checkpoint coverage differ")
    with np.load(source_dir / "surface_evidence.npz", allow_pickle=False) as data:
        xyz, rgb = data["xyz_m"].copy(), data["rgb"].copy()
    base = max(model.engine.next_instance_id, 2)
    unresolved = np.zeros(len(xyz), np.int32)
    frame_keys, records = [], []
    for item in manifest:
        path = Path(item["support_file"])
        if sha256_file(path) != item["support_sha256"]:
            raise ValueError("Raw region source changed")
        with np.load(path, allow_pickle=False) as data:
            local, points = data["mask_local_id"], data["surface_point_index"]
            digest = str(data["source_mask_sha256"][0])
            if int(data["frame_id"][0]) != item["frame_id"] or len(local) != len(points):
                raise ValueError("Invalid raw region frame")
            if int(data["pixel_stride"][0]) != original["pixel_stride"] or float(data["max_surface_distance_m"][0]) != original["max_surface_distance_m"]:
                raise ValueError("Raw region projection settings differ")
            if np.any(points < 0) or np.any(points >= len(xyz)) or np.any(local <= 0):
                raise ValueError("Invalid region indices")
            lookup = np.full(max([int(local.max(initial=0))] + [r["mask_local_id"] for r in by_frame[item["frame_id"]]])+1, -1, np.int32)
            known = set()
            for row in by_frame[item["frame_id"]]:
                key, mask = row["observation_id"], row["mask_local_id"]
                raw, assignment = model.ledger.raw_support_store[key], model.ledger.assignment_store[key]
                if row["source_mask_sha256"] != digest or raw.observation.source_mask_sha256 != digest:
                    raise ValueError("Raw mask hash differs")
                known.add(mask)
                if assignment.status == ACCEPTED:
                    lookup[mask] = assignment.persistent_instance_id
                else:
                    # Each distinct raw observation contributes at most one unresolved upper count per point.
                    unique = np.unique(points[local == mask])
                    unresolved[unique] += 1
            if not set(int(x) for x in np.unique(local)).issubset(known):
                raise ValueError("An unresolved region was silently dropped")
            accepted = lookup[local] > 0
            keys = frame_instance_keys(points[accepted], local[accepted], lookup, base)
        frame_keys.append(keys)
        records.append({**item, "frame_instance_votes": len(keys), "map_version": model.map_version})
    all_keys = np.concatenate(frame_keys)
    keys, votes = np.unique(all_keys, return_counts=True)
    evidence = reduce_surface_votes(keys, votes.astype(np.int32), len(xyz), base,
                                   original["min_confirmed_votes"], original["min_confirmed_ratio"])
    publication = protect_publication(evidence, unresolved, original["min_confirmed_votes"], original["min_confirmed_ratio"])
    output_dir.mkdir(parents=True, exist_ok=False)
    version = {"map_version": np.asarray([model.map_version], np.int64),
               "association_decisions_sha256": np.asarray([sha256_file(Path(associations))])}
    np.savez_compressed(output_dir / "surface_evidence.npz", xyz_m=xyz, rgb=rgb, **evidence,
        **{k: v for k, v in publication.items() if k != "instance_id"}, **version)
    np.savez_compressed(output_dir / "instance_surface.npz", xyz_m=xyz, rgb=rgb, **publication, **version)
    accepted_only = np.where(evidence["state"] == CONFIRMED, evidence["top1_instance_id"], -1).astype(np.int32)
    np.savez_compressed(output_dir / "p0_accepted_only.npz", xyz_m=xyz, rgb=rgb, instance_id=accepted_only, **version)
    np.savez_compressed(output_dir / "surface_instance_frame_votes.npz", surface_point_index=(keys//base).astype(np.int32),
                        instance_id=(keys%base).astype(np.int32), frame_votes=votes.astype(np.int32), **version)
    (output_dir / "frame_region_support_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    report = {"status": "PASS", "mode": model.mode, "map_version": model.map_version,
        "scene_id": original["scene_id"], "frame_count": frame_count, "observation_count": len(rows),
        "state_counts": {name: int(np.count_nonzero(evidence["state"] == i)) for i, name in enumerate(STATE_NAMES)},
        "publication_reason_counts": {name: int(np.count_nonzero(publication["publication_reason"] == i)) for i, name in enumerate(REASON_NAMES)},
        "labeled_surface_fraction": float(np.mean(publication["instance_id"] > 0)),
        "accepted_only_labeled_surface_fraction": float(np.mean(accepted_only > 0)),
        "pending_surface_fraction": float(np.mean(unresolved > 0)),
        "protected_surface_fraction": float(np.mean(publication["diffusion_protected_mask"])),
        "surface_point_count": len(xyz), "unresolved_observation_surface_pairs": int(unresolved.sum()),
        "ground_truth_used": False, "final_geometry_used_for_offline_publication_only": True,
        "online_prefix_quality_claimed": False, "raw_region_source_dir": str(source_dir),
        "config_sha256": sha256_file(Path(config)), "observation_catalog_sha256": sha256_file(Path(observations)),
        "association_decisions_sha256": sha256_file(Path(associations)), "identity_checkpoint_sha256": sha256_file(Path(checkpoint)),
        "raw_region_manifest_sha256": original["frame_region_support_manifest_sha256"],
        "surface_evidence_sha256": sha256_file(output_dir / "surface_evidence.npz"),
        "instance_surface_sha256": sha256_file(output_dir / "instance_surface.npz"),
        "total_seconds": time.perf_counter()-started}
    write_json(output_dir / "materialization_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-dir", "observations", "associations", "identity-checkpoint", "config", "output-dir"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(rebuild_surface(args.source_dir, args.observations, args.associations,
        args.identity_checkpoint, args.config, args.output_dir, args.frame_count)))


if __name__ == "__main__":
    main()
