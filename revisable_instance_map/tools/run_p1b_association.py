#!/usr/bin/env python3
"""Independent causal B0/B1/B2 entry; observations and GT never interchange."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.deferred_association import DeferredAssociator
from revisable_instance_map.assignment_ledger import ACCEPTED, PENDING
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.observations import RawInstanceObservation
from run_baseline_association import sha256_file, write_json


def write_rows(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "observations", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("B0", "B1", "B2"), default="B2")
    parser.add_argument("--frame-count", type=int, default=400)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.frame_count < 1:
        raise ValueError("Invalid frame count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = ReplicaFrameSource(args.config)
    frame_ids = source.frame_ids[:args.frame_count]
    if len(frame_ids) != args.frame_count:
        raise ValueError("Invalid requested input prefix")
    catalog = defaultdict(list)
    for line in args.observations.read_text().splitlines():
        row = json.loads(line)
        catalog[row["frame_id"]].append(RawInstanceObservation(**row))
    parameters = json.loads(args.parameters.read_text()) if args.parameters else None
    model = DeferredAssociator.from_checkpoint(args.resume, source.load_frame) if args.resume else DeferredAssociator(source.load_frame, args.mode, parameters)
    if model.mode != args.mode or (parameters is not None and model.parameters != {**model.parameters, **parameters}):
        raise ValueError("Resume mode/parameters differ")
    if args.resume:
        if model.engine.last_processed_frame_id not in frame_ids:
            raise ValueError("Resume checkpoint is beyond or outside the requested prefix")
        expected = {o.observation_id:o for f in frame_ids if f<=model.engine.last_processed_frame_id for o in catalog[f]}
        if set(expected)!=set(model.ledger.raw_support_store):
            raise ValueError("Resume checkpoint source catalog differs")
        for key,observation in expected.items():
            if json.loads(json.dumps(asdict(observation)))!=json.loads(json.dumps(asdict(model.ledger.raw_support_store[key].observation))):
                raise ValueError("Resume raw observation metadata changed")
    started = time.perf_counter()
    summaries = []
    for step, frame_id in enumerate(frame_ids, 1):
        if frame_id <= model.engine.last_processed_frame_id:
            continue
        frame = source.load_frame(frame_id)
        observations = tuple(catalog[frame_id])
        mask_path = source.mask_root / source.config["source"]["mask_pattern"].format(frame=frame_id)
        digest = sha256_file(mask_path)
        if any(o.source_mask_sha256 != digest for o in observations):
            raise ValueError("Source mask checksum changed")
        decisions = model.process_frame(frame, observations)
        counts = Counter(a.status for a in model.ledger.assignment_store.values())
        summaries.append({"frame_id": frame_id, "processing_step": step, "map_version": model.map_version,
            "arrival_observations": len(decisions), "assignment_counts": dict(counts),
            "formal_instances": len(model.engine.instances), "packets": len(model.packets),
            "review_queue": len(model.review_queue), "reviews": len(model.review_log),
            "seconds": time.perf_counter()-started})
        if step == 50:
            write_rows(args.output_dir / "assignments_at_step050.jsonl", model.assignment_rows())
            np.savez_compressed(args.output_dir / "identity_counts_at_step050.npz", **model.engine.identity_evidence.export_arrays())
        if step % 25 == 0 or step == len(frame_ids):
            write_json(args.output_dir / "progress.json", summaries[-1])
            print(json.dumps(summaries[-1]), flush=True)
    model.validate()
    write_rows(args.output_dir / "decisions_at_arrival.jsonl", model.arrivals)
    write_rows(args.output_dir / "associations.jsonl", model.assignment_rows())
    write_rows(args.output_dir / "decision_events.jsonl", model.ledger.events)
    write_rows(args.output_dir / "review_log.jsonl", model.review_log)
    write_rows(args.output_dir / "repair_tickets.jsonl", model.repair_tickets)
    model.save_checkpoint(args.output_dir / "identity_checkpoint.npz")
    model.engine.save_checkpoint(args.output_dir / "accepted_identity_checkpoint.npz")
    model.engine.save_identity_counts(args.output_dir / "identity_voxel_counts_3cm.npz")
    delays = [a.last_decision_frame_id - model.ledger.raw_support_store[key].source_frame_id
              for key, a in model.ledger.assignment_store.items() if a.status == ACCEPTED]
    source_steps = {f: i for i, f in enumerate(frame_ids)}
    step_delays = [source_steps[a.last_decision_frame_id]-source_steps[model.ledger.raw_support_store[key].source_frame_id]
                   for key, a in model.ledger.assignment_store.items() if a.status == ACCEPTED]
    counts = Counter(a.status for a in model.ledger.assignment_store.values())
    total = len(model.ledger.assignment_store)
    delayed = [d for d in step_delays if d > 0]
    small = [k for k, r in model.ledger.raw_support_store.items() if r.observation.pixel_count <= 1024]
    report = {"status": "PASS", "mode": model.mode, "parameters": model.parameters,
        "parameters_hash": model.parameters_hash, "association_parameters": model.engine.checkpoint_parameters(),
        "config_path": str(args.config), "config_sha256": sha256_file(args.config),
        "observation_catalog_path": str(args.observations), "observation_catalog_sha256": sha256_file(args.observations),
        "map_version": model.map_version, "frame_count": args.frame_count, "observation_count": total,
        "assignment_counts": dict(counts), "formal_instances": len(model.engine.instances),
        "packet_count": len(model.packets), "pending_fraction": sum(counts[k] for k in PENDING)/max(total, 1),
        "small_observation_definition": "source_pixel_count<=1024; not GT physical object size",
        "small_observation_count": len(small),
        "small_observation_pending_fraction": sum(model.ledger.assignment_store[k].status in PENDING for k in small)/max(len(small), 1),
        "accepted_fraction": counts[ACCEPTED]/max(total, 1), "delayed_accept_count": len(delayed),
        "accept_delay_source_frames_median_p95": np.percentile(delays, [50, 95]).tolist() if delays else None,
        "accept_delay_processing_steps_median_p95": np.percentile(step_delays, [50, 95]).tolist() if step_delays else None,
        "delayed_only_processing_steps_median_p95": np.percentile(delayed, [50, 95]).tolist() if delayed else None,
        "review_observation_count": len(model.review_log), "repair_ticket_count": len(model.repair_tickets),
        "packet_lifecycle_counts": dict(Counter(p.lifecycle for p in model.packets.values())),
        "next_instance_id": model.engine.next_instance_id, "next_packet_id": model.next_packet_id,
        "full_source_coverage_verified": True, "active_cache_rebuilt_and_verified": True,
        "ground_truth_used": False, "future_geometry_used": False,
        "association_decisions_sha256": sha256_file(args.output_dir / "associations.jsonl"),
        "identity_checkpoint_sha256": sha256_file(args.output_dir / "identity_checkpoint.npz"),
        "frame_summaries": summaries, "total_seconds": time.perf_counter()-started,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(args.output_dir / "association_report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "mode", "assignment_counts", "pending_fraction", "total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
