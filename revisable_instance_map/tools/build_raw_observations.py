#!/usr/bin/env python3
"""Create a lossless-indexed raw observation catalog for the fixed 400 frames."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource  # noqa: E402
from revisable_instance_map.observations import (  # noqa: E402
    extract_frame_observations, source_pixel_indices,
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ReplicaFrameSource(args.config)
    if len(source.frame_ids) != 400:
        raise ValueError("Expected frozen 400-frame input protocol")
    metadata_path = source.mask_root / source.config["source"]["mask_metadata"]
    metadata = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        frame_id = int(record["frame"])
        if frame_id in metadata:
            raise ValueError(f"Duplicate mask metadata for frame {frame_id}")
        metadata[frame_id] = record

    started = time.perf_counter()
    observation_path = args.output_dir / "observations.jsonl"
    temporary_path = observation_path.with_suffix(".jsonl.tmp")
    seen_ids = set()
    frame_summaries = []
    pixel_sizes = []
    total_projectable = 0
    total_mask_pixels = 0
    empty_geometry = 0
    source_pixel_checks = 0
    with temporary_path.open("w", encoding="utf-8") as output:
        for index, frame_id in enumerate(source.frame_ids, start=1):
            frame = source.load_frame(frame_id)
            mask_path = source.mask_root / source.config["source"]["mask_pattern"].format(frame=frame_id)
            digest = sha256_file(mask_path)
            expected = metadata.get(frame_id)
            if expected is None or digest != expected["mask_sha256"]:
                raise ValueError(f"Mask checksum or metadata mismatch at frame {frame_id}")
            observations = extract_frame_observations(
                frame, source.config["dataset"], source.config["scene"], digest,
                observation_namespace=source.config["mask"].get("observation_namespace", ""),
            )
            if len(observations) != int(expected["instances"]):
                raise ValueError(f"Mask instance count mismatch at frame {frame_id}")
            mask_pixels = int(np.count_nonzero(frame.mask_local))
            projectable = np.isfinite(frame.depth_m) & (frame.depth_m > 0) & (frame.depth_m < 10.0)
            projectable_mask_pixels = int(np.count_nonzero(projectable & (frame.mask_local != 0)))
            if sum(item.pixel_count for item in observations) != mask_pixels:
                raise ValueError(f"Mask pixels lost or duplicated at frame {frame_id}")
            if sum(item.projectable_pixel_count for item in observations) != projectable_mask_pixels:
                raise ValueError(f"Projectable mask pixels mismatch at frame {frame_id}")
            for observation in observations:
                if observation.observation_id in seen_ids:
                    raise ValueError(f"Duplicate observation ID: {observation.observation_id}")
                seen_ids.add(observation.observation_id)
                x0, y0, x1, y1 = observation.bbox_xyxy_exclusive
                if not (0 <= x0 < x1 <= frame.camera.width and 0 <= y0 < y1 <= frame.camera.height):
                    raise ValueError(f"Invalid bounding box: {observation.observation_id}")
                if observation.projectable_pixel_count == 0:
                    empty_geometry += 1
                pixel_sizes.append(observation.pixel_count)
                output.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")
            if frame_id in (0, 1000, 1995):
                for observation in observations:
                    source_pixel_indices(frame, observation)
                    source_pixel_checks += 1
            frame_summaries.append({
                "frame_id": frame_id,
                "source_mask_sha256": digest,
                "observation_count": len(observations),
                "nonzero_mask_pixels": mask_pixels,
                "projectable_mask_pixels": projectable_mask_pixels,
                "background_pixels": int(frame.mask_local.size - mask_pixels),
            })
            total_mask_pixels += mask_pixels
            total_projectable += projectable_mask_pixels
            if index % args.checkpoint_every == 0:
                progress = {
                    "frames_processed": index,
                    "last_frame_id": frame_id,
                    "observations": len(seen_ids),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write_json(args.output_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    temporary_path.replace(observation_path)

    report = {
        "status": "PASS",
        "purpose": "raw_per_frame_instance_proposals_without_association_or_repair",
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "input_mask_metadata_path": str(metadata_path),
        "frame_count": len(frame_summaries),
        "observation_count": len(seen_ids),
        "raw_mask_pixels": total_mask_pixels,
        "projectable_mask_pixels": total_projectable,
        "observations_without_projectable_depth": empty_geometry,
        "observations_with_fewer_than_100_pixels": int(np.count_nonzero(np.asarray(pixel_sizes) < 100)),
        "observation_pixel_count_min": int(min(pixel_sizes)),
        "observation_pixel_count_median": float(np.median(pixel_sizes)),
        "observation_pixel_count_max": int(max(pixel_sizes)),
        "observations_per_frame_min": int(min(item["observation_count"] for item in frame_summaries)),
        "observations_per_frame_median": float(np.median([item["observation_count"] for item in frame_summaries])),
        "observations_per_frame_max": int(max(item["observation_count"] for item in frame_summaries)),
        "source_pixel_recovery_checks": source_pixel_checks,
        "frame_summaries": frame_summaries,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "observations_jsonl_path": str(observation_path),
        "observations_jsonl_bytes": observation_path.stat().st_size,
    }
    write_json(args.output_dir / "observation_manifest.json", report)
    print(f"PASS: {report['frame_count']} frames, {report['observation_count']} observations, {report['total_seconds']:.1f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

