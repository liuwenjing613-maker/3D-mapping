#!/usr/bin/env python3
"""Fuse frozen CropFormer masks with OVI-MAP geometric masks, without GT."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.ovimap_refinement import fuse_ovimap_masks


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_by_frame(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frame = int(row["frame"])
        if frame in result:
            raise ValueError("Duplicate frame metadata: %d" % frame)
        result[frame] = row
    return result


def write_json(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--geometry-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--refined-config", type=Path, required=True)
    p.add_argument("--frame-count", type=int, default=400)
    a = p.parse_args()
    source = ReplicaFrameSource(a.config)
    selected = source.frame_ids[:a.frame_count]
    if len(selected) != a.frame_count or a.frame_count < 1:
        raise ValueError("Invalid frame count")
    raw_meta = rows_by_frame(source.mask_root / source.config["source"]["mask_metadata"])
    geometry_meta = rows_by_frame(a.geometry_dir / "frames.jsonl")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = a.output_dir / "frames.jsonl"
    region_path = a.output_dir / "segments.jsonl"
    if frame_path.exists() or region_path.exists():
        raise FileExistsError("Refined metadata already exists; choose a fresh output directory")
    began = time.perf_counter()
    totals = {
        "frames": 0, "raw_regions": 0, "geometry_regions": 0,
        "refined_regions": 0, "split_regions": 0, "majority_regions": 0,
        "raw_positive_pixels": 0, "refined_positive_pixels": 0,
        "raw_pixels_removed": 0, "raw_background_pixels_added": 0,
        "changed_positive_labels": 0, "projectable_refined_pixels": 0,
    }
    with frame_path.open("w", encoding="utf-8") as frames_out, region_path.open("w", encoding="utf-8") as regions_out:
        for index, frame_id in enumerate(selected, 1):
            frame = source.load_frame(frame_id)
            src = source.config["source"]
            raw_path = source.mask_root / src["mask_pattern"].format(frame=frame_id)
            rgb_path = source.scene_root / src["rgb_pattern"].format(frame=frame_id)
            depth_path = source.scene_root / src["depth_pattern"].format(frame=frame_id)
            geo_path = a.geometry_dir / ("%05d_mask.png" % frame_id)
            expected_raw = raw_meta.get(frame_id)
            expected_geo = geometry_meta.get(frame_id)
            if expected_raw is None or expected_geo is None:
                raise ValueError("Missing source metadata at frame %d" % frame_id)
            if sha256_file(raw_path) != expected_raw["mask_sha256"]:
                raise ValueError("CropFormer checksum mismatch at frame %d" % frame_id)
            if (sha256_file(rgb_path) != expected_geo["rgb_sha256"]
                    or sha256_file(depth_path) != expected_geo["depth_sha256"]
                    or sha256_file(geo_path) != expected_geo["geometric_sha256"]):
                raise ValueError("Geometric source checksum mismatch at frame %d" % frame_id)
            with Image.open(geo_path) as image:
                geometry = np.asarray(image)
            if geometry.shape != frame.mask_local.shape or geometry.dtype != np.uint8:
                raise ValueError("Bad geometric label image at frame %d" % frame_id)
            refined, regions, background, stats = fuse_ovimap_masks(frame.mask_local, geometry)
            if len(regions) > 65535:
                raise OverflowError("Too many refined labels")
            target = a.output_dir / ("frame%06d.png" % frame_id)
            if target.exists():
                raise FileExistsError(target)
            temporary = a.output_dir / ("frame%06d.tmp.png" % frame_id)
            Image.fromarray(refined).save(temporary, format="PNG")
            temporary.replace(target)
            mask_sha = sha256_file(target)
            raw_positive = frame.mask_local != 0
            refined_positive = refined != 0
            same = np.zeros(refined.shape, dtype=bool)
            valid_depth = np.isfinite(frame.depth_m) & (frame.depth_m > 0) & (frame.depth_m < 10.0)
            for region in regions:
                support = refined == region.local_id
                if int(np.count_nonzero(support)) != region.pixel_count:
                    raise AssertionError("Refined region support mismatch")
                matching = support & (frame.mask_local == region.cropformer_id)
                same |= matching
                row = asdict(region)
                row.update({
                    "frame_id": frame_id,
                    "observation_id": "%s/%s/ovimap-refined-v1/f%06d/m%03d" % (
                        source.config["dataset"], source.config["scene"], frame_id, region.local_id),
                    "projectable_pixel_count": int(np.count_nonzero(support & valid_depth)),
                    "same_cropformer_pixels": int(np.count_nonzero(matching)),
                    "other_cropformer_pixels": int(np.count_nonzero(support & raw_positive & ~matching)),
                    "cropformer_background_pixels": int(np.count_nonzero(support & ~raw_positive)),
                    "source_raw_mask_sha256": expected_raw["mask_sha256"],
                    "source_geometric_mask_sha256": expected_geo["geometric_sha256"],
                    "refined_mask_sha256": mask_sha,
                })
                regions_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            removed = int(np.count_nonzero(raw_positive & ~refined_positive))
            added = int(np.count_nonzero(~raw_positive & refined_positive))
            projectable = int(np.count_nonzero(refined_positive & valid_depth))
            if sum(x.pixel_count for x in regions) != int(np.count_nonzero(refined)):
                raise AssertionError("Refined pixel count mismatch")
            frame_row = {
                "frame": frame_id,
                "mask_sha256": mask_sha,
                "instances": len(regions),
                "raw_mask_sha256": expected_raw["mask_sha256"],
                "geometric_mask_sha256": expected_geo["geometric_sha256"],
                "raw_instances": int(expected_raw["instances"]),
                "geometry_segments": int(expected_geo["segments"]),
                "raw_positive_pixels": int(np.count_nonzero(raw_positive)),
                "refined_positive_pixels": int(np.count_nonzero(refined_positive)),
                "raw_pixels_removed": removed,
                "raw_background_pixels_added": added,
                "projectable_refined_pixels": projectable,
                "changed_positive_labels": int(np.count_nonzero(refined_positive & ~same)),
                "background_regions": len(background),
                "stats": stats,
            }
            frames_out.write(json.dumps(frame_row, ensure_ascii=False) + "\n")
            frames_out.flush()
            regions_out.flush()
            totals["frames"] += 1
            totals["raw_regions"] += frame_row["raw_instances"]
            totals["geometry_regions"] += frame_row["geometry_segments"]
            totals["refined_regions"] += frame_row["instances"]
            totals["split_regions"] += stats["split_regions"]
            totals["majority_regions"] += stats["majority_regions"]
            for key in ("raw_positive_pixels", "refined_positive_pixels", "raw_pixels_removed", "raw_background_pixels_added", "changed_positive_labels", "projectable_refined_pixels"):
                totals[key] += frame_row[key]
            if index % 25 == 0 or index == 1:
                print(json.dumps({"frames": index, "last": frame_id, "refined_regions": totals["refined_regions"], "elapsed_s": round(time.perf_counter() - began, 1)}), flush=True)
    if a.frame_count == len(source.frame_ids):
        refined_config = json.loads(a.config.read_text(encoding="utf-8"))
        refined_config["purpose"] = "ovimap_depth_refined_no_repair_input"
        refined_config["source"]["mask_root"] = str(a.output_dir)
        refined_config["source"]["mask_pattern"] = "frame{frame:06d}.png"
        refined_config["source"]["mask_metadata"] = "frames.jsonl"
        refined_config["mask"] = {
            "encoding": "uint16_instance_id_image", "background_id": 0,
            "observation_namespace": "ovimap-refined-v1",
            "provenance": "OVI-MAP depth_segmentation_py and official MaskFusion thresholds; see segments.jsonl",
        }
        write_json(a.refined_config, refined_config)
    report = {
        "status": "PASS", "method": "official_OVI-MAP_depth_segmentation_and_MaskFusion",
        "official_revision": "f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424",
        "source_config_sha256": sha256_file(a.config),
        "raw_metadata_sha256": sha256_file(source.mask_root / source.config["source"]["mask_metadata"]),
        "geometry_metadata_sha256": sha256_file(a.geometry_dir / "frames.jsonl"),
        "refined_frames_sha256": sha256_file(frame_path),
        "refined_segments_sha256": sha256_file(region_path),
        "refined_config_sha256": sha256_file(a.refined_config) if a.refined_config.exists() else None,
        "observation_namespace": "ovimap-refined-v1",
        "totals": totals,
        "seconds": time.perf_counter() - began,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "output_dir": str(a.output_dir),
    }
    write_json(a.output_dir / "refinement_report.json", report)
    print("PASS fusion: %d frames, %d refined regions, %.1f s" % (totals["frames"], totals["refined_regions"], report["seconds"]), flush=True)


if __name__ == "__main__":
    main()