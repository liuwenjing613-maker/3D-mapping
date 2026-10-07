#!/usr/bin/env python3
"""Keep OVI refined support while restoring each pixel's raw CropFormer ID."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-config", type=Path, required=True)
    parser.add_argument("--refined-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    raw_cfg = json.loads(args.raw_config.read_text(encoding="utf-8"))
    refined_cfg = json.loads(args.refined_config.read_text(encoding="utf-8"))
    if (raw_cfg["scene"] != refined_cfg["scene"] or
            raw_cfg["frame_selection"] != refined_cfg["frame_selection"]):
        raise ValueError("Raw and refined configs describe different protocols")
    selection = raw_cfg["frame_selection"]
    frame_ids = range(selection["start"], selection["stop_exclusive"],
                      selection["stride"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "frames.jsonl"
    report_path = args.output_dir / "regroup_report.json"
    if manifest.exists() or report_path.exists() or args.output_config.exists():
        raise FileExistsError("Regrouped output already exists; choose a fresh directory")
    raw_root = Path(raw_cfg["source"]["mask_root"])
    refined_root = Path(refined_cfg["source"]["mask_root"])
    records = []
    totals = {"retained_pixels": 0, "removed_pixels": 0, "instances": 0}
    for frame_id in frame_ids:
        raw_path = raw_root / raw_cfg["source"]["mask_pattern"].format(frame=frame_id)
        refined_path = refined_root / refined_cfg["source"]["mask_pattern"].format(frame=frame_id)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        refined = cv2.imread(str(refined_path), cv2.IMREAD_UNCHANGED)
        if raw is None or refined is None or raw.shape != refined.shape:
            raise ValueError(f"Mask error in frame {frame_id}")
        kept = np.where(refined > 0, raw, 0).astype(raw.dtype)
        if np.count_nonzero((kept > 0) & (raw == 0)):
            raise AssertionError("Regrouping introduced pixels absent from raw mask")
        output = args.output_dir / f"frame{frame_id:06d}.png"
        temporary = args.output_dir / f"frame{frame_id:06d}.tmp.png"
        if output.exists() or not cv2.imwrite(str(temporary), kept):
            raise FileExistsError(output)
        temporary.replace(output)
        instance_count = int(np.count_nonzero(np.unique(kept)))
        retained = int(np.count_nonzero(kept))
        removed = int(np.count_nonzero((raw > 0) & (kept == 0)))
        totals["retained_pixels"] += retained
        totals["removed_pixels"] += removed
        totals["instances"] += instance_count
        records.append({
            "frame": frame_id, "mask_sha256": sha256_file(output),
            "instances": instance_count, "raw_mask_sha256": sha256_file(raw_path),
            "refined_mask_sha256": sha256_file(refined_path),
        })
    manifest_tmp = manifest.with_suffix(".jsonl.tmp")
    manifest_tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in records), encoding="utf-8")
    manifest_tmp.replace(manifest)
    output_cfg = json.loads(json.dumps(raw_cfg))
    output_cfg["purpose"] = "p0_depth_refined_regrouped_by_raw_source"
    output_cfg["source"]["mask_root"] = str(args.output_dir)
    output_cfg["source"]["mask_pattern"] = "frame{frame:06d}.png"
    output_cfg["source"]["mask_metadata"] = "frames.jsonl"
    output_cfg["mask"]["observation_namespace"] = "ovimap-parent-regrouped-v1"
    output_cfg["mask"]["provenance"] = (
        "raw CropFormer ID intersected with positive OVI refined support")
    write_json(args.output_config, output_cfg)
    report = {
        "status": "PASS", "scene_id": raw_cfg["scene"],
        "frames": len(records), "ground_truth_used": False,
        "raw_config_sha256": sha256_file(args.raw_config),
        "refined_config_sha256": sha256_file(args.refined_config),
        "output_config_sha256": sha256_file(args.output_config),
        "manifest_sha256": sha256_file(manifest), **totals,
    }
    write_json(report_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
