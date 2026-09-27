#!/usr/bin/env python3
"""Check one fixed Replica input contract without changing its source files."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(aggregate, frame_id, path):
    checksum = sha256_file(path)
    aggregate.update(f"{frame_id}\0{checksum}\n".encode("ascii"))
    return checksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    camera = config["camera"]
    selection = config["frame_selection"]
    frame_ids = list(range(selection["start"], selection["stop_exclusive"], selection["stride"]))
    scene_root = Path(source["scene_root"])
    mask_root = Path(source["mask_root"])
    trajectory = scene_root / source["trajectory"]
    metadata_path = mask_root / source["mask_metadata"]
    errors = []

    pose_lines = trajectory.read_text(encoding="utf-8").splitlines()
    if len(pose_lines) < selection["stop_exclusive"]:
        errors.append("trajectory has fewer lines than the selected frame range")

    metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata = {row["frame"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        errors.append("duplicate frame IDs in mask metadata")
    if sorted(metadata) != frame_ids:
        errors.append("mask metadata frame IDs differ from selected frame IDs")

    aggregates = {name: hashlib.sha256() for name in ("rgb", "depth", "mask")}
    samples = {}
    sample_ids = {frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]}
    expected_size = (camera["width"], camera["height"])

    for frame_id in frame_ids:
        paths = {
            "rgb": scene_root / source["rgb_pattern"].format(frame=frame_id),
            "depth": scene_root / source["depth_pattern"].format(frame=frame_id),
            "mask": mask_root / source["mask_pattern"].format(frame=frame_id),
        }
        if frame_id < len(pose_lines):
            pose = [float(item) for item in pose_lines[frame_id].split()]
            if len(pose) != 16 or not all(math.isfinite(value) for value in pose):
                errors.append(f"frame {frame_id}: invalid camera pose")
            elif any(abs(a - b) > 1e-5 for a, b in zip(pose[12:], (0, 0, 0, 1))):
                errors.append(f"frame {frame_id}: invalid pose last row")

        for kind, path in paths.items():
            if not path.is_file():
                errors.append(f"frame {frame_id}: missing {kind}: {path}")
                continue
            checksum = add_file(aggregates[kind], frame_id, path)
            row = metadata.get(frame_id, {})
            expected_hash = row.get("input_sha256" if kind == "rgb" else "mask_sha256")
            if kind in ("rgb", "mask") and checksum != expected_hash:
                errors.append(f"frame {frame_id}: {kind} checksum differs from cache metadata")

            with Image.open(path) as image:
                if image.size != expected_size:
                    errors.append(f"frame {frame_id}: {kind} image size {image.size} differs from {expected_size}")
                if kind == "mask" and image.mode != "L":
                    errors.append(f"frame {frame_id}: mask mode is {image.mode}, expected L")
                if kind == "depth" and image.mode not in ("I;16", "I;16B", "I"):
                    errors.append(f"frame {frame_id}: depth mode is {image.mode}, expected 16-bit")
                if frame_id in sample_ids:
                    entry = samples.setdefault(str(frame_id), {})
                    entry[kind] = {"mode": image.mode, "size": list(image.size)}
                    if kind == "mask":
                        colors = image.getcolors(maxcolors=256)
                        entry[kind]["labels"] = sorted(value for _, value in colors) if colors else None

    report = {
        "status": "PASS" if not errors else "FAIL",
        "contract_version": config["contract_version"],
        "scene": config["scene"],
        "selected_frame_count": len(frame_ids),
        "first_frame": frame_ids[0],
        "last_frame": frame_ids[-1],
        "frame_ids_sha256": hashlib.sha256("\n".join(map(str, frame_ids)).encode("ascii")).hexdigest(),
        "config_sha256": sha256_file(args.config),
        "trajectory_sha256": sha256_file(trajectory),
        "mask_metadata_sha256": sha256_file(metadata_path),
        "selected_file_digests": {name: digest.hexdigest() for name, digest in aggregates.items()},
        "sample_images": samples,
        "error_count": len(errors),
        "errors": errors[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(frame_ids)} frames, {len(errors)} errors; {args.output}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
