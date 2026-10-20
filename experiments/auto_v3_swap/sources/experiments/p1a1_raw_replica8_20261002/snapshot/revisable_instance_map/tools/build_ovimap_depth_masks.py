#!/usr/bin/env python3
"""Regenerate OVI-MAP geometric labels with its compiled depth_segmentation_py.

Run inside the existing OVI native container; this file is piped to python3.8.
The explicit os._exit avoids a destructor crash in the upstream pybind module.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import depth_segmentation_py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-root", type=Path, default=Path("/datasets/Replica/room0"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-cache", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=2000)
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    metadata_path = a.output / "frames.jsonl"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    K = np.array([[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    segmenter = depth_segmentation_py.DepthSegmentation_py(680, 1200, cv2.CV_32FC1, K)
    began = time.perf_counter()
    with metadata_path.open("w", encoding="utf-8") as meta:
        for index, frame_id in enumerate(range(a.start, a.stop, a.stride), 1):
            rgb_path = a.scene_root / "results" / ("frame%06d.jpg" % frame_id)
            depth_path = a.scene_root / "results" / ("depth%06d.png" % frame_id)
            image = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if image is None or raw_depth is None or image.shape != (680, 1200, 3) or raw_depth.shape != (680, 1200) or raw_depth.dtype != np.uint16:
                raise ValueError("Invalid RGB-D input for frame %d" % frame_id)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            depth = raw_depth.astype(np.float32) / 6553.5
            started = time.perf_counter()
            segmenter.depthSegment(depth, rgb)
            layers = segmenter.get_segmentMasks()
            if layers.ndim != 3 or layers.shape[1:] != depth.shape or layers.shape[0] > 255:
                raise ValueError("Invalid geometric segment stack for frame %d: %s" % (frame_id, layers.shape))
            labels = np.zeros(depth.shape, dtype=np.uint8)
            for segment_index, layer in enumerate(layers):
                labels[layer.astype(bool)] = segment_index + 1
            reference_equal = None
            if a.reference_cache:
                reference_path = a.reference_cache / ("%05d_mask.png" % frame_id)
                reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
                reference_equal = bool(np.array_equal(reference, labels))
                if not reference_equal:
                    raise ValueError("Official geometric cache mismatch at frame %d" % frame_id)
            target = a.output / ("%05d_mask.png" % frame_id)
            if target.exists():
                raise FileExistsError(target)
            temporary = a.output / ("%05d_mask.tmp.png" % frame_id)
            if not cv2.imwrite(str(temporary), labels):
                raise IOError("Could not write geometric mask for frame %d" % frame_id)
            temporary.replace(target)
            row = {
                "frame": frame_id,
                "rgb_sha256": digest(rgb_path),
                "depth_sha256": digest(depth_path),
                "geometric_sha256": digest(target),
                "segments": int(layers.shape[0]),
                "labeled_pixels": int(np.count_nonzero(labels)),
                "reference_cache_equal": reference_equal,
                "seconds": time.perf_counter() - started,
            }
            meta.write(json.dumps(row, sort_keys=True) + "\n")
            meta.flush()
            if index % 25 == 0 or index == 1:
                print(json.dumps({"frames": index, "last": frame_id, "elapsed_s": round(time.perf_counter() - began, 1)}), flush=True)
    print("PASS depth masks: %d frames, %.1f s" % (index, time.perf_counter() - began), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print("FAIL depth masks: %r" % exc, file=sys.stderr, flush=True)
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
