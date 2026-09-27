"""Report native-map point-to-reference distances without evaluating method quality."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from unified_eval.io import sha256_file
from unified_eval.replica import load_existing_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=200000)
    args = parser.parse_args()
    if args.sample_limit < 1:
        raise ValueError("sample-limit must be positive")
    gt = load_existing_reference(args.reference_root, args.scene)
    ref = np.asarray(gt.xyz_ref[gt.valid_vertex_mask & ~gt.ignore_vertex_mask])
    tree = cKDTree(ref)
    rows = []
    rng = np.random.default_rng(20260927)
    for path in args.map:
        # Only run on trusted local ConceptGraphs outputs: pickle can execute code.
        with gzip.open(path, "rb") as handle:
            raw = pickle.load(handle)
        clouds = [np.asarray(obj["pcd_np"], dtype=np.float64) for obj in raw["objects"]]
        sizes = np.array([len(cloud) for cloud in clouds])
        if not len(sizes) or sizes.sum() == 0:
            raise ValueError(f"{path}: no native points")
        all_points = np.concatenate(clouds)
        chosen = (rng.choice(len(all_points), min(args.sample_limit, len(all_points)), replace=False)
                  if len(all_points) > args.sample_limit else np.arange(len(all_points)))
        distances, _ = tree.query(all_points[chosen], k=1, workers=-1)
        row = {"source_map": str(path), "source_map_sha256": sha256_file(path),
               "scene": args.scene, "native_instance_count": len(clouds),
               "native_point_count": int(len(all_points)), "sample_count": int(len(chosen)),
               "distance_median_m": float(np.median(distances)),
               "distance_p90_m": float(np.percentile(distances, 90)),
               "distance_p95_m": float(np.percentile(distances, 95)),
               "distance_p99_m": float(np.percentile(distances, 99)),
               "status": "DEBUG_ONLY / GEOMETRY_DISTRIBUTION / NON_OFFICIAL"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scene": args.scene,
        "reference_source_sha256": gt.metadata["source_reference_sha256"],
        "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
