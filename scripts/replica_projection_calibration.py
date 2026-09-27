"""Reproducible GT-only sensitivity run for the Replica-CA-v2 geometry mapper.

Example:
  python scripts/replica_projection_calibration.py \
    --reference-root /path/to/existing/reference \
    --config unified_eval/configs/replica_ca_v2.pending.json \
    --out /data/chenkejun/CVPR/evaluation_results/replica_projection_calibration \
    --scenes room0 room1 office0 --perturb-scenes room0 \
    --debug-min-valid-instance-vertices 100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from unified_eval.calibration import (
    eligible_oracle_ids, evaluate_oracle, oracle_clouds, oracle_identity_pass,
)
from unified_eval.io import sha256_file
from unified_eval.replica import SCENES, load_existing_reference
from unified_eval.schema import Protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, required=True)
    parser.add_argument("--perturb-scenes", nargs="*", choices=SCENES, default=[])
    parser.add_argument("--deltas-m", nargs="+", type=float,
                        default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05])
    parser.add_argument("--debug-min-valid-instance-vertices", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--gate-delta-m", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.scenes) != len(set(args.scenes)) or not set(args.perturb_scenes) <= set(args.scenes):
        raise ValueError("Scenes must be unique and perturb-scenes must be among scenes")
    if any(not np.isfinite(d) or d <= 0 for d in args.deltas_m):
        raise ValueError("All mapping distances must be finite and positive")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if raw.get("name") != "Replica-CA-v2" or raw.get("frozen") is True:
        raise ValueError("Calibration requires the pending, unfrozen Replica-CA-v2 config")
    raw["geometry_mapping"]["max_distance_m"] = args.deltas_m[0]
    raw["instance_filter"]["min_valid_instance_vertices"] = args.debug_min_valid_instance_vertices
    # Only required by unrelated merge/split diagnostics; no AP/F1 effect.
    raw["significant_overlap"] = {"min_intersection_vertices": 10,
                                   "min_gt_fraction": 0.05}
    protocol = Protocol.from_dict(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    scene_sources = []
    for scene in args.scenes:
        gt = load_existing_reference(args.reference_root, scene)
        ids = eligible_oracle_ids(gt, protocol.min_valid_instance_vertices)
        scene_sources.append({"scene_id": scene, "source_reference": gt.metadata["source_reference"],
                              "source_reference_sha256": gt.metadata["source_reference_sha256"],
                              "eligible_instance_count": int(len(ids))})
        if not len(ids):
            raise ValueError(f"{scene} has no eligible GT instances")
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(ids))
        exact_ids = ids[order]
        exact_clouds = oracle_clouds(gt, exact_ids)
        plans = [("exact", exact_ids, exact_clouds, 0.0, 1.0)]
        if scene in args.perturb_scenes:
            for k in (20, 40):
                if k < len(ids):
                    plans.append((f"subset_{k}", exact_ids[:k], exact_clouds[:k], 0.0, 1.0))
            for sigma in (0.002, 0.005, 0.01):
                plans.append((f"gaussian_{sigma:g}m", exact_ids,
                              oracle_clouds(gt, exact_ids, sigma_m=sigma, seed=args.seed),
                              sigma, 1.0))
            plans.append(("density_25pct", exact_ids,
                          oracle_clouds(gt, exact_ids, keep_fraction=0.25, seed=args.seed),
                          0.0, 0.25))
        for name, plan_ids, clouds, sigma, keep in plans:
            for delta in args.deltas_m:
                result = evaluate_oracle(gt, protocol, plan_ids, clouds, delta_m=delta)
                result.update({"condition": name, "noise_sigma_m": sigma,
                               "point_keep_fraction": keep,
                               "identity_pass": oracle_identity_pass(result) if name == "exact" else None,
                               "status": "DEBUG_ONLY / GT_ORACLE / NON_OFFICIAL"})
                rows.append(result)
                print(json.dumps({k: result[k] for k in
                      ("scene_id", "condition", "delta_m", "TP", "FP", "FN",
                       "AP50", "foreign_instance_vertex_claims", "identity_pass")}), flush=True)
    manifest = {"status": "DEBUG_ONLY / GT_ORACLE / NON_OFFICIAL",
                "purpose": "geometry calibration independent of method ranking",
                "config": str(args.config), "config_sha256": sha256_file(args.config),
                "reference_root": str(args.reference_root), "scenes": scene_sources,
                "seed": args.seed, "deltas_m": args.deltas_m,
                "debug_min_valid_instance_vertices": args.debug_min_valid_instance_vertices,
                "noise_model": "independent isotropic Gaussian displacement per retained GT vertex",
                "density_model": "deterministic uniform 25% vertex sampling",
                "subset_order": "seeded random permutation of eligible GT IDs",
                "significant_overlap_diagnostic_only": raw["significant_overlap"],
                "identity_gate": "Necessary exact-oracle check only: TP=N, FP=FN=0, CA-AP50_uniform=F1=1; PQ N/A if projected masks overlap. Passing does not freeze the protocol.",
                "formal_threshold_frozen": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fields = ["scene_id", "condition", "delta_m", "oracle_instance_count", "AP50", "F1_0_5",
              "TP", "FP", "FN", "self_iou_min", "self_iou_mean",
              "foreign_instance_vertex_claims", "overlapped_ref_vertices", "PQ", "PQ_status",
              "identity_pass", "status"]
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if args.gate_delta_m is not None:
        gates = [r for r in rows if r["condition"] == "exact" and
                 abs(r["delta_m"] - args.gate_delta_m) < 1e-12]
        if len(gates) != len(args.scenes) or not all(r["identity_pass"] for r in gates):
            raise SystemExit("Exact oracle gate failed; protocol cannot be frozen")


if __name__ == "__main__":
    main()
