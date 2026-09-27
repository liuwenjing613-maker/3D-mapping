"""GT-only Replica-CA-v3 geometry audit; never generates formal benchmark scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from unified_eval.calibration import eligible_oracle_ids, evaluate_oracle, oracle_clouds
from unified_eval.io import sha256_file
from unified_eval.replica import SCENES, load_existing_reference
from unified_eval.schema import Protocol


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, required=True)
    parser.add_argument("--stress-scenes", nargs="*", choices=SCENES, default=[])
    parser.add_argument("--deltas-m", nargs="+", type=float, default=[.005, .01, .015, .02, .03, .05])
    parser.add_argument("--debug-min-valid-instance-vertices", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def main():
    args = args_parser()
    if len(set(args.scenes)) != len(args.scenes) or not set(args.stress_scenes) <= set(args.scenes):
        raise ValueError("Scenes must be unique and stress scenes must be included")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if raw.get("name") != "Replica-CA-v3" or raw.get("frozen") is True:
        raise ValueError("Use the pending, unfrozen Replica-CA-v3 config")
    raw["geometry_mapping"]["max_distance_m"] = args.deltas_m[0]
    raw["diagnostic_mapping"]["max_distance_m"] = args.deltas_m[0]
    raw["instance_filter"]["min_valid_instance_vertices"] = args.debug_min_valid_instance_vertices
    raw["significant_overlap"] = {"min_intersection_vertices": 10, "min_gt_fraction": .05}
    protocol = Protocol.from_dict(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    for scene in args.scenes:
        gt = load_existing_reference(args.reference_root, scene)
        ids = eligible_oracle_ids(gt, protocol.min_valid_instance_vertices)
        if not len(ids):
            raise ValueError(f"{scene}: no eligible GT")
        rng = np.random.default_rng(args.seed)
        ids = ids[rng.permutation(len(ids))]
        exact = oracle_clouds(gt, ids)
        sources.append({"scene_id": scene, "eligible_instances": len(ids),
                        "reference_sha256": gt.metadata["source_reference_sha256"]})
        plans = [("exact", ids, exact)]
        if scene in args.stress_scenes:
            for count in (20, 40):
                if count < len(ids):
                    plans.append((f"subset_{count}", ids[:count], exact[:count]))
            for density in (.5, .25, .125):
                plans.append((f"density_{density:g}", ids,
                              oracle_clouds(gt, ids, keep_fraction=density, seed=args.seed)))
            for sigma in (.002, .005, .01):
                plans.append((f"gaussian_{sigma:g}m", ids,
                              oracle_clouds(gt, ids, sigma_m=sigma, seed=args.seed)))
        for condition, plan_ids, clouds in plans:
            for delta in args.deltas_m:
                result = evaluate_oracle(gt, protocol, plan_ids, clouds, delta_m=delta)
                result.update({"condition": condition,
                    "strict_identity_pass": (result["TP"] == len(ids) and result["FP"] == 0
                        and result["FN"] == 0 and result["AP50"] == 1
                        and result["F1_0_5"] == 1 and result["PQ"] == 1)
                        if condition == "exact" else None,
                    "status": "DEBUG_ONLY / GT_ORACLE / NON_OFFICIAL"})
                rows.append(result)
                print(json.dumps({k: result[k] for k in ("scene_id", "condition", "delta_m",
                      "TP", "FP", "FN", "AP50", "PQ", "self_iou_mean",
                      "foreign_instance_vertex_claims", "strict_identity_pass")}), flush=True)
    manifest = {"status": "DEBUG_ONLY / GT_ORACLE / NON_OFFICIAL",
        "config": str(args.config), "config_sha256": sha256_file(args.config),
        "reference_root": str(args.reference_root), "sources": sources,
        "deltas_m": args.deltas_m, "seed": args.seed,
        "debug_min_valid_instance_vertices": args.debug_min_valid_instance_vertices,
        "stress_scenes": args.stress_scenes,
        "formal_threshold_frozen": False,
        "gate": "exact GT: TP=N, FP=FN=0, AP50=F1=PQ=1; inspect self IoU and cross-instance claims separately"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out / "results.csv").open("w", newline="") as handle:
        fields = ["scene_id", "condition", "delta_m", "oracle_instance_count", "TP", "FP", "FN",
                  "AP50", "F1_0_5", "PQ", "self_iou_min", "self_iou_mean",
                  "foreign_instance_vertex_claims", "strict_identity_pass", "status"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
