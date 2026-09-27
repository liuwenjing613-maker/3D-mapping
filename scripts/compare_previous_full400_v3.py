"""Re-adapt prior complete room0 maps with one unfrozen Replica-CA-v3 debug protocol.

All inputs are existing run manifests and native outputs. The script refuses to
mix frame lists or GT meshes, and never writes an official summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from unified_eval.io import load_prediction, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--ovimap-adapter-manifest", type=Path, required=True)
    parser.add_argument("--ovimap-eval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group", choices=("all", "main"), default="all")
    parser.add_argument("--delta-geo-m", type=float, required=True)
    parser.add_argument("--delta-diag-m", type=float, required=True)
    parser.add_argument("--min-gt-vertices", type=int, required=True)
    parser.add_argument("--significant-min-vertices", type=int, required=True)
    parser.add_argument("--significant-min-gt-fraction", type=float, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("name") != "Replica-CA-v3" or config.get("frozen") is not False:
        raise ValueError("Use the pending Replica-CA-v3 config")
    sources = json.loads(args.source_index.read_text(encoding="utf-8"))
    if args.group == "main":
        sources = [row for row in sources if row["group"] == "main"]
    if not sources:
        raise ValueError("No source runs")
    gt_sha = sha256_file(args.gt)
    for row in sources:
        if row["source_frame_count"] != 400 or row["gt_sha256"] != gt_sha:
            raise ValueError(f"{row['name']}: frame count or GT differs")
        if not Path(row["source_map"]).is_file():
            raise FileNotFoundError(row["source_map"])
    ovi_adapter = json.loads(args.ovimap_adapter_manifest.read_text(encoding="utf-8"))
    ovi_eval = json.loads(args.ovimap_eval_manifest.read_text(encoding="utf-8"))
    if ovi_adapter["gt_file_sha256"] != gt_sha:
        raise ValueError("OVI-MAP GT differs")
    common = ["--config", str(args.config),
        "--debug-max-distance-m", str(args.delta_geo_m),
        "--debug-diagnostic-max-distance-m", str(args.delta_diag_m),
        "--debug-min-valid-instance-vertices", str(args.min_gt_vertices),
        "--debug-significant-min-vertices", str(args.significant_min_vertices),
        "--debug-significant-min-gt-fraction", str(args.significant_min_gt_fraction)]
    args.out.mkdir(parents=True, exist_ok=True)
    items = []
    for row in sources:
        slug = Path(row["directory"]).name
        items.append({"slug": slug, "name": row["name"], "kind": "conceptgraphs", "source": row})
    if args.group == "all":
        items.append({"slug": "OVI-MAP", "name": "OVI-MAP", "kind": "ovimap"})
    results = []
    expected_frame_hash = None
    for item in items:
        out = args.out / item["slug"]
        adapter = out / "adapter"
        evaluation = out / "evaluation"
        if item["kind"] == "conceptgraphs":
            source = item["source"]
            cmd = [sys.executable, "-m", "unified_eval.cli", "adapt-conceptgraphs", *common,
                "--gt", str(args.gt), "--map", source["source_map"],
                "--experiment-manifest", source["source_experiment_manifest"],
                "--method-name", item["name"], "--method-commit", source["method_commit"],
                "--out", str(adapter)]
        else:
            cmd = [sys.executable, "-m", "unified_eval.cli", "adapt-ovimap", *common,
                "--gt", str(args.gt), "--export-npz", ovi_adapter["source_export"],
                "--export-manifest", ovi_adapter["source_export_manifest"],
                "--input-alignment", ovi_adapter["source_input_alignment"],
                "--method-name", "OVI-MAP", "--method-commit", ovi_eval["method_commit"],
                "--out", str(adapter)]
        print("ADAPT", item["name"], flush=True)
        subprocess.run(cmd, check=True)
        pred_file = adapter / "canonical_prediction.npz"
        diag_file = adapter / "diagnostic_support_prediction.npz"
        print("EVALUATE", item["name"], flush=True)
        subprocess.run([sys.executable, "-m", "unified_eval.cli", "eval-scene", *common,
            "--gt", str(args.gt), "--pred", str(pred_file),
            "--diagnostic-pred", str(diag_file), "--out", str(evaluation)], check=True)
        metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))
        frame_hash = manifest["prediction_source"].get("frame_list_sha256")
        frame_count = manifest["prediction_source"].get("frame_count_from_source")
        if frame_count != 400 or not frame_hash:
            raise ValueError(f"{item['name']}: incomplete frame provenance")
        if expected_frame_hash is None:
            expected_frame_hash = frame_hash
        if frame_hash != expected_frame_hash or manifest["gt_sha256"] != gt_sha:
            raise ValueError(f"{item['name']}: frame list or GT differs")
        if metrics["status"] != "DEBUG_ONLY / NON_OFFICIAL":
            raise ValueError("Unfrozen v3 result was not marked DEBUG_ONLY")
        prf = metrics["CA_PRF1_0_5"]
        structure = metrics["structure"]
        diag = metrics["diagnostics"]
        result = {"name": item["name"], "slug": item["slug"], "native_kind": item["kind"],
            "native_instances": len(load_prediction(pred_file).instances),
            "AP25": metrics["CA_AP25_uniform"], "AP50": metrics["CA_AP50_uniform"],
            "AP": metrics["CA_AP_uniform"], "P": prf["P"], "R": prf["R"],
            "F1": prf["F1"], "PQ": metrics["CA_PQ"]["PQ"],
            "SQ": metrics["CA_PQ"]["SQ"], "RQ": metrics["CA_PQ"]["RQ"],
            "TP": prf["TP"], "FP": prf["FP"], "FN": prf["FN"],
            "split_gt_rate": structure["split_gt_rate"],
            "merge_prediction_rate": structure["merge_prediction_rate"],
            "duplicate_prediction_rate": structure["duplicate_prediction_rate"],
            "gt_surface_coverage": diag["gt_surface_coverage"],
            "macro_best_gt_recall": diag["macro_best_gt_recall"],
            "macro_prediction_purity": diag["macro_prediction_purity"],
            "frame_list_sha256": frame_hash, "gt_sha256": gt_sha,
            "status": metrics["status"],
            "source_map_sha256": manifest["prediction_source"].get("source_map_sha256")
                or manifest["prediction_source"].get("source_export_sha256"),
            "metrics_file": str(evaluation / "metrics.json")}
        results.append(result)
        print("DONE", item["name"], "F1", result["F1"], "AP50", result["AP50"], flush=True)
        (args.out / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["name", "native_instances", "AP25", "AP50", "AP", "P", "R", "F1",
        "PQ", "SQ", "RQ", "TP", "FP", "FN", "split_gt_rate", "merge_prediction_rate",
        "duplicate_prediction_rate", "gt_surface_coverage", "macro_best_gt_recall",
        "macro_prediction_purity", "status"]
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    (args.out / "run_manifest.json").write_text(json.dumps({
        "status": "DEBUG_ONLY / NON_OFFICIAL", "protocol": "Replica-CA-v3",
        "evaluator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": str(args.config), "config_sha256": sha256_file(args.config),
        "gt": str(args.gt), "gt_sha256": gt_sha, "frame_count": 400,
        "frame_list_sha256": expected_frame_hash,
        "delta_geo_m": args.delta_geo_m, "delta_diag_m": args.delta_diag_m,
        "min_gt_vertices": args.min_gt_vertices,
        "significant_min_vertices": args.significant_min_vertices,
        "significant_min_gt_fraction": args.significant_min_gt_fraction,
        "source_index": str(args.source_index), "source_index_sha256": sha256_file(args.source_index),
        "items": [{"name": row["name"], "source_map_sha256": row["source_map_sha256"]}
                  for row in results]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
