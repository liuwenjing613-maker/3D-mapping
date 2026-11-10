#!/usr/bin/env python3
"""Fresh A0/A1/A2 association runs, fixed P0 publication, and uniform v3 eval.

Identity inference uses only RGB-D observations seen so far. GT is opened only
after a completed surface output has been hashed. Thresholds remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "revisable_instance_map/tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revisable_instance_map/src"))
from unified_eval.geometry import map_instances_to_reference_v3
from unified_eval.io import load_gt, save_prediction
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.identity_evidence import FrameIdentityEvidence
from run_baseline_association import sha256_file, write_json

SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
MODES = {"A0": "legacy", "A1": "binary-ledger", "A2": "probabilistic"}
V3_FLAGS = ("--debug-max-distance-m", "0.01", "--debug-diagnostic-max-distance-m", "0.02",
            "--debug-min-valid-instance-vertices", "100", "--debug-significant-min-vertices", "10",
            "--debug-significant-min-gt-fraction", "0.05")


def stage(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        subprocess.run([str(item) for item in command], check=True, cwd=ROOT, stdout=output,
                       stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(ROOT)})


def evaluate_surface(scene, variant, path, gt_path, out, method_commit):
    # Maps are fixed before loading evaluation GT.
    surface_hash = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        xyz, labels = data["xyz_m"].copy(), data["instance_id"].copy()
    gt = load_gt(gt_path)
    ids = np.unique(labels[labels > 0])
    order = np.argsort(labels, kind="stable")
    sorted_labels, sorted_xyz = labels[order], xyz[order]
    clouds = [sorted_xyz[np.searchsorted(sorted_labels, key, side="left"):
                         np.searchsorted(sorted_labels, key, side="right")] for key in ids]
    adapted = map_instances_to_reference_v3(clouds, gt.xyz_ref, .01, scene_id=scene,
        method_name="P1A_" + variant, method_commit=method_commit, adapter_version="shared_tsdf_surface_v3",
        protocol_version="Replica-CA-v3", diagnostic_distance_m=.02,
        metadata={"source_map_sha256": surface_hash, "source_instance_surface": str(path),
                  "frame_count_from_source": 400, "debug_only": True})
    for prediction in (adapted.prediction, adapted.diagnostic_prediction):
        prediction.metadata["source_map_sha256"] = surface_hash
        for key, instance in zip(ids, prediction.instances):
            instance.instance_uid = f"p1a-id:{int(key)}"
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_prediction(adapter_dir / "canonical_prediction.npz", adapted.prediction)
    save_prediction(adapter_dir / "diagnostic_support_prediction.npz", adapted.diagnostic_prediction)
    write_json(adapter_dir / "adapter_stats.json", adapted.statistics)
    stage([sys.executable, "-m", "unified_eval.cli", "eval-scene", "--config",
           ROOT / "unified_eval/configs/replica_ca_v3.pending.json", *V3_FLAGS,
           "--gt", gt_path, "--pred", adapter_dir / "canonical_prediction.npz",
           "--diagnostic-pred", adapter_dir / "diagnostic_support_prediction.npz",
           "--out", out / "evaluation"], out / "evaluation.log")
    metrics = json.loads((out / "evaluation/metrics.json").read_text())
    return {"AP": metrics["CA_AP_uniform"], "AP50": metrics["CA_AP50_uniform"],
            "PQ": metrics["CA_PQ"]["PQ"], "F1": metrics["CA_PRF1_0_5"]["F1"],
            "diagnostics": metrics["diagnostics"], "status": metrics["status"]}


def run_variant(scene, variant, args, code_version):
    base = args.input_root / scene
    out = args.output_root / scene / variant
    out.mkdir(parents=True, exist_ok=False)
    config, observations = base / "configs/p0_parent_regrouped.json", base / "observations/observations.jsonl"
    assoc = out / "association"
    print(f"{scene} {variant}: fresh online association", flush=True)
    stage([sys.executable, TOOLS / "run_baseline_association.py", "--config", config,
           "--observations", observations, "--output-dir", assoc,
           "--association-mode", MODES[variant], "--allow-multiple-observations-per-instance-per-frame",
           "--legacy-association-sampling"], out / "association.log")
    print(f"{scene} {variant}: fixed P0 surface publication", flush=True)
    stage([sys.executable, TOOLS / "rebuild_p0_from_regions.py", "--source-dir", base / "surface_p0",
           "--observations", observations, "--config", config,
           "--associations", assoc / "associations.jsonl", "--identity-checkpoint", assoc / "observation_support_3cm.npz",
           "--output-dir", out / "surface_p0"], out / "surface.log")
    association_report = json.loads((assoc / "association_report.json").read_text())
    surface_report = json.loads((out / "surface_p0/materialization_report.json").read_text())
    print(f"{scene} {variant}: v3 evaluation", flush=True)
    metrics = evaluate_surface(scene, variant, out / "surface_p0/instance_surface.npz",
                               base / "ground_truth/gt.npz", out / "v3", code_version)
    summary = {"scene": scene, "variant": variant, "association_mode": MODES[variant],
               "instances": association_report["instance_count"],
               "association_seconds": association_report["total_seconds"],
               "association_peak_rss_mb": association_report["peak_process_rss_mb"],
               "decision_counts": association_report["decision_counts"],
               "state_counts": surface_report["state_counts"],
               "published_surface_fraction": surface_report["labeled_surface_fraction"], **metrics}
    write_json(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("scene", "variant", "AP50", "F1", "instances")}), flush=True)
    return summary


def assert_old_fields_equal(expected, actual):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, key
            assert_old_fields_equal(value, actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_old_fields_equal(left, right)
    else:
        assert expected == actual, (expected, actual)


def audit_scene(scene, args):
    folder = args.output_root / scene
    old = [json.loads(line) for line in (args.input_root / scene / "association/associations.jsonl").read_text().splitlines()]
    a0 = [json.loads(line) for line in (folder / "A0/association/associations.jsonl").read_text().splitlines()]
    a1 = [json.loads(line) for line in (folder / "A1/association/associations.jsonl").read_text().splitlines()]
    assert_old_fields_equal(old, a0)
    assert_old_fields_equal(a0, a1)
    for variant in ("A0", "A1"):
        with np.load(args.input_root / scene / "surface_p0/surface_evidence.npz") as reference:
            with np.load(folder / variant / "surface_p0/surface_evidence.npz") as actual:
                for key in reference.files:
                    np.testing.assert_array_equal(reference[key], actual[key], err_msg=variant + ":" + key)
    audit = {"status": "PASS", "A0_equals_original_all_decision_fields": True,
             "A1_equals_A0_all_original_decision_fields": True,
             "A0_A1_equals_original_all_P0_evidence_arrays": True}
    if args.audit_prefix:
        prefix_dir = folder / "A2_prefix50"
        base = args.input_root / scene
        stage([sys.executable, TOOLS / "run_baseline_association.py", "--config", base / "configs/p0_parent_regrouped.json",
               "--observations", base / "observations/observations.jsonl", "--frame-count", "50",
               "--output-dir", prefix_dir, "--association-mode", "probabilistic",
               "--allow-multiple-observations-per-instance-per-frame", "--legacy-association-sampling"], folder / "prefix.log")
        prefix = [json.loads(line) for line in (prefix_dir / "associations.jsonl").read_text().splitlines()]
        full = [json.loads(line) for line in (folder / "A2/association/associations.jsonl").read_text().splitlines()]
        assert prefix == [row for row in full if row["frame_id"] < 250]
        state = OnlineVoxelAssociator.from_checkpoint(folder / "A2/association/observation_support_3cm.npz")
        rebuilt = FrameIdentityEvidence.rebuild(entry for entry in state.observation_support.values() if entry.frame_id < 250)
        with np.load(prefix_dir / "identity_voxel_counts_3cm.npz") as saved:
            for key, value in rebuilt.export_arrays().items():
                np.testing.assert_array_equal(value, saved[key])
        audit["independent_50_frame_prefix_decisions_and_counts_equal_full_run_prefix"] = True
    write_json(folder / "invariant_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=["room0"])
    parser.add_argument("--input-root", type=Path, default=Path("/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--audit-prefix", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for pattern in (
        "revisable_instance_map/src/revisable_instance_map/*.py", "revisable_instance_map/tools/*identity*.py",
        "revisable_instance_map/tools/run_baseline_association.py", "revisable_instance_map/tools/rebuild_p0_from_regions.py",
        "revisable_instance_map/tools/run_p1a_validation.py") for path in ROOT.glob(pattern)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_version = commit + "+working-" + hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    write_json(args.output_root / "run_manifest.json", {
        "status": "RUNNING", "base_commit": commit, "code_version": code_version,
        "source_sha256": source_hashes, "scenes": args.scenes, "variants": MODES,
        "frame_ids": list(range(0, 2000, 5)), "thresholds_changed": False,
        "ground_truth_used_for_mapping": False, "v3_flags": V3_FLAGS,
        "protocol_sha256": sha256_file(ROOT / "unified_eval/configs/replica_ca_v3.pending.json"),
    })
    summaries = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_variant, scene, variant, args, code_version)
                   for scene in args.scenes for variant in MODES]
        for future in as_completed(futures):
            summaries.append(future.result())
    audits = {scene: audit_scene(scene, args) for scene in args.scenes}
    summaries.sort(key=lambda item: (args.scenes.index(item["scene"]), item["variant"]))
    write_json(args.output_root / "summary.json", {"status": "PASS", "per_scene": summaries,
        "invariant_audits": audits, "seconds": time.perf_counter() - started,
        "macro_scene_means": {variant: {key: float(np.mean([item[key] for item in summaries if item["variant"] == variant]))
                                       for key in ("AP", "AP50", "PQ", "F1", "published_surface_fraction")}
                              for variant in MODES}})
    with (args.output_root / "summary.csv").open("w", newline="") as output:
        fields = ("scene", "variant", "AP", "AP50", "PQ", "F1", "instances", "published_surface_fraction", "association_seconds", "association_peak_rss_mb")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    manifest = json.loads((args.output_root / "run_manifest.json").read_text())
    manifest["status"] = "PASS"
    manifest["summary_sha256"] = sha256_file(args.output_root / "summary.json")
    write_json(args.output_root / "run_manifest.json", manifest)
    print("PASS: fresh A0/A1/A2, binary equivalence, causal prefix, uniform v3", flush=True)


if __name__ == "__main__":
    main()
