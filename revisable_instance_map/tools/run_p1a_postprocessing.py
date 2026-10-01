#!/usr/bin/env python3
"""Apply the unchanged main diffusion to A0/A1/A2 and pool uniform v3 metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_p1a_validation import ROOT, TOOLS, V3_FLAGS, stage, evaluate_surface
from run_baseline_association import sha256_file, write_json


def pool_metrics(root, scenes, variant, input_root, output):
    batch = {"scenes": [dict(gt=str(input_root / scene / "ground_truth/gt.npz"),
        prediction=str(root / scene / variant / "v3/adapter/canonical_prediction.npz"),
        diagnostic_prediction=str(root / scene / variant / "v3/adapter/diagnostic_support_prediction.npz")) for scene in scenes]}
    output.mkdir(parents=True, exist_ok=True)
    batch_path = output / "scenes.json"
    write_json(batch_path, batch)
    stage([sys.executable, "-m", "unified_eval.cli", "eval-batch", "--config",
           ROOT / "unified_eval/configs/replica_ca_v3.pending.json", *V3_FLAGS,
           "--scenes", batch_path, "--out", output], output / "pool.log")
    return json.loads((output / "summary.json").read_text())


def run_diffusion(scene, variant, args, code_version):
    native = args.native_root / scene / variant
    out = args.output_root / scene / variant
    out.mkdir(parents=True, exist_ok=False)
    source = native / "surface_p0"
    source_hash = sha256_file(source / "instance_surface.npz")
    print(f"{scene} {variant}: unchanged main diffusion", flush=True)
    stage([sys.executable, TOOLS / "repair_p0_surface.py", "--source-dir", source,
           "--geometry", args.input_root / scene / "geometry/surface.ply",
           "--output-dir", out / "diffusion"], out / "diffusion.log")
    report = json.loads((out / "diffusion/materialization_report.json").read_text())
    if sha256_file(source / "instance_surface.npz") != source_hash:
        raise ValueError("Postprocessing modified native observed evidence")
    surface = out / "diffusion/holes_geodesic/instance_surface.npz"
    metrics = evaluate_surface(scene, variant + "_same_diffusion", surface,
        args.input_root / scene / "ground_truth/gt.npz", out / "v3", code_version)
    summary = {"scene": scene, "variant": variant, "settings_sha256": report["settings_sha256"],
        "native_surface_sha256": source_hash,
        "identity_checkpoint_sha256": sha256_file(native / "association/observation_support_3cm.npz"),
        "map_version": json.loads((native / "association/association_report.json").read_text())["map_version"],
        "postprocessed_surface_sha256": sha256_file(surface), "ground_truth_used_for_postprocessing": False,
        "diffusion_seconds": report["seconds"], **metrics}
    write_json(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("scene", "variant", "AP50", "F1")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    native = json.loads((args.native_root / "summary.json").read_text())
    source_manifest = json.loads((args.native_root / "run_manifest.json").read_text())
    if native["status"] != "PASS" or source_manifest["status"] != "PASS":
        raise ValueError("Complete native P1-A validation before postprocessing")
    for name, expected_hash in source_manifest["source_sha256"].items():
        if sha256_file(ROOT / name) != expected_hash:
            # A later checkpoint I/O improvement need not invalidate an already
            # frozen native map. Verify the original source snapshot instead.
            snapshot = subprocess.check_output(["git", "show", source_manifest["base_commit"] + ":" + name], cwd=ROOT)
            if hashlib.sha256(snapshot).hexdigest() != expected_hash:
                raise ValueError("Cannot verify native experiment source snapshot: " + name)
    scenes = source_manifest["scenes"]
    variants = ("A0", "A1", "A2")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    code_version = source_manifest["code_version"]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_diffusion, scene, variant, args, code_version)
                   for scene in scenes for variant in variants]
        for future in as_completed(futures):
            summaries.append(future.result())
    if len({item["settings_sha256"] for item in summaries}) != 1:
        raise ValueError("Diffusion settings differ across variants")
    for scene in scenes:
        for name in ("baseline", "holes_geodesic"):
            left = args.output_root / scene / "A0/diffusion" / name / "instance_surface.npz"
            right = args.output_root / scene / "A1/diffusion" / name / "instance_surface.npz"
            with np.load(left) as a, np.load(right) as b:
                for key in ("xyz_m", "rgb", "instance_id"):
                    np.testing.assert_array_equal(a[key], b[key])
    pooled = {"native": {}, "same_diffusion": {}}
    for variant in variants:
        pooled["native"][variant] = pool_metrics(args.native_root, scenes, variant, args.input_root,
                                                 args.output_root / "pooled/native" / variant)
        pooled["same_diffusion"][variant] = pool_metrics(args.output_root, scenes, variant, args.input_root,
                                                        args.output_root / "pooled/same_diffusion" / variant)
    write_json(args.output_root / "summary.json", {"status": "PASS", "per_scene": summaries,
        "A0_A1_same_diffusion_maps_identical": True, "pooled": pooled,
        "settings_sha256": summaries[0]["settings_sha256"],
        "native_run_manifest_sha256": sha256_file(args.native_root / "run_manifest.json"),
        "postprocessing_entry_sha256": sha256_file(Path(__file__)),
        "postprocessing_code_sha256": sha256_file(TOOLS / "repair_p0_surface.py")})
    print("PASS: native and unchanged diffusion, pooled uniform v3", flush=True)


if __name__ == "__main__":
    main()
