#!/usr/bin/env python3
"""Run the frozen P0 no-repair pipeline on all eight Replica scenes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np


SCENES = ("room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = Path("/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main")
REPLICA_ROOT = Path("/home/chenkejun/beauty/conceptgraphs/data/Replica")
RAW_MASK_ROOT = Path("/data/chenkejun/ovimap_runtime_20260908/sem_aligned")
OVI_RESULT_ROOT = Path("/data/chenkejun/ovimap_runtime_20260908/results")
REFERENCE_ROOT = Path("/home/chenkejun/beauty/ovimap_aligned_eval_20260908/reference")
PYTHON = Path("/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python")
PROTOCOL = REPO_ROOT / "unified_eval/configs/replica_ca_v1.json"
PROGRESS_LOCK = threading.Lock()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_passed(path):
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in {
            "PASS", "protocol_configured"}
    except (OSError, ValueError):
        return False


def raw_config(scene):
    return {
        "contract_version": 1,
        "dataset": "Replica", "scene": scene,
        "purpose": "p0_replica8_no_repair",
        "frame_selection": {"start": 0, "stop_exclusive": 2000, "stride": 5},
        "source": {
            "scene_root": str(REPLICA_ROOT / scene),
            "rgb_pattern": "results/frame{frame:06d}.jpg",
            "depth_pattern": "results/depth{frame:06d}.png",
            "trajectory": "traj.txt",
            "mask_root": str(RAW_MASK_ROOT / scene),
            "mask_pattern": "frame{frame:06d}.png",
            "mask_metadata": "frames.jsonl",
        },
        "camera": {
            "width": 1200, "height": 680, "fx": 600.0, "fy": 600.0,
            "cx": 599.5, "cy": 339.5,
            "depth_png_units_per_meter": 6553.5,
            "trajectory_matrix": "camera_to_world_4x4_row_major",
        },
        "mask": {
            "encoding": "uint8_instance_id_image", "background_id": 0,
            "provenance": "frozen CropFormer cache with per-frame checksums",
        },
        "rules": {
            "input_is_read_only": True, "use_ground_truth_for_mapping": False,
            "use_future_frames": False, "allow_repair": False,
        },
    }


def ensure_exact_json(path, value):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"Existing config differs: {path}")
    else:
        write_json(path, value)


def prepare_geometry_cache(scene, scene_dir):
    output = scene_dir / "official_geometry_cache"
    report_path = output / "cache_report.json"
    if json_passed(report_path):
        return
    output.mkdir(parents=True, exist_ok=True)
    source = OVI_RESULT_ROOT / f"{scene}_full400/geometrics"
    source_scene = REPLICA_ROOT / scene
    rows = []
    for frame_id in range(0, 2000, 5):
        source_mask = source / f"{frame_id:05d}_mask.png"
        link = output / source_mask.name
        if not source_mask.is_file():
            raise FileNotFoundError(source_mask)
        if link.exists() or link.is_symlink():
            if link.resolve() != source_mask.resolve():
                raise ValueError(f"Unexpected geometry link: {link}")
        else:
            link.symlink_to(source_mask)
        rgb = source_scene / f"results/frame{frame_id:06d}.jpg"
        depth = source_scene / f"results/depth{frame_id:06d}.png"
        mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (680, 1200) or mask.dtype != np.uint8:
            raise ValueError(f"Invalid official geometry mask: {source_mask}")
        rows.append({
            "frame": frame_id, "rgb_sha256": sha256_file(rgb),
            "depth_sha256": sha256_file(depth),
            "geometric_sha256": sha256_file(source_mask),
            "segments": int(np.max(mask)),
            "labeled_pixels": int(np.count_nonzero(mask)),
            "reference_cache_equal": True,
        })
    manifest = output / "frames.jsonl"
    temporary = output / "frames.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(manifest)
    write_json(report_path, {
        "status": "PASS", "scene_id": scene, "frame_count": len(rows),
        "ground_truth_used": False, "source": str(source),
        "manifest_sha256": sha256_file(manifest),
        "storage": "symlinks_to_frozen_official_OVI_geometry_cache",
    })


def update_progress(root, scene, stage, status, seconds=None, error=None):
    path = root / "progress.json"
    with PROGRESS_LOCK:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "status": "RUNNING", "scenes": {}}
        record = value["scenes"].setdefault(scene, {"stages": {}})
        record["stages"][stage] = {"status": status}
        if seconds is not None:
            record["stages"][stage]["seconds"] = seconds
        if error is not None:
            record["stages"][stage]["error"] = error
        record["updated_unix"] = time.time()
        write_json(path, value)


def run_command(scene, stage, marker, command, scene_dir):
    if marker.suffix == ".npz" and marker.is_file():
        update_progress(DATA_ROOT, scene, stage, "SKIPPED_COMPLETE")
        return
    if marker.suffix != ".npz" and json_passed(marker):
        update_progress(DATA_ROOT, scene, stage, "SKIPPED_COMPLETE")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    log_dir = scene_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    update_progress(DATA_ROOT, scene, stage, "RUNNING")
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": list(map(str, command))}) + "\n")
        log.flush()
        result = subprocess.run(list(map(str, command)), cwd=PACKAGE_ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        update_progress(DATA_ROOT, scene, stage, "FAILED", elapsed,
                        f"exit={result.returncode}; log={log_path}")
        raise RuntimeError(f"{scene}/{stage} failed; see {log_path}")
    passed = marker.is_file() if marker.suffix == ".npz" else json_passed(marker)
    if not passed:
        update_progress(DATA_ROOT, scene, stage, "FAILED", elapsed,
                        f"completion marker invalid: {marker}")
        raise RuntimeError(f"{scene}/{stage} produced no valid marker")
    update_progress(DATA_ROOT, scene, stage, "PASS", elapsed)


def run_scene(scene):
    scene_dir = DATA_ROOT / scene
    config_dir = scene_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw = config_dir / "raw.json"
    refined = config_dir / "refined.json"
    parent = config_dir / "p0_parent_regrouped.json"
    ensure_exact_json(raw, raw_config(scene))
    stage_started = time.perf_counter()
    update_progress(DATA_ROOT, scene, "geometry_cache", "RUNNING")
    prepare_geometry_cache(scene, scene_dir)
    update_progress(DATA_ROOT, scene, "geometry_cache", "PASS",
                    time.perf_counter() - stage_started)
    refined_dir = scene_dir / "refined_masks"
    run_command(scene, "refine_masks", refined_dir / "refinement_report.json", [
        PYTHON, PACKAGE_ROOT / "tools/build_ovimap_refined_masks.py",
        "--config", raw, "--geometry-dir", scene_dir / "official_geometry_cache",
        "--output-dir", refined_dir, "--refined-config", refined,
    ], scene_dir)
    parent_dir = scene_dir / "parent_masks"
    run_command(scene, "regroup_masks", parent_dir / "regroup_report.json", [
        PYTHON, PACKAGE_ROOT / "tools/build_refined_parent_masks.py",
        "--raw-config", raw, "--refined-config", refined,
        "--output-dir", parent_dir, "--output-config", parent,
    ], scene_dir)
    geometry_dir = scene_dir / "geometry"
    run_command(scene, "geometry", geometry_dir / "geometry_full400.json", [
        PYTHON, PACKAGE_ROOT / "tools/profile_full_geometry.py",
        "--config", parent, "--output-dir", geometry_dir, "--block-count", "100000",
    ], scene_dir)
    observations_dir = scene_dir / "observations"
    run_command(scene, "observations", observations_dir / "observation_manifest.json", [
        PYTHON, PACKAGE_ROOT / "tools/build_raw_observations.py",
        "--config", parent, "--output-dir", observations_dir,
    ], scene_dir)
    association_dir = scene_dir / "association"
    run_command(scene, "association", association_dir / "association_report.json", [
        PYTHON, PACKAGE_ROOT / "tools/run_baseline_association.py",
        "--config", parent,
        "--observations", observations_dir / "observations.jsonl",
        "--output-dir", association_dir, "--frame-count", "400",
        "--allow-multiple-observations-per-instance-per-frame",
    ], scene_dir)
    p0_dir = scene_dir / "surface_p0"
    run_command(scene, "surface_p0", p0_dir / "materialization_report.json", [
        PYTHON, PACKAGE_ROOT / "tools/materialize_surface_evidence.py",
        "--config", parent,
        "--observations", observations_dir / "observations.jsonl",
        "--associations", association_dir / "associations.jsonl",
        "--tsdf-surface", geometry_dir / "surface.ply", "--output-dir", p0_dir,
        "--frame-count", "400", "--pixel-stride", "2",
        "--max-surface-distance-m", "0.015",
        "--min-confirmed-votes", "2", "--min-confirmed-ratio", "0.67",
    ], scene_dir)
    gt = scene_dir / "ground_truth/gt.npz"
    run_command(scene, "ground_truth", gt, [
        PYTHON, "-m", "unified_eval.cli", "export-replica-gt",
        "--reference-root", REFERENCE_ROOT, "--scene", scene, "--out", gt,
    ], scene_dir)
    adapter_dir = scene_dir / "evaluation/adapter"
    run_command(scene, "adapter", adapter_dir / "adapter_manifest.json", [
        PYTHON, PACKAGE_ROOT / "tools/adapt_instance_map_replica_ca.py",
        "--instance-surface", p0_dir / "instance_surface.npz",
        "--materialization-report", p0_dir / "materialization_report.json",
        "--association-report", association_dir / "association_report.json",
        "--gt", gt, "--protocol", PROTOCOL, "--output-dir", adapter_dir,
    ], scene_dir)
    metrics_dir = scene_dir / "evaluation/metrics"
    run_command(scene, "metrics", metrics_dir / "metrics.json", [
        PYTHON, "-m", "unified_eval.cli", "eval-scene", "--config", PROTOCOL,
        "--gt", gt, "--pred", adapter_dir / "canonical_prediction.npz",
        "--out", metrics_dir,
    ], scene_dir)
    metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "status": "PASS", "scene_id": scene,
        "AP50": metrics["CA_AP50_uniform"], "AP25": metrics["CA_AP25_uniform"],
        "AP": metrics["CA_AP_uniform"], "PQ": metrics["CA_PQ"]["PQ"],
        "TP": metrics["CA_PQ"]["TP"], "FP": metrics["CA_PQ"]["FP"],
        "FN": metrics["CA_PQ"]["FN"],
    }
    write_json(scene_dir / "scene_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(DATA_ROOT / "run_manifest.json", {
        "status": "RUNNING", "architecture": "P0_no_repair",
        "scenes": args.scenes, "workers": args.workers,
        "protocol": str(PROTOCOL), "ground_truth_used_for_mapping": False,
        "started_unix": time.time(),
    })
    summaries = []
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_scene = {executor.submit(run_scene, scene): scene for scene in args.scenes}
        for future in concurrent.futures.as_completed(future_scene):
            scene = future_scene[future]
            try:
                summaries.append(future.result())
            except Exception as exc:
                failures[scene] = repr(exc)
    summaries.sort(key=lambda x: SCENES.index(x["scene_id"]))
    status = "PASS" if not failures else "FAILED"
    manifest = {
        "status": status, "architecture": "P0_no_repair",
        "scenes": summaries, "failures": failures, "finished_unix": time.time(),
    }
    write_json(DATA_ROOT / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
