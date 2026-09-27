#!/usr/bin/env python3
"""Offline adapter: labeled shared TSDF surface to frozen Replica-CA-v1 vertices.

GT is loaded only here, after mapping has finished. It is a query geometry and
metric target, never an input to observation association or instance fusion.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "/home/chenkejun/CVPR")
from unified_eval.geometry import map_point_labels_to_reference  # noqa: E402
from unified_eval.io import load_gt, save_prediction, sha256_file  # noqa: E402
from unified_eval.schema import Protocol  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-surface", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--association-report", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = Protocol.from_dict(json.loads(args.protocol.read_text(encoding="utf-8")))
    gt = load_gt(args.gt)
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    association = json.loads(args.association_report.read_text(encoding="utf-8"))
    if gt.scene_id != "room0" or association["frame_count"] != materialization["frame_count"]:
        raise ValueError("Input scene or frame protocol mismatch")
    with np.load(args.instance_surface, allow_pickle=False) as data:
        xyz = data["xyz_m"]
        labels = data["instance_id"]
    if len(xyz) != materialization["tsdf_surface_points"]:
        raise ValueError("Surface point count mismatch")
    code_paths = [
        Path("/home/chenkejun/CVPR/revisable_instance_map/src/revisable_instance_map/association.py"),
        Path("/home/chenkejun/CVPR/revisable_instance_map/tools/materialize_instance_map.py"),
        Path(__file__),
    ]
    code_hashes = {str(path): sha256_file(path) for path in code_paths}
    code_digest = hashlib.sha256("".join(code_hashes.values()).encode()).hexdigest()
    result = map_point_labels_to_reference(
        xyz, labels, gt.xyz_ref, protocol.geometry_mapping_max_distance_m,
        scene_id=gt.scene_id,
        method_name="revisable_instance_map_no_repair_baseline",
        method_commit="uncommitted-" + code_digest[:12],
        adapter_version="shared_tsdf_surface_nn_v1",
        protocol_version=protocol.name,
        metadata={
            "source_instance_surface": str(args.instance_surface),
            "source_instance_surface_sha256": sha256_file(args.instance_surface),
            "source_materialization_report": str(args.materialization_report),
            "source_materialization_report_sha256": sha256_file(args.materialization_report),
            "source_association_report": str(args.association_report),
            "source_association_report_sha256": sha256_file(args.association_report),
            "frame_count": int(materialization["frame_count"]),
            "no_repair": True,
            "code_hashes": code_hashes,
            "mapping_uses_gt_labels": False,
        },
    )
    prediction_path = args.output_dir / "canonical_prediction.npz"
    save_prediction(prediction_path, result.prediction)
    write_json(args.output_dir / "adapter_stats.json", result.statistics)
    write_json(args.output_dir / "adapter_manifest.json", {
        "status": "protocol_configured",
        "protocol": protocol.name,
        "protocol_sha256": sha256_file(args.protocol),
        "gt_sha256": sha256_file(args.gt),
        "instance_surface_sha256": sha256_file(args.instance_surface),
        "method_commit": result.prediction.method_commit,
        "code_hashes": code_hashes,
        "peak_process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "total_seconds": time.perf_counter() - started,
        "canonical_prediction": str(prediction_path),
    })
    print(json.dumps({
        "status": "PASS",
        "reference_vertices": gt.vertex_count,
        "native_surface_points": len(xyz),
        "mapped_reference_vertices": result.statistics["mapped_ref_vertices"],
        "mapping_coverage": result.statistics["mapping_coverage"],
        "predicted_instances": len(result.prediction.instances),
        "seconds": time.perf_counter() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
