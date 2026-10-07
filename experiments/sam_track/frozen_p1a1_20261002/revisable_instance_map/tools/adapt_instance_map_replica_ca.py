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

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
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
    scene_id = materialization.get("scene_id")
    if scene_id is None:
        config_path = Path(association["config_path"])
        if not config_path.is_absolute():
            config_path = PACKAGE_ROOT / config_path
        scene_id = json.loads(config_path.read_text(encoding="utf-8"))["scene"]
    if gt.scene_id != scene_id or association["frame_count"] != materialization["frame_count"]:
        raise ValueError("Input scene or frame protocol mismatch")
    with np.load(args.instance_surface, allow_pickle=False) as data:
        xyz = data["xyz_m"]
        labels = data["instance_id"]
    if len(xyz) != materialization["tsdf_surface_points"]:
        raise ValueError("Surface point count mismatch")
    code_root = PACKAGE_ROOT
    purpose = materialization.get("purpose", "")
    if purpose == "surface_centric_multiview_instance_evidence_P0":
        materializer_paths = [
            code_root / "tools/materialize_surface_evidence.py",
            code_root / "src/revisable_instance_map/surface_evidence.py",
        ]
    elif purpose == "surface_to_frame_projective_instance_evidence_P0_1":
        materializer_paths = [
            code_root / "tools/materialize_surface_projective_evidence.py",
            code_root / "src/revisable_instance_map/surface_projective_evidence.py",
        ]
    elif purpose == "p0_local_surface_decoder":
        materializer_paths = [
            code_root / "tools/materialize_surface_evidence.py",
            code_root / "src/revisable_instance_map/surface_evidence.py",
            code_root / "tools/decode_p0_surface.py",
            code_root / "src/revisable_instance_map/local_surface_decoder.py",
        ]
    else:
        materializer_paths = [code_root / "tools/materialize_instance_map.py"]
    code_paths = [code_root / "src/revisable_instance_map/association.py", *materializer_paths, Path(__file__)]
    if any(not path.is_file() for path in code_paths):
        raise FileNotFoundError("Materialization code path missing for provenance: " + purpose)
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
