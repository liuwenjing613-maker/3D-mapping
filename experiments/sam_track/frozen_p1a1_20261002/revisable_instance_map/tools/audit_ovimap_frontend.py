#!/usr/bin/env python3
"""Audit complete OVI-MAP front-end run and record immutable provenance."""
import hashlib
import json
from pathlib import Path

CODE=Path("/home/chenkejun/CVPR/revisable_instance_map")
DATA=Path("/data/chenkejun/CVPR/revisable_instance_map")
OFFICIAL=Path("/home/chenkejun/beauty/ovimap_aligned_eval_20260908/official_ovimap")
RUNTIME=Path("/data/chenkejun/ovimap_runtime_20260908")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    depth=DATA/"ovimap_depth_room0_stride5"
    fusion=DATA/"ovimap_refined_room0_stride5"
    observation=DATA/"ovimap_refined_observations_room0_stride5"
    geo=list(lines(depth/"frames.jsonl"))
    frames=list(lines(fusion/"frames.jsonl"))
    parity=read(fusion/"official_fusion_parity.json")
    fusion_report=read(fusion/"refinement_report.json")
    obs_report=read(observation/"observation_manifest.json")
    assert len(geo)==len(frames)==400
    assert all(x["reference_cache_equal"] is True for x in geo)
    assert parity["status"]=="PASS" and parity["frames"]==400
    assert parity["positive_regions"]==fusion_report["totals"]["refined_regions"]==obs_report["observation_count"]==16138
    assert fusion_report["refined_segments_sha256"]==sha(fusion/"segments.jsonl")
    assert fusion_report["refined_frames_sha256"]==sha(fusion/"frames.jsonl")
    assert sum(x["instances"] for x in frames)==16138
    source_ids={x["observation_id"] for x in lines(DATA/"raw_observations_room0_stride5/observations.jsonl")}
    refined_ids={x["observation_id"] for x in lines(observation/"observations.jsonl")}
    assert len(source_ids)==12046 and len(refined_ids)==16138 and source_ids.isdisjoint(refined_ids)
    code_files=[CODE/"tools/build_ovimap_depth_masks.py",CODE/"tools/build_ovimap_refined_masks.py",CODE/"tools/check_ovimap_fusion_parity.py",CODE/"src/revisable_instance_map/ovimap_refinement.py",CODE/"src/revisable_instance_map/frame_io.py",CODE/"src/revisable_instance_map/observations.py",CODE/"tools/build_raw_observations.py",CODE/"src/revisable_instance_map/association.py",CODE/"tools/run_baseline_association.py",CODE/"tools/materialize_instance_map.py",CODE/"tools/adapt_instance_map_replica_ca.py",CODE/"tools/audit_ovimap_frontend.py"]
    source_files=[OFFICIAL/"scripts/utils/common_scannet_nyu.py",OFFICIAL/"mapping_ros_ws/devel/lib/depth_segmentation_py.cpython-38-x86_64-linux-gnu.so",RUNTIME/"models/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth"]
    key_files=[CODE/"configs/replica_room0_stride5.json",CODE/"configs/replica_room0_stride5_ovimap_refined.json",depth/"frames.jsonl",fusion/"frames.jsonl",fusion/"segments.jsonl",fusion/"official_fusion_parity.json",observation/"observations.jsonl"]
    variants={}
    for key,assoc,mat,ev in [
        ("raw","association_full400_local_support","instance_map_full400_no_repair","evaluation_room0_no_repair"),
        ("refined_fixed","association_full400_ovimap_refined","instance_map_full400_ovimap_refined","evaluation_room0_ovimap_refined"),
        ("refined_multi","association_full400_ovimap_refined_multi","instance_map_full400_ovimap_refined_multi","evaluation_room0_ovimap_refined_multi")]:
        a=read(DATA/assoc/"association_report.json")
        m=read(DATA/mat/"materialization_report.json")
        e=read(DATA/ev/"metrics/metrics.json")
        variants[key]={"observation_decisions":a["decision_count"],"persistent_instance_ids":a["instance_count"],"surface_label_fraction":m["labeled_surface_fraction"],"CA_AP50_uniform":e["CA_AP50_uniform"],"CA_PQ":e["CA_PQ"]["PQ"],"TP":e["CA_PQ"]["TP"],"FP":e["CA_PQ"]["FP"],"FN":e["CA_PQ"]["FN"],"association_report_sha256":sha(DATA/assoc/"association_report.json"),"materialization_report_sha256":sha(DATA/mat/"materialization_report.json"),"metrics_sha256":sha(DATA/ev/"metrics/metrics.json")}
    report={
        "status":"PASS",
        "scene":"Replica/room0",
        "frames":400,
        "official_ovimap_revision":"f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424",
        "depth_masks_equal_to_native_cache":400,
        "full_fusion_parity_regions":parity["positive_regions"],
        "source_observations":len(source_ids),
        "refined_observations":len(refined_ids),
        "observation_id_collisions":0,
        "refinement_totals":fusion_report["totals"],
        "variants":variants,
        "code_sha256":{str(p):sha(p) for p in code_files},
        "official_dependency_sha256":{str(p):sha(p) for p in source_files},
        "input_and_output_sha256":{str(p):sha(p) for p in key_files},
        "mapping_uses_ground_truth":False,
        "evaluation_uses_ground_truth":True,
        "local_sync":False,
    }
    path=DATA/"ovimap_frontend_manifest.json"
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)
    print(json.dumps({"status":report["status"],"manifest":str(path),"frames":400,"regions":len(refined_ids)}))

if __name__=="__main__":
    main()
