"""Build a read-only Chinese inspection UI from existing, verified Replica artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from instance_repair.two_audit_sources import file_record, verify_record, sha256_file
from instance_repair.two_audit_cache import write_json


def build(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).absolute()
    if output.exists():
        raise FileExistsError("Inspection output already exists")
    validation = json.loads((root/"validation.json").read_text())
    if (validation.get("status") != "PASS" or validation.get("dataset") != "Replica"
            or validation.get("scenes") != ["room0", "office2"]):
        raise ValueError("Requires the confirmed Replica pilot")
    records = [file_record(root/"validation.json")]
    scenes = []
    images_to_check = []
    for scene in validation["scenes"]:
        audit = root/"instance_anomalies"/scene
        manifest = json.loads((audit/"manifest.json").read_text())
        records.append(file_record(audit/"manifest.json"))
        for name in ("candidate_pairs.json", "instance_anomalies.json"):
            record = file_record(audit/name)
            if record["sha256"] != manifest["artifact_sha256"][name]:
                raise ValueError("Audit artifact changed: "+name)
            records.append(record)
        pairs = json.loads((audit/"candidate_pairs.json").read_text())
        anomalies = json.loads((audit/"instance_anomalies.json").read_text())
        render = json.loads((audit/"review/render_validation.json").read_text())
        if render.get("status") != "PASS":
            raise ValueError("Review images are incomplete")
        records.append(file_record(audit/"review/render_validation.json"))
        by_key = defaultdict(list)
        for figure in render["figures"]:
            path = audit/"review"/figure["image"]
            if sha256_file(path) != figure["image_sha256"]:
                raise ValueError("Review image changed")
            images_to_check.append(path)
            by_key[figure["pair_key"]].append(dict(
                src="images/"+scene+"/"+Path(figure["image"]).name,
                frame=figure["frame_id"], view_group=figure["joint_view_group_id"]))
        compact = []
        for p in pairs:
            if p["status"] != "POTENTIAL_OVERMERGE":
                continue
            r = p["roi_comparison"]
            compact.append(dict(key=p["pair_key"], old_id=p["target_instance_id"],
                tracks=p["track_ids"], frames=p["joint_frame_count"],
                views=p["independent_joint_view_groups"],
                min_fraction=p["minimum_joint_target_fraction"],
                median_fraction=p["median_joint_target_fraction"],
                roi_outside=r["outside_all_selected_roi_context"],
                roi_relation=r["full_mask_footprint"]["relation"],
                roi_fraction=r["full_mask_footprint"]["context_fraction"],
                core_fraction=r["full_mask_footprint"]["core_fraction"],
                target_roi_fraction=r["target_instance_footprint"]["context_fraction"],
                images=by_key[p["pair_key"]],
                events=[dict(frame=e["frame_id"], view_group=e["joint_view_group_id"],
                    masks=e["mask_local_ids"], counts=e["target_support_points"],
                    totals=e["mask_support_points"], fractions=e["target_support_fractions"])
                    for e in p["joint_observations"]]))
        stage2=json.loads((root/"stable_tracks"/scene/"manifest.json").read_text())
        stage1=json.loads((root/"observations"/scene/"manifest.json").read_text())
        scenes.append(dict(name=scene, frames=400, observations=stage1["totals"]["observations"],
            stable_tracks=stage2["totals"]["stable_tracks"],
            suspected_ids=len(anomalies), pair_count=len(compact),
            insufficient_view_pairs=manifest["totals"]["insufficient_joint_view_pairs"],
            roi_outside_ids=manifest["totals"]["potential_instances_with_outside_roi_pair"],
            roi_outside_pairs=manifest["totals"]["potential_pairs_outside_all_roi_context"],
            exported_images=render["images"], pairs=compact,
            targets=[dict(id=a["target_instance_id"], pair_count=a["pair_count"],
                outside_pairs=a["pairs_outside_all_roi_context"]) for a in anomalies]))
    output.mkdir(parents=True,exist_ok=False)
    (output/"images").mkdir()
    for scene in scenes:
        (output/"images"/scene["name"]).symlink_to(
            root/"instance_anomalies"/scene["name"]/"review/images", target_is_directory=True)
    template = Path(__file__).with_name("two_audit_inspection.html")
    (output/"index.html").write_text(template.read_text(),encoding="utf-8")
    data=dict(dataset="Replica", date="2026-11-04", pipeline_status="PASS",
              human_review_status="未人工判真", scenes=scenes)
    write_json(output/"data.json",data)
    for record in records:
        verify_record(record)
    report=dict(status="PASS",dataset="Replica",scenes=[s["name"] for s in scenes],
        target_count=sum(s["suspected_ids"] for s in scenes),
        candidate_pair_count=sum(s["pair_count"] for s in scenes),
        exported_images=len(images_to_check),
        all_candidate_pairs_listed=True,
        images_are_representative_subset=True,
        original_results_modified=False,
        sources=records, builder=file_record(Path(__file__)),template=file_record(template),
        index=file_record(output/"index.html"),data=file_record(output/"data.json"))
    write_json(output/"page_validation.json",report)
    return {k:report[k] for k in ("status","dataset","target_count","candidate_pair_count","exported_images")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.root,args.output)))
