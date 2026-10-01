#!/usr/bin/env python3
"""Fresh A2/B0/B1/B2, causal prefix, checkpoint, pending and fixed v3 audits."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from run_p1a_validation import ROOT, TOOLS, V3_FLAGS, SCENES, stage, assert_old_fields_equal
from run_baseline_association import sha256_file, write_json
from run_p1a_postprocessing import pool_metrics
from revisable_instance_map.deferred_association import DeferredAssociator
from revisable_instance_map.assignment_ledger import ACCEPTED, PENDING
from revisable_instance_map.frame_io import ReplicaFrameSource
from unified_eval.geometry import map_instances_to_reference_v3
from unified_eval.io import load_gt, save_prediction, load_prediction


def evaluate(scene, variant, path, gt_path, out, code_version, frame_count):
    source_hash = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        xyz, labels = archive["xyz_m"].copy(), archive["instance_id"].copy()
    gt = load_gt(gt_path)
    ids = np.unique(labels[labels > 0])
    order = np.argsort(labels, kind="stable")
    sorted_labels, sorted_xyz = labels[order], xyz[order]
    clouds = [sorted_xyz[np.searchsorted(sorted_labels,k,"left"):np.searchsorted(sorted_labels,k,"right")] for k in ids]
    adapted = map_instances_to_reference_v3(clouds, gt.xyz_ref, .01, scene_id=scene,
        method_name="P1B_"+variant, method_commit=code_version, adapter_version="shared_tsdf_surface_v3",
        protocol_version="Replica-CA-v3", diagnostic_distance_m=.02,
        metadata={"source_map_sha256":source_hash, "source_instance_surface":str(path),
                  "frame_count_from_source":frame_count, "debug_only":True})
    for prediction in (adapted.prediction, adapted.diagnostic_prediction):
        prediction.metadata["source_map_sha256"] = source_hash
        for key, instance in zip(ids, prediction.instances):
            instance.instance_uid = f"p1b-id:{int(key)}"
    adapter_dir = out/"adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_prediction(adapter_dir/"canonical_prediction.npz", adapted.prediction)
    save_prediction(adapter_dir/"diagnostic_support_prediction.npz", adapted.diagnostic_prediction)
    write_json(adapter_dir/"adapter_stats.json", adapted.statistics)
    stage([sys.executable,"-m","unified_eval.cli","eval-scene","--config",
        ROOT/"unified_eval/configs/replica_ca_v3.pending.json", *V3_FLAGS, "--gt",gt_path,
        "--pred",adapter_dir/"canonical_prediction.npz","--diagnostic-pred",adapter_dir/"diagnostic_support_prediction.npz",
        "--out",out/"evaluation"],out/"evaluation.log")
    metrics = json.loads((out/"evaluation/metrics.json").read_text())
    # Small GT definition is fixed and reported; this diagnostic uses the same
    # canonical vertex support as v3 and strict IoU>0.5, with no mapping feedback.
    valid_eval = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    valid = valid_eval & (gt.instance_id>=0)
    gt_ids, sizes = np.unique(gt.instance_id[valid],return_counts=True)
    ignored_small_ids = gt_ids[sizes<100]
    gt_ids, sizes = gt_ids[sizes>=100], sizes[sizes>=100]
    cutoff = float(np.percentile(sizes,25)) if len(sizes) else 0.
    small_ids = gt_ids[sizes<=cutoff]
    matched_small = set()
    for instance in adapted.prediction.instances:
        vertices = instance.vertex_indices
        vertices = vertices[valid_eval[vertices] & ~np.isin(gt.instance_id[vertices],ignored_small_ids)]
        labels_at_vertices = gt.instance_id[vertices]
        for key in small_ids:
            intersection = np.count_nonzero(labels_at_vertices==key)
            total = int(sizes[np.flatnonzero(gt_ids==key)[0]])
            if intersection/(len(vertices)+total-intersection) > .5:
                matched_small.add(int(key))
    return {"AP":metrics["CA_AP_uniform"],"AP50":metrics["CA_AP50_uniform"],
        "PQ":metrics["CA_PQ"]["PQ"],"F1":metrics["CA_PRF1_0_5"]["F1"],
        "diagnostics":metrics["diagnostics"],"evaluation_status":metrics["status"],
        "small_GT_definition":"within-scene bottom quartile by valid GT vertex count, minimum100; strict canonical IoU>0.5",
        "small_GT_vertex_cutoff":cutoff,"small_GT_count":len(small_ids),
        "small_GT_recalled_count":len(matched_small),"small_GT_recall":len(matched_small)/len(small_ids) if len(small_ids) else None}


def delayed_release_gt_audit(model, source_dir, gt_path, out):
    """Offline proxy: raw-region GT dominance versus identity excluding source frame."""
    gt = load_gt(gt_path)
    with np.load(source_dir/"surface_evidence.npz",allow_pickle=False) as data:
        xyz=data["xyz_m"]
    distances, nearest = cKDTree(gt.xyz_ref).query(xyz,workers=8)
    valid = (distances<.01)&gt.valid_vertex_mask[nearest]&~gt.ignore_vertex_mask[nearest]&(gt.instance_id[nearest]>=0)
    labels = np.where(valid,gt.instance_id[nearest],-1)
    manifest = [json.loads(l) for l in (source_dir/"frame_region_support_manifest.jsonl").read_text().splitlines()]
    lookup = {(r.source_frame_id,r.observation.mask_local_id):key for key,r in model.ledger.raw_support_store.items()}
    by_observation, identity_votes, frame_votes = {},defaultdict(Counter),{}
    for row in manifest:
        frame=row["frame_id"]
        if frame>model.engine.last_processed_frame_id:
            continue
        with np.load(row["support_file"],allow_pickle=False) as data:
            points, local = data["surface_point_index"],data["mask_local_id"]
            accepted_by_id=defaultdict(list)
            for mask in np.unique(local):
                key=lookup[(frame,int(mask))]
                vertices=np.unique(points[local==mask])
                ids=labels[vertices]
                by_observation[key]=Counter(int(k) for k in ids if k>=0)
                a=model.ledger.assignment_store[key]
                if a.status==ACCEPTED:
                    accepted_by_id[a.persistent_instance_id].append(vertices)
            for identity,parts in accepted_by_id.items():
                vertices=np.unique(np.concatenate(parts))
                counts=Counter(int(k) for k in labels[vertices] if k>=0)
                frame_votes[(frame,identity)]=counts
                identity_votes[identity].update(counts)
    rows=[]
    for key,a in model.ledger.assignment_store.items():
        if a.status!=ACCEPTED or a.last_decision_frame_id<=model.ledger.raw_support_store[key].source_frame_id:
            continue
        source=model.ledger.raw_support_store[key]
        observation_counts=by_observation.get(key,Counter())
        remaining=identity_votes[a.persistent_instance_id]-frame_votes.get((source.source_frame_id,a.persistent_instance_id),Counter())
        first=observation_counts.most_common(1)
        second=remaining.most_common(1)
        assessable=bool(first and second and sum(observation_counts.values())>=3
            and first[0][1]/sum(observation_counts.values())>=.7 and second[0][1]/sum(remaining.values())>=.7)
        rows.append({"observation_id":key,"source_frame_id":source.source_frame_id,
            "decision_frame_id":a.last_decision_frame_id,"instance_id":a.persistent_instance_id,
            "source_gt_dominant":first[0][0] if first else None,
            "leave_source_frame_out_identity_gt_dominant":second[0][0] if second else None,
            "assessable":assessable,"correct_under_defined_proxy":first[0][0]==second[0][0] if assessable else None})
    assessed=[r for r in rows if r["assessable"]]
    correct=sum(r["correct_under_defined_proxy"] for r in assessed)
    gt_ids, sizes = np.unique(gt.instance_id[gt.valid_vertex_mask & ~gt.ignore_vertex_mask & (gt.instance_id>=0)],return_counts=True)
    gt_ids, sizes = gt_ids[sizes>=100], sizes[sizes>=100]
    small_ids = set(int(x) for x in gt_ids[sizes<=np.percentile(sizes,25)]) if len(sizes) else set()
    small_sources = []
    for key, counts in by_observation.items():
        majority=counts.most_common(1)
        if majority and majority[0][0] in small_ids and majority[0][1]/sum(counts.values())>=.7:
            small_sources.append(key)
    report={"definition":"offline 1cm nearest-GT surface-region dominance >=.7; identity reference excludes entire source frame; diagnostic proxy, not official association ground truth",
        "GT_used_after_completed_map_only":True,"delayed_observations":len(rows),"assessable":len(assessed),
        "correct_under_proxy":correct,"wrong_under_proxy":len(assessed)-correct,
        "unassessable":len(rows)-len(assessed),"proxy_accuracy":correct/len(assessed) if assessed else None,
        "proxy_error_rate":(len(assessed)-correct)/len(assessed) if assessed else None,
        "small_GT_source_observation_count":len(small_sources),
        "small_GT_source_pending_fraction":sum(model.ledger.assignment_store[k].status in PENDING for k in small_sources)/len(small_sources) if small_sources else None,
        "small_GT_source_accepted_fraction":sum(model.ledger.assignment_store[k].status==ACCEPTED for k in small_sources)/len(small_sources) if small_sources else None,
        "source_observations_without_projected_GT_support":len(model.ledger.raw_support_store)-sum(bool(c) for c in by_observation.values()),
        "rows":rows}
    write_json(out,report)
    return {k:v for k,v in report.items() if k!="rows"}


def run(scene,variant,args,code_version):
    base=args.input_root/scene
    out=args.output_root/scene/variant
    out.mkdir(parents=True,exist_ok=False)
    config=base/"configs/p0_parent_regrouped.json"
    catalog=base/"observations/observations.jsonl"
    association=out/"association"
    if variant=="A2":
        command=[sys.executable,TOOLS/"run_baseline_association.py","--association-mode","probabilistic",
                 "--allow-multiple-observations-per-instance-per-frame","--legacy-association-sampling"]
        checkpoint="observation_support_3cm.npz"
        publisher="rebuild_p0_from_regions.py"
    else:
        command=[sys.executable,TOOLS/"run_p1b_association.py","--mode",variant]
        if args.parameters:
            command+= ["--parameters",args.parameters]
        checkpoint="identity_checkpoint.npz"
        publisher="rebuild_p1b_surface.py"
    print(scene,variant,"fresh association",flush=True)
    stage([*command,"--config",config,"--observations",catalog,"--output-dir",association,
           "--frame-count",args.frame_count],out/"association.log")
    print(scene,variant,"native publication",flush=True)
    stage([sys.executable,TOOLS/publisher,"--source-dir",base/"surface_p0","--observations",catalog,
        "--associations",association/"associations.jsonl","--identity-checkpoint",association/checkpoint,
        "--config",config,"--output-dir",out/"surface_p0","--frame-count",args.frame_count],out/"surface.log")
    association_report=json.loads((association/"association_report.json").read_text())
    surface_report=json.loads((out/"surface_p0/materialization_report.json").read_text())
    print(scene,variant,"fixed v3 evaluation",flush=True)
    metrics=evaluate(scene,variant,out/"surface_p0/instance_surface.npz",base/"ground_truth/gt.npz",out/"v3",code_version,args.frame_count)
    summary={"scene":scene,"variant":variant,"frame_count":args.frame_count,
        "association_seconds":association_report["total_seconds"],"published_surface_fraction":surface_report["labeled_surface_fraction"],**metrics}
    if variant!="A2":
        model=DeferredAssociator.from_checkpoint(association/checkpoint,ReplicaFrameSource(config).load_frame)
        model.validate()
        summary.update({k:association_report[k] for k in ("assignment_counts","pending_fraction","small_observation_pending_fraction",
            "accepted_fraction","delayed_accept_count","delayed_only_processing_steps_median_p95","packet_lifecycle_counts")})
        summary.update({k:surface_report[k] for k in ("pending_surface_fraction","protected_surface_fraction","accepted_only_labeled_surface_fraction")})
        with np.load(association/"identity_voxel_counts_3cm.npz") as saved:
            for key,value in model.engine.identity_evidence.export_arrays().items():
                np.testing.assert_array_equal(value,saved[key])
        summary["checkpoint_restore_audit"]=True
        if variant in ("B1","B2"):
            summary["release_gt_proxy"]=delayed_release_gt_audit(model,base/"surface_p0",base/"ground_truth/gt.npz",out/"release_gt_audit.json")
    write_json(out/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("scene","variant","AP50","F1","published_surface_fraction")}),flush=True)
    return summary


def equivalence_audit(scene,args):
    folder=args.output_root/scene
    original=[json.loads(l) for l in (folder/"A2/association/associations.jsonl").read_text().splitlines()]
    actual=[json.loads(l) for l in (folder/"B0/association/decisions_at_arrival.jsonl").read_text().splitlines()]
    assert_old_fields_equal(original,actual)
    for file in ("surface_evidence.npz","surface_instance_frame_votes.npz","instance_surface.npz"):
        with np.load(folder/"A2/surface_p0"/file) as left,np.load(folder/"B0/surface_p0"/file) as right:
            for key in left.files:
                if key not in ("association_decisions_sha256",):
                    np.testing.assert_array_equal(left[key],right[key])
    with np.load(folder/"A2/association/identity_voxel_counts_3cm.npz") as left,np.load(folder/"B0/association/identity_voxel_counts_3cm.npz") as right:
        for key in left.files:
            np.testing.assert_array_equal(left[key],right[key])
    write_json(folder/"B0_equivalence_audit.json",{"status":"PASS","all_old_decision_fields_equal":True,
        "identity_count_arrays_equal":True,"P0_original_arrays_equal":True,"pending_count":0})


def prefix_audit(scene,args):
    base=args.input_root/scene
    folder=args.output_root/scene
    target=folder/"independent_prefix50"
    command=[sys.executable,TOOLS/"run_p1b_association.py","--mode","B2","--config",base/"configs/p0_parent_regrouped.json",
        "--observations",base/"observations/observations.jsonl","--output-dir",target,"--frame-count",50]
    if args.parameters:
        command += ["--parameters",args.parameters]
    stage(command,folder/"independent_prefix50.log")
    for file in ("assignments_at_step050.jsonl",):
        assert (target/file).read_bytes()==(folder/"B2/association"/file).read_bytes()
    with np.load(target/"identity_voxel_counts_3cm.npz") as left,np.load(folder/"B2/association/identity_counts_at_step050.npz") as right:
        for key in right.files:
            np.testing.assert_array_equal(left[key],right[key])
    write_json(folder/"causal_prefix_audit.json",{"status":"PASS","independent_50_frame_assignments_and_counts_equal":True,
        "final_geometry_not_used_by_online_association":True,"online_prefix_surface_quality_not_claimed":True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=Path("/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main"))
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--scenes",nargs="+",choices=SCENES,default=["room0"])
    parser.add_argument("--variants",nargs="+",choices=("A2","B0","B1","B2"),default=["A2","B0","B1","B2"])
    parser.add_argument("--jobs",type=int,default=1)
    parser.add_argument("--frame-count",type=int,default=400)
    parser.add_argument("--parameters",type=Path)
    parser.add_argument("--audit-prefix",action="store_true")
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for pattern in (
        "revisable_instance_map/src/revisable_instance_map/*.py","revisable_instance_map/tools/*p1b*.py") for p in ROOT.glob(pattern)}
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    code_version=commit+"+working-"+__import__("hashlib").sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest={"status":"RUNNING","base_commit":commit,"code_version":code_version,"source_sha256":hashes,
        "scenes":args.scenes,"variants":args.variants,"frame_count":args.frame_count,
        "ground_truth_used_for_mapping":False,"v3_flags":V3_FLAGS,"diffusion_enabled":False,
        "parameter_selection":"fixed development defaults, no per-scene search",
        "parameters_sha256":sha256_file(args.parameters) if args.parameters else None,
        "protocol_sha256":sha256_file(ROOT/"unified_eval/configs/replica_ca_v3.pending.json")}
    write_json(args.output_root/"run_manifest.json",manifest)
    summaries=[]
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures=[pool.submit(run,scene,variant,args,code_version) for scene in args.scenes for variant in args.variants]
            for future in as_completed(futures):
                summaries.append(future.result())
        for scene in args.scenes:
            if "A2" in args.variants and "B0" in args.variants:
                equivalence_audit(scene,args)
            if args.audit_prefix and "B2" in args.variants and args.frame_count>=50:
                prefix_audit(scene,args)
        pooled={variant:pool_metrics(args.output_root,args.scenes,variant,args.input_root,args.output_root/"pooled"/variant) for variant in args.variants}
        if any(sha256_file(ROOT/p)!=digest for p,digest in hashes.items()):
            raise ValueError("Implementation changed during validation; results are not from a frozen revision")
        if args.parameters and sha256_file(args.parameters)!=manifest["parameters_sha256"]:
            raise ValueError("Parameters changed during validation")
    except Exception as error:
        manifest.update(status="FAILED",error=str(error))
        write_json(args.output_root/"run_manifest.json",manifest)
        raise
    manifest["status"]="PASS"
    write_json(args.output_root/"run_manifest.json",manifest)
    write_json(args.output_root/"summary.json",{"status":"PASS","per_scene":summaries,"pooled":pooled,
        "implementation_validation_passed":True,"method_performance_success_requires_B2_vs_B1_analysis":True,
        "current_v3_status":"DEBUG_ONLY / NON_OFFICIAL"})


if __name__=="__main__":
    main()
