#!/usr/bin/env python3
"""Summarize frozen maps and review logs; this tool never changes assignments."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def recovery(folder):
    arrival={r["observation_id"]:r for r in read_rows(folder/"association/decisions_at_arrival.jsonl")}
    current={r["observation_id"]:r for r in read_rows(folder/"association/associations.jsonl")}
    pending={key for key,r in arrival.items() if r.get("status","").startswith("PENDING")}
    released={key for key in pending if current[key]["status"]=="ACCEPTED"}
    old_bind={key for key in released if current[key]["reason"]=="accepted_bind_independent_raw_witnesses"}
    born={key for key in released if current[key]["reason"]=="confirmed_birth_raw_multiview"}
    last={}
    for row in read_rows(folder/"association/review_log.jsonl"):
        candidates=row["candidate_diagnostics"]
        anchors=defaultdict(set)
        for witness in row["witnesses"]:
            if witness["anchor_instance_id"] is not None:
                anchors[witness["anchor_instance_id"]].add(witness["group_id"])
        last[row["observation_id"]]={"candidate_complete":candidates["candidate_complete"],
            "mixed":row["mixed"],"raw_status_counts":dict(Counter(c["status"] for c in row["raw_checks"])),
            "largest_independent_anchor_group_count":max((len(g) for g in anchors.values()),default=0)}
    blocked=Counter()
    for key,r in current.items():
        if not r["status"].startswith("PENDING"): continue
        review=last.get(key)
        if review is None: reason="no_review_within_budget"
        elif not review["candidate_complete"]: reason="candidate_incomplete"
        elif review["mixed"]: reason="multi_region_contradiction"
        elif r["status"]=="PENDING_BIND" and review["largest_independent_anchor_group_count"]<2:
            reason="fewer_than_two_independent_old_identity_anchors"
        elif r["status"]=="PENDING_BIRTH": reason="birth_raw_evidence_not_confirmed"
        else: reason="other_competing_evidence_or_unfinished_review"
        blocked[reason]+=1
    proxy=json.loads((folder/"release_gt_audit.json").read_text())
    by_kind={}
    for kind,keys in (("old_bind",old_bind),("birth",born)):
        rows=[r for r in proxy["rows"] if r["observation_id"] in keys]
        assessable=[r for r in rows if r["assessable"]]
        correct=sum(r["correct_under_defined_proxy"] for r in assessable)
        by_kind[kind]={"recovered_count":len(keys),"delayed_proxy_rows":len(rows),
            "proxy_assessable":len(assessable),"proxy_correct":correct,
            "proxy_wrong":len(assessable)-correct,
            "proxy_accuracy":correct/len(assessable) if assessable else None,
            "same_frame_review_accept_count":sum(current[k]["last_decision_frame_id"]==current[k]["frame_id"] for k in keys)}
    return {"ever_pending_count":len(pending),"ever_pending_fraction":len(pending)/len(current),
        "final_recovered_from_initial_pending":len(released),
        "initial_pending_recovery_fraction":len(released)/len(pending) if pending else None,
        "initial_pending_still_inactive":len(pending)-len(released),
        "initial_pending_reason_counts":dict(Counter(arrival[k]["decision"] for k in pending)),
        "final_pending_blocker_categories":dict(blocked),"recovery_kind_proxy":by_kind,
        "proxy_limitation":"post-map GT dominance proxy covers delayed releases; same-frame review releases are counted separately, not included in proxy accuracy"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    manifest=json.loads((args.root/"run_manifest.json").read_text())
    if args.require_complete and manifest["status"]!="PASS": raise ValueError("Validation is incomplete")
    result={"status":manifest["status"],"per_scene":[]}
    for scene in manifest["scenes"]:
        for variant in manifest["variants"]:
            folder=args.root/scene/variant
            path=folder/"summary.json"
            if not path.exists(): continue
            summary=json.loads(path.read_text())
            if variant in ("B1","B2"): summary["recovery_audit"]=recovery(folder)
            result["per_scene"].append(summary)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"completed_scene_variants":len(result["per_scene"])}))


if __name__=="__main__": main()
