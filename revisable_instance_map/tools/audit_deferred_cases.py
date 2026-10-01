#!/usr/bin/env python3
"""Controlled failure injection using the behavioral RGB-D fixtures, never real GT."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from test_deferred_association import make_frame, observations, setup_model, seed
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.association_review import score_complete
from revisable_instance_map.assignment_ledger import ACCEPTED, PENDING


def wrong_score(rows):
    """Inject a near-tied, wrong winning identity; later scores are untouched."""
    right = next(c for c in rows if c["instance_id"] == 2)
    return [{**right,"instance_id":1,"score":.6}, {**right,"score":.59}]


def weak_wrong_winner(injected_count=1):
    split = ((-.15,0.,1),(0.,.15,2))
    frames = [make_frame(0,regions=split),make_frame(5,regions=split)]
    frames += [make_frame(10,regions=((0.,.15,2),)),
               make_frame(15,.025,((0.,.15,2),)),make_frame(20,.05,((0.,.15,2),)),
               make_frame(25,.075,((0.,.15,2),))]
    target = observations(frames[2])[0].observation_id
    injected_keys = {observations(f)[0].observation_id for f in frames[2:2+injected_count]}
    baseline = OnlineVoxelAssociator(association_mode="probabilistic", valid_first_sampling=False,
        allow_multiple_observations_per_instance_per_frame=True)
    original = baseline._score_candidates
    def injected(frame,observation,voxels,**kwargs):
        rows = original(frame,observation,voxels,**kwargs)
        return wrong_score(rows) if observation.observation_id in injected_keys else rows
    baseline._score_candidates = injected
    a2 = []
    for frame in frames:
        a2.extend(baseline.process_frame(frame,observations(frame)))
    results = {}
    for mode in ("B1","B2"):
        model = setup_model(frames,mode)
        seed(model,frames[:2])
        def risky(engine,frame,observation,voxels,parameters,excluded_ids=()):
            scores=score_complete(engine,frame,observation,voxels,parameters,excluded_ids)
            if observation.observation_id in injected_keys:
                scores["candidates"]=wrong_score(scores["candidates"])
            return scores
        with patch("revisable_instance_map.deferred_association.score_complete",risky):
            for frame in frames[2:]:
                model.process_frame(frame,observations(frame))
        model.validate()
        results[mode]={"target_status":model.ledger.assignment_store[target].status,
            "target_id":model.ledger.assignment_store[target].persistent_instance_id,
            "target_decision_frame":model.ledger.assignment_store[target].last_decision_frame_id,
            "wrong_right_object_active_observations":sum(e.instance_id==1 for k,e in model.engine.observation_support.items()
                if model.ledger.raw_support_store[k].observation.mask_local_id==2),
            "future_right_object_ids":[model.ledger.assignment_store[observations(f)[0].observation_id].persistent_instance_id for f in frames[4:]]}
    propagated=[d["instance_id"] for d in a2 if d["frame_id"]>=10]
    assert propagated[:injected_count]==[1]*injected_count
    assert results["B1"]["target_status"] in PENDING
    assert results["B2"]["target_id"]==2
    assert results["B2"]["target_decision_frame"]==10+5*injected_count
    assert results["B2"]["wrong_right_object_active_observations"]==0
    return {"injection_frames":[f.frame_id for f in frames[2:2+injected_count]],
        "injection":"wrong ID1 score.60, correct ID2 score.59; subsequent frames use unmodified scores",
        "synthetic_truth":{"left_object":1,"right_object":2},
        "A2_ids_at_frames10_15_20_25":propagated,
        "A2_wrong_future_decisions":sum(i!=2 for i in propagated[2:]),
        "A2_wrong_observations_retained":sum(e.instance_id==1 and e.frame_id>=10 for e in baseline.observation_support.values()),
        "B1":results["B1"],"B2":results["B2"]}


def mixed_region():
    split=((-.15,0.,1),(0.,.15,2))
    frames=[make_frame(0,regions=split),make_frame(5,regions=split),make_frame(10),
            make_frame(15,.025,split),make_frame(20,.05,split)]
    model=setup_model(frames)
    seed(model,frames[:2])
    for frame in frames[2:]: model.process_frame(frame,observations(frame))
    key=observations(frames[2])[0].observation_id
    assignment=model.ledger.assignment_store[key]
    assert assignment.status in PENDING and key not in model.engine.observation_support
    assert any(t["observation_id"]==key for t in model.repair_tickets)
    return {"mixed_status":assignment.status,"formal_votes_added_by_mixed_mask":0,
        "repair_tickets_for_mixed_mask":sum(t["observation_id"]==key for t in model.repair_tickets)}


def small_birth():
    regions=((-.075,.075,1),)
    frames=[make_frame(0,regions=regions),make_frame(5,regions=regions),make_frame(10,.025,regions)]
    model=setup_model(frames)
    counts=[]
    for frame in frames:
        model.process_frame(frame,observations(frame))
        counts.append(len(model.engine.instances))
    assert counts==[0,0,1]
    assert all(a.status==ACCEPTED for a in model.ledger.assignment_store.values())
    return {"source_pixel_counts":[observations(f)[0].pixel_count for f in frames],
        "formal_instance_counts_after_three_arrivals":counts,"accepted_observations":3}


def systematic_error():
    frames=[make_frame(f,f*.005) for f in (0,5,10,15)]
    model=setup_model(frames)
    seed(model,frames[:2])
    for frame in frames[2:]: model.process_frame(frame,observations(frame))
    assert set(model.engine.instances)=={1}
    return {"no_alternative_raw_hypothesis":True,"still_one_identity":True,
        "corrective_identity_invented":False,"limitation":"consistent wrong frontend masks are outside P1-B historical repair scope"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    report={"status":"PASS","type":"controlled behavioral failure injection, not real-scene accuracy",
        "real_GT_used":False,"single_weak_wrong_winner":weak_wrong_winner(),
        "two_weak_wrong_winners":weak_wrong_winner(2),"mixed_region":mixed_region(),
        "small_cold_start":small_birth(),"systematic_error_boundary":systematic_error()}
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__": main()
