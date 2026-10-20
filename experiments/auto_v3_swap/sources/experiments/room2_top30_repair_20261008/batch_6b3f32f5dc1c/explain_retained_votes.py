"""Attribute retained original votes to frame-level rejection or partial support."""
from pathlib import Path
import json
from collections import Counter, defaultdict
import numpy as np

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
OUT=ROOT/'root_cause_20261008'
audit=json.loads((OUT/'root_cause_audit.json').read_text())
provenance=json.loads((OUT/'vote_provenance.json').read_text())
decisions=json.loads((ROOT/'validated/full_track/frame_decisions.json').read_text())
frames={row['frame']:row for row in decisions['frames']}
with np.load(OUT/'newly_gray_diagnostics.npz') as z:
    index=z['surface_point_index']; top1=z['top1_id'];top2=z['top2_id'];baseline=z['baseline_instance_id']
main_new=np.full(audit['points'],-1,np.int32)
main_new[index]=np.where(top1>audit['old_identity_max'],top1,np.where(top2>audit['old_identity_max'],top2,-1))
baseline_id=np.full(audit['points'],-1,np.int32);baseline_id[index]=baseline
by_pid={row['persistent_id']:row for row in audit['tracks'] if row['persistent_id']>audit['old_identity_max']}
counts=Counter();reasons=Counter();by_identity=defaultdict(Counter)
example_frames=defaultdict(list)
for transaction in sorted((ROOT/'validated/full_track/transactions').glob('*.npz')):
    fid=int(transaction.stem[1:]);frame=frames[fid]
    with np.load(transaction) as z:
        retained=z['retained_old_frame_keys'];base=int(z['instance_base'][0]);updated=z['new_frame_keys']
    points=retained//base;ids=retained%base
    take=(main_new[points]>0)&(baseline_id[points]==ids)
    for pid in np.unique(main_new[points[take]]):
        selected=points[take & (main_new[points]==pid)]
        tracks=by_pid[int(pid)]['member_track_ids']
        accepted=any(frame['objects'][str(track)]['accepted'] for track in tracks)
        category='target_accepted_but_baseline_vote_retained' if accepted else 'target_rejected_original_vote_retained'
        counts[category]+=len(selected);by_identity[int(pid)][category]+=len(selected)
        if not accepted:
            failed={reason for track in tracks for reason in frame['objects'][str(track)]['reasons']}
            for reason in failed:reasons[reason]+=len(selected)
            by_identity[int(pid)]['all_rejected_votes_have_insufficient_visible_anchor']+=len(selected) if 'insufficient_visible_anchor' in failed else 0
        for example in provenance['actual_point_examples']:
            point=example['surface_point_index']
            if int(main_new[point])==pid and point in selected:
                example_frames[point].append({'frame':fid,'target_pid':int(pid),'tracks':tracks,'accepted':accepted,
                    'reasons':{str(track):frame['objects'][str(track)]['reasons'] for track in tracks},
                    'new_identity_vote_in_same_frame':bool(point*base+pid in updated)})
result={'status':'PASS','scope':'Retained baseline-identity votes on newly gray points whose top two include a newly created identity; updated frames only',
        'counts':dict(counts),'rejection_reason_vote_counts_nonexclusive':dict(reasons),
        'by_new_identity':{str(pid):dict(c) for pid,c in by_identity.items()},'example_retained_frames':dict(example_frames),
        'predictions_votes_masks_unchanged':True}
(OUT/'retained_vote_explanation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'example_retained_frames':{str(k):{'count':len(v),'accepted':sum(x['accepted'] for x in v),'reason_counts_nonexclusive':dict(Counter(reason for x in v for rr in x['reasons'].values() for reason in rr))} for k,v in example_frames.items()}},indent=2))
