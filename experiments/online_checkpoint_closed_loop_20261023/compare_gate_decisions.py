from pathlib import Path
import json
O=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
a,b=[json.loads((O/m/'tracking/gate/frame_decisions.json').read_text()) for m in ['reference','checkpoint']]
rows=[];reasons=[]
for x,y in zip(a['frames'],b['frames']):
    assert x['frame']==y['frame']
    if x['accepted_track_ids']!=y['accepted_track_ids']:rows.append(dict(frame=x['frame'],left=x['accepted_track_ids'],right=y['accepted_track_ids']))
    decision=lambda row:{tid:{k:v.get(k) for k in ['accepted','reasons']} for tid,v in row['objects'].items()}
    if decision(x)!=decision(y):reasons.append(x['frame'])
r=dict(status='PASS' if not rows and not reasons else 'DIFFERENT',mapping_frames=len(a['frames']),
       object_frame_decisions=sum(len(x['objects']) for x in a['frames']),accepted_track_differences=rows,
       object_decision_or_rejection_reason_different_frames=reasons)
(O/'gate_decision_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
