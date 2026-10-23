from pathlib import Path
import json
O=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
a,b=[{r['frame']:r for r in read(O/m/'tracking/gate/frame_decisions.json')['frames']} for m in ['reference','checkpoint']]
rows=[]
for fid in [445,640]:
    for field in ['objects','initial_objects']:
        for tid in sorted(a[fid][field]):
            x,y=a[fid][field][tid],b[fid][field][tid]
            if (x.get('accepted'),x.get('reasons'))!=(y.get('accepted'),y.get('reasons')):
                rows.append(dict(frame=fid,stage=field,track_id=tid,left={k:v for k,v in x.items() if k!='other_seed_overlaps'},
                    right={k:v for k,v in y.items() if k!='other_seed_overlaps'},
                    other_seed_overlap_differences=[dict(track_id=p['track_id'],left=p,right=q) for p,q in zip(x.get('other_seed_overlaps',[]),y.get('other_seed_overlaps',[])) if p!=q]))
code=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_checkpoint_closed_loop_20261023/reference/tracking_code/200/A')
r=dict(differences=rows,gate_policy=read(code/'gate_policy.json'))
(O/'gate_difference_diagnosis.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
