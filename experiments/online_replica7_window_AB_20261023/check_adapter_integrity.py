from pathlib import Path
import hashlib,json,ast
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
S=C.parent/'online_full_scene_room0_AB_20261023'
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/prefix100')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bad=[];checks=0
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 for p in (C/scene).rglob('*.py'):ast.parse(p.read_text())
 for n in [50,100,200,300,400]:
  for g in ['A','B']:
   for rel in ['v2/filter_rules.py','v2/semantic_exemption.py','v1/noise_filter.py','v1/crop_filter.py','v1/source_policy.py']:
    p=C/scene/'selection_code'/str(n)/g/rel;checks+=1
    if sha(p)!=sha(BASE/rel):bad.append(str(p))
   for name in ['track_case.py','gate_v1.py','objectwise_ops.py','strict_vote_ops.py','joint_batch_ops.py','tracking_policy.json','gate_policy.json']:
    p=C/scene/'tracking_code'/str(n)/g/name;checks+=1
    if sha(p)!=sha(S/'tracking_code/400'/g/name):bad.append(str(p))
result=dict(status='FAIL' if bad else 'PASS',checks=checks,mismatched=bad)
Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023/frozen_kernel_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert not bad
