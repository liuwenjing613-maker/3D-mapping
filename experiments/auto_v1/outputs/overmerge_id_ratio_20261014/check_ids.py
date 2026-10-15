from pathlib import Path
import numpy as np
b=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain')
with np.load(b/'final/instance_surface.npz') as s,np.load(b/'surface_evidence.npz') as e:
 m=(e['state']==2)&(e['top1_instance_id']>0)
 print('confirmed',int(m.sum()),'equal native',int((s['instance_id'][m]==e['top1_instance_id'][m]).sum()))
 print('native IDs',s['native_instance_ids'].tolist())
 print('confirmed pair IDs',np.unique(np.column_stack([s['instance_id'][m],e['top1_instance_id'][m]]),axis=0).tolist())
