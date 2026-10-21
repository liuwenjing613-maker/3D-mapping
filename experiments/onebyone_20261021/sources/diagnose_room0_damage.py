"""Post-freeze GT diagnostics only; never imported by the mapper."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_PROTOCOL
from unified_eval.io import load_gt
R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021/automatic/room0')
read=lambda p:json.loads(Path(p).read_text())
result=read(R/'evaluation_r2/comparison.json');src=Path(read(R/'policy.json')['source'])
gt=load_gt(CURRENT_PROTOCOL['canonical_GT']['room0']['file'])
cache=np.load('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache/room0/fixed_gt_to_tsdf_r2.npz')
nearest=cache['nearest_native_index'];inverse=cache['native_unique_index'];allpoints={};report=[]
def counts(a):return {str(int(k)):int(v) for k,v in zip(*np.unique(a,return_counts=True))}
surfaces={v:np.load(result['source_locks'][v+'_final']['source'])['instance_id'] for v in ['baseline','partial','direct','e1']}
for gid in [44,56,80,65]:
 v=np.flatnonzero(gt.valid_vertex_mask&(gt.instance_id==gid));p=nearest[v];p=p[p>=0]
 p=np.flatnonzero(np.isin(inverse,np.unique(inverse[p])))
 allpoints[gid]=p
 report.append(dict(GT_ID=gid,GT_vertices=len(v),TSDF_points=len(p),labels={k:counts(a[p]) for k,a in surfaces.items()},
                    committed_to=Counter(),old_to_new=Counter(),committed_frame_points=0))
base=read(src/'global_association.json')['instance_base'];assoc=read(src/'global_association.json')['objects']
for f in sorted((R/'transactions').glob('f*.npz')):
 with np.load(f) as z:
  committed=z['committed_keys'];old=z['baseline_candidate_keys']
  for row in report:
   points=allpoints[row['GT_ID']];k=committed[np.isin(committed//base,points)]
   row['committed_to'].update(map(int,k%base));row['committed_frame_points']+=len(k)
   for key in k:
    oldids=old[old//base==key//base]%base
    for oldid in oldids:row['old_to_new'][f'{oldid}->{key%base}']+=1
for row in report:
 row['committed_to']=dict(row['committed_to'].most_common());row['old_to_new']=dict(row['old_to_new'].most_common(12))
 targets={int(k) for k in row['committed_to']}
 row['associated_repair_targets']=[a for a in assoc if a['persistent_id'] in targets]
 pids={int(k) for k in row['labels']['baseline'] if int(k)>0}
 row['targets_previously_associated_to_same_ID']=sorted(pids&targets)
out=R.parents[1]/'diagnostics';out.mkdir(exist_ok=True)
(out/'room0_automatic_regressions.json').write_text(json.dumps(dict(status='PASS',GT_used_only_after_prediction_freeze=True,objects=report),indent=2))
print(json.dumps(report))
