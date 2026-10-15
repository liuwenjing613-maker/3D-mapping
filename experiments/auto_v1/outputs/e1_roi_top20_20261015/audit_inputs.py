from pathlib import Path
import json,hashlib
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from prepare_inputs import ROOT,CATALOG,sha,write,read
M=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
m=read(CATALOG/'candidate_manifest.json')
for k in ['surface','evidence','graph','camera_config']:assert sha(m[k+'_file'])==m[k+'_sha256']
assert sha(CATALOG/'membership.npz')==m['membership_sha256']
freeze=read('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/predictions_freeze.json')['scenes']['room2']['predictions']['strict_baseline_final']
assert m['surface_file']==freeze['path'] and m['surface_sha256']==freeze['sha256']
with np.load(m['surface_file']) as z:xyz=z['xyz_m'];ids=z['instance_id']
with np.load(m['evidence_file']) as z:assert np.array_equal(xyz,z['xyz_m']);states=z['state']
alarm=(ids<=0)&np.isin(states,[0,1,3]);g=load_npz(m['graph_file']);assert g.shape==(len(xyz),len(xyz)) and (g!=g.T).nnz==0
seeds=np.flatnonzero(alarm);closed=np.union1d(seeds,g[seeds].indices)
n,labels=connected_components(g[closed][:,closed],directed=False)
components={c['uid']:c for c in m['unique_components']};selected=[]
with np.load(CATALOG/'membership.npz') as z:
 for group,state in [('CONFLICT',3),('U',0),('T',1)]:
  counts=np.bincount(labels,weights=(alarm[closed]&(states[closed]==state)),minlength=n).astype(np.int64)
  expected=sorted(np.flatnonzero(counts>0),key=lambda i:(-counts[i],i))[:20]
  entries=[e for e in m['entries'] if e['group']==group and e['rank']<=20]
  assert len(entries)==20
  for e,cid in zip(entries,expected):
   c=components[e['uid']];assert c['parent_component']==cid and e['count']==counts[cid]
   k=c['component_index'];core=z['core_point_index'][z['core_offsets'][k]:z['core_offsets'][k+1]]
   np.testing.assert_array_equal(np.sort(core),closed[(labels==cid)&alarm[closed]])
   assert hashlib.sha256(core.astype('<i4').tobytes()).hexdigest()==e['core_sha256']
   assert len(e['views'])==3
   for v in e['views']:
    assert sha(CATALOG/v['RGB_file'])==v['RGB_sha256'] and sha(CATALOG/v['alarm_file'])==v['alarm_sha256']
   selected.append({k:e[k] for k in ['id','uid','group','rank','count','counts']})
architecture=read(M/'instance_repair/configs/current_architecture.json');assert architecture['architecture']=='P1-A1-strict-vote-E1-20261012'
out=dict(status='PASS',scene='room2',baseline_repaired=False,GT_used=False,map_modified=False,exact_full_component_ranking=True,ranked_entries=60,unique_ROIs=len({e['uid'] for e in selected}),groups={g:20 for g in ['CONFLICT','U','T']},entries=selected,source_candidate_sha256=sha(CATALOG/'candidate_manifest.json'),source_surface=m['surface_file'],source_surface_sha256=m['surface_sha256'],source_evidence=m['evidence_file'],source_evidence_sha256=m['evidence_sha256'],architecture=architecture,architecture_sha256=sha(M/'instance_repair/configs/current_architecture.json'),state_definition=m['state_definition'],ranking=m['ranking'],history_views=180,depth_tolerance_m=.02)
write(ROOT/'ranking_audit.json',out);print('RANKING_PASS',out['ranked_entries'],out['unique_ROIs'],flush=True)
