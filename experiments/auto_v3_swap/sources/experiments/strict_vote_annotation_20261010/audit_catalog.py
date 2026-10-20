from pathlib import Path
import json,hashlib
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
R=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for s in ['room0','room2']:
 p=R/'catalog'/s;m=read(p/'candidate_manifest.json')
 for k in ['surface','evidence','graph','camera_config']:assert sha(Path(m[k+'_file']))==m[k+'_sha256']
 assert sha(p/'membership.npz')==m['membership_sha256']
 with np.load(m['surface_file']) as z:xyz=z['xyz_m'];ids=z['instance_id']
 with np.load(m['evidence_file']) as z:assert np.array_equal(xyz,z['xyz_m']);st=z['state']
 alarm=(ids<=0)&np.isin(st,[0,1,3]);g=load_npz(m['graph_file']);ix=np.flatnonzero(alarm)
 closed=np.union1d(ix,g[ix].indices);n,labels=connected_components(g[closed][:,closed],directed=False)
 ccounts=np.bincount(labels,weights=(alarm[closed]&(st[closed]==3)).astype(np.int64),minlength=n).astype(np.int64)
 expected=sorted(np.flatnonzero(ccounts>0),key=lambda j:(-ccounts[j],j))[:30]
 entries=[e for e in m['entries'] if e['group']=='CONFLICT'];comps={c['uid']:c for c in m['unique_components']}
 with np.load(p/'membership.npz') as z:
  for e,cid in zip(entries,expected):
   c=comps[e['uid']];assert c['parent_component']==cid and e['count']==ccounts[cid]
   k=c['component_index'];core=z['core_point_index'][z['core_offsets'][k]:z['core_offsets'][k+1]]
   assert np.array_equal(np.sort(core),np.sort(closed[(labels==cid)&alarm[closed]]))
   assert hashlib.sha256(core.astype('<i4').tobytes()).hexdigest()==e['core_sha256']
   assert len(e['views'])==3
   for v in e['views']:
    assert sha(p/v['RGB_file'])==v['RGB_sha256'] and sha(p/v['alarm_file'])==v['alarm_sha256']
 old=read(Path('/data/chenkejun/CVPR/revisable_instance_map/roi_selection_cut30_20261006')/s/'candidate_manifest.json')
 oldc=[e for e in old['entries'] if e['group']=='CONFLICT']
 rows.append(dict(scene=s,status='PASS',exact_full_component_ranking=True,source_hashes_verified=True,GT_used=False,baseline_repaired=False,final_unassigned=int(alarm.sum()),top30_conflict_points=sum(e['count'] for e in entries),top5=[e['count'] for e in entries[:5]],same_core_as_old_top30=len({e['uid'] for e in entries}&{e['uid'] for e in oldc}),manifest_sha256=sha(p/'candidate_manifest.json')))
 print(json.dumps(rows[-1]),flush=True)
(R/'catalog_audit.json').write_text(json.dumps(rows,indent=2))
