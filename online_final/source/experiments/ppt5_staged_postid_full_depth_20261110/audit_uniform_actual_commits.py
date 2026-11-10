from pathlib import Path
import json,hashlib,sys
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110')
scenes=['room0','room1','room2','office0','office1','office2','office3','office4']
def rd(p):return json.loads(p.read_text())
rows=[];pending=[]
for scene in scenes:
 for n,old in [(50,0),(100,50),(200,100),(300,200),(400,300)]:
  root=R/'online'/scene/str(n)/'B';p=root/'repair/commit_complete.json'
  if not p.exists():pending.append([scene,n]);continue
  commit=rd(p);assert commit['status']=='PASS' and commit['processed_frames']==n
  sel=rd(root/'candidates/fixedH/result.json');t=sel['threshold_tuning']
  assert sel['status']=='PASS' and not sel['GT_used']
  assert (t['inherit_dice_strict'],t['min_inherit_support'],t['min_new_support'],t['id_assignment_after_sameview'])==(.25,100,3500,True)
  selected={x['mask_id']:x for x in sel['selected_records']};inherited=[]
  for mid,r in selected.items():
   assert old*5<=r['frame']<n*5 and r['support']>=100
   assert r.get('conflict_winner') is None or r['conflict_winner'] in selected
   if r['id_action']=='inherit':
    assert r['id_candidates'][0]['dice']>.25 and r['id_candidates'][0]['id']==r['target_id'];inherited.append(r['target_id'])
   else:assert r['support']>=3500
  assert len(inherited)==len(set(inherited))
  ids=rd(root/'repair/seed_id_map.json')['objects'];assoc={x['canonical_track_id']:x for x in rd(root/'repair/global_association.json')['objects']}
  assert {x['mask_id'] for x in ids}==set(selected)
  for x in ids:
   r=selected[x['mask_id']];a=assoc[x['track_id']]
   expected='visible_surface_dice' if r['id_action']=='inherit' else 'distinct_new_identity'
   assert a['association']==expected and a['fixedH_id_action']==r['id_action']
   if r['id_action']=='inherit':assert x['persistent_id']==r['target_id']
   tr=rd(root/'tracking_sam3'/x['mask_id']/'status.json')
   assert tr['status']=='PASS' and tr['seed_sha256']==r['mask_sha256'] and tr['seed_frame']==r['frame'] and tr['frame_count']==(n-1)*5+1
  g=rd(root/'geometry_parity_audit.json');assert g['status']=='PASS' and g['same_PPT_full_depth_xyz_and_rgb']
  rows.append(dict(scene=scene,checkpoint=n,selected=len(selected),new=sum(r['id_action']=='new' for r in selected.values()),actual_committed=True,uniform_rule=True))
out=dict(status='PASS_ALL40_ACTUAL_COMMITS' if not pending else 'PASS_COMPLETED_ONLY',completed=len(rows),total=40,pending=pending,nodes=rows,online_thresholds_uniform=True,GT_used_for_filter=False)
(R/'actual_commit_uniformity_audit.json').write_text(json.dumps(out,indent=2))
print({k:v for k,v in out.items() if k!='nodes'})
if '--require-complete' in sys.argv:assert not pending
