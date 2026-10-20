"""Adapt the approved r4 unique primaries to E1, preserving every seed pixel."""
from bootstrap import C,R,S,F,digest
from pathlib import Path
from dataclasses import replace
import json,collections,time
import numpy as np
from PIL import Image
import run_repairs as r
import fixed_surface_repair as f

def main():
 assert not (R/'input_freeze.json').exists()
 R.mkdir(parents=True,exist_ok=True)
 src=S/'seed_manifest.json'
 assert digest(src)=='bc02522e9bc83d880e99397ea9154ace33ede26ffa4c6abdb18384d5a68a0b22'
 source=json.loads(src.read_text());groups=source['groups'];assert len(groups)==90
 selection_path=C/'confirmed_selection.json'
 assert digest(selection_path)=='b7188a47abf40974e87f317f3b489d1fe45c4f42fe60bf6b26b9e15196a22ae0'
 selection=json.loads(selection_path.read_text())
 assert selection['status']=='confirmed' and selection['version']=='auto35-20261019' and not selection['qa_only']
 assert selection['manifest_sha256']==digest(src)
 assert digest(Path(selection['selected_source_manifest']))==selection['selected_source_manifest_sha256']
 assert digest(Path(selection['selected_source_decisions']))==selection['selected_source_decisions_sha256']
 chosen=json.loads(Path(selection['selected_source_manifest']).read_text())
 assert {g['id'] for g in chosen['groups']}-set(selection['excluded_this_round'])=={x['group_id'] for x in selection['included']}
 assert selection['pr_filter_applied'] and selection['user_retained_exceptions']==['G017','G025']
 selected={x['original_track_id']:x for x in selection['included']}
 assert len(selected)==35 and len(selection['excluded'])==55
 assert set(selected)|{x['original_track_id'] for x in selection['excluded']}==set(range(1,91))
 group_by_tid={tid:g for tid,g in enumerate(groups,1) if tid in selected}
 for tid,g in group_by_tid.items():
  q=selected[tid];p=g['primary']
  assert (g['id'],p['id'],p['frame'],p['mask_sha256'],p['mask'])==(q['group_id'],q['mask_id'],q['frame'],q['mask_sha256'],q['mask_relative_path'])
 groups=list(group_by_tid.values())

 scene=r.Scene('room2');strictpath=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/final/instance_surface.npz')
 assert digest(strictpath)=='3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3'
 strict=r.load_npz(strictpath);np.testing.assert_array_equal(strict['xyz_m'],scene.xyz)
 scene.final=strict['instance_id'];scene.source_hashes[str(strictpath)]=digest(strictpath)
 batches={};sourcehash={str(src):digest(src),str(selection_path):digest(selection_path),selection['selected_source_manifest']:selection['selected_source_manifest_sha256'],selection['selected_source_decisions']:selection['selected_source_decisions_sha256']};masks={};seedmap=[]
 for tid,g in group_by_tid.items():
  p=g['primary'];path=S/p['mask'];sourcehash[str(path)]=digest(path);assert digest(path)==p['mask_sha256']
  m=np.asarray(Image.open(path))>0;assert m.shape==(680,1200) and int(m.sum())==p['area']
  assert not p['noise_decision']['exclude'] and not p['representative']['touches_crop']
  if any(a['semantic_hit'] for a in p['aliases']):assert p['representative']['model']==2
  masks[tid]=m;slots=batches.setdefault(p['frame'],[])
  for slot in slots:
   if not any((m&masks[q]).any() for q in slot):slot.append(tid);break
  else:slots.append([tid])
  seedmap.append({'track_id':tid,'group_id':g['id'],'mask_id':p['id'],'frame':p['frame'],'source':p})
 manifest=[];diagnostics=[];support={};seedhash={}
 for fid,slots in sorted(batches.items()):
  original=scene.source.load_frame(fid)
  for bi,ids in enumerate(slots,1):
   uid=f'auto35-f{fid:06d}-b{bi:02d}';out=R/'cases'/uid;out.mkdir(parents=True,exist_ok=True)
   selected=np.zeros((680,1200),np.uint16)
   for tid in ids:
    assert not selected[masks[tid]].any();selected[masks[tid]]=tid
   seedfile=out/'prepared_seed.png';Image.fromarray(selected).save(seedfile)
   pts,local,stats=r.project_frame_regions(replace(original,mask_local=selected),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
   f.atomic_npz(out/'seed_projected_support.npz',surface_point_index=pts,track_id=local)
   objects=[];families=set()
   for tid in ids:
    p=group_by_tid[tid]['primary'];m=selected==tid;np.testing.assert_array_equal(m,masks[tid]);sp=np.unique(pts[local==tid]);assert len(sp)>0,(tid,p['id'])
    mids,counts=np.unique(original.mask_local[m],return_counts=True)
    hist=sorted([{'old_local_id':int(mid),'old_persistent_id':int(scene.lookup[(fid,int(mid))]) if mid else -1,'pixels':int(n),'fraction':float(n/m.sum())} for mid,n in zip(mids,counts)],key=lambda x:-x['pixels'])
    positive=[x for x in hist if x['old_persistent_id']>0]
    assert positive,('NO_OLD_OBSERVATION_FAMILY',tid,p['id'])
    family=positive[0]['old_persistent_id'];families.add(family)
    oldids,n=np.unique(scene.final[sp],return_counts=True)
    obj={'track_id':tid,'native_mask_id':int(p['representative']['label']),'mask_id':p['id'],'group_id':group_by_tid[tid]['id'],'model':p['representative']['model'],
     'seed_pixels':int(m.sum()),'projected_seed_points':len(sp),'selection_reverse_projected_points':p['support_count'],'old_family_id':family,'old_pixel_overlap':hist,
     'baseline_seed_support_labels':dict(zip(oldids.astype(str).tolist(),n.tolist())),'seed_xyz_bounds_m':np.stack([scene.xyz[sp].min(0),scene.xyz[sp].max(0)]).tolist()}
    objects.append(obj);support[tid]=sp
    next(x for x in seedmap if x['track_id']==tid).update(case_uid=uid,E1_projected_points=len(sp),old_family_id=family)
   choice={'ROI':','.join(group_by_tid[i]['id'] for i in ids),'frame':fid,'choice':'approved_exact_binary_primary','crop_xyxy':[0,0,1200,680]}
   cfg={'schema_version':1,'scene':'room2','case_uid':uid,'experiment_mode':'offline_retrospective','seed_mode':'new_mask_reassociate','seed_frame':fid,'seed_manifest':str(R/'seed_manifest.json'),
    'objects':[{'track_id':tid} for tid in ids],'old_family_ids':sorted(families),'policy_file':str(C/'tracking_policy.json'),'output_dir':str(out),
    'tracking':{'checkpoint':'/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt','checkpoint_sha256':'2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
     'repo':'/home/chenkejun/CVPR/repos/sam2_1','python':'/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python','cuda_visible_devices':'assigned_by_queue_from_verified_idle_GPU_0_1_2_6','directions':['forward','reverse']}}
   cp=C/'cases'/(uid+'.json');f.atomic_json(cp,cfg)
   f.atomic_json(out/'frozen_association.json',{'status':'PREPARED_FOR_TRACKING_ONLY','scene':'room2','case_uid':uid,'seed_frame':fid,'seed_file':str(seedfile),'seed_sha256':digest(seedfile),'objects':objects,'GT_used':False,'persistent_identity_assignment_pending_global_review':True})
   row={'status':'READY','scene':'room2','case_uid':uid,'source_choice':choice,'objects':objects,'seed_file':str(seedfile.relative_to(R)),'seed_sha256':digest(seedfile)}
   manifest.append(row);diagnostics.append({'case_uid':uid,'source_choice':choice,'objects':objects,'old_family_ids':sorted(families),'projection':stats,'seed_sha256':digest(seedfile),'config_path':str(cp)})
   for path in [cp,seedfile,out/'frozen_association.json',out/'seed_projected_support.npz']:seedhash[str(path)]=digest(path)
 assert sum(len(c['objects']) for c in diagnostics)==35
 authorization={'user_requested_tracking_and_E1_repair':True,'source_manifest':str(src),'source_sha256':digest(src),'selected_primary_masks':35,'confirmed_selection_sha256':digest(selection_path),'confirmed_version':selection['version'],'source_policy':'window/cabinet SAM only; no removed masks restored','batching':'same frame and exactly disjoint seed masks only; overlapping masks run separately','GT_used':False}
 f.atomic_json(R/'inputs/input_selection.json',authorization)
 f.atomic_json(R/'inputs/local_ingest.json',{'no_reliable_seed_cases':[],'unannotated_cases':[],'annotation_sha256':digest(src)})
 f.atomic_json(R/'seed_manifest.json',{'status':'PASS','scene':'room2','annotation_sha256':digest(src),'repair_cases':len(diagnostics),'selected_masks':35,'seeds':manifest,'GT_used':False})
 pairs=[]
 for a,pa in support.items():
  for b,pb in support.items():
   if b<=a:continue
   n=len(np.intersect1d(pa,pb))
   if n:pairs.append({'track_a':a,'track_b':b,'intersection':n,'dice':2*n/(len(pa)+len(pb)),'coverage_a':n/len(pa),'coverage_b':n/len(pb)})
 f.atomic_json(R/'seed_diagnostics.json',{'status':'PASS','GT_used':False,'cases':diagnostics,'cross_case_surface_overlap':pairs})
 f.atomic_json(R/'seed_id_map.json',{'source_manifest':str(src),'objects':seedmap})
 f.atomic_json(R/'alias_review.json',{'status':'FROZEN','GT_used':False,'aliases':{},'rule':'35 retained already-grouped unique primaries; do not merge again','source_manifest_sha256':digest(src)})
 raw=json.loads((r.INPUT/'room2/configs/raw.json').read_text())['source'];rr=Path(raw['scene_root'])
 rgbhash={str(rr/raw['rgb_pattern'].format(frame=fid)):digest(rr/raw['rgb_pattern'].format(frame=fid)) for fid in range(2000)}
 for p in [R/'seed_manifest.json',R/'seed_diagnostics.json',R/'seed_id_map.json',R/'alias_review.json',R/'inputs/input_selection.json']:seedhash[str(p)]=digest(p)
 f.atomic_json(R/'input_freeze.json',{'status':'FROZEN_BEFORE_INFERENCE','original_baseline_source_hashes':scene.source_hashes,'annotation_sha256':digest(src),'raw_rgb_hashes':rgbhash,'seed_and_config_hashes':{**seedhash,**sourcehash},'prepare_code_sha256':digest(__file__),'tracking_code_path':str(C/'track_case.py'),'tracking_code_sha256':digest(C/'track_case.py'),'GT_used':False})
 scene.verify_unchanged()
 summary={'status':'PASS','cases':len(diagnostics),'objects':35,'seed_pixels_exact':True,'min_E1_projected_points':min(len(x) for x in support.values()),'same_frame_overlap_split':True,'windows_cabinets_SAM_only':True}
 f.atomic_json(R/'prepare_complete.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
