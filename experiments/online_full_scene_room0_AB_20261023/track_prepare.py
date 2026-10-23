"""Freeze exact approved candidates on current post-repair100 geometry; no map changes."""
from bootstrap_continue import *
from dataclasses import replace
import collections,numpy as np
from PIL import Image
import fixed_surface_repair as f
def main():
 tick=time.perf_counter();R.mkdir(parents=True,exist_ok=True)
 if (R/'prepare_complete.json').exists():
  print('PREPARE_ALREADY_COMPLETE');return
 ready=read(CANDIDATE_ROOT/'online_qa.json')
 assert ready.get('status')=='PASS' or ready.get('masks_ready') or ready.get('qa_status')=='PASS',ready
 inputs=read(TRACKROOT/'inputs_complete.json');assert inputs['raw_cutoff']==CURRENT_RAW and inputs['raw_frames']==RAW_COUNT
 src=S/'seed_manifest.json';source=read(src);chosen=read(CHOSEN)
 selected={int(g['id'][1:]):g for g in chosen['groups']}
 original={int(g['id'][1:]):g for g in source['groups']}
 groups={tid:original[tid] for tid in selected};N=len(groups)
 scene=r.Scene('room0');sourcehash={str(src):digest(src),str(CHOSEN):digest(CHOSEN)}
 masks={};batches={};seedmap=[];support={};seedhash={};diagnostics=[];manifest=[]
 for tid,g in groups.items():
  p=g['primary'];assert p['mask_sha256']==selected[tid]['primary']['mask_sha256']
  assert not p['touches_any_edge'] and not p['noise_decision']['exclude']
  assert not p['representative']['touches_crop']
  if any(a['semantic_hit'] for a in p['aliases']):assert p['representative']['model']==2
  fid=p['frame'];assert fid in FIDS and (GROUP=='A' or fid>=OLD_COUNT*5)
  path=S/p['mask'];sourcehash[str(path)]=digest(path);assert digest(path)==p['mask_sha256']
  m=np.asarray(Image.open(path))>0;assert m.shape==(680,1200) and int(m.sum())==p['area']
  assert not (m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
  masks[tid]=m;slots=batches.setdefault(fid,[])
  for slot in slots:
   if not any((m&masks[q]).any() for q in slot):slot.append(tid);break
  else:slots.append([tid])
  seedmap.append(dict(track_id=tid,group_id=g['id'],mask_id=p['id'],frame=fid,source=p))
 for fid,slots in sorted(batches.items()):
  frame=scene.source.load_frame(fid);layers=scene.identity_layers(fid,frame)
  for bi,tids in enumerate(slots,1):
   ids=sorted(tids);uid=f'room0v2-f{fid:06d}-b{bi:02d}';out=R/'cases'/uid;out.mkdir(parents=True,exist_ok=True)
   labels=np.zeros((680,1200),np.uint16)
   for tid in ids:
    assert not labels[masks[tid]].any();labels[masks[tid]]=tid
   seedfile=out/'prepared_seed.png';Image.fromarray(labels).save(seedfile)
   pts,local,stats=r.project_frame_regions(replace(frame,mask_local=labels),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
   f.atomic_npz(out/'seed_projected_support.npz',surface_point_index=pts,track_id=local)
   objects=[];families=set()
   for tid in ids:
    p=groups[tid]['primary'];mask=labels==tid;np.testing.assert_array_equal(mask,masks[tid]);sp=np.unique(pts[local==tid]);assert len(sp)>0
    counts=collections.Counter()
    for layer in layers:
     vals,nums=np.unique(layer[mask],return_counts=True)
     for val,num in zip(vals,nums):
      if val>0:counts[int(val)]+=int(num)
    assert counts,('NO_CURRENT_OBSERVATION_FAMILY',tid,p['id'])
    hist=[dict(old_persistent_id=k,pixels=v,fraction=v/int(mask.sum())) for k,v in sorted(counts.items(),key=lambda x:(-x[1],x[0]))]
    family=hist[0]['old_persistent_id'];families.add(family)
    oldids,n=np.unique(scene.final[sp],return_counts=True)
    obj=dict(track_id=tid,native_mask_id=int(p['representative']['label']),mask_id=p['id'],group_id=groups[tid]['id'],model=p['representative']['model'],
      seed_pixels=int(mask.sum()),projected_seed_points=len(sp),selection_reverse_projected_points=p['support_count'],old_family_id=family,old_pixel_overlap=hist,
      baseline_seed_support_labels=dict(zip(oldids.astype(str).tolist(),n.tolist())),seed_xyz_bounds_m=np.stack([scene.xyz[sp].min(0),scene.xyz[sp].max(0)]).tolist(),
      current_history_source='committed_history_layers' if fid<OLD_COUNT*5 else 'post_repair_online_association',
      seed_support_ID91_points=int((scene.final[sp]==91).sum()))
    objects.append(obj);support[tid]=sp
    next(x for x in seedmap if x['track_id']==tid).update(case_uid=uid,E1_projected_points=len(sp),old_family_id=family)
   choice=dict(ROI=','.join(groups[i]['id'] for i in ids),frame=fid,choice='approved_exact_binary_primary',crop_xyxy=[0,0,1200,680])
   cfg=dict(schema_version=1,scene='room0',case_uid=uid,experiment_mode='online_prefix_bidirectional',seed_mode='new_mask_reassociate',seed_frame=fid,seed_manifest=str(R/'seed_manifest.json'),
     objects=[dict(track_id=tid) for tid in ids],old_family_ids=sorted(families),policy_file=str(C/'tracking_policy.json'),output_dir=str(out),
     tracking=dict(checkpoint='/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt',checkpoint_sha256='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
      repo='/home/chenkejun/CVPR/repos/sam2_1',python=SAM,cuda_visible_devices='assigned_by_queue',directions=['forward','reverse']))
   cp=C/'cases'/(uid+'.json');f.atomic_json(cp,cfg)
   f.atomic_json(out/'frozen_association.json',dict(status='PREPARED_FOR_TRACKING_ONLY',scene='room0',case_uid=uid,seed_frame=fid,seed_file=str(seedfile),seed_sha256=digest(seedfile),objects=objects,GT_used=False))
   fingerprint_data=dict(raw_rgb_sequence_sha256=inputs['raw_rgb_sequence_sha256'],seed_frame=fid,
    ordered_seed_binary_sha256=[hashlib.sha256(np.packbits(labels==i).tobytes()).hexdigest() for i in ids],
    checkpoint_sha256=cfg['tracking']['checkpoint_sha256'],tracking_code_sha256=digest(C/'track_case.py'),sam_source_sha256=inputs['sam_source_sha256'],
    model_config='configs/sam2.1/sam2.1_hiera_l.yaml',directions=['forward','reverse'],raw_frames=RAW_COUNT,seed_mode='new_mask_reassociate',
    torch_seed=0,tf32=True,autocast='bfloat16',apply_postprocessing=False,independent_direction_states=True)
   fingerprint=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   row=dict(case_uid=uid,source_choice=choice,objects=objects,old_family_ids=sorted(families),projection=stats,seed_sha256=digest(seedfile),config_path=str(cp),
    fingerprint=fingerprint,fingerprint_data=fingerprint_data)
   diagnostics.append(row)
   manifest.append(dict(status='READY',scene='room0',case_uid=uid,source_choice=choice,objects=objects,seed_file=str(seedfile.relative_to(R)),seed_sha256=digest(seedfile)))
   for path in [cp,seedfile,out/'frozen_association.json',out/'seed_projected_support.npz']:seedhash[str(path)]=digest(path)
 assert sum(len(x['objects']) for x in diagnostics)==N
 f.atomic_json(R/'seed_manifest.json',dict(status='PASS',scene='room0',seeds=manifest,selected_masks=N,repair_cases=len(diagnostics),GT_used=False))
 f.atomic_json(R/'seed_diagnostics.json',dict(status='PASS',group=GROUP,cases=diagnostics,GT_used=False))
 f.atomic_json(R/'seed_id_map.json',dict(source_manifest=str(CHOSEN),objects=seedmap))
 f.atomic_json(R/'alias_review.json',dict(status='FROZEN',GT_used=False,aliases={},rule='retained already-grouped unique primaries; do not merge again',source_manifest_sha256=digest(CHOSEN)))
 for p in [R/'seed_manifest.json',R/'seed_diagnostics.json',R/'seed_id_map.json',R/'alias_review.json']:seedhash[str(p)]=digest(p)
 f.atomic_json(R/'input_freeze.json',dict(status='FROZEN_BEFORE_INFERENCE',original_baseline_source_hashes=scene.source_hashes,raw_rgb_hashes=inputs['raw_rgb_hashes'],
   seed_and_config_hashes={**seedhash,**sourcehash},tracking_code_path=str(C/'track_case.py'),tracking_code_sha256=digest(C/'track_case.py'),GT_used=False))
 scene.verify_unchanged()
 summary=dict(status='PASS',group=GROUP,cases=len(diagnostics),objects=N,seconds=time.perf_counter()-tick,seed_pixels_exact=True,
  raw_prefix=[0,CURRENT_RAW],mapping_frames=CHECKPOINT,candidate_source_frames=[0,CURRENT_RAW] if GROUP=='A' else [OLD_COUNT*5,CURRENT_RAW],whole_history_used_for_gate=True,
  current_history_state_used=True,repair50_applied=True,repair100_applied=True,current_checkpoint_repaired=False,previous_checkpoint=OLD_COUNT,checkpoint=CHECKPOINT,source_manifest=str(CHOSEN),source_manifest_sha256=digest(CHOSEN))
 dump(R/'prepare_complete.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
