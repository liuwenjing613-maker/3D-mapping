"""Read-only execution of the unchanged metrics, gate, and conflict-abstention functions."""
from pathlib import Path
from dataclasses import replace
import time,copy,json
import numpy as np
from PIL import Image
def evaluate(b,scene,cases,objects,canonical,policy,tracking,out=None,progress=None):
 started=time.perf_counter();records=[];counts=dict(mapping_frames=len(scene.records),accepted_frames_by_track={str(i):0 for i in objects},cross_case_conflict_pixels=0)
 for index,fid in enumerate(sorted(scene.records),1):
  frame=scene.source.load_frame(fid)
  visible={oid:b.c.visible_seed(scene,frame,obj['samples'],policy) for oid,obj in objects.items()}
  labels_by_case={};checks={};accepted=[]
  family_views={uid:scene.family_view(fid,frame,data['config']['old_family_ids']) for uid,data in cases.items()}
  def metric(oid,labels,p,t,uid):
   present,foreign=family_views[uid]
   # metrics_for consumes mask_local only for its diagnostic foreign-family fraction.
   # Encode that exact current-state foreign-pixel union while keeping all other frame data.
   diagnostic_frame=replace(frame,mask_local=foreign.astype(np.uint8))
   m=b.metrics_for(oid,labels==oid,p[t==oid],visible,objects,cases[uid],diagnostic_frame,[],scene,policy)
   ok,reasons=b.object_reliability(m,policy)
   return m,ok,reasons
  for uid,data in cases.items():
   labels=np.asarray(Image.open(tracking[uid][fid]['label_file']),np.uint16)
   assert labels.shape==frame.mask_local.shape and set(np.unique(labels)).issubset({0,*data['ids']})
   labels_by_case[uid]=labels
   p,t,_=b.r.project_frame_regions(replace(frame,mask_local=labels),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
   for oid in data['ids']:
    m,ok,reasons=metric(oid,labels,p,t,uid)
    if not family_views[uid][0]:ok=False;reasons.append('no_identified_old_family_observation')
    checks[str(oid)]={**m,'accepted':ok,'reasons':reasons}
    if ok:accepted.append(oid)
  initial=copy.deepcopy(checks)
  conflict=np.zeros(frame.mask_local.shape,bool)
  merged=np.zeros(frame.mask_local.shape,np.uint16)
  if accepted:
   merged,conflict=b.j.merge_foregrounds([(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted],canonical)
   counts['cross_case_conflict_pixels']+=int(conflict.sum())
   if conflict.any():
    revised=[]
    for uid,data in cases.items():
     labels=labels_by_case[uid].copy();labels[conflict]=0;labels_by_case[uid]=labels
     p,t,_=b.r.project_frame_regions(replace(frame,mask_local=labels),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
     for oid in [x for x in accepted if objects[x]['case_uid']==uid]:
      m,ok,reasons=metric(oid,labels,p,t,uid)
      checks[str(oid)]={**m,'accepted':ok,'reasons':reasons,'rechecked_after_overlap_abstention':True}
      if ok:revised.append(oid)
    accepted=sorted(revised)
    if accepted:merged,_=b.j.merge_foregrounds([(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted],canonical)
    else:merged[:]=0
  for oid in accepted:counts['accepted_frames_by_track'][str(oid)]+=1
  records.append(dict(frame=fid,objects=checks,initial_objects=initial,accepted_track_ids=accepted,
   cross_case_conflict_pixels=int(conflict.sum())))
  if out:
   masks=Path(out)/'accepted_masks';masks.mkdir(parents=True,exist_ok=True)
   Image.fromarray(merged.astype(np.uint16)).save(masks/f'f{fid:06d}.png')
  if progress and index%10==0:progress(index,len(scene.records),time.perf_counter()-started)
 return dict(status='PASS',seconds=time.perf_counter()-started,counts=counts,frames=records,
  evaluation_only=True,repair_applied=False,partial_executed=False,E1_executed=False,state_committed=False)
