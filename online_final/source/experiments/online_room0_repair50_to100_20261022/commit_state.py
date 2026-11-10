from bootstrap import *
from dataclasses import replace
import numpy as np
from PIL import Image
import time,traceback
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.identity_evidence import ObservationSupport,FrameIdentityEvidence
from revisable_instance_map.observations import RawInstanceObservation,source_pixel_indices,extract_frame_observations
from history_core_ops import strict_frame
import run_history_core as h
def sample_pixels(frame,observation,associator):
 pix=source_pixel_indices(frame,observation)
 if associator.valid_first_sampling:
  dep=frame.depth_m.ravel()[pix];pix=pix[np.isfinite(dep)&(dep>0)&(dep<10)]
  if not len(pix):return pix,np.empty((0,3),np.int32)
 n=min(len(pix),associator.max_points_per_observation)
 pix=pix[np.linspace(0,len(pix)-1,num=n,dtype=np.int64)]
 yy,xx=np.divmod(pix,frame.camera.width);z=frame.depth_m.ravel()[pix].astype(np.float64)
 ok=np.isfinite(z)&(z>0)&(z<10);pix=pix[ok];yy=yy[ok];xx=xx[ok];z=z[ok]
 cam=np.column_stack(((xx-frame.camera.cx)*z/frame.camera.fx,(yy-frame.camera.cy)*z/frame.camera.fy,z))
 xyz=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
 return pix,np.floor(xyz/associator.voxel_size_m).astype(np.int32)
def vt(x):return tuple(tuple(int(v) for v in row) for row in np.unique(x,axis=0))
def main():
 start=time.perf_counter();assert read(R/'repair_complete.json')['status']=='PASS'
 scene=r.Scene('room0');a=OnlineVoxelAssociator.from_checkpoint(B/'association/observation_support_3cm.npz')
 assert a.last_processed_frame_id==245;a.identity_evidence.validate()
 params=a.checkpoint_parameters();old_version=a.map_version;old_next=a.next_instance_id
 originals=[RawInstanceObservation(**json.loads(line)) for line in (B/'association/observations.jsonl').read_text().splitlines()]
 byframe={}
 for ob in originals:byframe.setdefault(ob.frame_id,[]).append(ob)
 entries=[];events=[];receipts=[];untouched_ids=[];revised_ids=[];added_records=[];all_frame_keys=[]
 base=read(R/'global_association.json')['instance_base']
 for fid in FIDS:
  frame=scene.source.load_frame(fid);journal=R/'commit_candidate/pixel_journal'
  historypath=journal/f'f{fid:06d}_history.png';adoptpath=journal/f'f{fid:06d}_adopted.png'
  history=np.asarray(Image.open(historypath));adopted=np.asarray(Image.open(adoptpath))
  hp,hi=h.project(frame,history,scene);np_,ni=h.project(frame,adopted,scene)
  raw=np.union1d(h.keys(hp,hi,base),h.keys(np_,ni,base));eff=strict_frame(raw,base)
  saved=r.load_npz(R/'partial/transactions'/f'f{fid:06d}.npz')
  np.testing.assert_array_equal(raw,saved['whole_to_partial_candidate_keys']);np.testing.assert_array_equal(eff,saved['whole_to_partial_frame_keys'])
  all_frame_keys.append(eff);frame_changed=0
  for ob in byframe[fid]:
   old=a.observation_support[ob.observation_id];pix,vox=sample_pixels(frame,ob,a)
   assert vt(vox)==old.voxels,(ob.observation_id,'sampling differs from original checkpoint')
   ids=history.ravel()[pix]
   if np.all(ids==old.instance_id):
    entries.append(old);untouched_ids.append(old.observation_id);continue
   frame_changed+=1;revised_ids.append(old.observation_id)
   pieces=[]
   for pid in sorted(np.unique(ids[ids>0]).tolist()):
    subvox=vt(vox[ids==pid]);suffix='' if pid==old.instance_id else f'/repair50-id{pid}'
    new=replace(old,observation_id=old.observation_id+suffix,instance_id=int(pid),voxels=subvox,assignment_version=old.assignment_version+1)
    entries.append(new);pieces.append(dict(observation_id=new.observation_id,instance_id=int(pid),voxels=len(subvox),sample_pixels=int((ids==pid).sum())))
   events.append(dict(event_id=len(events)+1,type='PIXEL_PARTIAL_E1',observation_id=old.observation_id,frame_id=fid,old_instance_id=old.instance_id,new_instance_id=int(pieces[0]['instance_id']) if pieces else old.instance_id,old_assignment_version=old.assignment_version,new_assignment_version=old.assignment_version+1,map_version_before=old_version,map_version_after=old_version+1,pieces=pieces,withdrawn_sample_pixels=int((ids==0).sum()),reason='approved50_partial_E1_source_pixel_journal'))
  adopted_frame=replace(frame,mask_local=adopted)
  newobs=extract_frame_observations(adopted_frame,'Replica','room0',digest(adoptpath),observation_namespace='repair50')
  for ob in newobs:
   vox=a._sample_observation_voxels(adopted_frame,ob);pid=ob.mask_local_id
   entries.append(ObservationSupport(ob.observation_id,fid,ob.source_mask_sha256,pid,vox,1))
   added_records.append(dict(observation_id=ob.observation_id,frame_id=fid,instance_id=pid,source_mask=str(adoptpath),source_mask_sha256=ob.source_mask_sha256,voxels=len(vox)))
  receipts.append(dict(frame=fid,revised_original_observations=frame_changed,added_observations=len(newobs),surface_candidates_match=True,strict_frame_votes_match=True))
 old_entries=a.observation_support
 a.identity_evidence=FrameIdentityEvidence.rebuild(entries);a.observation_support=a.identity_evidence.entries
 a._rebuild_instance_index();a.map_version=old_version+1;a.revision_events.extend(events)
 reserved=[x['persistent_id'] for x in read(R/'global_association.json')['objects']]
 a.next_instance_id=max(old_next,max([*a.instances,*reserved])+1)
 assert a.last_processed_frame_id==245 and a.checkpoint_parameters()==params
 for obsid in untouched_ids:assert a.observation_support[obsid]==old_entries[obsid]
 a.identity_evidence.validate()
 out=R/'commit_candidate';out.mkdir(exist_ok=True)
 a.save_checkpoint(out/'association_checkpoint.npz');a.save_identity_counts(out/'identity_counts.npz')
 restored=OnlineVoxelAssociator.from_checkpoint(out/'association_checkpoint.npz')
 assert restored.observation_support==a.observation_support and restored.next_instance_id==a.next_instance_id
 restored.identity_evidence.validate()
 assert restored.voxel_to_instances==a.voxel_to_instances
 k,v=np.unique(np.concatenate(all_frame_keys),return_counts=True)
 finalpath=R/'partial/whole_to_partial/room0'
 expected=r.load_npz(finalpath/'native/surface_instance_frame_votes.npz');ek,ev=h.encode(expected,base)
 np.testing.assert_array_equal(k,ek);np.testing.assert_array_equal(v,ev)
 evidence=r.reduce_surface_votes(k,v,len(scene.xyz),base,2,.67);saved=r.load_npz(finalpath/'native/surface_evidence.npz')
 for field in r.FIELDS:np.testing.assert_array_equal(evidence[field],saved[field])
 journal_hashes={str(p):digest(p) for p in (out/'pixel_journal').glob('*.png')}
 report=dict(status='PASS',trigger_processed_frames=50,last_raw_frame=245,seconds=time.perf_counter()-start,old_map_version=old_version,map_version=a.map_version,next_instance_id=a.next_instance_id,old_next_instance_id=old_next,unchanged_original_observations=len(untouched_ids),revised_original_observations=len(revised_ids),added_observations=len(added_records),identity_entries=len(a.observation_support),roundtrip_exact=True,unchanged_supports_preserved=True,surface_journal_reproduces_all_candidates_and_votes=True,strict_max_one_vote_per_frame_surface_point=True,geometric_indices_not_used_as_persistent_keys=True,source_pixel_journal_sha256=journal_hashes,association_checkpoint_sha256=digest(out/'association_checkpoint.npz'),final_map=str(finalpath/'final/instance_surface.npz'),final_map_sha256=digest(finalpath/'final/instance_surface.npz'),GT_used=False,frames=receipts)
 dump(out/'added_observations.json',added_records);dump(out/'commit_audit.json',report)
 # Publish only after every consistency check succeeds; continuation requires this pointer.
 dump(TOP/'committed50.json',dict(status='COMMITTED',checkpoint=str(out/'association_checkpoint.npz'),checkpoint_sha256=report['association_checkpoint_sha256'],pixel_journal=str(out/'pixel_journal'),audit=str(out/'commit_audit.json'),audit_sha256=digest(out/'commit_audit.json'),last_raw_frame=245,processed_frames=50,map_version=a.map_version,final_map=report['final_map'],final_map_sha256=report['final_map_sha256'],geometry=str(B/'geometry/tsdf_grid.npz'),geometry_sha256=digest(B/'geometry/tsdf_grid.npz')))
 print(json.dumps({k:report[k] for k in ('status','seconds','unchanged_original_observations','revised_original_observations','added_observations','next_instance_id')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'commit_failure.json',dict(traceback=traceback.format_exc()));raise
