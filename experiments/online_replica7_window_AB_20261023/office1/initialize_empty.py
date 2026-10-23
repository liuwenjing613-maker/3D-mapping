"""Create a genuine zero-frame association checkpoint, not a repaired history."""
from common_continue import *
import sys,numpy as np
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src')
from revisable_instance_map.association import OnlineVoxelAssociator

def main():
 out=ROOT.parent.parent/'initial'/GROUP
 if (out/'initial_complete.json').exists():return
 out.mkdir(parents=True,exist_ok=True)
 a=OnlineVoxelAssociator(allow_multiple_observations_per_instance_per_frame=True,improved_association=False,
  valid_first_sampling=False,association_mode='probabilistic',relative_weight_temperature=.1,null_candidate_score=None)
 assert a.map_version==0 and a.last_processed_frame_id==-1 and not a.observation_support
 a.save_checkpoint(out/'association_checkpoint.npz');a.save_identity_counts(out/'identity_counts.npz')
 b=OnlineVoxelAssociator.from_checkpoint(out/'association_checkpoint.npz')
 assert b.checkpoint_parameters()==a.checkpoint_parameters() and b.map_version==0 and not b.observation_support
 np.savez_compressed(out/'source_pixel_support.npz',observation_ids=np.array([],dtype='U1'),frame_ids=np.array([],np.int32),
  instance_ids=np.array([],np.int32),layer_index=np.array([],np.int32),offsets=np.array([0],np.int64),
  flat_source_pixel_index=np.array([],np.int32),sampled_voxel_coordinates=np.empty((0,3),np.int32))
 dump(out/'observation_lineage.json',dict(status='PASS',entries=[],state='INITIAL_EMPTY'))
 dump(out/'journal_manifest.json',dict(status='PASS',frames=[],state='INITIAL_EMPTY'))
 dump(out/'initial_audit.json',dict(status='PASS',zero_frame_initialization=True,roundtrip_checkpoint_exact=True,repair_applied=False))
 dump(out/'initial_geometry.json',dict(status='PASS',state='NOT_INTEGRATED',integrated_frames=[]))
 np.savez_compressed(out/'initial_surface.npz',xyz_m=np.empty((0,3),np.float32),rgb=np.empty((0,3),np.float32),instance_id=np.array([],np.int32))
 (out/'pixel_journal').mkdir(exist_ok=True)
 for sub in ['association','inputs/scene']:(out/sub).mkdir(parents=True,exist_ok=True)
 for rel in ['association/associations.jsonl','association/observations.jsonl','inputs/scene/traj.txt']:(out/rel).write_text('')
 paths=dict(checkpoint=out/'association_checkpoint.npz',identity_counts=out/'identity_counts.npz',source_pixel_support=out/'source_pixel_support.npz',
  observation_lineage=out/'observation_lineage.json',journal_manifest=out/'journal_manifest.json',audit=out/'initial_audit.json',
  geometry=out/'initial_geometry.json',final_map=out/'initial_surface.npz')
 row=dict(status='PASS',state='INITIAL_EMPTY',scene=SCENE,group=GROUP,processed_frames=0,last_raw_frame=-1,map_version=0,
  next_instance_id=a.next_instance_id,pixel_journal=str(out/'pixel_journal'),repair_applied=False)
 for k,p in paths.items():row[k]=str(p);row[k+'_sha256']=digest(p)
 dump(out/'initial_complete.json',row)
 print('INITIAL_EMPTY_VERIFIED',SCENE,GROUP)
if __name__=='__main__':main()
