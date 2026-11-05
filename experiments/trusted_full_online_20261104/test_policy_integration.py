"""Integration checks for archived partition feedback and checkpoint persistence."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys,tempfile
import numpy as np
C=Path(__file__).resolve().parent
sys.path[:0]=[str(C/'online/room0'),'/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src']
from trusted_online import TrustedPartition
from trusted_partition import partition_candidates,strict_keys
from revisable_instance_map.frame_io import Frame,Camera
from revisable_instance_map.observations import extract_frame_observations,source_pixel_indices
from revisable_instance_map.association import OnlineVoxelAssociator
checks=[]
p=np.arange(40);l=np.ones(40,int);fit=np.full(40,1)
for own,mode in [(np.r_[np.ones(15),np.full(15,2),np.zeros(10)],'multi'),(np.full(40,2),'single_mismatch'),(np.r_[np.ones(39),[2]],'partial_conflict'),(np.zeros(40),'normal')]:
 _,audit=partition_candidates(p,l,own,fit);assert audit[0]['mode']==mode
 checks.append(mode)
# Duplicate projections must not satisfy the ten-point threshold.
_,a=partition_candidates(np.zeros(30,int),np.ones(30,int),np.full(30,2),np.ones(30,int))
assert a[0]['mode']=='partial_conflict' and a[0]['significant']==[]
assert strict_keys([0,0,1,1,2],[1,1,1,2,-1],10).tolist()==[1]
checks.extend(['deduplicated_significance','strict_frame_vote'])
W,H=40,4
camera=Camera(W,H,20.,20.,0.,0.,1000.)
mask=np.ones((H,W),np.uint16);depth=np.ones((H,W),np.float32)
frame=Frame(0,np.zeros((H,W,3),np.uint8),depth,mask,np.eye(4),camera)
v,u=np.indices((H,W));xyz=np.column_stack((u.ravel()/20,v.ravel()/20,np.ones(H*W))).astype(np.float32)
owner=np.where(u.ravel()<W//2,1,2)
policy=TrustedPartition(xyz,[(xyz,owner)])
np.testing.assert_array_equal(policy.owner,owner)
overlap=TrustedPartition(xyz,[(xyz,owner),(xyz[:1],np.array([3]))])
assert overlap.owner[0]==-2 and np.all(overlap.owner[1:]==owner[1:])
checks.append('trusted_overlap_abstention')
assoc=OnlineVoxelAssociator(association_mode='binary-ledger',improved_association=True,valid_first_sampling=True,max_points_per_observation=47)
obs=extract_frame_observations(frame,'Replica','synthetic','a'*64)
decisions=assoc.process_frame(frame,obs);assert [d['instance_id'] for d in decisions]==[1]
layer,audit=policy.apply(frame,obs,decisions,assoc)
assert audit[0]['mode']=='multi'
np.testing.assert_array_equal(layer.ravel(),owner)
assert obs[0].observation_id not in assoc.observation_support
assert set(assoc.instances)=={1,2}
assert len(assoc.observation_support)==2
assoc.identity_evidence.validate()
# Independently reconstruct the source-pixel provenance used by commit_continue.
pix=source_pixel_indices(frame,obs[0]);pix=pix[np.linspace(0,len(pix)-1,num=min(len(pix),assoc.max_points_per_observation),dtype=np.int64)];world=xyz.astype(np.float64)
for oid,entry in assoc.observation_support.items():
 assert oid==obs[0].observation_id+'/partition-id'+str(entry.instance_id)
 chosen=pix[layer.ravel()[pix]==entry.instance_id]
 vox=np.floor(world[chosen]/assoc.voxel_size_m).astype(np.int32)
 # Float32 xyz used only for trustworthy lookup; actual pixel projection is float64.
 vv,uu=np.divmod(chosen,W)
 vox=np.floor(np.column_stack((uu/20,vv/20,np.ones(len(chosen))))/assoc.voxel_size_m).astype(np.int32)
 expected=tuple(tuple(map(int,row)) for row in np.unique(vox,axis=0))
 assert entry.voxels==expected
checks.append('source_pixel_partition_reconstruction')
out=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/policy_tests');out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=out) as td:
 path=Path(td)/'checkpoint.npz';assoc.save_checkpoint(path)
 restored=OnlineVoxelAssociator.from_checkpoint(path)
 assert restored.observation_support==assoc.observation_support
 restored.identity_evidence.validate()
 nextmask=np.where(u<W//2,1,2).astype(np.uint16)
 nextframe=replace(frame,frame_id=5,mask_local=nextmask)
 nextobs=extract_frame_observations(nextframe,'Replica','synthetic','b'*64)
 decisions=restored.process_frame(nextframe,nextobs)
 assert {d['mask_local_id']:d['instance_id'] for d in decisions}=={1:1,2:2},decisions
checks.extend(['checkpoint_roundtrip','next_frame_uses_partitioned_support'])
# Single-mismatch abstention removes the indivisible observation from feedback.
assoc2=OnlineVoxelAssociator(association_mode='binary-ledger',improved_association=True)
obs2=extract_frame_observations(frame,'Replica','synthetic','c'*64)
decisions2=assoc2.process_frame(frame,obs2)
policy2=TrustedPartition(xyz,[(xyz,np.full(len(xyz),2))])
layer2,audit2=policy2.apply(frame,obs2,decisions2,assoc2)
assert audit2[0]['mode']=='single_mismatch' and not np.any(layer2)
assert not assoc2.observation_support and not assoc2.instances
assoc2.identity_evidence.validate()
checks.append('abstention_withdraws_feedback')
report={'status':'PASS','checks':checks,'policy_sha256':{n:hashlib.sha256((C/'online/room0'/n).read_bytes()).hexdigest() for n in ['trusted_online.py','trusted_partition.py']}}
(out/'integration_test.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
