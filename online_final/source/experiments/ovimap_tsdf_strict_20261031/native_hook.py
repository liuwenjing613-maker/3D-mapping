from pathlib import Path
import hashlib,json,os,time
import numpy as np
MODE=os.environ.get('CVPR_READOUT_MODE','trace')
COUNT=int(os.environ['CVPR_EXPECTED_COUNT'])
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,data):
 path=Path(path);q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(data,indent=2));q.replace(path)
def save(path,**d):
 path=Path(path);q=path.with_suffix('.tmp.npz');np.savez_compressed(q,**d);q.replace(path)
def on_frame(node,root,fid,loader,K,raw_mask):
 root=Path(root);count=fid//5+1
 if MODE=='terminal' and count!=COUNT:return
 start=time.perf_counter()
 root.joinpath('geometry').mkdir(exist_ok=True);root.joinpath('ovi').mkdir(exist_ok=True)
 before=node.cvprTsdfState()
 hashes={k:digest(v) for k,v in before.items()}
 if count==COUNT:save(root/'tsdf_final.npz',**before)
 del before
 raw=node.cvprSurface()
 xyz,index,inverse=np.unique(raw['xyz'],axis=0,return_index=True,return_inverse=True)
 ids=np.asarray(raw['instance_id'],np.int32)
 inventory=np.unique(ids[ids>0])
 lo=np.full(len(xyz),np.iinfo(np.int32).max,np.int32);hi=np.full(len(xyz),-1,np.int32)
 np.minimum.at(lo,inverse,ids);np.maximum.at(hi,inverse,ids)
 owner=np.where((lo==hi)&(lo>0),lo,-1).astype(np.int32)
 after=node.cvprTsdfState()
 for k,v in after.items():assert digest(v)==hashes[k],('readout_mutated_state',fid,k)
 del after
 save(root/'geometry'/('f%06d.npz'%fid),xyz_m=xyz)
 save(root/'ovi'/('f%06d.npz'%fid),xyz_m=xyz,instance_id=owner,native_instance_ids=inventory)
 rgb,depth,pose=loader.getDataFromIndex(fid)
 row={'frame_id':fid,'processed_frames':count,'tsdf_sha256':hashes,'geometry_sha256':digest(xyz),
  'points':len(xyz),'raw_mesh_entries':len(ids),'conflicting_coordinate_groups':int(np.sum(lo!=hi)),
  'native_inventory':len(inventory),'readonly_verified':True,'geometry_source':'native_OVI_TSDF_current_prefix',
  'depth_decoded_sha256':digest(depth),'mask_decoded_sha256':digest(raw_mask.astype(np.int32)),
  'pose':pose.tolist(),'intrinsics':K.tolist(),'seconds':time.perf_counter()-start}
 with (root/'prefix_audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 write(root/'progress.json',row)
 print('CVPR_PREFIX',count,len(xyz),round(row['seconds'],3),flush=True)
 if count==COUNT:write(root/'readout_complete.json',{'status':'PASS','processed_frames':count,'mode':MODE})
