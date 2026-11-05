"""Online trusted-region partition, operating solely on committed prefix geometry."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from revisable_instance_map.observations import source_pixel_indices
from revisable_instance_map.surface_evidence import project_frame_regions

class TrustedPartition:
 def __init__(self,xyz,regions):
  self.xyz=np.asarray(xyz,dtype=np.float32); self.tree=cKDTree(self.xyz) if len(self.xyz) else None
  # Query each owner separately: no finite-neighbour truncation can hide a conflict.
  self.owner=np.full(len(self.xyz),-1,np.int32)
  for region_xyz,owner in regions:
   owner=np.asarray(owner);region_xyz=np.asarray(region_xyz)
   for pid in np.unique(owner[owner>0]):
    d,_=cKDTree(region_xyz[owner==pid]).query(self.xyz,workers=4,distance_upper_bound=.015)
    found=d<.015
    self.owner[found & (self.owner==-1)]=pid
    self.owner[found & (self.owner>0) & (self.owner!=pid)]=-2
 def pixel_projection(self,frame,pixels):
  pixels=np.asarray(pixels,dtype=np.int64); y,x=np.divmod(pixels,frame.camera.width); z=frame.depth_m.ravel()[pixels].astype(np.float64)
  valid=np.isfinite(z)&(z>0)&(z<10)
  owners=np.full(len(pixels),-1,np.int32)
  if self.tree is None or not np.any(valid): return owners
  cam=np.column_stack(((x[valid]-frame.camera.cx)*z[valid]/frame.camera.fx,(y[valid]-frame.camera.cy)*z[valid]/frame.camera.fy,z[valid]))
  world=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
  d,i=self.tree.query(world,workers=4); out=np.where(d<.015,self.owner[i],-1)
  owners[valid]=out; return owners
 def apply(self,frame,observations,decisions,associator):
  from revisable_instance_map.observations import source_pixel_indices
  from trusted_partition import partition_candidates
  lookup={int(d['mask_local_id']):int(d['instance_id']) for d in decisions}
  table=np.zeros(int(frame.mask_local.max())+1,np.uint16)
  for lid,pid in lookup.items():
   if pid>65535: raise ValueError('Persistent ID exceeds journal representation')
   table[lid]=pid
  image=table[frame.mask_local]; audits=[]
  if self.tree is None: return image,audits
  pts,lids,_=project_frame_regions(frame,self.tree,pixel_stride=2,max_distance_m=.015,workers=4)
  fits=np.asarray([lookup[int(i)] for i in lids],dtype=np.int64)
  _,audits=partition_candidates(pts,lids,self.owner[pts],fits)
  byid={a['mask_local_id']:a for a in audits}
  for obs in observations:
   a=byid.get(obs.mask_local_id)
   if not a or a['mode']=='normal':continue
   pix=source_pixel_indices(frame,obs)
   if a['mode']=='single_mismatch': image.ravel()[pix]=0
   elif a['mode']=='partial_conflict':
    owner=self.pixel_projection(frame,pix)
    conflict=((owner>0)&(owner!=a['fitted_id']))|(owner==-2)
    image.ravel()[pix[conflict]]=0
   else:
    owner=self.pixel_projection(frame,pix); owner[~np.isin(owner,a['significant'])]=0
    image.ravel()[pix]=np.maximum(owner,0).astype(np.uint16)
   # Remove the indivisible proposal committed by the legacy associator, then add
   # per-owner pieces based on the original sampling, before processing next frame.
   old=associator.observation_support[obs.observation_id]
   associator.identity_evidence._remove_votes(old); del associator.observation_support[obs.observation_id]
   pix=source_pixel_indices(frame,obs)
   if associator.valid_first_sampling:
    dep=frame.depth_m.ravel()[pix];pix=pix[np.isfinite(dep)&(dep>0)&(dep<10)]
   n=min(len(pix),associator.max_points_per_observation)
   pix=pix[np.linspace(0,len(pix)-1,num=n,dtype=np.int64)] if n else pix
   y,x=np.divmod(pix,frame.camera.width);z=frame.depth_m.ravel()[pix].astype(np.float64)
   good=np.isfinite(z)&(z>0)&(z<10);pix=pix[good];y=y[good];x=x[good];z=z[good]
   cam=np.column_stack(((x-frame.camera.cx)*z/frame.camera.fx,(y-frame.camera.cy)*z/frame.camera.fy,z))
   world=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
   vox=np.floor(world/associator.voxel_size_m).astype(np.int32)
   pids=image.ravel()[pix]
   for pid in np.unique(pids[pids>0]):
    part=replace(obs,observation_id=obs.observation_id+'/partition-id'+str(int(pid)))
    cells=tuple(tuple(int(v) for v in row) for row in np.unique(vox[pids==pid],axis=0))
    associator._commit(int(pid),part,cells)
  associator._rebuild_instance_index()
  return image,audits
