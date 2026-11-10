"""Equivalent parallel query implementation; strict vote semantics are unchanged."""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from strict_votes import BASE,RADIUS,coord_keys,nearest,frame_votes,count_keys,IncrementalVotes as SerialVotes
def cells(xyz):
 q=np.floor(np.asarray(xyz,np.float64)/.03).astype(np.int64)
 if np.any(q<-(1<<20)) or np.any(q>=(1<<20)):raise ValueError('Spatial index range')
 return q
def pack(q):
 return ((q[:,0]+(1<<20))*(1<<21)+q[:,1]+(1<<20))*(1<<21)+q[:,2]+(1<<20)
class IncrementalVotes(SerialVotes):
 def refresh(self,xyz,history):
  fids=tuple(r['frame_id'] for r in history)
  if fids[:len(self.frames)]!=self.frames or len(set(fids))!=len(fids):raise ValueError('History changed or duplicated')
  if any(a>=b for a,b in zip(fids,fids[1:])):raise ValueError('Unordered history')
  xyz=np.asarray(xyz,np.float32)
  common,oi,ni=np.intersect1d(coord_keys(self.xyz),coord_keys(xyz),assume_unique=True,return_indices=True)
  old_only=np.ones(len(self.xyz),bool);new_only=np.ones(len(xyz),bool);old_only[oi]=False;new_only[ni]=False
  changed=np.concatenate((self.xyz[old_only],xyz[new_only]))
  stable=np.zeros(len(xyz),bool);stable[ni]=True
  if len(changed) and len(ni):
   distance=cKDTree(changed).query(xyz[ni],workers=4,distance_upper_bound=np.nextafter(np.nextafter(2*RADIUS,np.inf),np.inf))[0]
   stable[ni[distance<=np.nextafter(2*RADIUS,np.inf)]]=False
  old_to_new=np.full(len(self.xyz),-1,np.int64);keepcommon=stable[ni];old_to_new[oi[keepcommon]]=ni[keepcommon]
  valid=old_to_new[self.keys//BASE]>=0 if len(self.keys) else np.empty(0,bool)
  arrays=[old_to_new[self.keys[valid]//BASE]*BASE+self.keys[valid]%BASE];weights=[self.votes[valid]]
  tree=cKDTree(xyz) if len(xyz) else None
  invalid=np.flatnonzero(~stable);dirty_cells=None
  # A 3 cm cell plus its 26 neighbors conservatively covers a 1.5 cm sphere.
  # The exact distance and surface assignment are still checked afterwards.
  if len(invalid) and len(invalid)<.6*len(xyz):
   q=np.unique(cells(xyz[invalid]),axis=0)
   dirty_cells=np.unique(np.concatenate([pack(q+np.array([x,y,z])) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1]]))
  oldframes=set(self.frames)
  if not hasattr(self,'sample_cells'):self.sample_cells={}
  def project(row):
   if tree is None:return np.empty(0,np.int64)
   old=row['frame_id'] in oldframes
   if old and not len(invalid):return np.empty(0,np.int64)
   selected=None
   if old and dirty_cells is not None:
    fid=row['frame_id']
    if fid not in self.sample_cells:self.sample_cells[fid]=pack(cells(row['world']))
    key=self.sample_cells[fid];pos=np.searchsorted(dirty_cells,key)
    keep=pos<len(dirty_cells);keep[keep]&=dirty_cells[pos[keep]]==key[keep]
    selected=np.flatnonzero(keep)
    if not len(selected):return np.empty(0,np.int64)
   world=row['world'] if selected is None else row['world'][selected]
   ids=row['ids'] if selected is None else row['ids'][selected]
   point,ok=nearest(tree,world)
   if old:ok&=~stable[np.minimum(point,len(xyz)-1)]
   return frame_votes(point[ok],ids[ok])
  with ThreadPoolExecutor(max_workers=8) as pool:
   for keys in pool.map(project,history):
    arrays.append(keys);weights.append(np.ones(len(keys),np.int32))
  keys,votes=count_keys(arrays,weights)
  if len(keys):
   points=keys//BASE;starts=np.r_[0,np.flatnonzero(points[1:]!=points[:-1])+1]
   assert np.max(np.add.reduceat(votes,starts))<=len(history)
  self.xyz=xyz.copy();self.keys=keys;self.votes=votes;self.frames=fids
  self.stats={'stable_points':int(stable.sum()),'replayed_points':int((~stable).sum()),'processed_frames':len(fids)}
  return keys,votes
