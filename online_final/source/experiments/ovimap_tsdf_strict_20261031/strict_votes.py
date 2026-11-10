"""Strict per-frame surface votes, with exact invalidation when the TSDF surface moves."""
import numpy as np
from scipy.spatial import cKDTree
BASE=1<<32
RADIUS=.015
def coord_keys(x):return np.ascontiguousarray(x,dtype=np.float32).view('V12').ravel()
def nearest(tree,world):
 if not len(world):return np.empty(0,np.int64),np.empty(0,bool)
 # Deterministic exact-distance ties choose the lexicographically first surface point.
 d,i=tree.query(world,k=2,workers=2)
 point=i[:,0].astype(np.int64);valid=np.isfinite(d[:,0])&(d[:,0]<RADIUS)
 tied=np.flatnonzero(valid & (d[:,0]==d[:,1]))
 for j in tied:
  candidates=tree.query_ball_point(world[j],np.nextafter(d[j,0],np.inf))
  ds=np.sum((tree.data[candidates]-world[j])**2,axis=1)
  cs=np.asarray(candidates,np.int64)
  point[j]=int(cs[ds==ds.min()].min())
 return point,valid
def frame_votes(point,ids):
 if not len(point):return np.empty(0,np.int64)
 if np.any(ids<=0) or np.any(ids>=BASE):raise ValueError('Invalid persistent ID')
 keys=np.unique(np.asarray(point,np.int64)*BASE+ids)
 points,n=np.unique(keys//BASE,return_counts=True)
 keys=keys[~np.isin(keys//BASE,points[n>1])]
 assert len(keys)==len(np.unique(keys//BASE))
 return keys
def count_keys(arrays,weights=None):
 if not arrays:return np.empty(0,np.int64),np.empty(0,np.int32)
 keys=np.concatenate(arrays)
 if not len(keys):return keys,np.empty(0,np.int32)
 if weights is None:return np.unique(keys,return_counts=True)
 w=np.concatenate(weights);order=np.argsort(keys,kind='stable');keys=keys[order];w=w[order]
 starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
 return keys[starts],np.add.reduceat(w,starts).astype(np.int32)
def full_replay(xyz,history):
 if not len(xyz):return np.empty(0,np.int64),np.empty(0,np.int32)
 tree=cKDTree(xyz);arrays=[]
 for row in history:
  point,ok=nearest(tree,row['world'])
  arrays.append(frame_votes(point[ok],row['ids'][ok]))
 return count_keys(arrays)
class IncrementalVotes:
 def __init__(self):
  self.xyz=np.empty((0,3),np.float32);self.keys=np.empty(0,np.int64);self.votes=np.empty(0,np.int32);self.frames=()
 def refresh(self,xyz,history):
  fids=tuple(r['frame_id'] for r in history)
  if fids[:len(self.frames)]!=self.frames or len(set(fids))!=len(fids):raise ValueError('History changed or duplicated')
  if any(a>=b for a,b in zip(fids,fids[1:])):raise ValueError('Unordered history')
  xyz=np.asarray(xyz,np.float32)
  oldkey,newkey=coord_keys(self.xyz),coord_keys(xyz)
  common,oi,ni=np.intersect1d(oldkey,newkey,assume_unique=True,return_indices=True)
  old_only=np.ones(len(self.xyz),bool);new_only=np.ones(len(xyz),bool)
  old_only[oi]=False;new_only[ni]=False
  changed=np.concatenate((self.xyz[old_only],xyz[new_only]))
  stable=np.zeros(len(xyz),bool);stable[ni]=True
  if len(changed) and len(ni):
   distance=cKDTree(changed).query(xyz[ni],workers=2)[0]
   stable[ni[distance<=np.nextafter(2*RADIUS,np.inf)]]=False
  old_to_new=np.full(len(self.xyz),-1,np.int64)
  keepcommon=stable[ni];old_to_new[oi[keepcommon]]=ni[keepcommon]
  valid=old_to_new[self.keys//BASE]>=0 if len(self.keys) else np.empty(0,bool)
  arrays=[old_to_new[self.keys[valid]//BASE]*BASE+self.keys[valid]%BASE];weights=[self.votes[valid]]
  tree=cKDTree(xyz) if len(xyz) else None
  for row in history:
   # Full historical reprojection is deliberately the safe first implementation.
   # Only invalidated points replace old counts; stable points add new frames once.
   if tree is None:continue
   point,ok=nearest(tree,row['world'])
   if row['frame_id'] in self.frames:ok&=~stable[np.minimum(point,len(xyz)-1)]
   keys=frame_votes(point[ok],row['ids'][ok])
   arrays.append(keys);weights.append(np.ones(len(keys),np.int32))
  keys,votes=count_keys(arrays,weights)
  if len(keys):
   points=keys//BASE;starts=np.r_[0,np.flatnonzero(points[1:]!=points[:-1])+1]
   assert np.max(np.add.reduceat(votes,starts))<=len(history)
  self.xyz=xyz.copy();self.keys=keys;self.votes=votes;self.frames=fids
  self.stats={'stable_points':int(stable.sum()),'replayed_points':int((~stable).sum()),'processed_frames':len(fids)}
  return keys,votes
 def save(self,path):
  np.savez_compressed(path,xyz=self.xyz,keys=self.keys,votes=self.votes,frames=np.asarray(self.frames,np.int32))
 @classmethod
 def load(cls,path):
  obj=cls()
  with np.load(path,allow_pickle=False) as z:
   obj.xyz=z['xyz'];obj.keys=z['keys'];obj.votes=z['votes'];obj.frames=tuple(int(x) for x in z['frames'])
  return obj
def oracle(xyz,history):
 # Independent tiny-scene reference: brute-force distance, sets and dictionaries.
 from collections import Counter
 counts=Counter()
 for row in history:
  candidates={}
  for p,ident in zip(row['world'],row['ids']):
   if not len(xyz):continue
   d=np.sqrt(np.sum((xyz.astype(np.float64)-p)**2,axis=1));i=int(np.argmin(d))
   if d[i]<RADIUS:candidates.setdefault(i,set()).add(int(ident))
  for i,ids in candidates.items():
   if len(ids)==1:counts[(i,next(iter(ids)))]+=1
 keys=np.asarray(sorted(i*BASE+j for i,j in counts),np.int64)
 return keys,np.asarray([counts[(int(k//BASE),int(k%BASE))] for k in keys],np.int32)
