from common100 import *
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from dataclasses import replace
from PIL import Image
import run_repairs as r
from history_core_ops import strict_frame
class Scene100:
 def __init__(self,scene):
  assert scene=='room0'
  assert read(B/'mapping_complete.json')['stopped_before_second_repair']
  self.scene=scene;self.native=B/'baseline/native';self.final_source=B/'baseline/final/instance_surface.npz'
  self.evidence=r.load_npz(B/'baseline/surface_evidence.npz');self.pair=r.load_npz(B/'baseline/surface_instance_frame_votes.npz')
  nm=r.load_npz(self.native/'instance_surface.npz');fm=r.load_npz(self.final_source)
  self.xyz,self.rgb=nm['xyz_m'],nm['rgb'];self.original,self.final=nm['instance_id'],fm['instance_id']
  np.testing.assert_array_equal(self.xyz,fm['xyz_m']);np.testing.assert_array_equal(self.xyz,self.evidence['xyz_m'])
  cloud=o3d.io.read_point_cloud(str(B/'geometry/surface.ply'))
  np.testing.assert_array_equal(np.asarray(cloud.points).astype(np.float32),self.xyz)
  self.normals=np.asarray(cloud.normals).astype(np.float32);self.tree=cKDTree(self.xyz.astype(np.float64))
  self.source=r.ReplicaFrameSource(BASEC/'prefix100/config.json');assert list(self.source.frame_ids)==FIDS
  self.lookup={(x['frame_id'],x['mask_local_id']):x['instance_id'] for x in [json.loads(s) for s in (B/'association/associations.jsonl').read_text().splitlines()]}
  self.commit=read(TOP/'committed50.json');self.journal=Path(self.commit['pixel_journal'])
  self.maximum_id=max(max(self.lookup.values()),int(self.final.max()),107)
  self.records={fid:{'frame_id':fid,'support_file':str(B/'baseline/frame_support'/f'f{fid:06d}.npz')} for fid in FIDS}
  self._support={};self._layers={}
  paths=[self.final_source,self.native/'instance_surface.npz',B/'baseline/surface_evidence.npz',B/'baseline/surface_instance_frame_votes.npz',B/'geometry/surface.ply',B/'association/associations.jsonl',BASEC/'prefix100/config.json',TOP/'committed50.json']
  self.source_hashes={str(p):digest(p) for p in paths}
 def support(self,fid):
  if fid not in self._support:self._support[fid]=r.load_npz(self.records[fid]['support_file'])
  return self._support[fid]
 def identity_layers(self,fid,frame=None):
  if fid not in self._layers:
   frame=frame or self.source.load_frame(fid)
   if fid<=245:
    self._layers[fid]=[np.asarray(Image.open(self.journal/f'f{fid:06d}_{kind}.png'),np.uint16) for kind in ['history','adopted']]
   else:
    table=np.zeros(int(frame.mask_local.max())+1,np.uint16)
    for mid in np.unique(frame.mask_local):
     if mid:table[mid]=self.lookup[(fid,int(mid))]
    self._layers[fid]=[table[frame.mask_local]]
  return self._layers[fid]
 def family_view(self,fid,frame,families):
  z=self.support(fid);present=bool(np.isin(z['candidate_persistent_id'],families).any())
  if fid<=245:
   layers=self.identity_layers(fid,frame)
   foreign=np.logical_or.reduce([(im>0)&~np.isin(im,families) for im in layers])
  else:
   local=z['mask_local_id'];mids=[int(mid) for mid in np.unique(local) if self.lookup[(fid,int(mid))] in families]
   foreign=(frame.mask_local>0)&~np.isin(frame.mask_local,mids)
  return present,foreign
 def old_frame(self,fid,base):
  frame=self.source.load_frame(fid);z=self.support(fid)
  p=z['surface_point_index'];pid=z['candidate_persistent_id']
  return frame,None,strict_frame(np.unique(p.astype(np.int64)*base+pid),base),p
 def verify_unchanged(self):
  for p,h in self.source_hashes.items():assert digest(p)==h,p
