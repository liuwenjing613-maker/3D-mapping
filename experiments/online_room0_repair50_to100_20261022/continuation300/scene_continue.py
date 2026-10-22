"""Read the current branch's causal map and all committed pixel identity layers."""
from common_continue import *
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image
import run_repairs as r
from history_core_ops import strict_frame
class SceneContinue:
 def __init__(self,scene):
  assert scene=='room0'
  done=read(B/'mapping_complete.json');assert done['status']=='PASS' and done['frame_count']==CHECKPOINT
  self.scene=scene;self.native=B/'baseline/native';self.final_source=B/'baseline/final/instance_surface.npz'
  self.evidence=r.load_npz(B/'baseline/surface_evidence.npz');self.pair=r.load_npz(B/'baseline/surface_instance_frame_votes.npz')
  nm=r.load_npz(self.native/'instance_surface.npz');fm=r.load_npz(self.final_source)
  self.xyz,self.rgb=nm['xyz_m'],nm['rgb'];self.original,self.final=nm['instance_id'],fm['instance_id']
  np.testing.assert_array_equal(self.xyz,fm['xyz_m']);np.testing.assert_array_equal(self.xyz,self.evidence['xyz_m'])
  np.testing.assert_array_equal(self.rgb,fm['rgb']);np.testing.assert_array_equal(self.rgb,self.evidence['rgb'])
  cloud=o3d.io.read_point_cloud(str(B/'geometry/surface.ply'))
  np.testing.assert_array_equal(np.asarray(cloud.points).astype(np.float32),self.xyz)
  self.normals=np.asarray(cloud.normals).astype(np.float32);self.tree=cKDTree(self.xyz.astype(np.float64))
  self.source=r.ReplicaFrameSource(CONFIG);assert list(self.source.frame_ids)==FIDS
  self.lookup={(x['frame_id'],x['mask_local_id']):x['instance_id'] for x in [json.loads(s) for s in (B/'association/associations.jsonl').read_text().splitlines()]}
  self.commit=read(B/'prior_commit.json');assert self.commit['status']=='PASS' and self.commit['processed_frames']==OLD_COUNT
  self.journal=Path(self.commit['pixel_journal'])
  manifest=read(self.commit['journal_manifest']);assert manifest['status']=='PASS'
  assert digest(self.commit['journal_manifest'])==self.commit['journal_manifest_sha256']
  self.prior_layers={x['frame_id']:x['layer_files'] for x in manifest['frames']}
  assert sorted(self.prior_layers)==FIDS[:OLD_COUNT]
  self.maximum_id=max(max(self.lookup.values()),int(self.final.max()),int(self.commit['next_instance_id'])-1)
  self.records={fid:{'frame_id':fid,'support_file':str(B/'baseline/frame_support'/f'f{fid:06d}.npz')} for fid in FIDS}
  self.settings=r.AssignmentSettings()
  self.report=dict(min_confirmed_votes=2,min_confirmed_ratio=.67)
  self._support={};self._layers={}
  paths=[self.final_source,self.native/'instance_surface.npz',B/'baseline/surface_evidence.npz',B/'baseline/surface_instance_frame_votes.npz',B/'geometry/surface.ply',B/'association/associations.jsonl',B/'association/observation_support_3cm.npz',CONFIG,B/'prior_commit.json',Path(self.commit['journal_manifest']),Path(self.commit['checkpoint'])]
  self.source_hashes={str(p):digest(p) for p in paths}
 def support(self,fid):
  assert fid in self.records
  if fid not in self._support:self._support[fid]=r.load_npz(self.records[fid]['support_file'])
  return self._support[fid]
 def expected_layer_count(self,fid):
  return len(self.prior_layers[fid]) if fid in self.prior_layers else 1
 def identity_layers(self,fid,frame=None):
  assert fid in self.records
  if fid not in self._layers:
   frame=frame or self.source.load_frame(fid)
   if fid in self.prior_layers:
    images=[]
    for row in self.prior_layers[fid]:
     p=Path(row['path']);sha=digest(p);assert sha==row['sha256']
     self.source_hashes[str(p)]=sha
     image=np.asarray(Image.open(p),np.uint16);assert image.shape==frame.mask_local.shape
     images.append(image)
    self._layers[fid]=images
   else:
    table=np.zeros(int(frame.mask_local.max())+1,np.uint16)
    for mid in np.unique(frame.mask_local):
     if mid:
      pid=self.lookup[(fid,int(mid))];assert 0<pid<=65535
      table[mid]=pid
    self._layers[fid]=[table[frame.mask_local]]
  return self._layers[fid]
 def family_view(self,fid,frame,families):
  z=self.support(fid);present=bool(np.isin(z['candidate_persistent_id'],families).any())
  layers=self.identity_layers(fid,frame)
  foreign=np.logical_or.reduce([(im>0)&~np.isin(im,families) for im in layers])
  return present,foreign
 def old_frame(self,fid,base):
  frame=self.source.load_frame(fid);z=self.support(fid)
  p=z['surface_point_index'];pid=z['candidate_persistent_id']
  return frame,None,strict_frame(np.unique(p.astype(np.int64)*base+pid),base),p
 def verify_unchanged(self):
  for p,h in self.source_hashes.items():assert digest(p)==h,p
