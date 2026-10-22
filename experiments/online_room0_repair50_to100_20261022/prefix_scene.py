from common import *
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from dataclasses import asdict
import run_repairs as r
from history_core_ops import strict_frame
class PrefixScene:
 def __init__(self,scene):
  assert scene=="room0"
  self.scene=scene;self.native=B/"baseline/native";self.final_source=B/"baseline/final/instance_surface.npz"
  self.evidence=r.load_npz(B/"baseline/surface_evidence.npz");self.pair=r.load_npz(B/"baseline/surface_instance_frame_votes.npz")
  nm=r.load_npz(self.native/"instance_surface.npz");fm=r.load_npz(self.final_source)
  self.xyz,self.rgb=nm["xyz_m"],nm["rgb"];self.original,self.final=nm["instance_id"],fm["instance_id"]
  for key in ("xyz_m","rgb"):
   np.testing.assert_array_equal(nm[key],fm[key]);np.testing.assert_array_equal(nm[key],self.evidence[key])
  cloud=o3d.io.read_point_cloud(str(B/"geometry/surface.ply"))
  np.testing.assert_array_equal(np.asarray(cloud.points).astype(np.float32),self.xyz)
  self.normals=np.asarray(cloud.normals).astype(np.float32);assert len(self.normals)==len(self.xyz)
  self.tree=cKDTree(self.xyz.astype(np.float64))
  self.source=r.ReplicaFrameSource(C/"mapping_config50.json");assert list(self.source.frame_ids)==FIDS
  self.lookup={}
  for line in (B/"association/associations.jsonl").read_text().splitlines():
   row=json.loads(line);self.lookup[(row["frame_id"],row["mask_local_id"])]=row["instance_id"]
  self.maximum_id=max(self.lookup.values())
  self.records={fid:{"frame_id":fid,"support_file":str(B/"baseline/frame_support"/f"f{fid:06d}.npz"),"support_sha256":digest(B/"baseline/frame_support"/f"f{fid:06d}.npz")} for fid in FIDS}
  self.report={"min_confirmed_votes":2,"min_confirmed_ratio":.67,"frame_records":list(self.records.values())}
  self.settings=r.AssignmentSettings();self.settings_sha=hashlib.sha256(json.dumps(asdict(self.settings),sort_keys=True,separators=(",",":")).encode()).hexdigest()
  self.source_hashes={str(p):digest(p) for p in (self.final_source,self.native/"instance_surface.npz",B/"baseline/surface_evidence.npz",B/"baseline/surface_instance_frame_votes.npz",B/"geometry/surface.ply",B/"association/associations.jsonl",B/"association/observation_support_3cm.npz",C/"mapping_config50.json")}
 def verify_unchanged(self):
  for p,h in self.source_hashes.items():assert digest(p)==h,p
 def old_frame(self,fid,base):
  assert fid in self.records
  frame=self.source.load_frame(fid);z=r.load_npz(self.records[fid]["support_file"])
  assert digest(self.source.mask_root/self.source.config["source"]["mask_pattern"].format(frame=fid))==str(z["source_mask_sha256"][0])
  p,l,_=r.project_frame_regions(frame,self.tree,pixel_stride=2,max_distance_m=.015,workers=8)
  np.testing.assert_array_equal(p,z["surface_point_index"]);np.testing.assert_array_equal(l,z["mask_local_id"])
  table=np.full(int(frame.mask_local.max())+1,-1,np.int32)
  for mid in np.unique(frame.mask_local):
   if mid:table[mid]=self.lookup[(fid,int(mid))]
  keys=strict_frame(np.unique(p.astype(np.int64)*base+table[l]),base)
  return frame,table,keys,p
